from dataclasses import dataclass, field
from typing import Type
import asyncio
import json
import os
import base64
import re

from llm import model_if_cache, local_embedding, multimodel_if_cache
from prompt import PROMPTS, GRAPH_FIELD_SEP
from storage import (
    BaseGraphStorage,
    BaseVectorStorage,
    BaseKVStorage,
    TextChunkSchema,
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from base import (
    logger, 
    truncate_list_by_token_size, 
    split_string_by_multi_markers,
    list_of_list_to_csv,
    read_config_to_dict,
    limit_async_func_call,
    get_latest_graphml_file,
    EmbeddingFunc
)

# 查询参数类，用于定义查询时的各种配置选项。
from parameter import QueryParam

def path_check(path):
    cache_path = os.getenv('CACHE_PATH')
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    pattern = r'.*(/images/image_\d+.jpg)'
    if not os.path.exists(path):
        path = re.sub(pattern, rf'{working_dir}\1', path)
    return path

def img_path2chunk_id(data,img_data):
    # 建立 image_path 到 chunk_id 的映射
    path_to_chunk = {v["image_path"]: v["chunk_id"] for v in img_data.values()}

    # 遍历原始数据字典，并进行替换和去重
    for key, value_set in data.items():
        updated_values = set()
        for value in value_set:
            if value.endswith('.jpg'):  # 检查是否为 jpg 文件路径
                # 替换为相应的 chunk_id
                chunk_id = path_to_chunk.get(value)
                if chunk_id:
                    updated_values.add(chunk_id)
            else:
                updated_values.add(value)  # 保留原始值
        # 更新字典中的值并去重（由于使用了 set，已自动去重）
        data[key] = updated_values
    return data

async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    异步函数：从实体中查找最相关的文本单元
    根据节点数据和查询参数，从文本块数据库和知识图谱实例中查找并返回最相关的文本单元

    参数:
    - node_datas: 节点数据列表，每个节点数据是一个字典
    - query_param: 查询参数对象，包含本地最大令牌大小等信息
    - text_chunks_db: 文本块存储对象，用于从数据库中获取文本块数据
    - knowledge_graph_inst: 知识图谱存储实例，用于获取节点及其边信息

    返回:
    - all_text_units: 所有相关文本单元的列表，按相关性和顺序排序
    """
    # 根据节点数据的源ID拆分文本单元
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
        if "source_id" in dp
    ]
    # 并发获取所有节点的边信息
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    # 收集所有一跳节点
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    # 并发获取所有一跳节点的数据
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    # 构建一跳节点的文本单元映射
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v
    }
    # 根据图像信息正则化一跳节点的文本单元映射
    cache_path = os.getenv('CACHE_PATH')
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    path = os.path.join(global_config['working_dir'], 'kv_store_image_data.json')
    with open(path, 'r') as file:
        image_data = json.load(file)
    all_one_hop_text_units_lookup = img_path2chunk_id(all_one_hop_text_units_lookup,image_data)
    # 初始化文本单元的全部映射
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if not c_id.startswith('chunk-'):
                continue
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    # 检查是否有缺失的文本块，并记录警告日志
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    # 构建所有文本单元的列表，并按顺序和关联计数排序
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    # 根据令牌大小截断文本单元列表
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
    )
    # 返回文本单元数据列表
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units

async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    异步函数：查找与实体最相关的边

    根据提供的节点数据，从知识图谱中找出最相关的边，并按照相关度排序

    参数:
    node_datas: list[dict] - 节点数据列表，每个节点数据是一个字典
    query_param: QueryParam - 查询参数，用于限制返回数据的token大小
    knowledge_graph_inst: BaseGraphStorage - 知识图谱存储实例

    返回:
    list - 包含相关边数据的列表，按相关度和权重降序排列
    """
    # 并发获取所有节点相关的边
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    # 创建一个集合，用于存储所有相关边的无序对
    all_edges = set()
    for this_edges in all_related_edges:
        # 将边转化为排序后的元组，确保边的顺序不影响后续处理
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    # 将集合转换为列表，以便后续处理
    all_edges = list(all_edges)
    # 并发获取所有边的详细信息
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    # 并发获取所有边的相关度
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    # 根据获取到的边信息、相关度和边的元数据，构建边的数据列表
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    # 根据查询参数中的token大小限制，裁剪边数据列表
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.local_max_token_for_local_context,
    )
    return all_edges_data

async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    """
    异步函数构建本地查询上下文。

    这个函数接受一个查询语句和多个数据存储对象，根据查询参数构建一个本地的查询上下文。
    它会从实体向量数据库中查询 top_k 个最相关的实体，并利用知识图谱存储获取这些实体的详细信息、文本片段及关系。

    参数:
    - query: 查询语句。
    - knowledge_graph_inst: 知识图谱实例，用于获取节点和边的信息。
    - entities_vdb: 实体向量存储，用于查询相关实体。
    - text_chunks_db: 文本块存储，用于查找关联的文本片段。
    - query_param: 查询参数，包含如 top_k 等查询选项。

    返回:
    - 一个字符串，包含了查询结果的上下文，格式为 CSV，如果未找到结果则为 None。
    """
    # 根据查询语句在实体向量数据库中查询 top_k 个最相关的实体
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return "", ""
    # 获取查询到的实体的节点数据
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    # 获取查询到的实体的节点度数
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    # 整合节点数据，包括实体名称、排名和节点度数
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    # 构建实体部分的 CSV 数据
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)
    # 构建关系部分的 CSV 数据
    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)
    # 构建文本单元部分的 CSV 数据
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    context = f"""
    -----Entities-----
    ```csv
    {entities_context}
    ```
    -----Relationships-----
    ```csv
    {relations_context}
    ```
    -----Sources-----
    ```csv
    {text_units_context}
    ```
    """
    return entities_context, context

@dataclass
class local_query:
    # 默认使用本地embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: local_embedding)
    # 最大并发请求数量
    embedding_func_max_async: int = 16
    # 键值存储，json，具体定义在storage.py
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    # 向量数据库存储,具体定义在storage.py
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    # 图数据库存储，默认为NetworkXStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    def __post_init__(self):
        # 获取全局设置
        cache_path = os.getenv('CACHE_PATH')
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        # 加载存储的文本块
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=global_config
        )
        # 加载知识图谱
        kg_namespace, _ = get_latest_graphml_file(global_config['working_dir'])
        self.knowledge_graph_merged = self.graph_storage_cls(
            namespace=kg_namespace, global_config=global_config
        )
        # 限制embedding函数的异步调用次数
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        # 加载实体向量数据库
        self.entities_database = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=global_config,
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
        )

    async def local_query(
        self,
        query,
        query_param: QueryParam,
    ) -> str:
        """
        根据查询和配置，执行本地查询并返回结果。

        该函数首先根据查询和一系列存储实例构建本地查询上下文。
        如果上下文为空，则返回预定义的查询失败响应。
        否则，将根据上下文和查询参数构造系统提示生成响应。

        参数:
            query: 查询字符串。
            query_param: 查询参数，包含查询类型和响应类型等信息。

        返回:
            根据查询和上下文生成的响应字符串。
        """
        # 获取index结果
        
        knowledge_graph_inst = self.knowledge_graph_merged

        entities_vdb = self.entities_database

        text_chunks_db = self.text_chunks
        # 从全局配置中获取最佳模型函数
        use_model_func = model_if_cache
        use_multimodel_func = multimodel_if_cache
        # 构建本地查询上下文
        entities_context, context = await _build_local_query_context(
            query,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
        # 将context字符串保存到 CSV 文件
        cache_path = os.getenv('CACHE_PATH')
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        context_path = os.path.join(global_config["working_dir"], "context.csv")
        with open(context_path, 'a', newline='') as file:
            file.write(f"{context}\n")
        # 如果上下文为空，则返回查询失败的响应
        if context is None:
            return PROMPTS["fail_response"]
        # 根据预定义模板构造系统提示，包含上下文数据和查询参数中的响应类型
        sys_prompt_temp = PROMPTS["local_rag_response_augmented"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context, response_type=query_param.response_type
        )
        # 生成查询响应
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
        )
        with open(context_path, 'a', newline='') as file:
            file.write(f"{response}\n")
        # 获取多模态实体
        img_entities = []
        for line in entities_context.split("\n")[1:]:  # 跳过第一行标题
            parts = line.split(",")
            if len(parts) >= 3 and parts[2].strip().strip('"') == "ORI_IMG":
                entity = parts[1].strip().strip('"')
                img_entities.append(entity)
        img_entities = [entity.lower() for entity in img_entities][:QueryParam.number_of_mmentities]
        logger.info(f'Using multimodal entities{img_entities}')
        if not img_entities:
            return response

        image_data_path = os.path.join(global_config["working_dir"], 'kv_store_image_data.json')
        # 加载image_data
        with open(image_data_path, 'r', encoding='utf-8') as file:
            image_data = json.load(file)
        # 获取图像路径
        image_paths = [image_data[entity]['image_path'] for entity in img_entities if entity in image_data]
        captions = [image_data[entity]['caption'] for entity in img_entities if entity in image_data]
        footnotes = [image_data[entity]['footnote'] for entity in img_entities if entity in image_data]
        #  base 64 编码格式
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        
        mm_response = []
        format_query = f'Query:{query}'
        for path, caption, footnote in zip(image_paths, captions, footnotes):
            path = path_check(path)
            image_base = encode_image(path)
            information = f"{caption}, {footnote}"
            mm_prompt = PROMPTS["local_rag_response_multimodal"].format(
                context_data=context, 
                response_type=query_param.response_type, 
                image_information=information
            )
            response = await use_multimodel_func(
                format_query,
                img_base=image_base,
                system_prompt=mm_prompt,
            )
            mm_response.append(response)
        with open(context_path, 'a', newline='') as file:
            file.write(f"mm_response:\n{mm_response}\n")
        merge_prompt = PROMPTS["local_rag_response_multimodal_merge"].format(mm_responses=mm_response)
        mm_merged_response = await use_model_func(
            query,
            system_prompt=merge_prompt,
        )
        with open(context_path, 'a', newline='') as file:
            file.write(f"merged_mm_response:\n{mm_merged_response}\n")
        final_prompt = PROMPTS["local_rag_response_merge"].format(response_type=query_param.response_type, mm_response=mm_merged_response, response=response)
        final_response = await use_model_func(
            format_query,
            system_prompt=final_prompt,
        )
        return final_response