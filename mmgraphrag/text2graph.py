from collections import defaultdict,Counter
from typing import Type, Union, cast
from dataclasses import dataclass
from functools import partial
import asyncio
import re
import os
import json

from prompt import GRAPH_FIELD_SEP, PROMPTS
from base import (
    logger,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    clean_str,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    is_float_regex,
    limit_async_func_call,
    read_config_to_dict,
)
from storage import (
    BaseGraphStorage,
    StorageNameSpace,
    BaseKVStorage,
    JsonKVStorage,
    NetworkXStorage,
    TextChunkSchema
)

from llm import model_if_cache

# 异步函数处理单个实体提取任务的结果
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
    处理单个实体提取任务。

    该函数负责验证并处理给定的实体记录属性，从中提取实体名称、类型和描述等信息，
    并返回一个字典，包含这些信息以及实体的来源标识。

    参数:
    - record_attributes: 一个字符串列表，包含实体的属性信息。预期列表中至少有4个元素，
      第一个元素为'entity'，标识这是一个实体记录。
    - chunk_key: 一个字符串，表示实体信息来源的唯一标识。

    返回:
    - 如果记录属性有效，返回一个字典，包含实体名称、类型、描述和来源标识。
    - 如果记录属性无效（如元素数量不足或第一个元素不是'entity'），则返回None。
    """
    # 检查record_attributes列表是否至少有4个元素，且第一个元素是否为'entity'
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )

async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """
    根据全局配置处理实体和关系的描述并生成摘要。

    参数:
    - entity_or_relation_name: 实体或关系的名称。
    - description: 实体或关系的描述。
    - global_config: 包含模型、令牌大小、总结最大令牌数等的全局配置。

    返回:
    - 生成的摘要或原始描述。
    """
    # 从全局配置中获取相应的函数和参数
    use_llm_func = model_if_cache
    llm_max_tokens = global_config["model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    # 编码描述信息
    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    # 如果描述信息的令牌数小于最大摘要令牌数，则直接返回描述信息
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    # 设置prompt
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    # 获取适合模型最大令牌数的描述信息
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    # 构建上下文基础信息
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    # 构建最终的prompt
    user_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    # 使用语言模型生成摘要
    summary = await use_llm_func(user_prompt, max_tokens=summary_max_tokens)
    return summary

async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )

async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    合并节点数据并更新或插入知识图谱中的节点。

    该函数首先尝试从知识图谱中获取已存在的节点信息，然后与新获取的节点数据进行合并。
    合并后的节点数据将根据给定的规则更新或插入到知识图谱中。

    参数:
    - entity_name (str): 实体名称，用于标识知识图谱中的节点。
    - nodes_data (list[dict]): 一组节点数据，每个节点数据是一个字典。
    - knwoledge_graph_inst (BaseGraphStorage): 知识图谱实例，用于操作知识图谱。
    - global_config (dict): 全局配置信息，可能用于配置合并或处理节点数据的规则。

    返回:
    - node_data (dict): 更新或插入后的节点数据。
    """
    # 初始化列表，用于存储已存在的节点信息
    already_entitiy_types = []
    already_source_ids = []
    already_description = []
    # 从知识图谱中获取已存在的节点信息
    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        # 如果节点存在，则将已存在的信息添加到对应的列表中
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
    # 合并新旧节点的实体类型，选择出现次数最多的作为新的实体类型
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    # 合并新旧节点的描述，使用分隔符连接所有不同的描述
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    # 合并新旧节点的源ID，使用分隔符连接所有不同的源ID
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    # 处理实体的描述信息
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    # 更新或插入节点到知识图谱
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    # 添加实体名称到节点数据中
    node_data["entity_name"] = entity_name
    return node_data

async def _merge_edges_then_upsert(
   
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    合并边数据并插入/更新知识图谱。
    该函数检查src_id和tgt_id之间是否存在边，如果存在，则获取当前边数据并
    与新的边数据(edges_data)合并；如果不存在，则直接插入新的边数据。同时，
    如果src_id或tgt_id在图中不存在相应的节点，则插入默认属性的节点。

    参数:
    - src_id (str): 边的起始节点ID。
    - tgt_id (str): 边的目标节点ID。
    - edges_data (list[dict]): 包含一个或多个边的数据。
    - knwoledge_graph_inst (BaseGraphStorage): 知识图谱实例。
    - global_config (dict): 全局配置。
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    # 如果src_id和tgt_id之间存在边，则获取现有边数据
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
     # 计算总权重
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
     # 合并并去重描述，排序后转为字符串
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    # 合并并去重源ID
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    # 确保src_id和tgt_id在图中存在节点，如果不存在则插入
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    # 插入/更新src_id和tgt_id之间的边
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )

async def extract_entities(
    cache_dir: BaseKVStorage,
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    """
    异步函数extract_entities从文本块中提取实体并更新知识图谱。

    参数:
    chunks (dict[str, TextChunkSchema]): 文本块字典，键为文本块标识，值为包含文本块内容的TextChunkSchema对象。
    knwoledge_graph_inst (BaseGraphStorage): 知识图谱实例，用于存储提取的实体和关系。
    entity_vdb (BaseVectorStorage): 实体向量数据库实例，用于存储实体的向量表示。
    global_config (dict): 全局配置字典，包含模型函数、最大迭代次数等参数。

    返回:
    Union[BaseGraphStorage, None]: 更新后的知识图谱实例，如果没有提取到任何实体，则返回None。
    """
    # 用来存储每个chunk的实体和关系
    output_json_path = f"{global_config['working_dir']}/kv_store_chunk_knowledge_graph.json"

    model_max_async: int = 16
    # 限制模型函数的异步调用次数，并为其配置哈希键值存储
    use_llm_func = limit_async_func_call(model_max_async)(
        partial(model_if_cache, hashing_kv=cache_dir)
    )
    # 从全局配置中获取实体提取最大迭代次数
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    # 将文本块排序，以便按顺序处理
    ordered_chunks = list(chunks.items())
    # 准备实体提取的提示模板和上下文基础信息
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entity_continue_extraction"]
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]
    # 初始化计数器，用于统计已处理的文本块、已提取的实体和关系数量
    already_processed = 0
    already_entities = 0
    already_relations = 0
    # 实现一个全局的结果字典用于与json格式对应
    chunk_knowledge_graph_info = {}

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """
        异步函数_process_single_content处理单个文本块的内容，提取实体并更新知识图谱。

        参数:
        chunk_key_dp (tuple[str, TextChunkSchema]): 文本块的键值对，包含文本块标识和内容。

        返回:
        dict: 包含从文本块中提取的可能的节点和边的字典。
        """
        # 初始化非局部变量，用于跟踪处理进度和统计信息
        nonlocal already_processed, already_entities, already_relations
        # 解析文本块的键和数据
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        chunk_order_index = chunk_dp["chunk_order_index"]
        # 构建提示信息并调用模型函数提取实体
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        # 构建对话历史并进行多次迭代以提取更多实体
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        # 循环执行最多 entity_extract_max_gleaning 次，进行迭代提取
        for now_glean_index in range(entity_extract_max_gleaning):
            # 调用模型函数继续提取
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            # 将新的结果添加到历史对话和最终结果中
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            # 检查是否继续迭代
            if now_glean_index == entity_extract_max_gleaning - 1:
                break
            # 调用模型函数检查是否需要继续迭代提取
            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            # 解析模型的返回值，去掉多余的引号和空白，并转换为小写
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            # 如果返回值不是 "yes"，则退出循环
            if if_loop_result != "yes":
                break
        # 将最终提取的结果按照多个分隔符分割成记录
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        # 使用 defaultdict 来存储可能的节点和边
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        # 为当前文本块初始化结果字典
        chunk_results = {
            "chunk_key": chunk_key,
            "entities": [],
            "relationships": [],
        }
        # 遍历每一条记录，处理节点和边的提取
        for record in records:
            # 使用正则表达式从记录中提取元组数据
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            # 按照元组分隔符分割记录属性
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            # 处理实体提取
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                chunk_results["entities"].append(if_entities)
                continue
            # 处理关系提取
            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
                chunk_results["relationships"].append(if_relation)
        # 更新处理进度和统计信息
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        # 存储该文本块的结果
        chunk_knowledge_graph_info[chunk_order_index] = chunk_results
        return dict(maybe_nodes), dict(maybe_edges)
    # 并发处理所有文本块
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear the progress bar
    # 将知识图谱信息保存为 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(chunk_knowledge_graph_info, json_file, ensure_ascii=False, indent=4)
    # 汇总所有可能的节点和边
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # 构建无向图时，按字典序排序边的节点
            maybe_edges[tuple(sorted(k))].extend(v)
    # 合并节点数据并更新知识图谱
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )

    # 合并边数据并更新知识图谱
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    # 如果没有提取到任何实体，发出警告并返回 None
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    return knwoledge_graph_inst

@dataclass
class extract_entities_from_text:
    # 实体提取函数
    text_entity_extraction_func: callable = extract_entities

    # 存储类型设置
    # 键值存储，json，具体定义在storage.py
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    # 图数据库存储，默认为NetworkXStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    def __post_init__(self):
        # 获取全局设置
        cache_path = os.getenv('CACHE_PATH')
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        # 根据配置初始化LLM响应缓存
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=global_config
            )
        )
        # 初始化图存储类实例，用于存储块实体关系图
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=global_config
        )
    async def text_entity_extraction(self, inserting_chunks):
        try:
            # ---------- extract/summary entity and upsert to graph
            # 提取新实体和关系，并更新到知识图谱中
            # 获取全局设置
            cache_path = os.getenv('CACHE_PATH')
            global_config_path = os.path.join(cache_path,"global_config.csv")
            global_config = read_config_to_dict(global_config_path)
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.text_entity_extraction_func(
                self.llm_response_cache,
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                global_config=global_config,
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
        finally:
            await self._text_entity_extraction_done()
    async def _text_entity_extraction_done(self):
        tasks = []
        for storage_inst in [
            self.llm_response_cache,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks) 