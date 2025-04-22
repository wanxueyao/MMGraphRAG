from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass, field
from typing import Type, cast
import math
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from prompt import PROMPTS, GRAPH_FIELD_SEP
from base import logger

import networkx as nx
import xml.etree.ElementTree as ET
import json
import asyncio
import os
import base64

from llm import local_embedding, get_llm_response, get_mmllm_response, normalize_to_json, normalize_to_json_list
from parameter import encode
from storage import (
    StorageNameSpace,
    BaseVectorStorage,
    NanoVectorDBStorage,
)
from base import (
    limit_async_func_call,
    read_config_to_dict,
    compute_mdhash_id,
    get_latest_graphml_file,
    EmbeddingFunc,
)
cache_path = os.getenv('CACHE_PATH')

# 获取全局设置
def get_image_data():
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    json_file_path1 = os.path.join(working_dir, 'kv_store_image_data.json')
    with open(json_file_path1, 'r', encoding='utf-8') as f:
        image_data = json.load(f)
    return image_data

def get_chunk_knowledge_graph():
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    json_file_path2 = os.path.join(working_dir, 'kv_store_chunk_knowledge_graph.json')
    with open(json_file_path2, 'r', encoding='utf-8') as f:
        chunk_knowledge_graph = json.load(f)
    return chunk_knowledge_graph

def get_text_chunks():
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    json_file_path3 = os.path.join(working_dir, 'kv_store_text_chunks.json')
    with open(json_file_path3, 'r', encoding='utf-8') as f:
        text_chunks = json.load(f)
    return text_chunks

def extract_entities_from_graph(graphml_file: str) -> list[dict]:
    # 加载GraphML文件
    graph = nx.read_graphml(graphml_file)
    
    # 存储所有实体节点的列表
    entity_list = []
    
    # 遍历图中的每个节点，提取所需的数据
    for node_id, node_data in graph.nodes(data=True):
        # 提取实体信息
        entity_info = {
            "entity_type": node_data.get('entity_type', 'UNKNOWN'),
            "description": node_data.get('description', ''),
            "source_id": node_data.get('source_id', ''),
            "entity_name": node_id
        }
        
        # 将实体信息加入到列表中
        entity_list.append(entity_info)
    
    return entity_list

@dataclass
class create_EntityVDB:
    # 默认使用本地embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: local_embedding)
    # 最大并发请求数量
    embedding_func_max_async: int = 16
    
    # 向量数据库存储,具体定义在storage.py
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage


    def __post_init__(self):
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        # 限制embedding函数的异步调用次数
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        # 根据配置初始化向量数据库存储类实例，用于存储实体
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=global_config,
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
        )
    async def create_vdb(self):
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        working_dir = global_config['working_dir']
        _, graph_file = get_latest_graphml_file(working_dir)
        all_entities_data = extract_entities_from_graph(graph_file)
        # 如果实体向量数据库不为空，则将提取到的实体存储到向量数据库中
        if self.entities_vdb is not None:
            # 将实体数据构造成向量数据库格式
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + dp["description"],
                    "entity_name": dp["entity_name"],
                }
                for dp in all_entities_data
            }
            # 将数据插入到向量数据库中
            await self.entities_vdb.upsert(data_for_vdb)
        return await self._create_vdb_done()
    async def _create_vdb_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

def get_nearby_chunks(data, index):
    # 获取前后两个数字的范围
    start_index = max(0, index - 1)  # 如果是0，则只取0和1
    end_index = min(len(data) - 1, index + 1)  # 如果是最后一个数字，则只取自己和前一个
    
    all_index = list(range(start_index, end_index + 1))
    nearby_chunks = []
    for key, value in data.items():
            if value.get("chunk_order_index") in all_index:
                nearby_chunks.append(value.get("content"))
    return nearby_chunks

def get_nearby_entities(data, index):
        # 获取前后两个数字的范围
        start_index = max(0, index - 1)  # 如果是0，则只取0和1
        end_index = min(len(data) - 1, index + 1)  # 如果是最后一个数字，则只取自己和前一个
        
        # 提取指定范围的entities
        entities = []
        for i in range(start_index, end_index + 1):
            entities.extend(data.get(str(i), {}).get("entities", []))
        # 去掉每个实体的 source_id
        for entity in entities:
            entity.pop("source_id", None)
        return entities

def get_nearby_relationships(data, index):
        # 获取前后两个数字的范围
        start_index = max(0, index - 1)  # 如果是0，则只取0和1
        end_index = min(len(data) - 1, index + 1)  # 如果是最后一个数字，则只取自己和前一个
        
        # 提取指定范围的relationships
        relationships = []
        for i in range(start_index, end_index + 1):
            relationships.extend(data.get(str(i), {}).get("relationships", []))
        # 去掉每个关系的 source_id
        for relationship in relationships:
            relationship.pop("source_id", None)
        return relationships

def align_single_image_entity(img_entity_name, text_chunks):
    image_data = get_image_data()
    image_path = image_data[img_entity_name]["image_path"]
    img_entity_description = image_data[img_entity_name]["description"]
    chunk_order_index = image_data[img_entity_name]["chunk_order_index"]
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_order_index)
    entity_type = PROMPTS["DEFAULT_ENTITY_TYPES"]
    entity_type = [item.upper() for item in entity_type]
    with open(image_path, "rb") as image_file:
        img_base = base64.b64encode(image_file.read()).decode("utf-8")
    alignment_prompt_user = PROMPTS["image_entity_alignment_user"].format(entity_type = entity_type, img_entity=img_entity_name, img_entity_description=img_entity_description, chunk_text=nearby_chunks)
    aligned_image_entity = get_mmllm_response(alignment_prompt_user, PROMPTS["image_entity_alignment_system"], img_base)
    return normalize_to_json(aligned_image_entity)

def judge_image_entity_alignment(image_entity_name, image_entity_description, possible_image_matched_entities, nearby_chunks):
    image_entity_judgement_user = PROMPTS["image_entity_judgement_user"].format(img_entity=image_entity_name, img_entity_description=image_entity_description, possible_matched_entities=possible_image_matched_entities, chunk_text=nearby_chunks)
    matched_entity_name = get_llm_response(image_entity_judgement_user, PROMPTS["image_entity_judgement_system"])
    return matched_entity_name

def get_possible_entities_image_clustering(
    image_entity_description, nearby_text_entity_list, nearby_relationship_list
):
    # Step 0: 排序关系列表，根据权重降序
    nearby_relationship_list = sorted(nearby_relationship_list, key=lambda x: x['weight'], reverse=True)
    
    # Step 1: 获取所有实体描述的嵌入
    descriptions = [entity["description"] for entity in nearby_text_entity_list]
    entity_names = [entity["entity_name"] for entity in nearby_text_entity_list]
    embeddings = encode(descriptions)

    # Step 2: spectral聚类
    # 计算相似度矩阵（余弦相似度）
    similarity_matrix = cosine_similarity(embeddings)

    # 根据关系权重修改度矩阵
    for relation in nearby_relationship_list:
        # 只有当 src_id 和 tgt_id 都在 entity_names 中时才执行
        if relation["src_id"] in entity_names and relation["tgt_id"] in entity_names:
            src_idx = entity_names.index(relation["src_id"])
            tgt_idx = entity_names.index(relation["tgt_id"])
        else:
            continue

        weight = relation["weight"]
        similarity_matrix[src_idx, tgt_idx] *= weight
        similarity_matrix[tgt_idx, src_idx] *= weight  # 确保邻接矩阵是对称的
    
    # 计算度矩阵
    degree_matrix = np.zeros_like(similarity_matrix)
    for i in range(len(similarity_matrix)):
        degree_matrix[i, i] = np.sum(similarity_matrix[i, :])

    # 计算拉普拉斯矩阵 L = D - A
    laplacian_matrix = degree_matrix - similarity_matrix

    # 计算拉普拉斯矩阵的特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(laplacian_matrix)

    # 选择前k个最小的特征值对应的特征向量
    k = max(2, math.ceil(math.sqrt(len(nearby_text_entity_list))))
    eigvecs_selected = eigvecs[:, np.argsort(eigvals)[:k]]
    # 前面拉普拉斯矩阵可能出现非实对称矩阵的情况导致出现复数，所以这里加一步取模值
    eigvecs_selected = np.abs(eigvecs_selected)

    # 使用 DBSCAN 聚类
    min_samples = max(1, math.ceil(len(nearby_text_entity_list) / 10))
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples)  # 调整 eps 和 min_samples 参数
    dbscan_labels = dbscan.fit_predict(eigvecs_selected)

    # 输出每个节点的聚类标签
    labels = dbscan_labels

    # 按照 nearby_text_entity_list 的顺序输出 labels
    labels = [labels[entity_names.index(entity["entity_name"])] for entity in nearby_text_entity_list]

    # Step 3: 判断输入描述的类别
    input_embedding = encode([image_entity_description])
    # 检查训练数据的样本数量
    n_samples_fit = embeddings.shape[0]
    # 设置 n_neighbors，确保它不会超过训练数据的样本数量
    n_neighbors = min(3, n_samples_fit)
    # 找到最近邻并使用 Pagerank,Leiden或Spectral的标签
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(embeddings)
    # 找到最近邻并使用 Pagerank,Leiden或Spectral的标签
    nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(embeddings)
    _, nearest_idx = nn.kneighbors(input_embedding)
    target_label = labels[nearest_idx[0][0]]

    # Step 4: 输出属于该类别的所有实体信息
    result_entities = [
        entity
        for entity, label in zip(nearby_text_entity_list, labels)
        if label == target_label
    ]

    return result_entities

def get_possible_entities_text_clustering(
   filtered_image_entity_list, nearby_text_entity_list, nearby_relationship_list
):
    """
    聚类和分类函数，支持 KMeans/DBSCAN/Pagerank/Leiden 聚类及 KNN/LLM 分类。

    Parameters:
        clustering_method (str): 聚类方法 ("KMeans", "DBSCAN", "Pagerank", 或 "Leiden")。
        classify_method (str): 分类方法 ("knn" 或 "llm")。
        filtered_image_entity_list (list): 过滤后的图像实体列表，每个实体包含 "entity_name" 和 "description"。
        nearby_text_entity_list (list): 附近文本实体列表，每个实体包含 "entity_name"、"entity_type" 和 "description"。
        nearby_relationship_list (list): 实体之间的关系列表，每个关系包含 "src_id"、"tgt_id"、"weight" 和 "description"。

    Returns:
        image_entity_with_labels (list): 图像实体及其对应类别的列表，每项为 {"entity_name": ..., "label": ..., "description": ..., "entity_type": ...}。
        text_clustering_results (list): 聚类后的文本实体列表，每项为 {"label": ..., "entities": [...]}。
    """
    # Step 0: 排序关系列表，根据权重降序
    nearby_relationship_list = sorted(nearby_relationship_list, key=lambda x: x['weight'], reverse=True)

    # Step 1: 获取文本实体描述的嵌入
    descriptions = [entity["description"] for entity in nearby_text_entity_list]
    entity_names = [entity["entity_name"] for entity in nearby_text_entity_list]
    embeddings = encode(descriptions)

    # Step 2: spectral聚类
    # 计算相似度矩阵（余弦相似度）
    similarity_matrix = cosine_similarity(embeddings)

    # 根据关系权重修改度矩阵
    for relation in nearby_relationship_list:
        # 只有当 src_id 和 tgt_id 都在 entity_names 中时才执行
        if relation["src_id"] in entity_names and relation["tgt_id"] in entity_names:
            src_idx = entity_names.index(relation["src_id"])
            tgt_idx = entity_names.index(relation["tgt_id"])
        else:
            continue

        weight = relation["weight"]
        similarity_matrix[src_idx, tgt_idx] *= weight
        similarity_matrix[tgt_idx, src_idx] *= weight  # 确保邻接矩阵是对称的
    
    # 计算度矩阵
    degree_matrix = np.zeros_like(similarity_matrix)
    for i in range(len(similarity_matrix)):
        degree_matrix[i, i] = np.sum(similarity_matrix[i, :])

    # 计算拉普拉斯矩阵 L = D - A
    laplacian_matrix = degree_matrix - similarity_matrix

    # 计算拉普拉斯矩阵的特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(laplacian_matrix)

    # 选择前k个最小的特征值对应的特征向量
    k = max(2, math.ceil(math.sqrt(len(nearby_text_entity_list))))
    eigvecs_selected = eigvecs[:, np.argsort(eigvals)[:k]]
    # 前面拉普拉斯矩阵可能出现非实对称矩阵的情况导致出现复数，所以这里加一步取模值
    eigvecs_selected = np.abs(eigvecs_selected)

    # 使用 DBSCAN 聚类
    min_samples = max(1, math.ceil(len(nearby_text_entity_list) / 10))
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples)  # 调整 eps 和 min_samples 参数
    dbscan_labels = dbscan.fit_predict(eigvecs_selected)

    # 输出每个节点的聚类标签
    labels = dbscan_labels

    # 创建一个字典，初始化所有实体标签为 -1，表示未分类
    entity_labels = {entity["entity_name"]: -1 for entity in nearby_text_entity_list}

    # 按照聚类结果分配标签
    for idx, entity in enumerate(nearby_text_entity_list):
        entity_labels[entity["entity_name"]] = labels[idx]

    # 按照 nearby_text_entity_list 的顺序输出 labels
    labels = [entity_labels[entity["entity_name"]] for entity in nearby_text_entity_list]

    # Step 3: 分类图像实体到聚类类别
    image_entity_with_labels = []
    input_embeddings = encode([entity["description"] for entity in filtered_image_entity_list])
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings)
    for image_entity, input_embedding in zip(filtered_image_entity_list, input_embeddings):
        _, nearest_idx = nn.kneighbors([input_embedding])
        target_label = labels[nearest_idx[0][0]]
        image_entity_with_labels.append({
            "entity_name": image_entity["entity_name"],
            "label": target_label,
            "description": image_entity["description"],
            "entity_type": image_entity.get("entity_type", "image")
        })
    
    # Step 4: 生成聚类结果
    text_clustering_results = []
    for label in set(labels):
        text_clustering_results.append({
            "label": label,
            "entities": [
                {
                    "entity_name": nearby_text_entity_list[idx]["entity_name"],
                    "entity_type": nearby_text_entity_list[idx]["entity_type"],
                    "description": nearby_text_entity_list[idx]["description"],
                }
                for idx in range(len(labels))
                if labels[idx] == label
            ]
        })

    return image_entity_with_labels, text_clustering_results

def judge_text_entity_alignment_clustering(image_entity_with_labels, text_clustering_results):
    """
    使用 LLM 判断是否需要融合实体，并输出融合结果。

    Parameters:
        image_entity_with_labels (list): 图像实体及其对应类别的列表，每项为 {"entity_name": ..., "label": ..., "description": ..., "entity_type": ...}。
        text_clustering_results (list): 聚类后的文本实体列表，每项为 {"label": ..., "entities": [...]}。

    Returns:
        merged_entities (list): 融合的实体列表，每项为 {
            "entity_name": ..., 
            "entity_type": ..., 
            "description": ..., 
            "source_image_entities": [...], 
            "source_text_entities": [...]
        }。
    """
    # 构建融合任务的上下文
    clusters_info = []
    for cluster in text_clustering_results:
        clusters_info.append({
            "label": cluster["label"],
            "text_entities": [
                {
                    "entity_name": entity["entity_name"],
                    "entity_type": entity["entity_type"],
                    "description": entity["description"],
                }
                for entity in cluster["entities"]
            ]
        })

    # 构建输入 prompt
    prompt_user = f"""
You are tasked with aligning image entities and text entities based on their labels and descriptions. Below are the clusters and the entities they contain.

Clusters information:
{{
    "clusters": [
        {", ".join([f'{{"label": {c["label"]}, "text_entities": {c["text_entities"]}}}' for c in clusters_info])}
    ]
}}

Image entities with labels:
{[
    {
        "entity_name": e["entity_name"],
        "label": e["label"],
        "description": e["description"],
        "entity_type": e["entity_type"]
    }
    for e in image_entity_with_labels
]}

Instruction:
1. For each image entity, look at the corresponding cluster (same label).
2. Compare the description and type of the image entity with the text entities in the same cluster.
3. Identify matching entities between the image entities and text entities within the same cluster (same label).
4. For each match, create a new unified entity by merging the descriptions and including the source entities under "source_image_entities" and "source_text_entities".
5. Output a JSON list where each item represents a merged entity with the following structure:
    {{
        "entity_name": "Newly merged entity name",
        "entity_type": "Type of the merged entity",
        "description": "Merged description of the entity",
        "source_image_entities": ["List of matched image entity names"],
        "source_text_entities": ["List of matched text entity names"]
    }}
Include only one JSON list as the output, strictly following the structure above.
"""
    prompt_system = """You are an AI assistant skilled in aligning entities based on semantic descriptions and cluster information. Use the provided instructions to merge entities accurately."""

    # 调用 LLM 获取融合结果
    merged_entities = get_llm_response(cur_prompt=prompt_user, system_content=prompt_system)
    normalized_merged_entities = normalize_to_json_list(merged_entities)
    return [
    item for item in normalized_merged_entities 
    if item.get("source_image_entities") and item.get("source_text_entities")
]

def extract_image_entities(img_entity_name):
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    # 构建 GraphML 文件路径
    image_knowledge_graph_path = os.path.join(working_dir, f"images/{img_entity_name}/graph_{img_entity_name}_entity_relation.graphml")
    
    # 检查文件是否存在
    if not os.path.exists(image_knowledge_graph_path):
        print(f"GraphML file not found: {image_knowledge_graph_path}")
        return

    # 解析 GraphML 文件
    tree = ET.parse(image_knowledge_graph_path)
    root = tree.getroot()
    image_entities = []
    # 定义命名空间
    namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    # 遍历所有 'node' 元素
    for node in root.findall('graphml:graph/graphml:node', namespace):
        # 提取实体信息
        entity_name = node.get('id').strip('"')
        for data in node.findall('graphml:data', namespace):
            if data.get('key') == 'd0':  # 'd0' 对应实体类型
                entity_type = data.text.strip('"')  # 获取实体类型并去掉引号
        for data in node.findall('graphml:data', namespace):
            if data.get('key') == 'd1':  # 'd1' 对应描述
                description = data.text.strip('"')  # 获取描述并去掉引号

        # 准备节点数据
        node_data = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": description
        }
        image_entities.append(node_data)
    return image_entities

def enhance_image_entities(enhanced_image_entity_list, nearby_chunks):
    enhance_image_entity_user = PROMPTS["enhance_image_entity_user"].format(enhanced_image_entity_list=enhanced_image_entity_list, chunk_text=nearby_chunks)
    enhanced_image_entities = get_llm_response(enhance_image_entity_user, PROMPTS["enhance_image_entity_system"])
    return normalize_to_json_list(enhanced_image_entities)

def ensure_quoted(entity_name):
    # 检查字符串是否以双引号开始和结束
    if not (entity_name.startswith('"') and entity_name.endswith('"')):
        # 如果没有双引号，则加上双引号
        entity_name = f'"{entity_name}"'
    return entity_name

def image_knowledge_graph_alignment(image_entity_name):
    image_data = get_image_data()
    chunk_knowledge_graph = get_chunk_knowledge_graph()
    chunk_order_index = image_data[image_entity_name].get("chunk_order_index")
    image_entity_list = extract_image_entities(image_entity_name)
    exclude_types=["ORI_IMG", "IMG"]
    filtered_image_entity_list = [entity for entity in image_entity_list if entity['entity_type'] not in exclude_types]
    nearby_text_entity_list = get_nearby_entities(chunk_knowledge_graph, chunk_order_index)
    nearby_relationship_list = get_nearby_relationships(chunk_knowledge_graph, chunk_order_index)
    image_entity_with_labels, text_clustering_results = get_possible_entities_text_clustering(filtered_image_entity_list, nearby_text_entity_list, nearby_relationship_list)
    aligned_text_entity_list = judge_text_entity_alignment_clustering(image_entity_with_labels, text_clustering_results)
    return aligned_text_entity_list

def enhanced_image_knowledge_graph(aligned_text_entity_list, image_entity_name):
    image_data = get_image_data()
    text_chunks = get_text_chunks()
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    img_kg_path = os.path.join(working_dir, f'images/{image_entity_name}/graph_{image_entity_name}_entity_relation.graphml')
    enhanced_img_kg_path = os.path.join(working_dir, f'images/{image_entity_name}/enhanced_graph_{image_entity_name}_entity_relation.graphml')
    image_entity_list = extract_image_entities(image_entity_name)
    exclude_types=["ORI_IMG", "IMG"]
    filtered_image_entity_list = [entity for entity in image_entity_list if entity['entity_type'] not in exclude_types]
    chunk_order_index = image_data[image_entity_name]["chunk_order_index"]
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_order_index)
    # Safely access 'source_image_entities' and ensure there are no errors
    source_image_entities = []
    for entity in aligned_text_entity_list:
        if 'source_image_entities' in entity and entity['source_image_entities']:
            source_image_entities.append(entity['source_image_entities'][0])
        else:
            # If 'source_image_entities' is missing or empty, we skip this entity
            print(f"Warning: 'source_image_entities' is missing or empty for entity {entity}. Skipping.")
    # Filter out image entities that have a matching 'entity_name'
    enhanced_image_entity_list = [entity for entity in filtered_image_entity_list if entity['entity_name'] not in source_image_entities]
    enhanced_image_entities = enhance_image_entities(enhanced_image_entity_list, nearby_chunks)
    
    # Step 1: Load the original knowledge graph
    G = nx.read_graphml(img_kg_path)
    
    # Step 2: Update the graph with enhanced entity details for nodes
    for entity in enhanced_image_entities:
        original_name = entity.get('original_name', None)  # Use .get() to avoid KeyError
        if original_name is None:
            print(f"Warning: 'original_name' is missing for entity {entity}. Skipping update.")
            continue  # Skip this iteration if 'original_name' is missing

        entity['entity_name'] = ensure_quoted(entity['entity_name'])

        # If 'description' is missing, skip this entity
        if 'description' not in entity:
            print(f"Warning: 'description' is missing for entity {entity}. Skipping update.")
            continue

        # Update nodes based on original name
        for node in G.nodes(data=True):
            node_id = node[0].strip('"')  # Remove extra quotes around node IDs
            if original_name == node_id:
                # Modify node_id to the new entity_name
                G = nx.relabel_nodes(G, {node[0]: entity['entity_name']})

                # Update the node's description with the enhanced entity's description
                G.nodes[entity['entity_name']].update({'description': entity['description']})
        
        # Update edges where the source or target node matches the original name
        for edge in G.edges(data=True):
            source, target, edge_data = edge
            
            if original_name == source or original_name == target:
                # Modify source_id and target_id in the edge based on the entity_name
                if original_name == source:
                    source = entity['entity_name']
                if original_name == target:
                    target = entity['entity_name']

    # Step 3: Save the updated graph to a new GraphML file
    nx.write_graphml(G, enhanced_img_kg_path)
    return enhanced_img_kg_path

def image_knowledge_graph_update(enhanced_img_kg_path, image_entity_name):
    image_data = get_image_data()
    text_chunks = get_text_chunks()
    chunk_knowledge_graph = get_chunk_knowledge_graph()
    img_kg_path = enhanced_img_kg_path
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    new_img_kg_path = os.path.join(working_dir, f'images/{image_entity_name}/new_graph_{image_entity_name}_entity_relation.graphml')
    
    image_entity = align_single_image_entity(image_entity_name, text_chunks)
    chunk_order_index = image_data[image_entity_name].get("chunk_order_index")
    nearby_chunks = get_nearby_chunks(text_chunks, chunk_order_index)
    nearby_text_entity_list = get_nearby_entities(chunk_knowledge_graph, chunk_order_index)
    nearby_relationship_list = get_nearby_relationships(chunk_knowledge_graph, chunk_order_index)
    
    
    if image_entity is not None:
        image_entity_name = image_entity.get("entity_name","no_match")
        image_entity_description = image_entity.get("description","None.")
    else:
        return img_kg_path
    
    if image_entity_name == "no_match" or image_entity_name == "no match":
        return img_kg_path
    
    possible_image_matched_entities = get_possible_entities_image_clustering(image_entity_description, nearby_text_entity_list, nearby_relationship_list) 
    matched_entity_name = judge_image_entity_alignment(image_entity_name, image_entity_description, possible_image_matched_entities, nearby_chunks)
    
    matched_entity_name_normalized = matched_entity_name.strip().replace(" ", "").replace("\\", "").lower()
    
    G = nx.read_graphml(img_kg_path)
    matched_entity_found = False  # Flag to check if matched_entity_name is found
    
    for entity in nearby_text_entity_list:
        entity_name_normalized = entity["entity_name"].strip().replace(" ", "").replace("\\", "").lower()
        
        if matched_entity_name_normalized == entity_name_normalized:
            matched_entity_found = True
            source_node_id = None
            for node, data in G.nodes(data=True):
                if data.get('entity_type') == '"ORI_IMG"' or '"UNKNOWN"':
                    source_node_id = node
                    break

            if source_node_id is None:
                raise ValueError("No node with entity_type 'ORI_IMG' found in the graph.")
            
            if len(G.edges(data=True)) > 0:
                first_edge = list(G.edges(data=True))[0]
                data_source_id_value = first_edge[2].get("source_id", "")
                data_order_value = first_edge[2].get("order", "")
            else:
                # Handle the case where there are no edges
                data_source_id_value = G.nodes[source_node_id].get('source_id', '')
                data_order_value = 1  # Set a default

            entity["entity_name"] = ensure_quoted(entity["entity_name"])
            G.add_edge(source_node_id, entity["entity_name"], 
                       weight=10.0, 
                       description=f"{source_node_id} is the image of {entity['entity_name']}.",
                       source_id=data_source_id_value,
                       order=data_order_value)
            G.add_node(entity["entity_name"], 
                        entity_type=entity["entity_type"], 
                        description=entity["description"],
                        source_id=data_source_id_value)
            
            # Save the updated graph to a new path
            nx.write_graphml(G, new_img_kg_path)
            break  # Exit the loop as we have processed the matched entity
    
    if not matched_entity_found:
        # If matched entity is not found, add a new node for image_entity_name
        source_node_id = None
        for node, data in G.nodes(data=True):
            if data.get('entity_type') == '"ORI_IMG"' or '"UNKNOWN"':
                source_node_id = node
                break
        
        if source_node_id is None:
            raise ValueError("No node with entity_type 'ORI_IMG' found in the graph.")
        
        
        # Create a relationship between the new node and the "ORI_IMG" node
        if len(G.edges(data=True)) > 0:
            first_edge = list(G.edges(data=True))[0]
            source_id_value = first_edge[2].get("source_id", "")
            order_value = first_edge[2].get("order", "")
        else:
            # Handle the case where there are no edges
            source_id_value = G.nodes[source_node_id].get('source_id', '')
            order_value = 1  # Set a default

        # Add new node with the image entity's name and description
        G.add_node(image_entity_name, 
                   entity_type="IMG_ENTITY", 
                   description=image_entity_description,
                   source_id=source_id_value)
        
        G.add_edge(source_node_id, image_entity_name, 
                   weight=10.0, 
                   description=f"{source_node_id} is the image of {image_entity_name}.",
                   source_id=source_id_value,
                   order=order_value)
        
        # Save the updated graph to a new path
        nx.write_graphml(G, new_img_kg_path)
    
    return new_img_kg_path

def merge_graphs(image_graph_path, graph_path, aligned_text_entity_list, image_entity_name):
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    merged_kg_path = os.path.join(working_dir, f'graph_merged_{image_entity_name}.graphml')

    # 步骤 1: 加载图像和文本知识图谱
    image_graph = nx.read_graphml(image_graph_path)
    text_graph = nx.read_graphml(graph_path)
    
    # 如果图谱加载失败，打印错误并返回
    if image_graph is None or text_graph is None:
        print(f"加载图谱失败，请检查文件路径。")
        return
    
    # 步骤 2: 合并两个图
    # 使用nx.compose将两个图谱的节点和边合并
    merged_graph = nx.compose(image_graph, text_graph)
    # 步骤 3: 遍历对齐的实体列表，进行融合
    for entity_info in aligned_text_entity_list:
        # 检查结果是否有问题，如果存在缺失的字段，则跳过该实体
        if not all(key in entity_info for key in ['entity_name', 'entity_type', 'description', 'source_image_entities', 'source_text_entities']):
            continue 
        entity_name = entity_info['entity_name']  # 融合后的实体名称
        entity_type = entity_info['entity_type']  # 融合后的实体类型
        description = entity_info['description']  # 融合后的实体描述
        
        # 获取图像和文本实体对应的节点
        source_image_entities = entity_info['source_image_entities']
        source_text_entities = entity_info['source_text_entities']
        
        # 获取图像和文本图谱中的source_id时，确保去掉引号
        source_image_entity = ensure_quoted(source_image_entities[0])
        source_text_entity = ensure_quoted(source_text_entities[0])
        
        # 确保图中存在这些节点
        if source_image_entity in image_graph.nodes:
            source_id_image = image_graph.nodes[source_image_entity].get('source_id', '')
        else:
            print(f"节点 {source_image_entity} 在图像图谱中不存在")
            continue

        if source_text_entity in text_graph.nodes:
            source_id_text = text_graph.nodes[source_text_entity].get('source_id', '')
        else:
            print(f"节点 {source_text_entity} 在文本图谱中不存在")
            continue
        # 将两个source_id连接起来
        source_id = GRAPH_FIELD_SEP.join([source_id_image, source_id_text])

        # 步骤 4: 融合节点
        # 假设 source_image_entities[0] 是目标节点，将所有其他实体连接到该节点
        target_entity = ensure_quoted(source_image_entities[0])

        # 合并 source_image_entities 和 source_text_entities 中的所有实体
        all_entities = source_image_entities + source_text_entities

        # 确保去掉重复的实体
        all_entities = list(set(all_entities))  # 去重

        # 先遍历所有实体，将它们与目标实体连接
        for entity in all_entities:
            entity = ensure_quoted(entity)
            if entity != target_entity and entity in merged_graph.nodes: 
                neighbors = list(merged_graph.neighbors(entity))  # 获取当前实体的邻居
                for neighbor in neighbors:
                    if not merged_graph.has_edge(target_entity, neighbor):
                        merged_graph.add_edge(target_entity, neighbor)  # 将邻居与目标实体连接
                    # 将边的属性合并到目标节点的边
                    edge_data = merged_graph.get_edge_data(entity, neighbor)
                    target_edge_data = merged_graph.get_edge_data(target_entity, neighbor)
                    
                    # 合并边的属性（传递所有属性，如 weight、description、source_id 和 order）
                    if target_edge_data:
                        # 如果边已经存在，合并现有的属性
                        for key in edge_data:
                            if key in ['weight', 'description', 'source_id', 'order']:
                                # 合并边的属性，如果已有就添加新值，或者保持已有的
                                target_edge_data[key] = edge_data.get(key, target_edge_data.get(key))
                    else:
                        # 如果边不存在，就添加新的边属性
                        merged_graph[target_entity][neighbor].update(edge_data)

                merged_graph.remove_node(entity)  # 删除已经合并的实体节点
        
        # 在更新之前，检查目标节点是否存在，如果不存在则创建
        if target_entity not in merged_graph.nodes:
            merged_graph.add_node(target_entity)
        # 修改目标节点的属性
        merged_graph.nodes[target_entity].update({
            'entity_type': entity_type,
            'description': description,
            'source_id': source_id
        })
        merged_graph = nx.relabel_nodes(merged_graph, {target_entity: ensure_quoted(entity_name)})
    
    # 步骤 5: 保存合并后的图谱
    # 使用NetworkX将合并后的图保存到指定路径
    nx.write_graphml(merged_graph, merged_kg_path)
    logger.info(f"合并后的知识图谱已保存到: {merged_kg_path}")
    return merged_kg_path

async def fusion(img_ids):
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    working_dir = global_config['working_dir']
    graph_path = os.path.join(working_dir, 'graph_chunk_entity_relation.graphml')
    for image_entity_name in img_ids:
        merged_kg_path = os.path.join(working_dir, f'graph_merged_{image_entity_name}.graphml')
        if os.path.exists(merged_kg_path):
            continue
        aligned_text_entity_list = image_knowledge_graph_alignment(image_entity_name)
        enhanced_img_kg_path = enhanced_image_knowledge_graph(aligned_text_entity_list, image_entity_name)
        image_graph_path = image_knowledge_graph_update(enhanced_img_kg_path, image_entity_name)
        graph_path = merge_graphs(image_graph_path, graph_path, aligned_text_entity_list, image_entity_name)
    createvdb = create_EntityVDB()
    return await createvdb.create_vdb()