from dataclasses import dataclass
from collections import defaultdict,Counter 
from typing import Type,cast,Union
from functools import partial
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

import numpy as np
import cv2
import asyncio
import os
import base64
import shutil
import re
import json

from prompt import GRAPH_FIELD_SEP,PROMPTS
from llm import multimodel_if_cache,model_if_cache
from base import (
    logger,
    clean_str,
    limit_async_func_call,
    read_config_to_dict,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    is_float_regex,
    split_string_by_multi_markers,
)
from storage import (
    BaseGraphStorage,
    StorageNameSpace,
    BaseKVStorage,
    JsonKVStorage,
    NetworkXStorage,
)

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

# 提取原始图像（单张）的特征块，并单独保存
async def extract_feature_chunks(single_image_path):
    cache_path = os.getenv('CACHE_PATH')
    # 获取全局设置
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    
    # 保存路径为feature_images文件夹下的原始图像名称
    save_dir = f"{global_config['working_dir']}/images/{Path(single_image_path).stem}"
    # 如果保存路径不存在，则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(global_config["working_dir"], 'kv_store_image_data.json')
    # 加载image_data，根据image_path获取current_image_data
    with open(path, 'r', encoding='utf-8') as file:
        image_data = json.load(file)
    segmentation = False
    for _, value in image_data.items():
        if value.get("image_path") == single_image_path:
            segmentation = value.get("segmentation", False)

    if not segmentation:
        return save_dir
    
    # 加载模型，这里是官方提供的默认模型，效果一般
    yolo_path = os.path.join(cache_path,"yolov8n-seg.pt")
    model = YOLO(yolo_path)
    # 执行预测
    results = model(single_image_path, device='cpu')

    # 迭代检测结果，因为默认输出results的结果是列表
    for result in results:
        # 拷贝原始图像
        img = np.copy(result.orig_img)
        # 获取图像的文件名
        img_name = Path(result.path).stem

        # 遍历检测到的目标轮廓 
        for ci, c in enumerate(result):
            # 获取目标类别
            label = c.names[c.boxes.cls.tolist().pop()]

            # 创建一个与原图像大小相同的空白掩膜，用于存放目标的掩膜轮廓
            b_mask = np.zeros(img.shape[:2], np.uint8)

            # 获取目标的轮廓信息，并将其转换为适合OpenCV处理的整数类型 
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)

            # 使用OpenCV绘制轮廓到掩膜中，填充轮廓区域
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

            # 将原始图像与掩膜结合，形成带有黑色背景的分割结果
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, img)
            # 透明背景
            # isolated = np.dstack([img, b_mask]) 

            # 获取边界框信息，将孤立对象裁剪到其边界框内
            x1, y1, x2, y2 = c.boxes.xyxy[0].cpu().numpy().astype(np.int32)
            iso_crop = isolated[y1:y2, x1:x2]

            # 创建完整的保存路径，包含保存目录、图片文件名、目标标签和序号
            save_path = Path(save_dir) / f"{img_name}_{label}-{ci}.jpg"

            # 将分割出的目标保存为JPG文件
            _ = cv2.imwrite(str(save_path),iso_crop)
    return save_dir

async def feature_image_entity_construction(feature_image_path,use_llm_func):
    entities = []

    # 检查文件夹是否为空（没有 .jpg 图片）
    if not any(filename.lower().endswith('.jpg') for filename in os.listdir(feature_image_path)):
        return entities  # 返回空列表

    feature_prompt_user = PROMPTS["feature_image_description_user"]
    feature_prompt_system = PROMPTS["feature_image_description_system"]
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"]
    for filename in os.listdir(feature_image_path):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(feature_image_path, filename)
            # 打开图像并检查尺寸
            with Image.open(image_path) as image:
                width, height = image.size
            if width > 28 and height > 28:
                # 读取并编码为 Base64
                with open(image_path, "rb") as image_file:
                    img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                
                # 调用 LLM 函数生成描述
                description = await use_llm_func(
                    user_prompt=feature_prompt_user,
                    img_base=img_base64,
                    system_prompt=feature_prompt_system
                )
                
                # 构建实体字符串
                entity = f'("entity"{tuple_delimiter}"{filename}"{tuple_delimiter}"img"{tuple_delimiter}"{description}"){record_delimiter}'
                # 替换多余符号，确保格式为你期望的 <|> 分隔符格式
                entity = entity.replace("('", "(").replace("')", ")")
                entities.append(entity.replace("\n", ""))
            else:
                logger.info(f"Image {image_path} dimensions too small: width={width}, height={height}")
                os.remove(image_path)
    return entities

async def feature_image_relationship_construction(feature_image_path,image_entities,use_llm_func):
    relationships = []

    # 检查文件夹是否为空（没有 .jpg 图片）
    if not any(filename.lower().endswith('.jpg') for filename in os.listdir(feature_image_path)):
        return relationships  # 返回空列表

    feature_prompt_user = PROMPTS["entity_alignment_user"]
    feature_prompt_system = PROMPTS["entity_alignment_system"]
    context_base_1 = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"]
        )
    prompt_system = feature_prompt_system.format(**context_base_1)

    for filename in os.listdir(feature_image_path):
        if filename.lower().endswith('.jpg'):
            context_base_2 = dict(
                entity_description = image_entities,
                feature_image_name = filename
                )
            prompt_user = feature_prompt_user.format(**context_base_2)
            image_path = os.path.join(feature_image_path, filename)
            with open(image_path, "rb") as image_file:
                img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                relationship = await use_llm_func(
                                    user_prompt = prompt_user,
                                    img_base = img_base64,
                                    system_prompt = prompt_system
                                )
                relationships.append(relationship)
    return relationships

async def extract_entities_from_image(single_image_path,use_llm_func):
    image_entity_extract_prompt = PROMPTS["image_entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )

    image_entity_system = image_entity_extract_prompt.format(**context_base)

    image_entity_user = """
    Please output the results in the format provided in the example.
    Output:
    """
    with open(single_image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    result = await use_llm_func(
                    user_prompt = image_entity_user,
                    img_base = img_base64,
                    system_prompt = image_entity_system   
            )
    return result

async def entity_of_original_image(image_path,result1,result2):
    # 初始化为列表
    result4 = []
    # 获取全局设置
    cache_path = os.getenv('CACHE_PATH')
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    path = os.path.join(global_config["working_dir"], 'kv_store_image_data.json')
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"]
    # 加载image_data，根据image_path获取current_image_data
    with open(path, 'r', encoding='utf-8') as file:
        image_data = json.load(file)
    for image_key, image_info in image_data.items():
        if image_info['image_path'] == image_path:
            current_image_data = image_info
            filename = image_key
            break
    # 根据current_image_data正则化为entity变量
    description = current_image_data["description"]
    entity = f'("entity"{tuple_delimiter}"{filename}"{tuple_delimiter}"ori_img"{tuple_delimiter}"{description}"){record_delimiter}'
    # 替换多余符号，确保格式为你期望的 <|> 分隔符格式
    entity = entity.replace("('", "(").replace("')", ")")
    result4.append(entity.replace("\n",""))
    # 根据result1正则化relationship变量
    entity_name_pattern = r'\"([^\"]+?\.jpg)\"'
    for entity_feature in result1:
        entity_name = re.findall(entity_name_pattern, entity_feature)[0]
        if entity_name:
            relationship1 = f'("relationship"{tuple_delimiter}"{entity_name}"{tuple_delimiter}"{filename}"{tuple_delimiter}"{entity_name}是{filename}的图像特征块。"{tuple_delimiter}10){record_delimiter}'
            result4.append(relationship1)
    # 根据result2正则化relationship变量
    entity_name_pattern2 = r'\"entity\"<\|>\"([^\"]+?)\"'
    entity_names = re.findall(entity_name_pattern2, result2)
    for entity_name2 in entity_names:
        relationship2 = f'("relationship"{tuple_delimiter}"{entity_name2}"{tuple_delimiter}"{filename}"{tuple_delimiter}"{entity_name2}是从{filename}中提取的实体。"{tuple_delimiter}10){record_delimiter}'
        result4.append(relationship2)
    return result4

def format_result(result):
    pattern = r'\("entity"<\|>"([^"]+)"<\|>"[^"]*"<\|>"([^"]+)"\)'
    entities = re.findall(pattern, result)
    formatted_result = "\n".join([f'"{entity}"-"{description}"' for entity, description in entities])
    return formatted_result

async def extract_entities(
    cache_dir: BaseKVStorage,
    image_path: str,
    feature_image_path:str,
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    model_max_async :int = 16
    # 限制模型函数的异步调用次数，并为其配置哈希键值存储
    use_llm_func = limit_async_func_call(model_max_async)(
        partial(multimodel_if_cache, hashing_kv=cache_dir)
    )
    
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )

    result1 = await feature_image_entity_construction(feature_image_path,use_llm_func)
    result2 = await extract_entities_from_image(image_path,use_llm_func)
    formatted_result2 = format_result(result2)
    result3 = await feature_image_relationship_construction(feature_image_path,formatted_result2,use_llm_func)
    result4 = await entity_of_original_image(image_path,result1,result2)
    final_result = "\n" + "\n".join(result1 + result3 + result4) + result2.strip()

    # 将最终提取的结果按照多个分隔符分割成记录
    records = split_string_by_multi_markers(
        final_result,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )
    # 使用 defaultdict 来存储可能的节点和边
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    # 为当前图像初始化结果字典
    image_results = {
        "image_path": image_path,
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
            record_attributes, image_path
        )
        if if_entities is not None:
            maybe_nodes[if_entities["entity_name"]].append(if_entities)
            image_results["entities"].append(if_entities)
            continue
        # 处理关系提取
        if_relation = await _handle_single_relationship_extraction(
            record_attributes, image_path
        )
        if if_relation is not None:
            maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                if_relation
            )
            image_results["relationships"].append(if_relation)
    m_nodes = defaultdict(list)
    m_edges = defaultdict(list)
    for k, v in maybe_nodes.items():
        m_nodes[k].extend(v)
    for k, v in maybe_edges.items():
        # 构建无向图时，按字典序排序边的节点
        m_edges[tuple(sorted(k))].extend(v)
    # 合并节点数据并更新知识图谱
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in m_nodes.items()
        ]
    )
    # 合并边数据并更新知识图谱
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in m_edges.items()
        ]
    )
    # 如果没有提取到任何实体，发出警告并返回 None
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    return knwoledge_graph_inst

@dataclass
class extract_entities_from_single_image:
    # 实体提取函数
    image_entity_extraction_func: callable = extract_entities

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
        self.multimodel_llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="multimodel_llm_response_cache", global_config=global_config
            )
        )
        # 初始化图存储类实例，用于存储块实体关系图
        self.image_entity_relation_graph = self.graph_storage_cls(
            namespace="image_entity_relation", global_config=global_config
        )
    async def single_image_entity_extraction(self, image_path):
        try:
            # 获取全局设置
            cache_path = os.getenv('CACHE_PATH')
            global_config_path = os.path.join(cache_path,"global_config.csv")
            global_config = read_config_to_dict(global_config_path)
            # yolo图像分割
            feature_image_path = await extract_feature_chunks(image_path)
            # ---------- extract/summary entity and upsert to graph
            # 提取新实体和关系，并更新到知识图谱中
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.image_entity_extraction_func(
                self.multimodel_llm_response_cache,
                image_path,
                feature_image_path,
                knwoledge_graph_inst=self.image_entity_relation_graph,
                global_config=global_config,
            )
            if maybe_new_kg is None:
                logger.warning("No entities found")
                return
            self.image_entity_relation_graph = maybe_new_kg
        finally:
            await self._single_image_entity_extraction_done()
    async def _single_image_entity_extraction_done(self):
        tasks = []
        for storage_inst in [
            self.multimodel_llm_response_cache,
            self.image_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

async def img2graph(image_path):
    # 获取全局设置
    cache_path = os.getenv('CACHE_PATH')
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    # 获取目录下所有的 .jpg 文件
    jpg_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    # 完整路径
    jpg_file_paths = [os.path.join(image_path, f) for f in jpg_files]
    for single_image_path in jpg_file_paths:
        extraction = extract_entities_from_single_image()
        if single_image_path.lower().endswith(('.jpg')):
            await extraction.single_image_entity_extraction(single_image_path)
            # 保存路径为feature_images文件夹下的原始图像名称
            destination_dir = f"{global_config['working_dir']}/images/{Path(single_image_path).stem}/graph_{Path(single_image_path).stem}_entity_relation.graphml"
            source = f"{global_config['working_dir']}/graph_image_entity_relation.graphml"
            shutil.move(source,destination_dir)
    return