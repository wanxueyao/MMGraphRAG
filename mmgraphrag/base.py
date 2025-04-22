import logging
import asyncio
from dataclasses import dataclass
import numpy as np
from hashlib import md5
from functools import wraps
from typing import Any
import tiktoken
import re
import os
import html
import json
import numbers
logger = logging.getLogger("multimodal-graphrag")

ENCODER = None

@dataclass
class EmbeddingFunc:
    """
    定义一个用于嵌入的函数类。

    参数:
    - embedding_dim: 嵌入向量的维度
    - max_token_size: 最大令牌大小
    - func: 可调用的对象，用于执行嵌入操作
    """
    embedding_dim: int
    max_token_size: int
    func: callable
    async def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        调用嵌入函数并返回结果。

        参数:
        - *args: 位置参数
        - **kwargs: 关键字参数

        返回:
        - np.ndarray: 嵌入结果
        """
        return await self.func(*args, **kwargs)

# -----------------------------------------------------------------------------------
# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input
    # 移除 HTML 转义字符
    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    # 移除控制字符和其他不需要的字符
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]

def pack_user_ass_to_openai_messages(*args: str):
    """
    将用户和助手的对话打包为OpenAI消息格式。

    该函数接受一系列字符串参数，成对地将它们包装成交替的用户和助手角色的消息。
    这对于将对话历史记录转换为可供OpenAI的API处理的格式特别有用。在_op.py里就是调用给history的。

    参数:
    *args (str): 一个或多个字符串参数，表示用户和助手之间的对话交替发言。

    返回:
    list: 一个字典列表，每个字典包含两个键值对:
        - 'role': 表示消息发送者的角色，根据参数序列中的位置交替为'user'或'assistant'。
        - 'content': 发送者发送的消息内容，来自输入参数序列中的对应位置。
    """
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]
# 计算参数的哈希值
def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()

# 计算md5哈希值
def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

# 使用指定的模型encode字符串，并返回tokens数量
def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens

# 使用指定的模型decode字符串，并返回tokens数量
def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content

# 判断是否为浮点数
def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))

def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""
    """
    为异步函数添加最大并发调用次数的限制。

    参数:
    - max_size: int, 允许的最大并发调用次数。
    - waitting_time: float, 当达到最大并发调用次数时，每次检查间隔的时间（秒），默认为0.0001秒。

    返回:
    - 返回一个装饰器函数，用于包装需要限制并发调用的异步函数。
    """
    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            # 如果当前调用数量达到最大值，等待一段时间后再检查
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro

# 将json对象写入文件
def write_json(json_obj, file_name):
    with open(file_name, "w", encoding='utf-8') as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

# 从文件中加载json对象
def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name) as f:
        return json.load(f)

import ast

def parse_value(value):
    """
    解析字符串值，将其转换为适当的 Python 类型（字典、数字、字符串等）。
    """
    try:
        # 使用 ast.literal_eval 解析字典或其他复杂结构
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # 如果无法解析，则返回原始字符串
        return value

def read_config_to_dict(file_path):
    config_dict = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            key, value = line.strip().split(',', 1)  # 仅分割一次
            config_dict[key] = parse_value(value)
    
    return config_dict

def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    """
    根据token大小截断列表数据。

    该函数的目的是确保列表中数据的总token数不超过指定的最大token大小。
    当数据的总token数超过最大允许大小时，函数将返回截断后的列表。

    参数:
    - list_data: list, 需要截断的列表，其中每个元素为一个数据项。
    - key: callable, 用于从列表数据项中提取用于计算token大小的字符串的函数。
    - max_token_size: int, 允许的最大token大小，用于决定列表数据的截断点。

    返回:
    - 截断后的列表。如果max_token_size小于等于0，返回空列表。

    注意:
    - 该函数使用tiktoken对字符串进行编码并计算token数量，请确保在使用前已安装tiktoken库。
    - 截断操作基于累计token数量首次超过max_token_size发生的索引位置。
    """
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data

# 将字符串括在双引号中
def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'

# 将多维列表转换为CSV格式
def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )

def get_latest_graphml_file(folder_path):
    # 正则表达式，用于匹配文件名中的数字部分
    pattern = r'graph_merged_image_(\d+)\.graphml'
    
    max_number = -1
    latest_file = None
    namespace = 'chunk_entity_relation'
    file_path = None
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否匹配目标格式
        match = re.match(pattern, filename)
        if match:
            # 提取文件名中的数字部分
            file_number = int(match.group(1))
            # 如果该数字比当前的最大值大，则更新最大值和文件名
            if file_number > max_number:
                max_number = file_number
                namespace = f"merged_image_{max_number}"
                latest_file = filename
                file_path = os.path.join(folder_path, latest_file)
    # 如果没有匹配的文件名，返回默认的 file_path
    if file_path is None:
        file_path = os.path.join(folder_path, 'graph_chunk_entity_relation.graphml')
    return namespace, file_path