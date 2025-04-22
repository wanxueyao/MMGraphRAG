from openai import AsyncOpenAI, OpenAI
from parameter import EMBED_MODEL, encode, API_KEY, MODEL, URL, MM_API_KEY, MM_MODEL, MM_URL
import numpy as np
import re
import json

from base import wrap_embedding_func_with_attrs,compute_args_hash
from storage import (
    BaseKVStorage,
)

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)

async def local_embedding(texts: list[str]) -> np.ndarray:
    return encode(texts)

async def model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_client = OpenAI(
        api_key=API_KEY, base_url=URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    # 如果缓存对象存在，计算当前请求的哈希值，尝试从缓存中获取结果
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = openai_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # 如果有缓存对象，将响应结果存入缓存
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content

async def multimodel_if_cache(
    user_prompt, img_base, system_prompt, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=MM_API_KEY, base_url=MM_URL
    )
    messages = []
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "system", "content": [
          {
            "type": "text",
            "text": system_prompt
          }
        ]})
    messages.append({"role": "user", "content": [
          {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base}"},
          },
          {
            "type": "text",
            "text": user_prompt
          }
        ]})
    # 如果缓存对象存在，计算当前请求的哈希值，尝试从缓存中获取结果
    if hashing_kv is not None:
        args_hash = compute_args_hash(MM_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=MM_MODEL, messages=messages, **kwargs
    )

    # 如果有缓存对象，将响应结果存入缓存
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MM_MODEL}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content

# 正则化处理函数
def normalize_to_json(output):
    # 使用正则提取JSON部分
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            # 验证JSON格式是否正确
            json_obj = json.loads(json_str)
            return json_obj  # 返回标准化的JSON对象
        except json.JSONDecodeError as e:
            print(f"JSON解码失败: {e}")
            return None
    else:
        print("未找到有效的JSON部分")
        return None

def normalize_to_json_list(output):
    """
    提取并验证JSON列表格式的字符串，返回解析后的JSON对象列表。
    即使JSON不完整，也尝试提取尽可能多的内容。
    """
    # 去除转义符和多余空白符
    cleaned_output = output.replace('\\"', '"').strip()
    
    # 使用宽松的正则表达式提取可能的JSON片段
    match = re.search(r"\[\s*(\{.*?\})*?\s*]", cleaned_output, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        
        # 移除多余逗号（可能由于截断导致多余逗号）
        json_str = re.sub(r",\s*]", "]", json_str)
        json_str = re.sub(r",\s*}$", "}", json_str)

        try:
            # 尝试完整解析
            json_obj = json.loads(json_str)
            if isinstance(json_obj, list):
                return json_obj
        except json.JSONDecodeError:
            # 如果完整解析失败，尝试逐项解析
            print("完整解析失败，尝试逐项解析...")
            items = []
            for partial_match in re.finditer(r"\{.*?\}", json_str, re.DOTALL):
                try:
                    item = json.loads(partial_match.group(0))
                    items.append(item)
                except json.JSONDecodeError:
                    print("跳过无效的JSON片段")
            return items if items else []
    else:
        print("未找到有效的JSON片段")
        return []

# 用LLM进行回答
def get_llm_response(cur_prompt, system_content):
    client = OpenAI(
        base_url=URL, api_key=API_KEY
    )

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": cur_prompt},
        ],
    )

    response = completion.choices[0].message.content
    return response

# 调用多模态LLM
def get_mmllm_response(cur_prompt, system_content, img_base):
    client = OpenAI(
        base_url=MM_URL, api_key=MM_API_KEY
    )

    completion = client.chat.completions.create(
        model=MM_MODEL,
        messages=[
            {"role": "system", "content": [
                    {
                        "type": "text",
                        "text": system_content
                    }
                    ]},
            {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base}"},
                    },
                    {
                        "type": "text",
                        "text": cur_prompt
                    }
                    ]},
        ],
    )

    response = completion.choices[0].message.content
    return response

