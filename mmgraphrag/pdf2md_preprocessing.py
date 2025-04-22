import re
import asyncio
import os
import json
import base64
import shutil
import subprocess
from typing import Callable, Dict, List, Optional, Type, Union, cast
from dataclasses import dataclass
from parameter import mineru_dir

from base import (
    logger,
    compute_mdhash_id,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    read_config_to_dict,
)

from storage import (
    BaseKVStorage,
    JsonKVStorage,
    StorageNameSpace,
)
from llm import multimodel_if_cache
from prompt import PROMPTS

def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    """
    根据token大小对文本进行分块。

    该函数用于将给定的文本内容按照指定的token大小限制进行分块，同时保证相邻块之间有重叠。
    主要用于处理大文本，使其能够适应如OpenAI的GPT系列模型的输入限制。

    参数:
    - content: str, 待分块的文本内容。
    - overlap_token_size: int, 默认128. 相邻文本块之间的重叠token数。
    - max_token_size: int, 默认1024. 每个文本块的最大token数。
    - tiktoken_model: str, 默认"gpt-4o". 用于token化和去token化的tiktoken模型。

    返回:
    - List[Dict[str, Any]], 包含每个文本块的tokens数量、文本内容和块顺序索引的列表。
    """
    # 使用指定的tiktoken模型对文本进行token化
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    # 初始化存储分块结果的列表
    results = []
    # 遍历tokens，根据max_token_size和overlap_token_size进行分块
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        # 根据当前分块的起始位置和最大token数限制，获取分块的tokens
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        # 将当前分块的tokens数量、文本内容和块顺序索引添加到结果列表中
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results

@dataclass
class text_chunking_func:
    # 分块
    chunk_func: Callable[[str, Optional[int], Optional[int], Optional[str]], List[Dict[str, Union[str, int]]]] = chunking_by_token_size
    # 分块大小
    chunk_token_size: int = 1200
    # 分块重叠数量
    chunk_overlap_token_size: int = 100
    

    # 键值存储，json，具体定义在storage.py
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    # 获取全局设置
    cache_path = os.getenv('CACHE_PATH')
    global_config_path = os.path.join(cache_path,"global_config.csv")
    global_config = read_config_to_dict(global_config_path)
    
    # tiktoken使用的模型名字，默认为gpt-4o，moonshot-v1-32k可以通用
    tiktoken_model_name = global_config["tiktoken_model_name"]

    def __post_init__(self):
        # 获取全局设置
        cache_path = os.getenv('CACHE_PATH')
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        # 初始化存储类实例，用于存储完整文档
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config = global_config
        )
        # 初始化存储类实例，用于存储文本块
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config = global_config
        )
    
    async def text_chunking(self,string_or_strings):
        try:
            # 如果输入是一个字符串，将其转换为列表
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            # 将字符串或字符串列表string_or_strings中的每个元素去除首尾空白后，作为文档内容。
            # 计算其MD5哈希值并添加前缀doc-作为键，内容本身作为值，生成一个新的字典new_docs。
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            # 筛选出需要添加的新文档ID
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            # 根据筛选结果更新新文档字典
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            # 如果没有新文档需要添加，记录日志并返回
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            # 记录插入新文档的日志
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking
            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                # 为每个文档生成片段
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in self.chunk_func(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            # 筛选出需要添加的新片段ID
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            # 根据筛选结果更新新片段字典
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            # 如果没有新片段需要添加，记录日志并返回
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            # 提交所有更新和索引操作
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._text_chunking_done()
    async def _text_chunking_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

image_description_prompt_user = PROMPTS["image_description_user"]
image_description_prompt_user_examples = PROMPTS["image_description_user_with_examples"]
image_description_prompt_system = PROMPTS["image_description_system"]

async def get_image_description(image_path, caption, footnote, context):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    caption_string = " ".join(caption) 
    footnote_string = " ".join(footnote) 
    user_prompt = image_description_prompt_user_examples.format(caption=caption_string, footnote=footnote_string, context=context)
    result =  await multimodel_if_cache(user_prompt=user_prompt,img_base=base64_image,system_prompt=image_description_prompt_system)
    
    description_match = re.search(r'"description": "([^"]*)"', result)
    segmentation_match = re.search(r'"segmentation": (\w+)', result)
    # 获取匹配的值
    if description_match:
        image_description = description_match.group(1)
    else:
        image_description = "No description."

    if segmentation_match:
        segmentation_str = segmentation_match.group(1)
    else:
        segmentation_str = "false"
    segmentation = True if segmentation_str.lower() == 'true' else False
    return image_description, segmentation

def find_chunk_for_image(text_chunks, context):
    """
    根据图片前后文本找到其所属 chunk。
    优先选择包含更多连续字符的 chunk，忽略换行符。
    """
    best_chunk_id = None
    best_match_count = 0

    # 如果合并的文本为空，则返回 None
    if not context:
        return None

    # 遍历所有的 chunk
    for chunk_id, chunk_data in text_chunks.items():
        # 去掉 chunk 中的换行符
        chunk_content = chunk_data['content'].replace('\n', '')

        # 计算组合文本与 chunk 内容的匹配度（基于词语匹配）
        match_count = sum(1 for word in context.split() if word in chunk_content)

        # 如果当前 chunk 的匹配度最高，则选择它
        if match_count > best_match_count:
            best_match_count = match_count
            best_chunk_id = chunk_id

    return best_chunk_id
def compress_image_to_size(input_image, output_image_path, target_size_mb=5, step=10, quality=90):
    """
    将图片压缩到目标大小以内（以MB为单位）。

    参数:
    input_image (PIL.Image): 输入图片的 PIL 对象。
    output_image_path (str): 输出图片的路径。
    target_size_mb (int): 目标大小，以MB为单位，默认为5MB。
    step (int): 每次降低的质量步长，默认为10。
    quality (int): 初始保存的图片质量，默认为90。

    返回:
    bool: 是否成功压缩到目标大小以内。
    """
    target_size_bytes = target_size_mb * 1024 * 1024  # 将目标大小转换为字节

    # 先保存图片并检查大小
    img = input_image
    img.save(output_image_path, quality=quality)
    if os.path.getsize(output_image_path) <= target_size_bytes:
        return True

    # 尝试逐步降低质量，直到图片大小小于目标大小
    while os.path.getsize(output_image_path) > target_size_bytes and quality > 10:
        quality -= step
        img.save(output_image_path, quality=quality)
    
    # 检查最终大小是否符合要求
    if os.path.getsize(output_image_path) <= target_size_bytes:
        return True
    else:
        print("无法将图片压缩到目标大小以内，请在preprocessing.py中调整初始质量或步长。")
        return False

def clear_images_in_md(content):
    # 使用正则表达式找到所有符合格式的图片内容并删除
    content = re.sub(r'!\[\]\([^)]*\)', '', content)
    return content

def image_move_remove(json_path, target_folder, folder_path):
    # 加载 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 计数器从1开始
    img_counter = 1

    # 遍历数据中的每个条目
    for entry in data:
        if 'img_path' in entry:
            # 获取当前图像路径
            img_path = entry['img_path']
            original_img_path = os.path.join(folder_path, img_path)
            # 设置新的图像文件名
            new_img_name = f"image_{img_counter}.jpg"
            new_img_path = os.path.join(target_folder, new_img_name)
            
            # 复制并重命名图像文件
            try:
                # 检查 original_img_path 是否是文件而非目录
                if os.path.isfile(original_img_path):
                    shutil.copy(original_img_path, new_img_path)
                    print(f"Copied {original_img_path} to {new_img_path}")
                    img_counter += 1  # 增加计数器
                else:
                    print(f"Skipped {original_img_path}, because it is a directory.")
            except FileNotFoundError:
                print(f"Image not found: {original_img_path}")

def get_content_list_json_file(folder_path):
    # 遍历指定文件夹中的文件
    for file_name in os.listdir(folder_path):
        # 仅选择以 "_content_list.json" 结尾的文件
        if file_name.endswith('_content_list.json'):
            file_path = os.path.join(folder_path, file_name)
    return file_path

def rename_images_in_json(data):
    # 初始化计数器
    image_counter = 1
    
    # 遍历 JSON 数据，找到所有 img_path 并重命名
    for item in data:
        if "img_path" in item and item["img_path"] != "":
            new_image_path = f"images/image_{image_counter}.jpg"
            # 更新 JSON 中的 img_path
            item["img_path"] = new_image_path
            image_counter += 1
    return data

# 用minerU对pdf预处理，输出结果保存在输出文件夹中
def pdf2markdown(pdf_path, output_dir):
    # 获取 PDF 文件的名称（去掉扩展名）
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    mineru_folder = os.path.join(mineru_dir, pdf_name, "auto")
    if os.path.exists(mineru_folder) and any(file.endswith(".md") for file in os.listdir(mineru_folder)):
        logger.info(f"MinerU already finished!")
        return mineru_folder

    output_folder = os.path.join(output_dir, pdf_name, "auto")

    # 检查 output_folder 下是否有 .md 文件
    if os.path.exists(output_folder) and any(file.endswith(".md") for file in os.listdir(output_folder)):
        logger.info(f"MinerU already finished!")
        return output_folder

    # 构造并运行命令
    try:
        logger.info(f"MinerU processing...")
        # 设置 GPU 设备的环境变量
        # gpu_id = 0
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # env_name = "MinerU"
        # command = ["conda", "run", "-n", env_name, 'magic-pdf', '-p', pdf_path, '-o', output_dir]
        command = ['magic-pdf', '-p', pdf_path, '-o', output_dir]
        subprocess.run(
            command,
            capture_output=True, text=True, check=True
        )
        logger.info(f"MinerU finished!")
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，打印错误信息
        print("Error:", e.stderr)
    return output_folder

async def extract_text_and_images_with_chunks(pdf_path, output_dir, context_length):
    """
    提取PDF中的文本块，并与图片关联。图片前后上下文文本提取整合。
    """
    folder_path = pdf2markdown(pdf_path, output_dir)
    files = os.listdir(folder_path)
    markdown_files = [file for file in files if file.endswith(".md")]

    if len(markdown_files) != 1:
        raise ValueError("No unique .md file was found in the folder. Please ensure there is only one .md file in the folder.")

    markdown_file_path = os.path.join(folder_path, markdown_files[0])
    with open(markdown_file_path, 'r', encoding='utf-8') as file:
        full_text = file.read()
    
    full_text = clear_images_in_md(full_text)

    # 首先实例化类
    text_chunking_instance = text_chunking_func()
    
    # 对文档文本进行分块
    await text_chunking_instance.text_chunking(full_text)
    
    filepath = os.path.join(output_dir, 'kv_store_text_chunks.json')
    with open(filepath, 'r') as file:
        text_chunks = json.load(file)

    # 创建保存图片的目录
    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # 读取json文件，按顺序重命名，并移动图片
    json_file_path = get_content_list_json_file(folder_path)
    image_move_remove(json_file_path, images_dir, folder_path)
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data = rename_images_in_json(data)

    image_data = {}
    image_counter = 1
     # 遍历 JSON 数据，提取图像信息
    for i, item in enumerate(data):
        if "img_path" in item and item["img_path"] != "":
            # 构建 image_id
            image_id = image_counter
            image_counter += 1
            
            # 获取并修改 image_path
            img_path = os.path.join(output_dir, item["img_path"])
            # 获取 caption 和 footnote
            if item["type"]=="image":
                caption = item.get("img_caption", [])
                footnote = item.get("img_footnote", [])
            elif item["type"] == "table":
                caption = item.get("table_caption", [])
                footnote = item.get("table_footnote", [])

            # 提取 context
            context = ""

            # 前向提取
            word_count = 0
            prev_index = i - 1
            while word_count < context_length and prev_index >= 0:
                prev_text = data[prev_index].get("text", "")
                prev_words = prev_text.split()  # 将前文分割成单词或汉字列表
                prev_remaining_words = context_length - word_count
                # 从结尾部分取指定数量的单词或汉字
                selected_words = prev_words[-prev_remaining_words:]
                context = " ".join(selected_words) + " " + context
                word_count += len(selected_words)
                prev_index -= 1

            # 后向提取
            word_count = 0
            next_index = i + 1
            while word_count < context_length and next_index < len(data):
                next_text = data[next_index].get("text", "")
                next_words = next_text.split()  # 将后文分割成单词或汉字列表
                next_remaining_words = context_length - word_count
                # 从开头部分取指定数量的单词或汉字
                selected_words = next_words[:next_remaining_words]
                context = " ".join(selected_words) + " " + context
                word_count += len(selected_words)
                next_index += 1

            # 获取 chunk_order_index 和 description
            associated_chunk_id = find_chunk_for_image(text_chunks, context)
            description, segmentation  = await get_image_description(img_path, caption, footnote, context)

            image_key = f"image_{image_id}"
            # 构建图像信息字典并添加到列表
            image_data[image_key] = {
                "image_id": image_id,
                "image_path": img_path,
                "caption": caption,
                "footnote": footnote,
                "context": context,
                "chunk_order_index": text_chunks[associated_chunk_id]['chunk_order_index'],
                "chunk_id": associated_chunk_id,
                "description": description,
                "segmentation": segmentation
            }
    return image_data

@dataclass
class chunking_func_pdf2md:
    # 图像提取上下文长度（各100，所以总长度为200）
    context_length: int = 100

    # 键值存储，json，具体定义在storage.py
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage

    def __post_init__(self):
        # 获取全局设置
        cache_path = os.getenv('CACHE_PATH')
        global_config_path = os.path.join(cache_path,"global_config.csv")
        global_config = read_config_to_dict(global_config_path)
        # 初始化存储类实例，用于存储图像属性
        self.image_data = self.key_string_value_json_storage_cls(
            namespace="image_data", global_config = global_config
        )
    
    async def extract_text_and_images(self, pdf_path):
        try:
            # 获取全局设置
            cache_path = os.getenv('CACHE_PATH')
            global_config_path = os.path.join(cache_path,"global_config.csv")
            global_config = read_config_to_dict(global_config_path)
            output_dir = global_config["working_dir"]
            context_length = self.context_length
            imagedata = await extract_text_and_images_with_chunks(pdf_path, output_dir, context_length)
            # 提交所有更新和索引操作
            await self.image_data.upsert(imagedata)
        finally:
            await self._chunking_done()

    async def _chunking_done(self):
        tasks = []
        for storage_inst in [self.image_data]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)