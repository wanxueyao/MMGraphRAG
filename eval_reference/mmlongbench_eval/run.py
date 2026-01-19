import os
import re
import json
import base64
import argparse
import fitz
import sys
import logging
import math
import shutil
import torch
import numpy as np

from openai import OpenAI
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from uuid import uuid4
from tqdm import tqdm
from time import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM

from extract_answer import extract_answer
from eval_score import eval_score, eval_acc_and_f1, show_results

mmlongbench_eval_dir = '/cpfs02/user/lidong1/xywan/project/eval/mmlongbench_eval'
data_dir = '/cpfs02/user/lidong1/xywan/temp_data/mmlongbench_mmllm_ovis'
# mmgraph, graph, llm, mmllm, naive
method_name = 'mmllm'
# llama, qwen, mistral; qwenvl, internvl, ovis
model_name = 'ovis'

# mineru预处理存储位置
preprocess_dir = '/cpfs02/user/lidong1/xywan/project/dataset/mmlongbench/mmlongbench_mineru'

API_KEY = "EMPTY"
MODEL = "llama-3.1-70b"
URL = "http://localhost:6000/v1"
MM_API_KEY = "EMPTY"
MM_MODEL = "qwen2-vl-72b"
MM_URL = "http://localhost:6008/v1"

# 将 mmgraphrag 文件夹添加到 sys.path
mmgraphrag_path = "/cpfs02/user/lidong1/xywan/project/mmgraphrag"
sys.path.append(mmgraphrag_path)

# 图像拼接参数
concat_num = 1
column_num = 3

# 加载嵌入模型
if method_name == 'naive':
    EMBED_MODEL = SentenceTransformer(
        '/cpfs02/user/lidong1/model/stella_en_1.5B_v5', trust_remote_code=True, device="cuda:0"
    )
# 加载ovis1.6模型
ovis_model_path = "/cpfs02/user/lidong1/model/Ovis1.6-Gemma2-27B"

if 'ovis' in model_name:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用 GPU 0
    model = AutoModelForCausalLM.from_pretrained(
        ovis_model_path,
        torch_dtype=torch.bfloat16,
        multimodal_max_length=8192,
        trust_remote_code=True
    ).cuda()

def chunk_text(text, chunk_size=500, overlap=100):
    """
    将文本分块，每块长度为 chunk_size，并有 overlap 个 token 的重叠
    :param text: 输入文本（str）
    :param chunk_size: 每块的 token 数量
    :param overlap: 每块之间的重叠 token 数量
    :return: 分块后的文本列表
    """
    tokens = text.split()  # 简单分词
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(tokens):  # 最后一块结束
            break
    return chunks

def compute_embeddings(chunks, model):
    """
    计算文本块的嵌入
    :param chunks: 文本块列表
    :param model: 嵌入模型
    :return: 文本块对应的嵌入向量列表
    """
    return model.encode(chunks, prompt_name="s2p_query", convert_to_tensor=True)

def select_top_chunks_by_embedding(chunks, question, model, top_k=5):
    """
    根据嵌入相似度选择最相关的 top_k 个文本块
    :param chunks: 文本块列表
    :param question: 用户问题
    :param model: 嵌入模型
    :param top_k: 选择的前 k 个文本块
    :return: 最相关的 top_k 文本块
    """
    # 计算文本块和问题的嵌入
    chunk_embeddings = compute_embeddings(chunks, model)
    question_embedding = model.encode([question], convert_to_tensor=True)
    
    # 计算问题和每个块之间的余弦相似度
    similarities = cosine_similarity(question_embedding.cpu(), chunk_embeddings.cpu()).flatten()
    
    # 找到相似度最高的 top_k 个文本块
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_k_indices]

tmp_dir = os.path.join(data_dir,f"tmp")
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# 将图像编码为 Base64 字符串的函数
def encode_image_to_base64(img):
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    tmp = os.path.join(tmp_dir, str(uuid4()) + '.jpg')
    img.save(tmp)
    with open(tmp, 'rb') as image_file:
        image_data = image_file.read()
    ret = base64.b64encode(image_data).decode('utf-8')
    os.remove(tmp)
    return ret

# 拼接图像
def concat_images(image_list, concat_num=1, column_num=3, max_size=65500):
    interval = max(math.ceil(len(image_list) / concat_num), 1)
    concatenated_image_list = list()

    for i in range(0, len(image_list), interval):
        image_path = "_".join(image_list[0].split("_")[:-1]) + "_concat{}_{}.jpg".format(concat_num, i // interval)
        if not os.path.exists(image_path):
            images_this_batch = [
                Image.open(filename) for filename in image_list[i:i + interval]
            ]
            
            if column_num == 1:
                total_height = images_this_batch[0].height * len(images_this_batch)
            else:
                total_height = images_this_batch[0].height * ((len(images_this_batch) - 1) // column_num + 1)

            total_width = images_this_batch[0].width * column_num
            
            # 检查拼接后的图像尺寸
            if total_width > max_size or total_height > max_size:
                print(f"Warning: The resulting image size exceeds the max allowed size ({max_size} pixels). Resizing images.")
                scale_factor = min(max_size / total_width, max_size / total_height)
                total_width = int(total_width * scale_factor)
                total_height = int(total_height * scale_factor)
                # 缩放图像
                images_this_batch = [image.resize((int(image.width * scale_factor), int(image.height * scale_factor))) for image in images_this_batch]
                
            concatenated_image = Image.new('RGB', (total_width, total_height), 'white')
            x_offset, y_offset = 0, 0
            for cnt, image in enumerate(images_this_batch):
                concatenated_image.paste(image, (x_offset, y_offset))
                x_offset += image.width
                if (cnt + 1) % column_num == 0:
                    y_offset += image.height
                    x_offset = 0
            concatenated_image.save(image_path)
        concatenated_image_list.append(image_path)

    return concatenated_image_list

#预处理函数列表
def process_sample_mmllm(sample, args):
    # 获取问题
    question = sample["question"]
    # 获取文档名称
    doc_name = re.sub("\.pdf$", "", sample["doc_id"]).split("/")[-1]
    # 存储编码图像的列表
    image_list = list()
    with fitz.open(os.path.join(args.document_path, sample["doc_id"])) as pdf:
        # 遍历指定页数内的页面
        for index, page in enumerate(pdf[:args.max_pages]):
            image_path = os.path.join(tmp_dir, f"{doc_name}_{index+1}.png")
            if not os.path.exists(image_path):
                # 渲染页面为图像
                image = page.get_pixmap(dpi=args.resolution)
                image.save(image_path)
            image_list.append(image_path)
    concat_image_list = concat_images(image_list, args.concat_num, args.column_num)
    if model_name == 'ovis':
        image_path = concat_image_list[0]
        # 存储 image_path 和 question 到 messages 列表中
        messages = []
        messages.append({'image': image_path})
        messages.append({'question': question})
        return messages
    else:
        encoded_image_list = list()
        for p in concat_image_list:
            image = Image.open(p)
            encoded_image = encode_image_to_base64(image)
            encoded_image_list.append(encoded_image)
        content = list()
        # 添加问题文本
        content.append(
            {
                "type": "text",
                "text": question,
            }
        )
        # 添加图像的 Base64 字符串
        for encoded_image in encoded_image_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        return messages

def process_sample_llm(sample, args):
    # 获取问题
    question = sample["question"]
    # 获取文档名称
    pdf_filename = re.sub("\.pdf$", "", sample["doc_id"]).split("/")[-1]
    # 判断txt文件是否已经存在
    txt_path = os.path.join(tmp_dir, f"{pdf_filename}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as file:
            file_content = file.read()
    else:
        """
        # 从PDF文件中提取文本
        pdf = fitz.open(os.path.join(args.document_path, sample["doc_id"]))
        file_content = ""
        image_index = 1
        for page_num in range(pdf.page_count):
            # 加载每一页
            page = pdf.load_page(page_num)
            text_blocks = page.get_text("dict")["blocks"]
            for block in text_blocks:
                if block["type"] == 0:# 如果是文本块
                    for line in block["lines"]:
                        for span in line["spans"]:
                            file_content += span["text"] # 提取文本内容
                elif block["type"] == 1:# 如果是图片块
                    file_content += f"[image{image_index}]"
                    image_index += 1
                file_content += "\n"
        """
        # 在预处理文件夹中查找与pdf文件名同名的文件夹
        target_folder = os.path.join(preprocess_dir, pdf_filename)
        # 检查目标文件夹是否存在
        if not os.path.isdir(target_folder):
            logging.error(f"没有找到与PDF文件名匹配的文件夹: {target_folder}")
        # 在文件夹中查找md文件
        md_file_path = None
        for file in os.listdir(target_folder):
            if file.endswith('.md'):
                md_file_path = os.path.join(target_folder, file)
                break
        # 获取目标文件夹中的唯一文件夹
        subfolders = [f for f in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, f))]
        # 找到唯一的文件夹（auto或者ocr）
        subfolder = os.path.join(target_folder, subfolders[0])
        # 在子文件夹中查找md文件
        md_file_path = None
        for file in os.listdir(subfolder):
            if file.endswith('.md'):
                md_file_path = os.path.join(subfolder, file)
                break
        
        # 如果没有找到md文件，记录错误
        if not md_file_path:
            logging.error(f"在文件夹 {target_folder} 中未找到MD文件")
            return None
        with open(md_file_path, 'r', encoding='utf-8') as md_file:
            file_content = md_file.read()

        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(file_content)
    
    assist_prompt = "Please answer the following questions based on the provided text:"
    q_prompt = assist_prompt + "\n" + question
    messages = [
        {
                "role": "system",
                "content": file_content,
        },
        {
            "role": "user",
            "content": q_prompt
        }
    ]
    return messages

def process_sample_graphrag(sample, args):
    from mmgraphrag import MMGraphRAG
    # 获取文档名称
    pdf_filename = re.sub("\.pdf$", "", sample["doc_id"]).split("/")[-1]
    # 建立工作路径
    working_dir = os.path.join(data_dir,f'graphrag_cache_{args.model_name}')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    storage_dir = os.path.join(data_dir,f'graphrag_{args.model_name}/{pdf_filename}')
    if os.path.exists(storage_dir):
        return sample
    # 建立txt文件
    txt_path = os.path.join(tmp_dir, f"{pdf_filename}.txt")
    # 在预处理文件夹中查找与pdf文件名同名的文件夹
    target_folder = os.path.join(preprocess_dir, pdf_filename)
    # 检查目标文件夹是否存在
    if not os.path.isdir(target_folder):
        logging.error(f"没有找到与PDF文件名匹配的文件夹: {target_folder}")
    # 在文件夹中查找md文件
    md_file_path = None
    for file in os.listdir(target_folder):
        if file.endswith('.md'):
            md_file_path = os.path.join(target_folder, file)
            break
    
    # 获取目标文件夹中的唯一文件夹
    subfolders = [f for f in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, f))]
    # 找到唯一的文件夹（auto或者ocr）
    subfolder = os.path.join(target_folder, subfolders[0])
    # 在子文件夹中查找md文件
    md_file_path = None
    for file in os.listdir(subfolder):
        if file.endswith('.md'):
            md_file_path = os.path.join(subfolder, file)
            break
    
    # 如果没有找到md文件，记录错误
    if not md_file_path:
        logging.error(f"在文件夹 {target_folder} 中未找到MD文件")
        return None
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        file_content = md_file.read()
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(file_content)
    # 进行索引处理
    def index(txt_path, working_dir):
        rag = MMGraphRAG(
            working_dir=working_dir,
            # mode 1 可以用来处理规整的pdf，这里输入txt文件可以实现单一模态的GraphRAG
            input_mode=1
        )
        start1 = time()
        rag.index(txt_path)
        print("开始索引PDF文件...")
        print('索引完成! ヾ(✿ﾟ▽ﾟ)ノ')
        print("索引耗时:", time() - start1)
    index(txt_path, working_dir)
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
        shutil.move(working_dir, storage_dir)
    return sample

def process_sample_naiverag(sample, args):
    # 获取问题
    question = sample["question"]
    # 获取文档名称
    pdf_filename = re.sub("\.pdf$", "", sample["doc_id"]).split("/")[-1]
    # 判断txt文件是否已经存在
    txt_path = os.path.join(tmp_dir, f"{pdf_filename}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as file:
            file_content = file.read()
    else:
        """
        # 从PDF文件中提取文本
        pdf = fitz.open(os.path.join(args.document_path, sample["doc_id"]))
        file_content = ""
        image_index = 1
        for page_num in range(pdf.page_count):
            # 加载每一页
            page = pdf.load_page(page_num)
            text_blocks = page.get_text("dict")["blocks"]
            for block in text_blocks:
                if block["type"] == 0:# 如果是文本块
                    for line in block["lines"]:
                        for span in line["spans"]:
                            file_content += span["text"] # 提取文本内容
                elif block["type"] == 1:# 如果是图片块
                    file_content += f"[image{image_index}]"
                    image_index += 1
                file_content += "\n"
        """
        # 在预处理文件夹中查找与pdf文件名同名的文件夹
        target_folder = os.path.join(preprocess_dir, pdf_filename)
        # 检查目标文件夹是否存在
        if not os.path.isdir(target_folder):
            logging.error(f"没有找到与PDF文件名匹配的文件夹: {target_folder}")
        # 在文件夹中查找md文件
        md_file_path = None
        for file in os.listdir(target_folder):
            if file.endswith('.md'):
                md_file_path = os.path.join(target_folder, file)
                break
        # 获取目标文件夹中的唯一文件夹
        subfolders = [f for f in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, f))]
        # 找到唯一的文件夹（auto或者ocr）
        subfolder = os.path.join(target_folder, subfolders[0])
        # 在子文件夹中查找md文件
        md_file_path = None
        for file in os.listdir(subfolder):
            if file.endswith('.md'):
                md_file_path = os.path.join(subfolder, file)
                break
        
        # 如果没有找到md文件，记录错误
        if not md_file_path:
            logging.error(f"在文件夹 {target_folder} 中未找到MD文件")
            return None
        with open(md_file_path, 'r', encoding='utf-8') as md_file:
            file_content = md_file.read()

        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(file_content)
    # Step 1: 分块
    chunk_size=500
    overlap=100
    chunks = chunk_text(file_content, chunk_size, overlap)
    
    # Step 2: 检索相关块
    top_k=5
    top_chunks = select_top_chunks_by_embedding(chunks, question, EMBED_MODEL, top_k)
    
    # Step 3: 合并上下文并生成答案
    context = "\n".join(top_chunks)
    assist_prompt = "Please answer the following questions based on the provided text:"
    q_prompt = assist_prompt + "\n" + question
    messages = [
        {
                "role": "system",
                "content": context,
        },
        {
            "role": "user",
            "content": q_prompt
        }
    ]
    return messages

def process_sample_mmgraphrag(sample, args):
    from mmgraphrag import MMGraphRAG
    # 获取文档名称
    pdf_filename = re.sub("\.pdf$", "", sample["doc_id"]).split("/")[-1]
    # 建立工作路径
    working_dir = os.path.join(data_dir,f'graphrag_cache_{args.model_name}')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    storage_dir = os.path.join(data_dir,f'graphrag_{args.model_name}/{pdf_filename}')
    if os.path.exists(storage_dir):
        return sample
    
    pdf_path = os.path.join(data_dir, f'data/documents', sample["doc_id"])
    
    # 进行索引处理
    def index(pdf_path, working_dir):
        rag = MMGraphRAG(
            working_dir=working_dir,
            input_mode=2
        )
        start1 = time()
        rag.index(pdf_path)
        print("开始索引PDF文件...")
        print('索引完成! ヾ(✿ﾟ▽ﾟ)ノ')
        print("索引耗时:", time() - start1)
    index(pdf_path, working_dir)
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
        shutil.move(working_dir, storage_dir)
    return sample

def process_sample(sample, args):
    if args.method=="mmllm":
        return process_sample_mmllm(sample, args)
    elif args.method=="llm":
        return process_sample_llm(sample, args)
    elif args.method=="graph":
        return process_sample_graphrag(sample, args)
    elif args.method=="naive":
        return process_sample_naiverag(sample, args)
    elif args.method=="mmgraph":
        return process_sample_mmgraphrag(sample, args)
    else:
        raise AssertionError()

# qa函数列表
def qa_mmllm(messages, args):
    if model_name == 'ovis':
        print("you are becoming a girl!")
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        image_path = messages[0]['image']
        image = Image.open(image_path)
        question = messages[1]['question']
        query = f'<image>\n{question}'
        # 格式化输入数据
        prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

        # 生成输出
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )
            
            # 调用模型生成
            output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            response = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return response
    else:
        client = OpenAI(
            base_url=MM_URL, api_key=MM_API_KEY
        )
        response = client.chat.completions.create(
                    model=MM_MODEL,
                    messages=messages,
                    # max_tokens=args.max_tokens,
                    # temperature=args.temperature
                )
        response = response.choices[0].message.content
        return response
     
def qa_llm(messages, args):
    client = OpenAI(
        base_url=URL, api_key=API_KEY
    )
    response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                # max_tokens=args.max_tokens,
                # temperature=args.temperature
            )
    response = response.choices[0].message.content
    return response

def qa_graphrag(messages, args):
    from mmgraphrag import MMGraphRAG
    sample = messages
    # 获取问题
    question = sample["question"]
    # 指定工作路径
    pdf_filename = re.sub("\.pdf$", "", sample["doc_id"]).split("/")[-1]
    working_dir = os.path.join(data_dir,f'graphrag_{args.model_name}/{pdf_filename}/graphrag_cache_{args.model_name}')
    # 进行查询
    start = time()
    def query():
        rag = MMGraphRAG(
            working_dir=working_dir,
            query_mode=True,
        )
        response = rag.query(question)
        return response
    response = query()
    print("所有查询完成! (๑•̀ㅂ•́)و✧")
    print("查询耗时:", time() - start)
    return response

def qa_naiverag(messages, args):
    client = OpenAI(
        base_url=URL, api_key=API_KEY
    )
    response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                # max_tokens=args.max_tokens,
                # temperature=args.temperature
            )
    response = response.choices[0].message.content
    return response

def qa_mmgraphrag(messages, args):
    from mmgraphrag import MMGraphRAG
    sample = messages
    # 获取问题
    question = sample["question"]
    # 指定工作路径
    pdf_filename = re.sub("\.pdf$", "", sample["doc_id"]).split("/")[-1]
    working_dir = os.path.join(data_dir,f'graphrag_{args.model_name}/{pdf_filename}/graphrag_cache_{args.model_name}')
    # 进行查询
    start = time()
    def query():
        rag = MMGraphRAG(
            working_dir=working_dir,
            query_mode=True,
        )
        response = rag.query(question)
        return response
    response = query()
    print("所有查询完成! (๑•̀ㅂ•́)و✧")
    print("查询耗时:", time() - start)
    return response

def qa(messages, args):
    if args.method=="mmllm":
        return qa_mmllm(messages, args)
    elif args.method=="llm":
        return qa_llm(messages, args)
    elif args.method=="graph":
        return qa_graphrag(messages, args)
    elif args.method=="naive":
        return qa_naiverag(messages, args)
    elif args.method=="mmgraph":
        return qa_mmgraphrag(messages, args)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    json_dir = os.path.join(data_dir, f"data/samples.json")
    parser.add_argument("--input_path", type=str, default=json_dir)
    doc_dir = os.path.join(data_dir, f"data/documents")
    parser.add_argument("--document_path", type=str, default=doc_dir)
    parser.add_argument("--method", type=str, default=method_name)
    parser.add_argument("--model_name", type=str, default=model_name)
    parser.add_argument("--max_pages", type=int, default=300)
    parser.add_argument("--resolution", type=int, default=144)
    parser.add_argument("--max_try", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concat_num", type=int, default=concat_num)
    parser.add_argument("--column_num", type=int, default=column_num)
    prompt_dir = os.path.join(mmlongbench_eval_dir, "prompt_for_answer_extraction.md")
    parser.add_argument("--extractor_prompt_path", type=str, default=prompt_dir)
    args = parser.parse_args()

    output_path = os.path.join(data_dir,f'data/res_{args.method}_{args.model_name}.json')
    # 加载提取答案的prompt
    with open(args.extractor_prompt_path) as f:
        prompt = f.read()
    
    # 加载样本，如果输出文件已存在则继续从中读取
    if os.path.exists(output_path):
        with open(output_path) as f:
            samples = json.load(f)
    else:
        with open(args.input_path, 'r') as f:
            samples = json.load(f)

    for sample in tqdm(samples):
        if "score" in sample:
            score = sample["score"]
        else:
            messages = process_sample(sample, args)
            
            try_cnt = 0
            is_success = False
            while True:
                try:
                    response = qa(messages, args)
                    is_success = True
                except:
                    try_cnt += 1
                    response = "Failed"
                if is_success or try_cnt>args.max_try:
                    break
                
            sample["response"] = response
            # 提取答案并计算得分
            extracted_res = extract_answer(sample["question"], response, prompt)
            sample["extracted_res"] = extracted_res
            try:
                print(extracted_res)
                pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
                score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
            except:
                pred_ans = "Failed to extract"
                score = 0.0
            sample["pred"] = pred_ans
            sample["score"] = score

        acc, f1 = eval_acc_and_f1(samples)
        print("--------------------------------------")
        print("Question: {}".format(sample["question"]))
        print("Response: {}".format(sample["response"]))
        print("Gt: {}\tPred: {}\tScore: {}".format(sample["answer"], sample["pred"], sample["score"]))
        print("Avg acc: {}".format(acc))
        print("Avg f1: {}".format(f1))
        
        with open(output_path, 'w') as f:
            json.dump(samples, f)
    
    show_results(samples, show_path=re.sub("\.json$", ".txt", output_path))