import logging
import json
import glob
import os
import sys
import shutil
import fitz
import base64
import math
import torch
import traceback

from uuid import uuid4
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from eval_llm import get_llm_response, get_mmllm_response, get_mmllm_response_ovis
from transformers import AutoModelForCausalLM

data_dir = '/cpfs02/user/lidong1/xywan/temp_data/docbench_mmgraph_llama_qwenvl'
# 方法名称 mmgraph, graph, llm, mmllm, naive
method_name = 'mmgraph'
# 使用的大模型 llama, qwen, mistral; internvl, qwenvl, ovis
system = 'llama_qwenvl'
# mineru预处理存储位置
preprocess_dir = '/cpfs02/user/lidong1/xywan/project/dataset/docbench/docbench_mineru'
# 加载ovis1.6模型
ovis_model_path = "/cpfs02/user/lidong1/model/Ovis1.6-Gemma2-27B"
if system == 'ovis':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用 GPU 0
    model = AutoModelForCausalLM.from_pretrained(
        ovis_model_path,
        torch_dtype=torch.bfloat16,
        multimodal_max_length=8192,
        trust_remote_code=True
    ).cuda()

if method_name == 'naive':
    from naive_rag import naive_rag_pipeline

# 图像拼接参数
max_pages = 300
concat_num = 1
column_num = 3
resolution = 144

tmp_dir = os.path.join(data_dir,f"tmp")
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# 将 mmgraphrag 文件夹添加到 sys.path
mmgraphrag_path = "/cpfs02/user/lidong1/xywan/project/mmgraphrag"
sys.path.append(mmgraphrag_path)

if method_name == 'mmgraph':
    from mmgraphrag import MMGraphRAG

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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

# 获取PDF路径和问题列表
def get_pdfpath_jsonlines_qstr(folder):
    # 读取 JSONL 文件中的问题
    path = os.path.join(data_dir,f'data/{folder}/{folder}_qa.jsonl')
    jsonlines = open(path, 'r').readlines()
    # 获取第一个PDF文件的路径，实际上就一个，所以就是目标pdf的路径
    path2 = os.path.join(data_dir,f'data/{folder}/*.pdf')
    pdf_path = glob.glob(path2)[0]
    q_string = ''
    for i, line in enumerate(jsonlines):
        question = json.loads(line)['question']
        # 构建问题字符串
        q_string += f'{i+1}. {question}\n'
    qstr_dir = os.path.join(data_dir,f'data/{folder}/{folder}_qstring.txt')
    # if not os.path.exists(qstr_dir):
    with open(qstr_dir, 'w') as f:
        f.write(q_string)
    return pdf_path, q_string

def qa_with_mmgraphrag(pdf_path, q_string, folder_num, system):
    # 建立工作路径
    working_dir = os.path.join(data_dir,f'graphrag_cache/{folder_num}/{system}')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    result_dir = os.path.join(data_dir,f'data/{folder_num}/{system}_results.txt')
    storage_dir = os.path.join(data_dir,f'data/{folder_num}/graphrag/{system}')
    if os.path.exists(storage_dir):
        print("索引已经完成了...是二周目呢！")
    else:
        # 进行索引处理，让控制台热闹起来
        def index():
            rag = MMGraphRAG(
                working_dir=working_dir,
                input_mode=2
            )
            rag.index(pdf_path)
        
        move_dir = os.path.join(data_dir,f'data/{folder_num}/graphrag')
        print("开始索引PDF文件...")
        index()
        print('索引完成! ヾ(✿ﾟ▽ﾟ)ノ')
        if not os.path.exists(move_dir):
            os.makedirs(move_dir)
        shutil.move(working_dir, move_dir)
    def query():
        rag = MMGraphRAG(
            working_dir=storage_dir,
            query_mode=True,
        )
        # 将提问字符串按换行符拆分为多个问题
        questions = q_string.strip().split('\n')

        # 打开结果文件并逐个写入每个问题的响应
        with open(result_dir, 'w') as f:
            for i, question in enumerate(questions, start=1):
                if question:  # 忽略空行
                    response = rag.query(question)
                    f.write(f"{i}. {response}\n\n")
    query()
    print("所有查询完成! (๑•̀ㅂ•́)و✧")

def qa_with_graphrag(pdf_path, q_string, folder_num, system):
    # 建立工作路径
    working_dir = os.path.join(data_dir,f'graphrag_cache/{folder_num}/{system}')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    result_dir = os.path.join(data_dir,f'data/{folder_num}/{system}_results.txt')
    content_dir = os.path.join(data_dir,f'data/{folder_num}/{folder_num}_content.txt')
    storage_dir = os.path.join(data_dir,f'data/{folder_num}/graphrag/{system}')
    if os.path.exists(storage_dir):
        print("索引已经完成了...是二周目呢！")
    else:
        # 获取pdf文件的名称（不带扩展名）
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        # 在预处理文件夹中查找与pdf文件名同名的文件夹
        target_folder = os.path.join(preprocess_dir, pdf_filename)
        # 检查目标文件夹是否存在
        if not os.path.isdir(target_folder):
            logger.error(f"没有找到与PDF文件名匹配的文件夹: {target_folder}")
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
            logger.error(f"在文件夹 {target_folder} 中未找到MD文件")
            return None
        with open(md_file_path, 'r', encoding='utf-8') as md_file:
            file_content = md_file.read()
        with open(content_dir, 'w') as f:
            f.write(file_content)
        # 进行索引处理，让控制台热闹起来
        def index():
            rag = MMGraphRAG(
                working_dir=working_dir,
                # mode 1 可以用来处理规整的pdf，这里输入txt文件可以实现单一模态的GraphRAG
                input_mode=1
            )
            rag.index(content_dir)
        
        move_dir = os.path.join(data_dir,f'data/{folder_num}/graphrag')
        print("开始索引PDF文件...")
        index()
        print('索引完成! ヾ(✿ﾟ▽ﾟ)ノ')
        if not os.path.exists(move_dir):
            os.makedirs(move_dir)
        shutil.move(working_dir, move_dir)
    def query():
        rag = MMGraphRAG(
            working_dir=storage_dir,
            query_mode=True,
        )
        # 将提问字符串按换行符拆分为多个问题
        questions = q_string.strip().split('\n')

        # 打开结果文件并逐个写入每个问题的响应
        with open(result_dir, 'w') as f:
            for i, question in enumerate(questions, start=1):
                if question:  # 忽略空行
                    response = rag.query(question)
                    f.write(f"{i}. {response}\n\n")
    query()
    print("所有查询完成! (๑•̀ㅂ•́)و✧")

def qa_with_llm(pdf_path, q_string, folder_num, system):
    result_dir = os.path.join(data_dir,f'data/{folder_num}/{system}_results.txt')
    """
    # 从PDF文件中提取文本
    pdf = fitz.open(pdf_path)
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
    # 获取pdf文件的名称（不带扩展名）
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    # 在预处理文件夹中查找与pdf文件名同名的文件夹
    target_folder = os.path.join(preprocess_dir, pdf_filename)
    # 检查目标文件夹是否存在
    if not os.path.isdir(target_folder):
        logger.error(f"没有找到与PDF文件名匹配的文件夹: {target_folder}")
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
        logger.error(f"在文件夹 {target_folder} 中未找到MD文件")
        return None
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        file_content = md_file.read()
    def truncate_file_content_by_char(file_content, max_length):
        """
        按字符数限制 file_content 的长度。
        """
        if len(file_content) > max_length:
            print(f"Original file content length: {len(file_content)}. Truncating to {max_length} characters.")
            return file_content[:max_length]
        return file_content
    max_length = 28000
    file_content = truncate_file_content_by_char(file_content, max_length)
    system_prompt = """You are an intelligent assistant capable of answering questions based solely on the context provided by the user. 
    Follow these rules strictly:
    Respond to questions only using the information from the provided text. Do not incorporate external knowledge or assumptions.
    If the answer cannot be found in the provided text, clearly inform the user: "The answer cannot be determined from the provided information."
    Follow the numbered format for answers, and ensure the response for each question is given in the form of a number, as shown below:
    1. Answer1
    2. Answer2
    3. Answer3
    ...
    """
    assist_prompt = "Please answer the following questions based on the provided text:"
    q_prompt = file_content + "\n" + assist_prompt + "\n" + q_string
    results = get_llm_response(q_prompt, system_prompt)
    with open(result_dir, 'w') as r:
        r.write(results)

def qa_with_mmllm(pdf_path, q_string, folder_num, system):
    result_dir = os.path.join(data_dir,f'data/{folder_num}/{system}_results.txt')
    # 获取pdf文件的名称（不带扩展名）
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    # 存储编码图像的列表
    image_list = list()
    with fitz.open(pdf_path) as pdf:
        # 遍历指定页数内的页面
        for index, page in enumerate(pdf[:max_pages]):
            image_path = os.path.join(tmp_dir, f"{pdf_filename}_{index+1}.png")
            if not os.path.exists(image_path):
                # 渲染页面为图像
                image = page.get_pixmap(dpi=resolution)
                image.save(image_path)
            image_list.append(image_path)
    concat_image_list = concat_images(image_list, concat_num, column_num)
    assist_prompt = "Please answer the following questions based on the provided image:"
    system_prompt = """You are an intelligent assistant capable of answering questions based solely on the context provided by the user. 
    Follow these rules strictly:
    Respond to questions only using the information from the provided image. Do not incorporate external knowledge or assumptions.
    If the answer cannot be found in the provided image, clearly inform the user: "The answer cannot be determined from the provided information."
    Follow the numbered format for answers, and ensure the response for each question is given in the form of a number, as shown below:
    1. Answer1
    2. Answer2
    3. Answer3
    ...
    """
    if system == 'ovis':
        text_prompt = system_prompt + "\n" + assist_prompt + "\n" + q_string
        results = get_mmllm_response_ovis(model, concat_image_list[0], text_prompt)
    else:
        encoded_image_list = list()
        for p in concat_image_list:
            image = Image.open(p)
            encoded_image = encode_image_to_base64(image)
            encoded_image_list.append(encoded_image)
        content = list()
        q_prompt = assist_prompt + "\n" + q_string
        # 添加问题文本
        content.append(
            {
                "type": "text",
                "text": q_prompt,
            }
        )
        # 添加图像的 Base64 字符串
        for encoded_image in encoded_image_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
        results = get_mmllm_response(content, system_prompt)
    with open(result_dir, 'w') as r:
        r.write(results)

def qa_with_naiverag(pdf_path, q_string, folder_num, system):
    # 建立工作路径
    result_dir = os.path.join(data_dir,f'data/{folder_num}/{system}_results.txt')
    content_dir = os.path.join(data_dir,f'data/{folder_num}/{folder_num}_content.txt')
    # 获取pdf文件的名称（不带扩展名）
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    # 在预处理文件夹中查找与pdf文件名同名的文件夹
    target_folder = os.path.join(preprocess_dir, pdf_filename)
    # 检查目标文件夹是否存在
    if not os.path.isdir(target_folder):
        logger.error(f"没有找到与PDF文件名匹配的文件夹: {target_folder}")
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
        logger.error(f"在文件夹 {target_folder} 中未找到MD文件")
        return None
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        file_content = md_file.read()
    with open(content_dir, 'w') as f:
        f.write(file_content)
    
    # 将提问字符串按换行符拆分为多个问题
    questions = q_string.strip().split('\n')

    # 打开结果文件并逐个写入每个问题的响应
    with open(result_dir, 'w') as f:
        for i, question in enumerate(questions, start=1):
            if question:  # 忽略空行
                response = naive_rag_pipeline(file_content, question)
                f.write(f"{i}. {response}\n\n")
    return

def qa_mmgraph(system):
    initial_folder_number = 0
    final_folder_number = 228
    for cur_folder_num in range(initial_folder_number, final_folder_number + 1):
        path = os.path.join(data_dir,f'data/{cur_folder_num}')
        # 检查系统结果文件是否存在，存在则跳过
        result_file = os.path.join(path, f"{system}_results.txt")
        if os.path.exists(result_file):
            logger.info(f"跳过文件夹 {cur_folder_num}，因为 {system}_results.txt 已存在。")
            continue
        if os.path.exists(path):
            logger.info(f"Folder = {cur_folder_num}")
            pdf_path, q_string = get_pdfpath_jsonlines_qstr(cur_folder_num)
            try:
                qa_with_mmgraphrag(pdf_path, q_string, cur_folder_num, system)
            except Exception as e:
                error_message = traceback.format_exc()
                logger.info(f"别气馁，再试一次！ヾ(◍°∇°◍)ﾉﾞ 这次的错误信息: {error_message}。")
                logger.info(f"Folder = {cur_folder_num}")
                try:
                    qa_with_mmgraphrag(pdf_path, q_string, cur_folder_num, system)
                except Exception as e:
                    error_message = traceback.format_exc()
                    logger.info(f"第二次执行仍失败，跳过当前目录。o(╥﹏╥)o 这次的错误信息: {error_message}。")
                    continue

def qa_graph(system):
    initial_folder_number = 0
    final_folder_number = 228
    
    for cur_folder_num in range(initial_folder_number, final_folder_number + 1):
        path = os.path.join(data_dir,f'data/{cur_folder_num}')
        # 检查系统结果文件是否存在，存在则跳过
        result_file = os.path.join(path, f"{system}_results.txt")
        if os.path.exists(result_file):
            logger.info(f"跳过文件夹 {cur_folder_num}，因为 {system}_results.txt 已存在。")
            continue
        if os.path.exists(path):
            logger.info(f"Folder = {cur_folder_num}")
            pdf_path, q_string = get_pdfpath_jsonlines_qstr(cur_folder_num)
            try:
                qa_with_graphrag(pdf_path, q_string, cur_folder_num, system)
            except Exception as e:
                logger.info(f"别气馁，再试一次！ヾ(◍°∇°◍)ﾉﾞ 这次的错误信息: {e}。")
                logger.info(f"Folder = {cur_folder_num}")
                try:
                    qa_with_graphrag(pdf_path, q_string, cur_folder_num, system)
                except Exception as e:
                    logger.info(f"第二次执行仍失败，跳过当前目录。o(╥﹏╥)o 这次的错误信息: {e}。")
                    continue

def qa_llm(system):
    initial_folder_number = 0
    final_folder_number = 228
    for cur_folder_num in range(initial_folder_number, final_folder_number + 1):
        path = os.path.join(data_dir,f'data/{cur_folder_num}')
        # 检查系统结果文件是否存在，存在则跳过
        result_file = os.path.join(path, f"{system}_results.txt")
        if os.path.exists(result_file):
            logger.info(f"跳过文件夹 {cur_folder_num}，因为 {system}_results.txt 已存在。")
            continue
        if os.path.exists(path):
            logger.info(f"Folder = {cur_folder_num}")
            pdf_path, q_string = get_pdfpath_jsonlines_qstr(cur_folder_num)
            try:
                qa_with_llm(pdf_path, q_string, cur_folder_num, system)
            except Exception as e:
                logger.info(f"别气馁，再试一次！ヾ(◍°∇°◍)ﾉﾞ 这次的错误信息: {e}。")
                logger.info(f"Folder = {cur_folder_num}")
                try:
                    qa_with_llm(pdf_path, q_string, cur_folder_num, system)
                except Exception as e:
                    logger.info(f"第二次执行仍失败，跳过当前目录。o(╥﹏╥)o 这次的错误信息: {e}。")
                    continue

def qa_mmllm(system):
    initial_folder_number = 0
    final_folder_number = 228
    for cur_folder_num in range(initial_folder_number, final_folder_number + 1):
        path = os.path.join(data_dir,f'data/{cur_folder_num}')
        # 检查系统结果文件是否存在，存在则跳过
        result_file = os.path.join(path, f"{system}_results.txt")
        if os.path.exists(result_file):
            logger.info(f"跳过文件夹 {cur_folder_num}，因为 {system}_results.txt 已存在。")
            continue
        if os.path.exists(path):
            logger.info(f"Folder = {cur_folder_num}")
            pdf_path, q_string = get_pdfpath_jsonlines_qstr(cur_folder_num)
            try:
                qa_with_mmllm(pdf_path, q_string, cur_folder_num, system)
            except Exception as e:
                logger.info(f"别气馁，再试一次！ヾ(◍°∇°◍)ﾉﾞ 这次的错误信息: {e}。")
                logger.info(f"Folder = {cur_folder_num}")
                try:
                    qa_with_llm(pdf_path, q_string, cur_folder_num, system)
                except Exception as e:
                    logger.info(f"第二次执行仍失败，跳过当前目录。o(╥﹏╥)o 这次的错误信息: {e}。")
                    continue

def qa_naive(system):
    initial_folder_number = 0
    final_folder_number = 228
    for cur_folder_num in range(initial_folder_number, final_folder_number + 1):
        path = os.path.join(data_dir,f'data/{cur_folder_num}')
        # 检查系统结果文件是否存在，存在则跳过
        result_file = os.path.join(path, f"{system}_results.txt")
        if os.path.exists(result_file):
            logger.info(f"跳过文件夹 {cur_folder_num}，因为 {system}_results.txt 已存在。")
            continue
        if os.path.exists(path):
            logger.info(f"Folder = {cur_folder_num}")
            pdf_path, q_string = get_pdfpath_jsonlines_qstr(cur_folder_num)
            try:
                qa_with_naiverag(pdf_path, q_string, cur_folder_num, system)
            except Exception as e:
                logger.info(f"别气馁，再试一次！ヾ(◍°∇°◍)ﾉﾞ 这次的错误信息: {e}。")
                logger.info(f"Folder = {cur_folder_num}")
                try:
                    qa_with_naiverag(pdf_path, q_string, cur_folder_num, system)
                except Exception as e:
                    logger.info(f"第二次执行仍失败，跳过当前目录。o(╥﹏╥)o 这次的错误信息: {e}。")
                    continue

# 简化重复运行的代码修改量
def qa(method_name, system):
    if method_name == "mmgraph":
        qa_mmgraph(system)
    elif method_name == "graph":
        qa_graph(system)
    elif method_name == "llm":
        qa_llm(system)
    elif method_name == "mmllm":
        qa_mmllm(system)
    elif method_name == "naive":
        qa_naive(system)
    else:
        logger.info("请输入正确的方法名！")

if __name__ == '__main__':
    qa(method_name, system)