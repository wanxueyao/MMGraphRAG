from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

# 加载嵌入模型
EMBED_MODEL = SentenceTransformer(
    '/cpfs02/user/lidong1/model/stella_en_1.5B_v5', trust_remote_code=True, device="cuda:0"
)

API_KEY = "EMPTY"
URL = "http://localhost:6002/v1"
MODEL = "qwen2.5-72b"

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

def generate_answer_with_llm(question, context):
    """
    将问题和上下文提交给 LLM 生成答案
    :param question: 用户问题
    :param context: 检索到的上下文文本（str）
    :param llm_function: 用于调用 LLM 的函数
    :return: LLM 生成的答案
    """
    prompt = f"Please answer the question based on the provided context. Only use a bullet-point format with (1), (2), etc., when the answer requires multiple points.\n\nContext：{context}\n\nQuestion：{question}\n\nAnswer："
    client = OpenAI(
        base_url=URL, api_key=API_KEY
    )

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    text_response = completion.choices[0].message.content
    return text_response

# 主函数
def naive_rag_pipeline(text, question, chunk_size=500, overlap=100, top_k=5):
    """
    实现 NaiveRAG 的完整管道
    :param text: 输入全文
    :param question: 用户问题
    :param chunk_size: 每块的 token 数量
    :param overlap: 每块之间的重叠 token 数量
    :param top_k: 选择的前 k 个文本块
    :param llm_function: 用于调用 LLM 的函数
    :return: LLM 生成的答案
    """
    # Step 1: 分块
    chunks = chunk_text(text, chunk_size, overlap)
    
    # Step 2: 检索相关块
    top_chunks = select_top_chunks_by_embedding(chunks, question, EMBED_MODEL, top_k)
    
    # Step 3: 合并上下文并生成答案
    context = "\n".join(top_chunks)
    answer = generate_answer_with_llm(question, context)
    return answer

# 示例测试
if __name__ == "__main__":
    # 示例文本
    txt_path = "./xywan/input/Bergen-Brochure-en-2022-23.txt"
    with open(txt_path, 'r', encoding='utf-8') as md_file:
        input_text = md_file.read()
    
    # 示例问题
    question = "Where the cuisine inspired by Nordic and continental traditions can be found?"

    # 调用管道
    answer = naive_rag_pipeline(input_text, question)
    print(answer)
