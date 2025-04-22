from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

@dataclass
class QueryParam:
    response_type: str = "Keep the responses as brief and accurate as possible. If you need to present information in a list format, use (1), (2), (3), etc., instead of numbered bullets like 1., 2., 3. "
    top_k: int = 10
    local_max_token_for_text_unit: int = 4000
    local_max_token_for_local_context: int = 6000
    # alpha: int = 0.5
    number_of_mmentities: int = 3

cache_path = './cache'

embedding_model_dir = './cache/all-MiniLM-L6-v2'
EMBED_MODEL = SentenceTransformer(embedding_model_dir, device="cpu")
# EMBED_MODEL = SentenceTransformer(embedding_model_dir, trust_remote_code=True, device="cuda:0")

def encode(content):
    return EMBED_MODEL.encode(content)
"""
def encode(content):
    return EMBED_MODEL.encode(content, prompt_name="s2p_query", convert_to_tensor=True).cpu()
"""

mineru_dir = "./example_input/mineru_result"
API_KEY = "Come on, hand over your API key! No more hiding it away!"
MODEL = "moonshot-v1-32k"
URL = "https://api.moonshot.cn/v1"
MM_API_KEY = "Alright, Sherlock, where’s the API key? Time to crack the case!"
MM_MODEL = "qwen-vl-max"
MM_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"