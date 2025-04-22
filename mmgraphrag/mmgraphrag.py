import asyncio
import os
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from base import logger
from parameter import cache_path
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
os.environ['CACHE_PATH'] = cache_path
from llm import model_if_cache
from parameter import QueryParam

# 返回类型是 asyncio.AbstractEventLoop，即事件循环对象。确保无论当前环境是否已经存在事件循环，都能返回一个有效的事件循环。
def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

@dataclass
class MMGraphRAG:
    # 表示工作目录的路径，默认是根据当前日期时间生成的目录
    working_dir: str = field(
        default_factory=lambda: f"./mm_graphrag_output_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # 实体提取最大“拾取”次数，也就是反复提取次数，默认不反复提取
    entity_extract_max_gleaning: int = 1
    # 实体摘要最大token数
    entity_summary_to_max_tokens: int = 500

    # 为向量数据库存储类提供可选的参数字典
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    # LLM相关调用
    model_func: callable = model_if_cache
    model_max_token_size: int = 32768

    # 批量大小，默认为32
    embedding_batch_num: int = 32

    # tiktoken使用的模型名字，默认为gpt-4o，多数模型可以通用
    tiktoken_model_name: str = "gpt-4o"

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    # 如果没有显式传入 node2vec_params，则会调用这个 lambda 函数，自动生成并赋值为这个默认的字典
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # 用于比较查询结果质量的阈值
    query_better_than_threshold: float = 0.2

    query_mode: bool = False
    # 载入文件方式，0代表docx文件，1代表pdf文件直接解析，2代表pdf2markdown方式解析
    input_mode: int = 2

    cache_path = cache_path

    # 在对象初始化后调用此方法，主要作用为打印配置信息和根据配置进行一些设置调整
    def __post_init__(self):
        # 将对象的属性以键值对的形式打印出来，用于调试和日志记录
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")
        global_config = asdict(self)

        global_config_path = os.path.join(cache_path,"global_config.csv")
        # 将global_config字典保存到 CSV 文件
        with open(global_config_path, 'w', newline='') as file:
            for key, value in global_config.items():
                file.write(f"{key},{value}\n")
        
        # 确保工作目录存在，如果不存在则创建
        if os.path.exists(self.working_dir):
            logger.info(f"Using existing working directory {self.working_dir}")
        else:
            os.makedirs(self.working_dir)
            logger.info(f"Creating working directory {self.working_dir}")
        
        from preprocessing import chunking_func
        from pdf_preprocessing import chunking_func_pdf
        from pdf2md_preprocessing import chunking_func_pdf2md
        from text2graph import extract_entities_from_text
        from query import local_query
        
        # 实例化类
        if self.query_mode:
            self.localquery = local_query()
        else:
            self.ChunkingFunc = chunking_func()
            self.ChunkingFunc_pdf = chunking_func_pdf()
            self.ChunkingFunc_pdf2md = chunking_func_pdf2md()
            self.ExtractEntitiesFromText = extract_entities_from_text()
    
    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        response = await self.localquery.local_query(
            query,
            param,
        )
        return response

    def index(self, path):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aindex(path))
        
    async def aindex(self,path):
        from img2graph import img2graph
        if self.input_mode == 0:
            await self.ChunkingFunc.extract_text_and_images(path)
        elif self.input_mode == 1:
            await self.ChunkingFunc_pdf.extract_text_and_images(path)
        elif self.input_mode == 2:
            # 定义 kv_store_image_data.json 文件路径
            kv_store_path = os.path.join(self.working_dir, "kv_store_image_data.json")
            if os.path.exists(kv_store_path):
                with open(kv_store_path, "r", encoding="utf-8") as file:
                    content = json.load(file)
                    # 判断 JSON 文件是否为空 {}
                    if content == {}:
                        logger.info(f"{kv_store_path} exists but is empty. Proceeding with preprocess.")
                        await self.ChunkingFunc_pdf2md.extract_text_and_images(path)
                    else:
                        logger.info(f"{kv_store_path} exists and is not empty. Skipping preprocess.")
            else:
                await self.ChunkingFunc_pdf2md.extract_text_and_images(path)
        filepath = os.path.join(self.working_dir, 'kv_store_text_chunks.json')
        with open(filepath, 'r') as file:
            chunks = json.load(file)
        await self.ExtractEntitiesFromText.text_entity_extraction(chunks)
        imgfolderpath = os.path.join(self.working_dir, 'images')
        await img2graph(imgfolderpath)
        filepath2 = os.path.join(self.working_dir, 'kv_store_image_data.json')
        with open(filepath2, 'r') as file:
            image_data = json.load(file)
        from fusion import fusion, create_EntityVDB
        # 检查 image_data 是否为空字典
        if image_data:
            img_ids = list(image_data.keys())
            await fusion(img_ids)
        else:
            print("没有提取出图片，跳过 fusion 操作")
            createvdb = create_EntityVDB()
            await createvdb.create_vdb()