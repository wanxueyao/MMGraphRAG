from openai import OpenAI
# 回答评测大模型
API_KEY = "EMPTY"
URL = "http://localhost:6000/v1"
MODEL = "llama-3.1-70b"
MM_API_KEY = ""
MM_MODEL = "qwen-vl-max"
MM_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 用LLM进行评测
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

    text_response = completion.choices[0].message.content
    return text_response

# 调用多模态LLM
def get_mmllm_response(cur_prompt, system_content):
    client = OpenAI(
        base_url=MM_URL, api_key=MM_API_KEY
    )

    completion = client.chat.completions.create(
        model=MM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": cur_prompt},
        ],
    )

    text_response = completion.choices[0].message.content
    return text_response

# 调用Ovis1.6
import torch
from PIL import Image

def get_mmllm_response_ovis(model, image_path, text_prompt):
    # 处理图片
    image = Image.open(image_path)
    query = f'<image>\n{text_prompt}'
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    
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
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        
    return output