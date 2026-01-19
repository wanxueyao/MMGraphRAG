from openai import OpenAI

key = "EMPTY"
model = "llama-3.1-70b"
url = "http://localhost:6000/v1"

client = OpenAI(
        base_url=url, api_key=key
    )

def extract_answer(question, output, prompt, model_name=model):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                "role": "assistant",
                "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output)
                }
            ],
            # temperature=0.0,
            # max_tokens=256,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0
        )
        response = response.choices[0].message.content
    except:
        response = "Failed"
    
    return response