from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn, json, datetime, torch, re

# 设备配置
DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}"

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post = await request.json()
    prompt = json_post.get('prompt')
    
    # 模型推理过程
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=8192)
    
    # 响应处理
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    think, answer = re.search(r'<think>(.*?)(.*)', response, re.DOTALL).groups()
    
    return {
        "response": response,
        "think": think.strip(),
        "answer": answer.strip(),
        "status": 200,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == '__main__':
    # 注意修改模型实际路径
    model_path = '/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                device_map=CUDA_DEVICE,
                                                torch_dtype=torch.bfloat16)
    uvicorn.run(app, host='0.0.0.0', port=6006)