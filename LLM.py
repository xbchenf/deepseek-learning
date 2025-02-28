from langchain.llms.base import LLM
from typing import Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class DeepSeek_R1_Distill_Qwen_LLM(LLM):
    """ 自定义DeepSeek-R1 LLM接入实现 """
    
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, mode_name_or_path: str):
        super().__init__()
        print("Initializing local model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            mode_name_or_path, 
            use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            mode_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.generation_config = GenerationConfig.from_pretrained(mode_name_or_path)
        
    def _call(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=8192
        )
        response = self.tokenizer.decode(
            generated_ids[0][len(model_inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        return response

    @property
    def _llm_type(self) -> str:
        return "DeepSeek_R1_Distill_Qwen_LLM"