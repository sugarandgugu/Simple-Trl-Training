'''
  src: https://github.com/QwenLM/Qwen
  ps: 注意这个inference文件是适配千问大模型的，以及qwen1.5没有model.chat()功能，官方提供的是以下代码。
'''
import os
device = '3'  # 本次实验需要用到的卡
os.environ["CUDA_VISIBLE_DEVICES"] = device
os.environ['CUDA_LAUNCH_BLOCKING'] = device
import torch
import warnings
warnings.filterwarnings("ignore")
from peft import PeftModel
from transformers import AutoTokenizer,AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained('',trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("",trust_remote_code=True, device_map="auto")


prompt = "你好，你们这里有输送管吗？"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("response---->:",response)
