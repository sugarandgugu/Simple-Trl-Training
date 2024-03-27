from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os
import torch
device = '3,6,7'  # 本次实验需要用到的卡
os.environ["CUDA_VISIBLE_DEVICES"] = device
os.environ['CUDA_LAUNCH_BLOCKING'] = device
'''
    https://github.com/yangjianxin1/Firefly/blob/master/script/merge_lora.py
'''

"""
使用该脚本，将lora的权重合并大base model中
"""


def merge_lora_to_base_model():
    # 大模型
    model_name_or_path = ''
    adapter_name_or_path = ''
    # 保存路径
    save_path = ''

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()