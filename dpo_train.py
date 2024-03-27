#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2024 Sugar. All Rights Reserved
#
########################################################################

"""
    File: dpo_train.py
    Desc: DPO训练代码
    Author: sugar(@google.com)
    Date: 2024-03-26 14:19
"""
import os
import torch
device = '3,6,7'  # 本次实验需要用到的卡
os.environ["CUDA_VISIBLE_DEVICES"] = device
os.environ['CUDA_LAUNCH_BLOCKING'] = device
from trl import DPOTrainer
from loguru import logger
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from utils import dpo_dataset
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,GPT2LMHeadModel,GPT2Tokenizer,DataCollatorForLanguageModeling,pipeline,DataCollatorForSeq2Seq
import warnings
warnings.filterwarnings("ignore")

def run():
    file = ''
    model_file = ''
    model_save_path = ''
    output_dir = ''
    tokenizer = AutoTokenizer.from_pretrained(model_file,trust_remote_code=True)
    # 量化
    model = AutoModelForCausalLM.from_pretrained(model_file,trust_remote_code=True,
                                        low_cpu_mem_usage=True, 
                                            quantization_config=BitsAndBytesConfig(
                                            load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.bfloat16,
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_quant_type='nf4'))
    model_ref = AutoModelForCausalLM.from_pretrained(model_file,trust_remote_code=True,
                                            low_cpu_mem_usage=True,                                
                                            quantization_config=BitsAndBytesConfig(
                                            load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.bfloat16,
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_quant_type='nf4'))
    # 加载数据集
    train_dataset = dpo_dataset(file = file, tokenizer = tokenizer, max_seq_length = 50)
    # Lora
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
                "q_proj",
                "v_proj",],
        task_type="CAUSAL_LM")
    # peft model
    model = get_peft_model(model,config)

    # 设置训练参数
    training_args = TrainingArguments(
        num_train_epochs = 1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing = True,
        learning_rate=3e-4,
        output_dir=output_dir,
        report_to="tensorboard",
        save_total_limit = 1,
        optim = "paged_adamw_32bit",
        logging_strategy = "steps",
        logging_steps = 50,
        seed = 103,
        fp16 = True,
        lr_scheduler_type = "constant_with_warmup",
        warmup_steps = 100,
    )
    # 设置dpo trainer
    dpo_trainer = DPOTrainer(
        model, # 经 SFT 的基础模型
        model_ref, # 一般为经 SFT 的基础模型的一个拷贝
        beta=0.1, # DPO 的温度超参
        train_dataset=train_dataset, # 上文准备好的数据集
        tokenizer=tokenizer, # 分词器
        args=training_args) # 训练参数，如: batch size, 学习率等

    # 开始训练
    logger.info("=============== starting training ===============")
    train_result = dpo_trainer.train()
    logger.info("=============== Saving Model ===============")
    dpo_trainer.save_model(model_save_path)
    logger.info("=============== Saving Metrics ===============")
    metrics = train_result.metrics
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()
    
if __name__ == '__main__':
    run()




















