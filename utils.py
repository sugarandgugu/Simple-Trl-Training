#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2024 Sugar. All Rights Reserved
#
########################################################################

"""
    File: utils.py
    Desc: 数据处理代码
    Author: sugar(@google.com)
    Date: 2024-03-27 15:19
"""
import torch
from loguru import logger
from datasets import load_dataset
from torch.utils.data import Dataset,DataLoader
from transformers import TrainingArguments, TrainerCallback


class dpo_dataset(Dataset):
    def __init__(self,file,tokenizer,max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # 打开json文件 用transformers
        self.data_list = load_dataset("json",data_files=file)['train']
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self,index):
        # 取出data_list的一条数据  --> {"chosen":xxx,"rejected":xxx,"prompt":xxx} 一条数据是这样的格式
        data = self.data_list[index]
        # 对prompt reject和chosen进行tokenize  判断是否需要截断 保证所有的input_ids都一样 不够长度的直接padding
        prompt_input_ids = self.tokenizer.encode(data['prompt'],add_special_tokens=False,max_length=self.max_seq_length,padding='max_length')
        chosen_input_ids = self.tokenizer.encode(data['chosen'],add_special_tokens=False,max_length=self.max_seq_length,padding='max_length')
        rejected_input_ids = self.tokenizer.encode(data['rejected'],add_special_tokens=False,max_length=self.max_seq_length,padding='max_length')

        # 设置labels
        chosen_labels = [-100] * len(prompt_input_ids) + chosen_input_ids
        rejected_labels = [-100] * len(prompt_input_ids) + rejected_input_ids
        chosen_input_ids = prompt_input_ids + chosen_input_ids
        rejected_input_ids = prompt_input_ids + rejected_input_ids

        assert len(chosen_labels) == len(chosen_input_ids)
        assert len(rejected_labels) == len(rejected_input_ids)

        inputs = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=[1]*len(prompt_input_ids),
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=[1]*len(chosen_input_ids),
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=[1]*len(rejected_input_ids),
            rejected_labels=rejected_labels,
        )
        return inputs
    def map(self, func, **kwargs):
        return self


class MyCallback(TrainerCallback):

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            print(state.global_step)

            data = random.choice(dataset['test'])
            input_ids = tokenizer.encode(data['prompt'],
                                         return_tensors='pt').to('cuda')

            out = generate(input_ids)

            print(tokenizer.decode(out[0]))
            print('=================')
            print(data['chosen'])
            print('=================')
