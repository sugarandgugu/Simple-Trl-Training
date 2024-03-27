#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2024 Sugar. All Rights Reserved
#
########################################################################

"""
    File: data_generate.py
    Desc: 生成DPO数据
    Author: sugar(@google.com)
    Date: 2024-03-25 14:58
"""


import requests
from tqdm import tqdm
import json
import time
import urllib
import re

username = ""
password = ""

prompt = '''

'''


# 这个函数用于请求文心
def req_qiaojiang_yiyan_new(text):
    req_url = ""
    header = {
        "userName": username,
        "password": password,
        "Content-Type": "application/json"
    }
    # print(text[0])
    data = {
        "modelReq": [{
            "prompt": text,
            "temperature": 0.1,
            "top_p": 0.9,
            "penalty_score":1.5
        }]
    }
    resp = requests.post(url=req_url, data=json.dumps(data), headers=header, timeout=60)
    content = json.loads(resp.content)
    # print(content)
    return content

'''
    args:
        text -> str: 待处理文本
'''
def remove_special_symbols(text):
    pattern = r"[^\w\s]"
    processed_text = re.sub(pattern, ' ', text)
    return processed_text

# 封装一个函数生成数据
def generate_data(prompt,output_file):
    '''
        args: 
            prompt -> str: 大模型的提示  
            output_file -> str: 生成数据的路径   
            输入exmples:
            prompt: 你好
            data_count: 10
            output_file: 'DPO_ChatGLM/dpo_data/chat_bot/test_json.json'
        return None
    '''
    # 初始化list 最终格式 [{"prompt":"xxx","chosen":"xxx","rejected":"xxx"},{},{}]
    dpo_data = []
    # 遍历调用 API 打开关键词文件
    with open('') as f:
        key_words = f.read().split(';')
        # key_words = key_words[1:4]
    for word in key_words[1001:4000]:
        word = remove_special_symbols(word)
        print("这次生成的关键词是{}".format(word))
        # prompt = prompt.format(word)
        # print("\n\n",prompt)
        content = req_qiaojiang_yiyan_new(prompt.format(word))
        # 解析content的数据
        for con in content['data']['modelRes']:
            text = con['text']
            text = text.replace("```", "")
            print(text,type(text))
            # load成json 做一些处理 因为模型有时候生成的数据不一样
            try:
                text_dict = json.loads(text)
                text_dict['prompt'] = 'human\n' + text_dict['prompt']
                text_dict['chosen'] = 'assistant\n' + text_dict['chosen']
                text_dict['rejected'] = 'assistant\n' + text_dict['rejected']
                # print("text_dict",text_dict)
                dpo_data.append(text_dict)
            except:
                # print("出错-->{}".format(text))
                print("出错")
                continue
    # print("dpo-data",dpo_data)
    # API生成的数据会重复，这里做去重操作
    dpo_unique_list = []
    seen_combinations = set()

    for d in dpo_data:
        combination = (d['prompt'], d['chosen'], d['rejected'])
        if combination not in seen_combinations:
            dpo_unique_list.append(d)
            seen_combinations.add(combination)
    # 写入文件    
    with open(output_file, 'w') as f:
        json.dump(dpo_unique_list,f,ensure_ascii=False,indent=4)
    print(f"列表已成功输出为 JSON 文件: {output_file}")


if __name__ == '__main__':
    output_file = ''
    generate_data(prompt = prompt,output_file = output_file)

