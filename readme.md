# Simple Trl Training

### 基本介绍👋

这个项目主要是基于Huggingface的Trl库，利用DPO算法对语言大模型进行偏好对齐训练。其项目结构如下:

模型自动使用peft库的量化以及lora。

```c++
simple-dpo-training
  utils.py   --- 这个文件主要用于处理dataloader一些数据处理的过程
  dpo_data   --- 主要存放数据集文件
  data_generate.py --- 这个文件是利用API生成需要的数据
  dpo_train.py    ---  训练文件
```

数据集格式如下:

```c++
[
    {
        "prompt": "你们这里有航天军工PCB板吗?",
        "chosen": "您好，关于航天军工PCB板，我们需要先确认一下。麻烦您留下联系方式。",
        "rejected": "对不起，我们这里没有航天军工PCB板，您可能需要去其他商家那里看看。"
    },
    {
        "prompt": "你们这里有高速高频PCB吗?",
        "chosen": "您好，关于高速高频PCB，我们需要先确认一下库存和型号。",
        "rejected": "对不起，我们这里没有高速高频PCB，您可能需要去其他商家那里看看。"
    },
]
```

### 使用该项目🤗

在使用之前请确保您已经按照格式准备了数据，下面需要修改以下路径，即可运行该项目，在dpo_train.py的run函数下: 注意file是一个json文件。

```python
file = ''
model_file = ''
model_save_path = ''
output_dir = ''
```

在命令行中:

```python
python dpo_train.py
```

后台启动该项目:
ps: 在后台挂载启动，这样关了服务器代码还是在运行的，不会断掉。
```python
nohub python dpo_train.py > train_log.log
```

启动tensorboard查看日志: 确保已经安装了tensorboard

```python
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```
tensorboard --logdir='your path'
```
### 报错
1、显存爆了，参考如下:
```python
# 这样写，指定的显卡会检测不到，一直用第0张显卡
import os
import torch
import bitsandbytes as bnb
import torch.nn as nn
device = '2,3,5'  # 本次实验需要用到的卡
os.environ["CUDA_VISIBLE_DEVICES"] = device
os.environ['CUDA_LAUNCH_BLOCKING'] = device
# ================ 分割线 ===================
# 这样写就能正常运行  (不知道啥原因)  ps: 是因为先导入了torch包，默认调用了第0张显卡，把torch放后面即可。
import os
device = '2,3,5'  # 本次实验需要用到的卡
os.environ["CUDA_VISIBLE_DEVICES"] = device
os.environ['CUDA_LAUNCH_BLOCKING'] = device
import torch
import bitsandbytes as bnb
import torch.nn as nn
```

### 参考资料

1、https://zhuanlan.zhihu.com/p/641620563

2、https://github.com/yangjianxin1/Firefly/tree/master

3、https://github.com/lansinuote/Simple_TRL/blob/main/1.dpo_trl%E8%AE%AD%E7%BB%83.ipynb

4、https://github.com/datawhalechina/self-llm/blob/master/Qwen/04-Qwen-7B-Chat%20Lora%20%E5%BE%AE%E8%B0%83.md
