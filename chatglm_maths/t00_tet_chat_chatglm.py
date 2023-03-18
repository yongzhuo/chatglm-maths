# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/14 21:24
# @author  : Mo
# @function:


import traceback
import platform
import logging
import time
import json
import sys
import os
cpu_nums = "9"
os.environ["OMP_NUM_THREADS"] = cpu_nums  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = cpu_nums  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = cpu_nums  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = cpu_nums  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = cpu_nums  # export NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
os.environ["USE_TORCH"] = "1"


from transformers import AutoTokenizer, AutoModel


path_model_dir = ""
# tokenizer = AutoTokenizer.from_pretrained(path_model_dir, trust_remote_code=True)
# model = AutoModel.from_pretrained(path_model_dir, trust_remote_code=True).half().cuda()
# model = AutoModel.from_pretrained(path_model_dir, trust_remote_code=True).bfloat16()

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(4).cuda()
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float()
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).bfloat16()
time_start = time.time()
response, history = model.chat(tokenizer, "你好", history=[])
print(time.time() - time_start)
print(str(response).encode("utf-8", "ignore").decode("utf-8", "ignore"))
print(history)
time_start = time.time()
# 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(str(response).encode("utf-8", "ignore").decode("utf-8", "ignore"))
print(time.time() - time_start)


while True:
    time_start = time.time()
    history = []
    print("请输入:")
    ques = input()
    try:
        if ques.strip().upper() == "CLEAR":
            history = []
            print("clear ok")
            continue
        else:
            response, history = model.chat(tokenizer, ques, history=history)
            res_ende = str(response).encode("utf-8", "ignore").decode("utf-8", "ignore")
            print(res_ende)
    except Exception as e:
        print(str(e))
    print(time.time()-time_start)



# nohup python p01_tet_chat_chatglm.py > tc.p01_tet_chat_chatglm.py.log 2>&1 &
# tail -n 1000  -f tc.p01_tet_chat_chatglm.py.log
# |yongzhuo|

