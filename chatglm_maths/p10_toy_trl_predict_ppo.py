# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/20 21:24
# @author  : Mo
# @function: many code from https://github.com/lvwerra/trl


import logging as logger
import traceback
import random
import math
import json
import copy
import sys
import os
import gc

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(path_root)
sys.path.append(path_root)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TORCH"] = "1"

# imports
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch, top_k_top_p_filtering
from transformers import AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm
import macropodus
import torch

from chatglm_maths.models.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from chatglm_maths.models.tokenization_chatglm import ChatGLMTokenizer
from chatglm_maths.models.ppo_trainer import PPOTrainer


class ChatGLMForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
    transformers_parent_class = ChatGLMForConditionalGeneration
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = ("summary_dropout_prob",
                      "v_head_initializer_range",
                      "v_head_init_strategy",
                      )
    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        self.is_peft_model = False
def load_model_state(model, path_dir="", model_name="pytorch_model.bin", device="cpu", model_save_dir="./"):
    """  仅加载模型参数(推荐使用)  """
    try:
        if path_dir:
            path_model = path_dir
        else:
            path_model = os.path.join(model_save_dir, model_name)
        model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))
        model.to(device)
        logger.info("******model loaded success******")
        logger.info("self.device: {}".format(device))
    except Exception as e:
        logger.info(str(e))
        raise Exception("******load model error******")
def respond_to_batch_new(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    end_ids = input_ids[:, -2:]
    start_ids = input_ids[:, :-2]
    for i in range(txt_len):
        # Get Logits
        outputs = model(torch.cat([start_ids, end_ids], dim=-1))
        next_token_logits = outputs[-1][:, -1, :]  # new-value
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        start_ids = torch.cat([start_ids, next_token.unsqueeze(-1)], dim=-1)
    return start_ids[:, -txt_len:]
def collect_score(ans, target, predict):
        """   计算得分   """
        score_1 = macropodus.sim(target, predict)
        try:
            predict_sp = predict.split("=")
            float_1 = eval(ans)
            float_2 = eval(predict_sp[1])
            score_2 = min(abs(float_1-float_2)/(float_1+1e-5), 1)
        except Exception as e:
            score_2 = 0.0
        scores = [score_1, score_2]
        return sum(scores) / len(scores)
def get_position_ids(seq, bos_token_id, gmask=True, position_encoding_2d=True):
    """  code from model_chatglm.py  """
    # context_length = seq.index(bos_token_id) + 1
    context_length = len(seq)
    position_ids = torch.arange(context_length, dtype=torch.long)
    if position_encoding_2d:
        seq_length = seq.index(bos_token_id)
        if not gmask:
            mask_position = seq_length - 1
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat((
            torch.zeros(seq_length, dtype=torch.long),
            torch.arange(context_length - seq_length, dtype=torch.long) + 1
        ))
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        if not gmask:
            seq_length = seq.index(bos_token_id)
            mask_position = seq_length - 1
            position_ids[context_length - 1:] = mask_position
    # position_ids = position_ids.unsqueeze(0)
    return position_ids
def get_masks(seq, bos_token_id):
    """  code from model_chatglm.py  """
    context_length = seq.index(bos_token_id)
    attention_mask = torch.ones((1, len(seq), len(seq)))
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    # attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()
    return attention_mask

# get models
pretrained_model_name_or_path = "THUDM/chatglm-6b"
model_save_path = "fine_tuning_t10"   # python c02_toy_gpu_train_small.py跑的模型
model_save_path_ppo = os.path.join(model_save_path, "ppo")
MAX_LEN=256
chatglm_config = ChatGLMConfig.from_json_file(os.path.join(model_save_path, "config.json"))
tokenizer = ChatGLMTokenizer.from_pretrained(pretrained_model_name_or_path)
model_chatglm = ChatGLMForConditionalGeneration(chatglm_config)
model = ChatGLMForCausalLMWithValueHead(pretrained_model=model_chatglm)
# model = ChatGLMForCausalLMWithValueHead.from_pretrained(model_save_path_ppo)
model = load_model_state(model, model_save_dir=model_save_path)
model = model.cuda()

# original_text = "1+1="
# response, history = model.pretrained_model.chat(tokenizer=tokenizer, query=original_text, history=[], max_length=256,
#                                num_beams=1, do_sample=True, top_p=0.7, temperature=0.95,
#                                )
# res_end = str(response).encode("utf-8", "ignore").decode("utf-8", "ignore")
# print(res_end)
# print("###############")
original_text = "1+1="
query_tensor = tokenizer.encode(original_text, return_tensors="pt").cuda()
response_tensor = respond_to_batch_new(model, query_tensor,
                        txt_len=MAX_LEN, top_k=0, top_p=1.0)
response_ids = response_tensor.detach().cpu().numpy().tolist()
response_text = tokenizer.decode(response_ids)
print(response_text)

