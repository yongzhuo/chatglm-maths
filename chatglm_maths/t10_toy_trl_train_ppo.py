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
from peft import prepare_model_for_int8_training
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
def respond_to_batch_new(model, queries, txt_len=128, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    end_ids = input_ids[:, -2:]
    start_ids = input_ids[:, :-2]
    for i in range(txt_len):
        # Get Logits
        outputs = model(torch.cat([start_ids, end_ids], dim=-1))
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        start_ids = torch.cat([start_ids, next_token.unsqueeze(-1)], dim=-1)
        # EOS
        if next_token.detach().cpu().numpy()[0] == tokenizer.eos_token_id:
            return start_ids
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
model_save_path = "./fine_tuning_t10"
MAX_LEN = 128
def save_model_state(model, config=None, model_save_dir="./", model_name="pytorch_model.bin", config_name="config.json"):
    """  仅保存模型参数(推荐使用)  """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # save config
    if config:
        path_config = os.path.join(model_save_dir, config_name)
        config.to_json_file(path_config)
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    torch.save(model.state_dict(), path_model)
    logger.info("******model_save_path is {}******".format(path_model))
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


tokenizer = ChatGLMTokenizer.from_pretrained(pretrained_model_name_or_path)
ID_PAD = tokenizer.convert_tokens_to_ids(["<pad>"])[0]
ID_UNK = tokenizer.convert_tokens_to_ids(["<unk>"])[0]
ID_BOS = tokenizer.convert_tokens_to_ids(["<s>"])[0]
ID_EOS = tokenizer.convert_tokens_to_ids(["</s>"])[0]
print(ID_PAD)
print(ID_UNK)
print(ID_BOS)
print(ID_EOS)


model_chatglm = ChatGLMForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path,
    # load_in_8bit=True,
    # device_map="auto"
    )
# model_chatglm = model_chatglm.quantize(8)
# model_chatglm = model_chatglm.half()
model_chatglm = prepare_model_for_int8_training(model_chatglm,
        use_gradient_checkpointing=True,
        output_embedding_layer_name="lm_head",
        #layer_norm_names=[],
        layer_norm_names=["post_attention_layernorm",
                          "input_layernorm",
                          "ln_f"
                          ],
        )
model = ChatGLMForCausalLMWithValueHead(pretrained_model=model_chatglm)

model = model.cuda()
model_ref = create_reference_model(model)
# initialize trainer
ppo_config = PPOConfig(model_name="ChatGLMForCausalLMWithValueHead",
                       steps=20000,
                       mini_batch_size=1,
                       learning_rate=1.41e-5,
                       adap_kl_ctrl=True,
                       init_kl_coef=0.2,
                       batch_size=1,
                       max_grad_norm=1,
                       seed=2023,
                       # log_with="tensorboard"
                       )
# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# dataset
path_dataset = "math23k_trainset.sample.json"
with open(path_dataset, mode="r", encoding="utf-8") as fj:
    math23k_list = json.load(fj)
    fj.close()

for math23k_dict in tqdm(math23k_list, desc="tqdm"):
    original_text = math23k_dict.get("original_text", "")
    equation = math23k_dict.get("equation", "")
    ans = math23k_dict.get("ans", "")
    target_text = equation.replace("x=", "") + "=" + ans

    # encode a query
    query_tensor = tokenizer.encode(original_text, return_tensors="pt").cuda()
    # get model response
    # print(query_tensor)

    response_tensor = respond_to_batch_new(model_ref, query_tensor, txt_len=MAX_LEN, top_k=0, top_p=1.0)
    # define a reward for response
    # (this could be any reward such as human feedback or output from another model)
    response_ids = response_tensor.detach().cpu().numpy().tolist()
    response_text = tokenizer.decode(response_ids)
    # print(response_ids)

    score_cal = collect_score(ans, target_text, response_text)
    reward = [torch.tensor(score_cal)]
    print(reward)
    # train model for one step with ppo
    # [torch.cat((query_tensor[0][:-1], torch.tensor(
    #     [tokenizer.bos_token_id], dtype=torch.long).cuda()))]
    # train_stats = ppo_trainer.step([query_tensor[0][:-2]], [response_tensor[0]], reward)
    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

save_model_state(model, config=None, model_save_dir=model_save_path + "/ppo")

"""
log
CUDA SETUP: Loading binary /anaconda3/envs/py371/lib/python3.7/site-packages/bitsandbytes/libbitsandbytes_cuda114_nocublaslt.so...
2023-04-09 22:42:35,951 - seg_basic.py[line:19] - INFO: path of dict cache is /anaconda3/envs/py371/lib/python3.7/site-packages/macropodus/data/cache/macropodus.cache!
2023-04-09 22:42:36,206 - textcleaner.py[line:37] - INFO: 'pattern' package not found; tag filters are not available for English
2023-04-09 22:42:36,207 - word2vec.py[line:19] - INFO: path of w2v cache is /anaconda3/envs/py371/lib/python3.7/site-packages/macropodus/data/cache/word2vec_char.cache!
20003
20000
20001
20002
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:11<00:00,  1.44s/it]
chatglm_maths/models/ppo_trainer.py:223: UserWarning: No dataset is provided. Make sure to set config.batch_size to the correct value before training.
  UserWarning,
tqdm:   0%|                                                               | 0/37 [00:00<?, ?it/s]      [tensor(0.0337, dtype=torch.float64)]
tqdm:   3%|████▍                                                          | 1/37 [00:11<06:55, 11.54s/it]      [tensor(0.0383, dtype=torch.float64)]
tqdm:   5%|████████▉                                                      | 2/37 [00:21<06:07, 10.50s/it]      [tensor(0.0356
7 [05:22<00:08,  8.94s/it][tensor(0.0283, dtype=torch.float64)]
tqdm: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 37/37 [05:31<00:00,  8.96s/it]
2023-04-09 22:48:57,715 - t10_toy_trl_train_ppo.py[line:127] - INFO: ******model_save_path is ./fine_tuning_t10/ppo/pytorch_model.bin******
"""

