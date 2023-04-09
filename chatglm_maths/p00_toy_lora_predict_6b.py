# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/16 21:54
# @author  : Mo
# @function: train total of 6b


import logging as logger
import traceback
import random
import math
import time
import copy
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(path_root)
sys.path.append(path_root)
CUDA_VISIBLE_DEVICES = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
cpu_nums = "9"
os.environ["OMP_NUM_THREADS"] = cpu_nums  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = cpu_nums  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = cpu_nums  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = cpu_nums  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = cpu_nums  # export NUMEXPR_NUM_THREADS=1
os.environ["USE_TORCH"] = "1"

from peft import PeftModel, get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup, GenerationConfig
from tqdm import tqdm, trange
import torch.nn as nn
import numpy as np
import datasets
import torch

from chatglm_maths.models.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from chatglm_maths.models.tokenization_chatglm import ChatGLMTokenizer


is_toy = True
if is_toy:
    use_cuda = True
    quantize_type = None  # None, 16, 8, 4
    batch_size = 16
    len_corpus = 64  # batch_size*3820
    num_layers = 28  # 1
    warmup_steps = 1
    pretrained_model_name_or_path = "THUDM/chatglm-6b"  # None
    evaluate_steps = 3820
else:
    use_cuda = False
    quantize_type = None  # None, 16, 8, 4
    batch_size = 16
    len_corpus = batch_size * 3820  # batch_size*3820
    num_layers = 28
    warmup_steps = 100
    pretrained_model_name_or_path = "THUDM/chatglm-6b"
    evaluate_steps = int(len_corpus / batch_size / 3) + 1  # 3820


model_save_path = "./fine_tuning_lora_c00"
# os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
quantize_type = None  # None, 16, 8, 4
seed = 2023
weight_decay = 5e-4
lr = 2e-5
eps = 1e-5  # 半精度需要设置大一点
betas = (0.9, 0.999)
grad_accum_steps = 1
stop_epochs = 3
epochs = 21
logger_steps = 100
max_grad_norm = 1
float_precision = 2
max_length = 256
max_coeff = 100  # 数据在 -max_coeff 到 max_coeff 之间
MAX_LENGTH_Q = 1024 - 2
MAX_LENGTH_A = 1024 - 2
MAX_LENGTH_QA = MAX_LENGTH_Q + MAX_LENGTH_A + 2
device = "cuda:{}".format(CUDA_VISIBLE_DEVICES) if (torch.cuda.is_available() \
            and use_cuda and CUDA_VISIBLE_DEVICES != "-1") else "cpu"
# attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)


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
    # torch.save(model.state_dict(), path_model)
    grad_params_dict = {k: v.to("cpu")
                        for k, v in model.named_parameters()
                        if v.requires_grad == True}
    torch.save(grad_params_dict, path_model)
    logger.info("******model_save_path is {}******".format(path_model))
def load_model_state(path_dir="", model=None, model_save_dir="./", model_name="pytorch_model.bin", device="cpu", model_save_path="./"):
    """  仅加载模型参数(推荐使用)  """
    try:
        if path_dir:
            path_model = path_dir
        else:
            path_model = os.path.join(model_save_dir, model_name)
        peft_config = LoraConfig.from_pretrained(model_save_dir)
        peft_config.inference_mode=True
        model = get_peft_model(model, peft_config)
        sae_dict_lora = torch.load(path_model, map_location=torch.device(device))
        model.load_state_dict(sae_dict_lora, strict=False)
        # model.to(device)
        logger.info("******model loaded success******")
        logger.info("self.device: {}".format(device))
    except Exception as e:
        logger.info(str(e))
        raise Exception("******load model error******")
    return model
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
def evaluate(model, tokenizer, len_corpus=batch_size, device="cpu"):
    """  验证  """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge import Rouge

    rouge = Rouge()
    smooth = SmoothingFunction().method1

    model.eval()
    pabr = tqdm(total=len_corpus//batch_size, desc="eval")
    ans_true = []
    ans_pred = []
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    for idx, (inputs, batch_qtext, batch_qans) in enumerate(generator.__iter__(len_corpus, max_coeff, device)):  # step
        pabr.update(1)
        for jdx, batch_qtext_i in enumerate(batch_qtext):
            try:
                response, history = model.chat(tokenizer=tokenizer, query=batch_qtext_i, max_length=max_length,
                                               num_beams=1, do_sample=True, top_p=0.7, temperature=0.95)
                batch_qans_i = batch_qans[jdx]
                ans_true.append(batch_qans_i)
                ans_pred.append(response)
                scores = rouge.get_scores(hyps=response, refs=batch_qans_i)
                rouge_1 += scores[0]["rouge-1"]["f"]
                rouge_2 += scores[0]["rouge-2"]["f"]
                rouge_l += scores[0]["rouge-l"]["f"]
                bleu += sentence_bleu(references=[list(batch_qans_i)],
                                      hypothesis=list(response),
                                      smoothing_function=smooth)
                if idx == 0 and jdx < 5:
                    print("batch_qtext_{}: {}".format(jdx, batch_qtext_i[:64]))
                    print("batch_qans_{}: {}".format(jdx, batch_qans_i[:64]))
                    print("response_{}: {}".format(jdx, response[:64]))
            except Exception as e:
                print(traceback.print_exc())
                print(batch_qtext_i)
    rouge_1, rouge_2, rouge_l, bleu = rouge_1 / len_corpus, rouge_2 / len_corpus, rouge_l / len_corpus, bleu / len_corpus
    score_dict = {"rouge-1": rouge_1, "rouge-2": rouge_2, "rouge-l": rouge_l, "bleu": bleu}
    print(score_dict)
    score_avg = round(sum(list(score_dict.values()))/len(score_dict.keys()), 5)
    return score_avg, score_dict
def set_random_seed(seed):
    """ 设置随机种子 """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
class Generator:
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self, batch_size=16, float_precision=3):
        self.float_precision = float_precision

    def generate_one(self, max_coeff=100, use_gaussian=True, use_flag=True, dim1=1, dim2=2):
        """   返回一个abs + encode 后的数据   """
        # int and float
        if use_flag:
            x = np.random.randint(low=-max_coeff, high=max_coeff, size=(dim1, dim2))
            return x
        else:
            if use_gaussian:
                x = np.array(max_coeff / math.sqrt(3.0) * np.random.randn(dim1, dim2))
            else:
                x = np.array(max_coeff * (2 * np.random.rand(dim1, dim2) - 1))
            return x

    def generate_line(self, op="+", max_coeff=100, use_gaussian=True, dim1=1, dim2=2):
        """   返回一行 encode + ans 后的数据   """
        use_flag = random.choice([0, 1])
        x_12 = self.generate_one(max_coeff, use_gaussian, use_flag, dim1, dim2)
        if use_flag:
            x_1 = abs(int(x_12[0][0]))
            x_2 = abs(int(x_12[0][1]))
        else:
            x_1 = round(abs(float(x_12[0][0])), self.float_precision)
            x_2 = round(abs(float(x_12[0][1])), self.float_precision)

        if op == "-":
            ans = x_1 - x_2
        elif op == "*":
            ans = x_1 * x_2
        elif op == "/":
            ans = x_1 / x_2
        else:
            ans = x_1 + x_2
        x_encode = str(x_1) + op + str(x_2)
        ans = round(ans, self.float_precision)
        if use_flag:
            ans = abs(int(ans))
        if random.choice([0, 1]):
            x_encode += "="
        y_encode = x_encode + "=" + str(ans)
        return x_encode, y_encode.replace("==", "=")

    def __iter__(self, len_corpus=50000, max_coeff=100, device="cpu"):
        for idx, _ in enumerate(range(len_corpus)):
            batch_xds_0 = []
            batch_xds_1 = []
            batch_qtext = []
            batch_qans = []
            for _ in range(batch_size):
                x, y = self.generate_line(op="+", max_coeff=max_coeff, use_gaussian=True, dim1=1, dim2=2)
                """
                "gigaword", "cnn_dm", "cnn_dm_original", "xsum":
                    source_tokens = [cls_id, mask_id] + " Content:"  + " " + source_text
                "squad_generation":
                    source_tokens = [cls_id] + source_text.rstrip() + \
                                    " Question:" + (" " + answer_patten) + \
                                    [mask_id] + " Answer: " + answer
                "squad"/"squad_v1":
                    source_tokens = [cls_id] + " " + source_text.rstrip()+
                                    " " + question + mask_id, period_id('.')\
                                    + source_tokens[:max_src_length]
                                    + " " + target_text + [eop_id]
                "cmrc":
                    source_tokens = [cls_id] + "问题：" + question + "答案：" + \
                                    [mask_id] + source_tokens[:max_src_length]
                """
                prompts = [("问:", "答:"), ("问题:", "答案:"), ("计算: ", "回答:"),
                           ("计算题:", "解答:"), ("口算:", "解:"), ("简便运算: ", "剖析:"),
                           ("数学题:", "点拨:"), ("初等数学: ", "解析:")]
                use_pormpts = False
                if use_pormpts:
                    prompt = random.choice(prompts)
                    x = prompt[0] + "\n" + x + "\n" + prompt[1] + "\n"
                x_encode = tokenizer.encode(x)  # encode自己多生成了一个空格_
                y_encode = tokenizer.encode(y)[:-2]
                if len(x) + len(y) > (MAX_LENGTH_Q + MAX_LENGTH_A):
                    x = x[:MAX_LENGTH_Q]
                    y = y[:MAX_LENGTH_A]
                if not x:
                    y = [ID_PAD, ID_BOS]
                if x[-1] != ID_BOS:
                    x += [ID_BOS]
                if not y:
                    y = [ID_PAD, ID_EOS]
                if y and y[-1] != ID_EOS:
                    y += [ID_EOS]
                batch_xds_0.append(x_encode)
                batch_xds_1.append(y_encode)
                batch_qtext.append(x)
                batch_qans.append(y)
            lens_01 = [len(batch_xds_0[i]) + len(batch_xds_1[i]) for i in range(len(batch_xds_0))]
            batch_attention_mask = []
            batch_position_ids = []
            batch_input_ids = []
            batch_labels = []
            lens_01_max = min(MAX_LENGTH_QA, max(lens_01) + 1)
            for jdx in range(len(lens_01)):
                x = batch_xds_0[jdx]
                y = batch_xds_1[jdx]
                len_padding = lens_01_max - len(x) - len(y) - 1
                labels = [-100] * len(x) + y + [ID_EOS] + [-100] * len_padding
                input_ids = x + y + [ID_EOS] * (len_padding + 1)
                tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
                tensor_labels = torch.tensor(labels, dtype=torch.long)
                position_ids = get_position_ids(input_ids, ID_BOS, gmask=True, position_encoding_2d=True)
                attention_mask = get_masks(input_ids, ID_BOS)
                batch_attention_mask.append(attention_mask)
                batch_position_ids.append(position_ids)
                batch_input_ids.append(tensor_input_ids)
                batch_labels.append(tensor_labels)
            if idx < 2:
                print("batch_attention_mask_0: {}".format(batch_attention_mask[0]))
                print("batch_position_ids_0: {}".format(batch_position_ids[0]))
                print("batch_input_ids_0: {}".format(batch_input_ids[0]))
                print("batch_labels_0: {}".format(batch_labels[0]))
                print("batch_qtext_0: {}".format(batch_qtext[0]))
                print("batch_qans_0: {}".format(batch_qans[0]))
            inputs = {"attention_mask": torch.stack(batch_attention_mask).to(device),
                      "position_ids": torch.stack(batch_position_ids).to(device),
                      "input_ids": torch.stack(batch_input_ids).to(device),
                      "labels": torch.stack(batch_labels).to(device),
                      }
            yield inputs, batch_qtext, batch_qans


set_random_seed(seed)
## 构建算式
generator = Generator(batch_size=batch_size, float_precision=2)
generator_line, y = generator.generate_line()
print("generator_calculate_line: {}".format(generator_line))


# The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
chatglm_config = ChatGLMConfig.from_pretrained(pretrained_model_name_or_path)
tokenizer = ChatGLMTokenizer.from_pretrained(pretrained_model_name_or_path)
text = ("1、2", "3、4")
x_encode = tokenizer.encode(text[0])
y_encode = tokenizer.encode(text[1])
tokens_id = x_encode[:-1] + y_encode[:-2] + [y_encode[-1]]
print(text)
print("tokenizer.encode_plus: {}".format(tokens_id))
print("tokenizer.vocab_size: {}".format(tokenizer.vocab_size))
ID_CLS = tokenizer.sp_tokenizer["<ENC>"]
# ID_SEP = tokenizer.sp_tokenizer["<pad>"]
ID_PAD = tokenizer.sp_tokenizer["<pad>"]
ID_MASK = tokenizer.sp_tokenizer["[MASK]"]
ID_gMASK = tokenizer.sp_tokenizer["[gMASK]"]
ID_sMASK = tokenizer.sp_tokenizer["[sMASK]"]
ID_SPACE = tokenizer.sp_tokenizer["▁"]
ID_BOS = tokenizer.sp_tokenizer["<sop>"]
ID_EOS = tokenizer.sp_tokenizer["<eop>"]
ID_S1 = tokenizer.sp_tokenizer["<s>"]
ID_S2 = tokenizer.sp_tokenizer["</s>"]


## 字典embedding也很大, 2w图像token, 8w英文token, 5w中文字词token(尝试剔除图片+英文(只保留计算+单词)---待实验?)
# model = ChatGLMForConditionalGeneration(chatglm_config)
model = ChatGLMForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
# model = model.bfloat16()
# model = model.half().cuda()  # .to(device)
# print("model cuda ok!")
# model = prepare_model_for_int8_training(model, use_gradient_checkpointing=False,
#             layer_norm_names=["layer_norm"])
            # layer_norm_names=["input_layernorm", "post_attention_layernorm",])
# print("prepare_model_for_int8_training!")
# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()
# model.disable_input_require_grads()
model.config.use_cache = False
model.supports_gradient_checkpointing = False
model.is_parallelizable = False
model.model_parallel = False
class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
# model.is_parallelizable = False
# model.model_parallel = False
model.lm_head = CastOutputToFloat(model.lm_head)
model = load_model_state(model=model, model_save_dir=model_save_path)
# model.eval()

def predict(text):
    prompt = text  # generate_prompt(text)
    inputs = tokenizer([prompt], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=0.95,
        top_p=0.95,
        top_k=50,
        num_beams=1,
        do_sample=True,
        penalty_alpha=1.5,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    # print(output.split("### Response:")[1].strip())
    # print(output[len(input_ids.detach().cpu().numpy().tolist()[0])-1:])
    print(output)

# model.device = device
# load_model_state(model_save_path=model_save_path)
# if use_cuda:
    # model = model.half().to(device)
    # model = model.half().cuda()
    # model = model.cuda()
    # print("model cuda ok!")
# else:
#     model = model.bfloat16()

model = model.bfloat16()
print("model.chat start")
predict(generator_line)

while True:
    time_start = time.time()
    history = []
    print("请输入:")
    ques = input()
    print("请稍等...")
    try:
        if ques.strip().upper() == "CLEAR":
            history = []
            print("clear ok")
            continue
        else:
            predict(ques)
    except Exception as e:
        print(str(e))
    print(time.time()-time_start)


"""
数学算式(加减乘除)微调, max_coeff=100以内
## 只训练最后一层的参数

"""


