# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/16 21:54
# @author  : Mo
# @function: train total of 6b


import logging as logger
import traceback
import random
import math
import copy
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(path_root)
sys.path.append(path_root)
CUDA_VISIBLE_DEVICES = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
# cpu_nums = "9"
# os.environ["OMP_NUM_THREADS"] = cpu_nums  # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = cpu_nums  # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = cpu_nums  # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = cpu_nums  # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = cpu_nums  # export NUMEXPR_NUM_THREADS=1
os.environ["USE_TORCH"] = "1"

from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import numpy as np
import torch

from chatglm_maths.models.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from chatglm_maths.models.tokenization_chatglm import ChatGLMTokenizer
from transformers import AutoTokenizer


is_toy = False
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
    use_cuda = True
    quantize_type = None  # None, 16, 8, 4
    batch_size = 16
    len_corpus = batch_size * 3820  # batch_size*3820
    num_layers = 28
    warmup_steps = 100
    pretrained_model_name_or_path = "THUDM/chatglm-6b"
    evaluate_steps = int(len_corpus / batch_size / 3) + 1  # 3820


model_save_path = "./fine_tuning_c00_gpu"
# os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
quantize_type = None  # None, 16, 8, 4
seed = 2023
weight_decay = 5e-4
lr = 2e-5
eps = 1e-5  # 半精度需要设置大一点
betas = (0.9, 0.999)
grad_accum_steps = 4
stop_epochs = 3
epochs = 21
logger_steps = 100
max_grad_norm = 1
float_precision = 2
max_length = 256
max_coeff = 20  # 数据在 -max_coeff 到 max_coeff 之间
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
    torch.save(model.state_dict(), path_model)
    logger.info("******model_save_path is {}******".format(path_model))
def load_model_state(path_dir="", model_name="pytorch_model.bin", device="cpu", model_save_path="./"):
    """  仅加载模型参数(推荐使用)  """
    try:
        if path_dir:
            path_model = path_dir
        else:
            path_model = os.path.join(model_save_path, model_name)
        model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))
        # model.to(device)
        logger.info("******model loaded success******")
        logger.info("self.device: {}".format(device))
    except Exception as e:
        logger.info(str(e))
        raise Exception("******load model error******")
def get_position_ids(seq, bos_token_id, gmask=False, position_encoding_2d=True):
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
def set_random_seed(seed):
    """ 设置随机种子 """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
        # use_flag = True
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
                    x = "\n" + prompt[0] + "\n" + x + "\n" + prompt[1] + "\n"
                x_encode = tokenizer.encode(x)  # encode自己多生成了一个空格_
                y_encode = tokenizer.encode(y)[:-2]
                if len(x_encode) + len(x_encode) > MAX_QA_LENGTH:
                    x_encode = x_encode[:MAX_Q_LENGTH]
                    y_encode = y_encode[:MAX_A_LENGTH]
                batch_xds_0.append(x_encode)
                batch_xds_1.append(y_encode)
                batch_qtext.append(x)
                batch_qans.append(y)
            lens_01 = [len(batch_xds_0[i]) + len(batch_xds_1[i]) for i in range(len(batch_xds_0))]
            batch_attention_mask = []
            batch_position_ids = []
            batch_input_ids = []
            batch_labels = []
            lens_01_max = max(lens_01) + 1
            for jdx in range(len(lens_01)):
                x = batch_xds_0[jdx]
                y = batch_xds_1[jdx]
                len_padding = lens_01_max - len(x) - len(y) - 1
                labels = [-100] * (len(x) - 1) + [ID_BOS] + y + [ID_EOS] + [-100] * len_padding
                input_ids = x + y + [ID_EOS] * (len_padding + 1)
                tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
                tensor_labels = torch.tensor(labels, dtype=torch.long)
                position_ids = get_position_ids(input_ids, ID_BOS, gmask=False, position_encoding_2d=True)
                attention_mask = get_masks(input_ids, ID_BOS)
                batch_attention_mask.append(attention_mask)
                batch_position_ids.append(position_ids)
                batch_input_ids.append(tensor_input_ids)
                batch_labels.append(tensor_labels)
            if idx < 5:
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
generator_line = generator.generate_line()
print("generator_calculate_line: {}".format(generator_line))


# The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
chatglm_config = ChatGLMConfig.from_pretrained(pretrained_model_name_or_path)
tokenizer = ChatGLMTokenizer.from_pretrained(pretrained_model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
text = ("1、2", "3、4")
# tokens_id = tokenizer.encode_plus(text[0], text[1]).input_ids
x_encode = tokenizer.encode(text[0])
y_encode = tokenizer.encode(text[1])
tokens_id = x_encode[:-1] + y_encode[:-2] + [y_encode[-1]]
print(text)
print("tokenizer.encode_plus: {}".format(tokens_id))
print("tokenizer.vocab_size: {}".format(tokenizer.vocab_size))
MAX_Q_LENGTH = 1024 - 2
MAX_A_LENGTH = 1024 - 2
MAX_QA_LENGTH = MAX_Q_LENGTH + MAX_A_LENGTH + 2
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

# ### test
# chatglm_config.num_layers = num_layers
# chatglm_config.torch_dtype = "float16"

## 字典embedding也很大, 2w图像token, 8w英文token, 5w中文字词token(尝试剔除图片+英文(只保留计算+单词)---待实验?)
model = ChatGLMForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
# ## 只训练最后一层的参数
layer_not_freeze = [27]  # [25, 26, 27]  # [str(i) for i in range(27) if i > 15]  # "27"
for k, v in model.named_parameters():
    flag_layer = False  # 不冻结的网络层
    if layer_not_freeze:
        for layer_i in layer_not_freeze:
            if "layers.{}.".format(layer_i) in k:
                flag_layer = True

    if ("lm_head" in k or "final_layernorm" in k):
        v.requires_grad = True
    elif flag_layer:
        v.requires_grad = True
    # elif "word_embeddings.weight" in k:
    #     v.requires_grad = True
    else:
        v.requires_grad = False
for k, v in model.named_parameters():
    print(k, v.requires_grad)

if use_cuda:
    model = model.half().cuda()  # .to(device)
    print("model cuda ok!")
else:
    model = model.bfloat16()


print("model.chat start")
response, history = model.chat(tokenizer, generator_line, max_length=max_length, history=[])
print(str(response).encode("utf-8", "ignore").decode("utf-8", "ignore"))

# 实验, 不计算, 1层
# time.sleep(300000)
# layer-1-init == 3588MiB  3856MiB
# layer-2-init == 3972MiB  4156MiB
# layer-3-init == 4356MiB  4636MiB
# 每增加一层: 4356-3972 = 3972-3588 = 384M

####   实验, 计算, 1层
# count = 0
# for inputs, batch_ans in generator.__iter__(len_corpus, max_coeff, device):
#     outputs = model(**inputs)
#     loss = outputs.loss / grad_accum_steps
#     count += 1
#     if count > 5:
#         import time
#         time.sleep(300000)
#     import time
#     time.sleep(300000)


params_no_decay = ["LayerNorm.weight", "bias"]
parameters_no_decay = [
    {"params": [p for n, p in model.named_parameters() if not any(pnd in n for pnd in params_no_decay)],
     "weight_decay": weight_decay},
    {"params": [p for n, p in model.named_parameters() if any(pnd in n for pnd in params_no_decay)],
     "weight_decay": 0.0}
    ]
optimizer = AdamW(parameters_no_decay, lr=lr, betas=betas, eps=eps)
# from lion_pytorch import Lion
# optimizer = Lion(parameters_no_decay, lr=lr)
# optimizer = torch.optim.SGD(parameters_no_decay, lr=lr, momentum=0.9, dampening=0.5, nesterov=False)


# 训练轮次
times_batch_size = len_corpus // grad_accum_steps // batch_size
num_training_steps = int(times_batch_size * epochs)
evaluate_steps = int(times_batch_size*0.382) if not evaluate_steps else evaluate_steps
# 如果选择-1不设置则为 半个epoch
num_warmup_steps = int((len_corpus // grad_accum_steps // 2)) if warmup_steps == -1 else warmup_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

epochs_store = []
global_steps = 0
best_mertics = 0
best_report = ""

# import torch.nn.functional as F
# loss_ce = F.cross_entropy
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
for epochs_i in trange(epochs, desc="epoch"):  # epoch
    print("epochs: ".format(epochs_i))
    model.train()  # train-type
    pabr = tqdm(total=times_batch_size, desc="epoch_{}_step".format(epochs_i))
    for idx, (inputs, batch_qtext, batch_qans) in enumerate(generator.__iter__(len_corpus, max_coeff, device)):  # step
        pabr.update(1)
        labels = inputs.get("labels")
        inputs.pop("labels")
        outputs = model(**inputs)
        ###   loss
        lm_logits = outputs.logits
        dtype = lm_logits.dtype
        lm_logits = lm_logits.to(torch.float32)
        # labels = labels.to(torch.float32)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        logits_1 = shift_logits.view(-1, shift_logits.size(-1))
        logits_2 = shift_labels.view(-1)
        # print(logits_1.detach().cpu().numpy())
        # print(logits_2.detach().cpu().numpy())
        loss = loss_fct(logits_1, logits_2)
        # loss_2 = loss_ce(logits_1, logits_2, ignore_index=-100)
        # print(loss.detach().cpu().numpy())
        # print(loss_2.detach().cpu().numpy())
        lm_logits = lm_logits.to(dtype)
        loss = loss.to(dtype) / grad_accum_steps
        # loss = outputs.loss / grad_accum_steps
        loss.backward()
        global_steps += 1
        if idx < 5 or idx % logger_steps == 0:
            print("epoch_global: {}, step_global: {}, step: {}, loss: {}".format(epochs_i, global_steps, idx, loss))
        #  梯度累计
        if (idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if epochs_i == 0 and idx+1 >= len_corpus:
            save_model_state(model, chatglm_config, model_save_path)
        # 评估算法/打印日志/存储模型, 1个epoch/到达保存的步数/或者是最后一轮最后一步
        if (evaluate_steps > 0 and global_steps % evaluate_steps == 0) or (epochs_i > 0 and idx == 0) \
                or (epochs_i + 1 == epochs and (idx + 1) >= len_corpus):
            score_avg, score_dict = evaluate(model, tokenizer, len_corpus=batch_size, device=device)  # 验证数据个数
            print("epoch_global: {}, step_global: {}, step: {}".format(epochs_i, global_steps, idx))
            print("best_score_avg: {}\n".format(score_avg))
            print("current_mertics: {}".format(score_dict))
            model.train()
            if score_avg > best_mertics:  # 只保留最优的指标
                epochs_store.append((epochs_i, idx))
                best_mertics = score_avg
                save_model_state(model, chatglm_config, model_save_path)
            if epochs_store and epochs_i - epochs_store[-1][0] >= stop_epochs:
                break


"""
数学算式(加减乘除)微调, max_coeff=100以内
## 只训练最后一层的参数
"""

# nohup python c00_toy_gpu_train_6b.py > tc.c00_toy_gpu_train_6b.py.log 2>&1 &
# tail -n 1000  -f tc.c00_toy_gpu_train_6b.py.log
# |yongzhuo|

