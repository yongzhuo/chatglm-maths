# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/16 21:54
# @author  : Mo
# @function: toy train with small config and cpu


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


is_toy = True
if is_toy:
    CUDA_VISIBLE_DEVICES = "0"
    use_cuda = True
    quantize_type = None  # None, 16, 8, 4
    use_half = True
    use_resume = False
    batch_size = 2
    len_corpus = 8  # batch_size*3820
    num_layers = 1  # 28
    warmup_steps = 1
    logger_steps = 2
    pretrained_model_name_or_path = "THUDM/chatglm-6b"  # None
    evaluate_steps = 3820
else:
    CUDA_VISIBLE_DEVICES = "0"
    use_cuda = True
    quantize_type = None  # None, 16, 8, 4
    use_half = True
    use_resume = False
    batch_size = 16
    len_corpus = batch_size * 3820
    num_layers = 28
    warmup_steps = 100
    logger_steps = 100
    pretrained_model_name_or_path = "THUDM/chatglm-6b"  # None
    evaluate_steps = int(len_corpus / batch_size / 3) + 1  # 3820


model_save_path = "./fine_tuning"
quantize_type = None  # None, 16, 8, 4
seed = 2023
weight_decay = 5e-4
lr = 2e-5
eps = 1e-5
betas = (0.9, 0.999)
grad_accum_steps = 1
stop_epochs = 3
epochs = 21
max_grad_norm = 5
float_precision = 2
max_coeff = 5  # 数据在 -max_coeff 到 max_coeff 之间
device = "cuda:{}".format(CUDA_VISIBLE_DEVICES) if (torch.cuda.is_available() \
                                                    and use_cuda and CUDA_VISIBLE_DEVICES != "-1") else "cpu"


def save_model_state(model, config=None, model_save_dir="./", model_name="tc.model", config_name="tc.config"):
    """  仅保存模型参数(推荐使用)  """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # save config
    if config:
        path_config = os.path.join(model_save_dir, config_name)
        config.to_json_file(path_config)
    # save model
    path_model = os.path.join(model_save_path, model_name)
    torch.save(model.state_dict(), path_model)
    logger.info("******model_save_path is {}******".format(path_model))


def load_model_state(path_dir="", model_name="tc.model", device="cpu", model_save_path="./"):
    """  仅加载模型参数(推荐使用)  """
    try:
        if path_dir:
            path_model = path_dir
        else:
            path_model = os.path.join(model_save_path, model_name)
        model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))
        model.to(device)
        logger.info("******model loaded success******")
        logger.info("self.device: {}".format(device))
    except Exception as e:
        logger.info(str(e))
        raise Exception("******load model error******")


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    code from https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


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
    pabr = tqdm(total=int(len_corpus//batch_size)+1, desc="eval")
    ans_true = []
    ans_pred = []
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    for idx, (inputs, batch_qtext, batch_qans) in enumerate(generator.__iter__(len_corpus, max_coeff, device)):  # step
        pabr.update(1)
        for jdx, batch_qtext_i in enumerate(batch_qtext):
            try:
                response, history = model.chat(tokenizer=tokenizer, query=batch_qtext_i, max_length=2048,
                                               num_beams=1, do_sample=True, top_p=0.7, temperature=0.95)
                batch_qans_i = batch_qans[jdx]
                ans_true.append(batch_qans_i)
                ans_pred.append(response)
                scores = rouge.get_scores(hyps=response, refs=batch_qans_i)
                rouge_1 += scores[0]["rouge-1"]["f"]
                rouge_2 += scores[0]["rouge-2"]["f"]
                rouge_l += scores[0]["rouge-l"]["f"]
                bleu += sentence_bleu(references=[batch_qans_i.split(" ")],
                                      hypothesis=response.split(" "),
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
    score_avg = round(sum(list(score_dict.values())) / len(score_dict.keys()), 5)
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
            batch_ids = []
            batch_qtext = []
            batch_qans = []
            for _ in range(batch_size):
                x, y = self.generate_line(op="+", max_coeff=max_coeff, use_gaussian=True, dim1=1, dim2=2)
                prompts = [("问: ", "答: "), ("问题: ", "答案: "), ("计算: ", "回答: "),
                           ("计算题: ", "解答: "), ("口算: ", "解: "), ("简便运算: ", "剖析: "),
                           ("数学题: ", "点拨: "), ("初等数学: ", "解析: ")]
                use_pormpts = True
                if use_pormpts:
                    prompt = random.choice(prompts)
                    x = prompt[0] + x + " " + prompt[1] + " "
                # x_prompt = prompt[0] + x + " " + prompt[1] + " \n [gMASK]" + y
                # batch_qtext.append(prompt[0] + x + " " + prompt[1] + " \n [gMASK]")
                x_encode = tokenizer.encode(x)
                y_encode = tokenizer.encode(y)
                # x [MASK]/[gMASK] y [eop]
                input_ids = x_encode[:-1] + y_encode[:-2] + [y_encode[-1]]
                batch_ids.append(input_ids)
                batch_qtext.append(x)
                batch_qans.append(y)
            if idx < 5:
                print("batch_query_0: {}".format(batch_ids[0]))
            """left-padding
            RuntimeError: The size of tensor a (40) must match the size of tensor b (27) at non-singleton dimension 0
            """
            batch_ids = sequence_padding(np.array(batch_ids), value=tokenizer.pad_token_id, mode="pre")
            batch_yds = sequence_padding(np.array(copy.deepcopy(batch_ids)), value=-100,
                                         mode="pre")  # ignore padding(loss-CE)
            inputs = {"input_ids": torch.tensor(batch_ids).long().to(device),
                      "labels": torch.tensor(batch_yds).long().to(device)}
            yield inputs, batch_qtext, batch_qans


set_random_seed(seed)
## 构建算式
generator = Generator(batch_size=batch_size, float_precision=2)
generator_line = generator.generate_line()
print("generator_calculate_line: {}".format(generator_line))

# The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
chatglm_config = ChatGLMConfig.from_pretrained(pretrained_model_name_or_path)
tokenizer = ChatGLMTokenizer.from_pretrained(pretrained_model_name_or_path)
print("tokenizer.vocab_size: {}".format(tokenizer.vocab_size))
### test
chatglm_config.num_layers = num_layers
chatglm_config.torch_dtype = "float16"

## 字典embedding也很大, 2w图像token, 8w英文token, 5w中文字词token(尝试剔除图片+英文(只保留计算+单词)---待实验?)
model = ChatGLMForConditionalGeneration(chatglm_config)
# layer_not_freeze = "0"
# for k, v in model.named_parameters():
#     if "lm_head" in k or "final_layernorm" in k:
#         v.requires_grad = True
#     elif layer_not_freeze and "layers.{}.".format(layer_not_freeze) in k:
#         v.requires_grad = True
#     else:
#         v.requires_grad = False
# for k, v in model.named_parameters():
#     print(k, v.requires_grad)

if os.path.exists(model_save_path) and use_resume:
    print("model load_model_state start!")
    load_model_state(model_save_path=model_save_path)
    print("model load_model_state end!")
## todo
# elif use_pretrain:
#     ee = 0
else:
    print("model _init ok!")
    model.apply(model._init_weights)
    print("model _init_weights ok!")
if use_cuda:
    model = model.half().to(device)
    print("model cuda ok!")
else:
    model = model.bfloat16()

print("model.chat start")
response, history = model.chat(tokenizer, generator_line, history=[])
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
evaluate_steps = int(times_batch_size * 0.382) if not evaluate_steps else evaluate_steps
# 如果选择-1不设置则为 半个epoch
num_warmup_steps = int((len_corpus // grad_accum_steps // 2)) if warmup_steps == -1 else warmup_steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_training_steps)

epochs_store = []
global_steps = 0
best_mertics = 0
best_report = ""
for epochs_i in trange(epochs, desc="epoch"):  # epoch
    print("epochs: ".format(epochs_i))
    model.train()  # train-type
    pabr = tqdm(total=times_batch_size, desc="epoch_{}_step".format(epochs_i))
    for idx, (inputs, batch_qtext, batch_qans) in enumerate(generator.__iter__(len_corpus, max_coeff, device)):  # step
        pabr.update(1)
        outputs = model(**inputs)
        loss = outputs.loss / grad_accum_steps
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

        if epochs_i == 0 and (idx + 1) * times_batch_size >= len_corpus - times_batch_size:
            save_model_state(model, chatglm_config, model_save_path)
        # 评估算法/打印日志/存储模型, 1个epoch/到达保存的步数/或者是最后一轮最后一步
        if (evaluate_steps > 0 and global_steps % evaluate_steps == 0) or (epochs_i > 0 and idx == 0) \
                or (epochs_i + 1 == epochs and (idx + 1) * times_batch_size >= len_corpus - times_batch_size):
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
数学算式(加减乘除)微调, max_coeff=10以内

AdamW
SGD
"""



