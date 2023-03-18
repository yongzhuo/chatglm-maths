# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/16 21:54
# @author  : Mo
# @function:


import logging as logger
import traceback
import random
import codecs
import time
import json
import math
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)

# cpu_nums = "9"
# os.environ["OMP_NUM_THREADS"] = cpu_nums  # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = cpu_nums  # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = cpu_nums  # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = cpu_nums  # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = cpu_nums  # export NUMEXPR_NUM_THREADS=1
os.environ["USE_TORCH"] = "1"

from tqdm import tqdm
import numpy as np
import torch

from chatglm_maths.models.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from chatglm_maths.models.tokenization_chatglm import ChatGLMTokenizer


is_toy = True
if is_toy:
    CUDA_VISIBLE_DEVICES = "-1"
    use_cuda = False
    quantize_type = None  # None, 16, 8, 4
    use_half = True
    batch_size = 2
    len_corpus = 16  # batch_size*3820
    num_layers = 1   # 28
    warmup_steps = 1
    logger_steps = 2
    pretrained_model_name_or_path = "THUDM/chatglm-6b"  # None
    evaluate_steps = 3820
else:
    CUDA_VISIBLE_DEVICES = "-1"
    use_cuda = False
    quantize_type = None  # None, 16, 8, 4
    use_half = True
    batch_size = 16
    len_corpus = batch_size*3820
    num_layers = 28
    warmup_steps = 100
    logger_steps = 100
    pretrained_model_name_or_path = "THUDM/chatglm-6b"  # None
    evaluate_steps = int(len_corpus / batch_size / 3) + 1  # 3820


model_save_path = "./fine_tuning"
quantize_type = None  # None, 16, 8, 4
seed = 2023
weight_decay = 5e-4
lr = 1e-5
eps = 1e-9
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
        with codecs.open(filename=path_config, mode="w", encoding="utf-8") as fc:
            json.dump(vars(config), fc, indent=4, ensure_ascii=False)
            fc.close()
    # save model
    path_model = os.path.join(model_save_path, model_name)
    torch.save(model.state_dict(), path_model)
    logger.info("******model_save_path is {}******".format(path_model))
def load_model_state(model_save_path="./", model_name="tc.model", device="cpu"):
    """  仅加载模型参数(推荐使用)  """
    try:
        path_model = os.path.join(model_save_path, model_name)
        model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))
        model.to(device)
        logger.info("******model loaded success******")
        logger.info("self.device: {}".format(device))
    except Exception as e:
        logger.info(str(e))
        raise Exception("******load model error******")
def sequence_padding(inputs, length=None, padding=0, length_max=2048):
    """
        将序列padding到同一长度(当前inputs最长)
    config:
        config: dict, enum of parms
    Returns:
        tokenizer: class
    """
    if length is None:
        length = min(max([len(x) for x in inputs]), length_max)
    outputs = []
    for x in inputs:
        if len(x) >= length:
            x = x[:length]
        else:
            x = x + [padding] * (length - len(x))
        outputs.append(x)
    return outputs
def load_json(path: str, encoding: str="utf-8"):
    """
    Read Line of List<json> form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        model_json: dict of word2vec, eg. [{"大漠帝国":132}]
    """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json
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
                    print("batch_qtext_{}: {}".format(jdx, batch_qtext_i))
                    print("batch_qans_{}: {}".format(jdx, batch_qans_i))
                    print("response_{}: {}".format(jdx, response))
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
            batch_query = []
            batch_qtext = []
            batch_qans = []
            for _ in range(batch_size):
                x, y = self.generate_line(op="+", max_coeff=max_coeff, use_gaussian=True, dim1=1, dim2=2)
                prompts = [("问: ", "答: "), ("问题: ", "答案: "), ("计算: ", "回答: "),
                           ("计算题: ", "解答: "), ("口算: ", "解: "), ("简便运算: ", "剖析: "),
                           ("数学题: ", "点拨: "), ("初等数学: ", "解析: ")]
                prompt = random.choice(prompts)
                x_prompt = prompt[0] + x + " " + prompt[1] + " " + y
                batch_query.append(x_prompt)
                batch_qtext.append(prompt[0] + x + " " + prompt[1] + " ")
                batch_qans.append(y)
            if idx < 5:
                print("batch_query: {}".format(batch_query))
            input_ids = tokenizer(batch_query, return_tensors="pt", padding="max_length",
                                  max_length=max([len(b) for b in batch_query])+2, truncation=True).input_ids.to(device)
            labels = tokenizer(batch_query, return_tensors="pt", padding="max_length", 
                               max_length=max([len(b) for b in batch_query])+2, truncation=True).input_ids.to(device)
            inputs= {"input_ids": input_ids, "labels": labels}
            yield inputs, batch_qtext, batch_qans


## 构建算式
generator = Generator(batch_size=batch_size, float_precision=2)
generator_line = generator.generate_line()
print("generator_calculate_line: {}".format(generator_line))

chatglm_config = ChatGLMConfig.from_json_file(os.path.join(model_save_path, "tc.config"))
tokenizer = ChatGLMTokenizer.from_pretrained(pretrained_model_name_or_path)

# ### test
# chatglm_config.num_layers = num_layers
# chatglm_config.torch_dtype = "float16"

model = ChatGLMForConditionalGeneration(chatglm_config)
load_model_state(model_save_path=model_save_path)
if use_cuda:
    model = model.half().to(device)
    print("model cuda ok!")
else:
    model = model.bfloat16()
model.eval()
score_avg, score_dict = evaluate(model, tokenizer, len_corpus=batch_size, device=device)  # 验证数据个数



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
            response, history = model.chat(tokenizer=tokenizer, query=ques, history=history, max_length=2048,
                                           num_beams=1, do_sample=True, top_p=0.7, temperature=0.95)
            res_ende = str(response).encode("utf-8", "ignore").decode("utf-8", "ignore")
            print(res_ende)
    except Exception as e:
        print(str(e))
    print(time.time()-time_start)




"""
数学算式(加减乘除)微调, max_coeff=100以内

"""



