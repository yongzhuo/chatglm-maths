# chatglm-maths
chatglm-6b微调/推理, 样本为自动生成的整数/小数加减乘除运算, 可gpu/cpu
chatglm-6b fine-tuning/inference, The sample is an automatically generated, integer/decimal of add, sub, mul and div operation, that can be gpu/cpu


## 数据集-中文
 - [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
 - [https://github.com/LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
 - [https://github.com/carbonz0/alpaca-chinese-dataset](https://github.com/carbonz0/alpaca-chinese-dataset)


## 踩坑
```python
1. eps=1e-5(不要改小), 单精度float16, 以及LN采用的是Sandwich-LN(Sandwich LayerNorm), 分支的ATtention前后都有LN, 目的是大模型为了防止梯度溢出等;
2. 模型输入输出, 默认的tokenization_chatglm.py/modeling_chatglm.py不能用, 因为那是完全为生成generate设置的, 需要自己写好所有缩入参数, 或者机子改成适配的;
   2.1 ChatGLMModel中, get_masks()正常, get_position_ids()函数中‘context_length = seq.index(150004) + 1’ 改为 ‘context_length = len(seq)’;
   2.2 训练输入input_ids格式暂定为(训练后post-padding, 推理前pre-padding[tokenization_chatglm.py默认pre-padding])
              a1. x1: [CLS] + prompt_1 + " " + text_1 + " " + prompt_2 + [gMASK] + [PAD]*N(post-padding)
              a2. x2: [SOP] + " " + text_2 + [PAD]*N(post-padding)
              a.  x = x1 + x2
   2.3 训练输入label_ids格式暂定为(CrossEntropyLoss默认忽略-100不参与计算loss)  
              b.  y = [-100]*len(x) + " " + text_2 + [EOP] + [-100]*N(post-padding)
   2.4 可参考GLM-1, https://github.com/THUDM/GLM/blob/main/tasks/seq2seq/dataset.py
3. 注意chatglm-6b权重是float16的, 不过计算loss时候会转成float32计算, 最后loss再转回float16更新梯度;
4. ChatGLMTokenizer有时候会报奇奇怪怪的错误, 建议生成时候设置max_new_tokens, 最大{"max_new_tokens": 2048}; decode有时候会出现不存在id;
5. 低秩自适应LORA, RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
   尝试 transformers升级到最新, get_peft_model后再.cuda(), device_map={'':torch.cuda.current_device()}, 
```

## 环境配置
```shell
transformers>=4.26.1
cpm_kernels==1.0.11
icetk==0.0.4
torch>=1.10.1
rouge==1.0.1
nltk==3.6.6
numpy
tqdm

lion_pytorch
```

## 微调-计算题
```shell
6b
微调: python c00_toy_cpu_train_6b.py
推理: python p00_toy_cpu_predit_6b.py

small-layer
微调: python c01_toy_cpu_train_small.py
推理: python p01_toy_cpu_predict_small.py
```


## 参考/感谢
 - [https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
 - [https://github.com/THUDM/GLM](https://github.com/THUDM/GLM)
 - [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
 - [https://github.com/LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
 - [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
 - [https://github.com/mymusise/ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)
 - [https://github.com/bojone/bert4keras](https://github.com/bojone/bert4keras)


## 推理日志toy
```cpu
generator_calculate_line: ('13+75=', '13+75=88')
tokenizer.vocab_size: 150344
eval:   0%|                                                                                                                                                                      | 0/1 [00:00<?, ?it/s]batch_query: ['简便运算: 98+83= 剖析: 98+83=181']
batch_qtext_0: 简便运算: 98+83= 剖析:
batch_qans_0: 98+83=181
response_0: 98+83=171
{'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
请输入:
25.31+86.35=
请稍等...
25.31+86.35=101.66
```


## 微调日志toy
```cpu
generator_calculate_line: ('13+75=', '13+75=88')
tokenizer.vocab_size: 150344
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:10<00:00,  1.31s/it]
transformer.word_embeddings.weight False
......
transformer.layers.26.mlp.dense_4h_to_h.bias False
transformer.layers.27.input_layernorm.weight True
transformer.layers.27.input_layernorm.bias True
transformer.layers.27.attention.query_key_value.weight True
transformer.layers.27.attention.query_key_value.bias True
transformer.layers.27.attention.dense.weight True
transformer.layers.27.attention.dense.bias True
transformer.layers.27.post_attention_layernorm.weight True
transformer.layers.27.post_attention_layernorm.bias True
transformer.layers.27.mlp.dense_h_to_4h.weight True
transformer.layers.27.mlp.dense_h_to_4h.bias True
transformer.layers.27.mlp.dense_4h_to_h.weight True
transformer.layers.27.mlp.dense_4h_to_h.bias True
transformer.final_layernorm.weight True
transformer.final_layernorm.bias True
model.chat start
13+75=88, but that's not the correct answer. The correct answer is 13+75=88, which is 90.
/anaconda3/envs/py371/lib/python3.7/site-packages/transformers/optimization.py:395: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  FutureWarning,   
epoch:   0%|                                                                                                                                                                    | 0/21 [00:00<?, ?it/s]epochs:
                                                                                                                                                                                                      batch_query: ['简便运算: 98+83= 剖析: 98+83=181']                                                                                                                                 | 0/8 [00:00<?, ?it/s]
epoch_global: 0, step_global: 1, step: 0, loss: 4.0625
batch_query: ['口算: 57.84+13.64 解: 57.84+13.64=71.48']
                                                                                                                                                                                                      epoch_global: 0, step_global: 2, step: 1, loss: 2.5625███▌                                                                                                                | 2/8 [00:17<00:51,  8.54s/it]
batch_query: ['计算题: 48+1 解答: 48+1=49']
                                                                                                                                                                                                      epoch_global: 0, step_global: 3, step: 2, loss: 4.15625█████████████████████▎                                                                                             | 3/8 [00:38<01:09, 13.94s/it]
batch_query: ['计算题: 61.65+33.05 解答: 61.65+33.05=94.7']
                                                                                                                                                                                                      epoch_global: 0, step_global: 4, step: 3, loss: 2.40625████████████████████████████████████████                                                                           | 4/8 [01:01<01:09, 17.43s/it]
batch_query: ['计算: 81+75 回答: 81+75=156']
                                                                                                                                                                                                      epoch_global: 0, step_global: 5, step: 4, loss: 3.546875█████████████████████████████████████████████████████████▊                                                        | 5/8 [01:27<01:01, 20.41s/it]
epoch:   5%|███████▎                                                                                                                                                 | 1/21 [03:07<1:02:30, 187.52s/it]epochs: step: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:41<00:00, 23.15s/it]
epoch_0_step: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [03:07<00:00, 23.44s/it]
batch_query: ['问题: 99+37 答案: 99+37=136']
epoch_global: 1, step_global: 9, step: 0, loss: 3.640625                                                                                                                         | 0/8 [00:00<?, ?it/s]
                                                                                                                                                                                                      batch_query: ['问题: 26.81+55.91 答案: 26.81+55.91=82.72']                                                                                                                        | 0/1 [00:00<?, ?it/s]
batch_qtext_0: 问题: 26.81+55.91 答案:
batch_qans_0: 26.81+55.91=82.72
response_0: 26.81+55.91=83.72
{'rouge-1': 0.749999995, 'rouge-2': 0.3333333283333334, 'rouge-l': 0.749999995, 'bleu': 0.0}
epoch_global: 1, step_global: 9, step: 0
best_score_avg: 0.45833

current_mertics: {'rouge-1': 0.749999995, 'rouge-2': 0.3333333283333334, 'rouge-l': 0.749999995, 'bleu': 0.0}
batch_query: ['数学题: 23.34+68.45 点拨: 23.34+68.45=91.79']
                                                                                                                                                                                                      epoch_global: 1, step_global: 10, step: 1, loss: 2.09375
batch_query: ['计算: 77+14 回答: 77+14=91']█████████████▌                                                                                                                | 2/8 [00:33<01:39, 16.58s/it]
                                                                                                                                                                                                      epoch_global: 1, step_global: 11, step: 2, loss: 3.265625
batch_query: ['口算: 79.69+17.43= 解: 79.69+17.43=97.12']██████████████████▎                                                                                             | 3/8 [00:35<00:53, 10.75s/it]
                                                                                                                                                                                                      epoch_global: 1, step_global: 12, step: 3, loss: 2.171875
batch_query: ['简便运算: 59.67+86.73 剖析: 59.67+86.73=146.4']████████████████████████████████                                                                           | 4/8 [00:37<00:29,  7.43s/it]
                                                                                                                                                                                                      epoch_global: 1, step_global: 13, step: 4, loss: 2.328125
epoch:  10%|██████████████▊                                                                                                                                            | 2/21 [03:56<33:33, 105.97s/it]epochs:
epoch_1_step: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:48<00:00,  6.11s/it]
batch_query: ['初等数学: 24.29+76.26 解析: 24.29+76.26=100.55']
epoch_global: 2, step_global: 17, step: 0, loss: 2.046875
epoch_2_step:   0%|                                                                                                                                                              | 0/8 [00:00<?, ?it/sbatch_query: ['计算题: 69.85+28.46= 解答: 69.85+28.46=98.31']
batch_qtext_0: 计算题: 69.85+28.46= 解答:                                                                                                                                        | 0/1 [00:00<?, ?it/s]
batch_qans_0: 69.85+28.46=98.31
response_0: 69.85+28.46=97.21
{'rouge-1': 0.4999999950000001, 'rouge-2': 0.3333333283333334, 'rouge-l': 0.4999999950000001, 'bleu': 0.0}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:07<00:00,  7.83s/it]
epoch_global: 2, step_global: 17, step: 0
best_score_avg: 0.33333

current_mertics: {'rouge-1': 0.4999999950000001, 'rouge-2': 0.3333333283333334, 'rouge-l': 0.4999999950000001, 'bleu': 0.0}
batch_query: ['问题: 113.79+81.78= 答案: 113.79+81.78=195.57']
                                                                                                                                                                                                      epoch_global: 2, step_global: 18, step: 1, loss: 1.8515625
batch_query: ['计算: 10.74+17.87= 回答: 10.74+17.87=28.61']
epoch_2_step:  25%|█████████████████████████████████████▌                                                                                                                | 2/8 [00:10<00:31,  5.21s/itepoch_global: 2, step_global: 19, step: 2, loss: 1.8203125
batch_query: ['计算: 11.64+25.07= 回答: 11.64+25.07=36.71']
epoch_2_step:  38%|████████████████████████████████████████████████████████▎                                                                                             | 3/8 [00:13<00:20,  4.15s/itepoch_global: 2, step_global: 20, step: 3, loss: 1.859375
batch_query: ['口算: 53.08+54.9 解: 53.08+54.9=107.98']
epoch_2_step:  50%|███████████████████████████████████████████████████████████████████████████                                                                           | 4/8 [00:15<00:14,  3.58s/itepoch_global: 2, step_global: 21, step: 4, loss: 2.078125
epoch:  14%|██████████████████████▎                                                                                                                                     | 3/21 [04:23<20:56, 69.80s/it]epochs: step:  62%|█████████████████████████████████████████████████████████████████████████████████████████████▊                                                        | 5/8 [00:18<00:09,  3.28s/it]
epoch_2_step: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:26<00:00,  3.34s/it]
batch_query: ['初等数学: 102.7+68.21= 解析: 102.7+68.21=170.91']█████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:24<00:00,  2.44s/it]
epoch_global: 3, step_global: 25, step: 0, loss: 1.5390625                                                                                                                       | 0/8 [00:00<?, ?it/s]
                                                                                                                                                                                                      batch_query: ['数学题: 94+19 点拨: 94+19=113']
batch_qtext_0: 数学题: 94+19 点拨:
batch_qans_0: 94+19=113                                                                                                                                                          | 0/1 [00:00<?, ?it/s]
response_0: 94+19=103
{'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.11s/it]
epoch_global: 3, step_global: 25, step: 0
best_score_avg: 0.0

current_mertics: {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
batch_query: ['数学题: 37.94+23.99 点拨: 37.94+23.99=61.93']
                                                                                                                                                                                                      epoch_global: 3, step_global: 26, step: 1, loss: 1.578125
batch_query: ['问: 51.16+14.21= 答: 51.16+14.21=65.37']█▌                                                                                                                | 2/8 [00:06<00:20,  3.41s/it]
                                                                                                                                                                                                      epoch_global: 3, step_global: 27, step: 2, loss: 1.7265625
batch_query: ['问题: 13.89+40.09 答案: 13.89+40.09=53.98']█████████████████▎                                                                                             | 3/8 [00:09<00:15,  3.08s/it]
                                                                                                                                                                                                      epoch_global: 3, step_global: 28, step: 3, loss: 1.9765625
batch_query: ['口算: 68+33 解: 68+33=101']████████████████████████████████████████████████████                                                                           | 4/8 [00:11<00:11,  2.83s/it]
                                                                                                                                                                                                      epoch_global: 3, step_global: 29, step: 4, loss: 3.125
epoch:  19%|█████████████████████████████▋                                                                                                                              | 4/21 [04:45<14:29, 51.16s/it]epochs:
epoch_3_step: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:22<00:00,  2.82s/it]
batch_query: ['简便运算: 52+48 剖析: 52+48=100']
epoch_global: 4, step_global: 33, step: 0, loss: 2.921875
epoch_4_step:   0%|                                                                                                                                                              | 0/8 [00:00<?, ?it/sbatch_query: ['口算: 11.71+0.36= 解: 11.71+0.36=12.07']
batch_qtext_0: 口算: 11.71+0.36= 解:                                                                                                                                             | 0/1 [00:00<?, ?it/s]
batch_qans_0: 11.71+0.36=12.07
response_0: 11.71+0.36=12.07
{'rouge-1': 0.999999995, 'rouge-2': 0.999999995, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.65s/it]
epoch_global: 4, step_global: 33, step: 0
best_score_avg: 0.79446

current_mertics: {'rouge-1': 0.999999995, 'rouge-2': 0.999999995, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
batch_query: ['计算题: 40.29+76.09 解答: 40.29+76.09=116.38']
                                                                                                                                                                                                      epoch_global: 4, step_global: 34, step: 1, loss: 1.40625
batch_query: ['计算: 64+26= 回答: 64+26=90']
epoch_4_step:  25%|█████████████████████████████████████▌                                                                                                                | 2/8 [00:27<01:23, 13.96s/itepoch_global: 4, step_global: 35, step: 2, loss: 2.328125
batch_query: ['问题: 48.54+9.56 答案: 48.54+9.56=58.1']
epoch_4_step:  38%|████████████████████████████████████████████████████████▎                                                                                             | 3/8 [00:30<00:46,  9.32s/itepoch_global: 4, step_global: 36, step: 3, loss: 1.90625
batch_query: ['计算题: 119.42+26.14 解答: 119.42+26.14=145.56']
epoch_4_step:  50%|███████████████████████████████████████████████████████████████████████████                                                                           | 4/8 [00:32<00:26,  6.54s/itepoch_global: 4, step_global: 37, step: 4, loss: 1.5859375
epoch:  24%|█████████████████████████████████████▏                                                                                                                      | 5/21 [05:29<12:57, 48.61s/it]epochs: step:  62%|█████████████████████████████████████████████████████████████████████████████████████████████▊                                                        | 5/8 [00:34<00:15,  5.06s/it]
epoch_4_step: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:44<00:00,  5.51s/it]
batch_query: ['计算题: 72+55 解答: 72+55=127']███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:42<00:00,  3.30s/it]
epoch_global: 5, step_global: 41, step: 0, loss: 2.21875                                                                                                                         | 0/8 [00:00<?, ?it/s]
                                                                                                                                                                                                      batch_query: ['计算: 54.37+23.56= 回答: 54.37+23.56=77.93']
batch_qtext_0: 计算: 54.37+23.56= 回答:
batch_qans_0: 54.37+23.56=77.93                                                                                                                                                  | 0/1 [00:00<?, ?it/s]
response_0: 54.37+23.56=87.03
{'rouge-1': 0.4999999950000001, 'rouge-2': 0.3333333283333334, 'rouge-l': 0.4999999950000001, 'bleu': 0.0}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.04s/it]
epoch_global: 5, step_global: 41, step: 0
best_score_avg: 0.33333

current_mertics: {'rouge-1': 0.4999999950000001, 'rouge-2': 0.3333333283333334, 'rouge-l': 0.4999999950000001, 'bleu': 0.0}
batch_query: ['初等数学: 11.66+124.17 解析: 11.66+124.17=135.83']
                                                                                                                                                                                                      epoch_global: 5, step_global: 42, step: 1, loss: 1.140625
batch_query: ['简便运算: 32.31+93.5= 剖析: 32.31+93.5=125.81']                                                                                                           | 2/8 [00:07<00:23,  3.97s/it]
                                                                                                                                                                                                      epoch_global: 5, step_global: 43, step: 2, loss: 2.03125
batch_query: ['计算题: 10+40 解答: 10+40=50']██████████████████████████████▎                                                                                             | 3/8 [00:10<00:17,  3.41s/it]
                                                                                                                                                                                                      epoch_global: 5, step_global: 44, step: 3, loss: 2.28125
batch_query: ['数学题: 26.19+58.61 点拨: 26.19+58.61=84.8']███████████████████████████████████                                                                           | 4/8 [00:13<00:12,  3.09s/it]
                                                                                                                                                                                                      epoch_global: 5, step_global: 45, step: 4, loss: 1.515625
epoch:  29%|████████████████████████████████████████████▌                                                                                                               | 6/21 [05:54<10:07, 40.50s/it]epochs:
epoch_5_step: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:24<00:00,  3.09s/it]
batch_query: ['简便运算: 83.94+43.41= 剖析: 83.94+43.41=127.35']
epoch_global: 6, step_global: 49, step: 0, loss: 1.6640625
epoch_6_step:   0%|                                                                                                                                                              | 0/8 [00:00<?, ?it/sbatch_query: ['问: 10+17= 答: 10+17=27']
batch_qtext_0: 问: 10+17= 答:                                                                                                                                                    | 0/1 [00:00<?, ?it/s]
batch_qans_0: 10+17=27
response_0: 10+17=27
{'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.13s/it]
epoch_global: 6, step_global: 49, step: 0
best_score_avg: 0.54446

current_mertics: {'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
batch_query: ['数学题: 77.73+51.97= 点拨: 77.73+51.97=129.7']
                                                                                                                                                                                                      epoch_global: 6, step_global: 50, step: 1, loss: 1.3671875
batch_query: ['口算: 57+56= 解: 57+56=113']
epoch_6_step:  25%|█████████████████████████████████████▌                                                                                                                | 2/8 [00:05<00:17,  2.86s/itepoch_global: 6, step_global: 51, step: 2, loss: 3.125
batch_query: ['问题: 59+24 答案: 59+24=83']
epoch_6_step:  38%|████████████████████████████████████████████████████████▎                                                                                             | 3/8 [00:08<00:13,  2.70s/itepoch_global: 6, step_global: 52, step: 3, loss: 2.671875
batch_query: ['计算题: 73+64 解答: 73+64=137']
epoch_6_step:  50%|███████████████████████████████████████████████████████████████████████████                                                                           | 4/8 [00:09<00:09,  2.36s/itepoch_global: 6, step_global: 53, step: 4, loss: 1.90625
epoch:  33%|████████████████████████████████████████████████████                                                                                                        | 7/21 [06:13<07:49, 33.51s/it]epochs: step:  62%|█████████████████████████████████████████████████████████████████████████████████████████████▊                                                        | 5/8 [00:11<00:06,  2.14s/it]
epoch_6_step: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:19<00:00,  2.39s/it]
batch_query: ['问题: 3+79 答案: 3+79=82']████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:17<00:00,  1.96s/it]
epoch_global: 7, step_global: 57, step: 0, loss: 3.328125                                                                                                                        | 0/8 [00:00<?, ?it/s]
                                                                                                                                                                                                      batch_query: ['计算题: 21.6+4.99 解答: 21.6+4.99=26.59']
batch_qtext_0: 计算题: 21.6+4.99 解答:
batch_qans_0: 21.6+4.99=26.59                                                                                                                                                    | 0/1 [00:00<?, ?it/s]
response_0: 21.6+4.99=26.67
{'rouge-1': 0.749999995, 'rouge-2': 0.6666666616666668, 'rouge-l': 0.749999995, 'bleu': 0.0}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.84s/it]
epoch_global: 7, step_global: 57, step: 0
best_score_avg: 0.54167

current_mertics: {'rouge-1': 0.749999995, 'rouge-2': 0.6666666616666668, 'rouge-l': 0.749999995, 'bleu': 0.0}
epoch:  38%|███████████████████████████████████████████████████████████▍                                                                                                | 8/21 [06:21<05:28, 25.27s/it]epochs:
epoch_7_step:  12%|██████████████████▊                                                                                                                                   | 1/8 [00:07<00:53,  7.62s/it]
batch_query: ['简便运算: 32.25+31.24= 剖析: 32.25+31.24=63.49']
epoch_global: 8, step_global: 58, step: 0, loss: 1.640625
epoch_8_step:   0%|                                                                                                                                                              | 0/8 [00:00<?, ?it/sbatch_query: ['简便运算: 4+18 剖析: 4+18=22']
batch_qtext_0: 简便运算: 4+18 剖析:                                                                                                                                              | 0/1 [00:00<?, ?it/s]
batch_qans_0: 4+18=22
response_0: 4+18=22
{'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.43s/it]
epoch_global: 8, step_global: 58, step: 0
best_score_avg: 0.54446

current_mertics: {'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
epoch:  43%|██████████████████████████████████████████████████████████████████▊                                                                                         | 9/21 [06:27<03:51, 19.26s/it]epochs:
epoch_8_step:  12%|██████████████████▊                                                                                                                                   | 1/8 [00:06<00:42,  6.05s/it]
batch_query: ['口算: 56.12+87.86= 解: 56.12+87.86=143.98']
epoch_global: 9, step_global: 59, step: 0, loss: 1.65625                                                                                                                         | 0/8 [00:00<?, ?it/s]
                                                                                                                                                                                                      batch_query: ['问: 84.48+26.75= 答: 84.48+26.75=111.23']
batch_qtext_0: 问: 84.48+26.75= 答:
batch_qans_0: 84.48+26.75=111.23                                                                                                                                                 | 0/1 [00:00<?, ?it/s]
response_0: 84.48+26.75=101.13
{'rouge-1': 0.4999999950000001, 'rouge-2': 0.3333333283333334, 'rouge-l': 0.4999999950000001, 'bleu': 0.0}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  6.06s/it]
epoch_global: 9, step_global: 59, step: 0
best_score_avg: 0.33333

current_mertics: {'rouge-1': 0.4999999950000001, 'rouge-2': 0.3333333283333334, 'rouge-l': 0.4999999950000001, 'bleu': 0.0}
epoch:  48%|█████████████████████████████████████████████████████████████████████████▊                                                                                 | 10/21 [06:35<02:55, 15.95s/it]epochs:
epoch_9_step:  12%|██████████████████▊                                                                                                                                   | 1/8 [00:08<00:59,  8.55s/it]
batch_query: ['计算: 76.94+92.36= 回答: 76.94+92.36=169.3']
epoch_global: 10, step_global: 60, step: 0, loss: 1.7421875
epoch_10_step:   0%|                                                                                                                                                             | 0/8 [00:00<?, ?it/sbatch_query: ['初等数学: 91+38= 解析: 91+38=129']
batch_qtext_0: 初等数学: 91+38= 解析:                                                                                                                                            | 0/1 [00:00<?, ?it/s]
batch_qans_0: 91+38=129
response_0: 91+38=129
{'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.50s/it]
epoch_global: 10, step_global: 60, step: 0
best_score_avg: 0.54446

current_mertics: {'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
epoch:  52%|█████████████████████████████████████████████████████████████████████████████████▏                                                                         | 11/21 [06:41<02:08, 12.89s/it]epochs:
epoch_10_step:  12%|██████████████████▋                                                                                                                                  | 1/8 [00:05<00:41,  5.93s/it]
batch_query: ['问题: 23.29+19.33 答案: 23.29+19.33=42.62']
epoch_global: 11, step_global: 61, step: 0, loss: 1.921875                                                                                                                       | 0/8 [00:00<?, ?it/s]
                                                                                                                                                                                                      batch_query: ['问: 62+93 答: 62+93=155']
batch_qtext_0: 问: 62+93 答:
batch_qans_0: 62+93=155                                                                                                                                                          | 0/1 [00:00<?, ?it/s]
response_0: 62+93=155
{'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.33s/it]
epoch_global: 11, step_global: 61, step: 0
best_score_avg: 0.54446

current_mertics: {'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
epoch:  57%|████████████████████████████████████████████████████████████████████████████████████████▌                                                                  | 12/21 [06:47<01:36, 10.70s/it]epochs:
epoch_11_step:  12%|██████████████████▋                                                                                                                                  | 1/8 [00:05<00:39,  5.70s/it]
batch_query: ['口算: 22.22+37.01 解: 22.22+37.01=59.23']
epoch_global: 12, step_global: 62, step: 0, loss: 1.7109375
epoch_12_step:   0%|                                                                                                                                                             | 0/8 [00:00<?, ?it/sbatch_query: ['口算: 7+24= 解: 7+24=31']
batch_qtext_0: 口算: 7+24= 解:                                                                                                                                                   | 0/1 [00:00<?, ?it/s]
batch_qans_0: 7+24=31
response_0: 7+24=29
{'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.83s/it]
epoch_global: 12, step_global: 62, step: 0
best_score_avg: 0.0

current_mertics: {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
epoch:  62%|███████████████████████████████████████████████████████████████████████████████████████████████▉                                                           | 13/21 [06:52<01:12,  9.03s/it]epochs:
epoch_12_step:  12%|██████████████████▋                                                                                                                                  | 1/8 [00:05<00:36,  5.17s/it]
batch_query: ['计算题: 48+5= 解答: 48+5=53']
epoch_global: 13, step_global: 63, step: 0, loss: 2.15625                                                                                                                        | 0/8 [00:00<?, ?it/s]
                                                                                                                                                                                                      batch_query: ['简便运算: 5+68= 剖析: 5+68=73']
batch_qtext_0: 简便运算: 5+68= 剖析:
batch_qans_0: 5+68=73                                                                                                                                                            | 0/1 [00:00<?, ?it/s]
response_0: 要简化这个算式,我们可以使用分配律,即:a+b=b+a。因此,5+68=68+5=73。

我们也可以使用长除法,将68除以5,
{'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:41<00:00, 41.89s/it]
epoch_global: 13, step_global: 63, step: 0
best_score_avg: 0.0

current_mertics: {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
epoch:  67%|███████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                   | 14/21 [07:36<02:16, 19.48s/it]epochs:
epoch_13_step:  12%|██████████████████▋                                                                                                                                  | 1/8 [00:43<05:05, 43.65s/it]
batch_query: ['计算题: 25.31+86.35 解答: 25.31+86.35=111.66']
epoch_global: 14, step_global: 64, step: 0, loss: 1.1796875
epoch_14_step:   0%|                                                                                                                                                             | 0/8 [00:00<?, ?it/sbatch_query: ['口算: 4+44= 解: 4+44=48']
batch_qtext_0: 口算: 4+44= 解:                                                                                                                                                   | 0/1 [00:00<?, ?it/s]
batch_qans_0: 4+44=48
response_0: 4+44=48
{'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.79s/it]
epoch_global: 14, step_global: 64, step: 0
best_score_avg: 0.54446

current_mertics: {'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
epoch:  71%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                            | 15/21 [07:41<01:31, 15.20s/it]epochs:
epoch_14_step:  12%|██████████████████▋                                                                                                                                  | 1/8 [00:05<00:36,  5.28s/it]
batch_query: ['计算题: 53+79= 解答: 53+79=132']
epoch_global: 15, step_global: 65, step: 0, loss: 1.9453125                                                                                                                      | 0/8 [00:00<?, ?it/s]
                                                                                                                                                                                                      batch_query: ['简便运算: 69+85 剖析: 69+85=154']
batch_qtext_0: 简便运算: 69+85 剖析:
batch_qans_0: 69+85=154                                                                                                                                                          | 0/1 [00:00<?, ?it/s]
response_0: 要简便运算,我们可以采用因数分解和通分的方法。

首先,将两个数进行因数分解:

69=3×33
85=5×53

然后,将两个
{'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:53<00:00, 53.75s/it]
epoch_global: 15, step_global: 65, step: 0
best_score_avg: 0.0

current_mertics: {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
epoch:  76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                     | 16/21 [08:37<02:16, 27.36s/it]epochs:
epoch_15_step:  12%|██████████████████▋                                                                                                                                  | 1/8 [00:55<06:29, 55.59s/it]
batch_query: ['问题: 11+28 答案: 11+28=39']
epoch_global: 16, step_global: 66, step: 0, loss: 2.609375
epoch_16_step:   0%|                                                                                                                                                             | 0/8 [00:00<?, ?it/sbatch_query: ['问: 5.65+10.67 答: 5.65+10.67=16.32']
batch_qtext_0: 问: 5.65+10.67 答:                                                                                                                                                | 0/1 [00:00<?, ?it/s]
batch_qans_0: 5.65+10.67=16.32
response_0: 5.65+10.67=16.32
{'rouge-1': 0.999999995, 'rouge-2': 0.999999995, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.89s/it]
epoch_global: 16, step_global: 66, step: 0
best_score_avg: 0.79446

current_mertics: {'rouge-1': 0.999999995, 'rouge-2': 0.999999995, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
epoch:  81%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                             | 17/21 [08:43<01:24, 21.15s/it]epochs:
epoch_16_step:  12%|██████████████████▋                                                                                                                                  | 1/8 [00:06<00:47,  6.72s/it]
batch_query: ['口算: 20+7= 解: 20+7=27']
epoch_global: 17, step_global: 67, step: 0, loss: 3.328125                                                                                                                       | 0/8 [00:00<?, ?it/s]
                                                                                                                                                                                                      batch_query: ['问题: 92.68+38.52= 答案: 92.68+38.52=131.2']
batch_qtext_0: 问题: 92.68+38.52= 答案:
batch_qans_0: 92.68+38.52=131.2                                                                                                                                                  | 0/1 [00:00<?, ?it/s]
response_0: 92.68+38.52=131.20
{'rouge-1': 0.749999995, 'rouge-2': 0.6666666616666668, 'rouge-l': 0.749999995, 'bleu': 0.0}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.39s/it]
epoch_global: 17, step_global: 67, step: 0
best_score_avg: 0.54167

current_mertics: {'rouge-1': 0.749999995, 'rouge-2': 0.6666666616666668, 'rouge-l': 0.749999995, 'bleu': 0.0}
epoch:  86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                      | 18/21 [08:50<00:50, 16.90s/it]epochs:
epoch_17_step:  12%|██████████████████▋                                                                                                                                  | 1/8 [00:06<00:48,  6.99s/it]
batch_query: ['简便运算: 54.87+42.4= 剖析: 54.87+42.4=97.27']
epoch_global: 18, step_global: 68, step: 0, loss: 1.7734375
epoch_18_step:   0%|                                                                                                                                                             | 0/8 [00:00<?, ?it/sbatch_query: ['计算题: 58.63+36.22= 解答: 58.63+36.22=94.85']
batch_qtext_0: 计算题: 58.63+36.22= 解答:                                                                                                                                        | 0/1 [00:00<?, ?it/s]
batch_qans_0: 58.63+36.22=94.85
response_0: 58.63+36.22=114.85
{'rouge-1': 0.749999995, 'rouge-2': 0.3333333283333334, 'rouge-l': 0.749999995, 'bleu': 0.0}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.30s/it]
epoch_global: 18, step_global: 68, step: 0
best_score_avg: 0.45833

current_mertics: {'rouge-1': 0.749999995, 'rouge-2': 0.3333333283333334, 'rouge-l': 0.749999995, 'bleu': 0.0}
epoch:  90%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏              | 19/21 [08:58<00:28, 14.15s/it]epochs:
epoch_18_step:  12%|██████████████████▋                                                                                                                                  | 1/8 [00:07<00:54,  7.76s/it]
batch_query: ['初等数学: 61.35+15.18 解析: 61.35+15.18=76.53']
epoch_global: 19, step_global: 69, step: 0, loss: 1.6953125                                                                                                                      | 0/8 [00:00<?, ?it/s]
                                                                                                                                                                                                      batch_query: ['简便运算: 65+92= 剖析: 65+92=157']
batch_qtext_0: 简便运算: 65+92= 剖析:
batch_qans_0: 65+92=157                                                                                                                                                          | 0/1 [00:00<?, ?it/s]
response_0: 要简便运算,我们需要知道将要计算的数的位数和个位上的数。在这个例子中,我们已经知道了个位上的数是9,我们需要将这个数转换为十进制
{'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:26<00:00, 26.37s/it]
epoch_global: 19, step_global: 69, step: 0
best_score_avg: 0.0

current_mertics: {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu': 0.0}
epoch:  95%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌       | 20/21 [09:27<00:18, 18.57s/it]epochs:
epoch_19_step:  12%|██████████████████▋                                                                                                                                  | 1/8 [00:28<03:22, 28.86s/it]
batch_query: ['问: 86.6+44.32= 答: 86.6+44.32=130.92']
epoch_global: 20, step_global: 70, step: 0, loss: 1.5546875
epoch_20_step:   0%|                                                                                                                                                             | 0/8 [00:00<?, ?it/sbatch_query: ['计算: 87+12= 回答: 87+12=99']
batch_qtext_0: 计算: 87+12= 回答:                                                                                                                                                | 0/1 [00:00<?, ?it/s]
batch_qans_0: 87+12=99
response_0: 87+12=99
{'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.82s/it]
epoch_global: 20, step_global: 70, step: 0
best_score_avg: 0.54446

current_mertics: {'rouge-1': 0.999999995, 'rouge-2': 0.0, 'rouge-l': 0.999999995, 'bleu': 0.1778279410038923}
epoch: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [09:32<00:00, 27.27s/it]
eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [06:22<00:00, 382.93s/it]
epoch_20_step:  12%|██████████████████▋                                                                                                                                  | 1/8 [00:06<00:45,  6.45s/it]

```


