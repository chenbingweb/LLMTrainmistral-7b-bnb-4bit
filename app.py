from src.myTrain import Tranner
mid = 'unsloth/mistral-7b-instruct-v0.2-bnb-4bit'
dataSrc='/root/autodl-tmp/DISC-Law-SFT-Pair.jsonl'
saveModel='/root/autodl-tmp/mistral-7b-instruct-v0.2-bnb-4bit-law-v1.0'

import subprocess
import os
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

my_trainer = Tranner(mid,TUNE_MODEL_NAME='mistral-7b-instruct-v0.2-bnb-4bit-law',TUNE_MODEL_VERSION='v1.0')
my_trainer.loadModel()
from datasets import load_dataset
dataset = load_dataset('json',data_files=dataSrc, split="train")  # 从 Hugging Face Hub 加载数据集

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ### Input:
        {}

        ### Response:
        {}"""
def formatting_prompts_func(examples):
    # instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = alpaca_prompt.format(input, output)+my_trainer.tokenizer.eos_token    # 必须添加 EOS_TOKEN 这个特殊符号，否则生成会无限循环。。
        texts.append(text)
    return {"text": texts, }
train_dataset,eval_dataset = my_trainer.load_data(dataset,alpaca_prompt,formatting_prompts_func)
my_trainer.createTrainner(train_dataset,eval_dataset)
my_trainer.startTrainAndSave(saveModel)

