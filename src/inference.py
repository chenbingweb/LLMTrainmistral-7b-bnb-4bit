from unsloth import FastLanguageModel
import os
import torch
from transformers import TextStreamer

class Inference:
    def __init__(self,model_name,load_in_4bit=True,max_seq_length=4096):
        torch.cuda.empty_cache()
        self.model_name =model_name
        self.model = None
        self.tokenizer = None
        self.load_in_4bit=load_in_4bit
        self.max_seq_length=max_seq_length
        self.dtype=None
        self.init()

    def init(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,  # 对比我们刚保存好的本地微调后模型名称
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            device_map="cuda"
        )
        self.model = model
        self.tokenizer = tokenizer
        FastLanguageModel.for_inference(model)

    def set_alpaca_prompt(self,alpaca_prompt):
        self.alpaca_prompt=alpaca_prompt

    def gennerate(self,input_text,max_new_tokens=500):
        inputs = self.tokenizer(
            [
                self.alpaca_prompt.format(
                    input_text,  # 用户指令
                    "",  # output - 留空以自动生成 / 不留空以填充
                )
            ], return_tensors="pt").to("cuda")
        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens)