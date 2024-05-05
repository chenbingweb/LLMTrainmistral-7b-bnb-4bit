import os
import torch
import numpy as np
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import wandb

torch.cuda.empty_cache()

class Tranner:
    def __init__(self,model_name,datasetSrc,max_seq_length=1024,load_in_4bit=True,TUNE_MODEL_NAME='myModel',TUNE_MODEL_VERSION='v1.0'):
        self.model_name=model_name
        self.max_seq_length=max_seq_length
        self.dtype=None
        self.load_in_4bit = load_in_4bit # 以4bit加载模型
        self.model=None
        self.tokenizer=None
        self.TUNE_MODEL_NAME=TUNE_MODEL_NAME
        self.TUNE_MODEL_VERSION=TUNE_MODEL_VERSION
        self.trainer=None
        self.datasetSrc=datasetSrc
    def loadModel(self):
        print('loading',self.model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,  # 以4bit加载模型
            attn_implementation="flash_attention_2"
        )
        self.model=model
        self.tokenizer=tokenizer
        print('loading finish',self.model_name)
    def get_peft_model(self,model):
        self.model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,  # Rank stabilized LoRA
            loftq_config=None,  # LoftQ
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-3-8b
        )

    def load_data(self,src):
        # Alpaca prompt
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            texts = []
            for instruction, input, output in zip(instructions, inputs, outputs):
                text = alpaca_prompt.format(instruction, input,
                                            output) + tokenizer.eos_token  # 必须添加 EOS_TOKEN 这个特殊符号，否则生成会无限循环。。
                texts.append(text)
            return {"text": texts, }

        # 加载数据集
        dataset = load_dataset(src, split="train")  # 从 Hugging Face Hub 加载数据集
        # random_indices = np.random.choice(len(dataset), 8000, replace=False)  # 随机选择500个样本
        # dataset = dataset.select(random_indices)
        dataset = dataset.map(formatting_prompts_func, batched=True, )
        split_index = int(len(dataset) * 0.9)
        train_dataset = dataset.select(range(split_index))  # 前90%用于训练
        eval_dataset = dataset.select(range(split_index, len(dataset)))  # 后10%用于评估

        return train_dataset,eval_dataset

    def createTrainner(self):
        train_dataset, eval_dataset = load_dataset(self.datasetSrc)
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=4,  # 设置为4以避免GPTQ量化问题
                warmup_steps=5,
                max_steps=500,  # 微调整体循环次数
                eval_steps=50,  # 每多少次循环评估一次
                learning_rate=2e-4,  # 建议 2e-4 用于4位int量化，1e-4 用于8位int量化
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                evaluation_strategy="steps",
                prediction_loss_only=True,
                eval_accumulation_steps=1,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",  # 也可以用线性： "linear"
                seed=1337,
                output_dir="models",  # 模型保存路径
                report_to="wandb",  # W&B 统计 (可选)
                run_name=self.TUNE_MODEL_NAME + '-' + self.TUNE_MODEL_VERSION,  # W&B 统计 (可选)
            ),
        )

    def startTrainAndSave(self,save_method='merged_4bit_forced'):
        trainer_stats = self.trainer.train()
        self.model.save_pretrained_merged(self.TUNE_MODEL_NAME + '-' + self.TUNE_MODEL_VERSION, self.tokenizer,save_method)
        wandb.finish()