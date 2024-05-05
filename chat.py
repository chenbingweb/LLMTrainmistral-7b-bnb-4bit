from src.inference import Inference
model_name='/root/autodl-tmp/mistral-7b-instruct-v0.2-bnb-4bit-law-v1.0'
inference=Inference(model_name)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ### Input:
        {}

        ### Response:
        {}"""
inference.set_alpaca_prompt(alpaca_prompt)
inference.gennerate('你好')
