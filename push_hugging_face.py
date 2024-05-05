from unsloth import FastLanguageModel
max_seq_length = 4096
dtype = None
load_in_4bit = True
model_name='/root/autodl-tmp/mistral-7b-instruct-v0.2-bnb-4bit-law-v1.0'
huggingface_token=''

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="cuda",
    attn_implementation="flash_attention_2"
)

MODEL_NAME = "mistral-7b-instruct-v0.2-bnb-4bit-law-v1.0"
model.push_to_hub_merged("" + MODEL_NAME, tokenizer, save_method="merged_4bit_forced", token=huggingface_token)
