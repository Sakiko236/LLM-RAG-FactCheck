import pathlib
import os

original_read_text = pathlib.Path.read_text
def patched_read_text(self, encoding=None, errors=None):
    if encoding is None:
        encoding = 'utf-8'
    return original_read_text(self, encoding=encoding, errors=errors)
pathlib.Path.read_text = patched_read_text
os.environ["PYTHONUTF8"] = "1"

import json
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("Loading data from JSON...")
with open('../../data/cot.json', 'r', encoding='utf-8') as f:
    cot_data = json.load(f)

df = pd.DataFrame(list(cot_data.values()))

print("Formatting data into ChatML...")

def format_chatml(row):

    return {
        "messages": [
            {"role": "system", "content": "You are a factual verification assistant. Think step-by-step to classify the claim based on the evidence."},
            {"role": "user", "content": str(row['claim_text'])},
            {"role": "assistant", "content": str(row['reasoning'])}
        ]
    }

chat_data = df.apply(format_chatml, axis=1).tolist()
train_dataset = Dataset.from_list(chat_data)


print("Loading Model and Tokenizer in 4-bit...")
model_id = "Qwen/Qwen3.5-2B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)

print("Applying LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("Starting Training...")
training_args = SFTConfig(
    output_dir="../../model/qwen-cot-lora",
    per_device_train_batch_size=2,      
    gradient_accumulation_steps=4,      
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1,                 
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",           
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_length=1024                 
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args,
)

trainer.train()

output_dir = "../../model/qwen-cot-lora-final"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Training complete! LoRA weights saved to {output_dir}")