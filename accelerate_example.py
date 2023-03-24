# no PEFT 
# try full fine-tuning with A100-80GB  
import numpy as np
import pandas as pd
import os, gc
from tqdm.auto import tqdm
from datetime import datetime, timezone, timedelta
import torch
from torch.utils.data import DataLoader ,Dataset
import datasets
from transformers import * 
from accelerate import Accelerator


tokenizer = AutoTokenizer.from_pretrained("kakaobrain/kogpt", revision="KoGPT6B-ryan1.5b-float16") 

train_df = pd.read_csv("train.csv") 
train_set = datasets.Dataset.from_pandas(train_df) 

def train_batch_preprocess(batch): 
    prompt = "{text} 한줄 요약:" 
    query_text = [prompt.format(text=text) for text in batch["text"]] 
    target_text = batch["summary"] 
    query = tokenizer(query_text) 
    target = tokenizer(target_text) 
    input_ids = [q + t + [tokenizer.eos_token_id] for q, t in zip(query["input_ids"], target["input_ids"])] 
    attention_mask = [q + t + [1] for q, t in zip(query["attention_mask"], target["attention_mask"])] 
    labels = [[-100] * len(q) + t + [tokenizer.eos_token_id] for q, t in zip(query["input_ids"], target["input_ids"])] 
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels} 

train_set = train_set.map(
    train_batch_preprocess, 
    remove_columns = ["id", "text", "summary"], 
    batched = True,  
    batch_size = 1000, 
) 

def left_pad(sequence, value, max_len): 
    return [value] * (max_len - len(sequence)) + sequence 

def collate_fn(batch, device="cuda"): 
    length = max(len(row["input_ids"]) for row in batch)
    input_ids = [
        left_pad(row["input_ids"], tokenizer.pad_token_id, length) 
        for row in batch
    ] 
    attention_mask = [
        left_pad(row["attention_mask"], 0, length) 
        for row in batch
    ] 
    labels = [
        left_pad(row["input_ids"], -100, length) 
        for row in batch
    ]  
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device), 
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device), 
        "labels": torch.tensor(labels, dtype=torch.long, device=device) 
    } 

train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn) 

model = AutoModelForCausalLM.from_pretrained("kakaobrain/kogpt", revision = "KoGPT6B-ryan1.5b-float16", torch_dtype = "auto", device_map="auto") 
accelerator = Accelerator()
device = accelerator.device 
optimizer = AdamW(model.parameters(), lr=3e-5)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

model.train()

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(tqdm(train_dataloader), start=1):
        if batch_idx == 30: 
            break 
        input_ids = batch["input_ids"].to(device) 
        attention_mask = batch["attention_mask"].to(device) 
        labels = batch["labels"].to(device) 
        outputs = model(
            input_ids = batch["input_ids"], 
            attention_mask = batch["attention_mask"], 
            labels = batch["labels"],
        )
        loss = outputs[0]
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1) 
        
