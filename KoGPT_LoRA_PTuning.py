import numpy as np 
import pandas as pd
import os, gc  
from tqdm.auto import tqdm 
from datetime import datetime, timezone, timedelta 
import torch  
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset 
import datasets 
import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM 
from peft import get_peft_model, PeftModel, TaskType, LoraConfig, PromptTuningConfig, PromptTuningInit

# define tokenizer 
'''
tokenizer = AutoTokenizer.from_pretrained(
    'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
)
''' 
tokenizer = AutoTokenizer.from_pretrained(
    "kakaobrain/kogpt", revision="KoGPT6B-ryan1.5b"
)

# train/validation split 
full_df = pd.read_csv("train.csv") 
train_size = int(full_df.shape[0] * 0.8) 
train_df = full_df.iloc[:train_size] 
val_df = full_df.iloc[train_size:] 

train_set = datasets.Dataset.from_pandas(train_df) 
val_set = datasets.Dataset.from_pandas(val_df) 

# save some memory 
del train_df
del val_df 

def batch_preprocess(batch):  
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
    batch_preprocess, 
    remove_columns = ["id", "text", "summary"], 
    batched = True, 
    batch_size = 1000
) 

val_set = val_set.map(
    batch_preprocess, 
    remove_columns = ["id", "text", "summary"], 
    batched = True,
    batch_size = 1000 
) 

def left_pad(sequence, value, max_len):
    return [value] * (max_len - len(sequence)) + sequence

def collate_fn(batch, device='cuda'):
    length = max(len(row['input_ids']) for row in batch)
    input_ids = [
        left_pad(row['input_ids'], tokenizer.pad_token_id, length)
        for row in batch
    ]
    attention_mask = [
        left_pad(row['attention_mask'], 0, length)
        for row in batch
    ]
    labels = [
        left_pad(row['input_ids'], -100, length)
        for row in batch
    ]
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long , device=device),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long , device=device),
        'labels': torch.tensor(labels, dtype=torch.long , device=device),
    }

train_loader = DataLoader(train_set, batch_size = 4, shuffle=True, collate_fn=collate_fn) 
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=collate_fn) 

# define base_model  
print("loading model") 
'''
base_model = AutoModelForCausalLM.from_pretrained(
    'kakaobrain/kogpt', revision = 'KoGPT6B-ryan1.5b-float16',
    torch_dtype = torch.float16,
    device_map = 'auto',
)
''' 


base_model = AutoModelForCausalLM.from_pretrained(
    "kakaobrain/kogpt", revision="KoGPT6B-ryan1.5b",
    torch_dtype="auto", 
    device_map="auto"
)


for path, dirs, files in os.walk("/root/.cache/huggingface/hub/models--kakaobrain--kogpt"):
    for file in files: 
        if file.endswith("tokenizer.json"):
            tokenizer_path = path 

print(tokenizer_path) 

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,
    prompt_tuning_init=PromptTuningInit.TEXT,
    
    prompt_tuning_init_text="다음 글을 읽고 요약해줘:",
    tokenizer_name_or_path=tokenizer_path  
)

peft_model = get_peft_model(base_model, peft_config)
peft_model.to("cuda") 


best_val_loss = 99999999999 # some large value 

learning_rate = 3e-5 
optimizer = torch.optim.Adam(peft_model.parameters(), lr=learning_rate) 
scaler = torch.cuda.amp.GradScaler() 

def training_step(model, batch, optimizer, scaler): 
    optimizer.zero_grad() 
    with torch.cuda.amp.autocast(): 
        outputs = model(
            input_ids = batch["input_ids"], 
            attention_mask = batch["attention_mask"], 
            labels = batch["labels"]
        ) 
        step_loss = outputs[0] 
    scaler.scale(step_loss).backward() 
    scaler.step(optimizer) 
    scaler.update() 
    return step_loss.detach() 

def validation_step(model, batch): 
    model.eval() 
    with torch.no_grad(): 
        with torch.cuda.amp.autocast(): 
            outputs = model(
                input_ids = batch["input_ids"], 
                attention_mask = batch["attention_mask"], 
                labels = batch["labels"] 
            ) 
            step_loss = outputs[0] 
    return step_loss.detach() 

NUM_EPOCHS = 3 

for epoch in tqdm(range(NUM_EPOCHS), position=0, leave=True, desc="Epochs"): 
    peft_model.train() 
    train_loss = 0 
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training"), start=1):
        step_loss = training_step(peft_model, batch, optimizer, scaler)
        train_loss += step_loss 
        if batch_idx % 300 == 0 and batch_idx > 0:
            print(f"current train loss : {train_loss / (batch_idx+1)}")
    val_loss = 0 
    peft_model.eval() 
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating"), start=1): 
        step_loss = validation_step(peft_model, batch) 
        val_loss += step_loss 
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"average train loss: {avg_train_loss} | average validation loss: {avg_val_loss}") 
    
    if avg_val_loss < best_val_loss: 
        print("saving best checkpoint!") 
        best_val_loss = avg_val_loss 
        peft_model.save_pretrained("best_peft_chkpt.pt") 
        
    
print("done training!") 
print(f"best_val_loss : {best_val_loss}")
