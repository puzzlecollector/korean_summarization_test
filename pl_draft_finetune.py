import numpy as np 
import pandas as pd 
import os
import torch 
import torch.nn.functional as F 
import torch.nn as nn 
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, IterableDataset 
import sys 
from pathlib import Path 
import shutil 
import pytorch_lightning as pl 
from pytorch_lightning.strategies.ddp import DDPStrategy 
from pytorch_lightning.core.saving import load_hparams_from_yaml, update_hparams
from tqdm.auto import tqdm 
from transformers import (
    AdamW, 
    AutoConfig, 
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup
)
import addict 
import argparse 
import time 
import re 
import datetime 
from sklearn.model_selection import train_test_split
import datasets 

'''
class SummarizationData(Dataset): 
    def __init__(self, path): 
        super().__init__() 
        self.data = [] 
        df = pd.read_csv(path) 
        texts = df["text"].values
        summaries = df["summary"].values 
        for i in range(len(texts)): 
            try: 
                data = [] 
                data.append(texts[i]) 
                data.append(summaries[i]) 
                self.data.append(data)
            except:
                continue 
    def __getitem__(self, index): 
        return self.data[index] 
    def __len__(self): 
        return len(self.data) 

class custom_collate(object): 
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("kakaobrain/kogpt", revision="KoGPT6B-ryan1.5b")
        self.max_length = 1024 
        self.mask_token_id = -100
    
    def left_pad(self, sequence, value, max_len):
        if len(sequence) <= self.max_length: 
            return [value]*(self.max_length - len(sequence)) + sequence 
        else:
            return sequence[:self.max_length] 
    
    def __call__(self, batch): 
        input_ids, attn_masks, target_ids = [], [], [] 

        length = 0 

        for idx, cur_batch in enumerate(batch): 
            original, summary = cur_batch 
            prompt = original + "의 한줄 요약:" 
            query = self.tokenizer(prompt) 
            target = self.tokenizer(summary) 
            query_input_ids = query["input_ids"] 
            query_attn_masks = query["attention_mask"] 
            target_input_ids = target["input_ids"] 
            target_attn_masks = target["attention_mask"] 

            query_input_ids = query_input_ids + target_input_ids + [self.tokenizer.eos_token_id] 
            query_attn_masks = query_input_ids + target_attn_masks + [1] 
            target_input_ids = [self.mask_token_id] * len(query_input_ids) + target_input_ids + [self.tokenizer.eos_token_id] 

            length = max(len(query_input_ids), length)


        for idx, cur_batch in enumerate(batch): 
            try: 
                original, summary = cur_batch 
                prompt = original + "의 한줄 요약:" 
                query = self.tokenizer(prompt)  
                target = self.tokenizer(summary) 
                
                query_input_ids = query["input_ids"] 
                query_attn_masks = query["attention_mask"] 
                target_input_ids = target["input_ids"]
                target_attn_masks = target["attention_mask"] 
                
                # add speical tokens 
                query_input_ids = query_input_ids + target_input_ids + [self.tokenizer.eos_token_id] 
                query_attn_masks = query_input_ids + target_attn_masks + [1] 
                target_input_ids = [self.mask_token_id] * len(query_input_ids) + target_input_ids + [self.tokenizer.eos_token_id]
                
                query_input_ids = self.left_pad(query_input_ids, self.tokenizer.pad_token_id, length) 
                query_attn_masks = self.left_pad(query_attn_masks, 0, length) 
                target_input_ids = self.left_pad(target_input_ids, self.mask_token_id, length) 
                
                input_ids.append(query_input_ids) 
                attn_masks.append(query_attn_masks) 
                target_ids.append(target_input_ids) 
                
            except Exception as e:
                print(e) 
                print("==="*100) 
                continue 
        input_ids = torch.tensor(input_ids, dtype=int) 
        attn_masks = torch.tensor(attn_masks, dtype=int) 
        target_ids = torch.tensor(target_ids, dtype=int) 
        return input_ids, attn_masks, target_ids 

'''
                

class NeuralSummarizer(pl.LightningModule): 
    def __init__(self, hparams=dict()):
        super(NeuralSummarizer, self).__init__() 
        self.hparams.update(hparams) 
        self.save_hyperparameters(ignore="hparams") 
        self.metric = nn.CrossEntropyLoss() 
        self.tokenizer = AutoTokenizer.from_pretrained("kakaobrain/kogpt", revision="KoGPT6B-ryan1.5b-float16")
        self.gpt = AutoModelForCausalLM.from_pretrained("kakaobrain/kogpt", revision="KoGPT6B-ryan1.5b-float16") 
        
    def forward(self, input_ids, attention_mask, target_ids): 
        net_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids) 
        # print(net_out.shape) 
        # net_out = torch.permute(net_out, (0, 2, 1)) 
        # print(net_out.shape)
        return net_out[0]  
    
    def configure_optimizers(self): 
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=float(self.hparams.lr), 
                                      weight_decay=float(self.hparams.weight_decay),
                                      eps=float(self.hparams.adam_epsilon))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches,
        ) 
        scheduler = {"scheduler": scheduler, "interval":"step", "frequency":1} 
        return [optimizer], [scheduler] 
    
    def training_step(self, batch, batch_idx): 
        input_ids, attn_masks, target_ids = batch 
        #self.print(input_ids.shape, attn_masks.shape, target_ids.shape)  
        cur_loss = self(input_ids, attn_masks, target_ids)  
        #self.print(cur_loss) 
        #self.print("="*100) 
        # print(logits.shape, target_ids.shape) 
        # loss = self.metric(logits, target_ids) 
        self.log("loss", cur_loss, batch_size=len(batch), prog_bar=True) 
        return {"loss":cur_loss} 
    
    def validation_step(self, batch, batch_idx): 
        input_ids, attn_masks, target_ids = batch 
        cur_loss = self(input_ids, attn_masks, target_ids) 
        #loss = self.metric(logits, target_ids) 
        self.log("val_loss", cur_loss, batch_size=len(batch), prog_bar=True) 
        return {"val_loss":cur_loss}  

    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int=0): 
        input_ids, attn_masks = batch 
        generated = self.generate(input_ids=input_ids, 
                                  attention_mask=attn_masks, 
                                  pad_token_id=self.tokenizer.pad_token_id, 
                                  max_new_tokens=100, 
                                  do_sample=False, 
                                  num_beams=1, 
                                  num_beam_groups = 1, 
                                  penalty_alpha = None, 
                                  use_cache = True, 
                                  temperature=1.0) 
        prompted_length = input_ids.size(-1) 
        summary_tokens = generated[:, prompted_length:] 


tokenizer = AutoTokenizer.from_pretrained("kakaobrain/kogpt", revision="KoGPT6B-ryan1.5b-float16")


def train_batch_preprocess(batch):
    prompt = "{text} 한줄 요약:"
    query_text = [prompt.format(text=text) for text in batch['text']]
    target_text = batch['summary']
    query = tokenizer(query_text)
    target = tokenizer(target_text)

    input_ids = [q + t + [tokenizer.eos_token_id] for q, t in zip(query['input_ids'], target['input_ids'])]
    attention_mask = [q + t + [1] for q, t in zip(query['attention_mask'], target['attention_mask'])]
    labels = [[-100] * len(q) + t + [tokenizer.eos_token_id] for q, t in zip(query['input_ids'], target['input_ids'])]

    # 결과로 돌려주는 값들이 추가됩니다.
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}



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
    input_ids = torch.tensor(input_ids, dtype=torch.long) 
    attention_mask = torch.tensor(attention_mask, dtype=torch.long) 
    target_ids = torch.tensor(labels, dtype=torch.long) 
    return input_ids, attention_mask, target_ids 
    '''
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long , device=device),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long , device=device),
        'labels': torch.tensor(labels, dtype=torch.long , device=device),
    }
    ''' 
        
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--setting", "--s", type=str, default="default.yaml", help="Experiment Setting") 
    args = parser.parse_args(args=[]) 
    hparams = addict.Addict(dict(load_hparams_from_yaml(args.setting))) 
    #train_set = SummarizationData("train_df.csv") 
    #val_set = SummarizationData("val_df.csv") 

    #collate = custom_collate() 
    #train_dataloader = DataLoader(train_set, batch_size=1, collate_fn=collate, shuffle=True)
    #valid_dataloader = DataLoader(val_set, batch_size=1, collate_fn=collate, shuffle=False) 
    train_df = pd.read_csv("train_df.csv")
    train_set = datasets.Dataset.from_pandas(train_df)
    train_set = train_set.map(
        train_batch_preprocess,
        remove_columns = ['id', 'text', 'summary'],
        batched = True,
        batch_size = 1000,
    )
    train_dataloader = DataLoader(
        train_set, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=collate_fn,
    ) 

    val_df = pd.read_csv("val_df.csv") 
    val_set = datasets.Dataset.from_pandas(val_df) 
    val_set = val_set.map(
        train_batch_preprocess, 
        remove_columns = ["id", "text", "summary"],
        batched = True, 
        batch_size = 1000,
    ) 
    valid_dataloader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=0, 
        collate_fn=collate_fn) 
    
    model = NeuralSummarizer(hparams) 
    chkpt_callback = pl.callbacks.ModelCheckpoint(
        monitor = "val_loss", 
        dirpath = "gen_chkpt/", 
        filename = "epoch_end_checkpoints-{epoch:02}-{val_loss:.8f}",
        save_top_k = 3, 
        mode = "min", 
        save_last = True) 
    device_cnt = torch.cuda.device_count() 
    trainer = pl.Trainer(devices=4, 
                         max_epochs = hparams.epochs, 
                         strategy = "deepspeed" if device_cnt > 1 else None, 
                         callbacks = [chkpt_callback], 
                         gradient_clip_val = 1.0, 
                         accumulate_grad_batches = 10, 
                         num_sanity_val_steps = 10,
                         accelerator = "gpu", 
                         precision="bf16") 
    print("start training model!") 
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader) 
