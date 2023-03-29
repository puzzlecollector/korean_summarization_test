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
from deepspeed.ops.adam import DeepSpeedCPUAdam


class NeuralSummarizer(pl.LightningModule): 
    def __init__(self, hparams=dict()):
        super(NeuralSummarizer, self).__init__() 
        self.hparams.update(hparams) 
        self.save_hyperparameters(ignore="hparams") 
        self.metric = nn.CrossEntropyLoss() 
        #self.tokenizer = AutoTokenizer.from_pretrained("kakaobrain/kogpt", revision="KoGPT6B-ryan1.5b-float16")
        #self.gpt = AutoModelForCausalLM.from_pretrained("kakaobrain/kogpt", revision="KoGPT6B-ryan1.5b-float16")  
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b") 
        self.gpt = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-5.8b")
        
    def forward(self, input_ids, attention_mask, target_ids): 
        net_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids) 
        # print(net_out.shape) 
        # net_out = torch.permute(net_out, (0, 2, 1)) 
        # print(net_out.shape)
        return net_out[0]  
    
    def configure_optimizers(self): 
        
        optimizer = DeepSpeedCPUAdam(self.parameters(), 
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


# tokenizer = AutoTokenizer.from_pretrained("kakaobrain/kogpt", revision="KoGPT6B-ryan1.5b-float16") 
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b") 


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
        left_pad(row['labels'], -100, length)
        for row in batch
    ]
    input_ids = torch.tensor(input_ids, dtype=torch.long) 
    attention_mask = torch.tensor(attention_mask, dtype=torch.long) 
    target_ids = torch.tensor(labels, dtype=torch.long) 
    return input_ids, attnetion_mask, target_ids 
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--setting", "--s", type=str, default="default.yaml", help="Experiment Setting") 
    args = parser.parse_args(args=[]) 
    hparams = addict.Addict(dict(load_hparams_from_yaml(args.setting))) 
    
    train_df = pd.read_csv("train_df.csv")
    train_set = datasets.Dataset.from_pandas(train_df)
    train_set = train_set.map(
        train_batch_preprocess,
        remove_columns = ['id', 'text', 'summary'],
        batched = True,
        batch_size = 1000,
    )
    train_dataloader = DataLoader(
        train_set, batch_size=8, shuffle=True, num_workers=0,
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
        val_set, batch_size=8, shuffle=False, num_workers=0, 
        collate_fn=collate_fn) 
    
    model = NeuralSummarizer(hparams) 
    chkpt_callback = pl.callbacks.ModelCheckpoint(
        monitor = "val_loss", 
        dirpath = "gen_chkpt/", 
        filename = "polyglot_fixed-{epoch:02}-{val_loss:.8f}",
        save_top_k = 3, 
        mode = "min", 
        save_last = True) 
    device_cnt = torch.cuda.device_count() 
    trainer = pl.Trainer(devices=4, 
                         max_epochs = hparams.epochs, 
                         strategy = "deepspeed_stage_3_offload" if device_cnt > 1 else None, 
                         callbacks = [chkpt_callback], 
                         gradient_clip_val = 1.0, 
                         accumulate_grad_batches = 10, 
                         num_sanity_val_steps = 10,
                         accelerator = "gpu", 
                         precision="bf16") 
    print("start training model!") 
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader) 
