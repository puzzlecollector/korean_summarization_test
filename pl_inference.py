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
from pytorch_lightning.callbacks import BasePredictionWriter 


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
        input_ids, attn_masks = batch["input_ids"], batch["attention_mask"] 
        generated = self.gpt.generate(input_ids=input_ids,
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
        summary = tokenizer.batch_decode(summary_tokens, skip_special_tokens=True) 
        print(summary_tokens) 
        print(summary) 
        return summary 




parser = argparse.ArgumentParser() 
parser.add_argument("--setting", "-s", type=str, default="default.yaml", help="Experiment settings")
args = parser.parse_args(args=[]) 
hparams = addict.Addict(dict(load_hparams_from_yaml(args.setting))) 

chkpt = torch.load("cur_chkpt.pt") 


model = NeuralSummarizer(hparams) 

model_states = model.state_dict() 

for k,v in tqdm(model_states.items(), position=0, leave=True, desc="transferring weights"):
    for cur_k, cur_v in chkpt.items():
        if cur_k[16:] == k: 
            model_states[k] = cur_v 

print("loading saved checkpoint!") 
print(model.load_state_dict(model_states))  
model.eval()
model.freeze() 


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b") 

### create test dataloader ### 
class SummaryTestDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self._data = pd.read_csv(data_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        row = self._data.iloc[idx]
        prompt = "{text} 한줄 요약:"
        input_text = prompt.format(text=row['text'])
        input_encoding = self.tokenizer(input_text)

        result = {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
        }

        return result

    def _left_pad(self, sequence, value, max_len):
        return [value] * (max_len - len(sequence)) + sequence

    def collate_fn(self, batch, device='cuda'):
        input_length = max(len(row['input_ids']) for row in batch)

        input_ids = [
            self._left_pad(row['input_ids'], self.tokenizer.pad_token_id, input_length)
            for row in batch
        ]
        attention_mask = [
            self._left_pad(row['attention_mask'], 0, input_length)
            for row in batch
        ]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long, device=device),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long, device=device),
        }


test_path = "aiconnect_test.csv" 
test_set = SummaryTestDataset(test_path, tokenizer)
test_dataloader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False, collate_fn=test_set.collate_fn) 


class CustomWriter(BasePredictionWriter): 
    def __init__(self, output_dir: str, write_interval: str): 
        super().__init__(write_interval) 
        self.output_dir = Path(output_dir) 
        self.output_dir.mkdir(exist_ok=True) 

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices: list, batch, batch_idx: int, dataloader_idx: int):
        batch_idx = str(batch_idx).zfill(9) 
        idx = 0
        while (self.output_dir / f"batch_idx-{batch_idx}_{idx}.pt").exists(): 
            idx += 1 
        torch.save(prediction, self.output_dir / f"batch_idx-{batch_idx}_{idx}.pt") 


output_dir = Path("./outputs2") 
prediction_callback = CustomWriter(output_dir, write_interval="batch")

device_cnt = 1 # 1 is enough for inference 

trainer = pl.Trainer(accelerator="gpu", 
                     devices=1, 
                     strategy="deepspeed" if device_cnt > 1 else "auto", 
                     callbacks=[prediction_callback], 
                     max_epochs=1) 


trainer.predict(model, dataloaders=test_dataloader, return_predictions=False) 
