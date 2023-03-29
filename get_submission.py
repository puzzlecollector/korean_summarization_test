import numpy as np 
import pandas as pd 
import torch
from tqdm.auto import tqdm 

predictions = [] 
version = 0
for i in tqdm(range(0, 125)): 
    batch_idx = str(i).zfill(9) 
    cur_arr = torch.load(f"./outputs/batch_idx-{batch_idx}_{version}.pt") 
    predictions.extend(cur_arr) 


submission = pd.read_csv("aiconnect_sample_submission.csv") 
submission["summary"] = cleaned

print(submission) 

submission.to_csv("clean_polyglot_1_epoch.csv", index=False) 

print("done!") 
