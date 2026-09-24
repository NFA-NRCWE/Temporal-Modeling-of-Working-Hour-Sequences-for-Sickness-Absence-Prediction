import os
import numpy as np
import pandas as pd
from torch import nn
import gzip
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import get_constant_schedule_with_warmup
import torch
from torch.amp import autocast
from sklearn.metrics import confusion_matrix, f1_score

from MLP_functions import(
    FinetuneDataset,
    collate_fn_mlp,
    MLPorderinvariant,
    train_mlp,
    )

#%%
load_dir = "."
cp_path = "."

# %%


with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    token_to_id = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    id_to_token = pickle.load(f)   


with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    all_sequences_val = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    all_position_ids_val = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    all_labels_val = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    sequence_ids_val = pickle.load(f)

# %%

vocab_size = len(token_to_id)
mask_token_id = token_to_id["[MASK]"]
pad_token_id = token_to_id["[PAD]"]

# %%

labels_val = torch.tensor(all_labels_val)

# %%
train_input, val_input, train_pos, val_pos, labels_train, labels_val, train_seq_ids, val_seq_ids  = train_test_split(
    all_sequences_val, all_position_ids_val, labels_val, sequence_ids_val,  test_size=0.3, random_state=123
)

train_dataset = FinetuneDataset(train_input, train_pos, labels_train, train_seq_ids)
val_dataset   = FinetuneDataset(val_input, val_pos, labels_val, val_seq_ids)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn_mlp)
val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False, collate_fn=collate_fn_mlp)

# %%

device = torch.device("cuda" if torch.cuda.is_avaiable() else "cpu")

num_pos = (labels_train == 1).sum()
num_neg = (labels_train == 0).sum()
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float, device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight/3)

# %%

model = MLPorderinvariant(
    vocab_size=vocab_size,
    n_days=365,
    hidden_dim=256,
    day_proj_dim=64,
    bg_dim=64,    
    dropout=0.1,
).to(device)


# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

train_mlp(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=30,
    cp_path=cp_path, 
)