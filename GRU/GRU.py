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
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)


from GRU_functions import(
    FinetuneDataset,
    collate_fn_daily_gru,
    SimpleDailyGRU,
    evaluate_daily_gru,
    train_daily_gru
    )

# %%

load_dir = "."
cp_path = "."

# %%
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    token_to_id = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    id_to_token = pickle.load(f)

vocab_size = len(token_to_id)
mask_token_id = token_to_id["[MASK]"]
pad_token_id = token_to_id["[PAD]"]

with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    all_sequences_val = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    all_position_ids_val = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    all_labels_val = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    sequence_ids_val = pickle.load(f)
    
# %%
labels_val = torch.tensor(all_labels_val)

# %%
special_tokens = ["[PAD]", "[MASK]"]
exclude_token_ids = [
    token_to_id[tok]
    for tok in special_tokens
    if tok in token_to_id
    ]

def collate_daily(batch):
    return collate_fn_daily_gru(
        batch=batch,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        n_days=365,
        exclude_token_ids=exclude_token_ids
        )

# %%

train_input, val_input, train_pos, val_pos, labels_train, labels_val, train_seq_ids, val_seq_ids  = train_test_split(
    all_sequences_val, all_position_ids_val, labels_val, sequence_ids_val,  test_size=0.3, random_state=123
)

train_dataset = FinetuneDataset(train_input, train_pos, labels_train, train_seq_ids)
val_dataset   = FinetuneDataset(val_input, val_pos, labels_val, val_seq_ids)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_daily)
val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False, collate_fn=collate_daily)

# %%
device = torch.device("cuda" if torch.cuda.is_avaiable() else "cpu")

num_pos = (labels_train == 1).sum()
num_neg = (labels_train == 0).sum()
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float, device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight/3)

# %%

model = SimpleDailyGRU(
    vocab_size=vocab_size,
    day_proj_dim=64,
    hidden_dim=256,
    dropout=0.1,
).to(device)


# %%

train_daily_gru(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    device=device,
    num_epochs=30,
    lr=1e-4,
    weight_decay=1e-4,
    cp_path=cp_path,
)