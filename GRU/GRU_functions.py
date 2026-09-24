import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
from torch import nn
import torch.nn.functional as F
import random
import gzip
import pickle
import os
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import torch

import torch
from functools import partial

# %%



class FinetuneDataset(Dataset):
    def __init__(self, input_ids_list, pos_ids_list, labels_list, seq_ids):
        self.input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids_list]
        self.pos_ids = [torch.tensor(x, dtype=torch.long) for x in pos_ids_list]
        self.labels = torch.tensor(labels_list, dtype=torch.float)
        self.seq_ids = seq_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.pos_ids[idx], self.labels[idx], self.seq_ids[idx]
    
def collate_fn_gru(
    batch,
    vocab_size,
    pad_token_id,
    n_days=365,
    exclude_token_ids=None,
):
    """
    Input: 
    input_ids: [seq_len]
    pos_ids: [seq_len]
    
    output:
    daily_x: [B, 365, vocab_size]
    bg_x: [B, vocab_size]
    labels: [B]
    """

    input_ids, pos_ids, labels, seq_ids = zip(*batch)
    B = len(input_ids)

    daily_x = torch.zeros(B, n_days, vocab_size, dtype=torch.float32)
    bg_x = torch.zeros(B, vocab_size, dtype=torch.float32)
    labels_tensor = torch.stack(labels).float().view(-1)

    exclude_token_ids = {int(x) for x in (exclude_token_ids or []) if x is not None}
    exclude_token_ids.add(int(pad_token_id))

    for b, (ids, pos) in enumerate(zip(input_ids, pos_ids)):
        ids = ids.long()
        pos = pos.long()

        valid = (pos >= 0) & (pos <= n_days)

        for special_id in exclude_token_ids:
            valid &= (ids != special_id)

        ids = ids[valid]
        pos = pos[valid]

        if ids.numel() == 0:
            continue

        # background tokens: position 0
        bg_mask = (pos == 0)
        if bg_mask.any():
            bg_ids = ids[bg_mask]
            bg_x[b].index_add_(
                0,
                bg_ids,
                torch.ones(bg_ids.size(0), dtype=torch.float32)
            )

        # daily tokens: positions 1-365
        day_mask = (pos >= 1) & (pos <= n_days)
        if day_mask.any():
            day_idx = pos[day_mask] - 1
            tok_idx = ids[day_mask]

            daily_x[b].index_put_(
                (day_idx, tok_idx),
                torch.ones(tok_idx.size(0), dtype=torch.float32),
                accumulate=True
            )
            
    # binary            
    daily_x.clamp_(0, 1)
    bg_x.clamp_(0, 1)

    return daily_x, bg_x, labels_tensor, list(seq_ids)
# %%
    
class SimpleDailyGRU(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 day_proj_dim=64, 
                 hidden_dim=256, 
                 dropout=0.1
                 ):
        super().__init__()

        self.day_encoder = nn.Sequential(
            nn.Linear(vocab_size, day_proj_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=day_proj_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.bg_encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, daily_x, bg_x):
        day_z = self.day_encoder(daily_x)   

        out, h_n = self.gru(day_z)          
        seq_summary = h_n[-1]               

        bg_z = self.bg_encoder(bg_x)        

        x = torch.cat([seq_summary, bg_z], dim=-1)

        return self.head(x).squeeze(-1)
    

# %%

@torch.no_grad()
def evaluate_daily_gru(model, loader, criterion, device, threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_n = 0

    all_probs = []
    all_labels = []

    for daily_x, bg_x, labels in loader:
        daily_x = daily_x.to(device, non_blocking=True)
        bg_x = bg_x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)

        logits = model(daily_x, bg_x)
        loss = criterion(logits.view(-1), labels)

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)

        probs = torch.sigmoid(logits).detach().cpu().view(-1)

        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().view(-1))

    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)

    y_pred = (probs >= threshold).astype(int)

    metrics = {
        "loss": total_loss / max(total_n, 1),
        "auroc": roc_auc_score(y_true, probs),
        "auprc": average_precision_score(y_true, probs),
        "f1": f1_score(y_true, y_pred),
    }

    cm = confusion_matrix(y_true, y_pred)

    return metrics, cm, probs, y_true
# %%


    
def train_daily_gru(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    num_epochs=50,
    lr=0.001,
    weight_decay=0.001,
    cp_path="path"
):

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        total_n = 0

        train_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1} [Train]",
            leave=False
        )

        for daily_x, bg_x, labels in train_iter:
            daily_x = daily_x.to(device, non_blocking=True)
            bg_x = bg_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            optimizer.zero_grad()

            logits = model(daily_x, bg_x)
            loss = criterion(logits.view(-1), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)

            train_iter.set_postfix(train_loss=loss.item())

        train_loss = total_loss / max(total_n, 1)

        val_metrics, val_cm, _, _ = evaluate_daily_gru(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUROC: {val_metrics['auroc']:.4f} | "
            f"Val AUPRC: {val_metrics['auprc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )
        print(val_cm)
        
        # save cp
        cp_path_ = os.path.join(cp_path, f"epoch_{epoch+1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            }, cp_path_)
        
    return criterion, optimizer