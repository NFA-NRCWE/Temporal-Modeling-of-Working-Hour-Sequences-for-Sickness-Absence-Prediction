# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:12:14 2026

@author: B317980
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from torch import nn
import torch.nn.functional as F
import random
import gzip
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_constant_schedule_with_warmup
import torch

# %%
with gzip.open(".", 'rb') as f:
    token_to_id = pickle.load(f)
with gzip.open(".", 'rb') as f:
    id_to_token = pickle.load(f)
    
pad_token_id = token_to_id["[PAD]"]
PAD_POS_ID = 366  
    
# %%

class SequenceDataset(Dataset):
    def __init__(self, input_ids_list, pos_ids_list):
        self.input_ids_list = [torch.tensor(x, dtype=torch.long) for x in input_ids_list]
        self.pos_ids_list = [torch.tensor(x, dtype=torch.long) for x in pos_ids_list]

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return self.input_ids_list[idx], self.pos_ids_list[idx]

def collate_fn(batch):
    input_ids, pos_ids = zip(*batch)

    input_ids_padded = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=token_to_id["[PAD]"]
    )
    pos_ids_padded = pad_sequence(
        pos_ids,
        batch_first=True,
        padding_value=366
    )

    padding_mask = (input_ids_padded == token_to_id["[PAD]"])

    return input_ids_padded, pos_ids_padded, padding_mask

# %%

class BERT_model(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=8, dropout=0.05):
        super().__init__()
        self.token_embed    = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.position_embed = nn.Embedding(367, embed_dim, padding_idx=PAD_POS_ID)

        self.emb_ln       = nn.LayerNorm(embed_dim)
        self.emb_dropout  = nn.Dropout(dropout)
        
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
            norm_first=True,
            activation="gelu",
            dim_feedforward=embed_dim*4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.mlm_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, bias=False)
        )

        self.logit_log_scale = nn.Parameter(torch.tensor(2.77))

    def forward(self, input_ids, pos_ids, padding_mask=None):
        x = self.token_embed(input_ids) + self.position_embed(pos_ids)
        x = self.emb_dropout(self.emb_ln(x))  
        h = self.encoder(x, src_key_padding_mask=padding_mask)

        z = F.normalize(self.mlm_proj(h), dim=-1)                   
        W = F.normalize(self.token_embed.weight, dim=-1).t()        

        scale = self.logit_log_scale.clamp(min=-2.0, max=4.0).exp()  
        logits = scale * (z @ W) + self.mlm_bias     
                           
        return logits
# %%

def mask_inputs_by_day_and_static(input_ids, pos_ids, mask_token_id, vocab_size,
                                  day_mask_prob=0.15, static_mask_prob=0.15,
                                  special_tokens={0,1}, PAD_POS_ID=366):

    device = input_ids.device
    B, T = input_ids.shape

    masked_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)

    for b in range(B):
        # ---- Daily masking ----
        days = pos_ids[b].unique()
        days = days[(days > 0) & (days < PAD_POS_ID)]
        if days.numel() > 0:
            num_to_mask = max(1, int(day_mask_prob * days.numel()))
            mask_days = days[torch.randperm(days.numel(), device=device)[:num_to_mask]]
            mask_selector = torch.isin(pos_ids[b], mask_days)
            for tok in special_tokens:
                mask_selector &= (input_ids[b] != tok)
            labels[b, mask_selector] = input_ids[b, mask_selector]
            
            # 80/10/10
            rand = torch.rand(masked_ids.shape[1], device=device)
            mask_mask = (rand < 0.8) & mask_selector
            masked_ids[b, mask_mask] = mask_token_id
            rand_mask = ((rand >= 0.8) & (rand < 0.9)) & mask_selector
            random_tokens = torch.randint(2, vocab_size, (masked_ids.shape[1],), device=device)
            masked_ids[b, rand_mask] = random_tokens[rand_mask]

        # ---- Static masking (pos==0) ----
        static_mask = (pos_ids[b] == 0)
        for tok in special_tokens:
            static_mask &= (input_ids[b] != tok)

        rand_static = torch.rand(masked_ids.shape[1], device=device)
        mask_static_selector = (rand_static < static_mask_prob) & static_mask

        labels[b, mask_static_selector] = input_ids[b, mask_static_selector]
        
        # 80/10/10
        rand = torch.rand(masked_ids.shape[1], device=device)
        mask_mask = (rand < 0.8) & mask_static_selector
        masked_ids[b, mask_mask] = mask_token_id
        rand_mask = ((rand >= 0.8) & (rand < 0.9)) & mask_static_selector
        random_tokens = torch.randint(3, vocab_size, (masked_ids.shape[1],), device=device)
        masked_ids[b, rand_mask] = random_tokens[rand_mask]

    return masked_ids, labels