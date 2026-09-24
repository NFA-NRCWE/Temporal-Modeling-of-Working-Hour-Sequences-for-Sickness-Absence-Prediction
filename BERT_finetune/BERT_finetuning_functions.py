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

def collate_fn_finetune(batch):
    input_ids, pos_ids, labels, seq_ids = zip(*batch)
    
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

    labels_tensor = torch.stack(labels)

    padding_mask = (input_ids_padded == token_to_id["[PAD]"])

    return input_ids_padded, pos_ids_padded, labels_tensor, padding_mask, list(seq_ids)


# %%

class Finetune_BERT_split_pool(nn.Module):
    def __init__(
        self,
        pretrained_model,
        hidden_dim=64,
        output_dim=1,
        dropout=0.2,
    ):
        super().__init__()

        self.token_embed    = pretrained_model.token_embed
        self.position_embed = pretrained_model.position_embed
        self.emb_ln         = pretrained_model.emb_ln
        self.emb_dropout    = pretrained_model.emb_dropout
        self.encoder        = pretrained_model.encoder

        d_model = self.token_embed.embedding_dim

        self.bg_attn = nn.Linear(d_model, 1, bias=False)
        self.daily_attn = nn.Linear(d_model, 1, bias=False)

        self.decoder_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )


    def masked_attention_pool(self, h, mask, attn_layer):

        attn_logits = attn_layer(h).squeeze(-1)         
        attn_logits = attn_logits.masked_fill(~mask, -1e9)

        weights = F.softmax(attn_logits, dim=1)        
        weights = weights * mask.float()

        summary = torch.bmm(weights.unsqueeze(1), h).squeeze(1) 

        return summary, weights

    def forward(self, 
                input_ids, 
                position_ids, 
                padding_mask=None, 
                return_attention=False
                ):
        x = self.token_embed(input_ids) + self.position_embed(position_ids)
        x = self.emb_dropout(self.emb_ln(x))

        h = self.encoder(x, src_key_padding_mask=padding_mask)

        if padding_mask is None:
            valid = torch.ones(h.size()[:2], dtype=torch.bool, device=h.device)
        else:
            valid = ~padding_mask

        bg_mask = valid & (position_ids == 0)
        daily_mask = valid & (position_ids >= 1) & (position_ids <= 365)

        bg_summary, bg_attn = self.masked_attention_pool(
            h,
            bg_mask,
            self.bg_attn,
        )

        daily_summary, daily_attn = self.masked_attention_pool(
            h,
            daily_mask,
            self.daily_attn,
        )

        summary = torch.cat([bg_summary, daily_summary], dim=-1)

        logits = self.decoder_head(summary).squeeze(-1)

        if return_attention:
            return logits, bg_attn, daily_attn, h

        return logits
    
# %%

def finetune_optimizer(finetune_model):

    N = 8

    for p in finetune_model.parameters():
        p.requires_grad = False

    for p in finetune_model.bg_attn.parameters():
        p.requires_grad = True
    for p in finetune_model.daily_attn.parameters():
        p.requires_grad = True
        
    for p in finetune_model.token_embed.parameters():
        p.requires_grad = True
    for p in finetune_model.position_embed.parameters():
        p.requires_grad = True
    for p in finetune_model.emb_ln.parameters():
        p.requires_grad = True

    for p in finetune_model.decoder_head.parameters():
        p.requires_grad = True

    layers = list(finetune_model.encoder.layers)
    if N > 0:
        for layer in layers[-N:]:
            for p in layer.parameters():
                p.requires_grad = True

    head_params = (
        list(finetune_model.bg_attn.parameters())
        + list(finetune_model.daily_attn.parameters())
        + list(finetune_model.decoder_head.parameters())
    )
    
    encoder_params = [
        p
        for layer in layers[-N:]
        for p in layer.parameters()
        if p.requires_grad
        ]
    
    token_embed_params = list(finetune_model.token_embed.parameters())
    position_embed_params = list(finetune_model.position_embed.parameters())
    emb_ln_params = list(finetune_model.emb_ln.parameters())



    optimizer = AdamW(
        [
            {"params": head_params,  "lr": 7e-5, "weight_decay": 1e-2},
            
            {"params": encoder_params,  "lr": 7e-5, "weight_decay": 1e-2},
            
            {"params": token_embed_params, "lr": 7e-5, "weight_decay": 1e-2},
            {"params": position_embed_params, "lr": 7e-6, "weight_decay": 1e-2},
            {"params": emb_ln_params, "lr": 7e-6, "weight_decay": 1e-2},
        ]
    )
   
    return optimizer