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

from BERT_finetuning_functions import(
    FinetuneDataset,
    collate_fn_finetune,
    Finetune_BERT_split_pool, 
    finetune_optimizer,
    )


from BERT_pretrain.BERT_pretrain_functions import BERT_model


#%%
load_dir = "."
save_dir = "."

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

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn_finetune)
val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False, collate_fn=collate_fn_finetune)

# %%


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrain_model = BERT_model.MLMMedBERT(vocab_size)
optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=0.001)

load_model = torch.load(".", map_location=device)
pretrain_model.load_state_dict(load_model["model_state_dict"])
optimizer.load_state_dict(load_model["optimizer_state_dict"])
pretrain_model = pretrain_model.to(device)

# %%

finetune_model = Finetune_BERT_split_pool(pretrain_model=pretrain_model,
                                          hidden_dim=256,
                                          dropout=0.0
                                          ).to(device)

optimizer = finetune_optimizer(finetune_model)

# %%

num_pos = (labels_train == 1).sum()
num_neg = (labels_train == 0).sum()
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float)
criterion_finetune = nn.BCEWithLogitsLoss(pos_weight=(pos_weight/3))

# %%

N=8
num_epochs = 100

layers = list(finetune_model.encoder.layers)
unfrozen_layers = layers[-N:]

def set_mode_partial(finetune_model, unfrozen_layers):
    finetune_model.train()                 
    finetune_model.encoder.eval()          
    finetune_model.emb_dropout.train()      
    for lyr in unfrozen_layers:            
        lyr.train()


for epoch in range(num_epochs):
    set_mode_partial(finetune_model, unfrozen_layers)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for input_ids, position_ids, labels, padding_mask,_ in train_iter:
        input_ids = input_ids.to(device)
        position_ids = position_ids.to(device)
        padding_mask = padding_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.bfloat16, device_type='cuda'):   
            logits = finetune_model(input_ids, position_ids, padding_mask=padding_mask)
            loss = criterion_finetune(logits.squeeze(-1), labels.float())
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_((p for p in finetune_model.parameters() if p.requires_grad), 1.0) 
        
        optimizer.step()
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds.view(-1) == labels).float().sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()
    
        train_iter.set_postfix(train_loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_samples

    # VALIDATION
    finetune_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_samples = 0
    all_probs = []
    all_labels = []
    val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    with torch.no_grad():
        for input_ids, position_ids, labels, padding_mask,_ in val_iter:
            input_ids = input_ids.to(device)
            position_ids = position_ids.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            logits = finetune_model(input_ids, position_ids, padding_mask=padding_mask)
            loss = criterion_finetune(logits.view(-1), labels.float())

            preds = (torch.sigmoid(logits) > 0.5).float()
            val_correct += (preds.view(-1) == labels).float().sum().item()
            val_samples += labels.size(0)
            val_loss += loss.item()

            val_iter.set_postfix(val_loss=loss.item())
            
            # til plot
            probs = torch.sigmoid(logits).detach().cpu()   
            all_probs.append(probs.view(-1))              
            all_labels.append(labels.cpu().view(-1))

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_correct / val_samples

    all_probs = torch.cat(all_probs).cpu()    
    all_labels = torch.cat(all_labels).cpu()  

    val_preds_epoch = (all_probs > 0.5).int().numpy()
    val_labels_epoch = all_labels.int().numpy()

    cm = confusion_matrix(val_labels_epoch, val_preds_epoch)
    f1 = f1_score(val_labels_epoch, val_preds_epoch)


    print(
        f"Epoch {epoch+1:3d} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | "
        f"Val F1: {f1:.4f}"
    )
    print(cm)


    checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': finetune_model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)


