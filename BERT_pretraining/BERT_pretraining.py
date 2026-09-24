

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

from BERT_pretraining_functions import(
    SequenceDataset,
    collate_fn,
    BERT_model, 
    mask_inputs_by_day_and_static,
    pad_token_id
    )

#%%
load_dir = "."
save_dir = "."

# %%

with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    token_to_id = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    id_to_token = pickle.load(f)   

with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    all_sequences = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    all_position_ids = pickle.load(f)
with gzip.open(os.path.join(load_dir, "."), 'rb') as f:
    sequence_ids = pickle.load(f)

#%%

batch_size = 512

train_input, val_input, train_pos, val_pos = train_test_split(
    all_sequences, all_position_ids, test_size=0.1, random_state=123
)

train_dataset = SequenceDataset(train_input, train_pos)
val_dataset   = SequenceDataset(val_input, val_pos)

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          collate_fn=collate_fn
                          )
val_loader   = DataLoader(val_dataset, 
                          batch_size=batch_size, 
                          shuffle=False,
                          collate_fn=collate_fn
                          )

#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(token_to_id)
mask_token_id = token_to_id["[MASK]"]

pretrain_model = BERT_model(vocab_size).to(device)
optimizer = AdamW(pretrain_model.parameters(), lr=8e-5, weight_decay=5e-3)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

num_epochs = 100
steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * num_epochs

warmup_steps = int(0.1 * total_steps)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)


#%%"

for epoch in range(num_epochs):
    # TRAINING
    pretrain_model.train()
    total_loss = 0.0
    count = 0
    
    train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for input_ids, position_ids, padding_mask in train_iter:
        
        input_ids = input_ids.to(device)
        position_ids = position_ids.to(device)
        padding_mask = padding_mask.to(device)

        masked_input_ids, mlm_labels = mask_inputs_by_day_and_static(
            input_ids, position_ids, mask_token_id, vocab_size
            )
        masked_input_ids = masked_input_ids.to(device)
        mlm_labels = mlm_labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(dtype=torch.bfloat, device_type="cuda"):
            mlm_logits = pretrain_model(masked_input_ids, position_ids, padding_mask=padding_mask)
            loss_mlm = criterion(mlm_logits.view(-1, vocab_size), mlm_labels.view(-1))

        loss_mlm.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss_mlm.item()

        train_iter.set_postfix(loss=loss_mlm.item(), lr=scheduler.get_last_lr()[0])

    avg_train_loss = total_loss / len(train_loader)
    
    # VALIDATION 
    pretrain_model.eval()
    total_val_loss = 0.0
    total_correct, total_masked = 0, 0
    count = 0

    val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    with torch.no_grad():
        
        for input_ids, position_ids, padding_mask in val_iter:
            traced = torch.jit.trace(pretrain_model, (input_ids, position_ids, padding_mask), strict=False) # Tensorboard graph
            input_ids = input_ids.to(device)
            position_ids = position_ids.to(device)
            padding_mask = (input_ids == pad_token_id).to(device)

            masked_input_ids, mlm_labels = mask_inputs_by_day_and_static(
                input_ids, position_ids, mask_token_id, vocab_size
                )
            masked_input_ids = masked_input_ids.to(device)
            mlm_labels = mlm_labels.to(device)
            
            with autocast(dtype=torch.bfloat, device_type="cuda"):
                mlm_logits = pretrain_model(masked_input_ids, position_ids, padding_mask=padding_mask)
                val_loss = criterion(mlm_logits.view(-1, vocab_size), mlm_labels.view(-1))
            
            total_val_loss += val_loss.item()

            m = (mlm_labels != -100)
            if m.any():
                preds = mlm_logits[m].argmax(-1)
                total_correct += (preds == mlm_labels[m]).sum().item()
                total_masked  += m.sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc  = (total_correct / total_masked) if total_masked else float("nan")

    print(f"\nEpoch {epoch+1:3d} | Train MLM Loss: {avg_train_loss:.3f} | "
          f"Val MLM Loss: {avg_val_loss:.3f} | val mask acc: {avg_val_acc:.3f}")
        
    # SAVE CHECKPOINT
    checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': pretrain_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, checkpoint_path)


  