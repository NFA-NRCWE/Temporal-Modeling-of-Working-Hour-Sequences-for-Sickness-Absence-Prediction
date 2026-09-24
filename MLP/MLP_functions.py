import torch
import torch.nn as nn


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

def collate_fn_mlp(batch):
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

class MLPorderinvariant(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_token_id,
        n_days=365,
        day_proj_dim=64,
        hidden_dim=256,
        bg_dim=64,
        dropout=0.1,
    ):
        super().__init__()

        self.n_days = n_days
        self.pad_token_id = pad_token_id

        self.day_embedding = nn.Embedding(
            vocab_size,
            day_proj_dim,
            padding_idx=pad_token_id,
        )

        self.bg_embedding = nn.Embedding(
            vocab_size,
            bg_dim,
            padding_idx=pad_token_id,
        )

        self.sequence_encoder = nn.Sequential(
            nn.Linear(day_proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.bg_encoder = nn.Sequential(
            nn.Linear(bg_dim, bg_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + bg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids,
        position_ids,
        padding_mask=None,
    ):

        if padding_mask is None:
            padding_mask = input_ids.eq(self.pad_token_id)

        B, T = input_ids.shape
        valid = ~padding_mask
        
        # Daily
        daily_mask = (
            valid
            & position_ids.ge(1)
            & position_ids.le(self.n_days)
        )

        day_idx = (position_ids - 1).clamp(
            min=0,
            max=self.n_days - 1,
        )

        token_z = self.day_embedding(input_ids)

        token_z = (token_z * daily_mask.unsqueeze(-1).to(token_z.dtype))

        D = token_z.size(-1)

        daily_sum = token_z.new_zeros(B, self.n_days, D,)

        daily_sum.scatter_add_(
            dim=1,
            index=day_idx.unsqueeze(-1).expand(-1, -1, D),
            src=token_z,
        )
        
        # token count
        daily_count = token_z.new_zeros(B, self.n_days, 1, )

        daily_count.scatter_add_(
            dim=1,
            index=day_idx.unsqueeze(-1),
            src=daily_mask.unsqueeze(-1).to(token_z.dtype),
        )

        # Mean-pool tokens 
        daily_z = daily_sum / daily_count.clamp_min(1.0)

        # dya rep
        daily_z = torch.relu(daily_z)

        # mean across days
        pooled_days = daily_z.mean(dim=1)

        seq_summary = self.sequence_encoder(pooled_days)

        # background
        bg_mask = valid & position_ids.eq(0)

        bg_token_z = self.bg_embedding(input_ids)

        bg_token_z = (
            bg_token_z
            * bg_mask.unsqueeze(-1).to(bg_token_z.dtype)
        )

        # sum background token rep
        bg_sum = bg_token_z.sum(dim=1)
        bg_summary = self.bg_encoder(bg_sum)

        combined = torch.cat(
            [seq_summary, bg_summary],
            dim=-1,
        )

        return self.head(combined).squeeze(-1)
    
# %%

@torch.no_grad()
def evaluate_mlp(model, loader, criterion, device, threshold=0.5):
    model.eval()

    total_loss = 0.0
    total_n = 0

    all_probs = []
    all_labels = []

    for daily_x, bg_x, labels, padding_mask, _ in loader:
        
        daily_x = daily_x.to(device)
        bg_x = bg_x.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)

        logits = model(daily_x, bg_x).view(-1)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)

        probs = torch.sigmoid(logits.float())

        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

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

def train_mlp(
    model,
    train_loader,
    val_loader,
    optimizer
    criterion,
    device,
    num_epochs=50,
    cp_path="path"
):
    
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        total_n = 0

        train_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1} [Train]"
        )

        for daily_x, bg_x, labels, padding_mask,_ in train_iter:
            daily_x = daily_x.to(device)
            bg_x = bg_x.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()
            
            with autocast(dtype=torch.bfloat16, device_type='cuda'):
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
        
    return model