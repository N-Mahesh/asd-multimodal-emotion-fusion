from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# The PrecomputedFeatures class is create by: https://github.com/julianrosen/erdos_dl_recanvo_project
class PrecomputedFeatures(torch.utils.data.Dataset):
    def __init__(self, df, labels=[], root_dir=None): #To construct a Features object, you feed in a dataframe with column "Filename" with the names of the audio files, as well as a list of expected labels.
        self.df=df
        self.root_dir= Path(root_dir) if root_dir is not None else (ROOT / "data/wav")
        if labels==[]:
            labels=self.df.Label.unique() # if no list of labels is provided, take the labels appearing in df
        self.labeldict=dict(zip(labels, range(len(labels)))) #keeps track of the labels and zips them with numerical values 0,1,...


    def __getitem__(self, idx): #idx is an index of our dataframe
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.df.Filename.iloc[idx]
        # Load precomputed HuBERT features saved by scripts/HuBERTexport.py
        # Note: directory name is 'HuBERT_features'
        feat_stem = filename[:-4] if filename.endswith('.wav') else filename
        feat_path = ROOT / 'data' / 'HuBERT_features' / f"{feat_stem}.pt"
        X = torch.load(feat_path).detach()
        y = torch.zeros(len(self.labeldict)).detach()
        y[self.labeldict[self.df.Label.iloc[idx]]] = 1
        z = self.labeldict[self.df.Label.iloc[idx]]
        # z is the integer corresponding to our label. We return z for use in train-test splits through sklearn - had issues using tensors in sklearn's stratify
        return X, y, z
    def __len__(self):
        return len(self.df)

ROOT = Path(__file__).resolve().parents[2]

DATA_CSV = ROOT / "data/directory_w_train_test.csv"
OUT_DIR = ROOT / "model_validation/hubert_dense"
MIN_PER_LABEL = 30
EPOCHS = 50
BATCH_SIZE = 64
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6
NUM_WORKERS = 2
SEED = 42
NO_CLASS_WEIGHTS = False 

data_dir = Path(DATA_CSV).resolve().parent
model_dir = Path(OUT_DIR)
model_dir.mkdir(parents=True, exist_ok=True)

# Checkpoint directory
checkpoint_dir = model_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load file list
file_df = pd.read_csv(DATA_CSV)
train_df = file_df[file_df["is_test"] == 0].copy()

min_per_label = int(MIN_PER_LABEL)
label_counts = train_df.Label.value_counts()
keep_labels = set(label_counts[label_counts >= min_per_label].index)
train_df = train_df[train_df.Label.isin(keep_labels)].reset_index(drop=True)

# Dataset and split
dataset = PrecomputedFeatures(train_df)
labels = list(dataset.labeldict.keys())
num_labels = len(labels)

# simple train/val split
_, _, train_idx, val_idx = train_test_split(
    train_df.index,
    train_df.index,
    stratify=train_df.Label,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)

train_loader = DataLoader(train_set, batch_size=int(BATCH_SIZE), shuffle=True, num_workers=int(NUM_WORKERS))
val_loader = DataLoader(val_set, batch_size=max(1, int(BATCH_SIZE) * 2), shuffle=False, num_workers=int(NUM_WORKERS))

# Classifier
class head_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout0 = nn.Dropout(0.3)
        self.norm = nn.BatchNorm1d(1280)
        self.layer1 = nn.Linear(1280, 1280)
        self.activation1 = nn.GELU()
        self.dropout1=nn.Dropout(0.2)
        self.layer2 = nn.Linear(1280, 256)
        self.activation2 = nn.GELU()
        self.dropout2=nn.Dropout(0.2)
        self.output = nn.Linear(256, num_labels)

    def forward(self, x):
        x = self.dropout0(x)
        x = self.norm(x)
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x

model = head_classifier()

model = model.to(device)

# Class weights (handle imbalance)
if NO_CLASS_WEIGHTS:
    class_weight_tensor = None
else:
    # Compute weights inversely proportional to class frequency
    counts = train_df.Label.value_counts()
    weights = []
    for label, idx in dataset.labeldict.items():
        c = counts.get(label, 0)
        weights.append(0.0 if c == 0 else 1.0 / c)
    weights = np.array(weights, dtype=np.float32)
    # Normalize to average 1.0 to keep loss scale
    if weights.sum() > 0:
        weights = weights * (len(weights) / weights.sum())
    class_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=float(LR), weight_decay=float(WEIGHT_DECAY))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5)

# Train
epochs = int(EPOCHS)
best_val_f1 = -1.0
epochs_since_improve = 0
for epoch in range(epochs):
    model.train()
    train_correct = 0
    train_total = 0
    train_preds = []
    train_targets = []
    for X, y, z in train_loader:
        optimizer.zero_grad()
        X = X.to(device)
        z_t = torch.tensor(z, dtype=torch.long, device=device)
        logits = model(X)
        # CrossEntropy expects class indices
        loss = criterion(logits, z_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Train accuracy and F1
        preds = logits.argmax(1)
        train_correct += (preds.cpu() == z_t.cpu()).sum().item()
        train_total += len(z)
        train_preds.extend(preds.cpu().tolist())
        train_targets.extend(list(z))

    train_acc = train_correct / max(1, train_total)
    train_f1 = f1_score(train_targets, train_preds, average="macro") if train_total > 0 else 0.0

    # quick val accuracy
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y, z in val_loader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(1)
            correct += (preds.cpu() == torch.tensor(z)).sum().item()
            total += len(z)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(list(z))
    acc = correct / max(1, total)
    val_f1 = f1_score(all_targets, all_preds, average="macro") if total > 0 else 0.0
    print(f"Epoch {epoch+1}/{epochs} - train_acc={train_acc:.3f} - train_macro_f1={train_f1:.3f} - val_acc={acc:.3f} - val_macro_f1={val_f1:.3f}")
    # LR schedule and early stopping
    scheduler.step(val_f1)

    # Save checkpoint every epoch
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_f1': val_f1,
        'acc': acc,
        'labeldict': dataset.labeldict
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_since_improve = 0
        torch.save(model.state_dict(), model_dir / "model.pt")
    else:
        epochs_since_improve += 1
        if epochs_since_improve >= int(PATIENCE):
            print(f"Early stopping at epoch {epoch+1} (no F1 improvement for {PATIENCE} epochs)")
            break

# Save label mapping
with open(model_dir / "labels.json", "w") as f:
    json.dump({label: i for label, i in dataset.labeldict.items()}, f, indent=2)

print("Saved:")
print(" -", model_dir / "model.pt")
print(" -", model_dir / "labels.json")
