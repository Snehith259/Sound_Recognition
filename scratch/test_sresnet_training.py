import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as tv_models

from Preproccesing.models import SResNet, BasicBlock

FEATURES_DIR = ROOT / "Preproccesing" / "processed" / "features"
SPLITS_DIR = ROOT / "Preproccesing" / "splits"

def load_split(split_name, feat_type):
    df = pd.read_csv(SPLITS_DIR / f"{split_name}.csv")
    subdir = FEATURES_DIR / feat_type
    xs, ys = [], []
    for _, row in df.iterrows():
        stem = Path(row["filename"]).stem
        p = subdir / f"{stem}.npy"
        if p.exists():
            xs.append(np.load(p).astype(np.float32))
            ys.append(int(row["target"]))
    X = torch.from_numpy(np.stack(xs)).unsqueeze(1).float()
    y = torch.tensor(ys, dtype=torch.long)
    return X, y

print("Loading data...")
X_tr_mfcc, y_tr_mfcc = load_split("fold_1_train", "mfcc")
X_va_mfcc, y_va_mfcc = load_split("fold_1_val", "mfcc")
X_te_mfcc, y_te_mfcc = load_split("fold_1_test", "mfcc")

X_tr_mel, y_tr_mel = load_split("fold_1_train", "log_mel")
X_va_mel, y_va_mel = load_split("fold_1_val", "log_mel")
X_te_mel, y_te_mel = load_split("fold_1_test", "log_mel")

print(f"Loaded: MFCC Train {X_tr_mfcc.shape}, Mel Train {X_tr_mel.shape}")

# Test 1: SResNet from models.py trained on MFCC
def train_eval(model, X_tr, y_tr, X_va, y_va, X_te, y_te, lr=1e-3, epochs=25, name="model"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    
    best_va = 0.0
    best_state = None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va_acc = (model(X_va).argmax(dim=1) == y_va).float().mean().item() * 100.0
            if va_acc > best_va:
                best_va = va_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_acc = (model(X_te).argmax(dim=1) == y_te).float().mean().item() * 100.0
    print(f"[{name}] Best Val: {best_va:.1f}% | Test: {te_acc:.1f}% | Time: {time.time()-t0:.1f}s")
    return te_acc

# 1. SResNet on MFCC
m1 = SResNet(in_channels=1, num_classes=50)
train_eval(m1, X_tr_mfcc, y_tr_mfcc, X_va_mfcc, y_va_mfcc, X_te_mfcc, y_te_mfcc, epochs=25, name="SResNet (models.py) on MFCC")

# 2. SResNet on Log-Mel
m2 = SResNet(in_channels=1, num_classes=50)
train_eval(m2, X_tr_mel, y_tr_mel, X_va_mel, y_va_mel, X_te_mel, y_te_mel, epochs=25, name="SResNet (models.py) on Log-Mel")
