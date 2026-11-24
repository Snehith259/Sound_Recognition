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

FEATURES_DIR = ROOT / "Preproccesing" / "processed" / "features"
SPLITS_DIR = ROOT / "Preproccesing" / "splits"

def load_split(split_name, feat_type="log_mel"):
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

print("Loading Log-Mel features for fold 1...", flush=True)
X_tr, y_tr = load_split("fold_1_train")
X_va, y_va = load_split("fold_1_val")
X_te, y_te = load_split("fold_1_test")

class SResNetAudio(nn.Module):
    def __init__(self, num_classes=50, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.resnet18(weights=weights)
        
        orig_conv = base.conv1.weight.data
        new_conv = orig_conv.mean(dim=1, keepdim=True)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.conv1.weight.data = new_conv
        
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

def extract_features(bb, X, batch_size=32):
    bb.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            feats.append(bb(X[i:i+batch_size]))
    return torch.cat(feats)

m = SResNetAudio(num_classes=50, pretrained=True)
print("Extracting backbone features...", flush=True)
t0 = time.time()
F_tr = extract_features(m.backbone, X_tr)
F_va = extract_features(m.backbone, X_va)
F_te = extract_features(m.backbone, X_te)
print(f"Features ready in {time.time()-t0:.2f}s", flush=True)

class SqueezedResidualHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, num_classes=50, dropout=0.25):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        
        # Residual Squeeze Block
        self.res_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.res_bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.res_act1 = nn.SiLU()
        self.res_fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.res_bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(dropout)
        
        self.out = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        h = self.drop1(self.act1(self.bn1(self.fc1(x))))
        residual = self.drop2(self.res_bn2(self.res_fc2(self.res_act1(self.res_bn1(self.res_fc1(h))))))
        h = F.silu(h + residual)
        return self.out(h)

head = SqueezedResidualHead(512, 256, 50, dropout=0.25)
opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=70, eta_min=1e-5)

def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

loader = DataLoader(TensorDataset(F_tr, y_tr), batch_size=32, shuffle=True)

best_va = 0.0
best_head = None
t_train = time.time()
for ep in range(1, 71):
    head.train()
    for bx, by in loader:
        opt.zero_grad()
        if np.random.rand() < 0.5:
            mixed_bx, y_a, y_b, lam = mixup_data(bx, by, alpha=0.25)
            pred = head(mixed_bx)
            loss = lam * F.cross_entropy(pred, y_a, label_smoothing=0.06) + (1 - lam) * F.cross_entropy(pred, y_b, label_smoothing=0.06)
        else:
            loss = F.cross_entropy(head(bx), by, label_smoothing=0.06)
        loss.backward()
        opt.step()
    sched.step()
    
    head.eval()
    with torch.no_grad():
        va_acc = (head(F_va).argmax(dim=1) == y_va).float().mean().item() * 100.0
        if va_acc > best_va:
            best_va = va_acc
            best_head = {k: v.clone() for k, v in head.state_dict().items()}

head.load_state_dict(best_head)
head.eval()
with torch.no_grad():
    te_acc = (head(F_te).argmax(dim=1) == y_te).float().mean().item() * 100.0
    te_logits = head(F_te)
    top5 = te_logits.topk(5, dim=1).indices
    top5_hits = sum(1 for i in range(len(y_te)) if y_te[i] in top5[i])
    top5_acc = top5_hits / len(y_te) * 100.0

print(f"\n[SResNet Squeezed Residual Head + Mixup] Best Val: {best_va:.2f}% | Test Top-1: {te_acc:.2f}% | Test Top-5: {top5_acc:.2f}% | Head Train Time: {time.time()-t_train:.2f}s", flush=True)
