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

from Preproccesing.models import SResNet

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

def init_sresnet_from_resnet18(sresnet_model):
    r18 = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    sd_src = r18.state_dict()
    sd_dst = sresnet_model.state_dict()
    
    # Map conv1
    conv1_w = sd_src["conv1.weight"].mean(dim=1, keepdim=True)[:32, :, 2:5, 2:5]
    sd_dst["conv1.weight"] = conv1_w
    sd_dst["bn1.weight"] = sd_src["bn1.weight"][:32]
    sd_dst["bn1.bias"] = sd_src["bn1.bias"][:32]
    sd_dst["bn1.running_mean"] = sd_src["bn1.running_mean"][:32]
    sd_dst["bn1.running_var"] = sd_src["bn1.running_var"][:32]
    
    # Map layers 1..4
    stages = [("layer1", 32, 32), ("layer2", 64, 32), ("layer3", 128, 64), ("layer4", 256, 128)]
    for stage_name, out_c, in_c in stages:
        for b in [0, 1]:
            for conv_idx in [1, 2]:
                k_w = f"{stage_name}.{b}.conv{conv_idx}.weight"
                if k_w in sd_dst and k_w in sd_src:
                    s_out, s_in = sd_dst[k_w].shape[:2]
                    sd_dst[k_w] = sd_src[k_w][:s_out, :s_in, :, :]
                
                k_bn = f"{stage_name}.{b}.bn{conv_idx}"
                for param in ["weight", "bias", "running_mean", "running_var"]:
                    k_p = f"{k_bn}.{param}"
                    if k_p in sd_dst and k_p in sd_src:
                        s_len = sd_dst[k_p].shape[0]
                        sd_dst[k_p] = sd_src[k_p][:s_len]
            
            # Shortcut
            k_sc_w = f"{stage_name}.{b}.shortcut.0.weight"
            if k_sc_w in sd_dst and k_sc_w in sd_src:
                s_out, s_in = sd_dst[k_sc_w].shape[:2]
                sd_dst[k_sc_w] = sd_src[k_sc_w][:s_out, :s_in, :, :]
            k_sc_bn = f"{stage_name}.{b}.shortcut.1"
            for param in ["weight", "bias", "running_mean", "running_var"]:
                k_p = f"{k_sc_bn}.{param}"
                if k_p in sd_dst and k_p in sd_src:
                    s_len = sd_dst[k_p].shape[0]
                    sd_dst[k_p] = sd_src[k_p][:s_len]
                    
    sresnet_model.load_state_dict(sd_dst)
    return sresnet_model

print("Loading data...", flush=True)
X_tr_mel, y_tr_mel = load_split("fold_1_train", "log_mel")
X_va_mel, y_va_mel = load_split("fold_1_val", "log_mel")
X_te_mel, y_te_mel = load_split("fold_1_test", "log_mel")

sres = SResNet(in_channels=1, num_classes=50)
sres = init_sresnet_from_resnet18(sres)

class SResNetBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.avgpool = m.avgpool
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)

backbone = SResNetBackbone(sres)
backbone.eval()

def extract(bb, X):
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            feats.append(bb(X[i:i+32]))
    return torch.cat(feats)

print("Extracting features...", flush=True)
t0 = time.time()
F_tr = extract(backbone, X_tr_mel)
F_va = extract(backbone, X_va_mel)
F_te = extract(backbone, X_te_mel)
print(f"Features extracted in {time.time()-t0:.2f}s: shape {F_tr.shape}", flush=True)

# Train head
head = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.SiLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 50)
)
opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
crit = nn.CrossEntropyLoss(label_smoothing=0.05)
loader = DataLoader(TensorDataset(F_tr, y_tr_mel), batch_size=32, shuffle=True)

best_va = 0.0
best_head = None
for ep in range(60):
    head.train()
    for bx, by in loader:
        opt.zero_grad()
        loss = crit(head(bx), by)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        va_acc = (head(F_va).argmax(dim=1) == y_va_mel).float().mean().item() * 100.0
        if va_acc > best_va:
            best_va = va_acc
            best_head = {k: v.clone() for k, v in head.state_dict().items()}

head.load_state_dict(best_head)
head.eval()
with torch.no_grad():
    te_acc = (head(F_te).argmax(dim=1) == y_te_mel).float().mean().item() * 100.0

print(f"SResNet + Transfer Head Test Top-1 Accuracy: {te_acc:.2f}% (Best Val: {best_va:.2f}%)", flush=True)
