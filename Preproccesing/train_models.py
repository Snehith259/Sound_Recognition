"""
SoundSense — Fold-1 ESC-50 Neural Training Diagnostic
Trains and exports real PyTorch checkpoints for all 4 neural architectures:
1. ResNet-18: Pretrained Deep Residual Network adapted for 1-channel Log-Mel spectrograms
2. VGG-16: Pretrained Visual Geometry Group network with BatchNorm for Log-Mel spectrograms
3. SResNetFast: Compact convolutional MFCC classifier; it has no residual blocks
4. CNN-GRU: Hybrid Convolutional Recurrent Network for Log-Mel temporal dynamics

This legacy script uses only fold 1. Its metrics are diagnostic and are not CV
evidence. Use eval/eval_esc50_cv.py only with an independently trained checkpoint
for each official held-out fold before publishing model scores.

Outputs:
- checkpoints/esc50_resnet18_v1.pt
- checkpoints/esc50_vgg16_v1.pt
- checkpoints/esc50_sresnet_linear_v09.pt
- checkpoints/esc50_cnngru_recurrent_v08.pt
- checkpoints/model_report.json
- checkpoints/manifest.json
"""

import os
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchaudio.functional as AF
import torchvision.models as tv_models

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from Preproccesing.acoustic_inference import ESC50_CATEGORIES

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
FEATURES_DIR = ROOT_DIR / "Preproccesing" / "processed" / "features"
SPLITS_DIR = ROOT_DIR / "Preproccesing" / "splits"


# ══════════════════════════════════════════════════════════════
# Feature Loading
# ══════════════════════════════════════════════════════════════

def load_split_features(split_name: str, feat_type: str = "log_mel") -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads precomputed feature arrays for a given split."""
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


def extract_features_batched(backbone: nn.Module, X: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    """Safely extracts embeddings without CPU memory overflow."""
    backbone.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = X[i:i + batch_size]
            out = backbone(bx)
            feats.append(out)
    return torch.cat(feats, dim=0)


# ══════════════════════════════════════════════════════════════
# Model Architectures
# ══════════════════════════════════════════════════════════════

class ResNet18Transfer(nn.Module):
    """End-to-end ResNet-18 adapted for 1-channel Log-Mel spectrograms."""
    def __init__(self, num_classes: int = 50, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.resnet18(weights=weights)

        # 1-channel conv1
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
        feats = self.backbone(x)
        return self.head(feats)


class VGG16Transfer(nn.Module):
    """End-to-end VGG-16 with BatchNorm adapted for 1-channel Log-Mel spectrograms."""
    def __init__(self, num_classes: int = 50, pretrained: bool = True):
        super().__init__()
        weights = tv_models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.vgg16_bn(weights=weights)

        orig_conv = base.features[0].weight.data
        new_conv = orig_conv.mean(dim=1, keepdim=True)
        base.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        base.features[0].weight.data = new_conv

        self.backbone = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


class SResNetTransfer(nn.Module):
    """Upgraded Squeezed Residual Network for 1-channel Log-Mel spectrograms.
    Combines an ImageNet-pretrained Residual Backbone with a Squeezed Bottleneck Residual Head.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 50, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if (pretrained and hasattr(tv_models, "ResNet18_Weights")) else None
        base = tv_models.resnet18(weights=weights)

        orig_conv = base.conv1.weight.data
        new_conv = orig_conv.mean(dim=1, keepdim=True)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.conv1.weight.data = new_conv

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


# Backward-compatible alias
SResNetFast = SResNetTransfer


class CNNGRUFast(nn.Module):
    """Hybrid Convolutional Recurrent Network for Log-Mel spectral dynamics."""
    def __init__(self, num_classes: int = 50):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, None))  # (B, 64, 8, T')
        )
        self.gru = nn.GRU(64 * 8, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.conv(x)
        B, C, F_dim, T_dim = feat.shape
        feat = feat.permute(0, 3, 1, 2).contiguous().view(B, T_dim, C * F_dim)
        gru_out, _ = self.gru(feat)
        pooled = gru_out.mean(dim=1)
        return self.fc(pooled)


# ══════════════════════════════════════════════════════════════
# V2 Architectures (3-Channel Delta Spectrograms + Diverse Backbones)
# ══════════════════════════════════════════════════════════════

def log_mel_to_imagenet_3ch(x: torch.Tensor) -> torch.Tensor:
    """Convert (B, 1, 128, T) [0, 1] Log-Mel into balanced [Mel, Delta, Delta-Delta] ImageNet 3-ch input."""
    mel = x.squeeze(1)  # (B, 128, T)
    # Match dynamic range of temporal derivatives (~0.04-0.08 std) to ImageNet RGB std (~0.23)
    raw_d1 = AF.compute_deltas(mel)
    raw_d2 = AF.compute_deltas(raw_d1)
    delta1 = (raw_d1 * 4.5).clamp(-1.0, 1.0) * 0.5 + 0.5
    delta2 = (raw_d2 * 10.0).clamp(-1.0, 1.0) * 0.5 + 0.5
    rgb = torch.stack([mel, delta1, delta2], dim=1)  # (B, 3, 128, T)
    mean = rgb.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = rgb.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (rgb - mean) / std


def _require_weights(enum_cls: Any, pretrained: bool) -> Any:
    if not pretrained:
        return None
    if not hasattr(enum_cls, "IMAGENET1K_V1"):
        raise RuntimeError(f"Required pretrained weights IMAGENET1K_V1 unavailable on {enum_cls}")
    return enum_cls.IMAGENET1K_V1


class ResNet18V2(nn.Module):
    """3-channel ImageNet ResNet-18 adapted for [Log-Mel, Delta, Delta-Delta]."""
    def __init__(self, num_classes: int = 50, pretrained: bool = True):
        super().__init__()
        weights = _require_weights(tv_models.ResNet18_Weights, pretrained)
        base = tv_models.resnet18(weights=weights)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(log_mel_to_imagenet_3ch(x)))


class VGG16V2(nn.Module):
    """3-channel ImageNet VGG-16-BN adapted for [Log-Mel, Delta, Delta-Delta]."""
    def __init__(self, num_classes: int = 50, pretrained: bool = True):
        super().__init__()
        weights = _require_weights(tv_models.VGG16_BN_Weights, pretrained)
        base = tv_models.vgg16_bn(weights=weights)
        self.backbone = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(log_mel_to_imagenet_3ch(x)))


class SResNetV2(nn.Module):
    """ConvNeXt-Tiny backbone for the SResNet slot, eliminating ResNet-18 duplication."""
    def __init__(self, num_classes: int = 50, pretrained: bool = True):
        super().__init__()
        weights = _require_weights(tv_models.ConvNeXt_Tiny_Weights, pretrained)
        base = tv_models.convnext_tiny(weights=weights)
        in_features = base.classifier[2].in_features
        self.backbone = nn.Sequential(base.features, base.avgpool)
        self.head = nn.Sequential(
            base.classifier[0],  # LayerNorm2d
            base.classifier[1],  # Flatten
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(log_mel_to_imagenet_3ch(x)))


class _SEConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        reduced = max(8, out_ch // 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, reduced, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(reduced, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.pool = nn.MaxPool2d(pool) if pool is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out * self.se(out)
        out = F.gelu(out + self.skip(x))
        return self.pool(out)


class CNNGRUV2(nn.Module):
    """4-stage SE-Residual CNN + 2-layer BiGRU with Attentive Statistics Pooling."""
    def __init__(self, num_classes: int = 50, hidden_size: int = 192):
        super().__init__()
        self.conv = nn.Sequential(
            _SEConvBlock(3, 32, pool=(2, 2)),
            _SEConvBlock(32, 64, pool=(2, 2)),
            _SEConvBlock(64, 128, pool=(2, 2)),
            _SEConvBlock(128, 256, pool=(2, 1)),
            nn.AdaptiveAvgPool2d((4, None)),  # (B, 256, 4, T')
        )
        self.gru = nn.GRU(
            input_size=256 * 4,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )
        feat_dim = hidden_size * 2
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(log_mel_to_imagenet_3ch(x))
        B, C, F_dim, T_dim = feat.shape
        seq = feat.permute(0, 3, 1, 2).contiguous().view(B, T_dim, C * F_dim)
        gru_out, _ = self.gru(seq)
        w = torch.softmax(self.attn(gru_out), dim=1)
        mean = torch.sum(w * gru_out, dim=1)
        var = torch.sum(w * (gru_out - mean.unsqueeze(1)) ** 2, dim=1).clamp_min(1e-6)
        std = torch.sqrt(var)
        return self.fc(torch.cat([mean, std], dim=1))


# ══════════════════════════════════════════════════════════════
# V3 Architectures (Dual Avg+Max Pooling & Multi-Scale Stem)
# ══════════════════════════════════════════════════════════════

class DualPoolHead(nn.Module):
    """Dual Adaptive Avg+Max pooling classifier head for mixed and pure acoustic maps."""
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int = 50, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels * 2),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.65),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        mx = F.adaptive_max_pool2d(feat_map, 1).flatten(1)
        return self.net(torch.cat([avg, mx], dim=1))


class _StemTailBackboneV3(nn.Module):
    """Shared stem+tail forward pipeline supporting cached stem training and full E2E inference."""
    stem: nn.Module
    tail: nn.Module
    head: nn.Module

    def forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(log_mel_to_imagenet_3ch(x))

    def forward_tail(self, stem_feats: torch.Tensor) -> torch.Tensor:
        return self.head(self.tail(stem_feats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tail(self.forward_stem(x))


class ResNet18V3(_StemTailBackboneV3):
    """3-ch ImageNet ResNet-18 with Dual Avg+Max Pooling for overlapping/mixed audio."""
    def __init__(self, num_classes: int = 50, pretrained: bool = True):
        super().__init__()
        weights = _require_weights(tv_models.ResNet18_Weights, pretrained)
        base = tv_models.resnet18(weights=weights)
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2,
        )
        self.tail = nn.Sequential(base.layer3, base.layer4)
        self.head = DualPoolHead(512, 256, num_classes, dropout=0.30)


class _SpatialMaxHead(nn.Module):
    """Combines (2,2) spatial average quadrants with (1,1) global peak pooling."""
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int = 50, dropout: float = 0.35):
        super().__init__()
        in_dim = in_channels * 4 + in_channels
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        avg22 = F.adaptive_avg_pool2d(feat_map, (2, 2)).flatten(1)
        mx11 = F.adaptive_max_pool2d(feat_map, (1, 1)).flatten(1)
        return self.net(torch.cat([avg22, mx11], dim=1))


class VGG16V3(_StemTailBackboneV3):
    """3-ch ImageNet VGG16-BN with multi-quadrant avg + global peak pooling."""
    def __init__(self, num_classes: int = 50, pretrained: bool = True):
        super().__init__()
        weights = _require_weights(tv_models.VGG16_BN_Weights, pretrained)
        base = tv_models.vgg16_bn(weights=weights)
        self.stem = nn.Sequential(*list(base.features.children())[:34])
        self.tail = nn.Sequential(*list(base.features.children())[34:])
        self.head = _SpatialMaxHead(512, 512, num_classes, dropout=0.35)


class SResNetV3(_StemTailBackboneV3):
    """3-ch ImageNet ConvNeXt-Tiny backbone with Dual Avg+Max Pooling."""
    def __init__(self, num_classes: int = 50, pretrained: bool = True):
        super().__init__()
        weights = _require_weights(tv_models.ConvNeXt_Tiny_Weights, pretrained)
        base = tv_models.convnext_tiny(weights=weights)
        in_features = base.classifier[2].in_features
        self.stem = nn.Sequential(*list(base.features.children())[:6])
        self.tail = nn.Sequential(*list(base.features.children())[6:])
        self.head = DualPoolHead(in_features, 256, num_classes, dropout=0.25)


class _CNNGRUSequenceTail(nn.Module):
    """EfficientNet MBConv tail + BiGRU temporal aggregator for stem-cached & E2E training."""
    def __init__(self, cnn_tail: nn.Module, cnn_channels: int = 1280, hidden_size: int = 160):
        super().__init__()
        self.cnn_tail = cnn_tail
        self.seq_proj = nn.Sequential(
            nn.Linear(cnn_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )
        feat_dim = hidden_size * 2
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.peak_proj = nn.Sequential(
            nn.Linear(cnn_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.out_dim = feat_dim * 2 + 256

    def forward(self, stem_feats: torch.Tensor) -> torch.Tensor:
        feat = self.cnn_tail(stem_feats)  # (B, 1280, 4, 7)
        cnn_peak = self.peak_proj(F.adaptive_max_pool2d(feat, 1).flatten(1))
        seq_raw = F.adaptive_avg_pool2d(feat, (1, feat.shape[-1])).squeeze(2).transpose(1, 2)  # (B, T, 1280)
        seq = self.seq_proj(seq_raw)
        gru_out, _ = self.gru(seq)
        w = torch.softmax(self.attn(gru_out), dim=1)
        mean = torch.sum(w * gru_out, dim=1)
        var = torch.sum(w * (gru_out - mean.unsqueeze(1)) ** 2, dim=1).clamp_min(1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std, cnn_peak], dim=1)


class CNNGRUV3(_StemTailBackboneV3):
    """Pretrained EfficientNet-B0 Spectral CNN + 2-layer BiGRU + Attentive Stats & Peak Pooling."""
    def __init__(self, num_classes: int = 50, hidden_size: int = 160, pretrained: bool = False):
        super().__init__()
        weights = _require_weights(tv_models.EfficientNet_B0_Weights, pretrained)
        base = tv_models.efficientnet_b0(weights=weights)
        self.stem = nn.Sequential(*list(base.features.children())[:5])  # (B, 80, 8, 14)
        cnn_tail = nn.Sequential(*list(base.features.children())[5:])
        self.tail = _CNNGRUSequenceTail(cnn_tail, cnn_channels=1280, hidden_size=hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(self.tail.out_dim),
            nn.Dropout(0.30),
            nn.Linear(self.tail.out_dim, 256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_classes),
        )


# ══════════════════════════════════════════════════════════════
# V4 Architectures (AudioSet-2M Pretrained Audio Backbones @ 32 kHz)
# ══════════════════════════════════════════════════════════════

class _AudioHeadV4(nn.Module):
    """Compact residual + bottleneck classifier head on AudioSet features."""
    def __init__(self, in_dim: int, hidden_dim: int = 320, num_classes: int = 50, dropout: float = 0.35):
        super().__init__()
        self.skip = nn.Linear(in_dim, num_classes)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.45 * self.skip(x) + 0.55 * self.mlp(x)


class _PannsAudioBaseV4(nn.Module):
    """Shared wrapper for PANNs ResNet22 and Cnn14 AudioSet-2M backbones."""
    backbone_cls: type
    repo_id: str

    def __init__(self, num_classes: int = 50, pretrained: bool = False):
        super().__init__()
        from Preproccesing.audioset_backbones import load_pretrained_panns
        self.backbone = self.backbone_cls()
        if pretrained:
            load_pretrained_panns(self.repo_id, self.backbone)
        in_dim = self.backbone.emb_dim + 527
        self.head = _AudioHeadV4(in_dim=in_dim, hidden_dim=320, num_classes=num_classes, dropout=0.38)

    def extract_frozen_features(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e_g, e_h1, e_h2, as_logits = self.backbone.extract_multi(wav)
        # Compact sub-half blend captures localized bursts inside mixtures without inflating feature dim
        loc = torch.max(F.normalize(e_h1, dim=-1), F.normalize(e_h2, dim=-1))
        emb = 0.65 * F.normalize(e_g, dim=-1) + 0.35 * loc
        feat = torch.cat([emb, torch.sigmoid(as_logits)], dim=-1)
        return feat, as_logits

    def forward_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)

    def forward_with_audioset(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat, as_logits = self.extract_frozen_features(wav)
        return self.head(feat), as_logits

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.forward_with_audioset(wav)[0]


class ResNet22AudioV4(_PannsAudioBaseV4):
    """PANNs ResNet22 AudioSet-2M backbone + localized mix head (`resnet18` v4 slot)."""
    def __init__(self, num_classes: int = 50, pretrained: bool = False):
        from Preproccesing.audioset_backbones import PannsResNet22Backbone
        self.backbone_cls = PannsResNet22Backbone
        self.repo_id = "nicofarr/panns_ResNet22"
        super().__init__(num_classes=num_classes, pretrained=pretrained)


class Cnn10AudioV4(_PannsAudioBaseV4):
    """PANNs Cnn10 AudioSet-2M VGG-family double-conv backbone (`vgg16` v4 slot)."""
    def __init__(self, num_classes: int = 50, pretrained: bool = False):
        from Preproccesing.audioset_backbones import PannsCnn10Backbone
        self.backbone_cls = PannsCnn10Backbone
        self.repo_id = "nicofarr/panns_Cnn10"
        super().__init__(num_classes=num_classes, pretrained=pretrained)


class Cnn14AudioV4(_PannsAudioBaseV4):
    """PANNs Cnn14 14-conv AudioSet-2M VGG-family backbone (`vgg16` v4 slot)."""
    def __init__(self, num_classes: int = 50, pretrained: bool = False):
        from Preproccesing.audioset_backbones import PannsCnn14Backbone
        self.backbone_cls = PannsCnn14Backbone
        self.repo_id = "nicofarr/panns_Cnn14"
        super().__init__(num_classes=num_classes, pretrained=pretrained)


class DyMNAudioV4(nn.Module):
    """EfficientAT DyMN Dynamic Squeezed Residual Net + sub-half localization (`sresnet` v4 slot)."""
    def __init__(self, num_classes: int = 50, pretrained: bool = False):
        super().__init__()
        from Preproccesing.audioset_backbones import (
            AudioDyMNBackbone,
            MelSTFT32k,
            load_pretrained_efficientat,
        )
        self.mel = MelSTFT32k()
        self.backbone = AudioDyMNBackbone(width_mult=1.0, num_classes=527)
        if pretrained:
            load_pretrained_efficientat(
                "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/dymn10_as.pt",
                self.backbone,
            )
        in_dim = self.backbone.out_channels * 3 + 527
        self.head = _AudioHeadV4(in_dim=in_dim, hidden_dim=320, num_classes=num_classes, dropout=0.35)

    def extract_frozen_features(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = self.mel(wav, augment=False)
        e_g, e_h1, e_h2, as_logits = self.backbone.extract_multi(spec)
        feat = torch.cat(
            [F.normalize(e_g, dim=-1), F.normalize(e_h1, dim=-1), F.normalize(e_h2, dim=-1), torch.sigmoid(as_logits)],
            dim=-1,
        )
        return feat, as_logits

    def forward_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)

    def forward_with_audioset(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat, as_logits = self.extract_frozen_features(wav)
        return self.head(feat), as_logits

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.forward_with_audioset(wav)[0]


class _MN20GRUHead(nn.Module):
    """2-layer BiGRU + Attentive Stats & Peak Pooling over MobileNetV3-AudioSet conv frames."""
    def __init__(self, seq_dim: int, clip_dim: int, num_classes: int = 50, hidden_size: int = 160):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(seq_dim, 256), nn.LayerNorm(256), nn.GELU())
        self.gru = nn.GRU(256, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.20)
        feat_dim = hidden_size * 2
        self.attn = nn.Linear(feat_dim, 1)
        self.skip = nn.Linear(clip_dim, num_classes)
        self.mlp = nn.Sequential(
            nn.LayerNorm(clip_dim + feat_dim * 2),
            nn.Dropout(0.35),
            nn.Linear(clip_dim + feat_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_classes),
        )

    def forward(self, clip_feat: torch.Tensor, seq_feat: torch.Tensor) -> torch.Tensor:
        g, _ = self.gru(self.proj(seq_feat))
        w = torch.softmax(self.attn(g), dim=1)
        mean = torch.sum(w * g, dim=1)
        mx = torch.max(g, dim=1)[0]
        return 0.45 * self.skip(clip_feat) + 0.55 * self.mlp(torch.cat([clip_feat, mean, mx], dim=-1))


class MN20GRUAudioV4(nn.Module):
    """EfficientAT MN2.0 AudioSet-2M CNN + 2-layer BiGRU temporal aggregator (`cnngru` v4 slot)."""
    def __init__(self, num_classes: int = 50, pretrained: bool = False):
        super().__init__()
        from Preproccesing.audioset_backbones import (
            AudioMobileNetBackbone,
            MelSTFT32k,
            load_pretrained_efficientat,
        )
        self.mel = MelSTFT32k()
        self.backbone = AudioMobileNetBackbone(width_mult=2.0, num_classes=527)
        if pretrained:
            load_pretrained_efficientat(
                "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn20_as_mAP_478.pt",
                self.backbone,
            )
        seq_dim = self.backbone.out_channels
        clip_dim = seq_dim + 527
        self.head = _MN20GRUHead(seq_dim=seq_dim, clip_dim=clip_dim, num_classes=num_classes)

    def extract_frozen_features(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = self.mel(wav, augment=False)
        emb, as_logits, t_seq = self.backbone.extract_sequence_and_global(spec)
        clip_feat = torch.cat([F.normalize(emb, dim=-1), torch.sigmoid(as_logits)], dim=-1)
        return clip_feat, t_seq, as_logits

    def forward_from_features(self, clip_feat: torch.Tensor, seq_feat: torch.Tensor) -> torch.Tensor:
        return self.head(clip_feat, seq_feat)

    def forward_with_audioset(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_feat, t_seq, as_logits = self.extract_frozen_features(wav)
        return self.head(clip_feat, t_seq), as_logits

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.forward_with_audioset(wav)[0]


# ══════════════════════════════════════════════════════════════
# Metrics & Temperature Scaling
# ══════════════════════════════════════════════════════════════

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        _, preds = torch.max(probs, 1)

        correct1 = (preds == targets).sum().item()
        top1_acc = round(correct1 / len(targets) * 100.0, 2)

        _, top5_preds = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)
        correct5 = (top5_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
        top5_acc = round(correct5 / len(targets) * 100.0, 2)

        num_classes = logits.shape[1]
        f1_scores = []
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().item()
            fp = ((preds == c) & (targets != c)).sum().item()
            fn = ((preds != c) & (targets == c)).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)
        macro_f1 = round(float(np.mean(f1_scores)) * 100.0, 2)

    return {"top1": top1_acc, "top5": top5_acc, "macroF1": macro_f1}


def calibrate_temperature(val_logits: torch.Tensor, val_targets: torch.Tensor) -> float:
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    logits = val_logits.detach().double().cpu()
    targets = val_targets.detach().long().cpu()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80)

    def eval_loss():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return round(float(log_temperature.detach().exp().clamp(0.05, 20.0).item()), 4)


# ══════════════════════════════════════════════════════════════
# High-Speed Multi-Model Trainer
# ══════════════════════════════════════════════════════════════

def train_and_export_all():
    print(f"\n{'='*70}\n SOUNDSENSE MULTI-MODEL ESC-50 TRAINING & EXPORT PIPELINE\n{'='*70}", flush=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("\n[Step 1/5] Loading preprocessed features into memory...", flush=True)
    t_data = time.time()
    X_tr_mel, y_tr_mel = load_split_features("fold_1_train", "log_mel")
    X_va_mel, y_va_mel = load_split_features("fold_1_val", "log_mel")
    X_te_mel, y_te_mel = load_split_features("fold_1_test", "log_mel")

    X_tr_mfcc, y_tr_mfcc = load_split_features("fold_1_train", "mfcc")
    X_va_mfcc, y_va_mfcc = load_split_features("fold_1_val", "mfcc")
    X_te_mfcc, y_te_mfcc = load_split_features("fold_1_test", "mfcc")
    print(f"Features ready in {time.time()-t_data:.2f}s: Mel train {X_tr_mel.shape}, MFCC train {X_tr_mfcc.shape}", flush=True)

    results = {}

    # ══════════════════════════════════════════════════════════
    # Model 1: ResNet-18
    # ══════════════════════════════════════════════════════════
    print("\n[Step 2/5] Training ResNet-18 (Log-Mel Spectrogram)...", flush=True)
    t0 = time.time()
    rn_model = ResNet18Transfer(num_classes=50)

    # Extract features via backbone in batches
    F_tr = extract_features_batched(rn_model.backbone, X_tr_mel)
    F_va = extract_features_batched(rn_model.backbone, X_va_mel)
    F_te = extract_features_batched(rn_model.backbone, X_te_mel)

    opt = torch.optim.AdamW(rn_model.head.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(TensorDataset(F_tr, y_tr_mel), batch_size=32, shuffle=True)

    best_va_acc = 0.0
    best_head = None
    for ep in range(40):
        rn_model.head.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(rn_model.head(bx), by)
            loss.backward()
            opt.step()
        rn_model.head.eval()
        with torch.no_grad():
            va_preds = rn_model.head(F_va).argmax(dim=1)
            va_acc = (va_preds == y_va_mel).float().mean().item() * 100.0
            if va_acc > best_va_acc:
                best_va_acc = va_acc
                best_head = {k: v.clone() for k, v in rn_model.head.state_dict().items()}

    rn_model.head.load_state_dict(best_head)
    rn_model.eval()

    with torch.no_grad():
        t_lat = time.time()
        te_logits = rn_model(X_te_mel[:32])
        lat_ms = round((time.time() - t_lat) / 32 * 1000, 2)

        te_logits_full = rn_model.head(F_te)
        va_logits_full = rn_model.head(F_va)

    metrics = compute_metrics(te_logits_full, y_te_mel)
    temp = calibrate_temperature(va_logits_full, y_va_mel)
    elapsed = round(time.time() - t0, 1)

    ckpt_path = CHECKPOINTS_DIR / "esc50_resnet18_v1.pt"
    param_count = sum(p.numel() for p in rn_model.parameters())
    torch.save({
        "format": "soundsense-v2",
        "model_name": "resnet18",
        "feature_type": "log_mel",
        "num_classes": 50,
        "temperature": temp,
        "parameters": param_count,
    "fold1_diagnostic_metrics": {**metrics, "latencyMs": lat_ms, "trainSeconds": elapsed},
        "state_dict": rn_model.state_dict()
    }, ckpt_path)
    print(f"  -> ResNet-18 done in {elapsed}s | Test Top-1: {metrics['top1']}% | Top-5: {metrics['top5']}% | Saved: {ckpt_path.name} ({ckpt_path.stat().st_size/1e6:.1f} MB)", flush=True)
    results["resnet18"] = {**metrics, "latencyMs": lat_ms, "sizeMb": round(ckpt_path.stat().st_size/1e6, 2), "paramsM": round(param_count/1e6, 2)}

    # ══════════════════════════════════════════════════════════
    # Model 2: VGG-16
    # ══════════════════════════════════════════════════════════
    print("\n[Step 3/5] Training VGG-16 (Log-Mel Spectrogram)...", flush=True)
    t0 = time.time()
    vgg_model = VGG16Transfer(num_classes=50)

    F_tr_vgg = extract_features_batched(vgg_model.backbone, X_tr_mel)
    F_va_vgg = extract_features_batched(vgg_model.backbone, X_va_mel)
    F_te_vgg = extract_features_batched(vgg_model.backbone, X_te_mel)

    opt = torch.optim.AdamW(vgg_model.head.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(TensorDataset(F_tr_vgg, y_tr_mel), batch_size=32, shuffle=True)

    best_va_acc = 0.0
    best_head_vgg = None
    for ep in range(35):
        vgg_model.head.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(vgg_model.head(bx), by)
            loss.backward()
            opt.step()
        vgg_model.head.eval()
        with torch.no_grad():
            va_preds = vgg_model.head(F_va_vgg).argmax(dim=1)
            va_acc = (va_preds == y_va_mel).float().mean().item() * 100.0
            if va_acc > best_va_acc:
                best_va_acc = va_acc
                best_head_vgg = {k: v.clone() for k, v in vgg_model.head.state_dict().items()}

    vgg_model.head.load_state_dict(best_head_vgg)
    vgg_model.eval()

    with torch.no_grad():
        t_lat = time.time()
        _ = vgg_model(X_te_mel[:32])
        lat_ms = round((time.time() - t_lat) / 32 * 1000, 2)
        te_logits_full = vgg_model.head(F_te_vgg)
        va_logits_full = vgg_model.head(F_va_vgg)

    metrics = compute_metrics(te_logits_full, y_te_mel)
    temp = calibrate_temperature(va_logits_full, y_va_mel)
    elapsed = round(time.time() - t0, 1)

    ckpt_path = CHECKPOINTS_DIR / "esc50_vgg16_v1.pt"
    param_count = sum(p.numel() for p in vgg_model.parameters())
    torch.save({
        "format": "soundsense-v2",
        "model_name": "vgg16",
        "feature_type": "log_mel",
        "num_classes": 50,
        "temperature": temp,
        "parameters": param_count,
        "fold1_diagnostic_metrics": {**metrics, "latencyMs": lat_ms, "trainSeconds": elapsed},
        "state_dict": vgg_model.state_dict()
    }, ckpt_path)
    print(f"  -> VGG-16 done in {elapsed}s | Test Top-1: {metrics['top1']}% | Top-5: {metrics['top5']}% | Saved: {ckpt_path.name} ({ckpt_path.stat().st_size/1e6:.1f} MB)", flush=True)
    results["vgg16"] = {**metrics, "latencyMs": lat_ms, "sizeMb": round(ckpt_path.stat().st_size/1e6, 2), "paramsM": round(param_count/1e6, 2)}

    # ══════════════════════════════════════════════════════════
    # Model 3: SResNet (Pretrained Squeezed Residual Network)
    # ══════════════════════════════════════════════════════════
    print("\n[Step 4/5] Training SResNet (Pretrained Squeezed Residual Log-Mel)...", flush=True)
    t0 = time.time()
    sr_model = SResNetTransfer(num_classes=50, pretrained=True)

    F_tr_sr = extract_features_batched(sr_model.backbone, X_tr_mel)
    F_va_sr = extract_features_batched(sr_model.backbone, X_va_mel)
    F_te_sr = extract_features_batched(sr_model.backbone, X_te_mel)

    opt = torch.optim.AdamW(sr_model.head.parameters(), lr=2e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.08)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    loader = DataLoader(TensorDataset(F_tr_sr, y_tr_mel), batch_size=32, shuffle=True)

    best_va_acc = 0.0
    best_head = None
    for ep in range(50):
        sr_model.head.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(sr_model.head(bx), by)
            loss.backward()
            opt.step()
        sched.step()
        sr_model.head.eval()
        with torch.no_grad():
            va_preds = sr_model.head(F_va_sr).argmax(dim=1)
            va_acc = (va_preds == y_va_mel).float().mean().item() * 100.0
            if va_acc > best_va_acc:
                best_va_acc = va_acc
                best_head = {k: v.clone() for k, v in sr_model.head.state_dict().items()}

    sr_model.head.load_state_dict(best_head)
    sr_model.eval()

    with torch.no_grad():
        t_lat = time.time()
        te_logits = sr_model.head(F_te_sr)
        lat_ms = round((time.time() - t_lat) / len(F_te_sr) * 1000, 2)
        va_logits = sr_model.head(F_va_sr)

    metrics = compute_metrics(te_logits, y_te_mel)
    temp = calibrate_temperature(va_logits, y_va_mel)
    elapsed = round(time.time() - t0, 1)

    ckpt_path = CHECKPOINTS_DIR / "esc50_sresnet_linear_v09.pt"
    param_count = sum(p.numel() for p in sr_model.parameters())
    torch.save({
        "format": "soundsense-v2",
        "model_name": "sresnet",
        "feature_type": "log_mel",
        "num_classes": 50,
        "temperature": temp,
        "parameters": param_count,
        "fold1_diagnostic_metrics": {**metrics, "latencyMs": lat_ms, "trainSeconds": elapsed},
        "state_dict": sr_model.state_dict()
    }, ckpt_path)
    print(f"  -> SResNet done in {elapsed}s | Test Top-1: {metrics['top1']}% | Top-5: {metrics['top5']}% | Saved: {ckpt_path.name} ({ckpt_path.stat().st_size/1e6:.1f} MB)", flush=True)
    results["sresnet"] = {**metrics, "latencyMs": lat_ms, "sizeMb": round(ckpt_path.stat().st_size/1e6, 2), "paramsM": round(param_count/1e6, 2)}

    # ══════════════════════════════════════════════════════════
    # Model 4: CNN-GRU (Log-Mel Dynamics)
    # ══════════════════════════════════════════════════════════
    print("\n[Step 5/5] Training CNN-GRU (Log-Mel Recurrent)...", flush=True)
    t0 = time.time()
    cg_model = CNNGRUFast(num_classes=50)
    opt = torch.optim.AdamW(cg_model.parameters(), lr=2e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(TensorDataset(X_tr_mel, y_tr_mel), batch_size=32, shuffle=True)

    best_va_acc = 0.0
    best_state = None
    for ep in range(30):
        cg_model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(cg_model(bx), by)
            loss.backward()
            opt.step()
        cg_model.eval()
        with torch.no_grad():
            va_preds = cg_model(X_va_mel).argmax(dim=1)
            va_acc = (va_preds == y_va_mel).float().mean().item() * 100.0
            if va_acc > best_va_acc:
                best_va_acc = va_acc
                best_state = {k: v.clone() for k, v in cg_model.state_dict().items()}

    cg_model.load_state_dict(best_state)
    cg_model.eval()

    with torch.no_grad():
        t_lat = time.time()
        te_logits = cg_model(X_te_mel)
        lat_ms = round((time.time() - t_lat) / len(X_te_mel) * 1000, 2)
        va_logits = cg_model(X_va_mel)

    metrics = compute_metrics(te_logits, y_te_mel)
    temp = calibrate_temperature(va_logits, y_va_mel)
    elapsed = round(time.time() - t0, 1)

    ckpt_path = CHECKPOINTS_DIR / "esc50_cnngru_recurrent_v08.pt"
    param_count = sum(p.numel() for p in cg_model.parameters())
    torch.save({
        "format": "soundsense-v2",
        "model_name": "cnngru",
        "feature_type": "log_mel",
        "num_classes": 50,
        "temperature": temp,
        "parameters": param_count,
        "fold1_diagnostic_metrics": {**metrics, "latencyMs": lat_ms, "trainSeconds": elapsed},
        "state_dict": cg_model.state_dict()
    }, ckpt_path)
    print(f"  -> CNN-GRU done in {elapsed}s | Test Top-1: {metrics['top1']}% | Top-5: {metrics['top5']}% | Saved: {ckpt_path.name} ({ckpt_path.stat().st_size/1e6:.1f} MB)", flush=True)
    results["cnngru"] = {**metrics, "latencyMs": lat_ms, "sizeMb": round(ckpt_path.stat().st_size/1e6, 2), "paramsM": round(param_count/1e6, 2)}

    # ══════════════════════════════════════════════════════════
    # Mark this as fold-1 diagnostic output; only eval_report.json feeds the UI.
    # ══════════════════════════════════════════════════════════
    report_path = CHECKPOINTS_DIR / "model_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.time(),
            "diagnosticOnly": True,
            "models": results,
            "status": "fold1_diagnostic_not_cross_validation",
            "note": "Run the official five-fold CV and external harnesses before publishing model scores.",
        }, f, indent=2)
    print(f"\nWrote fold-1 diagnostic report to: {report_path.name}", flush=True)

    inventory = {}
    for model_name, entry in {
        "resnet18": ("esc50_resnet18_v1.pt", "resnet18"),
        "vgg16": ("esc50_vgg16_v1.pt", "vgg16"),
        "sresnet": ("esc50_sresnet_linear_v09.pt", "sresnet"),
        "cnngru": ("esc50_cnngru_recurrent_v08.pt", "cnngru"),
    }.items():
        filename, _ = entry
        path = CHECKPOINTS_DIR / filename
        inventory[model_name] = {
            "filename": filename,
            "sizeBytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "trainingEvidence": "fold1_diagnostic_only",
        }
    manifest_path = CHECKPOINTS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"schemaVersion": 1, "status": "checkpoint_inventory_only", "models": inventory}, f, indent=2)
    print(f"Updated checkpoint inventory at: {manifest_path.name}", flush=True)
    print("Four checkpoint files were exported. Run five-fold CV and external evaluations before publishing scores.", flush=True)


if __name__ == "__main__":
    train_and_export_all()
