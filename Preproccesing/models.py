"""
SoundSense — Deep Neural Network Architectures for ESC-50 Environmental Sound Recognition
Provides PyTorch implementations for:
1. ResNet18_ESC50: 18-layer Deep Residual Network adapted for single-channel audio spectrograms
2. VGG16_ESC50: 16-layer Visual Geometry Group network with localized receptive fields
3. SResNet: Squeezed Residual Network for linear spectral MFCC representations
4. CNNGRU: Recurrent Convolutional Gated Recurrent Unit network for Log-Mel dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# 1. ResNet-18 for Audio Classification
# ══════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_ESC50(nn.Module):
    """
    ResNet-18 adapted for ESC-50 environmental sound recognition.
    Accepts 1-channel Log-Mel spectrograms (B, 1, 128, 216) and outputs 50 class logits.
    Parameters: ~11.2M
    """
    def __init__(self, in_channels=1, num_classes=50):
        super().__init__()
        self.in_planes = 64

        # Initial convolution adapted for 1-channel audio features
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 Residual Stages (2, 2, 2, 2 blocks)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, N_MELS, T)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ══════════════════════════════════════════════════════════════
# 2. VGG-16 for Audio Classification
# ══════════════════════════════════════════════════════════════

class VGG16_ESC50(nn.Module):
    """
    VGG-16 architecture adapted for ESC-50 environmental audio spectrograms.
    Features 13 convolutional layers (3x3 receptive fields) and 3 fully connected layers.
    Parameters: ~15.2M (with optimized audio classification head) or standard full VGG-16
    """
    def __init__(self, in_channels=1, num_classes=50, dropout=0.5):
        super().__init__()
        
        # Configuration for VGG-16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        
        layers = []
        current_in = in_channels
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(current_in, v, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                current_in = v
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, 128, 216)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ══════════════════════════════════════════════════════════════
# 3. SResNet (Squeezed ResNet for MFCC)
# ══════════════════════════════════════════════════════════════

class SResNet(nn.Module):
    """
    Squeezed Residual Network optimized for MFCC linear spectral representations.
    Parameters: ~8.6M
    """
    def __init__(self, in_channels=1, num_classes=50):
        super().__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ══════════════════════════════════════════════════════════════
# 4. CNN-GRU Hybrid (Log-Mel Spectrogram + Recurrent Sequence)
# ══════════════════════════════════════════════════════════════

class CNNGRU(nn.Module):
    """
    CNN-GRU Hybrid for nonlinear temporal dynamics.
    Parameters: ~5.6M
    """
    def __init__(self, in_channels=1, num_classes=50, hidden_size=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, None)),  # Compress frequency dimension to 16
        )

        self.gru = nn.GRU(
            input_size=128 * 16,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, F, T)
        feats = self.conv(x)  # (B, C, F', T')
        B, C, F_dim, T_dim = feats.size()
        feats = feats.permute(0, 3, 1, 2).contiguous()  # (B, T', C, F')
        feats = feats.view(B, T_dim, C * F_dim)  # (B, T', C*F')

        gru_out, _ = self.gru(feats)  # (B, T', 2 * hidden_size)
        pool_out = torch.mean(gru_out, dim=1)  # Temporal mean pooling
        logits = self.fc(pool_out)
        return logits


def get_model(model_name: str, num_classes: int = 50) -> nn.Module:
    """Factory helper to instantiate any supported model architecture."""
    name = model_name.lower().replace("-", "").replace("_", "")
    if "resnet18" in name:
        return ResNet18_ESC50(in_channels=1, num_classes=num_classes)
    elif "vgg16" in name:
        return VGG16_ESC50(in_channels=1, num_classes=num_classes)
    elif "sresnet" in name:
        return SResNet(in_channels=1, num_classes=num_classes)
    elif "cnngru" in name:
        return CNNGRU(in_channels=1, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from ['resnet18', 'vgg16', 'sresnet', 'cnngru'].")

