"""Self-contained AudioSet-2M pretrained audio backbones for SoundSense V4 models.

Contains standalone PyTorch implementations of:
  - PANNs ResNet22 & Cnn10 / Cnn14 (Kong et al., IEEE/ACM TASLP 2020)
  - EfficientAT MobileNetV3 (MN) & Dynamic MobileNet (DyMN) (Schmid et al., ICASSP 2023)
  - 32 kHz Kaldi-compatible Log-Mel STFT frontend

Weights are bundled into trained deployment checkpoints so runtime inference
requires zero internet downloads or external hub repositories.
"""
from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchvision.ops import Conv2dNormActivation

ROOT_DIR = Path(__file__).resolve().parent.parent
AUDIOSET_CACHE_DIR = ROOT_DIR / "checkpoints" / ".cache" / "audioset"

# Static AudioSet-527 ontology indices mapped to ESC-50 concepts vs clear Non-ESC-50 OOD concepts
# (Speech, Singing, Musical Instruments, Weapons/Explosions, Synthetic Tones/Noise - never tuned on eval sets)
ESC50_AUDIOSET_INDICES: Tuple[int, ...] = (
    9, 16, 17, 18, 20, 21, 23, 24, 41, 43, 45, 47, 48, 49, 53, 54, 63, 67, 69, 74,
    75, 78, 79, 80, 81, 82, 83, 84, 85, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101,
    107, 109, 111, 112, 113, 114, 115, 117, 118, 121, 122, 126, 127, 128, 129, 130,
    131, 132, 133, 200, 201, 202, 203, 206, 283, 286, 287, 289, 290, 291, 292, 293,
    294, 295, 298, 299, 303, 304, 306, 308, 309, 311, 318, 319, 322, 323, 324, 325,
    326, 329, 330, 331, 332, 333, 335, 336, 337, 338, 339, 340, 343, 344, 347, 348,
    349, 350, 351, 352, 355, 359, 360, 361, 368, 370, 374, 375, 376, 377, 384, 385,
    386, 390, 395, 396, 397, 398, 400, 401, 406, 407, 408, 421, 432, 433, 441, 443,
    448, 449, 450, 451, 486, 491, 492,
)
OOD_AUDIOSET_INDICES: Tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 7, 27, 28, 32, 33, 34, 35, 37, 70, 137, 138, 140, 141, 142,
    143, 144, 145, 152, 153, 154, 155, 156, 157, 161, 162, 163, 164, 165, 167, 168,
    178, 179, 186, 187, 191, 192, 195, 196, 197, 214, 216, 217, 219, 227, 230, 233,
    234, 235, 237, 239, 240, 243, 245, 246, 247, 248, 249, 252, 253, 254, 256, 258,
    259, 260, 261, 262, 264, 265, 267, 268, 269, 270, 272, 273, 274, 275, 276, 277,
    278, 279, 280, 281, 282, 327, 402, 403, 426, 427, 428, 430, 460, 481, 496, 501,
    502, 503, 505, 513, 514, 515, 516, 517, 518, 520, 521,
)


def audioset_ood_probability(as_logits: torch.Tensor | np.ndarray) -> np.ndarray:
    """Estimate non-ESC-50 out-of-domain dominance from pretrained AudioSet-527 logits."""
    if isinstance(as_logits, torch.Tensor):
        probs = torch.sigmoid(as_logits).detach().cpu().numpy()
    else:
        probs = 1.0 / (1.0 + np.exp(-np.clip(as_logits, -30.0, 30.0)))
    in_max = np.max(probs[..., ESC50_AUDIOSET_INDICES], axis=-1)
    ood_max = np.max(probs[..., OOD_AUDIOSET_INDICES], axis=-1)
    return (ood_max / (in_max + ood_max + 1e-6)).astype(np.float32)


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _cnn_out_size(in_size: int, padding: int, dilation: int, kernel: int, stride: int) -> int:
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


# ══════════════════════════════════════════════════════════════
# PANNs (Kong et al.) Blocks
# ══════════════════════════════════════════════════════════════

class PannsConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, pool_size: Tuple[int, int] = (2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class PannsSTFT(nn.Module):
    def __init__(self, n_fft: int = 1024, hop: int = 320):
        super().__init__()
        self.conv_real = nn.Conv1d(1, n_fft // 2 + 1, n_fft, stride=hop, padding=0, bias=False)
        self.conv_imag = nn.Conv1d(1, n_fft // 2 + 1, n_fft, stride=hop, padding=0, bias=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = F.pad(wav.unsqueeze(1), (512, 512), mode="reflect")
        r = self.conv_real(x)
        im = self.conv_imag(x)
        return (r * r + im * im).clamp_min(1e-10).transpose(1, 2)


class _PannsBasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.avg_pool2d(x, kernel_size=(2, 2)) if self.stride == 2 else x
        out = F.relu_(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu_(out + identity)


class _PannsResNetLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride) if stride != 1 else nn.Identity(),
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [_PannsBasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_PannsBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class PannsResNet22Backbone(nn.Module):
    """PANNs AudioSet-2M ResNet22 backbone (outputs temporal-pooled & half-localized features)."""
    emb_dim = 2048

    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = nn.Module()
        self.spectrogram_extractor.stft = PannsSTFT()
        self.logmel_extractor = nn.Module()
        self.logmel_extractor.register_buffer("melW", torch.zeros(513, 64))
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = PannsConvBlock(1, 64)
        self.resnet = _PannsResNetLayers()
        self.conv_block_after1 = PannsConvBlock(512, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, 527, bias=True)

    def extract_multi(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.matmul(self.spectrogram_extractor.stft(wav), self.logmel_extractor.melW)
        x = (10.0 * torch.log10(x.clamp_min(1e-10))).clamp_min(-100.0).unsqueeze(1)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        x = self.conv_block1(x, (2, 2))
        x = F.avg_pool2d(self.resnet(x), (2, 2))
        x = self.conv_block_after1(x, (1, 1))
        xm = torch.mean(x, dim=3)  # (B, 2048, T)
        half = max(1, xm.shape[2] // 2)
        e_g = F.relu_(self.fc1(torch.max(xm, 2)[0] + torch.mean(xm, 2)))
        e_h1 = F.relu_(self.fc1(torch.max(xm[:, :, :half], 2)[0] + torch.mean(xm[:, :, :half], 2)))
        e_h2 = F.relu_(self.fc1(torch.max(xm[:, :, half:], 2)[0] + torch.mean(xm[:, :, half:], 2)))
        return e_g, e_h1, e_h2, self.fc_audioset(e_g)


class PannsCnn10Backbone(nn.Module):
    """PANNs AudioSet-2M Cnn10 VGG-style double-conv backbone."""
    emb_dim = 512

    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = nn.Module()
        self.spectrogram_extractor.stft = PannsSTFT()
        self.logmel_extractor = nn.Module()
        self.logmel_extractor.register_buffer("melW", torch.zeros(513, 64))
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = PannsConvBlock(1, 64)
        self.conv_block2 = PannsConvBlock(64, 128)
        self.conv_block3 = PannsConvBlock(128, 256)
        self.conv_block4 = PannsConvBlock(256, 512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, 527, bias=True)

    def extract_multi(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.matmul(self.spectrogram_extractor.stft(wav), self.logmel_extractor.melW)
        x = (10.0 * torch.log10(x.clamp_min(1e-10))).clamp_min(-100.0).unsqueeze(1)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for blk in (self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4):
            x = blk(x, (2, 2))
        xm = torch.mean(x, dim=3)
        half = max(1, xm.shape[2] // 2)
        e_g = F.relu_(self.fc1(torch.max(xm, 2)[0] + torch.mean(xm, 2)))
        e_h1 = F.relu_(self.fc1(torch.max(xm[:, :, :half], 2)[0] + torch.mean(xm[:, :, :half], 2)))
        e_h2 = F.relu_(self.fc1(torch.max(xm[:, :, half:], 2)[0] + torch.mean(xm[:, :, half:], 2)))
        return e_g, e_h1, e_h2, self.fc_audioset(e_g)


class PannsCnn14Backbone(nn.Module):
    """PANNs AudioSet-2M Cnn14 VGG-style 6-block ConvNet backbone."""
    emb_dim = 2048

    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = nn.Module()
        self.spectrogram_extractor.stft = PannsSTFT()
        self.logmel_extractor = nn.Module()
        self.logmel_extractor.register_buffer("melW", torch.zeros(513, 64))
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = PannsConvBlock(1, 64)
        self.conv_block2 = PannsConvBlock(64, 128)
        self.conv_block3 = PannsConvBlock(128, 256)
        self.conv_block4 = PannsConvBlock(256, 512)
        self.conv_block5 = PannsConvBlock(512, 1024)
        self.conv_block6 = PannsConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, 527, bias=True)

    def extract_multi(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.matmul(self.spectrogram_extractor.stft(wav), self.logmel_extractor.melW)
        x = (10.0 * torch.log10(x.clamp_min(1e-10))).clamp_min(-100.0).unsqueeze(1)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for blk, ps in (
            (self.conv_block1, (2, 2)),
            (self.conv_block2, (2, 2)),
            (self.conv_block3, (2, 2)),
            (self.conv_block4, (2, 2)),
            (self.conv_block5, (2, 2)),
            (self.conv_block6, (1, 1)),
        ):
            x = blk(x, ps)
        xm = torch.mean(x, dim=3)
        half = max(1, xm.shape[2] // 2)
        e_g = F.relu_(self.fc1(torch.max(xm, 2)[0] + torch.mean(xm, 2)))
        e_h1 = F.relu_(self.fc1(torch.max(xm[:, :, :half], 2)[0] + torch.mean(xm[:, :, :half], 2)))
        e_h2 = F.relu_(self.fc1(torch.max(xm[:, :, half:], 2)[0] + torch.mean(xm[:, :, half:], 2)))
        return e_g, e_h1, e_h2, self.fc_audioset(e_g)


# ══════════════════════════════════════════════════════════════
# EfficientAT (Schmid et al., MIT License) Frontend + MN + DyMN
# ══════════════════════════════════════════════════════════════

class MelSTFT32k(nn.Module):
    """Kaldi-compatible 32 kHz 128-bin Log-Mel extractor with optional SpecAugment in train mode."""
    def __init__(self, n_mels: int = 128, sr: int = 32000, win_length: int = 800, hopsize: int = 320, n_fft: int = 1024, fmax: float = 15000.0):
        super().__init__()
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = 0.0
        self.fmax = fmax
        self.hopsize = hopsize
        self.register_buffer("window", torch.hann_window(win_length, periodic=False), persistent=False)
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-0.97, 1.0]]]), persistent=False)
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels, self.n_fft, self.sr, self.fmin, self.fmax, vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0
        )
        self.register_buffer("mel_basis", F.pad(mel_basis, (0, 1), mode="constant", value=0.0), persistent=False)
        self.freqm = torchaudio.transforms.FrequencyMasking(24, iid_masks=True)
        self.timem = torchaudio.transforms.TimeMasking(96, iid_masks=True)

    def forward(self, wav: torch.Tensor, augment: bool = False) -> torch.Tensor:
        x = F.conv1d(wav.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        stft_c = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length, center=True, normalized=False, window=self.window, return_complex=True)
        pow_spec = stft_c.real ** 2 + stft_c.imag ** 2
        melspec = torch.matmul(self.mel_basis, pow_spec)
        melspec = (melspec + 1e-5).log()
        if augment and self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)
        return ((melspec + 4.5) / 5.0).unsqueeze(1)


class _SqueezeExciteDim(nn.Module):
    def __init__(self, input_dim: int, squeeze_dim: int, se_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, squeeze_dim)
        self.fc2 = nn.Linear(squeeze_dim, input_dim)
        self.se_dim = [1, 2, 3]
        self.se_dim.remove(se_dim)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.mean(x, self.se_dim, keepdim=True)
        shape = scale.size()
        scale = self.fc2(self.activation(self.fc1(scale.squeeze(2).squeeze(2))))
        return self.scale_activation(scale).view(shape) * x


class _ConcurrentSEBlock(nn.Module):
    def __init__(self, c_dim: int, f_dim: int, t_dim: int, se_dims: Sequence[int] = (1,), se_r: int = 4):
        super().__init__()
        dims = [c_dim, f_dim, t_dim]
        self.conc_se_layers = nn.ModuleList(
            [_SqueezeExciteDim(dims[d - 1], _make_divisible(dims[d - 1] // se_r, 8), d) for d in se_dims]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.stack([layer(x) for layer in self.conc_se_layers], dim=0), dim=0)[0]


class _InvertedResidualConfig:
    def __init__(self, in_ch: int, kernel: int, exp_ch: int, out_ch: int, use_se: bool, act: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = _make_divisible(in_ch * width_mult, 8)
        self.kernel = kernel
        self.expanded_channels = _make_divisible(exp_ch * width_mult, 8)
        self.out_channels = _make_divisible(out_ch * width_mult, 8)
        self.use_se = use_se
        self.use_hs = act == "HS"
        self.stride = stride
        self.dilation = dilation
        self.f_dim = 0
        self.t_dim = 0

    def out_size(self, in_size: int) -> int:
        padding = (self.kernel - 1) // 2 * self.dilation
        return _cnn_out_size(in_size, padding, self.dilation, self.kernel, self.stride)


class _InvertedResidual(nn.Module):
    def __init__(self, cnf: _InvertedResidualConfig, se_dims: Optional[Sequence[int]], norm_layer: Callable[..., nn.Module]):
        super().__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        act = nn.Hardswish if cnf.use_hs else nn.ReLU
        layers: List[nn.Module] = []
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(Conv2dNormActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=act))
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(Conv2dNormActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel, stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels, norm_layer=norm_layer, activation_layer=act))
        if cnf.use_se and se_dims is not None:
            layers.append(_ConcurrentSEBlock(cnf.expanded_channels, cnf.f_dim, cnf.t_dim, se_dims=se_dims))
        layers.append(Conv2dNormActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None))
        self.block = nn.Sequential(*layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        res = self.block(inp)
        return res + inp if self.use_res_connect else res


class AudioMobileNetBackbone(nn.Module):
    """Self-contained EfficientAT MobileNetV3 AudioSet-2M backbone."""
    def __init__(self, width_mult: float = 2.0, num_classes: int = 527):
        super().__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        bneck = partial(_InvertedResidualConfig, width_mult=width_mult)
        cfgs = [
            bneck(16, 3, 16, 16, False, "RE", 1, 1),
            bneck(16, 3, 64, 24, False, "RE", 2, 1),
            bneck(24, 3, 72, 24, False, "RE", 1, 1),
            bneck(24, 5, 72, 40, True, "RE", 2, 1),
            bneck(40, 5, 120, 40, True, "RE", 1, 1),
            bneck(40, 5, 120, 40, True, "RE", 1, 1),
            bneck(40, 3, 240, 80, False, "HS", 2, 1),
            bneck(80, 3, 200, 80, False, "HS", 1, 1),
            bneck(80, 3, 184, 80, False, "HS", 1, 1),
            bneck(80, 3, 184, 80, False, "HS", 1, 1),
            bneck(80, 3, 480, 112, True, "HS", 1, 1),
            bneck(112, 3, 672, 112, True, "HS", 1, 1),
            bneck(112, 5, 672, 160, True, "HS", 2, 1),
            bneck(160, 5, 960, 160, True, "HS", 1, 1),
            bneck(160, 5, 960, 160, True, "HS", 1, 1),
        ]
        f_dim, t_dim = _cnn_out_size(128, 1, 1, 3, 2), _cnn_out_size(1000, 1, 1, 3, 2)
        layers: List[nn.Module] = [Conv2dNormActivation(1, cfgs[0].input_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish)]
        for cnf in cfgs:
            f_dim, t_dim = cnf.out_size(f_dim), cnf.out_size(t_dim)
            cnf.f_dim, cnf.t_dim = f_dim, t_dim
            layers.append(_InvertedResidual(cnf, (1,), norm_layer))
        lastconv_in = cfgs[-1].out_channels
        self.out_channels = 6 * lastconv_in
        last_channel = _make_divisible(1280 * width_mult, 8)
        layers.append(Conv2dNormActivation(lastconv_in, self.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(self.out_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

    def extract_sequence_and_global(self, spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fmap = self.features(spec)
        emb = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        as_logits = self.classifier(fmap)
        t_seq = F.adaptive_avg_pool2d(fmap, (1, fmap.shape[-1])).squeeze(2).transpose(1, 2)
        return emb, as_logits, t_seq


class _DynamicConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, context_dim: int, kernel_size: int, stride: int = 1, padding: int = 0, groups: int = 1, k: int = 4, temperature: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.k = k
        self.temperature = temperature
        self.residuals = nn.Sequential(nn.Linear(context_dim, k))
        weight = torch.randn(k, out_channels, in_channels // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight.view(1, 1, k, -1), requires_grad=True)

    def forward(self, x: torch.Tensor, g: Any = None) -> torch.Tensor:
        b, _, f, t = x.size()
        g_c = g[0].view(b, -1)
        att = F.softmax(self.residuals(g_c).view(b, 1, 1, -1) / self.temperature, dim=-1)
        agg_w = (att @ self.weight).transpose(1, 2).reshape(b * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        out = F.conv2d(x.view(1, -1, f, t), weight=agg_w, bias=None, stride=self.stride, padding=self.padding, groups=self.groups * b)
        return out.view(b, self.out_channels, out.size(-2), out.size(-1))


class _DyReLUB(nn.Module):
    def __init__(self, channels: int, context_dim: int, M: int = 2):
        super().__init__()
        self.channels = channels
        self.M = M
        self.coef_net = nn.Sequential(nn.Linear(context_dim, 2 * M * channels))
        self.sigmoid = nn.Sigmoid()
        self.register_buffer("lambdas", torch.tensor([1.0] * M + [0.5] * M, dtype=torch.float32))
        self.register_buffer("init_v", torch.tensor([1.0] + [0.0] * (2 * M - 1), dtype=torch.float32))

    def forward(self, x: torch.Tensor, g: Any) -> torch.Tensor:
        b = x.size(0)
        theta = 2.0 * self.sigmoid(self.coef_net(g[0].view(b, -1))) - 1.0
        coefs = theta.view(-1, self.channels, 1, 1, 2 * self.M) * self.lambdas + self.init_v
        x_m = x.unsqueeze(-1) * coefs[:, :, :, :, :self.M] + coefs[:, :, :, :, self.M:]
        return torch.maximum(x_m[:, :, :, :, 0], x_m[:, :, :, :, 1])


class _CoordAtt(nn.Module):
    def forward(self, x: torch.Tensor, g: Any) -> torch.Tensor:
        return x * g[1].sigmoid() * g[2].sigmoid()


class _DynamicWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, g: Any = None) -> torch.Tensor:
        return self.module(x)


class _ContextGen(nn.Module):
    def __init__(self, context_dim: int, in_ch: int, exp_ch: int, norm_layer: Callable[..., nn.Module], stride: int = 1):
        super().__init__()
        self.joint_conv = nn.Conv2d(in_ch, context_dim, kernel_size=1, bias=False)
        self.joint_norm = norm_layer(context_dim)
        self.joint_act = nn.Hardswish(inplace=True)
        self.conv_f = nn.Conv2d(context_dim, exp_ch, kernel_size=1)
        self.conv_t = nn.Conv2d(context_dim, exp_ch, kernel_size=1)
        self.pool_f = nn.AvgPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)) if stride > 1 else nn.Sequential()
        self.pool_t = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, stride), padding=(0, 1)) if stride > 1 else nn.Sequential()

    def forward(self, x: torch.Tensor, g: Any = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cf = F.adaptive_avg_pool2d(x, (None, 1))
        ct = F.adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
        f, t = cf.size(2), ct.size(2)
        g_cat = self.joint_act(self.joint_norm(self.joint_conv(torch.cat([cf, ct], dim=2))))
        h_cf, h_ct = torch.split(g_cat, [f, t], dim=2)
        h_c = torch.mean(g_cat, dim=2, keepdim=True)
        return h_c, self.conv_f(self.pool_f(h_cf)), self.conv_t(self.pool_t(h_ct.permute(0, 1, 3, 2)))


class _DYBlock(nn.Module):
    def __init__(self, cnf: _InvertedResidualConfig, width_mult: float = 1.0):
        super().__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        context_dim = int(np.clip(_make_divisible(cnf.expanded_channels // 4, 8), _make_divisible(32 * width_mult, 8), _make_divisible(128 * width_mult, 8)))
        act = nn.Hardswish if cnf.use_hs else nn.ReLU
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        if cnf.expanded_channels != cnf.input_channels:
            self.exp_conv = _DynamicConv(cnf.input_channels, cnf.expanded_channels, context_dim, kernel_size=1)
            self.exp_norm = norm_layer(cnf.expanded_channels)
            self.exp_act = _DynamicWrapper(act(inplace=True))
        else:
            self.exp_conv = _DynamicWrapper(nn.Identity())
            self.exp_norm = nn.Identity()
            self.exp_act = _DynamicWrapper(nn.Identity())
        padding = (cnf.kernel - 1) // 2
        self.depth_conv = _DynamicConv(cnf.expanded_channels, cnf.expanded_channels, context_dim, kernel_size=cnf.kernel, groups=cnf.expanded_channels, stride=cnf.stride, padding=padding)
        self.depth_norm = norm_layer(cnf.expanded_channels)
        self.depth_act = _DyReLUB(cnf.expanded_channels, context_dim, M=2)
        self.ca = _CoordAtt()
        self.proj_conv = _DynamicConv(cnf.expanded_channels, cnf.out_channels, context_dim, kernel_size=1)
        self.proj_norm = norm_layer(cnf.out_channels)
        self.context_gen = _ContextGen(context_dim, cnf.input_channels, cnf.expanded_channels, norm_layer=norm_layer, stride=cnf.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        g = self.context_gen(x)
        x = self.exp_act(self.exp_norm(self.exp_conv(x, g)), g)
        x = self.ca(self.depth_act(self.depth_norm(self.depth_conv(x, g)), g), g)
        x = self.proj_norm(self.proj_conv(x, g))
        return x + inp if self.use_res_connect else x


class AudioDyMNBackbone(nn.Module):
    """Self-contained EfficientAT Dynamic MobileNet (DyMN) AudioSet-2M backbone."""
    def __init__(self, width_mult: float = 1.0, num_classes: int = 527):
        super().__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        bneck = partial(_InvertedResidualConfig, width_mult=width_mult)
        cfgs = [
            bneck(16, 3, 16, 16, True, "RE", 1, 1),
            bneck(16, 3, 64, 24, True, "RE", 2, 1),
            bneck(24, 3, 72, 24, True, "RE", 1, 1),
            bneck(24, 5, 72, 40, True, "RE", 2, 1),
            bneck(40, 5, 120, 40, True, "RE", 1, 1),
            bneck(40, 5, 120, 40, True, "RE", 1, 1),
            bneck(40, 3, 240, 80, True, "HS", 2, 1),
            bneck(80, 3, 200, 80, True, "HS", 1, 1),
            bneck(80, 3, 184, 80, True, "HS", 1, 1),
            bneck(80, 3, 184, 80, True, "HS", 1, 1),
            bneck(80, 3, 480, 112, True, "HS", 1, 1),
            bneck(112, 3, 672, 112, True, "HS", 1, 1),
            bneck(112, 5, 672, 160, True, "HS", 2, 1),
            bneck(160, 5, 960, 160, True, "HS", 1, 1),
            bneck(160, 5, 960, 160, True, "HS", 1, 1),
        ]
        self.in_c = Conv2dNormActivation(1, cfgs[0].input_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish)
        self.layers = nn.ModuleList([_DYBlock(cnf, width_mult=width_mult) for cnf in cfgs])
        lastconv_in = cfgs[-1].out_channels
        self.out_channels = 6 * lastconv_in
        last_channel = _make_divisible(1280 * width_mult, 8)
        self.out_c = Conv2dNormActivation(lastconv_in, self.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(self.out_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

    def _encode_spec(self, spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_c(spec)
        for layer in self.layers:
            x = layer(x)
        x = self.out_c(x)
        emb = F.adaptive_avg_pool2d(x, 1).flatten(1)
        as_logits = self.classifier(x)
        return emb, as_logits

    def extract_multi(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb, as_logits = self._encode_spec(mel_spec)
        half = max(1, mel_spec.shape[-1] // 2)
        e_h1, _ = self._encode_spec(mel_spec[:, :, :, :half])
        e_h2, _ = self._encode_spec(mel_spec[:, :, :, half:])
        return emb, e_h1, e_h2, as_logits


def load_pretrained_panns(repo_id: str, module: nn.Module) -> None:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    wrap = nn.Module()
    wrap.backbone = module
    sd = load_file(hf_hub_download(repo_id, "model.safetensors"))
    wrap.load_state_dict(sd, strict=True)


def load_pretrained_efficientat(url: str, module: nn.Module) -> None:
    AUDIOSET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sd = torch.hub.load_state_dict_from_url(url, model_dir=str(AUDIOSET_CACHE_DIR), map_location="cpu")
    module.load_state_dict(sd, strict=True)
