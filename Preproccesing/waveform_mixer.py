"""Acoustic Waveform Mixture & Domain-Robust Training Sample Generator.

Generates physically realistic training mixtures and multi-domain perturbations directly
from real waveforms (never touches test folds or external evaluation data):
  1. Real 2-sound waveform superposition (y = norm(y1) + g * norm(y2), energy-weighted soft targets)
  2. Variable-length active burst cropping & smooth edge zero-padding (fixes <3s external clips)
  3. Spectral tilt & bandwidth low-pass simulation (mimics MP3/AAC/phone/microphone rolloff)
  4. Colored background floor noise (pink/brown/white at SNR 12-30 dB)
  5. Exact `extract_log_mel_spectrogram()` parity with serving inference.
"""
from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from Preproccesing.feature_extractor import extract_log_mel_spectrogram

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "Preproccesing" / "processed" / "audio_22k"
ESC50_RAW_DIR = ROOT / "ESC-50" / "wav_files"


def _locate_audio(filename: str) -> Path:
    stem = Path(filename).stem
    for d in (AUDIO_DIR, ESC50_RAW_DIR):
        p = d / filename
        if p.is_file():
            return p
        p2 = d / f"{stem}.wav"
        if p2.is_file():
            return p2
    raise FileNotFoundError(f"Audio clip not found for {filename} in {AUDIO_DIR} or {ESC50_RAW_DIR}")


def load_fold_waveforms(manifest_df: pd.DataFrame, target_sr: int = 22050) -> tuple[list[np.ndarray], np.ndarray]:
    """Load real waveforms and integer targets for a given split dataframe."""
    waves: list[np.ndarray] = []
    targets: list[int] = []
    target_len = int(target_sr * 5.0)
    for row in manifest_df.itertuples(index=False):
        p = _locate_audio(str(row.filename))
        y, sr = sf.read(str(p), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if y.size < target_len:
            y = np.pad(y, (0, target_len - y.size))
        elif y.size > target_len:
            y = y[:target_len]
        waves.append(np.asarray(y, dtype=np.float32))
        targets.append(int(row.target))
    return waves, np.asarray(targets, dtype=np.int64)


def _active_slice(y: np.ndarray, target_dur_samples: int, sr: int = 22050, rng: random.Random | None = None) -> np.ndarray:
    """Extract an active sub-segment and smooth edge fade without boundary clicks."""
    if y.size <= target_dur_samples:
        return y.copy()
    hop = max(1, sr // 4)
    energies = []
    starts = list(range(0, y.size - target_dur_samples + 1, hop))
    for s in starts:
        seg = y[s:s + target_dur_samples]
        energies.append(float(np.mean(seg ** 2)))
    max_e = max(energies) if energies else 0.0
    valid = [s for s, e in zip(starts, energies) if e >= 0.20 * max_e] or starts
    start = rng.choice(valid) if rng else valid[0]
    seg = y[start:start + target_dur_samples].copy()
    fade = min(660, seg.size // 6)
    if fade > 8:
        win = np.hanning(2 * fade).astype(np.float32)
        seg[:fade] *= win[:fade]
        seg[-fade:] *= win[fade:]
    return seg


def _apply_spectral_tilt_and_lowpass(y: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    """Simulate microphone/codec frequency response (spectral tilt + high-freq lowpass)."""
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(y.size, d=1.0 / sr)
    norm_f = freqs / max(1.0, sr * 0.5)
    # Random tilt (-4 to +4 dB across octave log-scale)
    tilt_db = rng.uniform(-4.0, 4.0)
    tilt_gain = 10.0 ** ((tilt_db * (norm_f - 0.3)) / 20.0)
    # Smooth Butterworth-like lowpass between 4.8 kHz and 10.2 kHz
    fc = rng.uniform(4800.0, 10200.0)
    order = 4
    lp_gain = 1.0 / np.sqrt(1.0 + (freqs / fc) ** (2 * order))
    out = np.fft.irfft(spec * tilt_gain * lp_gain, n=y.size)
    return out.astype(np.float32)


def _add_colored_noise(y: np.ndarray, snr_db: float, np_rng: np.random.Generator) -> np.ndarray:
    white = np_rng.normal(0.0, 1.0, size=y.size).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(y.size, d=1.0 / 22050)
    alpha = float(np_rng.uniform(0.35, 0.9))  # between pink and brownian ambient room noise
    colored = np.fft.irfft(spec / np.maximum(freqs, 20.0) ** alpha, n=y.size).astype(np.float32)
    sig_rms = float(np.sqrt(np.mean(y ** 2))) + 1e-8
    n_rms = float(np.sqrt(np.mean(colored ** 2))) + 1e-8
    target_n_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    return (y + colored * (target_n_rms / n_rms)).astype(np.float32)


def generate_mixed_spectrogram_dataset(
    waves: Sequence[np.ndarray],
    targets: Sequence[int],
    n_samples: int,
    num_classes: int = 50,
    mix_fraction: float = 0.55,
    short_crop_fraction: float = 0.15,
    label_smoothing: float = 0.04,
    seed: int = 2026,
    max_workers: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build `(n_samples, 1, 128, 216)` Log-Mel tensor and soft target distribution `(n_samples, 50)`."""
    n_base = len(waves)
    if n_base == 0:
        raise ValueError("Empty waveform list")
    target_len = waves[0].size
    sr = 22050
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Pre-index by class so 2-sound mixtures always pick distinct classes
    by_class: dict[int, list[int]] = {}
    for i, c in enumerate(targets):
        by_class.setdefault(int(c), []).append(i)
    class_list = list(by_class.keys())

    # First construct waveforms + soft targets deterministically on main thread (very fast vector ops)
    synth_waves: list[np.ndarray] = []
    soft_targets = np.full((n_samples, num_classes), label_smoothing / num_classes, dtype=np.float32)

    for idx in range(n_samples):
        i1 = idx % n_base if idx < n_base else rng.randrange(n_base)
        c1 = int(targets[i1])
        y1 = waves[i1].copy()

        # Keep clean exact pass on the very first epoch pass for canonical anchors
        if idx < n_base:
            peak = float(np.max(np.abs(y1)))
            if peak > 1e-8:
                y1 = y1 / peak
            synth_waves.append(y1)
            soft_targets[idx, c1] += 1.0 - label_smoothing
            continue

        # Random circular time shift on base clip
        shift1 = rng.randint(0, target_len - 1)
        y1 = np.roll(y1, shift1)

        roll_mode = rng.random()
        if roll_mode < mix_fraction and len(class_list) >= 2:
            # Acoustic 2-sound waveform superposition
            c2 = rng.choice([c for c in class_list if c != c1])
            i2 = rng.choice(by_class[c2])
            y2 = np.roll(waves[i2].copy(), rng.randint(0, target_len - 1))

            # Occasionally make one component a localized short active burst (0.8s..3.0s) inside the 5s mix!
            if rng.random() < 0.35:
                burst_len = int(sr * rng.uniform(0.9, 3.2))
                burst = _active_slice(y2, burst_len, sr=sr, rng=rng)
                y2_pad = np.zeros(target_len, dtype=np.float32)
                off = rng.randint(0, target_len - burst.size)
                y2_pad[off:off + burst.size] = burst
                y2 = y2_pad

            rms1 = float(np.sqrt(np.mean(y1 ** 2))) + 1e-8
            rms2 = float(np.sqrt(np.mean(y2 ** 2))) + 1e-8
            # Realistic background/co-occurring gain (0.22..0.75) so primary sound keeps sharp confidence
            g = rng.uniform(0.22, 0.75)
            y_out = (y1 / rms1) + g * (y2 / rms2)

            if rng.random() < 0.35:
                y_out = _apply_spectral_tilt_and_lowpass(y_out, sr, rng)
            if rng.random() < 0.25:
                y_out = _add_colored_noise(y_out, float(np_rng.uniform(18.0, 32.0)), np_rng)

            e1 = 1.0
            e2 = g * g
            w1 = e1 / (e1 + e2)
            w2 = 1.0 - w1
            soft_targets[idx, c1] += w1 * (1.0 - label_smoothing)
            soft_targets[idx, c2] += w2 * (1.0 - label_smoothing)
        else:
            # Single-source domain-robust variant
            y_out = y1
            if roll_mode < mix_fraction + short_crop_fraction:
                crop_len = int(sr * rng.uniform(0.75, 3.5))
                seg = _active_slice(y_out, crop_len, sr=sr, rng=rng)
                padded = np.zeros(target_len, dtype=np.float32)
                # Mix left-aligned padding (matching short-clip inference) with random positions
                off = 0 if rng.random() < 0.45 else rng.randint(0, target_len - seg.size)
                padded[off:off + seg.size] = seg
                y_out = padded
            if rng.random() < 0.45:
                y_out = _apply_spectral_tilt_and_lowpass(y_out, sr, rng)
            if rng.random() < 0.35:
                y_out = _add_colored_noise(y_out, float(np_rng.uniform(16.0, 32.0)), np_rng)
            soft_targets[idx, c1] += 1.0 - label_smoothing

        peak = float(np.max(np.abs(y_out)))
        if peak > 1e-8:
            y_out = y_out / peak
        synth_waves.append(y_out.astype(np.float32))

    workers = max_workers or min(10, max(1, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        mels = list(ex.map(lambda w: extract_log_mel_spectrogram(w, sr=sr), synth_waves))

    x_tensor = torch.from_numpy(np.stack(mels)).unsqueeze(1).float()
    y_tensor = torch.from_numpy(soft_targets).float()
    return x_tensor, y_tensor
