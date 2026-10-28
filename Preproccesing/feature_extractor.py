"""
Feature extraction module for Dual-Modal Environmental Sound Classification:
- Macroscopic Harmonic Features:
  * 128-band Log-Mel Spectrogram (Shape: [128, 216])
  * 40-coefficient MFCC (Shape: [40, 216])
- Microscopic Chaotic Dynamic Features:
  * Threshold-Free Recurrence Plot (Shape: [128, 128])
    Computes continuous Euclidean distance trajectories on phase-space PAA vectors.
"""

import numpy as np
import librosa
from typing import Dict, Any

from . import config
from .audio_processor import piecewise_aggregate_approximation


def extract_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    n_mels: int = config.N_MELS,
    fmin: float = config.F_MIN,
    fmax: float = config.F_MAX,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract 128-band Log-Mel Spectrogram.
    Output shape: (n_mels, time_steps) -> (128, 216) for 5s @ 22,050 Hz.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    # Convert power to decibel scale
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if normalize:
        # Normalize to [0.0, 1.0]
        min_val = log_mel.min()
        max_val = log_mel.max()
        if max_val - min_val > 1e-6:
            log_mel = (log_mel - min_val) / (max_val - min_val)
        else:
            log_mel = np.zeros_like(log_mel)

    return log_mel.astype(np.float32)


def extract_mfcc(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_mfcc: int = config.N_MFCC,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract 40-coefficient MFCC.
    Output shape: (n_mfcc, time_steps) -> (40, 216).
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    if normalize:
        mean = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True) + 1e-6
        mfcc = (mfcc - mean) / std

    return mfcc.astype(np.float32)


def extract_threshold_free_recurrence_plot(
    audio: np.ndarray,
    paa_dim: int = config.PAA_DIM,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Generate Threshold-Free Continuous Recurrence Plot (RP).
    Based on Eckmann (1987) & EURASIP 2024 Dual-Modal ESC.
    1. Reduces the waveform length using PAA to preserve macro-dynamics.
    2. Builds the continuous Euclidean distance matrix:
       R(i, j) = || x_i - x_j ||
    3. Normalizes to continuous gradient [0.0, 1.0] without binarization threshold,
       preserving subtle acoustic chaotic trajectories and phase dynamics.
    Output shape: (paa_dim, paa_dim) -> (128, 128).
    """
    # Step 1: PAA representation
    x_paa = piecewise_aggregate_approximation(audio, target_dim=paa_dim)

    # Step 2: Distance matrix computation
    diff = x_paa[:, np.newaxis] - x_paa[np.newaxis, :]
    rp = np.abs(diff)

    # Step 3: Threshold-free min-max normalization
    rp_min = rp.min()
    rp_max = rp.max()
    denom = rp_max - rp_min
    if denom > eps:
        rp = (rp - rp_min) / denom
    else:
        rp = np.zeros_like(rp)

    return rp.astype(np.float32)


def extract_all_features(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
) -> Dict[str, np.ndarray]:
    """Extract full feature bundle for a single audio signal."""
    log_mel = extract_log_mel_spectrogram(audio, sr=sr)
    mfcc = extract_mfcc(audio, sr=sr)
    rp = extract_threshold_free_recurrence_plot(audio)

    return {
        "log_mel": log_mel,               # Shape: (128, 216)
        "mfcc": mfcc,                     # Shape: (40, 216)
        "recurrence_plot": rp,            # Shape: (128, 128)
    }
