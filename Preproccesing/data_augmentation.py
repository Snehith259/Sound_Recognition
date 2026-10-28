"""
Audio data augmentation module for ESC-50 training expansion.
Supports:
- Time stretching (rate variations: 0.9x, 1.1x)
- Pitch shifting (+/- 2 semitones)
- Additive Gaussian / environmental noise
- Random temporal shift / cyclic roll
"""

import numpy as np
import librosa
from typing import List, Tuple

from . import config


def time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Apply time stretching without modifying pitch."""
    if rate == 1.0 or len(audio) == 0:
        return audio
    y_stretched = librosa.effects.time_stretch(y=audio, rate=rate)
    target_len = len(audio)
    if len(y_stretched) < target_len:
        y_stretched = np.pad(y_stretched, (0, target_len - len(y_stretched)), mode="constant")
    else:
        y_stretched = y_stretched[:target_len]
    return y_stretched.astype(np.float32)


def pitch_shift(audio: np.ndarray, sr: int = config.SAMPLE_RATE, n_steps: float = 0.0) -> np.ndarray:
    """Shift audio pitch by n_steps semitones without modifying speed."""
    if n_steps == 0.0 or len(audio) == 0:
        return audio
    y_shifted = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
    return y_shifted.astype(np.float32)


def add_gaussian_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add low-amplitude zero-mean Gaussian background noise."""
    noise = np.random.normal(0, noise_factor, len(audio)).astype(np.float32)
    augmented = audio + noise
    max_val = np.max(np.abs(augmented))
    if max_val > 1.0:
        augmented /= max_val
    return augmented.astype(np.float32)


def cyclic_time_shift(audio: np.ndarray, shift_ratio: float = 0.1) -> np.ndarray:
    """Shift audio cyclically along the time axis."""
    shift_samples = int(len(audio) * shift_ratio)
    return np.roll(audio, shift_samples)


def generate_augmented_variants(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
) -> List[Tuple[str, np.ndarray]]:
    """
    Generate 2 augmented variants for an audio waveform to expand the training set
    from 2,000 samples to 6,000 samples (as described in EURASIP 2024 paper).
    """
    variants = []

    # Variant 1: Slight time stretch (1.1x) + subtle pitch shift (+1 semitone)
    v1 = time_stretch(audio, rate=1.1)
    v1 = pitch_shift(v1, sr=sr, n_steps=1.0)
    variants.append(("aug_stretch_pitch_up", v1))

    # Variant 2: Slight time stretch (0.9x) + subtle pitch shift (-1 semitone) + minor noise
    v2 = time_stretch(audio, rate=0.9)
    v2 = pitch_shift(v2, sr=sr, n_steps=-1.0)
    v2 = add_gaussian_noise(v2, noise_factor=0.004)
    variants.append(("aug_stretch_pitch_down", v2))

    return variants
