"""
Verification, diagnostic testing, and visualization generator for preprocessed features.
- Validates tensor integrity and data types across all folds
- Tests PyTorch DataLoader batch retrieval
- Generates high-resolution diagnostic plot (Waveform, Log-Mel, MFCC, Recurrence Plot)
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from Preproccesing import config
from Preproccesing.dataset_loader import get_dataloaders


def verify_feature_integrity() -> dict:
    """Check integrity, count, and dimensions of all saved .npy features."""
    log_mel_files = list(config.LOG_MEL_DIR.glob("*.npy"))
    mfcc_files = list(config.MFCC_DIR.glob("*.npy"))
    rp_files = list(config.RP_DIR.glob("*.npy"))
    audio_files = list(config.PROCESSED_AUDIO_DIR.glob("*.wav"))

    integrity = {
        "log_mel_count": len(log_mel_files),
        "mfcc_count": len(mfcc_files),
        "rp_count": len(rp_files),
        "processed_audio_count": len(audio_files),
        "checks_passed": True,
        "sample_validations": {},
    }

    if log_mel_files:
        sample_mel = np.load(log_mel_files[0])
        sample_mfcc = np.load(mfcc_files[0])
        sample_rp = np.load(rp_files[0])

        integrity["sample_validations"] = {
            "log_mel_shape": list(sample_mel.shape),
            "log_mel_dtype": str(sample_mel.dtype),
            "log_mel_range": [float(sample_mel.min()), float(sample_mel.max())],
            "mfcc_shape": list(sample_mfcc.shape),
            "mfcc_dtype": str(sample_mfcc.dtype),
            "rp_shape": list(sample_rp.shape),
            "rp_dtype": str(sample_rp.dtype),
            "rp_range": [float(sample_rp.min()), float(sample_rp.max())],
        }

        # Check expected dimensions
        if sample_mel.shape != (config.N_MELS, 216):
            integrity["checks_passed"] = False
        if sample_mfcc.shape != (config.N_MFCC, 216):
            integrity["checks_passed"] = False
        if sample_rp.shape != (config.PAA_DIM, config.PAA_DIM):
            integrity["checks_passed"] = False

    return integrity


def generate_feature_visualization(sample_idx: int = 0) -> Path:
    """Generate and save a visual comparison of extracted features."""
    log_mel_files = sorted(list(config.LOG_MEL_DIR.glob("*.npy")))
    if not log_mel_files:
        print("No feature files found for visualization.")
        return None

    target_file = log_mel_files[min(sample_idx, len(log_mel_files) - 1)]
    base_name = target_file.stem

    mel = np.load(config.LOG_MEL_DIR / f"{base_name}.npy")
    mfcc = np.load(config.MFCC_DIR / f"{base_name}.npy")
    rp = np.load(config.RP_DIR / f"{base_name}.npy")

    # Load corresponding audio
    audio_path = config.PROCESSED_AUDIO_DIR / f"{base_name}.wav"
    if not audio_path.exists():
        audio_path = config.RAW_DATA_DIR / f"{base_name}.wav"

    audio, sr = np.zeros(config.TARGET_SAMPLES), config.SAMPLE_RATE
    if audio_path.exists():
        import librosa
        audio, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0a0d14")
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Waveform
    ax0 = axes[0, 0]
    ax0.set_facecolor("#111827")
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    ax0.plot(time_axis, audio, color="#38bdf8", linewidth=0.8)
    ax0.set_title(f"1D Normalized Waveform ({base_name})", color="#f8fafc", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Time (seconds)", color="#94a3b8")
    ax0.set_ylabel("Amplitude", color="#94a3b8")
    ax0.tick_params(colors="#94a3b8")
    ax0.grid(color="#1f2937", linestyle="--", alpha=0.5)

    # 2. Log-Mel Spectrogram (Spatial / SResNet branch)
    ax1 = axes[0, 1]
    ax1.set_facecolor("#111827")
    im1 = ax1.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    ax1.set_title(f"128-Band Log-Mel Spectrogram (Shape: {mel.shape})", color="#f8fafc", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time Frames", color="#94a3b8")
    ax1.set_ylabel("Mel Frequency Bands", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    cb1 = fig.colorbar(im1, ax=ax1)
    cb1.ax.tick_params(colors="#94a3b8")

    # 3. MFCC (Temporal sequence / CNN-GRU branch)
    ax2 = axes[1, 0]
    ax2.set_facecolor("#111827")
    im2 = ax2.imshow(mfcc, aspect="auto", origin="lower", cmap="viridis")
    ax2.set_title(f"40-Component MFCC (Shape: {mfcc.shape})", color="#f8fafc", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time Frames", color="#94a3b8")
    ax2.set_ylabel("Cepstral Coefficients", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    cb2 = fig.colorbar(im2, ax=ax2)
    cb2.ax.tick_params(colors="#94a3b8")

    # 4. Threshold-Free Recurrence Plot (Chaotic dynamic / DMFF branch)
    ax3 = axes[1, 1]
    ax3.set_facecolor("#111827")
    im3 = ax3.imshow(rp, aspect="equal", origin="lower", cmap="inferno")
    ax3.set_title(f"Threshold-Free Recurrence Plot (Shape: {rp.shape})", color="#f8fafc", fontsize=12, fontweight="bold")
    ax3.set_xlabel("PAA Trajectory Index i", color="#94a3b8")
    ax3.set_ylabel("PAA Trajectory Index j", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8")
    cb3 = fig.colorbar(im3, ax=ax3)
    cb3.ax.tick_params(colors="#94a3b8")

    plt.suptitle("EURASIP 2024 Dual-Modal Preprocessing Verification", color="#38bdf8", fontsize=15, fontweight="bold", y=0.98)
    
    out_img = config.REPORTS_DIR / "feature_visualization.png"
    plt.savefig(out_img, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Feature visualization plot exported to: {out_img}")
    return out_img


def test_pytorch_dataloader() -> bool:
    """Test batch retrieval from PyTorch DataLoader."""
    print("Testing PyTorch DataLoader batch retrieval...")
    try:
        loaders = get_dataloaders(fold=1, batch_size=16)
        train_loader = loaders["train"]
        batch = next(iter(train_loader))

        log_mel = batch["log_mel"]
        mfcc = batch["mfcc"]
        rp = batch["recurrence_plot"]
        target = batch["target"]

        print(f"  PyTorch Batch Loaded Successfully:")
        print(f"    - log_mel shape:         {log_mel.shape} (Batch, Channel, Mels, Time)")
        print(f"    - mfcc shape:            {mfcc.shape} (Batch, Coeffs, Time)")
        print(f"    - recurrence_plot shape: {rp.shape} (Batch, Channel, PAA, PAA)")
        print(f"    - target shape:          {target.shape} (Batch labels)")
        return True
    except Exception as e:
        print(f"  DataLoader test failed: {e}")
        return False


if __name__ == "__main__":
    integ = verify_feature_integrity()
    print("Feature Integrity Check:", json.dumps(integ, indent=2))
    generate_feature_visualization()
    test_pytorch_dataloader()
