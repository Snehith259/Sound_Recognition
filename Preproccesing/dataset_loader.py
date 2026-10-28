"""
PyTorch Dataset and DataLoader module for Environmental Sound Classification.
Loads preprocessed multi-modal features (.npy) or raw audio on-the-fly:
- Log-Mel Spectrogram: (1, 128, 216) -> SResNet spatial branch
- MFCC: (40, 216) -> CNN-GRU temporal sequence branch
- Recurrence Plot: (1, 128, 128) -> Dual-Modal Feature Fusion (DMFF) branch
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from . import config
from .audio_processor import load_and_resample, peak_normalize
from .feature_extractor import extract_all_features


class ESC50MultiModalDataset(Dataset):
    """
    Dataset that loads preprocessed multimodal features from disk.
    If precomputed .npy files exist in features/ directory, loads them directly for maximum I/O speed.
    Otherwise, computes features dynamically from raw .wav audio files.
    """

    def __init__(
        self,
        split_csv: str | Path,
        features_dir: Path = config.FEATURES_DIR,
        audio_dir: Path = config.RAW_DATA_DIR,
        transform: Optional[Any] = None,
    ):
        self.df = pd.read_csv(split_csv)
        self.features_dir = Path(features_dir)
        self.audio_dir = Path(audio_dir)
        self.transform = transform

        self.log_mel_dir = self.features_dir / "log_mel"
        self.mfcc_dir = self.features_dir / "mfcc"
        self.rp_dir = self.features_dir / "recurrence_plot"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        filename = row["filename"]
        base_name = Path(filename).stem
        target = int(row["target"])
        category = str(row["category"])

        # Check if precomputed .npy features exist
        log_mel_file = self.log_mel_dir / f"{base_name}.npy"
        mfcc_file = self.mfcc_dir / f"{base_name}.npy"
        rp_file = self.rp_dir / f"{base_name}.npy"

        if log_mel_file.exists() and mfcc_file.exists() and rp_file.exists():
            log_mel = np.load(log_mel_file)
            mfcc = np.load(mfcc_file)
            rp = np.load(rp_file)
        else:
            # Fallback to direct computation from audio
            wav_path = self.audio_dir / filename
            audio, sr = load_and_resample(wav_path)
            audio = peak_normalize(audio)
            features = extract_all_features(audio, sr=sr)
            log_mel = features["log_mel"]
            mfcc = features["mfcc"]
            rp = features["recurrence_plot"]

        # Convert to PyTorch tensors with channel dimensions
        # log_mel: (1, 128, 216)
        # mfcc: (40, 216)
        # rp: (1, 128, 128)
        tensor_log_mel = torch.from_numpy(log_mel).unsqueeze(0).float()
        tensor_mfcc = torch.from_numpy(mfcc).float()
        tensor_rp = torch.from_numpy(rp).unsqueeze(0).float()

        if self.transform:
            tensor_log_mel = self.transform(tensor_log_mel)

        return {
            "log_mel": tensor_log_mel,
            "mfcc": tensor_mfcc,
            "recurrence_plot": tensor_rp,
            "target": torch.tensor(target, dtype=torch.long),
            "category": category,
            "filename": filename,
        }


def get_dataloaders(
    fold: int = 1,
    splits_dir: Path = config.SPLITS_DIR,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Construct train, validation, and test PyTorch DataLoaders for a specific ESC-50 fold.
    """
    train_csv = splits_dir / f"fold_{fold}_train.csv"
    val_csv = splits_dir / f"fold_{fold}_val.csv"
    test_csv = splits_dir / f"fold_{fold}_test.csv"

    train_ds = ESC50MultiModalDataset(train_csv)
    val_ds = ESC50MultiModalDataset(val_csv)
    test_ds = ESC50MultiModalDataset(test_csv)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return loaders
