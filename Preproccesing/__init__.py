"""
Preproccesing package for ESC-50 Environmental Sound Classification.
Implements the EURASIP 2024 Dual-Modal Feature Fusion preprocessing pipeline.
"""

from . import config
from .audio_processor import load_and_resample, peak_normalize, detect_active_segment, piecewise_aggregate_approximation
from .feature_extractor import extract_log_mel_spectrogram, extract_mfcc, extract_threshold_free_recurrence_plot, extract_all_features
from .dataset_splits import generate_5fold_splits
from .dataset_loader import ESC50MultiModalDataset, get_dataloaders
