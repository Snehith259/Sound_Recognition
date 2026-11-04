"""
Preproccesing package for ESC-50 Environmental Sound Classification.
Supports ResNet-18, VGG-16, SResNet, and CNN-GRU sound recognition pipelines.
"""

from . import config
from .audio_processor import load_and_resample, peak_normalize, detect_active_segment, piecewise_aggregate_approximation
from .feature_extractor import extract_log_mel_spectrogram, extract_mfcc, extract_threshold_free_recurrence_plot, extract_all_features
from .dataset_splits import generate_5fold_splits
from .dataset_loader import ESC50MultiModalDataset, get_dataloaders
from .models import ResNet18_ESC50, VGG16_ESC50, SResNet, CNNGRU, get_model
