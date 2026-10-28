"""
Configuration module for Environmental Sound Classification (ESC-50) Preprocessing.
Based on EURASIP 2024 Dual-Modal Feature Fusion architecture specifications.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "ESC-50" / "wav_files"
LABELS_CSV = BASE_DIR / "ESC-50" / "esc50_labels.csv"

PREPROCESSING_DIR = BASE_DIR / "Preproccesing"
PROCESSED_AUDIO_DIR = PREPROCESSING_DIR / "processed" / "audio_22k"
FEATURES_DIR = PREPROCESSING_DIR / "processed" / "features"
LOG_MEL_DIR = FEATURES_DIR / "log_mel"
MFCC_DIR = FEATURES_DIR / "mfcc"
RP_DIR = FEATURES_DIR / "recurrence_plot"
SPLITS_DIR = PREPROCESSING_DIR / "splits"
REPORTS_DIR = PREPROCESSING_DIR / "reports"

# Acoustic Signal Specifications (EURASIP 2024 Benchmark)
SAMPLE_RATE = 22050        # Target sampling rate in Hz
TARGET_DURATION = 5.0      # Duration in seconds
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)  # 110,250 samples

# Spectral Parameters (Linear & Harmonic Representation)
N_FFT = 1024               # FFT window size
HOP_LENGTH = 512           # Frame hop length
N_MELS = 128               # Number of Mel filterbands (spatial spectral resolution)
N_MFCC = 40                # Number of Cepstral coefficients
F_MIN = 0.0                # Minimum frequency
F_MAX = 11025.0            # Nyquist frequency

# Dynamical Parameters (Microscopic Chaotic Representation)
# PAA (Piecewise Aggregate Approximation) dimension for continuous distance matrix
PAA_DIM = 128              # Dimension after PAA compression
RP_NORM_METHOD = "minmax"  # Normalization method for Threshold-Free Recurrence Plot

# Cross-Validation
NUM_FOLDS = 5              # 5-fold cross-validation standard (ESC-50)
RANDOM_SEED = 42

# Augmentation Parameters
TIME_STRETCH_RATES = [0.9, 1.1]
PITCH_SHIFT_STEPS = [-2, 2]
NOISE_FACTORS = [0.005, 0.01]
