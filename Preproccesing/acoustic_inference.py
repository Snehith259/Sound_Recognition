"""Audio decoding and inference for the SoundSense ESC-50 classifiers.

Predictions require the requested model checkpoint. The bundled MLP is retained
for offline research, but is never substituted for a missing UI model.
"""

import io
import hashlib
import json
import os
import sys
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Librosa uses Numba's on-disk cache for feature helpers. Put it in the existing
# ignored checkpoint cache instead of relying on a protected site-packages/temp path.
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT_DIR / "checkpoints" / ".cache" / "numba"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn

# Tiny single-clip inference workloads become dramatically slower when PyTorch
# fans out across every host core. Keep the API responsive while allowing an
# explicit deployment override.
_default_threads = min(4, max(1, os.cpu_count() or 1))
try:
    torch.set_num_threads(max(1, int(os.environ.get("SOUNDSENSE_TORCH_THREADS", str(_default_threads)))))
except (RuntimeError, ValueError):
    pass

# ── 50 Standard ESC-50 Classes ──────────────────────────────────────
ESC50_CATEGORIES = [
    {"id": 0, "name": "dog", "emoji": "🐕", "group": "Animals"},
    {"id": 1, "name": "rooster", "emoji": "🐓", "group": "Animals"},
    {"id": 2, "name": "pig", "emoji": "🐖", "group": "Animals"},
    {"id": 3, "name": "cow", "emoji": "🐄", "group": "Animals"},
    {"id": 4, "name": "frog", "emoji": "🐸", "group": "Animals"},
    {"id": 5, "name": "cat", "emoji": "🐈", "group": "Animals"},
    {"id": 6, "name": "hen", "emoji": "🐔", "group": "Animals"},
    {"id": 7, "name": "insects", "emoji": "🦗", "group": "Animals"},
    {"id": 8, "name": "sheep", "emoji": "🐑", "group": "Animals"},
    {"id": 9, "name": "crow", "emoji": "🐦‍⬛", "group": "Animals"},
    {"id": 10, "name": "rain", "emoji": "🌧️", "group": "Natural"},
    {"id": 11, "name": "sea_waves", "emoji": "🌊", "group": "Natural"},
    {"id": 12, "name": "crackling_fire", "emoji": "🔥", "group": "Natural"},
    {"id": 13, "name": "crickets", "emoji": "🦗", "group": "Natural"},
    {"id": 14, "name": "chirping_birds", "emoji": "🐦", "group": "Natural"},
    {"id": 15, "name": "water_drops", "emoji": "💧", "group": "Natural"},
    {"id": 16, "name": "wind", "emoji": "💨", "group": "Natural"},
    {"id": 17, "name": "pouring_water", "emoji": "🫗", "group": "Natural"},
    {"id": 18, "name": "toilet_flush", "emoji": "🚽", "group": "Natural"},
    {"id": 19, "name": "thunderstorm", "emoji": "⛈️", "group": "Natural"},
    {"id": 20, "name": "crying_baby", "emoji": "👶", "group": "Human"},
    {"id": 21, "name": "sneezing", "emoji": "🤧", "group": "Human"},
    {"id": 22, "name": "clapping", "emoji": "👏", "group": "Human"},
    {"id": 23, "name": "breathing", "emoji": "🫁", "group": "Human"},
    {"id": 24, "name": "coughing", "emoji": "😷", "group": "Human"},
    {"id": 25, "name": "footsteps", "emoji": "👣", "group": "Human"},
    {"id": 26, "name": "laughing", "emoji": "😂", "group": "Human"},
    {"id": 27, "name": "brushing_teeth", "emoji": "🪥", "group": "Human"},
    {"id": 28, "name": "snoring", "emoji": "💤", "group": "Human"},
    {"id": 29, "name": "drinking_sipping", "emoji": "🥤", "group": "Human"},
    {"id": 30, "name": "door_wood_knock", "emoji": "🚪", "group": "Interior"},
    {"id": 31, "name": "mouse_click", "emoji": "🖱️", "group": "Interior"},
    {"id": 32, "name": "keyboard_typing", "emoji": "⌨️", "group": "Interior"},
    {"id": 33, "name": "door_wood_creaks", "emoji": "🚪", "group": "Interior"},
    {"id": 34, "name": "can_opening", "emoji": "🥫", "group": "Interior"},
    {"id": 35, "name": "washing_machine", "emoji": "🫧", "group": "Interior"},
    {"id": 36, "name": "vacuum_cleaner", "emoji": "🧹", "group": "Interior"},
    {"id": 37, "name": "clock_alarm", "emoji": "⏰", "group": "Interior"},
    {"id": 38, "name": "clock_tick", "emoji": "⏱️", "group": "Interior"},
    {"id": 39, "name": "glass_breaking", "emoji": "🔨", "group": "Interior"},
    {"id": 40, "name": "helicopter", "emoji": "🚁", "group": "Urban"},
    {"id": 41, "name": "chainsaw", "emoji": "🪚", "group": "Urban"},
    {"id": 42, "name": "siren", "emoji": "🚨", "group": "Urban"},
    {"id": 43, "name": "car_horn", "emoji": "📢", "group": "Urban"},
    {"id": 44, "name": "engine", "emoji": "🚗", "group": "Urban"},
    {"id": 45, "name": "train", "emoji": "🚆", "group": "Urban"},
    {"id": 46, "name": "church_bells", "emoji": "🔔", "group": "Urban"},
    {"id": 47, "name": "airplane", "emoji": "✈️", "group": "Urban"},
    {"id": 48, "name": "fireworks", "emoji": "🎆", "group": "Urban"},
    {"id": 49, "name": "hand_saw", "emoji": "🪚", "group": "Urban"},
]

OUT_OF_SCOPE_CATEGORY = {
    "id": -1,
    "name": "Not in ESC-50 Classes",
    "emoji": "❓",
    "group": "Out of Domain",
    "description": "Sound signature does not match any of the 50 standardized ESC-50 environmental categories."
}

# ── Architecture checkpoint inference ──────────────────────────────
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
ANALYSIS_WINDOW_SECONDS = 5.0
ANALYSIS_WINDOW_HOP_SECONDS = 2.5
MAX_ANALYSIS_WINDOWS = 12
SILENCE_DBFS_THRESHOLD = float(os.environ.get("SOUNDSENSE_SILENCE_DBFS", "-58"))

# ── AST Deployed Checkpoint Configuration ──────────────────────────
import threading

AST_MODEL_ID = os.environ.get(
    "SOUNDSENSE_AST_MODEL", "Adam-ousse/ast-esc50-finetuned-fold1"
)
AST_REVISION = "fb2443ee3c362d6683a8f493eab2d15598360fb6"
AST_CACHE_DIR = CHECKPOINTS_DIR / ".cache" / "huggingface"
AST_SAMPLE_RATE = 16000
_AST_ENGINE: Optional[Dict[str, Any]] = None
_AST_LOCK = threading.Lock()


def _load_ast_engine() -> Dict[str, Any]:
    """Load the local ESC-50 AST checkpoint once. No network I/O."""
    global _AST_ENGINE
    if _AST_ENGINE is not None:
        return _AST_ENGINE
    with _AST_LOCK:
        if _AST_ENGINE is not None:
            return _AST_ENGINE

        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        kwargs = {
            "cache_dir": str(AST_CACHE_DIR),
            "local_files_only": True,
            "revision": AST_REVISION,
        }
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_ID, **kwargs)
            model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_ID, **kwargs)
        except Exception as err:
            raise FileNotFoundError(
                f"AST checkpoint not installed or failed to load offline: {err}. "
                "Run `python -m eval.fetch_ast` to bootstrap weights."
            ) from err

        model.eval()

        name_to_id = {category["name"]: category["id"] for category in ESC50_CATEGORIES}
        label_map: Dict[int, int] = {}
        for raw_index, label in model.config.id2label.items():
            output_index = int(raw_index)
            key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
            if key in name_to_id:
                label_map[output_index] = name_to_id[key]
            elif key.startswith("label_") and 0 <= output_index < len(ESC50_CATEGORIES):
                # Generic LABEL_n checkpoints use canonical ESC-50 order.
                label_map[output_index] = output_index
        if len(label_map) != len(ESC50_CATEGORIES):
            raise RuntimeError(
                f"AST label mapping incomplete: {len(label_map)}/{len(ESC50_CATEGORIES)}"
            )

        _AST_ENGINE = {
            "feature_extractor": feature_extractor,
            "model": model,
            "label_map": label_map,
            "sampling_rate": AST_SAMPLE_RATE,
            "model_id": AST_MODEL_ID,
            "revision": AST_REVISION,
        }
    return _AST_ENGINE


def _ast_probabilities(y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """Mean class probabilities over 5 s windows. No filename or metadata use."""
    engine = _load_ast_engine()
    if sr != AST_SAMPLE_RATE:
        y = librosa.resample(
            np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=AST_SAMPLE_RATE
        )
    y = np.asarray(y, dtype=np.float32)

    window = int(AST_SAMPLE_RATE * ANALYSIS_WINDOW_SECONDS)
    hop = int(AST_SAMPLE_RATE * ANALYSIS_WINDOW_HOP_SECONDS)
    if y.size <= window:
        starts = [0]
    else:
        starts = list(range(0, y.size - window + 1, hop))
        final_start = y.size - window
        if starts[-1] != final_start:
            starts.append(final_start)
        if len(starts) > MAX_ANALYSIS_WINDOWS:
            starts = starts[:MAX_ANALYSIS_WINDOWS - 1] + [final_start]

    accumulated = np.zeros(len(ESC50_CATEGORIES), dtype=np.float64)
    for start in starts:
        chunk = y[start:start + window]
        if chunk.size < window:
            chunk = np.pad(chunk, (0, window - chunk.size))
        inputs = engine["feature_extractor"](
            chunk, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt"
        )
        with torch.no_grad():
            logits = engine["model"](**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).numpy()
        mapped = np.zeros(len(ESC50_CATEGORIES), dtype=np.float64)
        for output_index, category_id in engine["label_map"].items():
            mapped[category_id] = probs[output_index]
        accumulated += mapped

    accumulated /= max(1, len(starts))
    accumulated /= max(float(accumulated.sum()), 1e-12)
    return accumulated.astype(np.float32), len(starts)


_ENGINE_STATUS: Dict[str, Any] = {
    "state": "not_loaded",
    "modelId": "architecture-checkpoints",
    "error": None,
}


def warm_up_engine() -> bool:
    """Pre-loads the default selected checkpoint (AST first, then resnet18). Safe to call in a background thread."""
    try:
        entry = load_model_checkpoint("ast")
        if entry is not None:
            _ENGINE_STATUS.update(state="ready", modelId=entry["filename"], error=None)
            return True
    except Exception:
        pass

    try:
        entry = load_model_checkpoint("resnet18")
        if entry is not None:
            _ENGINE_STATUS.update(state="ready", modelId=entry["filename"], error=None)
            return True
        _ENGINE_STATUS.update(
            state="unavailable",
            modelId="none",
            error="No trained checkpoints available (AST and ResNet-18 missing).",
        )
        return False
    except Exception as err:
        _ENGINE_STATUS.update(state="unavailable", modelId="none", error=str(err))
        return False


def get_engine_status() -> Dict[str, Any]:
    """Returns the current state of the deployed inference engine."""
    return dict(_ENGINE_STATUS)


def model_status() -> Dict[str, Any]:
    """Return current checkpoint inventory and only harness-generated metrics."""
    report_file = CHECKPOINTS_DIR / "eval_report.json"

    report_data = {}
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        except Exception:
            pass

    expected = {
        "resnet18": "esc50_resnet18_v1.pt",
        "vgg16": "esc50_vgg16_v1.pt",
        "sresnet": "esc50_sresnet_linear_v09.pt",
        "cnngru": "esc50_cnngru_recurrent_v08.pt",
    }
    checkpoints = {}

    # Check local snapshot for AST model.safetensors
    manifest_path = CHECKPOINTS_DIR / "ast_manifest.json"
    ast_manifest = {}
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                ast_manifest = json.load(f)
        except Exception:
            pass
    ast_snapshot_dir = (
        AST_CACHE_DIR
        / f"models--{AST_MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / AST_REVISION
    )
    ast_model_file = ast_snapshot_dir / "model.safetensors"
    ast_present = ast_model_file.is_file()
    checkpoints["ast"] = {
        "filename": f"hf:{AST_MODEL_ID}",
        "available": ast_present,
        "sha256": ast_manifest.get("sha256"),
        "sizeBytes": ast_manifest.get("sizeBytes")
        or (ast_model_file.stat().st_size if ast_present else None),
    }

    for model_name, filename in expected.items():
        path = CHECKPOINTS_DIR / filename
        present = path.is_file()
        digest = None
        size_bytes = None
        if present:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
            size_bytes = path.stat().st_size
        checkpoints[model_name] = {
            "filename": filename,
            "available": present,
            "sha256": digest,
            "sizeBytes": size_bytes,
        }

    return {
        "engineStatus": get_engine_status(),
        "report": report_data,
        "checkpointsDir": str(CHECKPOINTS_DIR),
        "checkpoints": checkpoints,
        "availableModels": [name for name, item in checkpoints.items() if item["available"]],
    }


# ── Local MLP Fallback ──────────────────────────────────────────────
class AcousticClassifierNet(nn.Module):
    def __init__(self, in_features=466, num_classes=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 384),
            nn.BatchNorm1d(384),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        return self.net(x)


_CLASSIFIER_CACHE: Optional[Dict[str, Any]] = None


def get_acoustic_classifier() -> Dict[str, Any]:
    global _CLASSIFIER_CACHE
    if _CLASSIFIER_CACHE is not None:
        return _CLASSIFIER_CACHE

    ckpt_path = CHECKPOINTS_DIR / "acoustic_classifier.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = AcousticClassifierNet(in_features=ckpt["in_features"], num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _CLASSIFIER_CACHE = {
        "model": model,
        "mean_vec": ckpt["mean_vec"],
        "std_vec": ckpt["std_vec"],
        "centroids": ckpt["class_centroids"],
        "radii": ckpt.get("class_radii", np.full(50, 22.0, dtype=np.float32)),
    }
    return _CLASSIFIER_CACHE


# ── Per-Architecture Checkpoint Registry ────────────────────────────
_MODEL_CACHE: Dict[str, Any] = {}

def load_model_checkpoint(model_name: str) -> Optional[Dict[str, Any]]:
    """Load the exact requested architecture; never substitute another model."""
    global _MODEL_CACHE
    norm = re.sub(r"[^a-z0-9]", "", model_name.lower())
    aliases = {
        "ast": "ast", "astesc50": "ast",
        "resnet18": "resnet18", "vgg16": "vgg16", "sresnet": "sresnet",
        "cnngru": "cnngru", "ensemble": "ensemble", "fusion": "ensemble",
    }
    matched_key = aliases.get(norm)
    if matched_key is None:
        raise ValueError(f"Unknown model '{model_name}'. Choose ast, resnet18, vgg16, sresnet, cnngru, or ensemble.")
    if matched_key == "ensemble":
        return None
    if matched_key == "ast":
        engine = _load_ast_engine()
        return {
            "model": engine["model"],
            "feat_type": "ast",
            "filename": f"hf:{AST_MODEL_ID}",
            "arch_key": "ast",
            "temperature": 1.0,
            "metrics": {},
        }
    if matched_key in _MODEL_CACHE:
        return _MODEL_CACHE[matched_key]

    files = {
        "resnet18": ("esc50_resnet18_v1.pt", "log_mel"),
        "vgg16": ("esc50_vgg16_v1.pt", "log_mel"),
        "sresnet": ("esc50_sresnet_linear_v09.pt", "mfcc"),
        "cnngru": ("esc50_cnngru_recurrent_v08.pt", "log_mel"),
    }
    filename, feat_type = files[matched_key]
    ckpt_path = CHECKPOINTS_DIR / filename
    if not ckpt_path.exists():
        return None

    from Preproccesing.train_models import ResNet18Transfer, VGG16Transfer, SResNetFast, CNNGRUFast
    classes = {
        "resnet18": ResNet18Transfer,
        "vgg16": VGG16Transfer,
        "sresnet": SResNetFast,
        "cnngru": CNNGRUFast,
    }
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in data:
        raise RuntimeError(f"Checkpoint '{filename}' has no state_dict.")
    if matched_key in {"resnet18", "vgg16"}:
        model = classes[matched_key](num_classes=50, pretrained=False)
    else:
        model = classes[matched_key](num_classes=50)
    model.load_state_dict(data["state_dict"], strict=True)
    model.eval()
    entry = {
        "model": model,
        "feat_type": feat_type,
        "filename": filename,
        "arch_key": matched_key,
        "temperature": float(data.get("temperature", 1.0)),
        "metrics": data.get("metrics", {}),
    }
    _MODEL_CACHE[matched_key] = entry
    return entry


def load_audio_waveform(
    audio_input: Any,
    target_sr: int = 22050,
    target_duration: Optional[float] = 5.0,
) -> Tuple[np.ndarray, int]:
    """
    Decodes audio from ndarray, raw bytes, or file path into float32 mono waveform,
    resampled to target_sr and aligned to target_duration (None keeps full length).
    """
    sr = target_sr

    if isinstance(audio_input, np.ndarray):
        y = audio_input.astype(np.float32)
    elif isinstance(audio_input, (bytes, bytearray)):
        # Try soundfile container decoding
        try:
            y, sr = sf.read(io.BytesIO(audio_input))
        except Exception:
            try:
                import subprocess
                cmd = ['ffmpeg', '-v', 'error', '-i', 'pipe:0', '-f', 'f32le', '-ar', str(target_sr), '-ac', '1', 'pipe:1']
                proc = subprocess.run(cmd, input=bytes(audio_input), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                if proc.returncode != 0 or not proc.stdout:
                    raise RuntimeError("ffmpeg pipe decode failed")
                y = np.frombuffer(proc.stdout, dtype=np.float32)
                sr = target_sr
            except Exception:
                try:
                    y, sr = librosa.load(io.BytesIO(audio_input), sr=target_sr, mono=True)
                except Exception as err:
                    raise ValueError("Audio could not be decoded as WAV, MP3, MPEG, or WebM.") from err
    else:
        # Load from file path
        try:
            y, sr = sf.read(str(audio_input))
        except Exception:
            try:
                import subprocess
                cmd = ['ffmpeg', '-v', 'error', '-i', str(audio_input), '-f', 'f32le', '-ar', str(target_sr), '-ac', '1', 'pipe:1']
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                if proc.returncode == 0 and len(proc.stdout) > 0:
                    y = np.frombuffer(proc.stdout, dtype=np.float32)
                    sr = target_sr
                else:
                    raise RuntimeError("ffmpeg file decode failed")
            except Exception:
                try:
                    y, sr = librosa.load(str(audio_input), sr=target_sr, mono=True)
                except Exception as err:
                    raise ValueError("Audio could not be decoded as WAV, MP3, MPEG, or WebM.") from err

    y = np.asarray(y, dtype=np.float32)

    # Convert to mono if multichannel
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample if needed. Never pass a waveform to the model at a false sample rate.
    if sr != target_sr and y.size:
        try:
            import soxr
            y = soxr.resample(y, sr, target_sr)
            sr = target_sr
        except Exception:
            try:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except Exception as librosa_error:
                raise ValueError(f"Audio resampling from {sr} Hz to {target_sr} Hz failed.") from librosa_error

    # Ensure valid numbers
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if target_duration is not None:
        target_samples = int(target_sr * target_duration)
        if y.size < target_samples:
            y = np.pad(y, (0, target_samples - y.size), mode="constant")
        elif y.size > target_samples:
            y = y[:target_samples]

    return y.astype(np.float32), sr


def compute_signal_metrics(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Computes RMS / ZCR / spectral flatness in pure vectorized NumPy for high-speed, deadlock-free execution."""
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    zcr = 0.0
    flatness = 0.0
    peak_ratio = 0.0
    dominant_frequency_spread_hz = 0.0
    dominant_frequency_median_hz = 0.0
    if y.size > 1024:
        try:
            # Pure NumPy Zero-Crossing Rate (fast & thread-safe)
            zcr = float(np.mean(np.abs(np.diff(np.sign(y))) > 0)) / 2.0

            # Windowed multi-frame measures distinguish tonal signals from noise.
            w_len = min(2048, y.size)
            hop = max(1, w_len // 2)
            window = np.hanning(w_len)
            flatnesses = []
            peak_ratios = []
            dominant_frequencies = []
            for start in range(0, max(1, y.size - w_len + 1), hop):
                frame = y[start:start + w_len]
                if frame.size < w_len:
                    frame = np.pad(frame, (0, w_len - frame.size))
                spec = np.abs(np.fft.rfft(frame * window)) ** 2 + 1e-20
                flatnesses.append(float(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-20)))
                peak_ratios.append(float(np.max(spec) / (np.sum(spec) + 1e-20)))
                dominant_bin = int(np.argmax(spec))
                dominant_frequencies.append(dominant_bin * sr / w_len)
            flatness = float(np.median(flatnesses))
            peak_ratio = float(np.median(peak_ratios))
            dominant_frequency_median_hz = float(np.median(dominant_frequencies)) if dominant_frequencies else 0.0
            if len(dominant_frequencies) >= 3:
                dominant_frequency_spread_hz = float(
                    np.percentile(dominant_frequencies, 90) - np.percentile(dominant_frequencies, 10)
                )
        except Exception:
            pass
    return {
        "rms": round(rms, 4),
        "dbfs": round(20.0 * np.log10(max(rms, 1e-12)), 2),
        "zcr": round(zcr, 4),
        "flatness": round(flatness, 6),
        "flatness_raw": flatness,
        "spectral_peak_ratio": round(peak_ratio, 6),
        "dominant_frequency_frame_count": len(dominant_frequencies) if y.size > 1024 else 0,
        "dominant_frequency_spread_hz": round(dominant_frequency_spread_hz, 2),
        "dominant_frequency_median_hz": round(dominant_frequency_median_hz, 2),
    }


def extract_acoustic_feature_vector(y: np.ndarray, sr: int = 22050) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extracts the 466-dimensional acoustic summary vector used by the local MLP
    fallback:
    - 128-band Log-Mel spectral envelope (mean, std, max across time)
    - 40-coefficient MFCC timbral profile (mean, std)
    - Macro temporal dynamics & crest factor
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128, fmin=0.0, fmax=11025.0, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1024, hop_length=512)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)

    mel_mean = log_mel.mean(axis=1)  # 128
    mel_std = log_mel.std(axis=1)    # 128
    mel_max = log_mel.max(axis=1)    # 128

    mfcc_mean = mfcc.mean(axis=1)    # 40
    mfcc_std = mfcc.std(axis=1)      # 40

    env = log_mel.mean(axis=0)
    temp_std = float(env.std())
    temp_crest = float(env.max() / (env.mean() + 1e-6))

    raw_feat = np.concatenate([mel_mean, mel_std, mel_max, mfcc_mean, mfcc_std, [temp_std, temp_crest]])

    metrics = compute_signal_metrics(y, sr)
    metrics["temp_std"] = round(temp_std, 4)
    return raw_feat.astype(np.float32), metrics


def _mlp_probabilities(y: np.ndarray, sr: int) -> Tuple[np.ndarray, bool]:
    """Runs the offline fallback MLP. Returns (probabilities, is_out_of_manifold)."""
    data = get_acoustic_classifier()
    feat, _ = extract_acoustic_feature_vector(y, sr)
    feat_norm = (feat.reshape(1, -1) - data["mean_vec"]) / data["std_vec"]

    with torch.no_grad():
        logits = data["model"](torch.tensor(feat_norm, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)[0].numpy()

    centroid_dists = np.linalg.norm(data["centroids"] - feat_norm, axis=1)
    nearest_id = int(np.argmin(centroid_dists))
    is_out_of_manifold = float(centroid_dists[nearest_id]) > float(data["radii"][nearest_id]) * 1.6
    return probs, is_out_of_manifold


def _build_silence_result(metrics: Dict[str, float], model_name: str, started_at: float) -> Dict[str, Any]:
    return {
        "success": True,
        "isOutOfScope": True,
        "category": {
            "id": -1,
            "name": "Not in ESC-50 Classes (Silence / Negligible Signal)",
            "emoji": "🔇",
            "group": "Out of Domain",
            "description": "Acoustic signal energy is too low to extract valid environmental sound signatures."
        },
        "explanation": (
            f"Rule-based silence gate: input level {metrics['dbfs']:.1f} dBFS is below "
            f"the configured {SILENCE_DBFS_THRESHOLD:.1f} dBFS threshold. No classifier confidence was computed."
        ),
        "confidence": None,
        "confidenceType": "not_computed_rule_gate",
        "oodScore": 1.0,
        "predictions": [
            {"category": OUT_OF_SCOPE_CATEGORY, "confidence": None},
        ],
        "metrics": {k: metrics[k] for k in ("rms", "zcr", "flatness", "dbfs")},
        "inferenceTime": round((time.time() - started_at) * 1000, 1),
        "engine": "silence-gate",
        "requestedModel": model_name,
    }


def _build_tone_result(metrics: Dict[str, float], model_name: str, started_at: float) -> Dict[str, Any]:
    return {
        "success": True,
        "isOutOfScope": True,
        "category": OUT_OF_SCOPE_CATEGORY,
        "explanation": (
            "Rule-based tonality gate detected a stable narrow-band signal. "
            "No classifier confidence was computed."
        ),
        "confidence": None,
        "confidenceType": "not_computed_rule_gate",
        "oodScore": 1.0,
        "predictions": [{"category": OUT_OF_SCOPE_CATEGORY, "confidence": None}],
        "metrics": {k: metrics[k] for k in ("rms", "zcr", "flatness", "dbfs", "spectral_peak_ratio", "dominant_frequency_frame_count", "dominant_frequency_spread_hz", "dominant_frequency_median_hz")},
        "inferenceTime": round((time.time() - started_at) * 1000, 1),
        "engine": "tonality-gate",
        "requestedModel": model_name,
    }


def run_acoustic_inference(
    audio_input: bytes | str | Path,
    model_name: str = "ast",
    confidence_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes true audio-based classification. Never reads or inspects the filename.
    Identifies sound solely through its acoustic fingerprint.

    The selected checkpoint is the predictor. Ensemble requests average the
    available architecture checkpoints; unavailable checkpoints fail explicitly
    instead of silently substituting another model.
    """
    t_start = time.time()

    # Measure the original decoded recording before padding or peak normalization.
    y, sr = load_audio_waveform(audio_input, target_sr=22050, target_duration=None)
    metrics = compute_signal_metrics(y, sr)

    silence_rms_threshold = 10.0 ** (SILENCE_DBFS_THRESHOLD / 20.0)
    if metrics["rms"] < silence_rms_threshold:
        return _build_silence_result(metrics, model_name, t_start)
    if (
        metrics["flatness_raw"] < 0.003
        and metrics["spectral_peak_ratio"] > 0.25
        and metrics["dominant_frequency_frame_count"] >= 3
        and metrics["dominant_frequency_median_hz"] > 30.0
        and metrics["dominant_frequency_spread_hz"] < 20.0
    ):
        return _build_tone_result(metrics, model_name, t_start)

    t_infer = time.time()
    norm_name = re.sub(r"[^a-z0-9]", "", model_name.lower())
    if norm_name in {"ast", "astesc50"}:
        probs, window_count = _ast_probabilities(y, sr)
        engine_name = "ast-esc50"
        confidence_type = "model_softmax"
    else:
        is_multi_ensemble = norm_name in {"ensemble", "fusion"}
        model_names = ["resnet18", "vgg16", "sresnet", "cnngru"] if is_multi_ensemble else [model_name]
        entries = [load_model_checkpoint(name) for name in model_names]
        missing = [name for name, entry in zip(model_names, entries) if entry is None]
        if missing:
            raise FileNotFoundError(f"No trained checkpoint is available for: {', '.join(missing)}.")

        window_samples = int(22050 * ANALYSIS_WINDOW_SECONDS)
        hop_samples = int(22050 * ANALYSIS_WINDOW_HOP_SECONDS)
        if y.size < window_samples:
            windows = [np.pad(y, (0, window_samples - y.size))]
        else:
            starts = list(range(0, max(1, y.size - window_samples + 1), hop_samples))
            final_start = y.size - window_samples
            if starts[-1] != final_start:
                starts.append(final_start)
            if len(starts) > MAX_ANALYSIS_WINDOWS:
                starts = starts[:MAX_ANALYSIS_WINDOWS - 1] + [final_start]
            windows = [y[start:start + window_samples] for start in starts]

        from Preproccesing.feature_extractor import extract_log_mel_spectrogram, extract_mfcc
        per_model_log_probs = []
        for entry in entries:
            window_log_probs = []
            for window in windows:
                peak = float(np.max(np.abs(window))) if window.size else 0.0
                model_window = window / peak if peak > 1e-8 else window
                if entry["feat_type"] == "mfcc":
                    feat = extract_mfcc(model_window, sr=22050)
                else:
                    feat = extract_log_mel_spectrogram(model_window, sr=22050)
                tx = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()
                with torch.no_grad():
                    logits = entry["model"](tx)
                temperature = max(0.05, float(entry.get("temperature", 1.0)))
                window_log_probs.append(torch.log_softmax(logits / temperature, dim=1)[0].numpy())
            per_model_log_probs.append(np.mean(window_log_probs, axis=0))

        mean_log_probs = np.mean(per_model_log_probs, axis=0)
        mean_log_probs -= np.max(mean_log_probs)
        probs = np.exp(mean_log_probs)
        probs /= np.sum(probs)
        model_entry = {"arch_key": "ensemble" if is_multi_ensemble else entries[0]["arch_key"]}
        engine_name = f"{model_entry['arch_key']}-checkpoint"
        confidence_type = "temperature_scaled_softmax_score"
        window_count = len(windows)

    order = np.argsort(probs)[::-1]
    top_id = int(order[0])
    top_prob = float(probs[top_id])

    predictions = [
        {"category": ESC50_CATEGORIES[int(idx)], "confidence": round(float(probs[int(idx)]), 4)}
        for idx in order[:5]
    ]

    # Low softmax alone has not been validated as an OOD detector here. Keep it
    # as a class score and let the user's confidence threshold flag uncertainty.
    is_out_of_scope = False
    ood_score = float(1.0 - top_prob)

    elapsed_ms = round((time.time() - t_infer) * 1000, 1)

    result: Dict[str, Any] = {
        "success": True,
        "engine": engine_name,
        "requestedModel": model_name,
        "metrics": {k: v for k, v in metrics.items() if k != "flatness_raw"},
        "inferenceTime": elapsed_ms,
        "windowCount": window_count,
        "classProbabilities": [float(p) for p in probs],
        "oodScore": round(max(0.0, min(1.0, ood_score)), 4),
        "confidenceType": confidence_type,
        "belowConfidenceThreshold": (
            top_prob < max(0.0, min(1.0, confidence_threshold))
            if confidence_threshold is not None else False
        ),
        "confidenceThreshold": (
            max(0.0, min(1.0, confidence_threshold))
            if confidence_threshold is not None else None
        ),
    }

    result.update({
        "isOutOfScope": False,
        "category": ESC50_CATEGORIES[top_id],
        "explanation": None,
        "confidence": round(top_prob, 4),
        "predictions": predictions,
    })
    return result


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        target_model = sys.argv[2] if len(sys.argv) > 2 else "ast"
        target_threshold = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
        try:
            if target_file == "-":
                content = sys.stdin.buffer.read()
            else:
                with open(target_file, "rb") as f:
                    content = f.read()
            out = run_acoustic_inference(content, target_model, target_threshold)
            print(json.dumps(out))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))
    else:
        print("Usage: python acoustic_inference.py <file_path or -> [model_name]")
