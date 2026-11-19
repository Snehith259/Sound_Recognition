"""
SoundSense — Real Acoustic Feature Extraction & Neural Audio Classification
Implements genuine inference based strictly on audio content and spectral signatures.
Zero filename-based guesswork. Supports Out-of-Distribution (OOD) sound detection.

Deployed engine: AST (Audio Spectrogram Transformer) fine-tuned on ESC-50
(93.5% top-1 on the ESC-50 fold-1 test split). A lightweight local MLP
checkpoint is kept as an offline fallback so the API still responds when the
Hugging Face weights cannot be reached.
"""

import io
import os
import re
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn

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

CATEGORY_BY_NAME = {cat["name"].lower(): cat for cat in ESC50_CATEGORIES}

OUT_OF_SCOPE_CATEGORY = {
    "id": -1,
    "name": "Not in ESC-50 Classes",
    "emoji": "❓",
    "group": "Out of Domain",
    "description": "Sound signature does not match any of the 50 standardized ESC-50 environmental categories."
}

# ── Deployed Pretrained ESC-50 Engine ───────────────────────────────
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
HF_CACHE_DIR = CHECKPOINTS_DIR / ".cache" / "huggingface"

# AST fine-tuned on ESC-50 (canonical ESC-50 label order, 93.5% fold-1 test accuracy).
HF_MODEL_ID = os.environ.get("SOUNDSENSE_HF_MODEL", "Adam-ousse/ast-esc50-finetuned-fold1")

AST_SAMPLE_RATE = 16000
ANALYSIS_WINDOW_SECONDS = 5.0
MAX_ANALYSIS_WINDOWS = 4
SILENCE_RMS_THRESHOLD = 0.005
PURE_TONE_FLATNESS_THRESHOLD = 1e-5
OOD_CONFIDENCE_THRESHOLD = float(os.environ.get("SOUNDSENSE_OOD_THRESHOLD", "0.30"))

_ENGINE_LOCK = threading.Lock()
_ENGINE: Optional[Dict[str, Any]] = None
_ENGINE_STATUS: Dict[str, Any] = {
    "state": "not_loaded",
    "modelId": HF_MODEL_ID,
    "error": None,
}


def _normalise_label(label: Any) -> str:
    """Normalises model label names ('Sea Waves', 'sea-waves') to ESC-50 names."""
    return re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")


def _load_ast_engine() -> Dict[str, Any]:
    """Lazily loads and caches the pretrained AST ESC-50 classifier."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE

        _ENGINE_STATUS.update(state="loading", error=None)
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                HF_MODEL_ID, cache_dir=str(HF_CACHE_DIR)
            )
            model = AutoModelForAudioClassification.from_pretrained(
                HF_MODEL_ID, cache_dir=str(HF_CACHE_DIR)
            )
            model.eval()

            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            label_to_category: Dict[int, Dict[str, Any]] = {}
            for out_idx, label in id2label.items():
                category = CATEGORY_BY_NAME.get(_normalise_label(label))
                if category is None and _normalise_label(label).startswith("label_"):
                    # Some checkpoints keep generic LABEL_n names; ESC-50 order is canonical.
                    if 0 <= out_idx < len(ESC50_CATEGORIES):
                        category = ESC50_CATEGORIES[out_idx]
                if category is not None:
                    label_to_category[out_idx] = category

            if not label_to_category:
                raise RuntimeError(
                    f"None of the model labels in '{HF_MODEL_ID}' map to ESC-50 classes."
                )

            _ENGINE = {
                "feature_extractor": feature_extractor,
                "model": model,
                "label_to_category": label_to_category,
                "sampling_rate": int(getattr(feature_extractor, "sampling_rate", AST_SAMPLE_RATE) or AST_SAMPLE_RATE),
                "model_id": HF_MODEL_ID,
            }
            _ENGINE_STATUS.update(state="ready", error=None)
        except Exception as err:
            _ENGINE_STATUS.update(state="unavailable", error=str(err))
            raise

    return _ENGINE


def warm_up_engine() -> bool:
    """Pre-loads the deployed model. Safe to call from a background thread."""
    try:
        _load_ast_engine()
        return True
    except Exception:
        return False


def get_engine_status() -> Dict[str, Any]:
    """Returns the current state of the deployed inference engine."""
    return dict(_ENGINE_STATUS)


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
                import scipy.io.wavfile as wavfile
                sr, y = wavfile.read(io.BytesIO(audio_input))
                if y.dtype == np.int16:
                    y = y.astype(np.float32) / 32768.0
                elif y.dtype == np.int32:
                    y = y.astype(np.float32) / 2147483648.0
            except Exception:
                try:
                    y, sr = librosa.load(io.BytesIO(audio_input), sr=target_sr, mono=True)
                except Exception:
                    # Raw PCM fallback
                    try:
                        y = np.frombuffer(audio_input, dtype=np.float32)
                    except Exception:
                        y = np.zeros(int(target_sr * (target_duration or 1.0)), dtype=np.float32)
    else:
        # Load from file path
        try:
            y, sr = sf.read(str(audio_input))
        except Exception:
            y, sr = librosa.load(str(audio_input), sr=target_sr, mono=True)

    y = np.asarray(y, dtype=np.float32)

    # Convert to mono if multichannel
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample if needed
    if sr != target_sr and y.size:
        try:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception:
            pass

    # Ensure valid numbers
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if target_duration is not None:
        target_samples = int(target_sr * target_duration)
        if y.size < target_samples:
            y = np.pad(y, (0, target_samples - y.size), mode="constant")
        elif y.size > target_samples:
            y = y[:target_samples]

    # Peak normalization
    max_val = float(np.max(np.abs(y))) if y.size else 0.0
    if max_val > 1e-6:
        y = y / max_val

    return y.astype(np.float32), sr


def compute_signal_metrics(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Computes RMS / ZCR / spectral flatness used for silence and OOD checks."""
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    zcr = 0.0
    flatness = 0.0
    if y.size > 1024:
        try:
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512)))
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=512)))
        except Exception:
            pass
    return {
        "rms": round(rms, 4),
        "zcr": round(zcr, 4),
        "flatness": round(flatness, 6),
        "flatness_raw": flatness,
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


def _ast_probabilities(y: np.ndarray, sr: int) -> np.ndarray:
    """Runs the pretrained AST ESC-50 classifier over 5-second windows."""
    engine = _load_ast_engine()
    feature_extractor = engine["feature_extractor"]
    model = engine["model"]
    target_sr = engine["sampling_rate"]

    if sr != target_sr and y.size:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    window = max(1, int(target_sr * ANALYSIS_WINDOW_SECONDS))
    chunks = [y[i:i + window] for i in range(0, max(1, y.size), window)][:MAX_ANALYSIS_WINDOWS]
    chunks = [chunk for chunk in chunks if chunk.size]
    if not chunks:
        chunks = [np.zeros(window, dtype=np.float32)]

    accumulated = np.zeros(len(ESC50_CATEGORIES), dtype=np.float64)
    for chunk in chunks:
        inputs = feature_extractor(chunk, sampling_rate=target_sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Map the model's output indices onto canonical ESC-50 class ids.
        mapped = np.zeros(len(ESC50_CATEGORIES), dtype=np.float64)
        for out_idx, category in engine["label_to_category"].items():
            if 0 <= out_idx < probs.shape[0]:
                mapped[category["id"]] += float(probs[out_idx])
        accumulated += mapped

    accumulated /= len(chunks)
    return accumulated.astype(np.float32)


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
        "explanation": "RMS energy is below acoustic detection threshold (ambient silence or muted mic).",
        "confidence": 0.95,
        "predictions": [
            {"category": OUT_OF_SCOPE_CATEGORY, "confidence": 0.95},
            {"category": ESC50_CATEGORIES[16], "confidence": 0.02},  # wind
            {"category": ESC50_CATEGORIES[23], "confidence": 0.01},  # breathing
            {"category": ESC50_CATEGORIES[38], "confidence": 0.01},  # clock tick
            {"category": ESC50_CATEGORIES[31], "confidence": 0.01},  # mouse click
        ],
        "metrics": {"rms": metrics["rms"], "zcr": metrics["zcr"], "flatness": metrics["flatness"]},
        "inferenceTime": round((time.time() - started_at) * 1000, 1),
        "engine": "silence-gate",
        "requestedModel": model_name,
    }


def run_acoustic_inference(audio_input: bytes | str | Path, model_name: str = "resnet18") -> Dict[str, Any]:
    """
    Executes true audio-based classification. Never reads or inspects the filename.
    Identifies sound solely through its acoustic fingerprint.

    `model_name` is accepted for API compatibility with the UI model selector; the
    deployed pretrained ESC-50 engine is used for every selection.
    """
    t_start = time.time()

    # Decode full waveform at 22050 Hz (MLP fallback feature rate).
    y, sr = load_audio_waveform(audio_input, target_sr=22050, target_duration=None)
    metrics = compute_signal_metrics(y, sr)

    # Silence / near-silent recording (< -46 dB)
    if metrics["rms"] < SILENCE_RMS_THRESHOLD:
        return _build_silence_result(metrics, model_name, t_start)

    engine_name = "ast-esc50"
    engine_error: Optional[str] = None
    is_out_of_manifold = False
    t_infer = time.time()

    try:
        probs = _ast_probabilities(y, sr)
    except Exception as err:
        # Offline fallback: local MLP checkpoint.
        engine_name = "acoustic-mlp-fallback"
        engine_error = str(err)
        probs, is_out_of_manifold = _mlp_probabilities(y, sr)

    order = np.argsort(probs)[::-1]
    top_id = int(order[0])
    top_prob = float(probs[top_id])

    predictions = [
        {"category": ESC50_CATEGORIES[int(idx)], "confidence": round(float(probs[int(idx)]), 4)}
        for idx in order[:5]
    ]

    is_pure_tone = metrics["flatness_raw"] < PURE_TONE_FLATNESS_THRESHOLD
    is_low_confidence = top_prob < OOD_CONFIDENCE_THRESHOLD
    is_out_of_scope = is_pure_tone or is_low_confidence or is_out_of_manifold

    elapsed_ms = round((time.time() - t_infer) * 1000, 1)

    result: Dict[str, Any] = {
        "success": True,
        "engine": engine_name,
        "requestedModel": model_name,
        "metrics": {k: v for k, v in metrics.items() if k != "flatness_raw"},
        "inferenceTime": elapsed_ms,
    }
    if engine_error:
        result["engineError"] = engine_error

    if is_out_of_scope:
        if is_pure_tone:
            ood_conf = 0.98
            ood_reason = (
                "Synthetic pure-tone or narrow-band signal detected. Environmental sounds "
                "exhibit natural multi-harmonic dispersion and broadband energy."
            )
        elif is_out_of_manifold:
            ood_conf = min(0.99, 1.0 - top_prob)
            ood_reason = (
                "Acoustic fingerprint lies outside the manifold of all 50 ESC-50 categories "
                "(nearest centroid distance exceeds that class's natural radius)."
            )
        else:
            ood_conf = 1.0 - top_prob
            ood_reason = (
                "Model confidence is too low to assign the sound to any of the 50 ESC-50 "
                "environmental categories."
            )
        ood_conf = round(max(0.5, min(0.99, ood_conf)), 4)

        result.update({
            "isOutOfScope": True,
            "category": OUT_OF_SCOPE_CATEGORY,
            "explanation": ood_reason,
            "confidence": ood_conf,
            "predictions": [
                {"category": OUT_OF_SCOPE_CATEGORY, "confidence": ood_conf},
                *predictions[:4],
            ],
        })
        return result

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
        target_model = sys.argv[2] if len(sys.argv) > 2 else "resnet18"
        try:
            with open(target_file, "rb") as f:
                content = f.read()
            out = run_acoustic_inference(content, target_model)
            print(json.dumps(out))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))
    else:
        print("Usage: python acoustic_inference.py <file_path> [model_name]")
