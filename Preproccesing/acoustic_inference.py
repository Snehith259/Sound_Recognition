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
AST_SHA256 = "f3b7ad3ffc95f5d867c7bb088da394bf048e649850535e9faa1f75276e0c6617"
AST_CACHE_DIR = CHECKPOINTS_DIR / ".cache" / "huggingface"
AST_FINETUNED_DIR = CHECKPOINTS_DIR / "ast_finetuned"
AST_SAMPLE_RATE = 16000
_AST_ENGINE: Optional[Dict[str, Any]] = None
_AST_LOCK = threading.Lock()


def _verify_file_sha256(path: Path, expected_sha256: str) -> None:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    actual = hasher.hexdigest().lower()
    if actual != expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 verification failed for {path.name}: expected {expected_sha256}, got {actual}"
        )


def _load_ast_engine() -> Dict[str, Any]:
    """Load the local ESC-50 AST checkpoint once, verifying SHA-256 when using the pinned model."""
    global _AST_ENGINE
    if _AST_ENGINE is not None:
        return _AST_ENGINE
    with _AST_LOCK:
        if _AST_ENGINE is not None:
            return _AST_ENGINE

        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        temperature = 1.0
        if (AST_FINETUNED_DIR / "model.safetensors").is_file():
            feature_extractor = AutoFeatureExtractor.from_pretrained(str(AST_FINETUNED_DIR), local_files_only=True)
            model = AutoModelForAudioClassification.from_pretrained(str(AST_FINETUNED_DIR), local_files_only=True)
            temp_file = AST_FINETUNED_DIR / "ast_temperature.json"
            if temp_file.is_file():
                temperature = float(json.loads(temp_file.read_text(encoding="utf-8")).get("temperature", 1.0))
            loaded_id = f"local:{AST_FINETUNED_DIR.name}"
        else:
            kwargs = {
                "cache_dir": str(AST_CACHE_DIR),
                "local_files_only": True,
                "revision": AST_REVISION,
            }
            try:
                feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_ID, **kwargs)
                model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_ID, **kwargs)
            except Exception as offline_err:
                try:
                    # If offline cache is absent (e.g. freshly deployed container without pre-baked cache),
                    # fetch from Hugging Face Hub and verify pinned revision.
                    kwargs["local_files_only"] = False
                    feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_ID, **kwargs)
                    model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_ID, **kwargs)
                except Exception as online_err:
                    raise FileNotFoundError(
                        f"AST checkpoint not installed offline ({offline_err}) and online download failed: {online_err}. "
                        "Run `python -m eval.fetch_ast` to bootstrap weights."
                    ) from online_err

            if AST_MODEL_ID == "Adam-ousse/ast-esc50-finetuned-fold1":
                sf_file = (
                    AST_CACHE_DIR
                    / f"models--{AST_MODEL_ID.replace('/', '--')}"
                    / "snapshots"
                    / AST_REVISION
                    / "model.safetensors"
                )
                if sf_file.is_file():
                    _verify_file_sha256(sf_file, AST_SHA256)
            loaded_id = AST_MODEL_ID

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
            "model_id": loaded_id,
            "revision": AST_REVISION,
            "temperature": max(0.05, min(20.0, temperature)),
        }
    return _AST_ENGINE


def _slice_analysis_windows(
    y: np.ndarray,
    sr: int,
    silence_rms_threshold: float,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Slice 5 s windows without tiling short clips or double-counting tiny codec tails."""
    window = int(sr * ANALYSIS_WINDOW_SECONDS)
    hop = int(sr * ANALYSIS_WINDOW_HOP_SECONDS)
    if y.size <= window:
        chunks = [np.pad(y, (0, window - y.size)) if y.size < window else y]
        return chunks, np.array([1.0], dtype=np.float64)

    starts = list(range(0, y.size - window + 1, hop))
    final_start = y.size - window
    # Ignore tiny <25% hop overflow caused by MP3/AAC encoder priming delay
    if (final_start - starts[-1]) >= max(1, int(0.25 * hop)):
        starts.append(final_start)
    if len(starts) > MAX_ANALYSIS_WINDOWS:
        starts = starts[:MAX_ANALYSIS_WINDOWS - 1] + [final_start]
    chunks = [y[s:s + window] for s in starts]
    w_rms = np.array([float(np.sqrt(np.mean(c ** 2))) if c.size else 0.0 for c in chunks], dtype=np.float64)
    max_rms = float(np.max(w_rms))
    if max_rms > 1e-8:
        win_weights = np.where(w_rms >= max(silence_rms_threshold, 0.15 * max_rms), w_rms / max_rms, 0.0)
        w_sum = float(np.sum(win_weights))
        win_weights = win_weights / w_sum if w_sum > 1e-8 else np.ones(len(chunks)) / len(chunks)
    else:
        win_weights = np.ones(len(chunks)) / len(chunks)
    return chunks, win_weights


def _ast_probabilities(y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """RMS-weighted, temperature-scaled class probabilities over 5 s windows. No filename or metadata use."""
    engine = _load_ast_engine()
    if sr != AST_SAMPLE_RATE:
        y = librosa.resample(
            np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=AST_SAMPLE_RATE
        )
    y = np.asarray(y, dtype=np.float32)

    silence_rms_threshold = 10.0 ** (SILENCE_DBFS_THRESHOLD / 20.0)
    chunks, win_weights = _slice_analysis_windows(y, AST_SAMPLE_RATE, silence_rms_threshold)

    temp = float(engine.get("temperature", 1.0))
    label_items = sorted(engine["label_map"].items())
    out_indices = np.array([out_idx for out_idx, _ in label_items], dtype=np.int64)
    cat_indices = np.array([cat_id for _, cat_id in label_items], dtype=np.int64)
    accumulated = np.zeros(len(ESC50_CATEGORIES), dtype=np.float64)
    batch_size = 6
    for start_idx in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start_idx:start_idx + batch_size]
        batch_weights = win_weights[start_idx:start_idx + batch_size]
        inputs = engine["feature_extractor"](
            batch_chunks, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = engine["model"](**inputs).logits / temp
        probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()
        mapped = np.zeros((probs_batch.shape[0], len(ESC50_CATEGORIES)), dtype=np.float64)
        mapped[:, cat_indices] = probs_batch[:, out_indices]
        accumulated += np.sum(mapped * batch_weights[:, None], axis=0)

    accumulated /= max(float(accumulated.sum()), 1e-12)
    return accumulated.astype(np.float32), len(chunks)


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
    finetuned_sf = AST_FINETUNED_DIR / "model.safetensors"
    ast_snapshot_dir = (
        AST_CACHE_DIR
        / f"models--{AST_MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / AST_REVISION
    )
    ast_model_file = finetuned_sf if finetuned_sf.is_file() else (ast_snapshot_dir / "model.safetensors")
    ast_present = ast_model_file.is_file() or (_AST_ENGINE is not None)
    ast_sha = None
    ast_size = None
    if finetuned_sf.is_file():
        ast_size = finetuned_sf.stat().st_size
        temp_file = AST_FINETUNED_DIR / "ast_temperature.json"
        if temp_file.is_file():
            try:
                ast_sha = json.loads(temp_file.read_text(encoding="utf-8")).get("sha256")
            except Exception:
                ast_sha = None
        if not ast_sha:
            hasher = hashlib.sha256()
            with finetuned_sf.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            ast_sha = hasher.hexdigest()
    checkpoints["ast"] = {
        "filename": f"local:{AST_FINETUNED_DIR.name}" if finetuned_sf.is_file() else f"hf:{AST_MODEL_ID}",
        "available": ast_present,
        "sha256": ast_sha or ast_manifest.get("sha256"),
        "sizeBytes": ast_size or ast_manifest.get("sizeBytes") or (ast_model_file.stat().st_size if ast_model_file.is_file() else None),
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
            "temperature": float(engine.get("temperature", 1.0)),
            "metrics": {},
        }
    if matched_key in _MODEL_CACHE:
        return _MODEL_CACHE[matched_key]

    files = {
        "resnet18": ("esc50_resnet18_v1.pt", "log_mel"),
        "vgg16": ("esc50_vgg16_v1.pt", "log_mel"),
        "sresnet": ("esc50_sresnet_linear_v09.pt", "log_mel"),
        "cnngru": ("esc50_cnngru_recurrent_v08.pt", "log_mel"),
    }
    filename, feat_type = files[matched_key]
    ckpt_path = CHECKPOINTS_DIR / filename
    if not ckpt_path.exists():
        return None

    from Preproccesing.train_models import (
        ResNet18Transfer,
        VGG16Transfer,
        SResNetTransfer,
        CNNGRUFast,
        ResNet18V2,
        VGG16V2,
        SResNetV2,
        CNNGRUV2,
        ResNet18V3,
        VGG16V3,
        SResNetV3,
        CNNGRUV3,
        ResNet22AudioV4,
        Cnn10AudioV4,
        DyMNAudioV4,
        MN20GRUAudioV4,
    )
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in data:
        raise RuntimeError(f"Checkpoint '{filename}' has no state_dict.")
    fmt = str(data.get("format", ""))
    default_ver = 4 if fmt.endswith("-v4") else (3 if fmt.endswith("-v3") else (2 if fmt.endswith("-v2") else 1))
    arch_version = int(data.get("arch_version") or default_ver)
    feat_type = str(data.get("feature_type") or feat_type)
    classes_by_version = {
        1: {
            "resnet18": ResNet18Transfer,
            "vgg16": VGG16Transfer,
            "sresnet": SResNetTransfer,
            "cnngru": CNNGRUFast,
        },
        2: {
            "resnet18": ResNet18V2,
            "vgg16": VGG16V2,
            "sresnet": SResNetV2,
            "cnngru": CNNGRUV2,
        },
        3: {
            "resnet18": ResNet18V3,
            "vgg16": VGG16V3,
            "sresnet": SResNetV3,
            "cnngru": CNNGRUV3,
        },
        4: {
            "resnet18": ResNet22AudioV4,
            "vgg16": Cnn10AudioV4,
            "sresnet": DyMNAudioV4,
            "cnngru": MN20GRUAudioV4,
        },
    }
    if arch_version not in classes_by_version:
        raise ValueError(f"Unsupported checkpoint arch_version={arch_version} in '{filename}'.")
    cls = classes_by_version[arch_version][matched_key]
    if matched_key in {"resnet18", "vgg16", "sresnet"} or arch_version >= 4:
        model = cls(num_classes=50, pretrained=False)
    else:
        model = cls(num_classes=50)
    # Checkpoints may store frozen AudioSet backbones in float16 for compact deployment
    state_dict = {
        k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
        for k, v in data["state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    entry = {
        "model": model,
        "feat_type": feat_type,
        "filename": filename,
        "arch_key": matched_key,
        "arch_version": arch_version,
        "temperature": float(data.get("temperature", 1.0)),
        "metrics": data.get("metrics", {}),
    }
    _MODEL_CACHE[matched_key] = entry
    return entry


def _ffmpeg_decode_audio(source: bytes | str) -> Tuple[np.ndarray, int]:
    """Decode arbitrary container audio via ffmpeg AU stream so native sample rate reaches soxr."""
    import subprocess
    inp = "pipe:0" if isinstance(source, (bytes, bytearray)) else str(source)
    cmd = ["ffmpeg", "-v", "error", "-i", inp, "-ac", "1", "-c:a", "pcm_f32be", "-f", "au", "pipe:1"]
    proc = subprocess.run(
        cmd,
        input=bytes(source) if isinstance(source, (bytes, bytearray)) else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='ignore')[:200]}")
    y_dec, sr_dec = sf.read(io.BytesIO(proc.stdout))
    if np.asarray(y_dec).size == 0:
        raise RuntimeError("ffmpeg produced empty audio stream")
    return np.asarray(y_dec, dtype=np.float32), int(sr_dec)


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
        # Try soundfile container decoding first
        try:
            y, sr = sf.read(io.BytesIO(audio_input))
        except Exception:
            try:
                y, sr = _ffmpeg_decode_audio(bytes(audio_input))
            except Exception:
                try:
                    y, sr = librosa.load(io.BytesIO(audio_input), sr=target_sr, mono=True)
                except Exception as err:
                    raise ValueError("Audio could not be decoded as WAV, MP3, MPEG, WebM, or AAC.") from err
    else:
        # Load from file path
        try:
            y, sr = sf.read(str(audio_input))
        except Exception:
            try:
                y, sr = _ffmpeg_decode_audio(str(audio_input))
            except Exception:
                try:
                    y, sr = librosa.load(str(audio_input), sr=target_sr, mono=True)
                except Exception as err:
                    raise ValueError("Audio could not be decoded as WAV, MP3, MPEG, WebM, or AAC.") from err

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

    norm_name = re.sub(r"[^a-z0-9]", "", model_name.lower())
    is_ast_only = norm_name in {"ast", "astesc50"}
    is_multi_ensemble = norm_name in {"ensemble", "fusion"}
    entries: list[Dict[str, Any]] = []
    if is_ast_only:
        decode_sr = AST_SAMPLE_RATE
    else:
        model_names = ["resnet18", "vgg16", "sresnet", "cnngru"] if is_multi_ensemble else [model_name]
        entries = [load_model_checkpoint(name) for name in model_names]
        missing = [name for name, entry in zip(model_names, entries) if entry is None]
        if missing:
            raise FileNotFoundError(f"No trained checkpoint is available for: {', '.join(missing)}.")
        decode_sr = 32000 if any(entry["feat_type"] == "wave_32k" for entry in entries) else 22050
    # Measure the original decoded recording before padding or peak normalization.
    y, sr = load_audio_waveform(audio_input, target_sr=decode_sr, target_duration=None)
    metrics = compute_signal_metrics(y, sr)

    silence_rms_threshold = 10.0 ** (SILENCE_DBFS_THRESHOLD / 20.0)
    if metrics["rms"] < silence_rms_threshold:
        return _build_silence_result(metrics, model_name, t_start)
    if (
        metrics["flatness_raw"] < 0.0005
        and metrics["spectral_peak_ratio"] > 0.45
        and metrics["dominant_frequency_frame_count"] >= 3
        and metrics["dominant_frequency_median_hz"] > 30.0
        and metrics["dominant_frequency_spread_hz"] < 5.0
    ):
        return _build_tone_result(metrics, model_name, t_start)

    t_infer = time.time()
    as_ood_probs: Optional[np.ndarray] = None
    if is_ast_only:
        probs, window_count = _ast_probabilities(y, sr)
        engine_name = "ast-esc50"
        confidence_type = "model_softmax"
    else:
        # Include AST in ensemble for stronger fusion when available
        include_ast_in_ensemble = False
        if is_multi_ensemble:
            try:
                _load_ast_engine()
                include_ast_in_ensemble = True
            except Exception:
                pass

        feat_types = {entry["feat_type"] for entry in entries}
        if feat_types & {"log_mel", "mfcc"}:
            if sr != 22050:
                try:
                    import soxr
                    y_22k = soxr.resample(y, sr, 22050)
                except Exception:
                    y_22k = librosa.resample(y, orig_sr=sr, target_sr=22050)
            else:
                y_22k = y
            windows_22k, win_weights_22k = _slice_analysis_windows(np.asarray(y_22k, dtype=np.float32), 22050, silence_rms_threshold)
        else:
            windows_22k, win_weights_22k = ([], np.array([1.0]))
        if "wave_32k" in feat_types:
            try:
                import soxr
                y_32k = soxr.resample(y, sr, 32000) if sr != 32000 else y
            except Exception:
                y_32k = librosa.resample(y, orig_sr=sr, target_sr=32000) if sr != 32000 else y
            windows_32k, win_weights_32k = _slice_analysis_windows(np.asarray(y_32k, dtype=np.float32), 32000, silence_rms_threshold)
        else:
            windows_32k, win_weights_32k = ([], np.array([1.0]))

        from Preproccesing.feature_extractor import extract_log_mel_spectrogram, extract_mfcc
        batch_by_feat: Dict[str, torch.Tensor] = {}
        weights_by_feat: Dict[str, np.ndarray] = {}
        for feat_type in feat_types:
            if feat_type == "wave_32k":
                tensors = []
                for window in windows_32k:
                    peak = float(np.max(np.abs(window))) if window.size else 0.0
                    model_window = window / peak if peak > 1e-8 else window
                    tensors.append(torch.from_numpy(model_window.astype(np.float32)).unsqueeze(0))
                batch_by_feat[feat_type] = torch.cat(tensors, dim=0)
                weights_by_feat[feat_type] = win_weights_32k
            else:
                tensors = []
                for window in windows_22k:
                    peak = float(np.max(np.abs(window))) if window.size else 0.0
                    model_window = window / peak if peak > 1e-8 else window
                    feat = extract_mfcc(model_window, sr=22050) if feat_type == "mfcc" else extract_log_mel_spectrogram(model_window, sr=22050)
                    tensors.append(torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float())
                batch_by_feat[feat_type] = torch.cat(tensors, dim=0)
                weights_by_feat[feat_type] = win_weights_22k

        ensemble_cfg_path = CHECKPOINTS_DIR / "ensemble_config.json"
        ensemble_cfg: Dict[str, Any] = {}
        if ensemble_cfg_path.is_file():
            try:
                ensemble_cfg = json.loads(ensemble_cfg_path.read_text(encoding="utf-8"))
            except Exception:
                ensemble_cfg = {}

        arch_versions = {int(entry.get("arch_version", 1)) for entry in entries}
        ensemble_arch = next(iter(arch_versions)) if len(arch_versions) == 1 else max(arch_versions)
        is_modern = ensemble_arch >= 2
        if is_multi_ensemble and ensemble_cfg.get("archVersion") == ensemble_arch:
            ensemble_weights = ensemble_cfg.get("weights") or {"sresnet": 0.34, "vgg16": 0.28, "resnet18": 0.26, "cnngru": 0.12}
            ens_temp = max(0.05, min(20.0, float(ensemble_cfg.get("temperature", 1.0))))
            ens_fusion = str(ensemble_cfg.get("fusion", "weighted_log_prob"))
        else:
            ensemble_weights = (
                {"sresnet": 0.34, "cnngru": 0.28, "vgg16": 0.20, "resnet18": 0.18}
                if ensemble_arch >= 4
                else ({"sresnet": 0.34, "vgg16": 0.28, "resnet18": 0.26, "cnngru": 0.12} if is_modern else {"sresnet": 0.30, "vgg16": 0.32, "resnet18": 0.28, "cnngru": 0.10})
            )
            ens_temp = 1.0
            ens_fusion = "hybrid" if ensemble_arch >= 4 else ("weighted_log_prob" if is_modern else "weighted_linear")

        if include_ast_in_ensemble:
            cfg_matches_ast = (
                is_multi_ensemble
                and bool(ensemble_cfg.get("includesAst"))
                and ensemble_cfg.get("archVersion") == ensemble_arch
            )
            if not cfg_matches_ast:
                if ensemble_arch >= 4:
                    ensemble_weights = {"sresnet": 0.285, "ast": 0.25, "cnngru": 0.255, "resnet18": 0.12, "vgg16": 0.09}
                    ens_temp = 1.0
                    ens_fusion = "hybrid"
                else:
                    ast_w = 0.72
                    cnn_total = 1.0 - ast_w
                    base_cnn_sum = max(1e-12, sum(float(ensemble_weights.get(k, 0.25)) for k in model_names))
                    ensemble_weights = {
                        k: float(ensemble_weights.get(k, 0.25)) * cnn_total / base_cnn_sum
                        for k in model_names
                    }
                    ensemble_weights["ast"] = ast_w
                    ens_temp = 1.0
                    ens_fusion = "weighted_linear"

        from Preproccesing.audioset_backbones import audioset_ood_probability

        per_model_probs = []
        model_weights = []
        as_ood_scores = []
        as_ood_weights = []
        for entry in entries:
            tx_batch = batch_by_feat[entry["feat_type"]]
            w_win = weights_by_feat[entry["feat_type"]]
            with torch.inference_mode():
                if hasattr(entry["model"], "forward_with_audioset"):
                    logits, as_logits = entry["model"].forward_with_audioset(tx_batch)
                    win_as_ood = float(np.sum(audioset_ood_probability(as_logits) * w_win))
                    as_ood_scores.append(win_as_ood)
                    as_ood_weights.append(float(ensemble_weights.get(entry["arch_key"], 0.25)))
                else:
                    logits = entry["model"](tx_batch)
            raw_temp = float(entry.get("temperature", 1.0))
            temperature = max(0.05, min(20.0, raw_temp)) if raw_temp > 0 else 1.0
            window_probs = torch.softmax(logits / temperature, dim=1).numpy()
            agg_probs = np.sum(window_probs * w_win[:, None], axis=0)
            agg_probs /= np.sum(agg_probs)
            per_model_probs.append(agg_probs)
            model_weights.append(ensemble_weights.get(entry["arch_key"], 0.25))

        if as_ood_scores:
            aw = np.asarray(as_ood_weights, dtype=np.float64)
            aw /= np.sum(aw)
            as_ood_probs = np.asarray(float(np.sum(np.asarray(as_ood_scores) * aw)))

        if include_ast_in_ensemble:
            ast_probs, _ast_wcount = _ast_probabilities(y, sr)
            per_model_probs.append(ast_probs)
            model_weights.append(ensemble_weights.get("ast", 0.25))

        w_arr = np.asarray(model_weights, dtype=np.float64)
        w_arr /= np.sum(w_arr)
        if is_multi_ensemble and ens_fusion in {"weighted_log_prob", "hybrid"}:
            fused_logits = np.sum([w * np.log(np.clip(p, 1e-12, 1.0)) for w, p in zip(w_arr, per_model_probs)], axis=0)
            scaled = fused_logits / ens_temp
            scaled -= np.max(scaled)
            probs_log = np.exp(scaled)
            probs_log /= np.sum(probs_log)
            if ens_fusion == "hybrid":
                probs_lin = np.sum([w * p for w, p in zip(w_arr, per_model_probs)], axis=0)
                probs = 0.5 * probs_lin + 0.5 * (probs_log / np.sum(probs_log))
                probs /= np.sum(probs)
            else:
                probs = probs_log
        else:
            probs = np.sum([w * p for w, p in zip(w_arr, per_model_probs)], axis=0)
            probs /= np.sum(probs)
        model_entry = {"arch_key": "ensemble" if is_multi_ensemble else entries[0]["arch_key"]}
        engine_name = f"{model_entry['arch_key']}-checkpoint"
        confidence_type = "temperature_scaled_softmax_score"
        window_count = len(windows_32k) if "wave_32k" in feat_types else len(windows_22k)

    order = np.argsort(probs)[::-1]
    top_id = int(order[0])
    top_prob = float(probs[top_id])
    second_prob = float(probs[int(order[1])]) if len(order) > 1 else 0.0
    norm_ent = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))) / np.log(len(ESC50_CATEGORIES)))
    # Concentrated 2-sound mixtures retain in-domain support when top-2 dominates remaining 48 classes
    mix_support = min(1.0, top_prob + 0.45 * min(second_prob, top_prob) * max(0.0, 1.0 - norm_ent))
    if as_ood_probs is not None:
        as_ood_f = float(as_ood_probs)
        ood_score = float(max(0.60 * (1.0 - mix_support) + 0.40 * as_ood_f, 0.72 * as_ood_f))
    else:
        ood_score = float(1.0 - mix_support)
    in_domain_conf = 1.0 - ood_score

    predictions = [
        {"category": ESC50_CATEGORIES[int(idx)], "confidence": round(float(probs[int(idx)]), 4)}
        for idx in order[:5]
    ]

    below_thresh = (
        in_domain_conf < max(0.0, min(1.0, confidence_threshold))
        if confidence_threshold is not None else False
    )
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
        "belowConfidenceThreshold": below_thresh,
        "confidenceThreshold": (
            max(0.0, min(1.0, confidence_threshold))
            if confidence_threshold is not None else None
        ),
    }

    if below_thresh:
        ood_cat = {
            "id": -1,
            "name": "Not in ESC-50 Classes (Low Confidence)",
            "emoji": "❓",
            "group": "Out of Domain",
            "description": (
                f"Acoustic in-domain match ({round(in_domain_conf * 100, 1)}%, "
                f"top class {round(top_prob * 100, 1)}%) is below the requested "
                f"threshold ({round(confidence_threshold * 100, 1)}%). Sound does not reliably match "
                f"any ESC-50 category."
            ),
        }
        result.update({
            "isOutOfScope": True,
            "category": ood_cat,
            "explanation": ood_cat["description"],
            "confidence": round(top_prob, 4),
            "predictions": [
                {"category": ood_cat, "confidence": round(top_prob, 4)},
                *predictions[:4],
            ],
        })
    else:
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
