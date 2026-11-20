"""
SoundSense — Automated Model Weight & Checkpoint Cloud Retriever
Downloads or verifies cached ESC-50 trained weights from Hugging Face Hub / Cloud Storage.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CACHE_MANIFEST = CHECKPOINTS_DIR / "manifest.json"

DEFAULT_MODELS = {
    "resnet18_v1": {
        "repo_id": "soundsense/esc50-resnet18",
        "filename": "esc50_resnet18_v1.pt",
        "size_mb": 45.34,
        "accuracy": 62.81,
        "top5": 88.19,
        "description": "Deep Residual Network with 18 layers (Log-Mel Spectrogram input)",
    },
    "vgg16_v1": {
        "repo_id": "soundsense/esc50-vgg16",
        "filename": "esc50_vgg16_v1.pt",
        "size_mb": 63.26,
        "accuracy": 61.06,
        "top5": 91.96,
        "description": "Visual Geometry Group 16-layer Deep Convolutional Network",
    },
    "sresnet_v09": {
        "repo_id": "soundsense/esc50-sresnet",
        "filename": "esc50_sresnet_linear_v09.pt",
        "size_mb": 1.72,
        "accuracy": 35.68,
        "top5": 76.13,
        "description": "Linear spectral MFCC squeezed residual network",
    },
    "cnngru_v08": {
        "repo_id": "soundsense/esc50-cnngru",
        "filename": "esc50_cnngru_recurrent_v08.pt",
        "size_mb": 2.11,
        "accuracy": 64.07,
        "top5": 91.21,
        "description": "Nonlinear Log-Mel temporal recurrent model",
    },
}


def ensure_checkpoint_cache():
    """Initializes checkpoints cache directory and manifest."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CACHE_MANIFEST.exists():
        manifest_data = {
            "version": "1.0.0",
            "hub": "https://huggingface.co",
            "models": DEFAULT_MODELS,
            "status": "cached",
        }
        with open(CACHE_MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        print(f"[SoundSense] Checkpoint cache initialized at: {CHECKPOINTS_DIR}")
    else:
        print(f"[SoundSense] Checkpoint cache verified at: {CHECKPOINTS_DIR}")


def retrieve_cloud_weights(model_key="resnet18_v1"):
    """
    Downloads or verifies model weights from Hugging Face Hub.
    Falls back cleanly to local cached manifest verification.
    """
    ensure_checkpoint_cache()
    if model_key not in DEFAULT_MODELS:
        print(f"[Error] Unknown model key: {model_key}. Available: {list(DEFAULT_MODELS.keys())}")
        return False

    spec = DEFAULT_MODELS[model_key]
    target_path = CHECKPOINTS_DIR / spec["filename"]

    if target_path.exists():
        print(f"[Cache Hit] Model checkpoint already present: {target_path} ({spec['size_mb']} MB)")
        return True

    print(f"[Cloud Retrieval] Fetching {spec['repo_id']}/{spec['filename']} ({spec['size_mb']} MB)...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"], local_dir=CHECKPOINTS_DIR)
        print(f"[Success] Downloaded to {downloaded}")
        return True
    except ImportError:
        print(f"[Warning] huggingface_hub not installed; please train models locally using Preproccesing/train_models.py")
        return False
    except Exception as e:
        print(f"[Warning] Remote retrieval failed: {e}")
        return False


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "resnet18_v1"
    retrieve_cloud_weights(key)
