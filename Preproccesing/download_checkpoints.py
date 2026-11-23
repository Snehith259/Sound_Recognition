"""Download a checkpoint only from an explicitly configured Hugging Face repo.

This helper does not keep a default repo URL or attach accuracy claims to a
download. Metrics come from `python -m eval.report` after the evaluation scripts.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
MODELS = {
    "resnet18": ("esc50_resnet18_v1.pt", "SOUNDSENSE_HF_REPO_RESNET18"),
    "vgg16": ("esc50_vgg16_v1.pt", "SOUNDSENSE_HF_REPO_VGG16"),
    "sresnet": ("esc50_sresnet_linear_v09.pt", "SOUNDSENSE_HF_REPO_SRESNET"),
    "cnngru": ("esc50_cnngru_recurrent_v08.pt", "SOUNDSENSE_HF_REPO_CNNGRU"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retrieve_cloud_weights(model_key: str = "resnet18") -> dict[str, Any]:
    """Fetch and validate the requested file, or explain why it was not fetched."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Choose one of {', '.join(MODELS)}.")

    filename, repo_env = MODELS[model_key]
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = CHECKPOINTS_DIR / filename
    repo_id = os.environ.get(repo_env, "").strip()

    if not target_path.is_file():
        if not repo_id:
            raise RuntimeError(
                f"No checkpoint is available and {repo_env} is unset. "
                "Set it to a Hugging Face repo that contains the trained checkpoint."
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as err:
            raise RuntimeError("Install huggingface_hub before downloading checkpoints.") from err
        downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename))
        if not downloaded.is_file() or downloaded.stat().st_size == 0:
            raise RuntimeError(f"The remote repository returned an empty or missing file: {filename}")
        if downloaded.resolve() != target_path.resolve():
            shutil.copy2(downloaded, target_path)

    # Check structural integrity; strict architecture loading remains the
    # responsibility of load_model_checkpoint before inference.
    import torch
    checkpoint = torch.load(target_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise RuntimeError(f"{filename} is not a trained SoundSense checkpoint with a state_dict.")

    result = {
        "model": model_key,
        "filename": filename,
        "sizeBytes": target_path.stat().st_size,
        "sha256": _sha256(target_path),
        "available": True,
    }
    print(f"Verified {result['filename']} ({result['sizeBytes']} bytes, SHA-256 {result['sha256']})")
    print("Run the ESC-50 CV and external evaluation scripts before publishing any model score.")
    return result


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    retrieve_cloud_weights(key)
