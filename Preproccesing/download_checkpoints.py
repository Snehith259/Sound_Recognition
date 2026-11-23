"""Download checkpoints from explicitly configured Hugging Face repos or AST repository.

This helper does not keep default repo URLs for the 4 architectures or attach
accuracy claims to a download. Metrics come from `python -m eval.report` after
the evaluation scripts.
"""
from __future__ import annotations

import hashlib
import json
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

AST_REPO = "Adam-ousse/ast-esc50-finetuned-fold1"
AST_REVISION = "fb2443ee3c362d6683a8f493eab2d15598360fb6"
AST_SHA256 = "f3b7ad3ffc95f5d867c7bb088da394bf048e649850535e9faa1f75276e0c6617"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_ast_weights(cache_dir: Path | str | None = None) -> dict[str, Any]:
    """Fetch and validate the AST ESC-50 checkpoint locally."""
    if cache_dir is None:
        cache_dir = CHECKPOINTS_DIR / ".cache" / "huggingface"
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise RuntimeError("Install huggingface_hub before downloading checkpoints.") from err

    snapshot_path = Path(snapshot_download(
        repo_id=AST_REPO,
        revision=AST_REVISION,
        cache_dir=str(cache_path),
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
    ))

    model_file = snapshot_path / "model.safetensors"
    if not model_file.is_file():
        # Look in snapshots directory if returned path was repo root
        matches = list(cache_path.glob(f"**/snapshots/{AST_REVISION}/model.safetensors"))
        if matches:
            model_file = matches[0]
            snapshot_path = model_file.parent
        else:
            raise RuntimeError(f"model.safetensors not found under {cache_path}")

    actual_sha = _sha256(model_file)
    actual_size = model_file.stat().st_size

    if actual_sha.lower() != AST_SHA256.lower():
        raise RuntimeError(
            f"AST model.safetensors SHA-256 mismatch! Expected {AST_SHA256}, got {actual_sha}"
        )

    manifest_data = {
        "modelId": AST_REPO,
        "revision": AST_REVISION,
        "sha256": actual_sha,
        "sizeBytes": actual_size,
        "license": "MIT",
        "source": f"https://huggingface.co/Adam-ousse/ast-esc50-finetuned-fold1",
        "featureType": "ast",
        "samplingRate": 16000,
        "windowSeconds": 5.0,
    }
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = CHECKPOINTS_DIR / "ast_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    result = {
        "model": "ast",
        "snapshotPath": str(snapshot_path),
        "modelFile": str(model_file),
        "sizeBytes": actual_size,
        "sha256": actual_sha,
        "available": True,
    }
    print(f"Verified AST ({result['sizeBytes']} bytes, SHA-256 {result['sha256']})")
    print(f"Wrote manifest to {manifest_file}")
    return result


def retrieve_cloud_weights(model_key: str = "resnet18") -> dict[str, Any]:
    """Fetch and validate the requested file, or explain why it was not fetched."""
    if model_key == "ast":
        return download_ast_weights()

    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Choose one of {', '.join(MODELS)} or 'ast'.")

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
    key = sys.argv[1] if len(sys.argv) > 1 else "ast"
    retrieve_cloud_weights(key)
