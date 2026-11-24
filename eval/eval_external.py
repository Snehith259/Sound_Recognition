"""Evaluate the same inference function called by the FastAPI prediction endpoint.

The local harness avoids ASGI test-client socket setup so it also runs on locked-down
Windows environments. Audio decoding, gates, model loading, and prediction are the
production inference code path.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from eval.metrics import binary_ood_metrics, classification_metrics

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "eval" / "external_manifest.csv"
REPORT_DIR = ROOT / "eval" / "reports"
MODEL_NAMES = ["ast", "resnet18", "vgg16", "sresnet", "cnngru", "ensemble"]


def _encode(audio_bytes: bytes, fmt: str) -> tuple[bytes, str, str]:
    if fmt == "wav":
        return audio_bytes, "audio/wav", ".wav"
    if fmt == "aac":
        output_format = "adts"
        codec = "aac"
        mime = "audio/aac"
        ext = ".aac"
        sr = "22050"
    elif fmt == "mp3":
        output_format = "mp3"
        codec = "libmp3lame"
        mime = "audio/mpeg"
        ext = ".mp3"
        sr = "22050"
    else:
        output_format = "webm"
        codec = "libopus"
        mime = "audio/webm"
        ext = ".webm"
        sr = "48000"
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-vn", "-ac", "1", "-ar", sr, "-c:a", codec, "-b:a", "128k", "-f", output_format, "pipe:1"],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"Could not encode {fmt}: {proc.stderr.decode(errors='replace')[-300:]}")
    return proc.stdout, mime, ext


def _eval_model(row: dict[str, str], model: str, fmt: str, index: int, threshold: float | None) -> dict[str, Any]:
    audio_bytes = (ROOT / "eval" / "external" / row["file"]).read_bytes()
    encoded, content_type, ext = _encode(audio_bytes, fmt)
    fake_name = f"clip_{index:05d}{ext}"
    try:
        from Preproccesing.acoustic_inference import run_acoustic_inference
        body = run_acoustic_inference(encoded, model, threshold)
        status_code = 200 if body.get("success") else 500
    except FileNotFoundError as exc:
        body = {"error": "Model unavailable", "details": str(exc)}
        status_code = 503
    except ValueError as exc:
        body = {"error": "Invalid inference request", "details": str(exc)}
        status_code = 400
    except Exception as exc:
        body = {"error": "Inference failed", "details": str(exc)}
        status_code = 500
    category = body.get("category") or {}
    predictions = body.get("predictions") or []
    return {
        "sourceFile": row["file"],
        "sourceDataset": row.get("source_dataset"),
        "sourceRecording": row.get("source_recording"),
        "split": row["split"],
        "label": row["label"],
        "allowedLabels": json.loads(row.get("allowed_labels") or "[]"),
        "inScope": row["in_scope"].lower() == "true",
        "model": model,
        "format": fmt,
        "requestFilename": fake_name,
        "statusCode": status_code,
        "predictedCategory": category.get("name"),
        "predictedClassId": category.get("id"),
        "confidence": body.get("confidence"),
        "confidenceType": body.get("confidenceType"),
        "oodScore": body.get("oodScore"),
        "isOutOfScope": bool(body.get("isOutOfScope", False)),
        "classProbabilities": body.get("classProbabilities"),
        "topPredictions": [
            {"id": p.get("category", {}).get("id"), "name": p.get("category", {}).get("name"), "confidence": p.get("confidence")}
            for p in predictions
        ],
        "latencyMs": body.get("inferenceTime"),
        "error": body.get("details") or body.get("detail") or body.get("error"),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in sorted({r["model"] for r in rows}):
        model_rows = [r for r in rows if r["model"] == model]
        completed = [r for r in model_rows if r["statusCode"] == 200 and r["predictedCategory"]]
        in_scope = [r for r in completed if r["inScope"]]
        out_scope = [r for r in completed if not r["inScope"]]
        correct = [r for r in in_scope if not r["isOutOfScope"] and r["predictedClassId"] in r["allowedLabels"]]
        ood_correct = [r for r in out_scope if r["isOutOfScope"]]
        confidence_rows = [r for r in in_scope if isinstance(r.get("confidence"), (int, float))]
        metrics: dict[str, Any] = {
            "n": len(completed),
            "failedRequests": len(model_rows) - len(completed),
            "inScopeCount": len(in_scope),
            "outOfScopeCount": len(out_scope),
            "top1": len(correct) / len(in_scope) if in_scope else None,
            "oodRejectionRate": len(ood_correct) / len(out_scope) if out_scope else None,
            "falseRejectionRate": sum(r["isOutOfScope"] for r in in_scope) / len(in_scope) if in_scope else None,
            "meanConfidenceInScope": float(np.mean([r["confidence"] for r in confidence_rows])) if confidence_rows else None,
            "meanLatencyMs": float(np.mean([r["latencyMs"] for r in completed if isinstance(r.get("latencyMs"), (int, float))])) if any(isinstance(r.get("latencyMs"), (int, float)) for r in completed) else None,
        }
        uniquely_labeled = [r for r in in_scope if len(r["allowedLabels"]) == 1 and isinstance(r.get("classProbabilities"), list)]
        if uniquely_labeled:
            targets = [r["allowedLabels"][0] for r in uniquely_labeled]
            probs = np.asarray([r["classProbabilities"] for r in uniquely_labeled], dtype=float)
            metrics["classificationMetrics"] = classification_metrics(targets, probs)
        if completed:
            ood_rows = [r for r in completed if isinstance(r.get("oodScore"), (int, float))]
            if ood_rows:
                metrics["oodRanking"] = binary_ood_metrics([not r["inScope"] for r in ood_rows], [r["oodScore"] for r in ood_rows])
        result[model] = metrics
    return result


def run(split: str = "dev", models: list[str] | None = None, formats: list[str] | None = None, threshold: float | None = 0.5, limit: int | None = None) -> dict[str, Any]:
    if split not in {"dev", "final"}:
        raise ValueError("split must be 'dev' or 'final'")
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        manifest = [row for row in csv.DictReader(stream) if row["split"] == split]
    if not manifest:
        raise RuntimeError(f"No clips in split '{split}'. Run python -m eval.build_external_set first.")
    if limit is not None and limit < len(manifest):
        rng = np.random.default_rng(20261121)
        indices = np.sort(rng.choice(len(manifest), size=limit, replace=False))
        manifest = [manifest[int(index)] for index in indices]
    selected_models = models or MODEL_NAMES
    selected_formats = formats or ["wav", "mp3", "webm", "aac"]
    if split == "final":
        print("Running the held-out final split once; do not use these results to tune thresholds.")

    records = []
    counter = 0
    for row in manifest:
        for model in selected_models:
            for fmt in selected_formats:
                counter += 1
                record = _eval_model(row, model, fmt, counter, threshold)
                records.append(record)
                if record["statusCode"] != 200:
                    print(f"{model}/{fmt}: HTTP {record['statusCode']} — {record['error']}")

    report = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "models": selected_models,
        "formats": selected_formats,
        "confidenceThreshold": threshold,
        "nClips": len(manifest),
        "nRequests": len(records),
        "metrics": _summarize(records),
        "results": records,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = REPORT_DIR / f"external_{split}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {target}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "final"], default="dev")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--formats", nargs="+", choices=["wav", "mp3", "webm", "aac"], default=["wav", "mp3", "webm", "aac"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, help="Evaluate a deterministic representative subset")
    args = parser.parse_args()
    run(args.split, args.models, args.formats, args.threshold, args.limit)
