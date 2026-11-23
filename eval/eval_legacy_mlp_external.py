"""Diagnostic only: score the bundled legacy MLP on external dev clips.

This checkpoint is not selected by the UI and is never a fallback for the four
architecture options. The report exists to inspect the only model file present
in the checkout, with every audio filename kept outside the model call.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.metrics import binary_ood_metrics, classification_metrics
from Preproccesing.acoustic_inference import (
    ESC50_CATEGORIES,
    extract_acoustic_feature_vector,
    get_acoustic_classifier,
    load_audio_waveform,
)

ROOT = Path(__file__).resolve().parents[1]


def run(split: str = "dev", limit: int | None = None) -> dict[str, Any]:
    if split != "dev":
        raise ValueError("The legacy MLP diagnostic intentionally supports dev only; final is reserved for trained models.")
    with (ROOT / "eval" / "external_manifest.csv").open(newline="", encoding="utf-8") as stream:
        clips = [row for row in csv.DictReader(stream) if row["split"] == "dev"]
    if limit is not None:
        clips = clips[:limit]
    if not clips:
        raise RuntimeError("No external dev clips. Run eval.build_external_set first.")
    data = get_acoustic_classifier()
    records = []
    for index, row in enumerate(clips, 1):
        audio, sr = load_audio_waveform(ROOT / "eval" / "external" / row["file"], target_sr=22050, target_duration=5.0)
        feature, signal_metrics = extract_acoustic_feature_vector(audio, sr)
        normalized = (feature.reshape(1, -1) - data["mean_vec"]) / data["std_vec"]
        with torch.no_grad():
            logits = data["model"](torch.tensor(normalized, dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=1)[0].numpy()
        distances = np.linalg.norm(data["centroids"] - normalized, axis=1)
        nearest = int(np.argmin(distances))
        radius = max(float(data["radii"][nearest]), 1e-8)
        manifold_reject = float(distances[nearest]) > radius * 1.6
        order = np.argsort(probabilities)[::-1]
        top_id = int(order[0])
        allowed = json.loads(row.get("allowed_labels") or "[]")
        in_scope = row["in_scope"].lower() == "true"

        distance_score = float(distances[nearest] / (radius * 1.6))
        records.append({
            "index": index,
            "sourceFile": row["file"],
            "sourceDataset": row.get("source_dataset"),
            "sourceRecording": row.get("source_recording"),
            "split": split,
            "inScope": in_scope,
            "label": row["label"],
            "allowedLabels": allowed,
            "predictedClassId": top_id,
            "predictedClass": ESC50_CATEGORIES[top_id]["name"],
            "correct": bool(in_scope and top_id in allowed),
            "maxClassProbability": float(probabilities[top_id]),
            "classProbabilities": probabilities.astype(float).tolist(),
            "top5": [{"id": int(i), "name": ESC50_CATEGORIES[int(i)]["name"], "probability": float(probabilities[int(i)])} for i in order[:5]],
            "centroidDistanceScore": distance_score,
            "centroidGateRejected": bool(manifold_reject),
            "isOutOfScopeByMaxSoftmax30": bool(float(probabilities[top_id]) < 0.30),
            "signalMetrics": signal_metrics,
        })
        if index % 10 == 0 or index == len(clips):
            print(f"Scored {index}/{len(clips)} external dev clips", flush=True)

    in_scope = [row for row in records if row["inScope"]]
    out_scope = [row for row in records if not row["inScope"]]
    unambiguous = [row for row in in_scope if len(row["allowedLabels"]) == 1]
    result = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": "legacy MLP diagnostic (not served by the UI)",
        "checkpoint": "checkpoints/acoustic_classifier.pt",
        "split": split,
        "nClips": len(records),
        "nInScope": len(in_scope),
        "nOutOfScope": len(out_scope),
        "top1AllowedLabelAccuracy": sum(row["correct"] for row in in_scope) / len(in_scope) if in_scope else None,
        "top5AllowedLabelAccuracy": sum(any(item["id"] in row["allowedLabels"] for item in row["top5"]) for row in in_scope) / len(in_scope) if in_scope else None,
        "meanConfidenceInScope": float(np.mean([row["maxClassProbability"] for row in in_scope])) if in_scope else None,
        "accuracyAtConfidence90": float(np.mean([row["correct"] for row in in_scope if row["maxClassProbability"] >= 0.90])) if any(row["maxClassProbability"] >= 0.90 for row in in_scope) else None,
        "coverageAtConfidence90": sum(row["maxClassProbability"] >= 0.90 for row in in_scope) / len(in_scope) if in_scope else 0.0,
        "classificationMetricsUnambiguousOnly": classification_metrics(
            [row["allowedLabels"][0] for row in unambiguous],
            np.asarray([row["classProbabilities"] for row in unambiguous]),
            [category["name"] for category in ESC50_CATEGORIES],
        ) if unambiguous else None,
        "oodMetricsMaxSoftmax": binary_ood_metrics(
            [row["inScope"] is False for row in records],
            [1.0 - row["maxClassProbability"] for row in records],
        ),
        "oodMetricsCentroidDistance": binary_ood_metrics(
            [row["inScope"] is False for row in records],
            [row["centroidDistanceScore"] for row in records],
        ),
        "centroidGateRejectionRateOOD": sum(row["centroidGateRejected"] for row in out_scope) / len(out_scope) if out_scope else None,
        "centroidGateFalseRejectionRateInScope": sum(row["centroidGateRejected"] for row in in_scope) / len(in_scope) if in_scope else None,
        "maxSoftmaxRejectionRateOOD": sum(row["isOutOfScopeByMaxSoftmax30"] for row in out_scope) / len(out_scope) if out_scope else None,
        "records": records,
        "limitations": [
            "This MLP was trained on the full ESC-50 set and is not an unbiased ESC-50 accuracy estimate.",
            "External scores are dev-only and must not be presented as metrics for the UI model choices.",
            "R², NLL, Brier, ECE, and selective accuracy are computed only on unambiguous in-scope labels.",
        ],
    }
    output = ROOT / "eval" / "reports" / "legacy_mlp_external_dev.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev"], default="dev")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = run(args.split, args.limit)
    print(f"Legacy MLP external dev accuracy: {report['top1AllowedLabelAccuracy']}")
    print(f"Legacy MLP external dev R² (unambiguous labels): {report['classificationMetricsUnambiguousOnly']['r2Probability'] if report['classificationMetricsUnambiguousOnly'] else None}")
