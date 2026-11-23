"""Evaluation script for the deployed AST ESC-50 classifier.

Evaluates the exact production code path (run_acoustic_inference) with fake
filenames and no metadata leakage. Supports ESC-50 fold-1 test, external dev,
and format parity verification.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT / "checkpoints"
REPORTS_DIR = ROOT / "eval" / "reports"
MANIFEST_FILE = ROOT / "eval" / "external_manifest.csv"

from Preproccesing.acoustic_inference import (
    ESC50_CATEGORIES,
    run_acoustic_inference,
)
from eval.metrics import binary_ood_metrics, classification_metrics

AST_MODEL_ID = "Adam-ousse/ast-esc50-finetuned-fold1"
AST_REVISION = "fb2443ee3c362d6683a8f493eab2d15598360fb6"
AST_SHA256 = "f3b7ad3ffc95f5d867c7bb088da394bf048e649850535e9faa1f75276e0c6617"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def wilson_interval(successes: int, total: int, z: float = 1.95996) -> tuple[float, float]:
    """Compute Wilson 95% score confidence interval."""
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    spread = (z / denom) * ((p * (1.0 - p) / total + z * z / (4.0 * total * total)) ** 0.5)
    return max(0.0, float(center - spread)), min(1.0, float(center + spread))


def evaluate_fold1(limit: int | None = None) -> dict[str, Any]:
    """Evaluate AST on ESC-50 fold-1 test split."""
    print("=" * 60)
    print(" Evaluating AST on ESC-50 Fold 1 Test Split")
    print("=" * 60)

    split_file = ROOT / "Preproccesing" / "splits" / "fold_1_test.csv"
    with split_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    if limit is not None:
        rows = rows[:limit]

    audio_dir = ROOT / "Preproccesing" / "processed" / "audio_22k"
    fallback_dir = ROOT / "ESC-50" / "wav_files"

    targets: List[int] = []
    probabilities: List[List[float]] = []
    latencies: List[float] = []
    records: List[Dict[str, Any]] = []

    for index, row in enumerate(rows, 1):
        filename = row["filename"]
        target = int(row["target"])
        path = audio_dir / filename
        if not path.is_file():
            path = fallback_dir / filename
        if not path.is_file():
            continue

        audio_bytes = path.read_bytes()
        # Anonymized fake filename to guarantee no metadata inference
        fake_filename = f"clip_{index:05d}.wav"

        t0 = time.time()
        res = run_acoustic_inference(audio_bytes, model_name="ast")
        t_ms = (time.time() - t0) * 1000

        if "classProbabilities" in res and res["classProbabilities"] is not None:
            probs = res["classProbabilities"]
            top_id = res["category"]["id"]
            confidence = res.get("confidence", 0.0)
        else:
            probs = [1.0 / len(ESC50_CATEGORIES)] * len(ESC50_CATEGORIES)
            top_id = res.get("category", {}).get("id", -1)
            confidence = 0.0

        targets.append(target)
        probabilities.append(probs)
        latencies.append(res.get("inferenceTime", t_ms))

        records.append({
            "index": index,
            "target": target,
            "predicted": top_id,
            "correct": top_id == target,
            "confidence": confidence,
            "latencyMs": res.get("inferenceTime", t_ms),
        })

        if index % 50 == 0 or index == len(rows):
            cur_acc = np.mean([r["correct"] for r in records])
            print(f"  Processed {index}/{len(rows)} clips: Running top-1 = {cur_acc * 100:.2f}%")

    prob_array = np.array(probabilities, dtype=np.float64)
    metrics = classification_metrics(
        targets,
        prob_array,
        class_names=[c["name"] for c in ESC50_CATEGORIES],
    )
    metrics["latencyMs"] = round(float(np.mean(latencies)), 1)

    ci_low, ci_high = wilson_interval(int(np.sum([r["correct"] for r in records])), len(targets))
    metrics["wilson95Top1"] = {"low": round(ci_low, 4), "high": round(ci_high, 4)}

    report = {
        "model": AST_MODEL_ID,
        "revision": AST_REVISION,
        "checkpointSha256": AST_SHA256,
        "windowPolicy": "5s windows, 2.5s hop, max 12, mean probabilities, no peak normalization",
        "inferencePath": "Preproccesing.acoustic_inference.run_acoustic_inference",
        "gitCommit": get_git_commit(),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "split": "fold_1_test",
        "n": len(targets),
        "metrics": metrics,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "ast_esc50_fold1.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote fold-1 report to {out_path}")
    print(f"Top-1: {metrics['top1'] * 100:.2f}% | Top-5: {metrics['top5'] * 100:.2f}% | ECE: {metrics['ece']:.4f} | R2: {metrics['r2Probability']:.4f}")
    return report


def evaluate_external(split: str = "dev", limit: int | None = None) -> dict[str, Any]:
    """Evaluate AST on external independent recordings (dev or final split)."""
    print("=" * 60)
    print(f" Evaluating AST on External Independent Audio ({split.upper()} split)")
    print("=" * 60)

    with MANIFEST_FILE.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))

    split_rows = [r for r in all_rows if r["split"] == split]
    if limit is not None:
        split_rows = split_rows[:limit]

    records: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for index, row in enumerate(split_rows, 1):
        rel_file = row["file"]
        file_path = ROOT / "eval" / "external" / rel_file
        if not file_path.is_file():
            print(f"  WARNING: Missing {file_path}")
            continue

        audio_bytes = file_path.read_bytes()
        t0 = time.time()
        res = run_acoustic_inference(audio_bytes, model_name="ast")
        t_ms = (time.time() - t0) * 1000

        top_id = res.get("category", {}).get("id", -1)
        confidence = res.get("confidence")
        ood_score = res.get("oodScore", 1.0)
        probs = res.get("classProbabilities") or ([1.0 / len(ESC50_CATEGORIES)] * len(ESC50_CATEGORIES))
        latencies.append(res.get("inferenceTime", t_ms))

        in_scope = row["in_scope"].lower() == "true"
        allowed = json.loads(row.get("allowed_labels") or "[]")
        correct_allowed = (top_id in allowed) if in_scope else None
        top5_ids = [p["category"]["id"] for p in res["predictions"]]
        top5_allowed = any(cid in allowed for cid in top5_ids) if in_scope else None

        records.append({
            "index": index,
            "file": rel_file,
            "sourceDataset": row.get("source_dataset"),
            "sourceRecording": row.get("source_recording"),
            "sourceName": row.get("source_name"),
            "label": row["label"],
            "allowed": allowed,
            "in_scope": in_scope,
            "is_ambient": row.get("source_dataset") == "kahst/BirdNET-Analyzer",
            "top_id": top_id,
            "top_name": res["category"]["name"],
            "confidence": confidence,
            "ood_score": ood_score,
            "correct_allowed": correct_allowed,
            "top5_allowed": top5_allowed,
            "probs": probs,
            "latencyMs": res.get("inferenceTime", t_ms),
        })

    # Metrics on unique clips
    in_scope_records = [r for r in records if r["in_scope"]]
    ood_records = [r for r in records if not r["in_scope"]]
    ambient_records = [r for r in in_scope_records if r["is_ambient"]]
    unambiguous_records = [r for r in in_scope_records if len(r["allowed"]) == 1]

    n_in_scope = len(in_scope_records)
    n_allowed_top1 = sum(1 for r in in_scope_records if r["correct_allowed"])
    n_allowed_top5 = sum(1 for r in in_scope_records if r["top5_allowed"])
    allowed_top1 = n_allowed_top1 / n_in_scope if n_in_scope else 0.0
    allowed_top5 = n_allowed_top5 / n_in_scope if n_in_scope else 0.0

    n_ambient = len(ambient_records)
    n_ambient_top1 = sum(1 for r in ambient_records if r["correct_allowed"])
    ambient_top1 = n_ambient_top1 / n_ambient if n_ambient else 0.0

    # Wilson intervals
    ci_allowed = wilson_interval(n_allowed_top1, n_in_scope)
    ci_ambient = wilson_interval(n_ambient_top1, n_ambient)

    # Unambiguous subset classification metrics
    unambiguous_targets = [r["allowed"][0] for r in unambiguous_records]
    unambiguous_probs = np.array([r["probs"] for r in unambiguous_records], dtype=np.float64)
    unambig_metrics = classification_metrics(
        unambiguous_targets,
        unambiguous_probs,
        class_names=[c["name"] for c in ESC50_CATEGORIES],
    )
    ci_unambig = wilson_interval(
        int(round(unambig_metrics["top1"] * len(unambiguous_targets))),
        len(unambiguous_targets),
    )
    unambig_metrics["wilson95Top1"] = {"low": round(ci_unambig[0], 4), "high": round(ci_unambig[1], 4)}

    # OOD metrics
    all_is_ood = [not r["in_scope"] for r in records]
    all_ood_scores = [r["ood_score"] for r in records]
    ood_metrics = binary_ood_metrics(all_is_ood, all_ood_scores)

    # Selective prediction & confidence analysis
    correct_confs = [r["confidence"] for r in in_scope_records if r["correct_allowed"]]
    wrong_confs = [r["confidence"] for r in in_scope_records if not r["correct_allowed"]]
    mean_conf_correct = float(np.mean(correct_confs)) if correct_confs else 0.0
    mean_conf_wrong = float(np.mean(wrong_confs)) if wrong_confs else 0.0

    sel90 = [r for r in in_scope_records if r["confidence"] >= 0.90]
    sel80 = [r for r in in_scope_records if r["confidence"] >= 0.80]
    acc_sel90 = float(np.mean([r["correct_allowed"] for r in sel90])) if sel90 else 0.0
    cov_sel90 = float(len(sel90) / n_in_scope) if n_in_scope else 0.0
    acc_sel80 = float(np.mean([r["correct_allowed"] for r in sel80])) if sel80 else 0.0
    cov_sel80 = float(len(sel80) / n_in_scope) if n_in_scope else 0.0

    # Per-source breakdown
    per_source: Dict[str, Any] = {}
    sources = set(r["sourceDataset"] for r in records if r["sourceDataset"])
    for src in sorted(sources):
        src_in_scope = [r for r in in_scope_records if r["sourceDataset"] == src]
        src_ood = [r for r in ood_records if r["sourceDataset"] == src]
        src_correct = sum(1 for r in src_in_scope if r["correct_allowed"])
        per_source[src] = {
            "inScopeCount": len(src_in_scope),
            "oodCount": len(src_ood),
            "allowedTop1": round(src_correct / len(src_in_scope), 4) if src_in_scope else None,
        }

    report = {
        "model": AST_MODEL_ID,
        "revision": AST_REVISION,
        "checkpointSha256": AST_SHA256,
        "windowPolicy": "5s windows, 2.5s hop, max 12, mean probabilities, no peak normalization",
        "inferencePath": "Preproccesing.acoustic_inference.run_acoustic_inference",
        "gitCommit": get_git_commit(),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "nClips": len(records),
        "inScopeCount": n_in_scope,
        "oodCount": len(ood_records),
        "ambientCount": n_ambient,
        "top1": round(allowed_top1, 6),
        "top5": round(allowed_top5, 6),
        "wilson95Top1": {"low": round(ci_allowed[0], 4), "high": round(ci_allowed[1], 4)},
        "ambientTop1": round(ambient_top1, 6),
        "ambientWilson95": {"low": round(ci_ambient[0], 4), "high": round(ci_ambient[1], 4)},
        "ambientN": n_ambient,
        "meanConfidenceCorrect": round(mean_conf_correct, 4),
        "meanConfidenceWrong": round(mean_conf_wrong, 4),
        "selectiveAccuracyAt90": round(acc_sel90, 4),
        "selectiveCoverageAt90": round(cov_sel90, 4),
        "selectiveAccuracyAt80": round(acc_sel80, 4),
        "selectiveCoverageAt80": round(cov_sel80, 4),
        "latencyMs": round(float(np.mean(latencies)), 1) if latencies else None,
        "unambiguous": {
            "n": len(unambiguous_targets),
            "top1": round(unambig_metrics["top1"], 6),
            "top5": round(unambig_metrics["top5"], 6),
            "macroF1": round(unambig_metrics["macroF1"], 6),
            "nll": round(unambig_metrics["nll"], 6),
            "brier": round(unambig_metrics["brier"], 6),
            "ece": round(unambig_metrics["ece"], 6),
            "r2Probability": round(unambig_metrics["r2Probability"], 6) if unambig_metrics.get("r2Probability") is not None else None,
            "meanConfidence": round(unambig_metrics["meanConfidence"], 4),
            "wilson95Top1": unambig_metrics["wilson95Top1"],
        },
        "ood": ood_metrics,
        "perSource": per_source,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / f"ast_external_{split}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote external {split} report to {out_file}")
    print(f"Allowed Top-1: {allowed_top1 * 100:.2f}% (95% CI: {ci_allowed[0]*100:.1f}–{ci_allowed[1]*100:.1f}%)")
    print(f"Unambiguous Top-1: {unambig_metrics['top1'] * 100:.2f}% | ECE: {unambig_metrics['ece']:.4f} | R2: {unambig_metrics['r2Probability']:.4f}")
    print(f"Ambient Top-1: {ambient_top1 * 100:.2f}% (n={n_ambient})")
    print(f"OOD AUROC: {ood_metrics['auroc']:.4f}")
    return report


def check_format_parity(split: str = "dev", max_clips: int = 20) -> dict[str, Any]:
    """Test WAV vs MP3 and WebM format parity using ffmpeg transcoding."""
    print("=" * 60)
    print(f" Checking WAV vs MP3/WebM Format Parity (up to {max_clips} clips)")
    print("=" * 60)

    with MANIFEST_FILE.open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == split and r["in_scope"].lower() == "true"]

    rows = rows[:max_clips]
    agreements_mp3 = []
    agreements_webm = []

    for idx, row in enumerate(rows, 1):
        file_path = ROOT / "eval" / "external" / row["file"]
        wav_bytes = file_path.read_bytes()
        res_wav = run_acoustic_inference(wav_bytes, model_name="ast")
        wav_top = res_wav["category"]["id"]

        # Transcode to MP3
        proc_mp3 = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-vn", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3", "pipe:1"],
            input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if proc_mp3.returncode == 0 and proc_mp3.stdout:
            res_mp3 = run_acoustic_inference(proc_mp3.stdout, model_name="ast")
            agreements_mp3.append(res_mp3["category"]["id"] == wav_top)

        # Transcode to WebM (Opus)
        proc_webm = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-vn", "-ar", "16000", "-c:a", "libopus", "-b:a", "96k", "-f", "webm", "pipe:1"],
            input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if proc_webm.returncode == 0 and proc_webm.stdout:
            res_webm = run_acoustic_inference(proc_webm.stdout, model_name="ast")
            agreements_webm.append(res_webm["category"]["id"] == wav_top)

    mp3_agreement = float(np.mean(agreements_mp3)) if agreements_mp3 else 0.0
    webm_agreement = float(np.mean(agreements_webm)) if agreements_webm else 0.0
    print(f"WAV vs MP3 Top-1 Agreement: {mp3_agreement * 100:.1f}% ({len(agreements_mp3)} clips)")
    print(f"WAV vs WebM Top-1 Agreement: {webm_agreement * 100:.1f}% ({len(agreements_webm)} clips)")
    return {"mp3Agreement": mp3_agreement, "webmAgreement": webm_agreement}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AST ESC-50 classifier")
    parser.add_argument("--fold1", action="store_true", help="Evaluate on ESC-50 fold-1 test set")
    parser.add_argument("--external-split", type=str, choices=["dev", "final"], help="Evaluate on external set split")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips")
    parser.add_argument("--check-formats", action="store_true", help="Run WAV/MP3/WebM format parity check")

    args = parser.parse_args()
    if not (args.fold1 or args.external_split or args.check_formats):
        parser.print_help()
        sys.exit(1)

    if args.fold1:
        evaluate_fold1(limit=args.limit)
    if args.external_split:
        evaluate_external(split=args.external_split, limit=args.limit)
    if args.check_formats:
        check_format_parity()
