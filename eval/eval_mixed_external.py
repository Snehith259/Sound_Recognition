"""Comprehensive baseline & post-upgrade evaluator for external mixed + pure audio.

Evaluates all 6 SoundSense models (ast, resnet18, vgg16, sresnet, cnngru, ensemble)
across pure single-source clips, multi-source mixed clips (~63% of corpus), and
out-of-domain distractors across multiple formats (wav/mp3/webm/aac optional).
Never leaks filenames or metadata into acoustic inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Preproccesing.acoustic_inference import run_acoustic_inference, warm_up_engine, load_model_checkpoint
from eval.metrics import binary_ood_metrics

MANIFEST = ROOT / "eval" / "external_mixed_manifest.csv"
AUDIO_DIR = ROOT / "eval" / "external_mixed"
REPORT_DIR = ROOT / "eval" / "reports"
ALL_MODELS = ["ast", "resnet18", "vgg16", "sresnet", "cnngru", "ensemble"]


def _transcode_bytes(wav_bytes: bytes, fmt: str) -> bytes:
    if fmt == "wav":
        return wav_bytes
    fmt_args = {
        "mp3": (["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"], "22050"),
        "aac": (["-c:a", "aac", "-b:a", "128k", "-f", "adts"], "22050"),
        "webm": (["-c:a", "libopus", "-b:a", "96k", "-f", "webm"], "48000"),
    }[fmt]
    args, sr = fmt_args
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-vn", "-ac", "1", "-ar", sr, *args, "pipe:1"],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"Encoding to {fmt} failed: {proc.stderr.decode(errors='replace')[-200:]}")
    return proc.stdout


def evaluate_corpus(
    models: list[str] | None = None,
    threshold: float | None = 0.50,
    multiformat_sample: int = 24,
    out_name: str = "external_mixed_eval.json",
) -> dict[str, Any]:
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected_models = models or ALL_MODELS

    # Pre-load checkpoints once so latency numbers reflect inference speed rather than cold load
    for m in selected_models:
        if m != "ensemble":
            load_model_checkpoint(m)

    # Load all WAV bytes once
    clips: list[tuple[dict[str, str], bytes]] = []
    for row in rows:
        wav_path = AUDIO_DIR / row["file"]
        if wav_path.is_file():
            clips.append((row, wav_path.read_bytes()))

    print(f"Loaded {len(clips)} external mixed+pure clips; testing models={selected_models}, threshold={threshold}", flush=True)

    summary: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []

    for model in selected_models:
        t_m0 = time.time()
        m_records = []
        for idx, (row, wav_bytes) in enumerate(clips, 1):
            res = run_acoustic_inference(wav_bytes, model_name=model, confidence_threshold=threshold)
            allowed = [int(x) for x in json.loads(row.get("allowed_labels") or "[]")]
            in_scope = row["in_scope"].lower() == "true"
            stype = row["sound_type"]
            preds = res.get("predictions") or []

            # Extract top-5 predicted ESC-50 class IDs (excluding OOD placeholder -1)
            esc_preds = [p for p in preds if (p.get("category") or {}).get("id", -1) >= 0]
            top5_ids = [int(p["category"]["id"]) for p in esc_preds[:5]]
            top3_ids = top5_ids[:3]
            raw_top1_id = top5_ids[0] if top5_ids else -1
            raw_top1_name = esc_preds[0]["category"]["name"] if esc_preds else "None"

            is_ood_pred = bool(res.get("isOutOfScope", False))
            acc_top1_id = -1 if is_ood_pred else int((res.get("category") or {}).get("id", -1))

            # Metrics flags
            in_allowed_top1_accepted = (not is_ood_pred) and (acc_top1_id in allowed) if in_scope else False
            in_allowed_top1_raw = (raw_top1_id in allowed) if in_scope else False
            in_allowed_top5 = any(cid in allowed for cid in top5_ids) if in_scope else False

            # Dual-source recall for cross-mixes with >= 2 distinct allowed IDs
            has_two_gt = in_scope and len(allowed) >= 2
            both_top3 = (allowed[0] in top3_ids and allowed[1] in top3_ids) if has_two_gt else None
            both_top5 = (allowed[0] in top5_ids and allowed[1] in top5_ids) if has_two_gt else None

            rec = {
                "file": row["file"],
                "soundType": stype,
                "sourceDataset": row["source_dataset"],
                "label": row["label"],
                "secondaryLabel": row.get("secondary_label", ""),
                "allowedLabels": allowed,
                "inScope": in_scope,
                "model": model,
                "predictedCategory": (res.get("category") or {}).get("name"),
                "predictedClassId": acc_top1_id,
                "rawTop1Id": raw_top1_id,
                "rawTop1Name": raw_top1_name,
                "top5Ids": top5_ids,
                "isOutOfScope": is_ood_pred,
                "confidence": res.get("confidence"),
                "oodScore": res.get("oodScore"),
                "latencyMs": res.get("inferenceTime"),
                "top1AcceptedOk": in_allowed_top1_accepted,
                "top1RawOk": in_allowed_top1_raw,
                "top5Ok": in_allowed_top5,
                "bothInTop3": both_top3,
                "bothInTop5": both_top5,
                "oodRejectedOk": is_ood_pred if not in_scope else None,
            }
            m_records.append(rec)
            all_records.append(rec)

        in_recs = [r for r in m_records if r["inScope"]]
        pure_recs = [r for r in in_recs if r["soundType"] == "pure"]
        mixed_recs = [r for r in in_recs if r["soundType"] == "mixed"]
        two_gt_recs = [r for r in mixed_recs if r["bothInTop5"] is not None]
        ood_recs = [r for r in m_records if not r["inScope"]]

        ood_scores_valid = [r for r in m_records if isinstance(r.get("oodScore"), (int, float))]
        ood_rank = binary_ood_metrics([not r["inScope"] for r in ood_scores_valid], [r["oodScore"] for r in ood_scores_valid]) if ood_scores_valid else {}

        m_summary = {
            "nTotal": len(m_records),
            "nInScope": len(in_recs),
            "nPure": len(pure_recs),
            "nMixed": len(mixed_recs),
            "nOOD": len(ood_recs),
            "inScopeAcceptedTop1": round(sum(r["top1AcceptedOk"] for r in in_recs) / max(1, len(in_recs)), 4),
            "inScopeRawTop1": round(sum(r["top1RawOk"] for r in in_recs) / max(1, len(in_recs)), 4),
            "inScopeTop5": round(sum(r["top5Ok"] for r in in_recs) / max(1, len(in_recs)), 4),
            "pureAcceptedTop1": round(sum(r["top1AcceptedOk"] for r in pure_recs) / max(1, len(pure_recs)), 4),
            "pureRawTop1": round(sum(r["top1RawOk"] for r in pure_recs) / max(1, len(pure_recs)), 4),
            "pureTop5": round(sum(r["top5Ok"] for r in pure_recs) / max(1, len(pure_recs)), 4),
            "mixedAcceptedTop1": round(sum(r["top1AcceptedOk"] for r in mixed_recs) / max(1, len(mixed_recs)), 4),
            "mixedRawTop1": round(sum(r["top1RawOk"] for r in mixed_recs) / max(1, len(mixed_recs)), 4),
            "mixedTop5": round(sum(r["top5Ok"] for r in mixed_recs) / max(1, len(mixed_recs)), 4),
            "mixedDualSourceBothInTop3": round(sum(bool(r["bothInTop3"]) for r in two_gt_recs) / max(1, len(two_gt_recs)), 4) if two_gt_recs else None,
            "mixedDualSourceBothInTop5": round(sum(bool(r["bothInTop5"]) for r in two_gt_recs) / max(1, len(two_gt_recs)), 4) if two_gt_recs else None,
            "falseRejectionRate": round(sum(r["isOutOfScope"] for r in in_recs) / max(1, len(in_recs)), 4),
            "oodRejectionRate": round(sum(r["isOutOfScope"] for r in ood_recs) / max(1, len(ood_recs)), 4),
            "oodAuroc": round(ood_rank.get("auroc"), 4) if ood_rank.get("auroc") is not None else None,
            "clockTickFalsePositives": sum(1 for r in in_recs if r["rawTop1Name"] == "clock_tick" and 38 not in r["allowedLabels"]),
            "meanLatencyMs": round(float(np.mean([r["latencyMs"] for r in m_records if r["latencyMs"] is not None])), 1),
            "elapsedSeconds": round(time.time() - t_m0, 1),
        }
        summary[model] = m_summary
        print(
            f"[{model:9s}] InScope Acc={m_summary['inScopeAcceptedTop1']*100:5.1f}% (Raw={m_summary['inScopeRawTop1']*100:5.1f}%, Top5={m_summary['inScopeTop5']*100:5.1f}%) | "
            f"Pure={m_summary['pureAcceptedTop1']*100:5.1f}% (Raw={m_summary['pureRawTop1']*100:5.1f}%) | "
            f"Mixed={m_summary['mixedAcceptedTop1']*100:5.1f}% (Raw={m_summary['mixedRawTop1']*100:5.1f}%, Both@5={((m_summary['mixedDualSourceBothInTop5'] or 0)*100):4.1f}%) | "
            f"FRR={m_summary['falseRejectionRate']*100:4.1f}% | OOD_Rej={m_summary['oodRejectionRate']*100:5.1f}% (AUROC={m_summary['oodAuroc']}) | "
            f"clk_fp={m_summary['clockTickFalsePositives']} | {m_summary['elapsedSeconds']}s",
            flush=True,
        )

    # Multi-format parity check (WAV vs MP3 vs WebM vs AAC) across representative mixed + pure clips
    parity_report = {}
    if multiformat_sample > 0:
        sample_clips = [c for c in clips if c[0]["in_scope"] == "true"][:multiformat_sample]
        for model in selected_models:
            matches = 0
            total = 0
            for row, wav_b in sample_clips:
                base_pred = run_acoustic_inference(wav_b, model_name=model, confidence_threshold=None)["category"]["id"]
                for fmt in ("mp3", "webm", "aac"):
                    enc = _transcode_bytes(wav_b, fmt)
                    pred = run_acoustic_inference(enc, model_name=model, confidence_threshold=None)["category"]["id"]
                    total += 1
                    if pred == base_pred:
                        matches += 1
            parity_report[model] = round(matches / max(1, total), 4)
        print(f"Format Parity vs WAV (MP3/WebM/AAC across {len(sample_clips)} clips): {parity_report}", flush=True)

    payload = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST),
        "threshold": threshold,
        "summary": summary,
        "formatParity": parity_report,
        "results": all_records,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / out_name
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--no-threshold", action="store_true")
    parser.add_argument("--multiformat-sample", type=int, default=20)
    parser.add_argument("--out", default="external_mixed_eval.json")
    args = parser.parse_args()
    thr = None if args.no_threshold else args.threshold
    evaluate_corpus(args.models, thr, args.multiformat_sample, args.out)
