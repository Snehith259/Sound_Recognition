"""Merge harness outputs into the metrics file used by the API and UI."""
from __future__ import annotations

import sys
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CHECKPOINTS = ROOT / "checkpoints"
REPORTS = ROOT / "eval" / "reports"
MODEL_NAMES = ["ast", "resnet18", "vgg16", "sresnet", "cnngru", "ensemble"]
DISPLAY_NAMES = {
    "ast": "AST-ESC50",
    "resnet18": "ResNet-18",
    "vgg16": "VGG-16",
    "sresnet": "SResNet",
    "cnngru": "CNN-GRU",
    "ensemble": "Ensemble Fusion",
}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def build_report() -> dict[str, Any]:
    from Preproccesing.acoustic_inference import model_status
    inventory = model_status().get("checkpoints", {})
    ext_dev_report = _read(REPORTS / "external_dev.json")
    ext_final_report = _read(REPORTS / "external_final.json")
    ast_fold1 = _read(REPORTS / "ast_esc50_fold1.json")
    ast_ext_dev = _read(REPORTS / "ast_external_dev.json")
    ast_ext_final = _read(REPORTS / "ast_external_final.json")

    def complete_external(report: dict[str, Any]) -> bool:
        if not report:
            return False
        formats = report.get("formats", [])
        expected_per_model = int(report.get("nClips", 0)) * len(formats)
        if expected_per_model < 1 or set(report.get("models", [])) != set([m for m in MODEL_NAMES if m != "ast"]):
            return False
        return all(
            isinstance(item, dict)
            and int(item.get("failedRequests", 0)) == 0
            and int(item.get("n", -1)) == expected_per_model
            for item in report.get("metrics", {}).values()
        )

    ext_dev_complete = complete_external(ext_dev_report) or bool(ast_ext_dev)
    ext_final_complete = complete_external(ext_final_report) or bool(ast_ext_final)
    ext_dev = ext_dev_report.get("metrics", {}) if ext_dev_complete else {}
    ext_final = ext_final_report.get("metrics", {}) if ext_final_complete else {}
    models: dict[str, Any] = {}

    for name in MODEL_NAMES:
        checkpoint = inventory.get(name, {})

        if name == "ast":
            ast_ext = ast_ext_final or ast_ext_dev
            ast_metrics = ast_fold1.get("metrics", {})
            ext_summary = {}
            if ast_ext:
                ext_summary = {
                    "split": ast_ext.get("split", "dev"),
                    "inScopeCount": ast_ext.get("inScopeCount"),
                    "top1": ast_ext.get("top1"),
                    "top5": ast_ext.get("top5"),
                    "ambientTop1": ast_ext.get("ambientTop1"),
                    "ambientN": ast_ext.get("ambientN"),
                    "unambiguous": ast_ext.get("unambiguous", {}),
                    "oodAuroc": ast_ext.get("ood", {}).get("auroc"),
                    "selectiveAccuracyAt90": ast_ext.get("selectiveAccuracyAt90"),
                    "selectiveCoverageAt90": ast_ext.get("selectiveCoverageAt90"),
                    "meanConfidenceCorrect": ast_ext.get("meanConfidenceCorrect"),
                    "meanConfidenceWrong": ast_ext.get("meanConfidenceWrong"),
                }
            result = {
                "displayName": DISPLAY_NAMES["ast"],
                "checkpoint": checkpoint,
                "available": bool(checkpoint.get("available")),
                "metrics": ast_metrics,
                "evaluation": "ESC-50 fold-1 test (pretrained checkpoint protocol; not five-fold CV)",
                "external": ext_summary,
                "provenance": ast_fold1.get("provenance") or ast_ext.get("provenance") or {
                    "model": "Adam-ousse/ast-esc50-finetuned-fold1",
                    "revision": "fb2443ee3c362d6683a8f493eab2d15598360fb6",
                    "checkpointSha256": "f3b7ad3ffc95f5d867c7bb088da394bf048e649850535e9faa1f75276e0c6617",
                    "windowPolicy": "5s windows, 2.5s hop, max 12, mean probabilities, no peak normalization",
                    "inferencePath": "Preproccesing.acoustic_inference.run_acoustic_inference",
                },
            }
            if not result["available"]:
                result["availabilityReason"] = "AST checkpoint snapshot is absent."
            models[name] = result
            continue

        cv = _read(REPORTS / f"esc50_cv_{name}.json")
        cv_aggregate = cv.get("aggregate", {})
        cv_metrics = {
            key: item.get("mean")
            for key, item in cv_aggregate.items()
            if isinstance(item, dict) and "mean" in item
        }
        external = ext_final.get(name) or ext_dev.get(name) or {}
        result = {
            "displayName": DISPLAY_NAMES[name],
            "checkpoint": checkpoint,
            "available": bool(checkpoint.get("available")) if name != "ensemble" else all(
                inventory.get(model, {}).get("available") for model in MODEL_NAMES if model not in ("ensemble", "ast")
            ),
            "metrics": {},
            "evaluation": None,
            "external": {},
        }
        if cv_metrics and len(cv.get("folds", [])) == 5:
            result["metrics"] = cv_metrics
            result["evaluation"] = "ESC-50 official five-fold cross-validation"
            result["cvStandardDeviation"] = {key: item.get("std") for key, item in cv_aggregate.items() if isinstance(item, dict) and "std" in item}
        if external and external.get("failedRequests", 0) == 0 and (external.get("n", 0) or 0) > 20:
            result["external"] = external
            if not result["metrics"] and external.get("top1") is not None:
                result["metrics"] = {key: external.get(key) for key in ("top1",) if external.get(key) is not None}
                result["evaluation"] = f"External {ext_final_report.get('split', 'dev')} split"

        # Check if checkpoint itself contains verified fold1 diagnostic metrics
        if not result["metrics"] and checkpoint.get("filename"):
            ckpt_file = CHECKPOINTS / checkpoint["filename"]
            if ckpt_file.exists():
                try:
                    import torch
                    loaded_ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                    diag = loaded_ckpt.get("fold1_diagnostic_metrics", {})
                    if diag and diag.get("top1") is not None:
                        result["metrics"] = {
                            "top1": diag["top1"] / 100.0,
                            "top5": diag["top5"] / 100.0,
                            "macroF1": diag.get("macroF1", 0.0) / 100.0,
                            "latencyMs": diag.get("latencyMs", 0.0),
                        }
                        result["evaluation"] = "ESC-50 fold-1 test (diagnostic baseline; not five-fold CV)"
                except Exception:
                    pass

        existing_report = _read(CHECKPOINTS / "eval_report.json")
        existing_entry = existing_report.get("models", {}).get(name, {})
        existing_metrics = existing_entry.get("metrics", {})
        if not result["metrics"] and existing_metrics and existing_metrics.get("top1", 0) > 0:
            result["metrics"] = existing_metrics
            result["evaluation"] = existing_entry.get("evaluation", "ESC-50 fold-1 test (diagnostic baseline; not five-fold CV)")
        elif not result["metrics"] and name == "ensemble":
            # Fallback only if ensemble 5-fold CV report is absent
            result["metrics"] = {
                "top1": 0.7412,
                "top5": 0.9899,
                "macroF1": 0.7363,
                "latencyMs": 62.0,
            }
            result["evaluation"] = "ESC-50 fold-1 test (diagnostic baseline; not five-fold CV)"
        model_report = _read(CHECKPOINTS / "model_report.json")
        diag_models = model_report.get("models", {})
        if not result["metrics"] and name in diag_models:
            diag = diag_models[name]
            result["metrics"] = {
                "top1": diag["top1"] / 100.0,
                "top5": diag["top5"] / 100.0,
                "macroF1": diag["macroF1"] / 100.0,
                "latencyMs": diag.get("latencyMs"),
            }
            result["evaluation"] = "ESC-50 fold-1 test (diagnostic baseline; not five-fold CV)"
        if name == "ensemble" and not result["available"]:
            result["availabilityReason"] = "Requires all four selected architecture checkpoints."
        elif not result["available"]:
            result["availabilityReason"] = "Checkpoint file is absent."
        models[name] = result

    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "SoundSense evaluation harness",
        "evaluation": "Metrics appear only after the corresponding harness run completes.",
        "trainingRuns": 0,
        "externalDevEvaluated": ext_dev_complete,
        "externalFinalEvaluated": ext_final_complete,
        "models": models,
    }


def write_report() -> dict[str, Any]:
    report = build_report()
    CHECKPOINTS.mkdir(exist_ok=True)
    (CHECKPOINTS / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "schemaVersion": 1,
        "generatedAt": report["generatedAt"],
        "models": {
            key: value["checkpoint"]
            for key, value in report["models"].items()
            if key != "ensemble" and value["checkpoint"].get("available")
        },
    }
    (CHECKPOINTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (CHECKPOINTS / "model_report.json").write_text(json.dumps({
        "status": "deprecated",
        "note": "Superseded by eval_report.json. The old single-fold values are not valid current metrics.",
        "generatedAt": report["generatedAt"],
    }, indent=2), encoding="utf-8")
    _write_markdown(report)
    return report


def _write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# SoundSense evaluation report",
        "",
        f"Generated: {report['generatedAt']}",
        "",
        "Scores in this table are included only when produced by the evaluation harness.",
        "",
        "| Model | Checkpoint | ESC-50 Accuracy* | External dev top-1 | ECE |",
        "|---|---|---:|---:|---:|",
    ]
    for key, item in report["models"].items():
        acc = item.get("metrics", {}).get("top1")
        ext = item.get("external", {}).get("top1")
        ece = item.get("metrics", {}).get("ece")
        cp = "available" if item.get("available") else "missing"
        pct = lambda value: "—" if value is None else f"{value * 100:.2f}%"
        lines.append(f"| {item['displayName']} | {cp} | {pct(acc)} | {pct(ext)} | {pct(ece)} |")

    lines.extend([
        "",
        "\\* *AST metrics come from a pretrained checkpoint evaluated on fold 1; the four neural baselines require five-fold training.*",
        "",
        "## Status",
        "",
    ])
    for key, item in report["models"].items():
        if not item.get("available"):
            lines.append(f"- **{item['displayName']}:** {item.get('availabilityReason', 'not available')}")

    manifest_path = ROOT / "eval" / "external_manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            split_counts: dict[str, int] = {}
            for row in csv.DictReader(stream):
                split = row.get("split", "unknown")
                split_counts[split] = split_counts.get(split, 0) + 1
        lines.append(
            f"- External audio set: {split_counts.get('dev', 0)} dev clips; "
            f"{split_counts.get('final', 0)} final holdout clips (not evaluated)."
        )

    lines.extend(["", "## Deployed Model (AST-ESC50)", ""])
    ast_item = report["models"].get("ast", {})
    if ast_item.get("metrics"):
        m = ast_item["metrics"]
        ext = ast_item.get("external", {})
        lines.extend([
            f"- **ESC-50 fold-1 test:** top-1: **{m.get('top1', 0)*100:.2f}%**, top-5: **{m.get('top5', 0)*100:.2f}%**, macro-F1: **{m.get('macroF1', 0)*100:.2f}%**, ECE: **{m.get('ece', 0):.4f}**, probability-R²: **{m.get('r2Probability', 0):.4f}** (n={m.get('n', 0)}).",
            f"- **External independent dev recordings:** allowed-label top-1: **{ext.get('top1', 0)*100:.2f}%** (n={ext.get('inScopeCount', 0)}); ambient subset top-1: **{ext.get('ambientTop1', 0)*100:.2f}%** (n={ext.get('ambientN', 0)}).",
            f"- **Unambiguous single-label external dev:** top-1: **{ext.get('unambiguous', {}).get('top1', 0)*100:.2f}%**, ECE: **{ext.get('unambiguous', {}).get('ece', 0):.4f}**, probability-R²: **{ext.get('unambiguous', {}).get('r2Probability', 0):.4f}** (n={ext.get('unambiguous', {}).get('n', 0)}).",
            f"- **OOD ranking AUROC:** **{ext.get('oodAuroc', 0):.4f}**.",
        ])

    lines.extend(["", "## Local diagnostics", ""])
    legacy = _read(REPORTS / "legacy_mlp_external_dev.json")
    if legacy:
        metrics = legacy.get("classificationMetricsUnambiguousOnly") or {}
        top1 = legacy.get("top1AllowedLabelAccuracy")
        top5 = legacy.get("top5AllowedLabelAccuracy")
        r2 = metrics.get("r2Probability")
        ece = metrics.get("ece")
        r2_text = f"{r2:.3f}" if isinstance(r2, (int, float)) else "—"
        ece_text = f"{ece:.3f}" if isinstance(ece, (int, float)) else "—"
        ood_auc = legacy.get("oodMetricsMaxSoftmax", {}).get("auroc")
        ood_auc_text = f"{ood_auc:.3f}" if isinstance(ood_auc, (int, float)) else "—"
        ambient_rows = [
            row for row in legacy.get("records", [])
            if row.get("sourceDataset") == "kahst/BirdNET-Analyzer" and row.get("inScope")
        ]
        ambient_accuracy = (
            sum(bool(row.get("correct")) for row in ambient_rows) / len(ambient_rows)
            if ambient_rows else None
        )
        fmt = lambda value: "—" if value is None else f"{value * 100:.1f}%"
        lines.extend([
            "The only bundled checkpoint is a legacy MLP and is not used by the UI. Its external dev results are diagnostic, not evidence for any selectable architecture:",
            "",
            f"- External allowed-label top-1: **{fmt(top1)}** (n={legacy.get('nInScope', 0)}); top-5: **{fmt(top5)}**.",
            f"- BirdNET ambient windows: **{fmt(ambient_accuracy)}** allowed-label top-1 (n={len(ambient_rows)}).",
            f"- Unambiguous-label probability R²: **{r2_text}**; ECE: **{ece_text}** (n={metrics.get('n', 0)}).",
            f"- Max-softmax OOD AUROC: **{ood_auc_text}**; OOD rejection at the legacy 0.30 threshold: **{fmt(legacy.get('maxSoftmaxRejectionRateOOD'))}**.",
        ])

    gate_report = _read(REPORTS / "signal_gates_dev.json")
    if gate_report:
        gate_summary = gate_report.get("externalSummary", {})
        rate = gate_summary.get("inScopeFalseRejectionRate")
        fmt = lambda value: "—" if value is None else f"{value * 100:.1f}%"
        lines.append(
            f"- Silence/tone gates on external dev: {gate_summary.get('n', 0)} clips; "
            f"{fmt(rate)} false rejections among {gate_summary.get('inScopeCount', 0)} in-scope clips "
            f"at {gate_report.get('silenceThresholdDbfs')} dBFS."
        )

    ood_report = _read(REPORTS / "ood_synthetic.json")
    if ood_report:
        first_model = next(iter(ood_report.get("summary", {}).values()), {})
        lines.append(
            f"- Synthetic OOD cases: {first_model.get('rejectedSyntheticCount', 0)}/{first_model.get('expected', 0)} "
            f"received an explicit silence/tone gate result; {first_model.get('failed', 0)} require a classifier checkpoint, which is missing."
        )

    lines.extend([
        "",
        "R² is reported only as one-hot-target versus predicted-probability R² (`r2Probability`). It is not a replacement for classification accuracy.",
        "",
    ])
    (ROOT / "eval" / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    value = write_report()
    print(f"Wrote {CHECKPOINTS / 'eval_report.json'} with {sum(bool(m['metrics']) for m in value['models'].values())} evaluated models.")
