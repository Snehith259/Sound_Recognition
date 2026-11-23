"""Metrics shared by ESC-50 and external audio evaluation."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def classification_metrics(
    targets: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str] | None = None,
    ece_bins: int = 15,
) -> dict[str, Any]:
    """Return standard classifier metrics, including the documented probability R²."""
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or len(y) != len(p) or not len(y):
        raise ValueError("targets and probabilities must describe the same non-empty batch")
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    n, n_classes = p.shape
    names = list(class_names) if class_names is not None else [str(i) for i in range(n_classes)]
    pred = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = pred == y
    top_k = np.argsort(p, axis=1)[:, -min(5, n_classes):]
    top5 = (top_k == y[:, None]).any(axis=1)

    recalls: dict[str, float | None] = {}
    f1s = []
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_id, pred_id in zip(y, pred):
        if 0 <= true_id < n_classes and 0 <= pred_id < n_classes:
            confusion[true_id, pred_id] += 1
    for class_id, name in enumerate(names):
        true_mask = y == class_id
        pred_mask = pred == class_id
        support = int(true_mask.sum())
        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = support - tp
        recall = tp / support if support else None
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.0
        recalls[name] = recall
        if support:
            f1s.append(f1)

    one_hot = np.eye(n_classes, dtype=np.float64)[y]
    sse = float(np.square(one_hot - p).sum())
    sst = float(np.square(one_hot - one_hot.mean()).sum())
    ece = 0.0
    bin_data = []
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    for idx in range(ece_bins):
        mask = (confidence >= edges[idx]) & (confidence < edges[idx + 1] if idx < ece_bins - 1 else confidence <= edges[idx + 1])
        count = int(mask.sum())
        if count:
            acc = float(correct[mask].mean())
            conf = float(confidence[mask].mean())
            ece += count / n * abs(acc - conf)
            bin_data.append({"lower": float(edges[idx]), "upper": float(edges[idx + 1]), "count": count, "accuracy": acc, "confidence": conf})

    selective = confidence >= 0.90
    return {
        "n": n,
        "top1": float(correct.mean()),
        "top5": float(top5.mean()),
        "macroF1": float(np.mean(f1s)) if f1s else 0.0,
        "perClassRecall": recalls,
        "confusionMatrix": confusion.tolist(),
        "nll": float(-np.log(p[np.arange(n), y]).mean()),
        "brier": float(np.square(p - one_hot).sum(axis=1).mean()),
        "ece": float(ece),
        "reliability": bin_data,
        "r2Probability": float(1.0 - sse / sst) if sst else None,
        "meanConfidence": float(confidence.mean()),
        "accuracyAtConfidence90": float(correct[selective].mean()) if selective.any() else None,
        "coverageAtConfidence90": float(selective.mean()),
    }


def binary_ood_metrics(is_ood: Sequence[bool], ood_scores: Sequence[float]) -> dict[str, Any]:
    """Compute OOD ranking and a transparent rejection rate at 90% in-scope acceptance."""
    y = np.asarray(is_ood, dtype=np.int64)
    scores = np.asarray(ood_scores, dtype=np.float64)
    if len(y) != len(scores) or not len(y):
        return {"n": 0, "auroc": None, "aupr": None, "rejectionAt90InScopeAcceptance": None}
    if len(np.unique(y)) < 2:
        auroc = aupr = None
    else:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auroc = float(roc_auc_score(y, scores))
        aupr = float(average_precision_score(y, scores))
    in_scope_scores = scores[y == 0]
    threshold = float(np.quantile(in_scope_scores, 0.90)) if len(in_scope_scores) else 1.0
    rejected = scores >= threshold
    return {
        "n": int(len(y)),
        "oodCount": int(y.sum()),
        "inScopeCount": int((y == 0).sum()),
        "auroc": auroc,
        "aupr": aupr,
        "rejectionAt90InScopeAcceptance": float(rejected[y == 1].mean()) if np.any(y == 1) else None,
        "thresholdAt90InScopeAcceptance": threshold,
    }
