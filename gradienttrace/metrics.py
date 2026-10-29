from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _nan() -> float:
    return float("nan")


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - probs[mask].mean())
    return float(ece)


def binary_detection_metrics(
    y_true, probs, target_fprs: tuple[float, ...] = (0.01, 0.05)
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    out: dict[str, float] = {"n": int(y.size), "pos_rate": float(y.mean()) if y.size else _nan()}
    for t in target_fprs:
        out[f"tpr_at_fpr_{t:g}"] = _nan()
    if y.size == 0 or y.min() == y.max():
        out.update(auroc=_nan(), auprc=_nan(), accuracy=_nan(), best_threshold=_nan(), ece=_nan())
        return out

    out["auroc"] = float(roc_auc_score(y, p))
    out["auprc"] = float(average_precision_score(y, p))
    out["accuracy"] = float(((p >= 0.5) == y).mean())

    fpr, tpr, thresholds = roc_curve(y, p)
    for t in target_fprs:
        mask = fpr <= t
        out[f"tpr_at_fpr_{t:g}"] = float(tpr[mask].max()) if mask.any() else 0.0

    youden = tpr - fpr
    best = int(np.argmax(youden))
    out["best_threshold"] = float(thresholds[best])
    out["ece"] = expected_calibration_error(y, p)
    return out


def sliced_report(y_true, probs, slices: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Overall metrics plus metrics for each value of every metadata slice."""
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)
    report: dict[str, dict[str, float]] = {"overall": binary_detection_metrics(y, p)}
    for name, values in slices.items():
        values = np.asarray(values)
        for value in sorted(set(values.tolist())):
            mask = values == value
            report[f"{name}:{value}"] = binary_detection_metrics(y[mask], p[mask])
    return report
