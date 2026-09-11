"""Binary classification metrics and operating-point selection."""

from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from VisualAD_lib.constants import REGION_NAMES


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """ROC-AUC, or NaN when only one class is present."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def safe_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Average precision, or NaN when only one class is present."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def compute_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Threshold probabilities and report the full metric panel."""
    predictions = (probabilities >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    if precision + recall == 0:
        f2 = 0.0
    else:
        f2 = 5.0 * precision * recall / (4.0 * precision + recall)

    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "threshold": float(threshold),
        "auc": safe_auc(labels, probabilities),
        "ap": safe_average_precision(labels, probabilities),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "f2": float(f2),
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def choose_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    objective: str = "recall_at_spec",
    minimum_specificity: float = 0.90,
) -> Tuple[float, Dict[str, float]]:
    """Pick an operating point on the validation set.

    ``recall_at_spec`` maximises recall subject to a specificity floor, which
    matches the clinical requirement of a low false-positive rate. ``f1`` and
    ``f2`` optimise the corresponding F-score directly. Ties are broken
    deterministically by the secondary keys below.
    """
    candidates = np.unique(
        np.concatenate([np.linspace(0.01, 0.99, 99), probabilities])
    )

    best_threshold = 0.5
    best_metrics = compute_binary_metrics(labels, probabilities, best_threshold)
    best_key: Optional[Tuple[float, ...]] = None

    for threshold in candidates:
        metrics = compute_binary_metrics(labels, probabilities, float(threshold))

        if objective == "recall_at_spec":
            feasible = metrics["specificity"] >= minimum_specificity
            key = (
                1.0 if feasible else 0.0,
                metrics["recall"] if feasible else metrics["specificity"],
                metrics["f2"],
                metrics["f1"],
                -abs(float(threshold) - 0.5),
            )
        elif objective == "f2":
            key = (
                metrics["f2"],
                metrics["recall"],
                metrics["specificity"],
                metrics["f1"],
            )
        else:
            key = (
                metrics["f1"],
                metrics["recall"],
                metrics["specificity"],
                metrics["precision"],
            )

        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def compute_region_metrics(
    predictions: Mapping[str, np.ndarray],
    threshold: float,
) -> Dict[str, Dict[str, float]]:
    """Per-region metric panel, keyed by region name."""
    result: Dict[str, Dict[str, float]] = {}

    for region_id, region_name in enumerate(REGION_NAMES):
        mask = predictions["region_id"] == region_id

        labels = predictions["label"][mask]
        probabilities = predictions["probability"][mask]

        if len(labels) == 0:
            result[region_name] = {"n": 0, "positive": 0, "negative": 0}
            continue

        metrics = compute_binary_metrics(labels, probabilities, threshold)
        metrics["n"] = int(len(labels))
        metrics["positive"] = int((labels == 1).sum())
        metrics["negative"] = int((labels == 0).sum())

        result[region_name] = metrics

    return result
