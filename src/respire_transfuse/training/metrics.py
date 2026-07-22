"""Compute the common binary prediction metrics used across experiments.

The helpers convert logits to probabilities, guard AUROC, AUPRC, log-loss, and
Brier calculations against degenerate inputs, select an F1 operating threshold
from validation data, and summarize threshold-dependent performance. A recursive
converter also makes NumPy and path values safe for JSON result files.
"""

import math

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    log_loss,
    brier_score_loss,
)


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    return obj


def safe_auroc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.sum() == 0:
        return float("nan")

    return float(average_precision_score(y_true, y_prob))


def safe_logloss(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-7, 1.0 - 1e-7)

    return float(log_loss(y_true, y_prob, labels=[0, 1]))


def safe_brier(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)

    return float(brier_score_loss(y_true, y_prob))


def choose_threshold_by_val_f1(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5, 0.0

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.nanargmax(f1))

    return float(thresholds[idx]), float(f1[idx])


def metrics_at_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    pred = (y_prob >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "auroc": safe_auroc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "log_loss": safe_logloss(y_true, y_prob),
        "brier": safe_brier(y_true, y_prob),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize_probabilities(y_true, y_prob, threshold=None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if threshold is None:
        threshold, best_f1 = choose_threshold_by_val_f1(y_true, y_prob)
    else:
        _, best_f1 = choose_threshold_by_val_f1(y_true, y_prob)

    out = metrics_at_threshold(y_true, y_prob, threshold)
    out["best_f1_threshold"] = float(threshold)
    out["best_f1"] = float(best_f1)
    out["prevalence"] = float(y_true.mean())
    out["n"] = int(len(y_true))
    out["positives"] = int(y_true.sum())
    out["prob_mean"] = float(y_prob.mean())
    out["prob_std"] = float(y_prob.std())

    return out


def summarize_logits(y_true, logits, threshold=None):
    prob = sigmoid_np(logits)
    return summarize_probabilities(y_true, prob, threshold=threshold)
