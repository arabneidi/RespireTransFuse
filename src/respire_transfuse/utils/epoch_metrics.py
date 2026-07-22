"""Compute and persist epoch-level metrics, predictions, and calibration bins."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)


def as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def to_probability(x):
    x = as_numpy(x).astype(np.float64)

    if np.nanmin(x) < 0.0 or np.nanmax(x) > 1.0:
        return sigmoid_np(x)

    return x


def binary_log_loss_np(y_true, prob, eps=1e-7):
    y_true = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(prob) + (1.0 - y_true) * np.log(1.0 - prob)))


def brier_score_np(y_true, prob):
    y_true = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    return float(np.mean((prob - y_true) ** 2))


def fixed_calibration_bins(y_true, prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        left = float(edges[i])
        right = float(edges[i + 1])

        if i == n_bins - 1:
            mask = (prob >= left) & (prob <= right)
        else:
            mask = (prob >= left) & (prob < right)

        n = int(mask.sum())

        if n == 0:
            rows.append({
                "bin": i + 1,
                "left": left,
                "right": right,
                "n": 0,
                "mean_prob": np.nan,
                "observed_rate": np.nan,
                "gap": np.nan,
                "abs_gap": np.nan,
            })
            continue

        mean_prob = float(prob[mask].mean())
        observed_rate = float(y_true[mask].mean())
        gap = mean_prob - observed_rate

        rows.append({
            "bin": i + 1,
            "left": left,
            "right": right,
            "n": n,
            "mean_prob": mean_prob,
            "observed_rate": observed_rate,
            "gap": gap,
            "abs_gap": abs(gap),
        })

    return pd.DataFrame(rows)


def adaptive_calibration_bins(y_true, prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)

    order = np.argsort(prob)
    y_sorted = y_true[order]
    p_sorted = prob[order]

    chunks = np.array_split(np.arange(len(y_true)), n_bins)
    rows = []

    for i, idx in enumerate(chunks):
        if len(idx) == 0:
            rows.append({
                "bin": i + 1,
                "n": 0,
                "min_prob": np.nan,
                "max_prob": np.nan,
                "mean_prob": np.nan,
                "observed_rate": np.nan,
                "gap": np.nan,
                "abs_gap": np.nan,
            })
            continue

        p_bin = p_sorted[idx]
        y_bin = y_sorted[idx]

        mean_prob = float(p_bin.mean())
        observed_rate = float(y_bin.mean())
        gap = mean_prob - observed_rate

        rows.append({
            "bin": i + 1,
            "n": int(len(idx)),
            "min_prob": float(p_bin.min()),
            "max_prob": float(p_bin.max()),
            "mean_prob": mean_prob,
            "observed_rate": observed_rate,
            "gap": gap,
            "abs_gap": abs(gap),
        })

    return pd.DataFrame(rows)


def calibration_error_from_bins(bins):
    valid = bins.dropna(subset=["abs_gap"]).copy()
    total = float(valid["n"].sum())

    if total <= 0:
        return np.nan, np.nan

    ece = float(((valid["n"] / total) * valid["abs_gap"]).sum())
    mce = float(valid["abs_gap"].max())

    return ece, mce


def best_f1_metrics(y_true, prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)

    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)

    best_idx = int(np.nanargmax(f1))

    if best_idx >= len(thresholds):
        threshold = 1.0
    else:
        threshold = float(thresholds[best_idx])

    pred = (prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision_at_threshold = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    return {
        "best_f1": float(f1[best_idx]),
        "best_f1_threshold": threshold,
        "best_f1_precision": float(precision_at_threshold),
        "best_f1_recall": float(sensitivity),
        "best_f1_specificity": float(specificity),
        "best_f1_npv": float(npv),
        "best_f1_accuracy": float(accuracy),
        "best_f1_fpr": float(fpr),
        "best_f1_fnr": float(fnr),
        "best_f1_tn": int(tn),
        "best_f1_fp": int(fp),
        "best_f1_fn": int(fn),
        "best_f1_tp": int(tp),
    }


def binary_epoch_metrics(y_true, pred_values, prefix, n_bins=10):
    y_true = as_numpy(y_true).astype(np.int64).reshape(-1)
    prob = to_probability(pred_values).reshape(-1)

    out = {}

    out[f"{prefix}_n"] = int(len(y_true))
    out[f"{prefix}_prevalence"] = float(y_true.mean())

    if len(np.unique(y_true)) == 2:
        out[f"{prefix}_auroc"] = float(roc_auc_score(y_true, prob))
        out[f"{prefix}_auprc"] = float(average_precision_score(y_true, prob))
    else:
        out[f"{prefix}_auroc"] = np.nan
        out[f"{prefix}_auprc"] = np.nan

    out[f"{prefix}_log_loss"] = binary_log_loss_np(y_true, prob)
    out[f"{prefix}_brier"] = brier_score_np(y_true, prob)

    fixed_bins = fixed_calibration_bins(y_true, prob, n_bins=n_bins)
    adaptive_bins = adaptive_calibration_bins(y_true, prob, n_bins=n_bins)

    ece, mce = calibration_error_from_bins(fixed_bins)
    adaptive_ece, adaptive_mce = calibration_error_from_bins(adaptive_bins)

    out[f"{prefix}_ece_{n_bins}"] = ece
    out[f"{prefix}_mce_{n_bins}"] = mce
    out[f"{prefix}_adaptive_ece_{n_bins}"] = adaptive_ece
    out[f"{prefix}_adaptive_mce_{n_bins}"] = adaptive_mce

    out[f"{prefix}_mean_prob"] = float(prob.mean())
    out[f"{prefix}_std_prob"] = float(prob.std())
    out[f"{prefix}_min_prob"] = float(prob.min())
    out[f"{prefix}_max_prob"] = float(prob.max())

    if np.any(y_true == 0):
        out[f"{prefix}_mean_prob_neg"] = float(prob[y_true == 0].mean())
        out[f"{prefix}_std_prob_neg"] = float(prob[y_true == 0].std())
    else:
        out[f"{prefix}_mean_prob_neg"] = np.nan
        out[f"{prefix}_std_prob_neg"] = np.nan

    if np.any(y_true == 1):
        out[f"{prefix}_mean_prob_pos"] = float(prob[y_true == 1].mean())
        out[f"{prefix}_std_prob_pos"] = float(prob[y_true == 1].std())
    else:
        out[f"{prefix}_mean_prob_pos"] = np.nan
        out[f"{prefix}_std_prob_pos"] = np.nan

    out[f"{prefix}_prob_gap_pos_neg"] = out[f"{prefix}_mean_prob_pos"] - out[f"{prefix}_mean_prob_neg"]
    out.update({f"{prefix}_{k}": v for k, v in best_f1_metrics(y_true, prob).items()})

    return out, fixed_bins, adaptive_bins


def prediction_frame(epoch, split, sample_ids, y_true, pred_values):
    y_true = as_numpy(y_true).astype(np.int64).reshape(-1)
    raw = as_numpy(pred_values).astype(np.float64).reshape(-1)
    prob = to_probability(raw).reshape(-1)

    if sample_ids is None:
        sample_ids = np.arange(len(y_true))

    sample_ids = as_numpy(sample_ids).reshape(-1)

    df = pd.DataFrame({
        "epoch": int(epoch),
        "split": split,
        "sample_id": sample_ids,
        "label": y_true,
        "probability": prob,
    })

    if np.nanmin(raw) < 0.0 or np.nanmax(raw) > 1.0:
        df["logit"] = raw

    return df


def bins_frame(epoch, split, bins):
    out = bins.copy()
    out.insert(0, "epoch", int(epoch))
    out.insert(1, "split", split)
    return out


def append_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)

    frame.to_csv(path, index=False)
    return path


def save_epoch_artifacts(save_dir, epoch, split, sample_ids, y_true, pred_values, n_bins=10, save_predictions=True):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics, fixed_bins, adaptive_bins = binary_epoch_metrics(
        y_true,
        pred_values,
        split,
        n_bins=n_bins,
    )

    fixed_bins_out = bins_frame(epoch, split, fixed_bins)
    adaptive_bins_out = bins_frame(epoch, split, adaptive_bins)

    append_csv(save_dir / f"calibration_bins_{n_bins}_by_epoch.csv", fixed_bins_out)
    append_csv(save_dir / f"adaptive_calibration_bins_{n_bins}_by_epoch.csv", adaptive_bins_out)

    if save_predictions:
        pred_dir = save_dir / "epoch_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_df = prediction_frame(epoch, split, sample_ids, y_true, pred_values)
        pred_df.to_csv(pred_dir / f"{split}_epoch_{int(epoch):03d}.csv", index=False)

    return metrics
