#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_train_only(X_raw, mask, split):
    X_raw = X_raw.astype(np.float32)
    mask = mask.astype(np.float32)
    split = np.asarray(split).astype(str)

    N, T, F = X_raw.shape

    X = np.zeros_like(X_raw, dtype=np.float32)
    feature_mean = np.zeros(F, dtype=np.float32)
    feature_std = np.ones(F, dtype=np.float32)

    train_rows = split == "train"

    if train_rows.sum() == 0:
        raise RuntimeError("No train rows found.")

    for j in range(F):
        xj = X_raw[:, :, j]
        mj = mask[:, :, j] > 0

        train_obs = train_rows[:, None] & mj & np.isfinite(xj)

        if train_obs.sum() == 0:
            mean = 0.0
            std = 1.0
        else:
            vals = xj[train_obs].astype(np.float64)
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals))

            if not np.isfinite(std) or std < 1e-6:
                std = 1.0

        feature_mean[j] = mean
        feature_std[j] = std

        obs = mj & np.isfinite(xj)
        X[:, :, j][obs] = ((xj[obs] - mean) / std).astype(np.float32)

    X[~np.isfinite(X)] = 0.0
    X[mask == 0] = 0.0

    return X, feature_mean, feature_std


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_npz",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--features_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="ehr_24h_final_train_ready_current_split.npz",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_npz = Path(args.input_npz)
    features_csv = Path(args.features_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_npz.exists():
        raise FileNotFoundError(input_npz)

    if not features_csv.exists():
        raise FileNotFoundError(features_csv)

    z = np.load(input_npz, allow_pickle=True)
    features = pd.read_csv(features_csv)

    required = [
        "X_raw",
        "mask",
        "y",
        "split",
        "sample_id",
        "variables",
        "labels",
        "sources",
        "itemids",
    ]

    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"Missing NPZ keys: {missing}. Available: {z.files}")

    X_raw = z["X_raw"].astype(np.float32)
    mask = z["mask"].astype(np.float32)
    y = z["y"].astype(np.int64)
    split = z["split"].astype(str)
    sample_id = z["sample_id"].astype(str)

    variables = z["variables"].astype(str)
    labels = z["labels"].astype(str)
    sources = z["sources"].astype(str)
    itemids = z["itemids"]

    if "value_sources" in z.files:
        value_sources = z["value_sources"].astype(str)
    else:
        value_sources = np.array(["valuenum"] * X_raw.shape[-1], dtype=str)

    if X_raw.ndim != 3:
        raise RuntimeError(f"Expected X_raw [N,T,F], got {X_raw.shape}")

    if X_raw.shape != mask.shape:
        raise RuntimeError(f"X_raw shape {X_raw.shape} does not match mask shape {mask.shape}")

    if X_raw.shape[-1] != 30:
        raise RuntimeError(f"Expected 30 final features, got {X_raw.shape[-1]}")

    if len(features) != 30:
        raise RuntimeError(f"Expected 30 rows in features CSV, got {len(features)}")

    X, feature_mean, feature_std = normalize_train_only(X_raw, mask, split)

    feature_names = np.array(
        [
            f"{sources[i]}::{itemids[i]}::{labels[i]}"
            for i in range(X_raw.shape[-1])
        ],
        dtype=str,
    )

    output_npz = output_dir / args.output_name
    output_norm_csv = output_dir / "ehr_24h_final_feature_normalization.csv"
    output_features_csv = output_dir / "ehr_24h_final_train_ready_features.csv"
    output_summary_json = output_dir / "ehr_24h_final_train_ready_summary.json"

    payload = {}

    for key in z.files:
        if key in ["X"]:
            continue
        payload[key] = z[key]

    payload["X"] = X.astype(np.float32)
    payload["X_raw"] = X_raw.astype(np.float32)
    payload["M"] = mask.astype(np.float32)
    payload["mask"] = mask.astype(np.float32)
    payload["y"] = y.astype(np.int64)
    payload["split"] = split.astype(str)
    payload["sample_id"] = sample_id.astype(str)
    payload["feature_names"] = feature_names.astype(str)
    payload["variables"] = variables.astype(str)
    payload["labels"] = labels.astype(str)
    payload["sources"] = sources.astype(str)
    payload["itemids"] = itemids
    payload["value_sources"] = value_sources.astype(str)
    payload["feature_mean"] = feature_mean.astype(np.float32)
    payload["feature_std"] = feature_std.astype(np.float32)
    payload["source"] = "final_30_feature_train_ready_current_split"

    np.savez_compressed(output_npz, **payload)

    features.to_csv(output_features_csv, index=False)

    norm = pd.DataFrame({
        "feature_order": np.arange(X_raw.shape[-1], dtype=int),
        "feature_name": feature_names,
        "source": sources,
        "itemid": itemids,
        "label": labels,
        "train_mean": feature_mean,
        "train_std": feature_std,
        "observed_count_total": mask.sum(axis=(0, 1)).astype(int),
        "observed_count_train": mask[split == "train"].sum(axis=(0, 1)).astype(int),
    })

    norm.to_csv(output_norm_csv, index=False)

    summary = {
        "input_npz": str(input_npz),
        "features_csv": str(features_csv),
        "output_npz": str(output_npz),
        "output_features_csv": str(output_features_csv),
        "output_normalization_csv": str(output_norm_csv),
        "shape": list(X.shape),
        "n_features": int(X.shape[-1]),
        "split_counts": pd.Series(split).value_counts().astype(int).to_dict(),
        "label_counts": pd.Series(y).value_counts().sort_index().astype(int).to_dict(),
        "has_X": True,
        "has_X_raw": True,
        "has_mask": True,
        "has_M": True,
    }

    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved train-ready NPZ:", output_npz)
    print("Saved features:", output_features_csv)
    print("Saved normalization:", output_norm_csv)
    print("Saved summary:", output_summary_json)
    print("X:", X.shape)
    print("X_raw:", X_raw.shape)
    print("mask:", mask.shape)
    print("features:", X.shape[-1])
    print("split counts:", summary["split_counts"])
    print("label counts:", summary["label_counts"])


if __name__ == "__main__":
    main()
