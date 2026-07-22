#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a train-ready EHR dataset from a selected-feature registry.

The script subsets a candidate NPZ to the approved ordered features, calculates
normalization statistics from observed training values only, and applies those
statistics consistently to train, validation, and test rows. It exports the
normalized tensor, aligned cohort metadata, feature list, scaling table, and a
JSON summary under the requested output name.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_train_only(X_raw, M, split):
    X_raw = X_raw.astype(np.float32)
    M = M.astype(np.float32)
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
        mj = M[:, :, j] > 0

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
    X[M == 0] = 0.0

    return X, feature_mean, feature_std


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--selected_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", required=True)

    args = parser.parse_args()

    input_npz = Path(args.input_npz)
    selected_csv = Path(args.selected_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(input_npz, allow_pickle=True)
    selected = pd.read_csv(selected_csv)

    required_npz = ["X_raw", "mask", "y", "split", "sample_id", "variables", "labels", "sources", "itemids"]
    for k in required_npz:
        if k not in z.files:
            raise KeyError(f"Missing NPZ key {k}. Available keys: {z.files}")

    if "feature_index" not in selected.columns:
        raise ValueError("Selected CSV must contain feature_index column.")

    idx = selected["feature_index"].astype(int).tolist()

    X_raw_all = z["X_raw"].astype(np.float32)
    M_all = z["mask"].astype(np.float32)

    if max(idx) >= X_raw_all.shape[-1]:
        raise RuntimeError(
            f"Selected index {max(idx)} exceeds NPZ feature count {X_raw_all.shape[-1]}. "
            "The selected CSV does not match this NPZ."
        )

    X_raw = X_raw_all[:, :, idx]
    M = M_all[:, :, idx]

    y = z["y"].astype(np.int64)
    split = z["split"].astype(str)
    sample_id = z["sample_id"].astype(str)

    variables = z["variables"].astype(str)[idx]
    labels = z["labels"].astype(str)[idx]
    sources = z["sources"].astype(str)[idx]
    itemids = z["itemids"][idx]

    if "value_sources" in z.files:
        value_sources = z["value_sources"].astype(str)[idx]
    else:
        value_sources = np.array(["valuenum"] * len(idx), dtype=str)

    X, feature_mean, feature_std = normalize_train_only(X_raw, M, split)

    # Human-readable feature names for train_ehr.py.
    feature_names = np.array(
        [
            f"{sources[i]}::{itemids[i]}::{labels[i]}"
            for i in range(len(idx))
        ],
        dtype=str,
    )

    out_npz = out_dir / f"ehr_{args.name}_24h_train_ready.npz"
    out_cohort = out_dir / f"cohort_for_{args.name}_training.csv"
    out_features = out_dir / f"ehr_{args.name}_selected_features.csv"
    out_norm = out_dir / f"ehr_{args.name}_feature_normalization.csv"
    out_summary = out_dir / f"ehr_{args.name}_summary.json"

    np.savez_compressed(
        out_npz,
        X=X.astype(np.float32),
        X_raw=X_raw.astype(np.float32),
        M=M.astype(np.float32),
        mask=M.astype(np.float32),
        y=y.astype(np.int64),
        split=split.astype(str),
        sample_id=sample_id.astype(str),
        feature_names=feature_names.astype(str),
        variables=variables.astype(str),
        labels=labels.astype(str),
        sources=sources.astype(str),
        itemids=itemids,
        value_sources=value_sources.astype(str),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        selected_feature_indices=np.array(idx, dtype=np.int64),
        selected_csv=str(selected_csv),
        source="selected_from_train_only_consensus_24h_current_clean_rebuild",
    )

    cohort = pd.DataFrame({
        "sample_id": sample_id,
        "label": y.astype(int),
        "split": split,
    })
    cohort.to_csv(out_cohort, index=False)

    selected.to_csv(out_features, index=False)

    norm = pd.DataFrame({
        "selected_order": np.arange(len(idx)),
        "feature_index": idx,
        "feature_name": feature_names,
        "source": sources,
        "itemid": itemids,
        "label": labels,
        "train_mean": feature_mean,
        "train_std": feature_std,
        "observed_count_total": M.sum(axis=(0, 1)).astype(int),
        "observed_count_train": M[split == "train"].sum(axis=(0, 1)).astype(int),
    })
    norm.to_csv(out_norm, index=False)

    summary = {
        "input_npz": str(input_npz),
        "selected_csv": str(selected_csv),
        "output_npz": str(out_npz),
        "output_cohort": str(out_cohort),
        "output_features": str(out_features),
        "output_normalization": str(out_norm),
        "shape": list(X.shape),
        "n_features": int(X.shape[-1]),
        "label_counts": pd.Series(y).value_counts().sort_index().astype(int).to_dict(),
        "split_counts": pd.Series(split).value_counts().astype(int).to_dict(),
        "split_label_counts": pd.crosstab(pd.Series(split, name="split"), pd.Series(y, name="label")).astype(int).to_dict(),
    }

    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 120)
    print("DONE")
    print("=" * 120)
    print("Selected CSV:", selected_csv)
    print("Output NPZ:", out_npz)
    print("Output cohort:", out_cohort)
    print("Output selected features:", out_features)
    print("Output normalization:", out_norm)
    print("Output summary:", out_summary)
    print("X:", X.shape)
    print("M:", M.shape)
    print("features:", len(feature_names))
    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"{i:02d} | {name}")


if __name__ == "__main__":
    main()
