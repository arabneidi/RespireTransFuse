#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter unusable features from a candidate EHR tensor without data leakage.

The script measures observation coverage using training rows only, removes
features that are never observed there, and applies the same retained columns to
all cohort splits. It writes the reduced NPZ, a per-feature retention report, and
a JSON summary while preserving the original sample order and metadata arrays.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Filter EHR features with no train-split observations.")

    parser.add_argument(
        "--input_npz",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_npz",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_features_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_summary_json",
        type=str,
        required=True,
    )

    return parser.parse_args()


def require_keys(z, keys):
    missing = [key for key in keys if key not in z.files]
    if missing:
        raise KeyError(f"Missing required NPZ keys: {missing}. Available keys: {list(z.files)}")


def main():
    args = parse_args()

    input_npz = Path(args.input_npz)
    output_npz = Path(args.output_npz)
    output_features_csv = Path(args.output_features_csv)
    output_summary_json = Path(args.output_summary_json)

    if not input_npz.exists():
        raise FileNotFoundError(input_npz)

    z = np.load(input_npz, allow_pickle=True)

    required = [
        "X_raw",
        "X",
        "mask",
        "y",
        "split",
        "sample_id",
        "variables",
        "labels",
        "sources",
        "itemids",
    ]
    require_keys(z, required)

    X_raw = z["X_raw"].astype(np.float32)
    X = z["X"].astype(np.float32)
    mask = z["mask"].astype(np.float32)
    split = z["split"].astype(str)

    variables = z["variables"].astype(str)
    labels = z["labels"].astype(str)
    sources = z["sources"].astype(str)
    itemids = z["itemids"]

    if "value_sources" in z.files:
        value_sources = z["value_sources"].astype(str)
    else:
        value_sources = np.array(["valuenum"] * len(labels), dtype=str)

    if X_raw.ndim != 3:
        raise ValueError(f"Expected X_raw with shape [N, T, F], got {X_raw.shape}")

    if X.shape != X_raw.shape:
        raise ValueError(f"X shape {X.shape} does not match X_raw shape {X_raw.shape}")

    if mask.shape != X_raw.shape:
        raise ValueError(f"mask shape {mask.shape} does not match X_raw shape {X_raw.shape}")

    if len(split) != X_raw.shape[0]:
        raise ValueError(f"split length {len(split)} does not match sample count {X_raw.shape[0]}")

    train_rows = split == "train"

    if train_rows.sum() == 0:
        raise ValueError("No train rows found.")

    train_observed_entries = mask[train_rows].sum(axis=(0, 1))
    full_observed_entries = mask.sum(axis=(0, 1))
    keep = train_observed_entries > 0

    feature_report = pd.DataFrame({
        "feature_index": np.arange(int(keep.sum()), dtype=int),
        "feature_index_original": np.where(keep)[0].astype(int),
        "source": sources[keep],
        "itemid": itemids[keep],
        "label": labels[keep],
        "variable": variables[keep],
        "value_source": value_sources[keep],
        "train_observed_entries": train_observed_entries[keep].astype(int),
        "full_observed_entries": full_observed_entries[keep].astype(int),
    })

    removed_report = pd.DataFrame({
        "feature_index_original": np.where(~keep)[0].astype(int),
        "source": sources[~keep],
        "itemid": itemids[~keep],
        "label": labels[~keep],
        "variable": variables[~keep],
        "value_source": value_sources[~keep],
        "train_observed_entries": train_observed_entries[~keep].astype(int),
        "full_observed_entries": full_observed_entries[~keep].astype(int),
    })

    save = {}

    for key in z.files:
        arr = z[key]

        if key in ["X_raw", "X", "mask"]:
            save[key] = arr[:, :, keep]

        elif key in ["variables", "labels", "sources", "itemids", "value_sources"]:
            save[key] = arr[keep]

        else:
            save[key] = arr

    save["feature_indices_before_filter"] = np.where(keep)[0].astype(np.int64)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_features_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_npz, **save)
    feature_report.to_csv(output_features_csv, index=False)

    summary = {
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "output_features_csv": str(output_features_csv),
        "output_summary_json": str(output_summary_json),
        "input_shape": list(X_raw.shape),
        "output_shape": list(save["X_raw"].shape),
        "input_features": int(X_raw.shape[-1]),
        "kept_features": int(keep.sum()),
        "removed_features": int((~keep).sum()),
        "train_rows": int(train_rows.sum()),
        "removed_feature_labels": removed_report["label"].astype(str).tolist(),
    }

    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 120)
    print("ZERO-TRAIN FEATURE FILTER")
    print("=" * 120)
    print("Input NPZ:", input_npz)
    print("Output NPZ:", output_npz)
    print("Input shape:", X_raw.shape)
    print("Output shape:", save["X_raw"].shape)
    print("Input features:", int(X_raw.shape[-1]))
    print("Kept features:", int(keep.sum()))
    print("Removed features:", int((~keep).sum()))
    print("Train rows:", int(train_rows.sum()))

    print("\nRemoved features:")
    if removed_report.empty:
        print("None")
    else:
        print(removed_report.to_string(index=False))

    print("\nSaved feature CSV:", output_features_csv)
    print("Saved summary JSON:", output_summary_json)


if __name__ == "__main__":
    main()
