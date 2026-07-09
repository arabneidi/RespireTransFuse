#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLINICAL_SELECTED = [
    ("chartevents", "220045", "Heart Rate"),
    ("chartevents", "220210", "Respiratory Rate"),
    ("chartevents", "220277", "O2 saturation pulseoxymetry"),
    ("chartevents", "220179", "Non Invasive Blood Pressure systolic"),
    ("chartevents", "220180", "Non Invasive Blood Pressure diastolic"),
    ("chartevents", "223761", "Temperature Fahrenheit"),
    ("chartevents", "223834", "O2 Flow"),
    ("chartevents", "223835", "Inspired O2 Fraction"),
    ("labevents", "50813", "Lactate"),
    ("labevents", "50820", "pH"),
    ("labevents", "50802", "Base Excess"),
    ("labevents", "50882", "Bicarbonate"),
    ("labevents", "50811", "Hemoglobin"),
    ("labevents", "51301", "White Blood Cells"),
    ("labevents", "50912", "Creatinine"),
    ("labevents", "51274", "PT"),
    ("labevents", "51237", "INR(PT)"),
    ("labevents", "50960", "Magnesium"),
    ("labevents", "50970", "Phosphate"),
    ("labevents", "50893", "Calcium, Total"),
]


ALGORITHMIC_SELECTED = [
    ("chartevents", "225638", "Differential-Bands"),
    ("chartevents", "220224", "Arterial O2 pressure"),
    ("labevents", "50884", "Bilirubin, Indirect"),
    ("chartevents", "226534", "Sodium (whole blood)"),
    ("chartevents", "226536", "Chloride (whole blood)"),
    ("chartevents", "229357", "Absolute Count - Neuts"),
    ("labevents", "50924", "Ferritin"),
    ("chartevents", "220632", "LDH"),
    ("labevents", "51214", "Fibrinogen, Functional"),
    ("labevents", "51283", "Reticulocyte Count, Automated"),
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo_root",
        type=str,
        default="/content/drive/MyDrive/respire-transfuse",
    )

    parser.add_argument(
        "--broad_npz",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--clinical_consensus_csv",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--algorithmic_consensus_csv",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_npz_name",
        type=str,
        default="ehr_24h_final_current_split.npz",
    )

    parser.add_argument(
        "--output_features_name",
        type=str,
        default="ehr_24h_final_selected_features.csv",
    )

    parser.add_argument(
        "--output_summary_name",
        type=str,
        default="ehr_24h_final_selected_summary.json",
    )

    return parser.parse_args()


def default_paths(repo_root: Path, args):
    broad_npz = Path(args.broad_npz) if args.broad_npz else (
        repo_root
        / "data/processed/ehr/ehr_broad_feature_selection_24h"
        / "ehr_24h_broad_current_split.npz"
    )

    clinical_consensus_csv = Path(args.clinical_consensus_csv) if args.clinical_consensus_csv else (
        repo_root
        / "data/processed/ehr/ehr_feature_selection_24h/features/selection_clinical"
        / "ehr_24h_current_split_nonzero_train_consensus_feature_evidence_train_only.csv"
    )

    algorithmic_consensus_csv = Path(args.algorithmic_consensus_csv) if args.algorithmic_consensus_csv else (
        repo_root
        / "data/processed/ehr/ehr_broad_feature_selection_24h/features/selection_strict_v4"
        / "ehr_24h_broad_current_split_consensus_feature_evidence_train_only.csv"
    )

    output_dir = Path(args.output_dir) if args.output_dir else (
        repo_root / "data/processed/ehr/ehr_final_24h"
    )

    return broad_npz, clinical_consensus_csv, algorithmic_consensus_csv, output_dir


def load_evidence_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing evidence CSV: {path}")

    df = pd.read_csv(path)

    required = ["source", "itemid", "label", "feature_index"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    df["source"] = df["source"].astype(str)
    df["itemid"] = df["itemid"].astype(str)
    df["label"] = df["label"].astype(str)

    return df


def select_rows(evidence_df: pd.DataFrame, selected_items, branch_name: str) -> pd.DataFrame:
    rows = []

    for rank, (source, itemid, label) in enumerate(selected_items, start=1):
        matched = evidence_df[
            (evidence_df["source"].astype(str) == str(source))
            & (evidence_df["itemid"].astype(str) == str(itemid))
            & (evidence_df["label"].astype(str) == str(label))
        ]

        if len(matched) == 0:
            nearby = evidence_df[
                evidence_df["label"].astype(str).str.contains(str(label).split()[0], case=False, na=False)
            ][["source", "itemid", "label", "recommendation"]].head(20)

            raise RuntimeError(
                f"Missing selected feature in {branch_name}: "
                f"source={source}, itemid={itemid}, label={label}\n"
                f"Nearby rows:\n{nearby.to_string(index=False)}"
            )

        if len(matched) > 1:
            raise RuntimeError(
                f"Selected feature matched multiple rows in {branch_name}: "
                f"source={source}, itemid={itemid}, label={label}"
            )

        row = matched.iloc[0].copy()
        row["final_branch"] = branch_name
        row["final_rank_within_branch"] = rank
        row["evidence_feature_index"] = int(row["feature_index"])
        rows.append(row)

    return pd.DataFrame(rows)


def match_broad_indices(final_df: pd.DataFrame, z) -> list:
    broad_sources = np.array([str(x) for x in z["sources"]])
    broad_itemids = np.array([str(x) for x in z["itemids"]])
    broad_labels = np.array([str(x) for x in z["labels"]])

    broad_indices = []

    for _, row in final_df.iterrows():
        matched = np.where(
            (broad_sources == str(row["source"]))
            & (broad_itemids == str(row["itemid"]))
            & (broad_labels == str(row["label"]))
        )[0]

        if len(matched) == 0:
            raise RuntimeError(
                "Feature not found in broad NPZ: "
                f"source={row['source']}, itemid={row['itemid']}, label={row['label']}"
            )

        if len(matched) > 1:
            raise RuntimeError(
                "Feature matched multiple broad NPZ columns: "
                f"source={row['source']}, itemid={row['itemid']}, label={row['label']}"
            )

        broad_indices.append(int(matched[0]))

    return broad_indices


def build_final_npz(z, broad_indices: list) -> dict:
    payload = {}

    for key in z.files:
        arr = z[key]

        if key in ["X_raw", "mask"]:
            payload[key] = arr[:, :, broad_indices]
        elif key == "X":
            payload[key] = arr[:, :, broad_indices]
        elif key in ["variables", "labels", "sources", "itemids", "value_sources"]:
            payload[key] = arr[broad_indices]
        else:
            payload[key] = arr

    return payload


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)

    broad_npz_path, clinical_csv, algorithmic_csv, output_dir = default_paths(repo_root, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_npz_path = output_dir / args.output_npz_name
    output_features_csv = output_dir / args.output_features_name
    output_summary_json = output_dir / args.output_summary_name
    output_clinical_csv = output_dir / "ehr_24h_final_clinical_features.csv"
    output_algorithmic_csv = output_dir / "ehr_24h_final_algorithmic_features.csv"

    if not broad_npz_path.exists():
        raise FileNotFoundError(f"Missing broad NPZ: {broad_npz_path}")

    clinical_df = load_evidence_csv(clinical_csv)
    algorithmic_df = load_evidence_csv(algorithmic_csv)

    final_clinical = select_rows(clinical_df, CLINICAL_SELECTED, "clinical")
    final_algorithmic = select_rows(algorithmic_df, ALGORITHMIC_SELECTED, "algorithmic")

    final_df = pd.concat([final_clinical, final_algorithmic], ignore_index=True)
    final_df["final_feature_order"] = np.arange(len(final_df), dtype=int)

    if len(final_clinical) != 20:
        raise RuntimeError(f"Expected 20 clinical features, got {len(final_clinical)}")

    if len(final_algorithmic) != 10:
        raise RuntimeError(f"Expected 10 algorithmic features, got {len(final_algorithmic)}")

    if len(final_df) != 30:
        raise RuntimeError(f"Expected 30 final features, got {len(final_df)}")

    duplicate_mask = final_df.duplicated(["source", "itemid", "label"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = final_df.loc[
            duplicate_mask,
            ["final_branch", "source", "itemid", "label"],
        ]
        raise RuntimeError(f"Duplicate final features found:\n{duplicate_rows.to_string(index=False)}")

    z = np.load(broad_npz_path, allow_pickle=True)

    required_npz_keys = ["X_raw", "mask", "sources", "itemids", "labels"]
    missing_npz_keys = [k for k in required_npz_keys if k not in z.files]
    if missing_npz_keys:
        raise KeyError(f"Broad NPZ missing keys: {missing_npz_keys}")

    broad_indices = match_broad_indices(final_df, z)
    final_df["broad_feature_index"] = broad_indices

    payload = build_final_npz(z, broad_indices)

    if payload["X_raw"].shape[2] != 30:
        raise RuntimeError(f"Expected final feature dimension 30, got {payload['X_raw'].shape}")

    np.savez_compressed(output_npz_path, **payload)

    final_df.to_csv(output_features_csv, index=False)
    final_clinical.to_csv(output_clinical_csv, index=False)
    final_algorithmic.to_csv(output_algorithmic_csv, index=False)

    summary = {
        "input_broad_npz": str(broad_npz_path),
        "clinical_consensus_csv": str(clinical_csv),
        "algorithmic_consensus_csv": str(algorithmic_csv),
        "output_npz": str(output_npz_path),
        "output_features_csv": str(output_features_csv),
        "output_clinical_csv": str(output_clinical_csv),
        "output_algorithmic_csv": str(output_algorithmic_csv),
        "n_final_features": int(len(final_df)),
        "n_clinical_features": int((final_df["final_branch"] == "clinical").sum()),
        "n_algorithmic_features": int((final_df["final_branch"] == "algorithmic").sum()),
        "final_shape": list(payload["X_raw"].shape),
        "clinical_labels": final_df.loc[final_df["final_branch"] == "clinical", "label"].astype(str).tolist(),
        "algorithmic_labels": final_df.loc[final_df["final_branch"] == "algorithmic", "label"].astype(str).tolist(),
        "broad_feature_indices": [int(x) for x in broad_indices],
    }

    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved final NPZ:", output_npz_path)
    print("Saved final features:", output_features_csv)
    print("Saved clinical features:", output_clinical_csv)
    print("Saved algorithmic features:", output_algorithmic_csv)
    print("Saved summary:", output_summary_json)
    print("Final X_raw shape:", payload["X_raw"].shape)

    display_cols = [
        "final_feature_order",
        "final_branch",
        "final_rank_within_branch",
        "source",
        "itemid",
        "label",
        "recommendation",
        "summary_mode",
        "valid_fraction",
        "hourly_coverage",
        "elasticnet_selection_probability",
        "consensus_score",
        "broad_feature_index",
    ]

    existing_cols = [c for c in display_cols if c in final_df.columns]

    print("\nFinal selected features:")
    print(final_df[existing_cols].to_string(index=False))


if __name__ == "__main__":
    main()
