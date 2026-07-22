#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a feature registry that combines clinical and algorithmic choices.

The script reads a consensus ranking table, matches named clinical variables,
adds the highest-ranked remaining data-driven candidates, and records the source
of every retained feature. The resulting CSV is an ordered, duplicate-free input
registry for tensor construction workflows that use the mixed selection scheme.
"""

import argparse
from pathlib import Path

import pandas as pd


CLINICAL_LABELS = [
    "Heart Rate",
    "Respiratory Rate",
    "Non Invasive Blood Pressure systolic",
    "Non Invasive Blood Pressure diastolic",
    "Arterial Blood Pressure systolic",
    "O2 saturation pulseoxymetry",
    "Temperature Fahrenheit",
    "Inspired O2 Fraction",
    "O2 Flow",
    "BiPap O2 Flow",
    "O2 Flow (additional cannula)",
    "Oxygen",
    "Oxygen Saturation",
    "Base Excess",
    "Bicarbonate",
    "pH",
    "Lactate",
    "Hemoglobin",
    "Glucose (serum)",
    "Glucose finger stick (range 70-100)",
]


ALGORITHM_LABELS = [
    "Free Calcium",
    "White Blood Cells",
    "Creatinine",
    "PT",
    "INR(PT)",
    "Magnesium",
    "Phosphate",
    "Monocytes",
    "Calcium, Total",
    "Neutrophils",
]


RECOMMENDATION_ORDER = {
    "core": 1,
    "strong": 2,
    "review": 3,
    "weak_or_noise": 4,
    "reject_or_review_sparse": 5,
    "reject_low_signal": 6,
    "reject_static_like_temporal": 7,
    "reject_possible_leakage": 8,
}


def normalize_label(x):
    return str(x).strip().lower()


def add_recommendation_order(df):
    if "recommendation_order" not in df.columns:
        df["recommendation_order"] = df["recommendation"].map(RECOMMENDATION_ORDER).fillna(99).astype(int)
    return df


def require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def select_one(df, label, group_name):
    label_norm = normalize_label(label)
    candidates = df[df["label"].map(normalize_label) == label_norm].copy()

    if candidates.empty:
        raise RuntimeError(f"Missing selected label: {label}")

    candidates = candidates.sort_values(
        [
            "recommendation_order",
            "consensus_score",
            "elasticnet_selection_probability",
            "mrmr_rank",
            "univariate_relevance_score",
            "valid_fraction",
        ],
        ascending=[True, False, False, True, False, False],
        na_position="last",
    )

    row = candidates.iloc[0].copy()
    row["selection_group"] = group_name
    row["selection_label"] = label

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    consensus_csv = Path(args.consensus_csv)
    output_csv = Path(args.output_csv)

    if not consensus_csv.exists():
        raise FileNotFoundError(consensus_csv)

    df = pd.read_csv(consensus_csv)

    require_columns(
        df,
        [
            "feature_index",
            "label",
            "recommendation",
            "consensus_score",
            "elasticnet_selection_probability",
            "mrmr_rank",
            "univariate_relevance_score",
            "valid_fraction",
        ],
    )

    df = add_recommendation_order(df)

    rows = []

    for label in CLINICAL_LABELS:
        rows.append(select_one(df, label, "clinical"))

    for label in ALGORITHM_LABELS:
        rows.append(select_one(df, label, "algorithmic"))

    out = pd.DataFrame(rows).reset_index(drop=True)

    duplicated = out[out["feature_index"].duplicated(keep=False)]
    if not duplicated.empty:
        raise RuntimeError("Duplicate feature_index selected:\n" + duplicated.to_string(index=False))

    expected_n = len(CLINICAL_LABELS) + len(ALGORITHM_LABELS)
    if len(out) != expected_n:
        raise RuntimeError(f"Expected {expected_n} features, got {len(out)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("=" * 120)
    print("MIXED SELECTED FEATURE SET")
    print("=" * 120)
    print("Consensus CSV:", consensus_csv)
    print("Output CSV:", output_csv)
    print("Selected features:", len(out))

    print("\nSelection groups:")
    print(out["selection_group"].value_counts().to_string())

    print("\nRecommendation counts:")
    print(out["recommendation"].value_counts().to_string())

    show_cols = [
        "selection_group",
        "selection_label",
        "recommendation",
        "feature_index",
        "source",
        "itemid",
        "label",
        "clinical_category",
        "summary_mode",
        "valid_fraction",
        "hourly_coverage",
        "mrmr_rank",
        "elasticnet_selection_probability",
        "consensus_score",
    ]

    show_cols = [c for c in show_cols if c in out.columns]

    print("\nSelected features:")
    print(out[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
