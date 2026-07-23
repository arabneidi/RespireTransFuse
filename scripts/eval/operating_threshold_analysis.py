#!/usr/bin/env python3
"""Validation-selected operating points with clustered confidence intervals."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import bootstrap_seven_models as bootstrap


ROOT = Path(__file__).resolve().parents[2]

COHORT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "cohorts"
    / "cohort.csv"
)

EHR_NPZ_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ehr"
    / "ehr_final_24h_train_ready"
    / "ehr_24h_final_train_ready_current_split.npz"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "operating_threshold_analysis"
)

METRIC_NAMES = [
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select model-specific thresholds on validation data "
            "using Youden's J and evaluate the fixed thresholds "
            "on the held-out test set."
        )
    )

    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def canonical_id(value) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        prefix = text[:-2]

        if prefix.replace("-", "").isdigit():
            text = prefix

    return text


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    lookup = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def load_reference() -> pd.DataFrame:
    if not COHORT_PATH.exists():
        raise FileNotFoundError(COHORT_PATH)

    if not EHR_NPZ_PATH.exists():
        raise FileNotFoundError(EHR_NPZ_PATH)

    cohort = pd.read_csv(COHORT_PATH)

    sample_column = find_column(
        cohort,
        [
            "sample_id",
            "sample",
            "case_id",
            "case",
        ],
    )

    label_column = find_column(
        cohort,
        [
            "label",
            "labels",
            "target",
            "y_true",
        ],
    )

    split_column = find_column(
        cohort,
        [
            "split",
            "partition",
            "set",
        ],
    )

    patient_column = find_column(
        cohort,
        [
            "subject_id",
            "patient_id",
            "patient",
        ],
    )

    if (
        sample_column is None
        or label_column is None
        or split_column is None
    ):
        raise RuntimeError(
            "The cohort must contain sample, label, and "
            "split columns.\n"
            f"Available columns: {cohort.columns.tolist()}"
        )

    required_columns = [
        sample_column,
        label_column,
        split_column,
    ]

    if "verified_image_path" in cohort.columns:
        required_columns.append(
            "verified_image_path"
        )

    cohort = cohort.dropna(
        subset=required_columns
    ).copy()

    if "image_exists" in cohort.columns:
        cohort = cohort[
            cohort["image_exists"] == True
        ].copy()

    if "image_decode_ok" in cohort.columns:
        cohort = cohort[
            cohort["image_decode_ok"] == True
        ].copy()

    cohort["sample_id"] = (
        cohort[sample_column]
        .map(canonical_id)
    )

    cohort["label"] = pd.to_numeric(
        cohort[label_column],
        errors="raise",
    ).astype(int)

    cohort["split"] = (
        cohort[split_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if patient_column is not None:
        cohort["patient_id"] = (
            cohort[patient_column]
            .map(canonical_id)
        )
    else:
        cohort["patient_id"] = (
            cohort["sample_id"]
        )

    ehr = np.load(
        EHR_NPZ_PATH,
        allow_pickle=True,
    )

    for key in [
        "sample_id",
        "y",
        "split",
    ]:
        if key not in ehr:
            raise RuntimeError(
                f"Missing EHR NPZ key: {key}"
            )

    npz_frame = pd.DataFrame(
        {
            "sample_id": [
                canonical_id(value)
                for value in ehr["sample_id"]
            ],
            "npz_idx": np.arange(
                len(ehr["sample_id"]),
                dtype=np.int64,
            ),
            "npz_label": (
                np.asarray(ehr["y"])
                .astype(int)
            ),
            "npz_split": np.char.lower(
                np.char.strip(
                    np.asarray(
                        ehr["split"]
                    ).astype(str)
                )
            ),
        }
    )

    merged = cohort.merge(
        npz_frame,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    if not np.array_equal(
        merged["label"].to_numpy(dtype=int),
        merged["npz_label"].to_numpy(dtype=int),
    ):
        raise RuntimeError(
            "Cohort and EHR NPZ labels do not match."
        )

    if not np.array_equal(
        merged["split"].astype(str).to_numpy(),
        merged["npz_split"].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "Cohort and EHR NPZ split assignments "
            "do not match."
        )

    reference = (
        merged[
            [
                "sample_id",
                "patient_id",
                "label",
                "split",
                "npz_idx",
            ]
        ]
        .sort_values("npz_idx")
        .reset_index(drop=True)
    )

    if reference["sample_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate sample IDs were found."
        )

    return reference


def load_model_split(
    model_name: str,
    split: str,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    spec = copy.deepcopy(
        bootstrap.MODEL_SPECS[
            model_name
        ]
    )

    run_dir = (
        spec["prediction_path"]
        .parent
    )

    spec["prediction_path"] = (
        run_dir
        / f"{split}_predictions.csv"
    )

    if model_name == "EHR-Only Transformer":
        # The EHR files store raw probability in "prob"
        # and post-hoc calibrated probability separately.
        spec["probability_columns"] = [
            "prob",
            "raw_prob",
            "prob_raw",
            "ehr_prob",
        ]

        spec["logit_columns"] = [
            "logit",
            "ehr_logit",
            "logits",
        ]

    split_reference = (
        reference[
            reference["split"] == split
        ]
        .sort_values("npz_idx")
        .reset_index(drop=True)
    )

    frame, source = (
        bootstrap.load_prediction_file(
            model_name,
            spec,
            split_reference,
        )
    )

    frame["patient_id"] = (
        split_reference[
            "patient_id"
        ].to_numpy()
    )

    return frame, source


def select_youden_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(np.unique(labels)) != 2:
        raise RuntimeError(
            "Threshold selection requires "
            "both outcome classes."
        )

    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(
            labels,
            probabilities,
            drop_intermediate=False,
        )
    )

    valid = (
        np.isfinite(thresholds)
        & (thresholds >= 0.0)
        & (thresholds <= 1.0)
    )

    if not np.any(valid):
        raise RuntimeError(
            "No valid probability threshold was found."
        )

    youden_values = (
        true_positive_rate
        - false_positive_rate
    )

    maximum_youden = float(
        np.max(
            youden_values[valid]
        )
    )

    candidates = np.flatnonzero(
        valid
        & np.isclose(
            youden_values,
            maximum_youden,
            rtol=0.0,
            atol=1e-12,
        )
    )

    # roc_curve returns thresholds in descending order.
    # Selecting the first tied threshold therefore uses
    # the most conservative threshold among equal maxima.
    selected_index = int(
        candidates[0]
    )

    return {
        "threshold": float(
            thresholds[
                selected_index
            ]
        ),
        "youden_j": maximum_youden,
        "validation_sensitivity": float(
            true_positive_rate[
                selected_index
            ]
        ),
        "validation_specificity": float(
            1.0
            - false_positive_rate[
                selected_index
            ]
        ),
    }


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return float("nan")

    return float(
        numerator
        / denominator
    )


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities
        >= float(threshold)
    ).astype(int)

    true_positive = int(
        np.sum(
            (labels == 1)
            & (predictions == 1)
        )
    )

    false_negative = int(
        np.sum(
            (labels == 1)
            & (predictions == 0)
        )
    )

    true_negative = int(
        np.sum(
            (labels == 0)
            & (predictions == 0)
        )
    )

    false_positive = int(
        np.sum(
            (labels == 0)
            & (predictions == 1)
        )
    )

    return {
        "TP": true_positive,
        "FN": false_negative,
        "TN": true_negative,
        "FP": false_positive,
        "Sensitivity": safe_ratio(
            true_positive,
            true_positive
            + false_negative,
        ),
        "Specificity": safe_ratio(
            true_negative,
            true_negative
            + false_positive,
        ),
        "PPV": safe_ratio(
            true_positive,
            true_positive
            + false_positive,
        ),
        "NPV": safe_ratio(
            true_negative,
            true_negative
            + false_negative,
        ),
    }


def percentile_interval(
    values: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    alpha = (
        1.0
        - confidence
    )

    return (
        float(
            np.quantile(
                values,
                alpha / 2.0,
            )
        ),
        float(
            np.quantile(
                values,
                1.0 - alpha / 2.0,
            )
        ),
    )


def format_interval(
    point: float,
    lower: float,
    upper: float,
) -> str:
    if not np.isfinite(point):
        return "NA"

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
    ):
        return (
            f"{point:.5f} (NA)"
        )

    return (
        f"{point:.5f} "
        f"({lower:.5f}--{upper:.5f})"
    )


def main() -> None:
    args = parse_args()

    if args.n_bootstraps < 100:
        raise ValueError(
            "n_bootstraps must be at least 100."
        )

    if not 0.0 < args.confidence < 1.0:
        raise ValueError(
            "confidence must be between 0 and 1."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = load_reference()

    validation_reference = (
        reference[
            reference["split"] == "val"
        ]
        .sort_values("npz_idx")
        .reset_index(drop=True)
    )

    test_reference = (
        reference[
            reference["split"] == "test"
        ]
        .sort_values("npz_idx")
        .reset_index(drop=True)
    )

    model_names = list(
        bootstrap.MODEL_SPECS.keys()
    )

    validation_frames = {}
    test_frames = {}
    probability_sources = {}
    threshold_information = {}

    print("=" * 100)
    print("VALIDATION-SELECTED YOUDEN THRESHOLDS")
    print("=" * 100)
    print(
        "Validation observations:",
        len(validation_reference),
    )
    print(
        "Validation positives:",
        int(
            validation_reference[
                "label"
            ].sum()
        ),
    )
    print(
        "Test observations:",
        len(test_reference),
    )
    print(
        "Test positives:",
        int(
            test_reference[
                "label"
            ].sum()
        ),
    )
    print()

    for model_name in model_names:
        validation_frame, validation_source = (
            load_model_split(
                model_name,
                "val",
                reference,
            )
        )

        test_frame, test_source = (
            load_model_split(
                model_name,
                "test",
                reference,
            )
        )

        if validation_source != test_source:
            raise RuntimeError(
                f"{model_name}: validation source "
                f"{validation_source} differs from "
                f"test source {test_source}."
            )

        validation_frames[
            model_name
        ] = validation_frame

        test_frames[
            model_name
        ] = test_frame

        probability_sources[
            model_name
        ] = validation_source

        threshold_information[
            model_name
        ] = select_youden_threshold(
            validation_frame[
                "label"
            ].to_numpy(dtype=int),
            validation_frame[
                "probability"
            ].to_numpy(dtype=float),
        )

        information = (
            threshold_information[
                model_name
            ]
        )

        print(
            f"{model_name:38s} | "
            f"threshold="
            f"{information['threshold']:.6f} | "
            f"Youden J="
            f"{information['youden_j']:.5f} | "
            f"val sensitivity="
            f"{information['validation_sensitivity']:.5f} | "
            f"val specificity="
            f"{information['validation_specificity']:.5f} | "
            f"source={validation_source}"
        )

    test_labels = (
        test_reference[
            "label"
        ].to_numpy(dtype=int)
    )

    point_metrics = {}

    print()
    print("=" * 100)
    print("FIXED-THRESHOLD TEST OPERATING POINTS")
    print("=" * 100)

    for model_name in model_names:
        if not np.array_equal(
            test_frames[
                model_name
            ]["label"].to_numpy(dtype=int),
            test_labels,
        ):
            raise RuntimeError(
                f"{model_name}: test labels "
                "do not match the common reference."
            )

        threshold = (
            threshold_information[
                model_name
            ]["threshold"]
        )

        metrics = calculate_metrics(
            test_labels,
            test_frames[
                model_name
            ]["probability"].to_numpy(
                dtype=float
            ),
            threshold,
        )

        point_metrics[
            model_name
        ] = metrics

        print(
            f"{model_name:38s} | "
            f"threshold={threshold:.6f} | "
            f"sensitivity="
            f"{metrics['Sensitivity']:.5f} | "
            f"specificity="
            f"{metrics['Specificity']:.5f} | "
            f"PPV={metrics['PPV']:.5f} | "
            f"NPV={metrics['NPV']:.5f}"
        )

    (
        patient_groups,
        patient_strata,
    ) = bootstrap.build_patient_groups(
        test_reference
    )

    probability_matrix = np.column_stack(
        [
            test_frames[
                model_name
            ]["probability"].to_numpy(
                dtype=float
            )
            for model_name in model_names
        ]
    )

    metric_draws = np.full(
        (
            args.n_bootstraps,
            len(model_names),
            len(METRIC_NAMES),
        ),
        np.nan,
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        args.seed
    )

    progress_interval = max(
        1,
        args.n_bootstraps // 10,
    )

    print()
    print("=" * 100)
    print("FIXED-THRESHOLD PATIENT-CLUSTER BOOTSTRAP")
    print("=" * 100)
    print(
        "Patients:",
        len(patient_groups),
    )
    print(
        "All-negative patients:",
        len(
            patient_strata[
                "all_negative"
            ]
        ),
    )
    print(
        "Mixed-label patients:",
        len(
            patient_strata[
                "mixed"
            ]
        ),
    )
    print(
        "All-positive patients:",
        len(
            patient_strata[
                "all_positive"
            ]
        ),
    )
    print(
        "Bootstrap samples:",
        args.n_bootstraps,
    )
    print(
        "Confidence level:",
        args.confidence,
    )
    print(
        "Seed:",
        args.seed,
    )
    print()

    for bootstrap_index in range(
        args.n_bootstraps
    ):
        sampled_indices = (
            bootstrap.create_bootstrap_indices(
                rng,
                patient_groups,
                patient_strata,
            )
        )

        sampled_labels = test_labels[
            sampled_indices
        ]

        for model_index, model_name in enumerate(
            model_names
        ):
            sampled_probabilities = (
                probability_matrix[
                    sampled_indices,
                    model_index,
                ]
            )

            metrics = calculate_metrics(
                sampled_labels,
                sampled_probabilities,
                threshold_information[
                    model_name
                ]["threshold"],
            )

            for metric_index, metric_name in enumerate(
                METRIC_NAMES
            ):
                metric_draws[
                    bootstrap_index,
                    model_index,
                    metric_index,
                ] = float(
                    metrics[
                        metric_name
                    ]
                )

        completed = (
            bootstrap_index
            + 1
        )

        if (
            completed
            % progress_interval
            == 0
            or completed
            == args.n_bootstraps
        ):
            print(
                f"Completed {completed:,}/"
                f"{args.n_bootstraps:,}"
            )

    summary_rows = []

    for model_index, model_name in enumerate(
        model_names
    ):
        information = (
            threshold_information[
                model_name
            ]
        )

        metrics = (
            point_metrics[
                model_name
            ]
        )

        row = {
            "Model": model_name,
            "Validation-selected threshold": (
                information[
                    "threshold"
                ]
            ),
            "Validation Youden J": (
                information[
                    "youden_j"
                ]
            ),
            "Validation sensitivity": (
                information[
                    "validation_sensitivity"
                ]
            ),
            "Validation specificity": (
                information[
                    "validation_specificity"
                ]
            ),
            "Test prevalence": float(
                np.mean(test_labels)
            ),
            "TP": metrics["TP"],
            "FN": metrics["FN"],
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "Probability source": (
                probability_sources[
                    model_name
                ]
            ),
        }

        for metric_index, metric_name in enumerate(
            METRIC_NAMES
        ):
            lower, upper = (
                percentile_interval(
                    metric_draws[
                        :,
                        model_index,
                        metric_index,
                    ],
                    args.confidence,
                )
            )

            row[metric_name] = (
                metrics[
                    metric_name
                ]
            )

            row[
                f"{metric_name} CI lower"
            ] = lower

            row[
                f"{metric_name} CI upper"
            ] = upper

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    bootstrap_frames = []

    for model_index, model_name in enumerate(
        model_names
    ):
        frame = pd.DataFrame(
            {
                "Bootstrap": np.arange(
                    1,
                    args.n_bootstraps + 1,
                ),
                "Model": model_name,
            }
        )

        for metric_index, metric_name in enumerate(
            METRIC_NAMES
        ):
            frame[metric_name] = (
                metric_draws[
                    :,
                    model_index,
                    metric_index,
                ]
            )

        bootstrap_frames.append(
            frame
        )

    bootstrap_results = pd.concat(
        bootstrap_frames,
        ignore_index=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "seven_model_operating_points.csv"
    )

    draws_path = (
        OUTPUT_DIR
        / "operating_point_bootstrap_draws.csv"
    )

    compressed_path = (
        OUTPUT_DIR
        / "operating_point_bootstrap_draws.npz"
    )

    metadata_path = (
        OUTPUT_DIR
        / "operating_point_metadata.json"
    )

    latex_path = (
        OUTPUT_DIR
        / "seven_model_operating_points_table.tex"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    bootstrap_results.to_csv(
        draws_path,
        index=False,
    )

    np.savez_compressed(
        compressed_path,
        model_names=np.asarray(
            model_names,
            dtype=object,
        ),
        metric_names=np.asarray(
            METRIC_NAMES,
            dtype=object,
        ),
        thresholds=np.asarray(
            [
                threshold_information[
                    model_name
                ]["threshold"]
                for model_name in model_names
            ],
            dtype=float,
        ),
        metric_draws=metric_draws,
        seed=np.asarray(
            [args.seed],
            dtype=int,
        ),
        confidence=np.asarray(
            [args.confidence],
            dtype=float,
        ),
    )

    metadata = {
        "threshold_selection": (
            "One model-specific threshold was selected "
            "only on validation data by maximizing "
            "Youden's J. When multiple thresholds tied, "
            "the highest threshold was retained."
        ),
        "test_evaluation": (
            "The validation-selected threshold was fixed "
            "before sensitivity, specificity, PPV, and "
            "NPV were calculated on the test set."
        ),
        "uncertainty": (
            "Patient-cluster bootstrap stratified by "
            "patient label composition. All observations "
            "from each sampled patient were retained "
            "together. Percentile intervals were used."
        ),
        "n_bootstraps": int(
            args.n_bootstraps
        ),
        "confidence": float(
            args.confidence
        ),
        "seed": int(
            args.seed
        ),
        "validation_observations": int(
            len(validation_reference)
        ),
        "validation_positives": int(
            validation_reference[
                "label"
            ].sum()
        ),
        "test_observations": int(
            len(test_reference)
        ),
        "test_positives": int(
            test_reference[
                "label"
            ].sum()
        ),
        "test_negatives": int(
            len(test_reference)
            - test_reference[
                "label"
            ].sum()
        ),
        "test_patients": int(
            len(patient_groups)
        ),
        "probability_sources": (
            probability_sources
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    latex = pd.DataFrame(
        {
            "Model": summary["Model"],
            "Threshold": (
                summary[
                    "Validation-selected threshold"
                ].map(
                    lambda value: (
                        f"{value:.5f}"
                    )
                )
            ),
        }
    )

    for metric_name in METRIC_NAMES:
        latex[
            f"{metric_name} (95\\% CI)"
        ] = [
            format_interval(
                point,
                lower,
                upper,
            )
            for point, lower, upper in zip(
                summary[
                    metric_name
                ],
                summary[
                    f"{metric_name} CI lower"
                ],
                summary[
                    f"{metric_name} CI upper"
                ],
            )
        ]

    latex_path.write_text(
        latex.to_latex(
            index=False,
            escape=False,
            column_format="lccccc",
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("OPERATING-POINT RESULTS WITH CONFIDENCE INTERVALS")
    print("=" * 100)

    result_columns = [
        "Model",
        "Validation-selected threshold",
        "Sensitivity",
        "Sensitivity CI lower",
        "Sensitivity CI upper",
        "Specificity",
        "Specificity CI lower",
        "Specificity CI upper",
        "PPV",
        "PPV CI lower",
        "PPV CI upper",
        "NPV",
        "NPV CI lower",
        "NPV CI upper",
    ]

    display = summary[
        result_columns
    ].copy()

    for column in display.columns:
        if column == "Model":
            continue

        display[column] = (
            display[column]
            .map(
                lambda value: (
                    f"{value:.5f}"
                )
            )
        )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(summary_path)
    print(draws_path)
    print(compressed_path)
    print(metadata_path)
    print(latex_path)


if __name__ == "__main__":
    main()
