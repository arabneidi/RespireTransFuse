#!/usr/bin/env python3
"""Estimate uncertainty in held-out discrimination for all seven models.

The script aligns each model's test predictions with the common patient-level
cohort and performs a stratified patient-cluster bootstrap so repeated samples
from one patient remain together. It reports percentile confidence intervals for
AUROC and AUPRC, paired differences between RespireTransFuse and each comparator,
and exports the point estimates, bootstrap draws, metadata, and formatted tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


BASE = Path(__file__).resolve().parents[2]

COHORT_PATH = (
    BASE
    / "data"
    / "processed"
    / "cohorts"
    / "cohort.csv"
)

OUTPUT_DIR = (
    BASE
    / "outputs"
    / "bootstrap_confidence_intervals"
)

MODEL_SPECS = {
    "Image-Only CNN": {
        "prediction_path": (
            BASE
            / "outputs"
            / "image_only"
            / "test_predictions.csv"
        ),
        "probability_columns": [
            "image_prob",
            "prob",
            "probability",
        ],
        "logit_columns": [
            "image_logit",
            "logit",
        ],
    },
    "Adapted MedFuse Uni-CXR": {
        "prediction_path": (
            BASE
            / "outputs"
            / "medfuse_cxr"
            / "test_predictions.csv"
        ),
        "probability_columns": [
            "prob",
            "probability",
            "medfuse_prob",
            "fusion_prob",
            "y_prob",
            "score",
        ],
        "logit_columns": [
            "logit",
            "logits",
        ],
    },
    "Adapted MedFuse Uni-EHR": {
        "prediction_path": (
            BASE
            / "outputs"
            / "medfuse_ehr"
            / "test_predictions.csv"
        ),
        "probability_columns": [
            "prob",
            "probability",
            "medfuse_prob",
            "fusion_prob",
            "y_prob",
            "score",
        ],
        "logit_columns": [
            "logit",
            "logits",
        ],
    },
    "Early Fusion": {
        "prediction_path": (
            BASE
            / "outputs"
            / "early_fusion"
            / "test_predictions.csv"
        ),
        "probability_columns": [
            "fusion_prob",
            "prob",
            "probability",
        ],
        "logit_columns": [
            "fusion_logit",
            "logit",
        ],
    },
    "EHR-Only Transformer": {
        "prediction_path": (
            BASE
            / "outputs"
            / "ehr_only"
            / "test_predictions.csv"
        ),
        "probability_columns": [
            "calibrated_prob",
            "prob_calibrated",
            "calibrated_probability",
            "probability_calibrated",
            "prob_cal",
            "prob",
            "ehr_prob",
            "probability",
            "raw_prob",
            "prob_raw",
        ],
        "logit_columns": [
            "ehr_logit",
            "logit",
        ],
    },
    "Adapted MedFuse Multimodal LSTM": {
        "prediction_path": (
            BASE
            / "outputs"
            / "medfuse_multimodal_lstm"
            / "test_predictions.csv"
        ),
        "probability_columns": [
            "prob",
            "probability",
            "medfuse_prob",
            "fusion_prob",
            "y_prob",
            "score",
        ],
        "logit_columns": [
            "logit",
            "logits",
        ],
    },
    "RespireTransFuse": {
        "prediction_path": (
            BASE
            / "outputs"
            / "respire_transfuse"
            / "test_predictions.csv"
        ),
        "probability_columns": [
            "fusion_prob",
            "prob",
            "probability",
        ],
        "logit_columns": [
            "fusion_logit",
            "logit",
        ],
    },
}

EXPECTED_METRICS = {
    "Image-Only CNN": {
        "auroc": 0.66625,
        "auprc": 0.25764,
    },
    "Adapted MedFuse Uni-CXR": {
        "auroc": 0.67019,
        "auprc": 0.23330,
    },
    "Adapted MedFuse Uni-EHR": {
        "auroc": 0.71035,
        "auprc": 0.35697,
    },
    "Early Fusion": {
        "auroc": 0.72946,
        "auprc": 0.38592,
    },
    "EHR-Only Transformer": {
        "auroc": 0.73603,
        "auprc": 0.36742,
    },
    "Adapted MedFuse Multimodal LSTM": {
        "auroc": 0.74535,
        "auprc": 0.34305,
    },
    "RespireTransFuse": {
        "auroc": 0.75919,
        "auprc": 0.40738,
    },
}

LABEL_COLUMNS = [
    "label",
    "labels",
    "target",
    "targets",
    "y_true",
    "gt",
]

SAMPLE_COLUMNS = [
    "sample_id",
    "sample",
    "case_id",
    "case",
]

SUBJECT_COLUMNS = [
    "subject_id",
    "patient_id",
    "patient",
]

SPLIT_COLUMNS = [
    "split",
    "partition",
    "set",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate paired bootstrap confidence intervals "
            "for held-out AUROC and AUPRC."
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

    parser.add_argument(
        "--expected_tolerance",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--skip_expected_check",
        action="store_true",
    )

    return parser.parse_args()


def canonical_id(value) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        numeric_part = text[:-2]

        if numeric_part.replace("-", "").isdigit():
            text = numeric_part

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


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = np.clip(
        values,
        -50.0,
        50.0,
    )

    return 1.0 / (
        1.0
        + np.exp(-values)
    )


def load_reference_cohort() -> pd.DataFrame:
    if not COHORT_PATH.exists():
        raise FileNotFoundError(COHORT_PATH)

    cohort = pd.read_csv(COHORT_PATH)

    sample_column = find_column(
        cohort,
        SAMPLE_COLUMNS,
    )

    label_column = find_column(
        cohort,
        LABEL_COLUMNS,
    )

    split_column = find_column(
        cohort,
        SPLIT_COLUMNS,
    )

    subject_column = find_column(
        cohort,
        SUBJECT_COLUMNS,
    )

    if sample_column is None:
        raise RuntimeError(
            "No sample identifier column was found in "
            f"{COHORT_PATH}.\n"
            f"Available columns: {cohort.columns.tolist()}"
        )

    if label_column is None:
        raise RuntimeError(
            "No label column was found in "
            f"{COHORT_PATH}."
        )

    if split_column is not None:
        split_values = (
            cohort[split_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        cohort = cohort[
            split_values == "test"
        ].copy()

    reference = pd.DataFrame(
        {
            "sample_id": (
                cohort[sample_column]
                .map(canonical_id)
            ),
            "label": pd.to_numeric(
                cohort[label_column],
                errors="coerce",
            ),
        }
    )

    if subject_column is not None:
        reference["patient_id"] = (
            cohort[subject_column]
            .map(canonical_id)
            .to_numpy()
        )
    else:
        reference["patient_id"] = (
            reference["sample_id"]
        )

    if reference["label"].isna().any():
        raise RuntimeError(
            "The reference test cohort contains invalid labels."
        )

    reference["label"] = (
        reference["label"]
        .astype(int)
    )

    if reference["sample_id"].duplicated().any():
        duplicates = (
            reference.loc[
                reference["sample_id"].duplicated(
                    keep=False
                ),
                "sample_id",
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            "Duplicate sample identifiers were found in "
            f"the test cohort: {duplicates}"
        )

    reference = (
        reference
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    if len(reference) != 1695:
        raise RuntimeError(
            "Expected 1,695 test observations, found "
            f"{len(reference)}."
        )

    if int(reference["label"].sum()) != 237:
        raise RuntimeError(
            "Expected 237 positive test observations, found "
            f"{int(reference['label'].sum())}."
        )

    return reference


def load_prediction_file(
    model_name: str,
    spec: dict,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    path = spec["prediction_path"]

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name}: missing prediction file:\n{path}"
        )

    frame = pd.read_csv(path)

    label_column = find_column(
        frame,
        LABEL_COLUMNS,
    )

    sample_column = find_column(
        frame,
        SAMPLE_COLUMNS,
    )

    probability_column = find_column(
        frame,
        spec["probability_columns"],
    )

    if label_column is None:
        raise RuntimeError(
            f"{model_name}: no label column was found.\n"
            f"Available columns: {frame.columns.tolist()}"
        )

    labels = pd.to_numeric(
        frame[label_column],
        errors="coerce",
    )

    if labels.isna().any():
        raise RuntimeError(
            f"{model_name}: invalid labels were found."
        )

    if probability_column is not None:
        probabilities = pd.to_numeric(
            frame[probability_column],
            errors="coerce",
        )

        prediction_source = probability_column

    else:
        logit_column = find_column(
            frame,
            spec["logit_columns"],
        )

        if logit_column is not None:
            logits = pd.to_numeric(
                frame[logit_column],
                errors="coerce",
            ).to_numpy(dtype=float)

            probabilities = pd.Series(
                sigmoid(logits)
            )

            prediction_source = (
                f"{logit_column} converted with sigmoid"
            )

        else:
            prediction_column = find_column(
                frame,
                [
                    "prediction",
                    "pred",
                ],
            )

            if prediction_column is None:
                raise RuntimeError(
                    f"{model_name}: no probability or "
                    "logit column was found.\n"
                    f"Available columns: {frame.columns.tolist()}"
                )

            prediction_values = pd.to_numeric(
                frame[prediction_column],
                errors="coerce",
            ).to_numpy(dtype=float)

            if (
                np.nanmin(prediction_values) < 0.0
                or np.nanmax(prediction_values) > 1.0
            ):
                probabilities = pd.Series(
                    sigmoid(prediction_values)
                )

                prediction_source = (
                    f"{prediction_column} converted with sigmoid"
                )
            else:
                probabilities = pd.Series(
                    prediction_values
                )

                prediction_source = prediction_column

    probabilities = pd.to_numeric(
        probabilities,
        errors="coerce",
    )

    if probabilities.isna().any():
        raise RuntimeError(
            f"{model_name}: invalid predictions were found."
        )

    probabilities = probabilities.to_numpy(
        dtype=float
    )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            f"{model_name}: predictions contain non-finite values."
        )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    model_frame = pd.DataFrame(
        {
            "label": labels.astype(int).to_numpy(),
            "probability": probabilities,
        }
    )

    if sample_column is not None:
        model_frame["sample_id"] = (
            frame[sample_column]
            .map(canonical_id)
            .to_numpy()
        )

        if model_frame["sample_id"].duplicated().any():
            raise RuntimeError(
                f"{model_name}: duplicate sample IDs were found."
            )

        model_ids = set(
            model_frame["sample_id"]
        )

        reference_ids = set(
            reference["sample_id"]
        )

        if model_ids != reference_ids:
            missing = sorted(
                reference_ids - model_ids
            )[:10]

            unexpected = sorted(
                model_ids - reference_ids
            )[:10]

            raise RuntimeError(
                f"{model_name}: prediction cohort does not "
                "match the reference test cohort.\n"
                f"Missing IDs: {missing}\n"
                f"Unexpected IDs: {unexpected}"
            )

        model_frame = (
            model_frame
            .set_index("sample_id")
            .loc[reference["sample_id"]]
            .reset_index()
        )

    else:
        if len(model_frame) != len(reference):
            raise RuntimeError(
                f"{model_name}: the prediction file has no "
                "sample_id column and its row count does not "
                "match the reference cohort."
            )

        if not np.array_equal(
            model_frame["label"].to_numpy(),
            reference["label"].to_numpy(),
        ):
            raise RuntimeError(
                f"{model_name}: the prediction file has no "
                "sample_id column and its row order cannot be "
                "verified against the test cohort."
            )

        model_frame.insert(
            0,
            "sample_id",
            reference["sample_id"].to_numpy(),
        )

    if not np.array_equal(
        model_frame["label"].to_numpy(),
        reference["label"].to_numpy(),
    ):
        raise RuntimeError(
            f"{model_name}: labels do not match the "
            "reference cohort after alignment."
        )

    return model_frame, prediction_source


def percentile_interval(
    values: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence

    lower = float(
        np.quantile(
            values,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            values,
            1.0 - alpha / 2.0,
        )
    )

    return lower, upper


def build_patient_groups(
    reference: pd.DataFrame,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    patient_groups = {}

    patient_ids = (
        reference["patient_id"]
        .astype(str)
        .to_numpy()
    )

    labels = (
        reference["label"]
        .to_numpy(dtype=int)
    )

    strata_lists = {
        "all_negative": [],
        "mixed": [],
        "all_positive": [],
    }

    for patient_id in pd.unique(patient_ids):
        patient_id = str(patient_id)

        indices = np.flatnonzero(
            patient_ids == patient_id
        )

        if len(indices) == 0:
            raise RuntimeError(
                f"No observations found for patient {patient_id}."
            )

        patient_groups[patient_id] = indices

        unique_labels = set(
            np.unique(
                labels[indices]
            ).tolist()
        )

        if unique_labels == {0}:
            strata_lists[
                "all_negative"
            ].append(patient_id)

        elif unique_labels == {1}:
            strata_lists[
                "all_positive"
            ].append(patient_id)

        elif unique_labels == {0, 1}:
            strata_lists[
                "mixed"
            ].append(patient_id)

        else:
            raise RuntimeError(
                "Unexpected patient-label composition for "
                f"patient {patient_id}: "
                f"{sorted(unique_labels)}"
            )

    patient_strata = {
        stratum_name: np.asarray(
            patient_list,
            dtype=object,
        )
        for stratum_name, patient_list
        in strata_lists.items()
    }

    total_stratified_patients = sum(
        len(patient_list)
        for patient_list
        in patient_strata.values()
    )

    if total_stratified_patients != len(
        patient_groups
    ):
        raise RuntimeError(
            "Patient-stratum accounting mismatch: "
            f"{total_stratified_patients} stratified versus "
            f"{len(patient_groups)} total."
        )

    return (
        patient_groups,
        patient_strata,
    )


def create_bootstrap_indices(
    rng: np.random.Generator,
    patient_groups: dict[str, np.ndarray],
    patient_strata: dict[str, np.ndarray],
) -> np.ndarray:
    sampled_cluster_indices = []

    for stratum_name in [
        "all_negative",
        "mixed",
        "all_positive",
    ]:
        stratum_patients = (
            patient_strata[
                stratum_name
            ]
        )

        if len(stratum_patients) == 0:
            continue

        sampled_patients = rng.choice(
            stratum_patients,
            size=len(stratum_patients),
            replace=True,
        )

        for patient_id in sampled_patients:
            sampled_cluster_indices.append(
                patient_groups[
                    str(patient_id)
                ]
            )

    if not sampled_cluster_indices:
        raise RuntimeError(
            "The bootstrap sample contains no patient clusters."
        )

    sampled_indices = np.concatenate(
        sampled_cluster_indices
    )

    rng.shuffle(
        sampled_indices
    )

    return sampled_indices


def main() -> None:
    args = parse_args()

    if args.n_bootstraps < 100:
        raise ValueError(
            "n_bootstraps must be at least 100."
        )

    if not 0.0 < args.confidence < 1.0:
        raise ValueError(
            "confidence must be between zero and one."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = load_reference_cohort()

    model_names = list(
        MODEL_SPECS.keys()
    )

    loaded = {}
    prediction_sources = {}

    for model_name in model_names:
        frame, source = load_prediction_file(
            model_name,
            MODEL_SPECS[model_name],
            reference,
        )

        loaded[model_name] = frame
        prediction_sources[model_name] = source

    labels = reference[
        "label"
    ].to_numpy(dtype=int)

    probability_matrix = np.column_stack(
        [
            loaded[model_name][
                "probability"
            ].to_numpy(dtype=float)
            for model_name in model_names
        ]
    )

    point_aurocs = np.asarray(
        [
            roc_auc_score(
                labels,
                probability_matrix[:, model_index],
            )
            for model_index in range(
                len(model_names)
            )
        ],
        dtype=float,
    )

    point_auprcs = np.asarray(
        [
            average_precision_score(
                labels,
                probability_matrix[:, model_index],
            )
            for model_index in range(
                len(model_names)
            )
        ],
        dtype=float,
    )

    print("=" * 100)
    print("POINT-ESTIMATE VERIFICATION")
    print("=" * 100)

    for model_index, model_name in enumerate(
        model_names
    ):
        calculated_auroc = float(
            point_aurocs[model_index]
        )

        calculated_auprc = float(
            point_auprcs[model_index]
        )

        expected = EXPECTED_METRICS[
            model_name
        ]

        auroc_difference = abs(
            calculated_auroc
            - expected["auroc"]
        )

        auprc_difference = abs(
            calculated_auprc
            - expected["auprc"]
        )

        print(
            f"{model_name:38s} | "
            f"AUROC={calculated_auroc:.5f} | "
            f"AUPRC={calculated_auprc:.5f} | "
            f"source={prediction_sources[model_name]}"
        )

        if not args.skip_expected_check:
            if (
                auroc_difference
                > args.expected_tolerance
                or auprc_difference
                > args.expected_tolerance
            ):
                raise RuntimeError(
                    f"{model_name}: calculated metrics do not "
                    "match the thesis point estimates within "
                    f"tolerance {args.expected_tolerance}.\n"
                    f"Expected AUROC={expected['auroc']:.5f}, "
                    f"calculated={calculated_auroc:.5f}\n"
                    f"Expected AUPRC={expected['auprc']:.5f}, "
                    f"calculated={calculated_auprc:.5f}"
                )

    (
        patient_groups,
        patient_strata,
    ) = build_patient_groups(
        reference
    )

    n_models = len(model_names)

    auroc_draws = np.empty(
        (
            args.n_bootstraps,
            n_models,
        ),
        dtype=np.float64,
    )

    auprc_draws = np.empty(
        (
            args.n_bootstraps,
            n_models,
        ),
        dtype=np.float64,
    )

    rng = np.random.default_rng(
        args.seed
    )

    print()
    print("=" * 100)
    print("PAIRED PATIENT-LEVEL BOOTSTRAP")
    print("=" * 100)
    print("Observations:", len(reference))
    print("Positive observations:", int(labels.sum()))
    print("Patients:", len(patient_groups))
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
    print("Bootstrap samples:", args.n_bootstraps)
    print("Confidence level:", args.confidence)
    print("Seed:", args.seed)
    print()

    progress_interval = max(
        1,
        args.n_bootstraps // 10,
    )

    for bootstrap_index in range(
        args.n_bootstraps
    ):
        sampled_indices = (
            create_bootstrap_indices(
                rng,
                patient_groups,
                patient_strata,
            )
        )

        bootstrap_labels = labels[
            sampled_indices
        ]

        bootstrap_probabilities = (
            probability_matrix[
                sampled_indices,
                :
            ]
        )

        for model_index in range(
            n_models
        ):
            model_probabilities = (
                bootstrap_probabilities[
                    :,
                    model_index,
                ]
            )

            auroc_draws[
                bootstrap_index,
                model_index,
            ] = roc_auc_score(
                bootstrap_labels,
                model_probabilities,
            )

            auprc_draws[
                bootstrap_index,
                model_index,
            ] = average_precision_score(
                bootstrap_labels,
                model_probabilities,
            )

        completed = bootstrap_index + 1

        if (
            completed % progress_interval == 0
            or completed == args.n_bootstraps
        ):
            print(
                f"Completed {completed:,}/"
                f"{args.n_bootstraps:,}"
            )

    summary_rows = []

    for model_index, model_name in enumerate(
        model_names
    ):
        auroc_lower, auroc_upper = (
            percentile_interval(
                auroc_draws[
                    :,
                    model_index,
                ],
                args.confidence,
            )
        )

        auprc_lower, auprc_upper = (
            percentile_interval(
                auprc_draws[
                    :,
                    model_index,
                ],
                args.confidence,
            )
        )

        summary_rows.append(
            {
                "Model": model_name,
                "AUROC": float(
                    point_aurocs[
                        model_index
                    ]
                ),
                "AUROC CI lower": (
                    auroc_lower
                ),
                "AUROC CI upper": (
                    auroc_upper
                ),
                "AUPRC": float(
                    point_auprcs[
                        model_index
                    ]
                ),
                "AUPRC CI lower": (
                    auprc_lower
                ),
                "AUPRC CI upper": (
                    auprc_upper
                ),
                "Prediction source": (
                    prediction_sources[
                        model_name
                    ]
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    respire_index = model_names.index(
        "RespireTransFuse"
    )

    paired_rows = []
    paired_draw_rows = []

    for comparator_index, comparator in enumerate(
        model_names
    ):
        if comparator == "RespireTransFuse":
            continue

        auroc_differences = (
            auroc_draws[
                :,
                respire_index,
            ]
            - auroc_draws[
                :,
                comparator_index,
            ]
        )

        auprc_differences = (
            auprc_draws[
                :,
                respire_index,
            ]
            - auprc_draws[
                :,
                comparator_index,
            ]
        )

        auroc_lower, auroc_upper = (
            percentile_interval(
                auroc_differences,
                args.confidence,
            )
        )

        auprc_lower, auprc_upper = (
            percentile_interval(
                auprc_differences,
                args.confidence,
            )
        )

        paired_rows.append(
            {
                "Comparator": comparator,
                "Respire AUROC difference": float(
                    point_aurocs[
                        respire_index
                    ]
                    - point_aurocs[
                        comparator_index
                    ]
                ),
                "AUROC difference CI lower": (
                    auroc_lower
                ),
                "AUROC difference CI upper": (
                    auroc_upper
                ),
                "AUROC CI excludes zero": bool(
                    auroc_lower > 0.0
                    or auroc_upper < 0.0
                ),
                "AUROC bootstrap support above zero": float(
                    np.mean(
                        auroc_differences > 0.0
                    )
                ),
                "Respire AUPRC difference": float(
                    point_auprcs[
                        respire_index
                    ]
                    - point_auprcs[
                        comparator_index
                    ]
                ),
                "AUPRC difference CI lower": (
                    auprc_lower
                ),
                "AUPRC difference CI upper": (
                    auprc_upper
                ),
                "AUPRC CI excludes zero": bool(
                    auprc_lower > 0.0
                    or auprc_upper < 0.0
                ),
                "AUPRC bootstrap support above zero": float(
                    np.mean(
                        auprc_differences > 0.0
                    )
                ),
            }
        )

        paired_draw_rows.append(
            pd.DataFrame(
                {
                    "Bootstrap": np.arange(
                        1,
                        args.n_bootstraps + 1,
                    ),
                    "Comparator": comparator,
                    "AUROC difference": (
                        auroc_differences
                    ),
                    "AUPRC difference": (
                        auprc_differences
                    ),
                }
            )
        )

    paired_summary = pd.DataFrame(
        paired_rows
    )

    metric_draw_rows = []

    for model_index, model_name in enumerate(
        model_names
    ):
        metric_draw_rows.append(
            pd.DataFrame(
                {
                    "Bootstrap": np.arange(
                        1,
                        args.n_bootstraps + 1,
                    ),
                    "Model": model_name,
                    "AUROC": auroc_draws[
                        :,
                        model_index,
                    ],
                    "AUPRC": auprc_draws[
                        :,
                        model_index,
                    ],
                }
            )
        )

    metric_draws = pd.concat(
        metric_draw_rows,
        ignore_index=True,
    )

    paired_draws = pd.concat(
        paired_draw_rows,
        ignore_index=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "seven_model_metric_confidence_intervals.csv"
    )

    paired_summary_path = (
        OUTPUT_DIR
        / "respiretransfuse_paired_differences.csv"
    )

    metric_draws_path = (
        OUTPUT_DIR
        / "bootstrap_metric_draws.csv"
    )

    paired_draws_path = (
        OUTPUT_DIR
        / "bootstrap_paired_difference_draws.csv"
    )

    compressed_draws_path = (
        OUTPUT_DIR
        / "bootstrap_draws.npz"
    )

    excel_path = (
        OUTPUT_DIR
        / "seven_model_bootstrap_results.xlsx"
    )

    metadata_path = (
        OUTPUT_DIR
        / "bootstrap_metadata.json"
    )

    latex_summary_path = (
        OUTPUT_DIR
        / "seven_model_confidence_intervals_table.tex"
    )

    latex_difference_path = (
        OUTPUT_DIR
        / "respiretransfuse_paired_differences_table.tex"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    paired_summary.to_csv(
        paired_summary_path,
        index=False,
    )

    metric_draws.to_csv(
        metric_draws_path,
        index=False,
    )

    paired_draws.to_csv(
        paired_draws_path,
        index=False,
    )

    np.savez_compressed(
        compressed_draws_path,
        model_names=np.asarray(
            model_names,
            dtype=object,
        ),
        auroc_draws=auroc_draws,
        auprc_draws=auprc_draws,
        seed=np.asarray(
            [args.seed],
            dtype=int,
        ),
        confidence=np.asarray(
            [args.confidence],
            dtype=float,
        ),
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:
        summary.to_excel(
            writer,
            sheet_name="Metric Intervals",
            index=False,
        )

        paired_summary.to_excel(
            writer,
            sheet_name="Paired Differences",
            index=False,
        )

    metadata = {
        "method": (
            "Paired patient-cluster bootstrap stratified by "
            "patient label composition, with percentile "
            "confidence intervals"
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
        "test_observations": int(
            len(reference)
        ),
        "test_positives": int(
            labels.sum()
        ),
        "test_negatives": int(
            len(labels) - labels.sum()
        ),
        "patients": int(
            len(patient_groups)
        ),
        "all_negative_patients": int(
            len(
                patient_strata[
                    "all_negative"
                ]
            )
        ),
        "mixed_label_patients": int(
            len(
                patient_strata[
                    "mixed"
                ]
            )
        ),
        "all_positive_patients": int(
            len(
                patient_strata[
                    "all_positive"
                ]
            )
        ),
        "paired_models": model_names,
        "prediction_sources": (
            prediction_sources
        ),
        "cohort_path": str(
            COHORT_PATH
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    latex_summary = summary.copy()

    latex_summary[
        "AUROC (95\\% CI)"
    ] = latex_summary.apply(
        lambda row: (
            f"{row['AUROC']:.5f} "
            f"({row['AUROC CI lower']:.5f}--"
            f"{row['AUROC CI upper']:.5f})"
        ),
        axis=1,
    )

    latex_summary[
        "AUPRC (95\\% CI)"
    ] = latex_summary.apply(
        lambda row: (
            f"{row['AUPRC']:.5f} "
            f"({row['AUPRC CI lower']:.5f}--"
            f"{row['AUPRC CI upper']:.5f})"
        ),
        axis=1,
    )

    latex_summary = latex_summary[
        [
            "Model",
            "AUROC (95\\% CI)",
            "AUPRC (95\\% CI)",
        ]
    ]

    latex_summary_path.write_text(
        latex_summary.to_latex(
            index=False,
            escape=False,
            column_format="lcc",
        ),
        encoding="utf-8",
    )

    latex_differences = paired_summary.copy()

    latex_differences[
        "$\\Delta$AUROC (95\\% CI)"
    ] = latex_differences.apply(
        lambda row: (
            f"{row['Respire AUROC difference']:.5f} "
            f"({row['AUROC difference CI lower']:.5f}--"
            f"{row['AUROC difference CI upper']:.5f})"
        ),
        axis=1,
    )

    latex_differences[
        "$\\Delta$AUPRC (95\\% CI)"
    ] = latex_differences.apply(
        lambda row: (
            f"{row['Respire AUPRC difference']:.5f} "
            f"({row['AUPRC difference CI lower']:.5f}--"
            f"{row['AUPRC difference CI upper']:.5f})"
        ),
        axis=1,
    )

    latex_differences = latex_differences[
        [
            "Comparator",
            "$\\Delta$AUROC (95\\% CI)",
            "$\\Delta$AUPRC (95\\% CI)",
        ]
    ]

    latex_difference_path.write_text(
        latex_differences.to_latex(
            index=False,
            escape=False,
            column_format="lcc",
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 100)

    display_summary = summary.copy()

    for column in [
        "AUROC",
        "AUROC CI lower",
        "AUROC CI upper",
        "AUPRC",
        "AUPRC CI lower",
        "AUPRC CI upper",
    ]:
        display_summary[column] = (
            display_summary[column]
            .map(
                lambda value: f"{value:.5f}"
            )
        )

    print(
        display_summary[
            [
                "Model",
                "AUROC",
                "AUROC CI lower",
                "AUROC CI upper",
                "AUPRC",
                "AUPRC CI lower",
                "AUPRC CI upper",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("PAIRED DIFFERENCES: RESPIRETRANSFUSE MINUS COMPARATOR")
    print("=" * 100)

    print(
        paired_summary.to_string(
            index=False,
            formatters={
                "Respire AUROC difference": (
                    lambda value: f"{value:.5f}"
                ),
                "AUROC difference CI lower": (
                    lambda value: f"{value:.5f}"
                ),
                "AUROC difference CI upper": (
                    lambda value: f"{value:.5f}"
                ),
                "AUROC bootstrap support above zero": (
                    lambda value: f"{value:.4f}"
                ),
                "Respire AUPRC difference": (
                    lambda value: f"{value:.5f}"
                ),
                "AUPRC difference CI lower": (
                    lambda value: f"{value:.5f}"
                ),
                "AUPRC difference CI upper": (
                    lambda value: f"{value:.5f}"
                ),
                "AUPRC bootstrap support above zero": (
                    lambda value: f"{value:.4f}"
                ),
            },
        )
    )

    print()
    print("Saved:")
    print(summary_path)
    print(paired_summary_path)
    print(metric_draws_path)
    print(paired_draws_path)
    print(compressed_draws_path)
    print(excel_path)
    print(metadata_path)
    print(latex_summary_path)
    print(latex_difference_path)


if __name__ == "__main__":
    main()
