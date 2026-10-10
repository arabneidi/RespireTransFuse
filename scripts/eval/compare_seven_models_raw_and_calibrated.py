#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(
        0,
        str(SRC),
    )

from respire_transfuse.training.calibration import (
    fit_temperature_bias,
    calibrated_probabilities,
)


COHORT_PATH = (
    ROOT
    / "data/processed/cohorts/cohort.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs/seven_model_raw_and_calibrated"
)

CONFIG_SNAPSHOT_DIR = (
    OUTPUT_DIR
    / "config_snapshots"
)

PREDICTION_EXPORT_DIR = (
    OUTPUT_DIR
    / "heldout_predictions"
)



MODEL_SPECS = {
    "Image-Only CNN": {
        "run_dir": ROOT / "outputs/image_only",
        "raw_probability_columns": [
            "image_prob",
            "prob",
            "probability",
        ],
        "raw_logit_columns": [
            "image_logit",
            "logit",
        ],
        "config_file": "config_used.yaml",
    },

    "EHR-Only Transformer": {
        "run_dir": ROOT / "outputs/ehr_only",
        "raw_probability_columns": [
            "prob",
            "ehr_prob",
            "probability",
            "raw_prob",
            "prob_raw",
        ],
        "raw_logit_columns": [
            "logit",
            "ehr_logit",
        ],
        "config_file": "config_used.yaml",
    },

    "Early Fusion": {
        "run_dir": ROOT / "outputs/early_fusion",
        "raw_probability_columns": [
            "fusion_prob",
            "prob",
            "probability",
        ],
        "raw_logit_columns": [
            "fusion_logit",
            "logit",
        ],
        "config_file": "config_used.yaml",
    },

    "Adapted MedFuse Uni-CXR": {
        "run_dir": ROOT / "outputs/medfuse_uni_cxr_linear_probe_513",
        "raw_probability_columns": [
            "prob",
            "probability",
        ],
        "raw_logit_columns": [
            "logit",
            "logits",
        ],
        "config_file": "args_used.json",
    },

    "Adapted MedFuse Uni-EHR": {
        "run_dir": ROOT / "outputs/medfuse_ehr",
        "raw_probability_columns": [
            "prob",
            "probability",
        ],
        "raw_logit_columns": [
            "logit",
            "logits",
        ],
        "config_file": "args_used.json",
    },

    "Adapted MedFuse Multimodal LSTM": {
        "run_dir": (
            ROOT
            / "outputs/medfuse_multimodal_lstm_26f_lr9e7_wd1e3_bs32"
        ),
        "raw_probability_columns": [
            "prob",
            "probability",
        ],
        "raw_logit_columns": [
            "logit",
            "logits",
        ],
        "config_file": "args_used.json",
    },

    "RespireTransFuse": {
        "run_dir": ROOT / "outputs/respire_transfuse",
        "raw_probability_columns": [
            "fusion_prob",
            "prob",
            "probability",
        ],
        "raw_logit_columns": [
            "fusion_logit",
            "logit",
        ],
        "config_file": "config_used.yaml",
    },
}


LABEL_COLUMNS = [
    "label",
    "labels",
    "y_true",
    "target",
    "targets",
]

SAMPLE_COLUMNS = [
    "sample_id",
    "sample",
    "case_id",
]



def canonical_id(value):
    text = str(value).strip()

    if text.endswith(".0"):
        candidate = text[:-2]

        if candidate.replace("-", "").isdigit():
            text = candidate

    return text


def safe_name(text):
    return (
        str(text)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def sha256_file(path):
    path = Path(path)

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def find_column(frame, candidates):
    lookup = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        key = str(candidate).strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def sigmoid(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = np.clip(
        values,
        -50.0,
        50.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-values)
        )
    )


def probability_to_logit(probabilities):
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    return np.log(
        probabilities
        / (
            1.0
            - probabilities
        )
    )


def ece_equal_width(
    y_true,
    probabilities,
    n_bins=10,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        int(n_bins) + 1,
    )

    bin_ids = np.digitize(
        probabilities,
        edges[1:-1],
        right=False,
    )

    ece = 0.0

    for bin_id in range(
        int(n_bins)
    ):
        mask = (
            bin_ids
            == bin_id
        )

        if not np.any(mask):
            continue

        weight = float(
            mask.mean()
        )

        mean_prob = float(
            probabilities[
                mask
            ].mean()
        )

        observed = float(
            y_true[
                mask
            ].mean()
        )

        ece += (
            weight
            * abs(
                observed
                - mean_prob
            )
        )

    return float(ece)


def calculate_metrics(
    y_true,
    probabilities,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    return {
        "n": int(
            len(y_true)
        ),

        "positives": int(
            y_true.sum()
        ),

        "prevalence": float(
            y_true.mean()
        ),

        "auroc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "auprc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "brier": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),

        "ece_10": float(
            ece_equal_width(
                y_true,
                probabilities,
                n_bins=10,
            )
        ),
    }



def load_reference_split(split_name):
    if not COHORT_PATH.exists():
        raise FileNotFoundError(
            COHORT_PATH
        )

    cohort = pd.read_csv(
        COHORT_PATH
    )

    required = [
        "sample_id",
        "subject_id",
        "label",
        "split",
    ]

    missing = [
        column
        for column in required
        if column not in cohort.columns
    ]

    if missing:
        raise RuntimeError(
            f"Cohort missing columns: {missing}"
        )

    frame = cohort[
        cohort["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        == str(split_name).lower()
    ].copy()

    frame["sample_id"] = (
        frame["sample_id"]
        .map(canonical_id)
    )

    frame["label"] = pd.to_numeric(
        frame["label"],
        errors="raise",
    ).astype(int)

    frame["subject_id"] = (
        frame["subject_id"]
        .map(canonical_id)
    )

    if frame["sample_id"].duplicated().any():
        raise RuntimeError(
            f"{split_name}: duplicate sample_id "
            "in reference cohort."
        )

    frame = (
        frame[
            [
                "sample_id",
                "subject_id",
                "label",
            ]
        ]
        .sort_values(
            "sample_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    if frame.empty:
        raise RuntimeError(
            f"No {split_name} rows found."
        )

    return frame



def load_raw_predictions(
    model_name,
    spec,
    split_name,
    reference,
):
    prediction_path = (
        Path(
            spec["run_dir"]
        )
        / f"{split_name}_predictions.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"{model_name}: missing "
            f"{prediction_path}"
        )

    frame = pd.read_csv(
        prediction_path
    )

    label_col = find_column(
        frame,
        LABEL_COLUMNS,
    )

    sample_col = find_column(
        frame,
        SAMPLE_COLUMNS,
    )

    if label_col is None:
        raise RuntimeError(
            f"{model_name}: no label column "
            f"in {prediction_path}"
        )

    raw_prob_col = find_column(
        frame,
        spec[
            "raw_probability_columns"
        ],
    )

    raw_logit_col = find_column(
        frame,
        spec[
            "raw_logit_columns"
        ],
    )

    if raw_prob_col is not None:
        raw_prob = pd.to_numeric(
            frame[raw_prob_col],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        probability_source = (
            raw_prob_col
        )

    elif raw_logit_col is not None:
        raw_logit_temp = pd.to_numeric(
            frame[raw_logit_col],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        raw_prob = sigmoid(
            raw_logit_temp
        )

        probability_source = (
            f"{raw_logit_col}->sigmoid"
        )

    else:
        raise RuntimeError(
            f"{model_name}: no RAW probability "
            f"or logit found in {prediction_path}. "
            f"Available columns: "
            f"{frame.columns.tolist()}"
        )

    raw_prob = np.clip(
        raw_prob,
        1e-7,
        1.0 - 1e-7,
    )

    if raw_logit_col is not None:
        raw_logit = pd.to_numeric(
            frame[raw_logit_col],
            errors="raise",
        ).to_numpy(
            dtype=np.float64
        )

        logit_source = raw_logit_col

    else:
        raw_logit = probability_to_logit(
            raw_prob
        )

        logit_source = (
            f"logit({probability_source})"
        )

    labels = pd.to_numeric(
        frame[label_col],
        errors="raise",
    ).astype(int).to_numpy()

    model_frame = pd.DataFrame(
        {
            "label": labels,
            "raw_logit": raw_logit,
            "raw_prob": raw_prob,
        }
    )

    if sample_col is not None:
        model_frame[
            "sample_id"
        ] = (
            frame[sample_col]
            .map(canonical_id)
            .to_numpy()
        )

        if model_frame[
            "sample_id"
        ].duplicated().any():
            raise RuntimeError(
                f"{model_name} {split_name}: "
                "duplicate prediction sample IDs."
            )

        reference_ids = set(
            reference["sample_id"]
        )

        model_ids = set(
            model_frame["sample_id"]
        )

        if model_ids != reference_ids:
            missing = sorted(
                reference_ids
                - model_ids
            )[:10]

            extra = sorted(
                model_ids
                - reference_ids
            )[:10]

            raise RuntimeError(
                f"{model_name} {split_name}: "
                "prediction IDs do not match "
                "reference cohort.\n"
                f"Missing: {missing}\n"
                f"Extra: {extra}"
            )

        model_frame = (
            model_frame
            .set_index(
                "sample_id"
            )
            .loc[
                reference[
                    "sample_id"
                ]
            ]
            .reset_index()
        )

    else:
        if len(
            model_frame
        ) != len(
            reference
        ):
            raise RuntimeError(
                f"{model_name} {split_name}: "
                "no sample_id and wrong row count."
            )

        if not np.array_equal(
            model_frame[
                "label"
            ].to_numpy(),
            reference[
                "label"
            ].to_numpy(),
        ):
            raise RuntimeError(
                f"{model_name} {split_name}: "
                "no sample_id and row order "
                "cannot be verified."
            )

        model_frame.insert(
            0,
            "sample_id",
            reference[
                "sample_id"
            ].to_numpy(),
        )

    if not np.array_equal(
        model_frame[
            "label"
        ].to_numpy(
            dtype=int
        ),
        reference[
            "label"
        ].to_numpy(
            dtype=int
        ),
    ):
        raise RuntimeError(
            f"{model_name} {split_name}: "
            "labels differ after alignment."
        )

    return (
        model_frame,
        prediction_path,
        probability_source,
        logit_source,
    )



def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFIG_SNAPSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_reference = (
        load_reference_split(
            "val"
        )
    )

    test_reference = (
        load_reference_split(
            "test"
        )
    )

    print("=" * 100)
    print("CURRENT REFERENCE COHORT")
    print("=" * 100)

    print(
        "validation:",
        len(val_reference),
        "positives:",
        int(
            val_reference[
                "label"
            ].sum()
        ),
    )

    print(
        "test:",
        len(test_reference),
        "positives:",
        int(
            test_reference[
                "label"
            ].sum()
        ),
    )

    metric_rows = []
    calibration_rows = []
    manifest_models = {}

    calibration_device = (
        torch.device("cpu")
    )

    for (
        model_name,
        spec,
    ) in MODEL_SPECS.items():

        print("\n" + "-" * 100)
        print(model_name)
        print("-" * 100)

        (
            val_frame,
            val_prediction_path,
            val_prob_source,
            val_logit_source,
        ) = load_raw_predictions(
            model_name,
            spec,
            "val",
            val_reference,
        )

        (
            test_frame,
            test_prediction_path,
            test_prob_source,
            test_logit_source,
        ) = load_raw_predictions(
            model_name,
            spec,
            "test",
            test_reference,
        )


        temperature, bias = (
            fit_temperature_bias(
                val_frame[
                    "raw_logit"
                ].to_numpy(
                    dtype=np.float64
                ),
                val_frame[
                    "label"
                ].to_numpy(
                    dtype=int
                ),
                calibration_device,
                max_iter=200,
            )
        )

        val_calibrated_prob = (
            calibrated_probabilities(
                val_frame[
                    "raw_logit"
                ].to_numpy(
                    dtype=np.float64
                ),
                temperature,
                bias,
            )
        )

        test_calibrated_prob = (
            calibrated_probabilities(
                test_frame[
                    "raw_logit"
                ].to_numpy(
                    dtype=np.float64
                ),
                temperature,
                bias,
            )
        )

        val_frame[
            "calibrated_prob"
        ] = val_calibrated_prob

        test_frame[
            "calibrated_prob"
        ] = test_calibrated_prob

        val_frame[
            "split"
        ] = "val"

        test_frame[
            "split"
        ] = "test"


        for split_name, frame in [
            ("val", val_frame),
            ("test", test_frame),
        ]:
            for calibration_state, prob_col in [
                (
                    "raw",
                    "raw_prob",
                ),
                (
                    "calibrated",
                    "calibrated_prob",
                ),
            ]:
                metrics = calculate_metrics(
                    frame[
                        "label"
                    ].to_numpy(
                        dtype=int
                    ),
                    frame[
                        prob_col
                    ].to_numpy(
                        dtype=float
                    ),
                )

                metric_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "prediction_state": (
                            calibration_state
                        ),
                        **metrics,
                    }
                )

        calibration_rows.append(
            {
                "model": model_name,
                "temperature": float(
                    temperature
                ),
                "bias": float(
                    bias
                ),
                "fit_split": "validation",
                "max_iter": 200,
                "method": (
                    "temperature_plus_bias"
                ),
            }
        )


        heldout = pd.concat(
            [
                val_frame,
                test_frame,
            ],
            axis=0,
            ignore_index=True,
        )

        heldout_path = (
            PREDICTION_EXPORT_DIR
            / (
                safe_name(
                    model_name
                )
                + "_heldout_predictions.csv"
            )
        )

        heldout.to_csv(
            heldout_path,
            index=False,
        )


        config_path = (
            Path(
                spec[
                    "run_dir"
                ]
            )
            / spec[
                "config_file"
            ]
        )

        config_snapshot_path = None

        if config_path.exists():
            suffix = (
                config_path.suffix
            )

            config_snapshot_path = (
                CONFIG_SNAPSHOT_DIR
                / (
                    safe_name(
                        model_name
                    )
                    + suffix
                )
            )

            shutil.copy2(
                config_path,
                config_snapshot_path,
            )

        manifest_models[
            model_name
        ] = {
            "run_dir": str(
                spec[
                    "run_dir"
                ]
            ),

            "val_prediction_file": str(
                val_prediction_path
            ),

            "val_prediction_sha256": (
                sha256_file(
                    val_prediction_path
                )
            ),

            "test_prediction_file": str(
                test_prediction_path
            ),

            "test_prediction_sha256": (
                sha256_file(
                    test_prediction_path
                )
            ),

            "raw_validation_probability_source": (
                val_prob_source
            ),

            "raw_validation_logit_source": (
                val_logit_source
            ),

            "raw_test_probability_source": (
                test_prob_source
            ),

            "raw_test_logit_source": (
                test_logit_source
            ),

            "config_file": (
                str(config_path)
                if config_path.exists()
                else None
            ),

            "config_sha256": (
                sha256_file(
                    config_path
                )
                if config_path.exists()
                else None
            ),

            "config_snapshot": (
                str(
                    config_snapshot_path
                )
                if config_snapshot_path
                is not None
                else None
            ),

            "heldout_export": str(
                heldout_path
            ),

            "heldout_export_sha256": (
                sha256_file(
                    heldout_path
                )
            ),

            "temperature": float(
                temperature
            ),

            "bias": float(
                bias
            ),
        }

        test_raw = calculate_metrics(
            test_frame[
                "label"
            ],
            test_frame[
                "raw_prob"
            ],
        )

        test_cal = calculate_metrics(
            test_frame[
                "label"
            ],
            test_frame[
                "calibrated_prob"
            ],
        )

        print(
            "RAW        "
            f"AUROC={test_raw['auroc']:.5f} | "
            f"AUPRC={test_raw['auprc']:.5f} | "
            f"Brier={test_raw['brier']:.5f} | "
            f"ECE={test_raw['ece_10']:.5f}"
        )

        print(
            "CALIBRATED "
            f"AUROC={test_cal['auroc']:.5f} | "
            f"AUPRC={test_cal['auprc']:.5f} | "
            f"Brier={test_cal['brier']:.5f} | "
            f"ECE={test_cal['ece_10']:.5f}"
        )

        print(
            "temperature:",
            round(
                float(temperature),
                6,
            ),
            "| bias:",
            round(
                float(bias),
                6,
            ),
        )


    metrics_df = pd.DataFrame(
        metric_rows
    )

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    metrics_path = (
        OUTPUT_DIR
        / "seven_model_raw_and_calibrated_metrics.csv"
    )

    calibration_path = (
        OUTPUT_DIR
        / "seven_model_calibration_parameters.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    calibration_df.to_csv(
        calibration_path,
        index=False,
    )


    manifest = {
        "method": {
            "primary_discrimination": (
                "raw held-out probabilities"
            ),

            "calibration": (
                "temperature+bias fitted independently "
                "for each model on validation predictions "
                "only, then applied unchanged to test"
            ),

            "calibration_max_iter": 200,

            "all_models_same_calibration_procedure": True,

            "test_set_used_for_calibrator_fit": False,

            "calibrated_ehr_preferred_over_raw": False,
        },

        "cohort": {
            "path": str(
                COHORT_PATH
            ),

            "sha256": sha256_file(
                COHORT_PATH
            ),

            "validation_n": int(
                len(
                    val_reference
                )
            ),

            "validation_positives": int(
                val_reference[
                    "label"
                ].sum()
            ),

            "test_n": int(
                len(
                    test_reference
                )
            ),

            "test_positives": int(
                test_reference[
                    "label"
                ].sum()
            ),
        },

        "models": (
            manifest_models
        ),

        "outputs": {
            "metrics_csv": str(
                metrics_path
            ),

            "calibration_parameters_csv": str(
                calibration_path
            ),

            "prediction_export_directory": str(
                PREDICTION_EXPORT_DIR
            ),

            "config_snapshot_directory": str(
                CONFIG_SNAPSHOT_DIR
            ),
        },
    }

    manifest_path = (
        OUTPUT_DIR
        / "reproducibility_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print("\n" + "=" * 100)
    print("FAIR SEVEN-MODEL CALIBRATION COMPARISON COMPLETE")
    print("=" * 100)

    print(
        "Metrics:",
        metrics_path,
    )

    print(
        "Calibration parameters:",
        calibration_path,
    )

    print(
        "Held-out predictions:",
        PREDICTION_EXPORT_DIR,
    )

    print(
        "Config snapshots:",
        CONFIG_SNAPSHOT_DIR,
    )

    print(
        "Manifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()
