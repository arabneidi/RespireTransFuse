"""Compare validation histories and calibration curves across seven models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import brier_score_loss


BASE = Path("/content/drive/MyDrive/respire-transfuse")
OUTPUTS = BASE / "outputs"

MODEL_SPECS = {
    "EHR-Only": {
        "run_dir": OUTPUTS / "ehr_only",
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
    "Image-Only": {
        "run_dir": OUTPUTS / "image_only",
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
    "Early Fusion": {
        "run_dir": OUTPUTS / "early_fusion",
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
    "MedFuse Uni-EHR": {
        "run_dir": OUTPUTS / "medfuse_ehr",
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
    "MedFuse Uni-CXR": {
        "run_dir": OUTPUTS / "medfuse_cxr_only_resnet34",
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
    "MedFuse Multimodal LSTM": {
        "run_dir": OUTPUTS / "medfuse_multimodal_lstm",
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
        "run_dir": OUTPUTS / "respire_transfuse",
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

MODEL_COLORS = {
    model_name: f"C{index}"
    for index, model_name in enumerate(MODEL_SPECS)
}

SAVE_PATH = (
    OUTPUTS
    / "seven_models_validation_and_calibration_curves.png"
)

START_EPOCH = 1
END_EPOCH = 20

LOSS_YLIM = (0.0, 1.0)
AUROC_YLIM = (0.0, 1.0)
AUPRC_YLIM = (0.0, 1.0)

N_BINS = 10

FIGSIZE = (18, 12)
LINE_WIDTH = 2.0
MARKER_SIZE = 4
GRID_ALPHA = 0.3

MAIN_TITLE_FONT_SIZE = 20
PLOT_TEXT_FONT_SIZE = 16
LEGEND_FONT_SIZE = 12
TICK_FONT_SIZE = 14


def set_tick_style(axis):
    axis.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
    )

    for label in axis.get_xticklabels():
        label.set_fontweight("bold")

    for label in axis.get_yticklabels():
        label.set_fontweight("bold")


def resolve_history_dir(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if (path / "history.csv").exists():
        return path

    history_files = sorted(
        path.rglob("history.csv"),
        key=lambda file: file.stat().st_mtime,
        reverse=True,
    )

    if not history_files:
        raise FileNotFoundError(
            f"No history.csv found under: {path}"
        )

    return history_files[0].parent


def get_numeric_column(frame, candidates):
    column_lookup = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        candidate_key = candidate.strip().lower()

        if candidate_key in column_lookup:
            original_column = column_lookup[candidate_key]

            return pd.to_numeric(
                frame[original_column],
                errors="coerce",
            )

    return None


def find_column(frame, candidates):
    column_lookup = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        candidate_key = candidate.strip().lower()

        if candidate_key in column_lookup:
            return column_lookup[candidate_key]

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

    return 1.0 / (
        1.0 + np.exp(-values)
    )


def load_validation_curves():
    curves = {}

    for model_name, spec in MODEL_SPECS.items():
        run_dir = resolve_history_dir(
            spec["run_dir"]
        )

        history_path = (
            run_dir
            / "history.csv"
        )

        history = pd.read_csv(
            history_path
        )

        epoch = get_numeric_column(
            history,
            [
                "epoch",
            ],
        )

        if epoch is None:
            epoch = pd.Series(
                np.arange(
                    1,
                    len(history) + 1,
                ),
                index=history.index,
            )

        val_loss = get_numeric_column(
            history,
            [
                "val_loss",
                "validation_loss",
                "loss_val",
                "loss val",
            ],
        )

        val_auroc = get_numeric_column(
            history,
            [
                "val_auroc",
                "validation_auroc",
                "val_auc",
                "auroc_val",
                "auroc val",
            ],
        )

        val_auprc = get_numeric_column(
            history,
            [
                "val_auprc",
                "validation_auprc",
                "val_ap",
                "auprc_val",
                "auprc val",
            ],
        )

        metrics = {
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
        }

        missing_metrics = [
            metric_name
            for metric_name, values in metrics.items()
            if values is None
        ]

        if missing_metrics:
            raise RuntimeError(
                f"{model_name} is missing metrics: "
                f"{missing_metrics}\n"
                f"Available columns: "
                f"{history.columns.tolist()}"
            )

        curves[model_name] = {
            "epoch": epoch,
            **metrics,
        }

        print(
            f"{model_name} history: {history_path}"
        )

    return curves


def load_test_predictions(
    model_name,
    spec,
):
    prediction_path = (
        spec["run_dir"]
        / "test_predictions.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            prediction_path
        )

    frame = pd.read_csv(
        prediction_path
    )

    label_column = find_column(
        frame,
        [
            "label",
            "labels",
            "target",
            "targets",
            "y_true",
            "gt",
        ],
    )

    if label_column is None:
        raise RuntimeError(
            f"{model_name}: no label column found. "
            f"Available columns: {frame.columns.tolist()}"
        )

    probability_column = find_column(
        frame,
        spec["probability_columns"],
    )

    if probability_column is None:
        probability_column = next(
            (
                column
                for column in frame.columns
                if "prob" in str(column).lower()
            ),
            None,
        )

    labels = pd.to_numeric(
        frame[label_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    if probability_column is not None:
        probabilities = pd.to_numeric(
            frame[probability_column],
            errors="coerce",
        ).to_numpy(dtype=float)

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

            probabilities = sigmoid(
                logits
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
                    f"{model_name}: no probability, "
                    "logit, or prediction column found. "
                    f"Available columns: "
                    f"{frame.columns.tolist()}"
                )

            prediction_values = pd.to_numeric(
                frame[prediction_column],
                errors="coerce",
            ).to_numpy(dtype=float)

            if (
                prediction_values.min() < 0.0
                or prediction_values.max() > 1.0
            ):
                probabilities = sigmoid(
                    prediction_values
                )

                prediction_source = (
                    f"{prediction_column} "
                    "converted with sigmoid"
                )
            else:
                probabilities = prediction_values
                prediction_source = prediction_column

    if len(labels) != len(probabilities):
        raise RuntimeError(
            f"{model_name}: labels and probabilities "
            "have different lengths."
        )

    if not np.isfinite(labels).all():
        raise RuntimeError(
            f"{model_name}: labels contain invalid values."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            f"{model_name}: probabilities contain "
            "invalid values."
        )

    labels = labels.astype(int)

    if not set(
        np.unique(labels).tolist()
    ).issubset({0, 1}):
        raise RuntimeError(
            f"{model_name}: labels must be binary."
        )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    return {
        "labels": labels,
        "probabilities": probabilities,
        "source": prediction_source,
    }


def calculate_calibration_bins(
    labels,
    probabilities,
    n_bins=N_BINS,
):
    sorted_indices = np.argsort(
        probabilities,
        kind="mergesort",
    )

    groups = np.array_split(
        sorted_indices,
        min(
            n_bins,
            len(sorted_indices),
        ),
    )

    rows = []

    for bin_number, indices in enumerate(
        groups,
        start=1,
    ):
        if len(indices) == 0:
            continue

        bin_labels = labels[
            indices
        ]

        bin_probabilities = probabilities[
            indices
        ]

        total = int(
            len(indices)
        )

        mean_probability = float(
            bin_probabilities.mean()
        )

        observed_rate = float(
            bin_labels.mean()
        )

        rows.append(
            {
                "bin": bin_number,
                "n": total,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_error": abs(
                    observed_rate
                    - mean_probability
                ),
            }
        )

    table = pd.DataFrame(
        rows
    )

    weights = (
        table["n"]
        / table["n"].sum()
    )

    ece = float(
        (
            weights
            * table["absolute_error"]
        ).sum()
    )

    brier = float(
        brier_score_loss(
            labels,
            probabilities,
        )
    )

    return table, ece, brier


def load_calibration_curves():
    curves = {}
    cohort_size = None
    positive_count = None

    for model_name, spec in MODEL_SPECS.items():
        data = load_test_predictions(
            model_name,
            spec,
        )

        labels = data["labels"]
        probabilities = data["probabilities"]

        current_size = int(
            len(labels)
        )

        current_positives = int(
            labels.sum()
        )

        if cohort_size is None:
            cohort_size = current_size
            positive_count = current_positives
        elif (
            current_size != cohort_size
            or current_positives != positive_count
        ):
            raise RuntimeError(
                f"{model_name}: test cohort differs. "
                f"Expected N={cohort_size}, "
                f"positives={positive_count}; "
                f"found N={current_size}, "
                f"positives={current_positives}."
            )

        table, ece, brier = (
            calculate_calibration_bins(
                labels,
                probabilities,
            )
        )

        curves[model_name] = {
            "table": table,
            "ece": ece,
            "brier": brier,
            "source": data["source"],
        }

        print(
            f"{model_name} predictions: "
            f"{data['source']} | "
            f"N={current_size} | "
            f"positives={current_positives} | "
            f"Brier={brier:.5f} | "
            f"ECE={ece:.5f}"
        )

    return curves


def create_valid_mask(
    epoch,
    values,
):
    return (
        epoch.notna()
        & values.notna()
        & epoch.between(
            START_EPOCH,
            END_EPOCH,
        )
    )


def plot_validation_metric(
    axis,
    curves,
    metric_key,
    title,
    ylabel,
    ylim,
):
    for model_name, item in curves.items():
        values = item[
            metric_key
        ]

        valid_mask = create_valid_mask(
            item["epoch"],
            values,
        )

        if not valid_mask.any():
            raise RuntimeError(
                f"{model_name} has no valid "
                f"{metric_key} values."
            )

        epochs = item[
            "epoch"
        ][valid_mask]

        metric_values = values[
            valid_mask
        ]

        axis.plot(
            epochs,
            metric_values,
            color=MODEL_COLORS[
                model_name
            ],
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=model_name,
        )

    axis.set_title(
        title,
        fontsize=PLOT_TEXT_FONT_SIZE,
        fontweight="bold",
    )

    axis.set_xlabel(
        "Epoch",
        fontsize=PLOT_TEXT_FONT_SIZE,
        fontweight="bold",
    )

    axis.set_ylabel(
        ylabel,
        fontsize=PLOT_TEXT_FONT_SIZE,
        fontweight="bold",
    )

    axis.set_xlim(
        START_EPOCH,
        END_EPOCH,
    )

    axis.set_ylim(*ylim)

    axis.set_xticks(
        range(
            START_EPOCH,
            END_EPOCH + 1,
        )
    )

    axis.grid(
        True,
        alpha=GRID_ALPHA,
    )

    axis.legend(
        prop={
            "size": LEGEND_FONT_SIZE,
            "weight": "bold",
        },
        loc="best",
    )

    set_tick_style(
        axis
    )


def plot_calibration(
    axis,
    curves,
):
    maximum_value = 0.0

    for item in curves.values():
        table = item[
            "table"
        ]

        maximum_value = max(
            maximum_value,
            float(
                table[
                    "mean_probability"
                ].max()
            ),
            float(
                table[
                    "observed_rate"
                ].max()
            ),
        )

    plot_limit = min(
        1.0,
        max(
            0.4,
            np.ceil(
                maximum_value * 10.0
            )
            / 10.0
            + 0.05,
        ),
    )

    axis.plot(
        [0.0, plot_limit],
        [0.0, plot_limit],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Perfect calibration",
    )

    for model_name, item in curves.items():
        table = item[
            "table"
        ].sort_values(
            "bin"
        )

        axis.plot(
            table[
                "mean_probability"
            ],
            table[
                "observed_rate"
            ],
            color=MODEL_COLORS[
                model_name
            ],
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=model_name,
        )

    axis.set_title(
        "Calibration",
        fontsize=PLOT_TEXT_FONT_SIZE,
        fontweight="bold",
    )

    axis.set_xlabel(
        "Mean predicted probability",
        fontsize=PLOT_TEXT_FONT_SIZE,
        fontweight="bold",
    )

    axis.set_ylabel(
        "Observed event rate",
        fontsize=PLOT_TEXT_FONT_SIZE,
        fontweight="bold",
    )

    axis.set_xlim(
        0.0,
        plot_limit,
    )

    axis.set_ylim(
        0.0,
        plot_limit,
    )

    axis.grid(
        True,
        alpha=GRID_ALPHA,
    )

    axis.legend(
        prop={
            "size": LEGEND_FONT_SIZE,
            "weight": "bold",
        },
        loc="best",
    )

    set_tick_style(
        axis
    )


validation_curves = (
    load_validation_curves()
)

calibration_curves = (
    load_calibration_curves()
)

figure, axes = plt.subplots(
    2,
    2,
    figsize=FIGSIZE,
)

plot_validation_metric(
    axes[0, 0],
    validation_curves,
    "val_loss",
    "Validation Loss",
    "Loss",
    LOSS_YLIM,
)

plot_validation_metric(
    axes[0, 1],
    validation_curves,
    "val_auroc",
    "Validation AUROC",
    "AUROC",
    AUROC_YLIM,
)

plot_validation_metric(
    axes[1, 0],
    validation_curves,
    "val_auprc",
    "Validation AUPRC",
    "AUPRC",
    AUPRC_YLIM,
)

plot_calibration(
    axes[1, 1],
    calibration_curves,
)

figure.suptitle(
    "Validation and Calibration Curves for All Seven Models",
    fontsize=MAIN_TITLE_FONT_SIZE,
    fontweight="bold",
    y=1.01,
)

figure.tight_layout()

figure.savefig(
    SAVE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(figure)

print("Saved:", SAVE_PATH)
