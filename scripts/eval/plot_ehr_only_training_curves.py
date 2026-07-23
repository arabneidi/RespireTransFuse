"""Plot training, validation, and calibration results for the EHR-only model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import beta
from sklearn.metrics import brier_score_loss


BASE = Path("/content/drive/MyDrive/respire-transfuse")
RUN_DIR = BASE / "outputs/ehr_only"
HISTORY_PATH = RUN_DIR / "history.csv"
PREDICTIONS_PATH = RUN_DIR / "test_predictions.csv"
SAVE_PATH = RUN_DIR / "ehr_only_training_validation_calibration.png"

START_EPOCH = 1
END_EPOCH = 20

LOSS_YLIM = (0.0, 1.0)
AUROC_YLIM = (0.0, 1.0)
AUPRC_YLIM = (0.0, 1.0)

N_BINS = 10
CONFIDENCE = 0.95

FIGSIZE = (15, 11)
LINE_WIDTH = 2.0
MARKER_SIZE = 5
GRID_ALPHA = 0.3
MAIN_TITLE_FONT_SIZE = 20
PLOT_TEXT_FONT_SIZE = 16
LEGEND_FONT_SIZE = 14
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


def find_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column

    return None


def load_test_predictions(path):
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    label_column = find_column(
        frame,
        [
            "label",
            "labels",
            "target",
            "targets",
            "y_true",
        ],
    )

    if label_column is None:
        raise RuntimeError(
            "No label column found. "
            f"Available columns: {frame.columns.tolist()}"
        )

    probability_column = find_column(
        frame,
        [
            "calibrated_prob",
            "prob_calibrated",
            "calibrated_probability",
            "probability_calibrated",
            "prob_cal",
        ],
    )

    if probability_column is None:
        probability_column = next(
            (
                column
                for column in frame.columns
                if "calibrat" in column.lower()
                and "prob" in column.lower()
            ),
            None,
        )

    if probability_column is None:
        probability_column = find_column(
            frame,
            [
                "prob",
                "ehr_prob",
                "probability",
                "raw_prob",
                "prob_raw",
            ],
        )

    if probability_column is None:
        probability_column = next(
            (
                column
                for column in frame.columns
                if "prob" in column.lower()
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
            [
                "ehr_logit",
                "logit",
            ],
        )

        if logit_column is None:
            raise RuntimeError(
                "No probability or logit column found. "
                f"Available columns: {frame.columns.tolist()}"
            )

        logits = pd.to_numeric(
            frame[logit_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        logits = np.clip(
            logits,
            -50.0,
            50.0,
        )

        probabilities = 1.0 / (
            1.0 + np.exp(-logits)
        )

        prediction_source = (
            f"{logit_column} converted with sigmoid"
        )

    if len(labels) != len(probabilities):
        raise RuntimeError(
            "Test labels and predictions have different lengths."
        )

    if not np.isfinite(labels).all():
        raise RuntimeError(
            "Test labels contain invalid values."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Test predictions contain invalid values."
        )

    labels = labels.astype(int)

    if not set(
        np.unique(labels).tolist()
    ).issubset({0, 1}):
        raise RuntimeError(
            "Test labels must be binary."
        )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    return labels, probabilities, prediction_source


def clopper_pearson_interval(
    positives,
    total,
    confidence=CONFIDENCE,
):
    alpha = 1.0 - confidence

    if positives == 0:
        lower = 0.0
    else:
        lower = beta.ppf(
            alpha / 2.0,
            positives,
            total - positives + 1,
        )

    if positives == total:
        upper = 1.0
    else:
        upper = beta.ppf(
            1.0 - alpha / 2.0,
            positives + 1,
            total - positives,
        )

    return float(lower), float(upper)


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
        min(n_bins, len(sorted_indices)),
    )

    rows = []

    for bin_number, indices in enumerate(
        groups,
        start=1,
    ):
        if len(indices) == 0:
            continue

        bin_labels = labels[indices]
        bin_probabilities = probabilities[indices]

        total = int(len(indices))
        positives = int(bin_labels.sum())

        mean_probability = float(
            bin_probabilities.mean()
        )

        observed_rate = float(
            bin_labels.mean()
        )

        ci_lower, ci_upper = (
            clopper_pearson_interval(
                positives,
                total,
            )
        )

        rows.append(
            {
                "bin": bin_number,
                "n": total,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "absolute_error": abs(
                    observed_rate - mean_probability
                ),
            }
        )

    table = pd.DataFrame(rows)

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

    return table, ece


def plot_metric(
    axis,
    epochs,
    train_values,
    validation_values,
    title,
    ylabel,
    ylim,
):
    axis.plot(
        epochs,
        train_values,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        label="Train",
    )

    axis.plot(
        epochs,
        validation_values,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        label="Validation",
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
    )

    set_tick_style(axis)


if not HISTORY_PATH.exists():
    raise FileNotFoundError(HISTORY_PATH)

history = pd.read_csv(HISTORY_PATH)

required_columns = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_auroc",
    "val_auroc",
    "train_auprc",
    "val_auprc",
]

missing_columns = [
    column
    for column in required_columns
    if column not in history.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing columns: {missing_columns}\n"
        f"Available columns: {history.columns.tolist()}"
    )

for column in required_columns:
    history[column] = pd.to_numeric(
        history[column],
        errors="coerce",
    )

history = (
    history[
        history["epoch"].between(
            START_EPOCH,
            END_EPOCH,
        )
    ]
    .sort_values("epoch")
    .reset_index(drop=True)
)

if history.empty:
    raise RuntimeError(
        f"No epochs found between "
        f"{START_EPOCH} and {END_EPOCH}."
    )

invalid_columns = [
    column
    for column in required_columns
    if history[column].isna().any()
]

if invalid_columns:
    raise RuntimeError(
        "Invalid numeric values in columns: "
        f"{invalid_columns}"
    )

epochs = history["epoch"].to_numpy()

labels, probabilities, prediction_source = (
    load_test_predictions(
        PREDICTIONS_PATH
    )
)

calibration_table, ece = (
    calculate_calibration_bins(
        labels,
        probabilities,
    )
)

brier = float(
    brier_score_loss(
        labels,
        probabilities,
    )
)

figure, axes = plt.subplots(
    2,
    2,
    figsize=FIGSIZE,
)

plot_metric(
    axes[0, 0],
    epochs,
    history["train_loss"],
    history["val_loss"],
    "Loss",
    "Loss",
    LOSS_YLIM,
)

plot_metric(
    axes[0, 1],
    epochs,
    history["train_auroc"],
    history["val_auroc"],
    "AUROC",
    "AUROC",
    AUROC_YLIM,
)

plot_metric(
    axes[1, 0],
    epochs,
    history["train_auprc"],
    history["val_auprc"],
    "AUPRC",
    "AUPRC",
    AUPRC_YLIM,
)

calibration_axis = axes[1, 1]

mean_probabilities = calibration_table[
    "mean_probability"
].to_numpy()

observed_rates = calibration_table[
    "observed_rate"
].to_numpy()

lower_errors = (
    calibration_table["observed_rate"]
    - calibration_table["ci_lower"]
).to_numpy()

upper_errors = (
    calibration_table["ci_upper"]
    - calibration_table["observed_rate"]
).to_numpy()

maximum_value = max(
    float(mean_probabilities.max()),
    float(observed_rates.max()),
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

calibration_axis.plot(
    [0.0, plot_limit],
    [0.0, plot_limit],
    linestyle="--",
    linewidth=1.8,
    label="Perfect calibration",
)

calibration_axis.errorbar(
    mean_probabilities,
    observed_rates,
    yerr=np.vstack(
        [
            lower_errors,
            upper_errors,
        ]
    ),
    marker="o",
    markersize=6,
    linewidth=2.0,
    capsize=4,
    label="EHR-Only",
)

calibration_axis.set_title(
    "Calibration",
    fontsize=PLOT_TEXT_FONT_SIZE,
    fontweight="bold",
)

calibration_axis.set_xlabel(
    "Mean predicted probability",
    fontsize=PLOT_TEXT_FONT_SIZE,
    fontweight="bold",
)

calibration_axis.set_ylabel(
    "Observed event rate",
    fontsize=PLOT_TEXT_FONT_SIZE,
    fontweight="bold",
)

calibration_axis.set_xlim(
    0.0,
    plot_limit,
)

calibration_axis.set_ylim(
    0.0,
    plot_limit,
)

calibration_axis.grid(
    True,
    alpha=GRID_ALPHA,
)

calibration_axis.legend(
    prop={
        "size": LEGEND_FONT_SIZE,
        "weight": "bold",
    },
    loc="lower right",
)

calibration_axis.text(
    0.04,
    0.96,
    (
        f"Brier = {brier:.3f}\n"
        f"ECE = {ece:.3f}"
    ),
    transform=calibration_axis.transAxes,
    va="top",
    fontsize=LEGEND_FONT_SIZE,
    fontweight="bold",
    bbox={
        "boxstyle": "round,pad=0.5",
        "facecolor": "white",
        "edgecolor": "#cccccc",
        "linewidth": 1.0,
        "alpha": 0.8,
    },
)

set_tick_style(calibration_axis)

figure.suptitle(
    "EHR-Only Training, Validation, and Calibration Curves",
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

print("Prediction column:", prediction_source)
print("Test samples:", len(labels))
print("Brier score:", f"{brier:.5f}")
print("ECE:", f"{ece:.5f}")
print("Saved:", SAVE_PATH)
