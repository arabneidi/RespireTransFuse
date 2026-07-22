"""Build the diagnostic and calibration report for MedFuse Uni-CXR.

The script reads the bundled trainer's histories and prediction outputs, derives
consistent validation and held-out summaries, and calculates the calibration
table used in the project comparison. It saves the supporting CSV reports and a
publication-ready figure in PNG and PDF formats alongside the model outputs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import beta
from sklearn.metrics import brier_score_loss


BASE = Path("/content/drive/MyDrive/respire-transfuse")

RUN_DIR = (
    BASE
    / "outputs"
    / "medfuse_cxr_only_resnet34"
)

HISTORY_PATH = (
    RUN_DIR
    / "history.csv"
)

PREDICTIONS_PATH = (
    RUN_DIR
    / "test_predictions.csv"
)

SAVE_PATH = (
    RUN_DIR
    / "medfuse_uni_cxr_training_validation_calibration.png"
)

SAVE_PDF_PATH = (
    RUN_DIR
    / "medfuse_uni_cxr_training_validation_calibration.pdf"
)

CALIBRATION_CSV_PATH = (
    RUN_DIR
    / "test_calibration_bins_10_equal_frequency.csv"
)

SUMMARY_CSV_PATH = (
    RUN_DIR
    / "medfuse_uni_cxr_plot_summary.csv"
)

START_EPOCH = 1
MAX_EPOCH = 30

LOSS_YLIM = (0.3, 0.5)
AUROC_YLIM = (0.5, 0.8)
AUPRC_YLIM = (0.0, 0.4)

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
            "gt",
        ],
    )

    if label_column is None:
        raise RuntimeError(
            "No label column found in "
            f"{path}.\n"
            f"Available columns: "
            f"{frame.columns.tolist()}"
        )

    probability_column = find_column(
        frame,
        [
            "prob",
            "probability",
            "medfuse_prob",
            "fusion_prob",
            "y_prob",
            "score",
        ],
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
        prediction_column = find_column(
            frame,
            [
                "prediction",
                "pred",
                "logit",
                "logits",
            ],
        )

        if prediction_column is None:
            raise RuntimeError(
                "No probability, prediction, or logit "
                f"column found in {path}.\n"
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
                f"{prediction_column} converted "
                "with sigmoid"
            )
        else:
            probabilities = (
                prediction_values
            )

            prediction_source = (
                prediction_column
            )

    if len(labels) != len(probabilities):
        raise RuntimeError(
            "Test labels and probabilities have "
            "different lengths."
        )

    if not np.isfinite(labels).all():
        raise RuntimeError(
            "Test labels contain invalid values."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Test probabilities contain invalid values."
        )

    labels = labels.astype(int)

    unique_labels = set(
        np.unique(labels).tolist()
    )

    if not unique_labels.issubset({0, 1}):
        raise RuntimeError(
            f"Test labels must be binary. "
            f"Found: {unique_labels}"
        )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    return (
        labels,
        probabilities,
        prediction_source,
    )


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

    return (
        float(lower),
        float(upper),
    )


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

        positives = int(
            bin_labels.sum()
        )

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
                "bin": int(bin_number),
                "n": total,
                "positives": positives,
                "probability_min": float(
                    bin_probabilities.min()
                ),
                "probability_max": float(
                    bin_probabilities.max()
                ),
                "mean_probability": (
                    mean_probability
                ),
                "observed_rate": (
                    observed_rate
                ),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "signed_error": float(
                    observed_rate
                    - mean_probability
                ),
                "absolute_error": float(
                    abs(
                        observed_rate
                        - mean_probability
                    )
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

    mce = float(
        table["absolute_error"].max()
    )

    return (
        table,
        ece,
        mce,
    )


def plot_metric(
    axis,
    epochs,
    train_values,
    validation_values,
    title,
    ylabel,
    ylim,
    end_epoch,
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
        end_epoch,
    )

    axis.set_ylim(
        *ylim
    )

    axis.set_xticks(
        range(
            START_EPOCH,
            end_epoch + 1,
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

    set_tick_style(
        axis
    )


if not HISTORY_PATH.exists():
    raise FileNotFoundError(
        HISTORY_PATH
    )

history = pd.read_csv(
    HISTORY_PATH
)

required_history_columns = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_auroc",
    "val_auroc",
    "train_auprc",
    "val_auprc",
]

missing_history_columns = [
    column
    for column in required_history_columns
    if column not in history.columns
]

if missing_history_columns:
    raise RuntimeError(
        "Missing history columns: "
        f"{missing_history_columns}\n"
        f"Available columns: "
        f"{history.columns.tolist()}"
    )

for column in required_history_columns:
    history[column] = pd.to_numeric(
        history[column],
        errors="coerce",
    )

history = (
    history[
        history["epoch"].between(
            START_EPOCH,
            MAX_EPOCH,
        )
    ]
    .sort_values("epoch")
    .reset_index(drop=True)
)

if history.empty:
    raise RuntimeError(
        f"No epochs found between "
        f"{START_EPOCH} and {MAX_EPOCH}."
    )

invalid_history_columns = [
    column
    for column in required_history_columns
    if history[column].isna().any()
]

if invalid_history_columns:
    raise RuntimeError(
        "Invalid history values in columns: "
        f"{invalid_history_columns}"
    )

end_epoch = int(
    history["epoch"].max()
)

labels, probabilities, prediction_source = (
    load_test_predictions(
        PREDICTIONS_PATH
    )
)

if len(labels) != 1695:
    raise RuntimeError(
        f"Expected 1695 test samples, "
        f"found {len(labels)}."
    )

if int(labels.sum()) != 237:
    raise RuntimeError(
        f"Expected 237 test positives, "
        f"found {int(labels.sum())}."
    )

calibration_table, ece, mce = (
    calculate_calibration_bins(
        labels,
        probabilities,
    )
)

calibration_table.to_csv(
    CALIBRATION_CSV_PATH,
    index=False,
)

brier = float(
    brier_score_loss(
        labels,
        probabilities,
    )
)

prevalence = float(
    labels.mean()
)

null_brier = float(
    prevalence
    * (
        1.0
        - prevalence
    )
)

brier_skill = float(
    1.0
    - brier
    / null_brier
)

summary = pd.DataFrame(
    [
        {
            "model": (
                "MedFuse Uni-CXR"
            ),
            "prediction_source": (
                prediction_source
            ),
            "test_n": int(
                len(labels)
            ),
            "test_positives": int(
                labels.sum()
            ),
            "test_negatives": int(
                len(labels)
                - labels.sum()
            ),
            "prevalence": prevalence,
            "mean_probability": float(
                probabilities.mean()
            ),
            "brier": brier,
            "null_brier": null_brier,
            "brier_skill": brier_skill,
            "ece_10_equal_frequency": ece,
            "mce_10_equal_frequency": mce,
        }
    ]
)

summary.to_csv(
    SUMMARY_CSV_PATH,
    index=False,
)

epochs = history[
    "epoch"
].to_numpy()

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
    end_epoch,
)

plot_metric(
    axes[0, 1],
    epochs,
    history["train_auroc"],
    history["val_auroc"],
    "AUROC",
    "AUROC",
    AUROC_YLIM,
    end_epoch,
)

plot_metric(
    axes[1, 0],
    epochs,
    history["train_auprc"],
    history["val_auprc"],
    "AUPRC",
    "AUPRC",
    AUPRC_YLIM,
    end_epoch,
)

calibration_axis = axes[
    1,
    1,
]

mean_probabilities = calibration_table[
    "mean_probability"
].to_numpy()

observed_rates = calibration_table[
    "observed_rate"
].to_numpy()

lower_errors = (
    calibration_table[
        "observed_rate"
    ]
    - calibration_table[
        "ci_lower"
    ]
).to_numpy()

upper_errors = (
    calibration_table[
        "ci_upper"
    ]
    - calibration_table[
        "observed_rate"
    ]
).to_numpy()

maximum_value = max(
    float(
        mean_probabilities.max()
    ),
    float(
        observed_rates.max()
    ),
)

plot_limit = min(
    1.0,
    max(
        0.4,
        np.ceil(
            maximum_value
            * 10.0
        )
        / 10.0
        + 0.05,
    ),
)

calibration_axis.plot(
    [
        0.0,
        plot_limit,
    ],
    [
        0.0,
        plot_limit,
    ],
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
    label="MedFuse Uni-CXR",
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

set_tick_style(
    calibration_axis
)

figure.suptitle(
    "MedFuse Uni-CXR Training, Validation, and Calibration Curves",
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

figure.savefig(
    SAVE_PDF_PATH,
    bbox_inches="tight",
)

plt.show()
plt.close(
    figure
)

print("=" * 90)
print("MEDFUSE UNI-CXR PLOT COMPLETE")
print("=" * 90)
print("History:", HISTORY_PATH)
print("Predictions:", PREDICTIONS_PATH)
print("Prediction column:", prediction_source)
print("Epochs plotted:", START_EPOCH, "to", end_epoch)
print("Test samples:", len(labels))
print("Test positives:", int(labels.sum()))
print("Test prevalence:", f"{prevalence:.5f}")
print("Mean probability:", f"{probabilities.mean():.5f}")
print("Brier score:", f"{brier:.5f}")
print("Brier skill:", f"{brier_skill:.5f}")
print("ECE:", f"{ece:.5f}")
print("MCE:", f"{mce:.5f}")
print()
print("Saved PNG:", SAVE_PATH)
print("Saved PDF:", SAVE_PDF_PATH)
print("Saved calibration table:", CALIBRATION_CSV_PATH)
print("Saved summary:", SUMMARY_CSV_PATH)
