"""Compare held-out discrimination for all seven evaluated models.

The script loads each model's test labels and probabilities, verifies that the
prediction cohorts are compatible, and calculates AUROC and average precision on
the common held-out set. It saves separate publication-ready ROC and
precision-recall figures together with a CSV table of the plotted summary values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


BASE = Path("/content/drive/MyDrive/respire-transfuse")

OUTPUT_DIR = (
    BASE
    / "outputs"
    / "seven_model_discrimination_curves"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_SPECS = {
    "Image-Only": {
        "run_dir": BASE / "outputs/image_only",
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
    "EHR-Only": {
        "run_dir": BASE / "outputs/ehr_only",
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
    "Early Fusion": {
        "run_dir": BASE / "outputs/early_fusion",
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
        "run_dir": BASE / "outputs/medfuse_ehr",
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
        "run_dir": BASE / "outputs/medfuse_cxr_only_resnet34",
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
        "run_dir": (
            BASE
            / "outputs/medfuse_multimodal_lstm"
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
        "run_dir": BASE / "outputs/respire_transfuse",
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

LABEL_COLUMNS = [
    "label",
    "labels",
    "target",
    "targets",
    "y_true",
    "gt",
]

MAIN_TITLE_FONT_SIZE = 20
PLOT_TEXT_FONT_SIZE = 16
LEGEND_FONT_SIZE = 11
TICK_FONT_SIZE = 14

FIGSIZE = (17, 7)
GRID_ALPHA = 0.25
STANDARD_LINE_WIDTH = 1.8
RESPIRE_LINE_WIDTH = 3.2


def flatten(values):
    values = np.asarray(values)
    values = np.squeeze(values)

    if values.ndim != 1:
        raise RuntimeError(
            f"Expected one-dimensional values, got {values.shape}"
        )

    return values


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


def set_tick_style(axis):
    axis.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
    )

    for label in axis.get_xticklabels():
        label.set_fontweight("bold")

    for label in axis.get_yticklabels():
        label.set_fontweight("bold")


def validate_predictions(
    labels,
    probabilities,
    model_name,
):
    labels = flatten(
        labels
    ).astype(float)

    probabilities = flatten(
        probabilities
    ).astype(float)

    if len(labels) != len(probabilities):
        raise RuntimeError(
            f"{model_name}: labels and predictions "
            "have different lengths."
        )

    if not np.isfinite(labels).all():
        raise RuntimeError(
            f"{model_name}: labels contain invalid values."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            f"{model_name}: predictions contain "
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

    return labels, probabilities


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
        LABEL_COLUMNS,
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
                    f"{model_name}: no probability, logit, "
                    "or prediction column found. "
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

    labels, probabilities = validate_predictions(
        labels,
        probabilities,
        model_name,
    )

    return {
        "labels": labels,
        "probabilities": probabilities,
        "source": prediction_source,
    }


loaded = {
    model_name: load_test_predictions(
        model_name,
        spec,
    )
    for model_name, spec in MODEL_SPECS.items()
}

reference_model = next(
    iter(loaded)
)

reference_size = len(
    loaded[reference_model]["labels"]
)

reference_positives = int(
    loaded[reference_model]["labels"].sum()
)

for model_name, data in loaded.items():
    labels = data["labels"]

    if (
        len(labels) != reference_size
        or int(labels.sum()) != reference_positives
    ):
        raise RuntimeError(
            f"{model_name}: test cohort differs. "
            f"Expected N={reference_size}, "
            f"positives={reference_positives}; "
            f"found N={len(labels)}, "
            f"positives={int(labels.sum())}."
        )

prevalence = float(
    reference_positives
    / reference_size
)

figure, axes = plt.subplots(
    1,
    2,
    figsize=FIGSIZE,
)

summary_rows = []

for model_name, data in loaded.items():
    labels = data["labels"]
    probabilities = data["probabilities"]

    auprc = float(
        average_precision_score(
            labels,
            probabilities,
        )
    )

    auroc = float(
        roc_auc_score(
            labels,
            probabilities,
        )
    )

    precision, recall, _ = (
        precision_recall_curve(
            labels,
            probabilities,
        )
    )

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            labels,
            probabilities,
        )
    )

    is_respire = (
        model_name == "RespireTransFuse"
    )

    line_width = (
        RESPIRE_LINE_WIDTH
        if is_respire
        else STANDARD_LINE_WIDTH
    )

    line_zorder = (
        10
        if is_respire
        else 2
    )

    axes[0].plot(
        recall[::-1],
        precision[::-1],
        color=MODEL_COLORS[model_name],
        linewidth=line_width,
        zorder=line_zorder,
        label=(
            f"{model_name} "
            f"(AUPRC={auprc:.3f})"
        ),
    )

    axes[1].plot(
        false_positive_rate,
        true_positive_rate,
        color=MODEL_COLORS[model_name],
        linewidth=line_width,
        zorder=line_zorder,
        label=(
            f"{model_name} "
            f"(AUROC={auroc:.3f})"
        ),
    )

    summary_rows.append(
        {
            "Model": model_name,
            "Prediction Source": data["source"],
            "Test N": int(len(labels)),
            "Test Positives": int(labels.sum()),
            "Prevalence": prevalence,
            "AUROC": auroc,
            "AUPRC": auprc,
            "AUPRC / Prevalence": (
                auprc / prevalence
            ),
        }
    )

axes[0].axhline(
    prevalence,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=(
        f"No-skill baseline "
        f"(prevalence={prevalence:.3f})"
    ),
)

axes[0].set_title(
    "Precision–Recall Curves",
    fontsize=PLOT_TEXT_FONT_SIZE,
    fontweight="bold",
)

axes[0].set_xlabel(
    "Recall",
    fontsize=PLOT_TEXT_FONT_SIZE,
    fontweight="bold",
)

axes[0].set_ylabel(
    "Precision",
    fontsize=PLOT_TEXT_FONT_SIZE,
    fontweight="bold",
)

axes[0].set_xlim(
    0.0,
    1.0,
)

axes[0].set_ylim(
    0.0,
    1.0,
)

axes[0].grid(
    alpha=GRID_ALPHA,
)

axes[0].legend(
    loc="upper right",
    prop={
        "size": LEGEND_FONT_SIZE,
        "weight": "bold",
    },
)

set_tick_style(
    axes[0]
)

axes[1].plot(
    [0.0, 1.0],
    [0.0, 1.0],
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Random ranking",
)

axes[1].set_title(
    "Receiver Operating Characteristic Curves",
    fontsize=PLOT_TEXT_FONT_SIZE,
    fontweight="bold",
)

axes[1].set_xlabel(
    "False-positive rate",
    fontsize=PLOT_TEXT_FONT_SIZE,
    fontweight="bold",
)

axes[1].set_ylabel(
    "True-positive rate",
    fontsize=PLOT_TEXT_FONT_SIZE,
    fontweight="bold",
)

axes[1].set_xlim(
    0.0,
    1.0,
)

axes[1].set_ylim(
    0.0,
    1.0,
)

axes[1].grid(
    alpha=GRID_ALPHA,
)

axes[1].legend(
    loc="lower right",
    prop={
        "size": LEGEND_FONT_SIZE,
        "weight": "bold",
    },
)

set_tick_style(
    axes[1]
)

figure.suptitle(
    "Test-Set Discrimination Curves",
    fontsize=MAIN_TITLE_FONT_SIZE,
    fontweight="bold",
)

figure.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94,
    ]
)

png_path = (
    OUTPUT_DIR
    / "seven_models_pr_and_roc_curves.png"
)

pdf_path = (
    OUTPUT_DIR
    / "seven_models_pr_and_roc_curves.pdf"
)

figure.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

figure.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.show()
plt.close(figure)

summary = (
    pd.DataFrame(summary_rows)
    .sort_values(
        "AUPRC",
        ascending=False,
    )
    .reset_index(drop=True)
)

summary.insert(
    0,
    "Rank by AUPRC",
    np.arange(
        1,
        len(summary) + 1,
    ),
)

summary_path = (
    OUTPUT_DIR
    / "seven_models_discrimination_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

print()
print(
    summary.to_string(
        index=False,
        formatters={
            "Prevalence": lambda value: f"{value:.5f}",
            "AUROC": lambda value: f"{value:.5f}",
            "AUPRC": lambda value: f"{value:.5f}",
            "AUPRC / Prevalence": (
                lambda value: f"{value:.2f}"
            ),
        },
    )
)

print()
print("Saved figure:")
print(png_path)
print(pdf_path)

print()
print("Saved summary:")
print(summary_path)
