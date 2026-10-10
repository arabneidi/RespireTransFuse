from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

BASE = Path("/content/drive/MyDrive/respire-transfuse")

RESPIRE_DIR = (
    BASE
    / "outputs"
    / "respire_transfuse"
)

OUTPUT_DIR = (
    BASE
    / "outputs"
    / "respire_calibration_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_PATH = (
    RESPIRE_DIR
    / "test_predictions.csv"
)

CALIBRATED_PATH = (
    RESPIRE_DIR
    / "test_predictions_calibrated.csv"
)



N_BINS = 10

LABEL_COLUMNS = [
    "label",
    "labels",
    "target",
    "targets",
    "y_true",
    "gt",
]

RAW_PROBABILITY_COLUMNS = [
    "fusion_prob",
    "prob",
    "probability",
    "raw_prob",
    "prob_raw",
]

CALIBRATED_PROBABILITY_COLUMNS = [
    "fusion_prob_calibrated",
    "calibrated_prob",
    "prob_calibrated",
    "calibrated_probability",
    "probability_calibrated",
]



def find_column(
    frame,
    candidates,
):
    lookup = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def load_predictions(
    path,
    probability_candidates,
    name,
):
    if not path.exists():
        raise FileNotFoundError(
            f"{name}: prediction file not found:\n{path}"
        )

    frame = pd.read_csv(
        path
    )

    label_column = find_column(
        frame,
        LABEL_COLUMNS,
    )

    probability_column = find_column(
        frame,
        probability_candidates,
    )

    if label_column is None:
        raise RuntimeError(
            f"{name}: no label column found.\n"
            f"Available columns: {frame.columns.tolist()}"
        )

    if probability_column is None:
        raise RuntimeError(
            f"{name}: no required probability column found.\n"
            f"Expected one of: {probability_candidates}\n"
            f"Available columns: {frame.columns.tolist()}"
        )

    labels = pd.to_numeric(
        frame[label_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    probabilities = pd.to_numeric(
        frame[probability_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(labels).all():
        raise RuntimeError(
            f"{name}: labels contain invalid values."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            f"{name}: probabilities contain invalid values."
        )

    labels = labels.astype(int)

    if not set(
        np.unique(labels).tolist()
    ).issubset({0, 1}):
        raise RuntimeError(
            f"{name}: labels must be binary."
        )

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    return (
        labels,
        probabilities,
        probability_column,
    )


def calibration_table(
    labels,
    probabilities,
    n_bins=10,
):
    order = np.argsort(
        probabilities,
        kind="mergesort",
    )

    groups = np.array_split(
        order,
        min(
            n_bins,
            len(order),
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

        mean_probability = float(
            bin_probabilities.mean()
        )

        observed_rate = float(
            bin_labels.mean()
        )

        rows.append(
            {
                "Bin": bin_number,
                "N": int(len(indices)),
                "Mean predicted probability": mean_probability,
                "Observed event rate": observed_rate,
                "Absolute error": abs(
                    observed_rate
                    - mean_probability
                ),
            }
        )

    table = pd.DataFrame(
        rows
    )

    weights = (
        table["N"]
        / table["N"].sum()
    )

    ece = float(
        (
            weights
            * table["Absolute error"]
        ).sum()
    )

    return table, ece



(
    raw_labels,
    raw_probabilities,
    raw_source,
) = load_predictions(
    RAW_PATH,
    RAW_PROBABILITY_COLUMNS,
    "Raw RespireTransFuse",
)



(
    calibrated_labels,
    calibrated_probabilities,
    calibrated_source,
) = load_predictions(
    CALIBRATED_PATH,
    CALIBRATED_PROBABILITY_COLUMNS,
    "Calibrated RespireTransFuse",
)



if len(
    raw_labels
) != len(
    calibrated_labels
):
    raise RuntimeError(
        "Raw and calibrated files have different "
        "numbers of test samples."
    )

if not np.array_equal(
    raw_labels,
    calibrated_labels,
):
    raise RuntimeError(
        "Raw and calibrated files do not contain "
        "the same ordered test labels."
    )



raw_table, raw_ece = calibration_table(
    raw_labels,
    raw_probabilities,
    N_BINS,
)

calibrated_table, calibrated_ece = calibration_table(
    calibrated_labels,
    calibrated_probabilities,
    N_BINS,
)



raw_brier = float(
    brier_score_loss(
        raw_labels,
        raw_probabilities,
    )
)

calibrated_brier = float(
    brier_score_loss(
        calibrated_labels,
        calibrated_probabilities,
    )
)

raw_auroc = float(
    roc_auc_score(
        raw_labels,
        raw_probabilities,
    )
)

calibrated_auroc = float(
    roc_auc_score(
        calibrated_labels,
        calibrated_probabilities,
    )
)

raw_auprc = float(
    average_precision_score(
        raw_labels,
        raw_probabilities,
    )
)

calibrated_auprc = float(
    average_precision_score(
        calibrated_labels,
        calibrated_probabilities,
    )
)

raw_mean_probability = float(
    raw_probabilities.mean()
)

calibrated_mean_probability = float(
    calibrated_probabilities.mean()
)

prevalence = float(
    raw_labels.mean()
)



fig, ax = plt.subplots(
    figsize=(8.5, 7)
)


ax.plot(
    [0.0, 0.5],
    [0.0, 0.5],
    color="black",
    linestyle="--",
    linewidth=1.6,
    label="Perfect calibration",
)


ax.plot(
    raw_table[
        "Mean predicted probability"
    ],
    raw_table[
        "Observed event rate"
    ],
    marker="o",
    markersize=8,
    linewidth=2.2,
    label=(
        f"Raw "
        f"(Brier={raw_brier:.4f}, "
        f"ECE={raw_ece:.4f})"
    ),
)


ax.plot(
    calibrated_table[
        "Mean predicted probability"
    ],
    calibrated_table[
        "Observed event rate"
    ],
    marker="s",
    markersize=8,
    linewidth=2.4,
    label=(
        f"Calibrated "
        f"(Brier={calibrated_brier:.4f}, "
        f"ECE={calibrated_ece:.4f})"
    ),
)


ax.axhline(
    prevalence,
    color="black",
    linestyle=":",
    linewidth=1.4,
    label=(
        f"Observed prevalence "
        f"({prevalence:.3f})"
    ),
)



all_values = np.concatenate(
    [
        raw_table[
            "Mean predicted probability"
        ].to_numpy(),

        raw_table[
            "Observed event rate"
        ].to_numpy(),

        calibrated_table[
            "Mean predicted probability"
        ].to_numpy(),

        calibrated_table[
            "Observed event rate"
        ].to_numpy(),
    ]
)

max_value = float(
    np.nanmax(
        all_values
    )
)

plot_limit = max(
    0.40,
    np.ceil(
        (
            max_value
            + 0.03
        )
        * 10
    )
    / 10,
)

plot_limit = min(
    1.0,
    plot_limit,
)

ax.lines[0].set_data(
    [0.0, plot_limit],
    [0.0, plot_limit],
)

ax.set_xlim(
    0.0,
    plot_limit,
)

ax.set_ylim(
    0.0,
    plot_limit,
)



ax.set_title(
    "RespireTransFuse Calibration Before and After Recalibration",
    fontsize=18,
    fontweight="bold",
)

ax.set_xlabel(
    "Mean predicted probability",
    fontsize=15,
    fontweight="bold",
)

ax.set_ylabel(
    "Observed deterioration rate",
    fontsize=15,
    fontweight="bold",
)

ax.tick_params(
    axis="both",
    labelsize=13,
)

for label in ax.get_xticklabels():
    label.set_fontweight("bold")

for label in ax.get_yticklabels():
    label.set_fontweight("bold")

ax.grid(
    alpha=0.25,
)

ax.legend(
    loc="upper left",
    fontsize=11,
    frameon=True,
)

fig.tight_layout()



PNG_PATH = (
    OUTPUT_DIR
    / "respire_calibration_raw_vs_calibrated.png"
)

JPG_PATH = (
    OUTPUT_DIR
    / "respire_calibration_raw_vs_calibrated.jpg"
)

PDF_PATH = (
    OUTPUT_DIR
    / "respire_calibration_raw_vs_calibrated.pdf"
)

fig.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    JPG_PATH,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)



raw_save = raw_table.copy()
raw_save.insert(
    0,
    "Version",
    "Raw",
)

calibrated_save = calibrated_table.copy()
calibrated_save.insert(
    0,
    "Version",
    "Calibrated",
)

comparison_table = pd.concat(
    [
        raw_save,
        calibrated_save,
    ],
    ignore_index=True,
)

CSV_PATH = (
    OUTPUT_DIR
    / "respire_calibration_bins_raw_vs_calibrated.csv"
)

comparison_table.to_csv(
    CSV_PATH,
    index=False,
)



print()
print("RespireTransFuse calibration comparison")
print("----------------------------------------")

print(
    f"Test N: {len(raw_labels)}"
)

print(
    f"Positives: {int(raw_labels.sum())}"
)

print(
    f"Prevalence: {prevalence:.6f}"
)

print()

print(
    f"Raw source: {raw_source}"
)

print(
    f"Raw AUROC: {raw_auroc:.6f}"
)

print(
    f"Raw AUPRC: {raw_auprc:.6f}"
)

print(
    f"Raw Brier: {raw_brier:.6f}"
)

print(
    f"Raw ECE: {raw_ece:.6f}"
)

print(
    f"Raw mean probability: "
    f"{raw_mean_probability:.6f}"
)

print()

print(
    f"Calibrated source: {calibrated_source}"
)

print(
    f"Calibrated AUROC: "
    f"{calibrated_auroc:.6f}"
)

print(
    f"Calibrated AUPRC: "
    f"{calibrated_auprc:.6f}"
)

print(
    f"Calibrated Brier: "
    f"{calibrated_brier:.6f}"
)

print(
    f"Calibrated ECE: "
    f"{calibrated_ece:.6f}"
)

print(
    f"Calibrated mean probability: "
    f"{calibrated_mean_probability:.6f}"
)

print()

print("Saved:")
print(PNG_PATH)
print(JPG_PATH)
print(PDF_PATH)
print(CSV_PATH)
