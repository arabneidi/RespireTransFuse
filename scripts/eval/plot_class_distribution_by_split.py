"""Summarize class balance across the patient-level data partitions.

The script reads the processed cohort, counts positive and negative outcomes in
the training, validation, and test splits, and compares split prevalence with the
overall cohort rate. It saves the underlying summary table and a publication-ready
two-panel figure in both PNG and PDF formats under the thesis figure directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter


BASE = Path("/content/drive/MyDrive/respire-transfuse")
COHORT_PATH = BASE / "data" / "processed" / "cohorts" / "cohort.csv"
OUT_DIR = BASE / "outputs" / "thesis_figures" / "data_descriptives"

OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUT_DIR / "02_class_distribution_and_prevalence.png"
PDF_PATH = OUT_DIR / "02_class_distribution_and_prevalence.pdf"
CSV_PATH = OUT_DIR / "02_class_distribution_and_prevalence.csv"

SPLIT_ORDER = ["train", "val", "test"]

SPLIT_LABELS = {
    "train": "Training",
    "val": "Validation",
    "test": "Test",
}

if not COHORT_PATH.exists():
    raise FileNotFoundError(COHORT_PATH)

cohort = pd.read_csv(COHORT_PATH)

label_col = next(
    (
        column
        for column in [
            "label",
            "outcome",
            "y",
            "target",
        ]
        if column in cohort.columns
    ),
    None,
)

if label_col is None:
    raise KeyError(
        f"Could not find the label column. Available columns: {cohort.columns.tolist()}"
    )

split_col = next(
    (
        column
        for column in [
            "split_std",
            "split",
            "dataset_split",
            "data_split",
            "partition",
            "set",
        ]
        if column in cohort.columns
    ),
    None,
)

if split_col is None:
    raise KeyError(
        f"Could not find the split column. Available columns: {cohort.columns.tolist()}"
    )

cohort = cohort.copy()

cohort["label"] = pd.to_numeric(
    cohort[label_col],
    errors="coerce",
)

split_mapping = {
    "train": "train",
    "training": "train",
    "0": "train",
    "val": "val",
    "valid": "val",
    "validate": "val",
    "validation": "val",
    "dev": "val",
    "1": "val",
    "test": "test",
    "testing": "test",
    "2": "test",
}

cohort["split_std"] = (
    cohort[split_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(split_mapping)
)

cohort = cohort[
    cohort["label"].isin([0, 1])
    & cohort["split_std"].isin(SPLIT_ORDER)
].copy()

cohort["label"] = cohort["label"].astype(int)

if cohort.empty:
    raise ValueError(
        "No valid training, validation, or test samples were found."
    )

split_counts = (
    pd.crosstab(
        cohort["split_std"],
        cohort["label"],
    )
    .reindex(
        index=SPLIT_ORDER,
        columns=[0, 1],
        fill_value=0,
    )
)

negative_counts = split_counts[0].astype(int)
positive_counts = split_counts[1].astype(int)
totals = negative_counts + positive_counts

if (totals == 0).any():
    empty_splits = totals[totals == 0].index.tolist()
    raise ValueError(
        f"No samples were found for these splits: {empty_splits}"
    )

prevalence = positive_counts / totals
overall_prevalence = cohort["label"].mean()

summary = pd.DataFrame(
    {
        "split": SPLIT_ORDER,
        "negative": negative_counts.to_numpy(),
        "positive": positive_counts.to_numpy(),
        "total": totals.to_numpy(),
        "positive_prevalence": prevalence.to_numpy(),
        "positive_prevalence_percent": prevalence.to_numpy() * 100,
    }
)

summary.to_csv(
    CSV_PATH,
    index=False,
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.edgecolor": "#111111",
        "axes.linewidth": 1.0,
        "savefig.facecolor": "white",
    }
)

plt.close("all")

fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 7.5),
)

fig.patch.set_facecolor("white")

positions = np.arange(len(SPLIT_ORDER))
split_labels = [
    SPLIT_LABELS[split_name]
    for split_name in SPLIT_ORDER
]

negative_bars = axes[0].bar(
    positions,
    negative_counts.to_numpy(),
    color="#4C78A8",
    edgecolor="#111111",
    linewidth=1.0,
    label="Negative",
)

positive_bars = axes[0].bar(
    positions,
    positive_counts.to_numpy(),
    bottom=negative_counts.to_numpy(),
    color="#F58518",
    edgecolor="#111111",
    linewidth=1.0,
    label="Positive",
)

axes[0].set_xticks(positions)
axes[0].set_xticklabels(split_labels)

axes[0].set_ylabel(
    "Number of samples",
    fontsize=16,
    fontweight="bold",
)

axes[0].set_title(
    "Outcome Distribution by Split",
    fontsize=16,
    fontweight="bold",
    pad=12,
)

axes[0].grid(
    axis="y",
    alpha=0.3,
    linewidth=0.7,
    color="#B0B0B0",
)

axes[0].set_axisbelow(True)

axes[0].yaxis.set_major_locator(
    MaxNLocator(nbins=6, integer=True)
)

axes[0].yaxis.set_major_formatter(
    FuncFormatter(
        lambda value, position: f"{int(value):,}"
    )
)

stacked_upper_limit = totals.max() * 1.20

axes[0].set_ylim(
    0,
    stacked_upper_limit,
)

legend = axes[0].legend(
    fontsize=14,
    frameon=True,
    loc="upper right",
)

for text in legend.get_texts():
    text.set_fontweight("bold")

for tick in axes[0].get_xticklabels():
    tick.set_fontweight("bold")

for tick in axes[0].get_yticklabels():
    tick.set_fontweight("bold")

for index, split_name in enumerate(SPLIT_ORDER):
    negative = int(negative_counts.loc[split_name])
    positive = int(positive_counts.loc[split_name])
    total = int(totals.loc[split_name])

    if negative > 0:
        axes[0].text(
            index,
            negative / 2,
            f"{negative:,}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white",
        )

    if positive > 0:
        axes[0].text(
            index,
            negative + positive / 2,
            f"{positive:,}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#111111",
        )

    axes[0].text(
        index,
        total + totals.max() * 0.025,
        f"Total: {total:,}",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#111111",
    )

prevalence_percent = prevalence.to_numpy() * 100

prevalence_bars = axes[1].bar(
    positions,
    prevalence_percent,
    color="#F58518",
    edgecolor="#111111",
    linewidth=1.0,
    label="Split prevalence",
)

axes[1].axhline(
    overall_prevalence * 100,
    color="#4C78A8",
    linestyle="--",
    linewidth=2.0,
    label=(
        "Overall prevalence "
        f"({overall_prevalence * 100:.2f}%)"
    ),
)

axes[1].set_xticks(positions)
axes[1].set_xticklabels(split_labels)

axes[1].set_ylabel(
    "Positive prevalence",
    fontsize=16,
    fontweight="bold",
)

axes[1].yaxis.set_major_formatter(
    PercentFormatter(xmax=100)
)

axes[1].yaxis.set_major_locator(
    MaxNLocator(nbins=6)
)

axes[1].set_title(
    "Positive Prevalence by Split",
    fontsize=16,
    fontweight="bold",
    pad=12,
)

axes[1].grid(
    axis="y",
    alpha=0.3,
    linewidth=0.7,
    color="#B0B0B0",
)

axes[1].set_axisbelow(True)

upper_limit = max(
    prevalence_percent.max() * 1.30,
    overall_prevalence * 100 * 1.25,
    5,
)

axes[1].set_ylim(
    0,
    upper_limit,
)

legend = axes[1].legend(
    fontsize=14,
    frameon=True,
    loc="upper right",
)

for text in legend.get_texts():
    text.set_fontweight("bold")

for tick in axes[1].get_xticklabels():
    tick.set_fontweight("bold")

for tick in axes[1].get_yticklabels():
    tick.set_fontweight("bold")

for bar, value in zip(
    prevalence_bars,
    prevalence_percent,
):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + upper_limit * 0.025,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#111111",
    )

fig.suptitle(
    "Class Imbalance and Patient-Level Split Balance",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)

fig.subplots_adjust(
    left=0.07,
    right=0.98,
    bottom=0.12,
    top=0.86,
    wspace=0.16,
)

plt.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    PDF_PATH,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

print("Cohort file:", COHORT_PATH)
print("Label column:", label_col)
print("Split column:", split_col)
print(summary.to_string(index=False))
print("Saved:", PNG_PATH)
print("Saved:", PDF_PATH)
print("Saved:", CSV_PATH)
