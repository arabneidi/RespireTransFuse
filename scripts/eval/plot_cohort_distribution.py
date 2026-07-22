"""Describe cohort composition and the timing of positive outcomes.

Using the final cohort and its respiratory-event timestamps, the script reports
how samples are distributed across outcome groups and when qualifying future
events occur relative to the index radiograph. It produces matching PNG and PDF
figures for the data-descriptive section of the project results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


BASE = Path("/content/drive/MyDrive/respire-transfuse")
OUT_DIR = BASE / "outputs" / "thesis_figures" / "data_descriptives"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUT_DIR / "cohort_composition_and_positive_event_timing.png"
PDF_PATH = OUT_DIR / "cohort_composition_and_positive_event_timing.pdf"

candidate_paths = [
    BASE / "data" / "processed" / "cohorts" / "cohort.csv",
    BASE / "outputs" / "clean_source_step0" / "respire_clean_source_base_cohort.csv",
]

cohort_path = next(
    (path for path in candidate_paths if path.exists()),
    None,
)

if cohort_path is None:
    raise FileNotFoundError(
        "Could not find the cohort CSV in the expected locations."
    )

df = pd.read_csv(cohort_path)

label_col = next(
    (
        column
        for column in ["label", "outcome", "y", "target"]
        if column in df.columns
    ),
    None,
)

if label_col is None:
    raise KeyError(
        f"Could not find the label column. Available columns: {df.columns.tolist()}"
    )

hours_col = next(
    (
        column
        for column in [
            "hours_to_future_resp_event",
            "future_resp_hours",
            "time_to_event_hours",
            "hours_to_event",
            "resp_event_hours",
            "future_event_hours",
            "event_hours",
        ]
        if column in df.columns
    ),
    None,
)

if hours_col is None:
    raise KeyError(
        f"Could not find the event-time column. Available columns: {df.columns.tolist()}"
    )

df = df.copy()
df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
df[hours_col] = pd.to_numeric(df[hours_col], errors="coerce")

negatives = df[df[label_col] == 0].copy()
all_positives = df[df[label_col] == 1].copy()

if all_positives.empty:
    raise ValueError("No positive samples were found.")

bin_edges = [0, 6, 12, 24, 48]
bin_labels = ["0–6 h", "6–12 h", "12–24 h", "24–48 h"]

positives = all_positives[
    all_positives[hours_col].between(0, 48, inclusive="both")
].copy()

if positives.empty:
    raise ValueError(
        f"No positive samples have valid {hours_col} values between 0 and 48 hours."
    )

positives["time_bin"] = pd.cut(
    positives[hours_col],
    bins=bin_edges,
    labels=bin_labels,
    right=True,
    include_lowest=True,
)

bin_counts = (
    positives["time_bin"]
    .value_counts(sort=False)
    .reindex(bin_labels, fill_value=0)
    .astype(int)
)

negative_total = len(negatives)
positive_total = int(bin_counts.sum())
cohort_total = negative_total + positive_total

overall_counts = pd.Series(
    [negative_total, *bin_counts.tolist()],
    index=["Stable negative", *bin_labels],
    dtype=int,
)

overall_percentages = 100 * overall_counts / cohort_total
positive_percentages = 100 * bin_counts / positive_total

colors_left = [
    "#4C78A8",
    "#F58518",
    "#ECA02C",
    "#D45087",
    "#B279A2",
]

colors_right = [
    "#F58518",
    "#ECA02C",
    "#D45087",
    "#B279A2",
]

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

ax_overall = axes[0]

overall_bars = ax_overall.bar(
    overall_counts.index,
    overall_counts.values,
    color=colors_left,
    edgecolor="#111111",
    linewidth=1.0,
    width=0.80,
)

ax_overall.set_title(
    "Overall cohort composition",
    fontsize=16,
    fontweight="bold",
    pad=12,
)

ax_overall.set_ylabel(
    "Number of samples",
    fontsize=16,
    fontweight="bold",
)

ax_overall.set_ylim(
    0,
    overall_counts.max() * 1.22,
)

ax_overall.yaxis.set_major_locator(
    MaxNLocator(nbins=6, integer=True)
)

ax_overall.yaxis.set_major_formatter(
    FuncFormatter(
        lambda value, position: f"{int(value):,}"
    )
)

ax_overall.grid(
    axis="y",
    linewidth=0.7,
    alpha=0.3,
    color="#B0B0B0",
)

ax_overall.set_axisbelow(True)

for tick in ax_overall.get_xticklabels():
    tick.set_fontweight("bold")

for tick in ax_overall.get_yticklabels():
    tick.set_fontweight("bold")

for bar, count, percentage in zip(
    overall_bars,
    overall_counts.values,
    overall_percentages.values,
):
    ax_overall.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + overall_counts.max() * 0.025,
        f"{count:,}\n({percentage:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#111111",
        linespacing=1.1,
    )

ax_positive = axes[1]

positive_bars = ax_positive.bar(
    bin_counts.index,
    bin_counts.values,
    color=colors_right,
    edgecolor="#111111",
    linewidth=1.0,
    width=0.80,
)

ax_positive.set_title(
    "Positive samples by event-time window",
    fontsize=16,
    fontweight="bold",
    pad=12,
)

ax_positive.set_ylabel(
    "Number of positive samples",
    fontsize=16,
    fontweight="bold",
)

ax_positive.set_ylim(
    0,
    bin_counts.max() * 1.25,
)

ax_positive.yaxis.set_major_locator(
    MaxNLocator(nbins=6, integer=True)
)

ax_positive.yaxis.set_major_formatter(
    FuncFormatter(
        lambda value, position: f"{int(value):,}"
    )
)

ax_positive.grid(
    axis="y",
    linewidth=0.7,
    alpha=0.3,
    color="#B0B0B0",
)

ax_positive.set_axisbelow(True)

for tick in ax_positive.get_xticklabels():
    tick.set_fontweight("bold")

for tick in ax_positive.get_yticklabels():
    tick.set_fontweight("bold")

for bar, count, percentage in zip(
    positive_bars,
    bin_counts.values,
    positive_percentages.values,
):
    ax_positive.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + bin_counts.max() * 0.025,
        f"{count:,}\n({percentage:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#111111",
        linespacing=1.1,
    )

fig.suptitle(
    "Class Imbalance and Positive Event-Time Distribution",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)


fig.subplots_adjust(
    left=0.07,
    right=0.98,
    bottom=0.12,
    top=0.82,
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

print("Cohort file:", cohort_path)
print("Label column:", label_col)
print("Event-time column:", hours_col)
print("Stable negatives:", f"{negative_total:,}")
print("Positive samples:", f"{positive_total:,}")
print("Saved:", PNG_PATH)
print("Saved:", PDF_PATH)
