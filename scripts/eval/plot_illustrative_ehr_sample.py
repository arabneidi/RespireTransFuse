"""Visualize the temporal coverage and values of a representative EHR sample."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


TEXT_SCALE_APPLIED = 1.5

BASE = Path("/content/drive/MyDrive/respire-transfuse")

COHORT_PATH = BASE / "data" / "processed" / "cohorts" / "cohort.csv"

EHR_PATH = (
    BASE
    / "data"
    / "processed"
    / "ehr"
    / "ehr_final_24h_train_ready"
    / "ehr_24h_final_train_ready_current_split.npz"
)

OUT_DIR = BASE / "outputs" / "thesis_figures" / "dataset_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_CANDIDATES_PATH = OUT_DIR / "illustrative_ehr_sample_candidates.csv"
PNG_PATH = OUT_DIR / "12_illustrative_ehr_sample_broad_temporal_coverage.png"
PDF_PATH = OUT_DIR / "12_illustrative_ehr_sample_broad_temporal_coverage.pdf"

TARGET_SPLIT = "train"
TARGET_LABEL = 1

MIN_DISTINCT_HOURS = 12
MIN_DISTINCT_FEATURES = 14
MIN_HOURS_PER_THIRD = 2
MIN_ALGORITHMIC_FEATURES = 2


def decode_text_array(values):
    decoded = []

    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            decoded.append(
                value.decode(
                    "utf-8",
                    errors="replace",
                )
            )
        else:
            decoded.append(str(value))

    return np.asarray(decoded, dtype=str)


def normalize_split(values):
    return (
        pd.Series(decode_text_array(values))
        .str.strip()
        .str.lower()
        .replace(
            {
                "training": "train",
                "validation": "val",
                "valid": "val",
                "dev": "val",
                "testing": "test",
            }
        )
        .to_numpy(dtype=str)
    )


for path in [COHORT_PATH, EHR_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

cohort = pd.read_csv(COHORT_PATH)

if "sample_id" not in cohort.columns:
    raise KeyError(
        f"sample_id was not found. Available columns: {cohort.columns.tolist()}"
    )

cohort["sample_id"] = cohort["sample_id"].astype(str)

with np.load(EHR_PATH, allow_pickle=True) as z:
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int64)
    split = normalize_split(z["split"])
    sample_ids = decode_text_array(z["sample_id"])

    if "M" in z.files:
        M = z["M"].astype(np.float32)
    elif "mask" in z.files:
        M = z["mask"].astype(np.float32)
    else:
        raise RuntimeError("The NPZ contains neither M nor mask.")

    if "labels" in z.files:
        feature_names = decode_text_array(z["labels"])
    elif "feature_names" in z.files:
        feature_names = decode_text_array(z["feature_names"])
    else:
        raise RuntimeError("The NPZ does not contain feature labels.")

if X.shape != M.shape:
    raise RuntimeError(
        f"X and mask shapes differ: {X.shape} and {M.shape}"
    )

if X.shape[1:] != (24, 30):
    raise RuntimeError(
        f"Expected an EHR shape of [N, 24, 30], found {X.shape}."
    )

if len(y) != len(X):
    raise RuntimeError("The number of labels does not match the EHR samples.")

if len(split) != len(X):
    raise RuntimeError("The number of split values does not match the EHR samples.")

if len(sample_ids) != len(X):
    raise RuntimeError("The number of sample IDs does not match the EHR samples.")

if len(feature_names) != X.shape[2]:
    raise RuntimeError(
        f"Expected {X.shape[2]} feature names, found {len(feature_names)}."
    )

candidate_mask = split == TARGET_SPLIT

if TARGET_LABEL is not None:
    candidate_mask &= y == TARGET_LABEL

candidate_indices = np.where(candidate_mask)[0]

if len(candidate_indices) == 0:
    raise RuntimeError(
        "No samples matched the requested split and label."
    )

candidate_observation_mask = M[candidate_indices] > 0

observed_cells = candidate_observation_mask.sum(axis=(1, 2))
hour_presence = candidate_observation_mask.any(axis=2)
feature_presence = candidate_observation_mask.any(axis=1)

distinct_hours = hour_presence.sum(axis=1)
distinct_features = feature_presence.sum(axis=1)

early_hours = hour_presence[:, 0:8].sum(axis=1)
middle_hours = hour_presence[:, 8:16].sum(axis=1)
late_hours = hour_presence[:, 16:24].sum(axis=1)

early_cells = candidate_observation_mask[:, 0:8, :].sum(axis=(1, 2))
middle_cells = candidate_observation_mask[:, 8:16, :].sum(axis=(1, 2))
late_cells = candidate_observation_mask[:, 16:24, :].sum(axis=(1, 2))

clinical_features = feature_presence[:, :20].sum(axis=1)
algorithmic_features = feature_presence[:, 20:].sum(axis=1)

balanced_third_hours = np.minimum.reduce(
    [
        early_hours,
        middle_hours,
        late_hours,
    ]
)

temporal_span = np.zeros(
    len(candidate_indices),
    dtype=int,
)

for row_index in range(len(candidate_indices)):
    observed_hour_indices = np.where(
        hour_presence[row_index]
    )[0]

    if len(observed_hour_indices) > 0:
        temporal_span[row_index] = (
            observed_hour_indices[-1]
            - observed_hour_indices[0]
            + 1
        )

cell_score = observed_cells / float(24 * 30)
hour_score = distinct_hours / 24.0
feature_score = distinct_features / 30.0
balance_score = balanced_third_hours / 8.0
span_score = temporal_span / 24.0

branch_score = (
    0.5 * clinical_features / 20.0
    + 0.5 * algorithmic_features / 10.0
)

selection_score = (
    0.25 * cell_score
    + 0.25 * hour_score
    + 0.20 * feature_score
    + 0.15 * balance_score
    + 0.10 * span_score
    + 0.05 * branch_score
)

eligible = (
    (distinct_hours >= MIN_DISTINCT_HOURS)
    & (distinct_features >= MIN_DISTINCT_FEATURES)
    & (early_hours >= MIN_HOURS_PER_THIRD)
    & (middle_hours >= MIN_HOURS_PER_THIRD)
    & (late_hours >= MIN_HOURS_PER_THIRD)
    & (algorithmic_features >= MIN_ALGORITHMIC_FEATURES)
)

selection_rule = "Strict broad-coverage criteria"

if not eligible.any():
    eligible = (
        (distinct_hours >= 10)
        & (distinct_features >= 12)
        & (early_hours >= 2)
        & (middle_hours >= 2)
        & (late_hours >= 2)
        & (algorithmic_features >= 1)
    )

    selection_rule = "Relaxed broad-coverage criteria"

if not eligible.any():
    eligible = np.ones(
        len(candidate_indices),
        dtype=bool,
    )

    selection_rule = "Highest overall coverage score"

candidate_table = pd.DataFrame(
    {
        "npz_index": candidate_indices,
        "sample_id": sample_ids[candidate_indices],
        "split": split[candidate_indices],
        "label": y[candidate_indices],
        "selection_score": selection_score,
        "observed_cells": observed_cells,
        "observed_fraction": observed_cells / 720.0,
        "distinct_hours": distinct_hours,
        "early_hours": early_hours,
        "middle_hours": middle_hours,
        "late_hours": late_hours,
        "early_cells": early_cells,
        "middle_cells": middle_cells,
        "late_cells": late_cells,
        "temporal_span": temporal_span,
        "distinct_features": distinct_features,
        "clinical_features": clinical_features,
        "algorithmic_features": algorithmic_features,
        "eligible": eligible,
    }
)

ranked_candidates = (
    candidate_table[candidate_table["eligible"]]
    .sort_values(
        [
            "selection_score",
            "distinct_hours",
            "distinct_features",
            "observed_cells",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

ranked_candidates.head(50).to_csv(
    TOP_CANDIDATES_PATH,
    index=False,
)

selected_candidate = ranked_candidates.iloc[0]

sample_index = int(selected_candidate["npz_index"])
sample_id = str(selected_candidate["sample_id"])
sample_label = int(selected_candidate["label"])
sample_split = str(selected_candidate["split"])

normalized_sample = X[sample_index].copy()
mask_sample = M[sample_index].copy()

event_column = next(
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
        if column in cohort.columns
    ),
    None,
)

event_hours = np.nan

if event_column is not None:
    cohort_matches = cohort[
        cohort["sample_id"] == sample_id
    ]

    if not cohort_matches.empty:
        possible_event_hours = pd.to_numeric(
            cohort_matches[event_column],
            errors="coerce",
        ).dropna()

        if not possible_event_hours.empty:
            event_hours = float(
                possible_event_hours.iloc[0]
            )

observed_values = normalized_sample[
    mask_sample > 0
]

observed_values = observed_values[
    np.isfinite(observed_values)
]

if len(observed_values) == 0:
    raise RuntimeError(
        "The selected sample has no finite observed values."
    )

robust_limit = float(
    np.percentile(
        np.abs(observed_values),
        98,
    )
)

robust_limit = min(
    max(robust_limit, 1.0),
    5.0,
)

plot_matrix = normalized_sample.T.copy()
plot_mask = mask_sample.T <= 0

masked_matrix = np.ma.masked_where(
    plot_mask,
    plot_matrix,
)

cmap = plt.get_cmap("coolwarm").copy()
cmap.set_bad("#F2F2F2")

normalizer = TwoSlopeNorm(
    vmin=-robust_limit,
    vcenter=0.0,
    vmax=robust_limit,
)

observed_hours_per_feature = (
    (mask_sample > 0)
    .sum(axis=0)
    .astype(int)
)

outcome_text = (
    "Positive Outcome"
    if sample_label == 1
    else "Negative Outcome"
)

split_text = {
    "train": "Training Split",
    "val": "Validation Split",
    "test": "Test Split",
}.get(
    sample_split,
    f"{sample_split.title()} Split",
)

event_text = ""

if np.isfinite(event_hours):
    event_text = (
        f" | Event {event_hours:.1f} Hours After the Index Radiograph"
    )

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 21,
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "axes.labelsize": 24,
        "axes.labelweight": "bold",
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "axes.edgecolor": "#111111",
        "axes.linewidth": 1.0,
        "savefig.facecolor": "white",
    }
)

plt.close("all")

fig = plt.figure(
    figsize=(22, 15),
)

fig.patch.set_facecolor("white")

grid = fig.add_gridspec(
    1,
    2,
    width_ratios=[5.8, 1.45],
    wspace=0.08,
)

ax_matrix = fig.add_subplot(
    grid[0, 0]
)

ax_availability = fig.add_subplot(
    grid[0, 1],
    sharey=ax_matrix,
)

image = ax_matrix.imshow(
    masked_matrix,
    aspect="auto",
    interpolation="nearest",
    cmap=cmap,
    norm=normalizer,
)

major_hour_positions = [
    0,
    3,
    6,
    9,
    12,
    15,
    18,
    21,
    23,
]

major_hour_labels = [
    "−24",
    "−21",
    "−18",
    "−15",
    "−12",
    "−9",
    "−6",
    "−3",
    "−1",
]

ax_matrix.set_xticks(
    major_hour_positions
)

ax_matrix.set_xticklabels(
    major_hour_labels,
    fontsize=21,
    fontweight="bold",
)

ax_matrix.set_xticks(
    np.arange(-0.5, 24, 1),
    minor=True,
)

ax_matrix.set_yticks(
    np.arange(len(feature_names))
)

ax_matrix.set_yticklabels(
    feature_names,
    fontsize=21,
    fontweight="bold",
)

ax_matrix.grid(
    which="minor",
    color="white",
    linewidth=0.55,
)

ax_matrix.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

ax_matrix.set_xlabel(
    "Hours Before the Index Chest Radiograph",
    fontsize=24,
    fontweight="bold",
)

ax_matrix.set_ylabel(
    "EHR Feature",
    fontsize=24,
    fontweight="bold",
)

ax_matrix.set_title(
    "Training-Standardized Feature Values",
    fontsize=24,
    fontweight="bold",
    pad=12,
)

availability_bars = ax_availability.barh(
    np.arange(len(feature_names)),
    observed_hours_per_feature,
    height=0.68,
    color="#4C78A8",
    edgecolor="#111111",
    linewidth=0.8,
)

ax_availability.set_xlim(
    0,
    26,
)

ax_availability.set_xticks(
    [0, 6, 12, 18, 24]
)

ax_availability.set_xticklabels(
    [0, 6, 12, 18, 24],
    fontsize=21,
    fontweight="bold",
)

ax_availability.set_xlabel(
    "Observed Hourly Bins",
    fontsize=24,
    fontweight="bold",
)

ax_availability.set_title(
    "Feature Availability",
    fontsize=24,
    fontweight="bold",
    pad=12,
)

ax_availability.tick_params(
    axis="y",
    labelleft=False,
)

ax_availability.grid(
    axis="x",
    alpha=0.3,
    linewidth=0.7,
    color="#B0B0B0",
)

ax_availability.set_axisbelow(True)



colorbar = fig.colorbar(
    image,
    ax=[ax_matrix, ax_availability],
    fraction=0.018,
    pad=0.018,
)

colorbar.set_label(
    "Training-Standardized Observed Value",
    fontsize=24,
    fontweight="bold",
)

colorbar.ax.tick_params(
    labelsize=21,
)

for tick in colorbar.ax.get_yticklabels():
    tick.set_fontweight("bold")

fig.suptitle(
    "Illustrative 24-Hour EHR Sample",
    fontsize=30,
    fontweight="bold",
    y=0.985,
)


fig.subplots_adjust(
    top=0.92,
    bottom=0.08,
    left=0.26,
    right=0.91,
)

fig.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.10,
)

fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.10,
)

plt.show()

print("Selected sample ID:", sample_id)
print("Selected NPZ index:", sample_index)
print("Selection rule:", selection_rule)
print("Outcome:", outcome_text)
print("Split:", sample_split)
print(
    "Observed cells:",
    (
        f"{int(selected_candidate['observed_cells'])}/720 "
        f"({selected_candidate['observed_fraction'] * 100:.2f}%)"
    ),
)
print(
    "Distinct observed hours:",
    int(selected_candidate["distinct_hours"]),
)
print(
    "Distinct observed features:",
    int(selected_candidate["distinct_features"]),
)
print(
    "Algorithmic features observed:",
    int(selected_candidate["algorithmic_features"]),
)
print("Candidate table:", TOP_CANDIDATES_PATH)
print("Saved:", PNG_PATH)
print("Saved:", PDF_PATH)
