"""Compare probability quality, calibration, and risk strata across seven models."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scipy.stats import beta
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


BASE = Path("/content/drive/MyDrive/respire-transfuse")

OUTPUT_DIR = (
    BASE
    / "outputs"
    / "seven_model_probability_and_risk_analysis"
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

LABEL_COLUMNS = [
    "label",
    "labels",
    "target",
    "targets",
    "y_true",
    "gt",
]

N_CALIBRATION_BINS = 10
N_BOOTSTRAPS = 5000
SEED = 42

MAIN_TITLE_FONT_SIZE = 20
PLOT_TEXT_FONT_SIZE = 16
LEGEND_FONT_SIZE = 17
TICK_FONT_SIZE = 14
ANNOTATION_FONT_SIZE = 13


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
            f"{model_name}: predictions contain invalid values."
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


def clopper_pearson_interval(
    positives,
    total,
    confidence=0.95,
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


def calibration_metrics(
    labels,
    probabilities,
    n_bins=N_CALIBRATION_BINS,
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
                "Calibration difference": (
                    observed_rate
                    - mean_probability
                ),
                "Absolute calibration error": abs(
                    observed_rate
                    - mean_probability
                ),
                "Minimum probability": float(
                    bin_probabilities.min()
                ),
                "Maximum probability": float(
                    bin_probabilities.max()
                ),
            }
        )

    table = pd.DataFrame(rows)

    weights = (
        table["N"]
        / table["N"].sum()
    )

    ece = float(
        (
            weights
            * table[
                "Absolute calibration error"
            ]
        ).sum()
    )

    return table, ece


def bootstrap_brier_interval(
    labels,
    probabilities,
    n_bootstraps=N_BOOTSTRAPS,
    seed=SEED,
):
    rng = np.random.default_rng(seed)

    positive_indices = np.flatnonzero(
        labels == 1
    )

    negative_indices = np.flatnonzero(
        labels == 0
    )

    scores = np.empty(
        n_bootstraps,
        dtype=float,
    )

    for bootstrap_index in range(
        n_bootstraps
    ):
        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )

        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [
                sampled_positive,
                sampled_negative,
            ]
        )

        scores[bootstrap_index] = (
            brier_score_loss(
                labels[sampled_indices],
                probabilities[sampled_indices],
            )
        )

    lower = float(
        np.percentile(
            scores,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            scores,
            97.5,
        )
    )

    return lower, upper


def build_risk_tertiles(
    model_name,
    labels,
    probabilities,
):
    frame = pd.DataFrame(
        {
            "label": labels,
            "probability": probabilities,
        }
    )

    frame["risk_group"] = pd.qcut(
        frame["probability"].rank(
            method="first"
        ),
        q=3,
        labels=[
            "Low risk",
            "Intermediate risk",
            "High risk",
        ],
    )

    total_positives = int(
        labels.sum()
    )

    prevalence = float(
        labels.mean()
    )

    rows = []

    for risk_group, group in frame.groupby(
        "risk_group",
        observed=False,
    ):
        total = int(
            len(group)
        )

        positives = int(
            group["label"].sum()
        )

        negatives = (
            total
            - positives
        )

        observed_rate = (
            positives
            / total
        )

        mean_probability = float(
            group["probability"].mean()
        )

        ci_lower, ci_upper = (
            clopper_pearson_interval(
                positives,
                total,
            )
        )

        rows.append(
            {
                "Model": model_name,
                "Risk group": str(risk_group),
                "N": total,
                "Positives": positives,
                "Negatives": negatives,
                "Observed event rate": observed_rate,
                "CI lower": ci_lower,
                "CI upper": ci_upper,
                "Mean predicted probability": mean_probability,
                "Observed minus predicted": (
                    observed_rate
                    - mean_probability
                ),
                "Enrichment over prevalence": (
                    observed_rate
                    / prevalence
                ),
                "Share of all positives": (
                    positives
                    / total_positives
                ),
                "Minimum predicted probability": float(
                    group["probability"].min()
                ),
                "Maximum predicted probability": float(
                    group["probability"].max()
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    group_order = [
        "Low risk",
        "Intermediate risk",
        "High risk",
    ]

    result["Risk group"] = pd.Categorical(
        result["Risk group"],
        categories=group_order,
        ordered=True,
    )

    return (
        result
        .sort_values("Risk group")
        .reset_index(drop=True)
    )


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

common_prevalence = float(
    reference_positives
    / reference_size
)

null_brier = float(
    common_prevalence
    * (
        1.0
        - common_prevalence
    )
)

null_log_loss = float(
    -(
        common_prevalence
        * np.log(common_prevalence)
        + (
            1.0
            - common_prevalence
        )
        * np.log(
            1.0
            - common_prevalence
        )
    )
)

metric_rows = []
risk_tables = []
calibration_tables = []

for model_index, (
    model_name,
    data,
) in enumerate(
    loaded.items()
):
    labels = data["labels"]
    probabilities = data[
        "probabilities"
    ]

    auroc = float(
        roc_auc_score(
            labels,
            probabilities,
        )
    )

    auprc = float(
        average_precision_score(
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

    brier_ci_lower, brier_ci_upper = (
        bootstrap_brier_interval(
            labels,
            probabilities,
            seed=SEED + model_index,
        )
    )

    model_log_loss = float(
        log_loss(
            labels,
            probabilities,
            labels=[0, 1],
        )
    )

    positive_probabilities = probabilities[
        labels == 1
    ]

    negative_probabilities = probabilities[
        labels == 0
    ]

    positive_brier = float(
        np.mean(
            (
                1.0
                - positive_probabilities
            )
            ** 2
        )
    )

    negative_brier = float(
        np.mean(
            negative_probabilities
            ** 2
        )
    )

    balanced_brier = float(
        0.5
        * (
            positive_brier
            + negative_brier
        )
    )

    calibration_table, ece = (
        calibration_metrics(
            labels,
            probabilities,
        )
    )

    calibration_table.insert(
        0,
        "Model",
        model_name,
    )

    calibration_tables.append(
        calibration_table
    )

    risk_table = build_risk_tertiles(
        model_name,
        labels,
        probabilities,
    )

    risk_tables.append(
        risk_table
    )

    low_row = risk_table[
        risk_table["Risk group"]
        == "Low risk"
    ].iloc[0]

    intermediate_row = risk_table[
        risk_table["Risk group"]
        == "Intermediate risk"
    ].iloc[0]

    high_row = risk_table[
        risk_table["Risk group"]
        == "High risk"
    ].iloc[0]

    if low_row["Observed event rate"] > 0:
        high_low_ratio = (
            high_row["Observed event rate"]
            / low_row["Observed event rate"]
        )
    else:
        high_low_ratio = np.inf

    metric_rows.append(
        {
            "Model": model_name,
            "Prediction source": data["source"],
            "Test N": int(len(labels)),
            "Test positives": int(labels.sum()),
            "Prevalence": common_prevalence,
            "AUROC": auroc,
            "AUPRC": auprc,
            "Brier score": brier,
            "Brier CI lower": brier_ci_lower,
            "Brier CI upper": brier_ci_upper,
            "Null Brier": null_brier,
            "Brier skill score": (
                1.0
                - brier
                / null_brier
            ),
            "Positive-class Brier": positive_brier,
            "Negative-class Brier": negative_brier,
            "Balanced Brier": balanced_brier,
            "Log loss": model_log_loss,
            "Null log loss": null_log_loss,
            "Log-loss skill": (
                1.0
                - model_log_loss
                / null_log_loss
            ),
            "ECE": ece,
            "Mean predicted probability": float(
                probabilities.mean()
            ),
            "Positive mean probability": float(
                positive_probabilities.mean()
            ),
            "Negative mean probability": float(
                negative_probabilities.mean()
            ),
            "Low-risk observed rate": float(
                low_row["Observed event rate"]
            ),
            "Intermediate-risk observed rate": float(
                intermediate_row[
                    "Observed event rate"
                ]
            ),
            "High-risk observed rate": float(
                high_row["Observed event rate"]
            ),
            "High-risk enrichment": float(
                high_row[
                    "Enrichment over prevalence"
                ]
            ),
            "High-to-low risk ratio": float(
                high_low_ratio
            ),
            "High-risk positive capture": float(
                high_row[
                    "Share of all positives"
                ]
            ),
            "High-risk positives": int(
                high_row["Positives"]
            ),
        }
    )

metrics = pd.DataFrame(
    metric_rows
)

metrics = (
    metrics
    .sort_values(
        "Brier score",
        ascending=True,
    )
    .reset_index(drop=True)
)

metrics.insert(
    0,
    "Rank by Brier",
    np.arange(
        1,
        len(metrics) + 1,
    ),
)

risk_tertiles = pd.concat(
    risk_tables,
    ignore_index=True,
)

calibration_bins = pd.concat(
    calibration_tables,
    ignore_index=True,
)

risk_summary = metrics[
    [
        "Model",
        "Prevalence",
        "Low-risk observed rate",
        "Intermediate-risk observed rate",
        "High-risk observed rate",
        "High-risk enrichment",
        "High-to-low risk ratio",
        "High-risk positive capture",
        "High-risk positives",
    ]
].copy()

metrics_csv_path = (
    OUTPUT_DIR
    / "seven_model_probability_metrics.csv"
)

risk_csv_path = (
    OUTPUT_DIR
    / "seven_model_risk_tertiles.csv"
)

risk_summary_csv_path = (
    OUTPUT_DIR
    / "seven_model_risk_summary.csv"
)

calibration_csv_path = (
    OUTPUT_DIR
    / "seven_model_calibration_bins.csv"
)

excel_path = (
    OUTPUT_DIR
    / "seven_model_probability_and_risk_analysis.xlsx"
)

metrics.to_csv(
    metrics_csv_path,
    index=False,
)

risk_tertiles.to_csv(
    risk_csv_path,
    index=False,
)

risk_summary.to_csv(
    risk_summary_csv_path,
    index=False,
)

calibration_bins.to_csv(
    calibration_csv_path,
    index=False,
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl",
) as writer:
    metrics.to_excel(
        writer,
        sheet_name="Probability Metrics",
        index=False,
    )

    risk_summary.to_excel(
        writer,
        sheet_name="Risk Summary",
        index=False,
    )

    risk_tertiles.to_excel(
        writer,
        sheet_name="Risk Tertiles",
        index=False,
    )

    calibration_bins.to_excel(
        writer,
        sheet_name="Calibration Bins",
        index=False,
    )

model_order = list(
    MODEL_SPECS.keys()
)

figure, axes = plt.subplots(
    4,
    2,
    figsize=(15, 22),
    sharey=True,
)

axes = axes.flatten()

group_order = [
    "Low risk",
    "Intermediate risk",
    "High risk",
]

x = np.arange(
    len(group_order)
)

for axis, model_name in zip(
    axes,
    model_order,
):
    model_table = risk_tertiles[
        risk_tertiles["Model"]
        == model_name
    ].copy()

    model_table[
        "Risk group"
    ] = pd.Categorical(
        model_table["Risk group"],
        categories=group_order,
        ordered=True,
    )

    model_table = model_table.sort_values(
        "Risk group"
    )

    observed_rates = (
        model_table[
            "Observed event rate"
        ].to_numpy()
        * 100.0
    )

    predicted_rates = (
        model_table[
            "Mean predicted probability"
        ].to_numpy()
        * 100.0
    )

    lower_errors = (
        model_table[
            "Observed event rate"
        ]
        - model_table[
            "CI lower"
        ]
    ).to_numpy() * 100.0

    upper_errors = (
        model_table[
            "CI upper"
        ]
        - model_table[
            "Observed event rate"
        ]
    ).to_numpy() * 100.0

    axis.errorbar(
        x,
        observed_rates,
        yerr=np.vstack(
            [
                lower_errors,
                upper_errors,
            ]
        ),
        marker="o",
        linewidth=2,
        markersize=7,
        capsize=5,
        label="Observed rate",
    )

    axis.plot(
        x,
        predicted_rates,
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label="Mean prediction",
    )

    axis.axhline(
        common_prevalence * 100.0,
        color="black",
        linestyle=":",
        linewidth=1.6,
        label="Overall prevalence",
    )

    axis.set_xticks(
        x,
        [
            "Low",
            "Intermediate",
            "High",
        ],
    )

    axis.set_title(
        model_name,
        fontsize=PLOT_TEXT_FONT_SIZE,
        fontweight="bold",
    )

    axis.set_xlabel(
        "Predicted-risk tertile",
        fontsize=PLOT_TEXT_FONT_SIZE,
        fontweight="bold",
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    set_tick_style(
        axis
    )

    for row_index, row in model_table.reset_index(
        drop=True
    ).iterrows():
        axis.annotate(
            (
                f"{row['Observed event rate'] * 100:.1f}%\n"
                f"{int(row['Positives'])}/{int(row['N'])}"
            ),
            (
                row_index,
                row[
                    "Observed event rate"
                ]
                * 100.0,
            ),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight="bold",
        )

# Seven models occupy seven of the eight panels.
axes[7].axis("off")

for axis in [
    axes[0],
    axes[2],
    axes[4],
    axes[6],
]:
    axis.set_ylabel(
        "Observed deterioration rate (%)",
        fontsize=PLOT_TEXT_FONT_SIZE,
        fontweight="bold",
    )

handles, labels = axes[0].get_legend_handles_labels()

figure.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=3,
    prop={
        "size": LEGEND_FONT_SIZE,
        "weight": "bold",
    },
    markerscale=1.45,
    handlelength=2.4,
    handletextpad=0.8,
    columnspacing=2.0,
    borderpad=0.55,
)

figure.suptitle(
    "Observed Deterioration Across Predicted-Risk Tertiles",
    fontsize=MAIN_TITLE_FONT_SIZE,
    fontweight="bold",
)

figure.tight_layout(
    rect=[
        0,
        0.01,
        1,
        0.89,
    ],
    h_pad=4.0,
)

figure.canvas.draw()

first_separator_y = (
    min(
        axes[0].get_position().y0,
        axes[1].get_position().y0,
    )
    + max(
        axes[2].get_position().y1,
        axes[3].get_position().y1,
    )
) / 2.0

second_separator_y = (
    min(
        axes[2].get_position().y0,
        axes[3].get_position().y0,
    )
    + max(
        axes[4].get_position().y1,
        axes[5].get_position().y1,
    )
) / 2.0

third_separator_y = (
    min(
        axes[4].get_position().y0,
        axes[5].get_position().y0,
    )
    + max(
        axes[6].get_position().y1,
        axes[7].get_position().y1,
    )
) / 2.0

for separator_y in [
    first_separator_y,
    second_separator_y,
    third_separator_y,
]:
    figure.add_artist(
        Line2D(
            [0.04, 0.96],
            [separator_y, separator_y],
            transform=figure.transFigure,
            color="#b5b5b5",
            linewidth=1.5,
        )
    )

risk_png_path = (
    OUTPUT_DIR
    / "seven_model_risk_tertiles_4x2.png"
)

risk_pdf_path = (
    OUTPUT_DIR
    / "seven_model_risk_tertiles_4x2.pdf"
)

figure.savefig(
    risk_png_path,
    dpi=300,
    bbox_inches="tight",
)

figure.savefig(
    risk_pdf_path,
    bbox_inches="tight",
)

plt.show()
plt.close(figure)

display_columns = [
    "Rank by Brier",
    "Model",
    "Brier score",
    "Brier CI lower",
    "Brier CI upper",
    "Null Brier",
    "Brier skill score",
    "Balanced Brier",
    "Log loss",
    "ECE",
    "AUROC",
    "AUPRC",
]

print()
print(
    metrics[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "Brier score": lambda value: f"{value:.5f}",
            "Brier CI lower": lambda value: f"{value:.5f}",
            "Brier CI upper": lambda value: f"{value:.5f}",
            "Null Brier": lambda value: f"{value:.5f}",
            "Brier skill score": lambda value: f"{value:.5f}",
            "Balanced Brier": lambda value: f"{value:.5f}",
            "Log loss": lambda value: f"{value:.5f}",
            "ECE": lambda value: f"{value:.5f}",
            "AUROC": lambda value: f"{value:.5f}",
            "AUPRC": lambda value: f"{value:.5f}",
        },
    )
)

print()
print("Common test cohort")
print("N:", reference_size)
print("Positives:", reference_positives)
print("Prevalence:", common_prevalence)
print("Null Brier:", null_brier)
print("ECE bins:", N_CALIBRATION_BINS)
print("Brier bootstrap samples:", N_BOOTSTRAPS)

print()
print("Saved:")
print(metrics_csv_path)
print(risk_csv_path)
print(risk_summary_csv_path)
print(calibration_csv_path)
print(excel_path)
print(risk_png_path)
print(risk_pdf_path)
