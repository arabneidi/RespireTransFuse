#!/usr/bin/env python3
"""Coordinate the complete cohort and EHR preprocessing workflow.

This cross-platform launcher checks the required MIMIC-IV and MIMIC-CXR inputs,
runs each cohort construction and feature-engineering stage in dependency order,
and validates the schema, split integrity, and sample alignment of every output.
Command-line options support clean rebuilds, resumable runs, and validation-only
checks while keeping all generated artifacts under the selected repository root.
"""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

RANDOM_SEED = 42

SPLIT_SEARCH_ITERATIONS = 50000
MAX_PREVALENCE_GAP = 0.0025
MAX_SPLIT_SIZE_DEVIATION = 0.01

EHR_WINDOW_HOURS = 24
EHR_CHUNKSIZE = 750000
MAX_CHART_FEATURES = 80
MAX_LAB_FEATURES = 80

CLINICAL_BOOTSTRAPS = 80
BROAD_BOOTSTRAPS = 80
BROAD_ELASTIC_TOP_K = 120
MIN_TRAIN_SAMPLE_COVERAGE = 0.005


def section(title):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)
    print()


def require_file(path):
    path = Path(path)

    if not path.exists():
        raise RuntimeError(
            f"Required output was not created: {path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"Required output is empty: {path}"
        )

    return path



def preflight_inputs(base):
    required_files = {
        "CXR metadata": (
            base
            / "data/raw/mimic_cxr/metadata"
            / "mimic-cxr-2.0.0-metadata.csv.gz"
        ),
        "ICU stays": (
            base
            / "data/raw/mimiciv/icu"
            / "icustays.csv.gz"
        ),
        "ICU item dictionary": (
            base
            / "data/raw/mimiciv/icu"
            / "d_items.csv.gz"
        ),
        "Procedure events": (
            base
            / "data/raw/mimiciv/icu"
            / "procedureevents.csv.gz"
        ),
        "Chart events": (
            base
            / "data/raw/mimiciv/icu"
            / "chartevents.csv.gz"
        ),
        "Lab events": (
            base
            / "data/raw/mimiciv/hosp"
            / "labevents.csv.gz"
        ),
        "Lab item dictionary": (
            base
            / "data/raw/mimiciv/hosp"
            / "d_labitems.csv.gz"
        ),
    }

    image_root = (
        base
        / "data/raw/mimic_cxr/images"
    )

    failures = []

    print()
    print("=" * 110)
    print("PREFLIGHT: REQUIRED INPUT FILES")
    print("=" * 110)

    for name, path in required_files.items():
        exists = path.is_file()
        nonempty = (
            exists
            and path.stat().st_size > 0
        )

        status = (
            "PASS"
            if nonempty
            else "FAIL"
        )

        size_gb = (
            path.stat().st_size
            / (1024 ** 3)
            if exists
            else 0.0
        )

        print(
            f"{status:4s} | "
            f"{name:24s} | "
            f"{size_gb:8.3f} GB | "
            f"{path}"
        )

        if not exists:
            failures.append(
                f"Missing file: {path}"
            )
        elif not nonempty:
            failures.append(
                f"Empty file: {path}"
            )

    print()
    print("=" * 110)
    print("PREFLIGHT: CXR IMAGE DIRECTORY")
    print("=" * 110)
    print(
        "SKIP | Image-file checking is deferred "
        "to build_cohort.py."
    )
    print(
        "PATH |",
        image_root,
    )

    if failures:
        print()
        print("=" * 110)
        print("PREFLIGHT FAILED")
        print("=" * 110)

        for failure in failures:
            print("-", failure)

        raise RuntimeError(
            f"Preprocessing cannot start because "
            f"{len(failures)} required input checks failed."
        )

    print()
    print("=" * 110)
    print("PREFLIGHT PASSED")
    print("=" * 110)
    print(
        "All required raw tabular files are available. "
        "Image checking is deferred to build_cohort.py."
    )


def npz_headers(path):
    path = require_file(path)
    output = {}

    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue

            key = Path(member).stem

            with archive.open(member, "r") as stream:
                version = np.lib.format.read_magic(stream)

                shape, fortran_order, dtype = (
                    np.lib.format._read_array_header(
                        stream,
                        version,
                    )
                )

            output[key] = {
                "shape": tuple(shape),
                "dtype": str(dtype),
                "fortran_order": bool(fortran_order),
            }

    return output


def require_npz_keys(headers, required, path):
    missing = [
        key
        for key in required
        if key not in headers
    ]

    if missing:
        raise RuntimeError(
            f"{path} is missing NPZ keys: {missing}"
        )


def load_cohort(base):
    cohort_path = (
        base
        / "data/processed/cohorts/cohort.csv"
    )

    require_file(cohort_path)

    cohort = pd.read_csv(cohort_path)

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
            f"Cohort is missing columns: {missing}"
        )

    cohort = cohort.copy()

    cohort["sample_id"] = (
        cohort["sample_id"]
        .astype(str)
    )

    cohort["subject_id"] = pd.to_numeric(
        cohort["subject_id"],
        errors="raise",
    ).astype(np.int64)

    cohort["label"] = pd.to_numeric(
        cohort["label"],
        errors="raise",
    ).astype(np.int64)

    cohort["split"] = (
        cohort["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if cohort["sample_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate sample_id values in cohort.csv."
        )

    if not set(
        cohort["label"].unique()
    ).issubset({0, 1}):
        raise RuntimeError(
            "Cohort labels are not binary."
        )

    return cohort


def validate_npz_alignment(base, npz_path):
    cohort = load_cohort(base)

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:
        for key in [
            "sample_id",
            "split",
            "y",
        ]:
            if key not in data.files:
                raise RuntimeError(
                    f"{npz_path} is missing {key}"
                )

        sample_ids = (
            data["sample_id"]
            .astype(str)
        )

        splits = (
            data["split"]
            .astype(str)
        )

        labels = (
            data["y"]
            .astype(np.int64)
        )

    npz_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "npz_split": splits,
            "npz_label": labels,
        }
    )

    if npz_df["sample_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate sample IDs in {npz_path}"
        )

    comparison = cohort[
        [
            "sample_id",
            "split",
            "label",
        ]
    ].merge(
        npz_df,
        on="sample_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    missing = comparison[
        comparison["_merge"] != "both"
    ]

    split_mismatch = comparison[
        comparison["split"]
        != comparison["npz_split"]
    ]

    label_mismatch = comparison[
        comparison["label"]
        != comparison["npz_label"]
    ]

    if len(missing) > 0:
        raise RuntimeError(
            f"Cohort/NPZ sample mismatch in "
            f"{npz_path}: {len(missing)} rows"
        )

    if len(split_mismatch) > 0:
        raise RuntimeError(
            f"Cohort/NPZ split mismatch in "
            f"{npz_path}: {len(split_mismatch)} rows"
        )

    if len(label_mismatch) > 0:
        raise RuntimeError(
            f"Cohort/NPZ label mismatch in "
            f"{npz_path}: {len(label_mismatch)} rows"
        )

    del cohort
    del npz_df
    del comparison
    gc.collect()


def find_json_values(obj, key):
    values = []

    if isinstance(obj, dict):
        for current_key, value in obj.items():
            if current_key == key:
                values.append(value)

            values.extend(
                find_json_values(
                    value,
                    key,
                )
            )

    elif isinstance(obj, list):
        for value in obj:
            values.extend(
                find_json_values(
                    value,
                    key,
                )
            )

    return values


def validate_cohort(
    base,
    seed,
    max_prevalence_gap,
    max_size_deviation,
):
    cohort_path = (
        base
        / "data/processed/cohorts/cohort.csv"
    )

    assignments_path = (
        base
        / "data/processed/cohorts"
        / "subject_split_assignments.csv"
    )

    manifest_path = (
        base
        / "data/processed/cohorts"
        / "cohort_manifest.json"
    )

    require_file(cohort_path)
    require_file(assignments_path)
    require_file(manifest_path)

    cohort = load_cohort(base)

    expected_splits = {
        "train",
        "val",
        "test",
    }

    actual_splits = set(
        cohort["split"].unique()
    )

    if actual_splits != expected_splits:
        raise RuntimeError(
            f"Unexpected cohort splits: "
            f"{sorted(actual_splits)}"
        )

    patient_split_counts = (
        cohort
        .groupby("subject_id")[
            "split"
        ]
        .nunique()
    )

    leaking_patients = int(
        (
            patient_split_counts > 1
        ).sum()
    )

    if leaking_patients != 0:
        raise RuntimeError(
            f"Patient leakage found for "
            f"{leaking_patients} patients."
        )

    summary = (
        cohort
        .groupby("split")
        .agg(
            rows=(
                "sample_id",
                "size",
            ),
            subjects=(
                "subject_id",
                "nunique",
            ),
            positives=(
                "label",
                "sum",
            ),
        )
        .reindex(
            [
                "train",
                "val",
                "test",
            ]
        )
    )

    if summary.isna().any().any():
        raise RuntimeError(
            "One or more cohort splits are empty."
        )

    summary["negatives"] = (
        summary["rows"]
        - summary["positives"]
    )

    summary["prevalence"] = (
        summary["positives"]
        / summary["rows"]
    )

    summary["row_fraction"] = (
        summary["rows"]
        / summary["rows"].sum()
    )

    prevalence_gap = float(
        summary["prevalence"].max()
        - summary["prevalence"].min()
    )

    target_fractions = np.array(
        [
            0.70,
            0.15,
            0.15,
        ],
        dtype=np.float64,
    )

    size_deviation = float(
        np.abs(
            summary[
                "row_fraction"
            ].to_numpy()
            - target_fractions
        ).max()
    )

    if prevalence_gap > float(
        max_prevalence_gap
    ):
        raise RuntimeError(
            f"Prevalence gap is too large: "
            f"{prevalence_gap:.8f}; "
            f"allowed={max_prevalence_gap:.8f}"
        )

    if size_deviation > float(
        max_size_deviation
    ):
        raise RuntimeError(
            f"Split-size deviation is too large: "
            f"{size_deviation:.8f}; "
            f"allowed={max_size_deviation:.8f}"
        )

    with open(
        assignments_path,
        "rb",
    ) as file:
        assignment_hash = hashlib.sha256(
            file.read()
        ).hexdigest()

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    algorithm = manifest.get(
        "split_algorithm",
        {},
    )

    manifest_hash = algorithm.get(
        "subject_assignments_sha256"
    )

    manifest_seed = algorithm.get(
        "seed",
        manifest.get(
            "random_seed"
        ),
    )

    if assignment_hash != manifest_hash:
        raise RuntimeError(
            "Subject-assignment SHA256 does not "
            "match the cohort manifest."
        )

    if int(manifest_seed) != int(seed):
        raise RuntimeError(
            f"Manifest seed is {manifest_seed}, "
            f"expected {seed}."
        )

    assignments = pd.read_csv(
        assignments_path
    )

    if assignments[
        "subject_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate subject IDs in split assignments."
        )

    cohort_subjects = set(
        cohort["subject_id"]
        .astype(int)
        .tolist()
    )

    assignment_subjects = set(
        pd.to_numeric(
            assignments["subject_id"],
            errors="raise",
        )
        .astype(int)
        .tolist()
    )

    if cohort_subjects != assignment_subjects:
        raise RuntimeError(
            "Cohort subjects and assignment-table "
            "subjects do not match."
        )

    print(summary.to_string())

    print(
        "\nMaximum prevalence gap:",
        f"{prevalence_gap:.8f}",
    )

    print(
        "Prevalence gap in percentage points:",
        f"{prevalence_gap * 100:.4f}",
    )

    print(
        "Maximum row-fraction deviation:",
        f"{size_deviation:.8f}",
    )

    print(
        "Patient leakage:",
        leaking_patients,
    )

    print(
        "Split seed:",
        int(seed),
    )

    print(
        "Subject assignment SHA256:",
        assignment_hash,
    )


def validate_candidate_tensor(
    base,
    npz_path,
    stage_name,
):
    npz_path = require_file(
        npz_path
    )

    headers = npz_headers(
        npz_path
    )

    required = [
        "X_raw",
        "mask",
        "y",
        "split",
        "sample_id",
        "variables",
        "labels",
        "sources",
        "itemids",
    ]

    require_npz_keys(
        headers,
        required,
        npz_path,
    )

    x_raw_shape = headers[
        "X_raw"
    ]["shape"]

    mask_shape = headers[
        "mask"
    ]["shape"]

    if len(x_raw_shape) != 3:
        raise RuntimeError(
            f"{stage_name}: expected X_raw [N,T,F], "
            f"got {x_raw_shape}"
        )

    if x_raw_shape != mask_shape:
        raise RuntimeError(
            f"{stage_name}: X_raw and mask "
            f"shapes differ: "
            f"{x_raw_shape} vs {mask_shape}"
        )

    if "X" in headers:
        x_shape = headers[
            "X"
        ]["shape"]

        if x_shape != x_raw_shape:
            raise RuntimeError(
                f"{stage_name}: X shape {x_shape} "
                f"does not match X_raw {x_raw_shape}"
            )

    n_samples, n_bins, n_features = (
        x_raw_shape
    )

    if n_samples <= 0:
        raise RuntimeError(
            f"{stage_name}: tensor has zero samples."
        )

    if n_bins != 24:
        raise RuntimeError(
            f"{stage_name}: expected 24 time bins, "
            f"got {n_bins}"
        )

    if n_features <= 0:
        raise RuntimeError(
            f"{stage_name}: tensor has zero features."
        )

    expected_sample_shapes = {
        "y": (n_samples,),
        "split": (n_samples,),
        "sample_id": (n_samples,),
    }

    for key, expected_shape in (
        expected_sample_shapes.items()
    ):
        actual_shape = headers[
            key
        ]["shape"]

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{stage_name}: {key} shape "
                f"{actual_shape} does not match "
                f"{expected_shape}"
            )

    expected_feature_shapes = {
        "variables": (n_features,),
        "labels": (n_features,),
        "sources": (n_features,),
        "itemids": (n_features,),
    }

    for key, expected_shape in (
        expected_feature_shapes.items()
    ):
        actual_shape = headers[
            key
        ]["shape"]

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{stage_name}: {key} shape "
                f"{actual_shape} does not match "
                f"{expected_shape}"
            )

    validate_npz_alignment(
        base,
        npz_path,
    )

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:
        split = (
            data["split"]
            .astype(str)
        )

        y = (
            data["y"]
            .astype(np.int64)
        )

        mask = (
            data["mask"]
            .astype(np.float32)
        )

    actual_splits = set(
        np.unique(split)
    )

    if actual_splits != {
        "train",
        "val",
        "test",
    }:
        raise RuntimeError(
            f"{stage_name}: invalid split values: "
            f"{sorted(actual_splits)}"
        )

    if not set(
        np.unique(y)
    ).issubset({0, 1}):
        raise RuntimeError(
            f"{stage_name}: labels are not binary."
        )

    if not np.isfinite(mask).all():
        raise RuntimeError(
            f"{stage_name}: mask contains "
            "non-finite values."
        )

    if not np.isin(
        mask,
        [0.0, 1.0],
    ).all():
        raise RuntimeError(
            f"{stage_name}: mask is not binary."
        )

    zero_ehr_samples = int(
        (
            mask.sum(
                axis=(1, 2)
            ) == 0
        ).sum()
    )

    print(
        "Tensor:",
        npz_path,
    )

    print(
        "Available NPZ keys:",
        sorted(
            headers.keys()
        ),
    )

    print(
        "Shape:",
        x_raw_shape,
    )

    print(
        "Split counts:",
        pd.Series(
            split
        ).value_counts().to_dict(),
    )

    print(
        "Label counts:",
        pd.Series(
            y
        ).value_counts().to_dict(),
    )

    print(
        "Samples with zero EHR:",
        zero_ehr_samples,
    )


def validate_filtered_tensor(
    base,
    input_npz,
    output_npz,
    features_csv,
    summary_json,
):
    validate_candidate_tensor(
        base,
        output_npz,
        "Filtered clinical tensor",
    )

    input_headers = npz_headers(
        input_npz
    )

    output_headers = npz_headers(
        output_npz
    )

    input_features = input_headers[
        "X_raw"
    ]["shape"][2]

    output_features = output_headers[
        "X_raw"
    ]["shape"][2]

    if output_features > input_features:
        raise RuntimeError(
            "Filtered tensor contains more features "
            "than the unfiltered tensor."
        )

    features_csv = require_file(
        features_csv
    )

    features = pd.read_csv(
        features_csv
    )

    if len(features) != output_features:
        raise RuntimeError(
            "Filtered feature CSV row count does "
            "not match NPZ feature count."
        )

    if (
        "train_observed_entries"
        not in features.columns
    ):
        raise RuntimeError(
            "Filtered feature report is missing "
            "train_observed_entries."
        )

    if (
        features[
            "train_observed_entries"
        ] <= 0
    ).any():
        raise RuntimeError(
            "A retained feature has no training "
            "observations."
        )

    require_file(
        summary_json
    )

    print(
        "Features before filtering:",
        input_features,
    )

    print(
        "Features after filtering:",
        output_features,
    )

    print(
        "Removed features:",
        input_features - output_features,
    )


def validate_feature_selection(
    output_dir,
    consensus_name,
    summary_stem,
    seed,
    stage_name,
):
    output_dir = Path(output_dir)

    consensus_path = (
        output_dir
        / consensus_name
    )

    require_file(
        consensus_path
    )

    consensus = pd.read_csv(
        consensus_path
    )

    required_columns = [
        "feature_index",
        "label",
        "recommendation",
        "consensus_score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in consensus.columns
    ]

    if missing:
        raise RuntimeError(
            f"{stage_name} consensus table is "
            f"missing columns: {missing}"
        )

    if len(consensus) == 0:
        raise RuntimeError(
            f"{stage_name} produced no "
            "feature evidence."
        )

    if consensus[
        "feature_index"
    ].duplicated().any():
        raise RuntimeError(
            f"{stage_name} produced duplicate "
            "feature_index values."
        )

    summary_path = (
        output_dir
        / f"{summary_stem}_feature_selection_summary.json"
    )

    if not summary_path.exists():
        matches = sorted(
            output_dir.glob(
                "*_feature_selection_summary.json"
            )
        )

        if len(matches) != 1:
            raise RuntimeError(
                f"Could not identify the {stage_name} "
                "feature-selection summary."
            )

        summary_path = matches[0]

    require_file(
        summary_path
    )

    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    random_states = find_json_values(
        summary,
        "random_state",
    )

    if not random_states:
        raise RuntimeError(
            f"{stage_name} summary does not record "
            "random_state."
        )

    if int(random_states[0]) != int(seed):
        raise RuntimeError(
            f"{stage_name} random_state is "
            f"{random_states[0]}, expected {seed}."
        )

    print(
        "Consensus evidence rows:",
        len(consensus),
    )

    print(
        "Recommendation counts:",
        consensus[
            "recommendation"
        ].value_counts(
            dropna=False
        ).to_dict(),
    )

    print(
        "Recorded random_state:",
        int(random_states[0]),
    )

    print(
        "Consensus table:",
        consensus_path,
    )


def validate_final_features(base):
    final_dir = (
        base
        / "data/processed/ehr/ehr_final_24h"
    )

    final_npz = (
        final_dir
        / "ehr_24h_final_current_split.npz"
    )

    features_csv = (
        final_dir
        / "ehr_24h_final_selected_features.csv"
    )

    require_file(
        final_npz
    )

    require_file(
        features_csv
    )

    headers = npz_headers(
        final_npz
    )

    require_npz_keys(
        headers,
        [
            "X_raw",
            "mask",
            "y",
            "split",
            "sample_id",
            "variables",
            "labels",
            "sources",
            "itemids",
        ],
        final_npz,
    )

    shape = headers[
        "X_raw"
    ]["shape"]

    if len(shape) != 3:
        raise RuntimeError(
            f"Final EHR tensor has invalid shape: "
            f"{shape}"
        )

    if shape[1] != 24:
        raise RuntimeError(
            f"Final EHR tensor has {shape[1]} "
            "time bins instead of 24."
        )

    if shape[2] != 30:
        raise RuntimeError(
            f"Final EHR tensor has {shape[2]} "
            "features instead of 30."
        )

    features = pd.read_csv(
        features_csv
    )

    if len(features) != 30:
        raise RuntimeError(
            f"Final selected feature CSV has "
            f"{len(features)} rows instead of 30."
        )

    if (
        "selection_branch"
        in features.columns
    ):
        print(
            "Selection branch counts:",
            features[
                "selection_branch"
            ].value_counts().to_dict(),
        )

    elif (
        "selection_group"
        in features.columns
    ):
        print(
            "Selection group counts:",
            features[
                "selection_group"
            ].value_counts().to_dict(),
        )

    validate_npz_alignment(
        base,
        final_npz,
    )

    print(
        "Final tensor shape:",
        shape,
    )

    print(
        "Final selected features:",
        len(features),
    )


def validate_train_ready(base):
    npz_path = (
        base
        / "data/processed/ehr"
        / "ehr_final_24h_train_ready"
        / "ehr_24h_final_train_ready_current_split.npz"
    )

    require_file(
        npz_path
    )

    headers = npz_headers(
        npz_path
    )

    required = [
        "X",
        "X_raw",
        "y",
        "split",
        "sample_id",
        "feature_mean",
        "feature_std",
    ]

    require_npz_keys(
        headers,
        required,
        npz_path,
    )

    mask_key = (
        "M"
        if "M" in headers
        else "mask"
    )

    if mask_key not in headers:
        raise RuntimeError(
            "Final train-ready NPZ contains "
            "neither M nor mask."
        )

    shape = headers[
        "X"
    ]["shape"]

    if shape != headers[
        "X_raw"
    ]["shape"]:
        raise RuntimeError(
            "Final X and X_raw shapes do not match."
        )

    if shape != headers[
        mask_key
    ]["shape"]:
        raise RuntimeError(
            "Final X and mask shapes do not match."
        )

    if len(shape) != 3:
        raise RuntimeError(
            f"Final train-ready tensor has "
            f"invalid shape: {shape}"
        )

    if shape[1] != 24:
        raise RuntimeError(
            f"Final train-ready tensor has "
            f"{shape[1]} time bins instead of 24."
        )

    if shape[2] != 30:
        raise RuntimeError(
            f"Final train-ready tensor has "
            f"{shape[2]} features instead of 30."
        )

    validate_npz_alignment(
        base,
        npz_path,
    )

    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:
        X = data[
            "X"
        ].astype(np.float32)

        mask = data[
            mask_key
        ].astype(np.float32)

        split = data[
            "split"
        ].astype(str)

        feature_mean = data[
            "feature_mean"
        ].astype(np.float64)

        feature_std = data[
            "feature_std"
        ].astype(np.float64)

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Final normalized X contains non-finite values."
        )

    if not np.isfinite(
        feature_mean
    ).all():
        raise RuntimeError(
            "Feature means contain non-finite values."
        )

    if not np.isfinite(
        feature_std
    ).all():
        raise RuntimeError(
            "Feature standard deviations contain "
            "non-finite values."
        )

    if (
        feature_std <= 0
    ).any():
        raise RuntimeError(
            "At least one feature standard deviation "
            "is not positive."
        )

    if not np.allclose(
        X[
            mask <= 0
        ],
        0.0,
        atol=1e-7,
    ):
        raise RuntimeError(
            "Masked values are not zero in the "
            "train-ready tensor."
        )

    train_rows = (
        split == "train"
    )

    if train_rows.sum() == 0:
        raise RuntimeError(
            "Final train-ready NPZ has no training rows."
        )

    observed_means = []
    observed_stds = []

    for feature_index in range(
        shape[2]
    ):
        observed = (
            train_rows[:, None]
            & (
                mask[
                    :,
                    :,
                    feature_index
                ] > 0
            )
        )

        values = X[
            :,
            :,
            feature_index
        ][observed]

        if len(values) == 0:
            raise RuntimeError(
                f"Final feature {feature_index} has "
                "no training observations."
            )

        observed_means.append(
            float(
                np.mean(values)
            )
        )

        observed_stds.append(
            float(
                np.std(values)
            )
        )

    observed_means = np.array(
        observed_means
    )

    observed_stds = np.array(
        observed_stds
    )

    if np.max(
        np.abs(
            observed_means
        )
    ) > 1e-4:
        raise RuntimeError(
            "Train-only normalized feature means "
            "are not sufficiently close to zero."
        )

    invalid_std = (
        (
            np.abs(
                observed_stds - 1.0
            ) > 1e-4
        )
        & (
            observed_stds > 1e-7
        )
    )

    if invalid_std.any():
        indices = np.where(
            invalid_std
        )[0].tolist()

        raise RuntimeError(
            "Unexpected train-only normalized "
            f"standard deviation for features: {indices}"
        )

    print(
        "Final train-ready shape:",
        shape,
    )

    print(
        "Train/val/test rows:",
        pd.Series(
            split
        ).value_counts().to_dict(),
    )

    print(
        "Maximum absolute normalized "
        "training mean:",
        float(
            np.max(
                np.abs(
                    observed_means
                )
            )
        ),
    )

    print(
        "Normalized training standard "
        "deviation range:",
        (
            float(
                observed_stds.min()
            ),
            float(
                observed_stds.max()
            ),
        ),
    )

    del X
    del mask
    gc.collect()


def run_step(
    number,
    name,
    command,
    base,
    environment,
    validator,
):
    section(
        f"STEP {number}: {name}"
    )

    print(
        "Command:"
    )

    print(
        " ".join(
            str(part)
            for part in command
        )
    )

    print()

    subprocess.run(
        [
            str(part)
            for part in command
        ],
        cwd=base,
        env=environment,
        check=True,
    )

    section(
        f"VERIFY STEP {number}: {name}"
    )

    validator()

    print()
    print(
        f"PASS: STEP {number}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        type=str,
        default=str(PROJECT_ROOT),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    parser.add_argument(
        "--split-search-iterations",
        type=int,
        default=SPLIT_SEARCH_ITERATIONS,
    )

    parser.add_argument(
        "--max-prevalence-gap",
        type=float,
        default=MAX_PREVALENCE_GAP,
    )

    parser.add_argument(
        "--max-split-size-deviation",
        type=float,
        default=MAX_SPLIT_SIZE_DEVIATION,
    )

    parser.add_argument(
        "--start-step",
        type=int,
        choices=range(1, 9),
        default=1,
        help=(
            "First preprocessing step to run. "
            "Use only after confirming earlier "
            "step outputs are valid."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    base = Path(
        args.base
    ).resolve()

    if not base.exists():
        raise FileNotFoundError(
            base
        )

    scripts = (
        base
        / "scripts/preprocess"
    )

    required_scripts = [
        "build_cohort.py",
        "build_ehr_tensor.py",
        "filter_ehr_features.py",
        "select_ehr_features.py",
        "build_broad_ehr_tensor.py",
        "select_broad_ehr_features.py",
        "build_final_ehr_features.py",
        "build_final_ehr_train_ready.py",
    ]

    for name in required_scripts:
        require_file(
            scripts / name
        )

    preflight_inputs(
        base
    )

    environment = os.environ.copy()

    environment[
        "PYTHONHASHSEED"
    ] = str(
        args.seed
    )

    environment[
        "OMP_NUM_THREADS"
    ] = "1"

    environment[
        "MKL_NUM_THREADS"
    ] = "1"

    environment[
        "OPENBLAS_NUM_THREADS"
    ] = "1"

    environment[
        "NUMEXPR_NUM_THREADS"
    ] = "1"

    python = sys.executable

    cohort_npz = (
        base
        / "data/processed/ehr"
        / "ehr_feature_selection_24h"
        / "tensors"
        / "ehr_24h_current_split.npz"
    )

    filtered_npz = (
        base
        / "data/processed/ehr"
        / "ehr_feature_selection_24h"
        / "tensors"
        / "ehr_24h_current_split_nonzero_train.npz"
    )

    filtered_features = (
        base
        / "data/processed/ehr"
        / "ehr_feature_selection_24h"
        / "features"
        / "ehr_24h_current_split_nonzero_train_features.csv"
    )

    filtered_summary = (
        base
        / "data/processed/ehr"
        / "ehr_feature_selection_24h"
        / "audits"
        / "ehr_24h_current_split_nonzero_train_filter_summary.json"
    )

    clinical_selection_dir = (
        base
        / "data/processed/ehr"
        / "ehr_feature_selection_24h"
        / "features"
        / "selection_clinical"
    )

    broad_npz = (
        base
        / "data/processed/ehr"
        / "ehr_broad_feature_selection_24h"
        / "ehr_24h_broad_current_split.npz"
    )

    broad_selection_dir = (
        base
        / "data/processed/ehr"
        / "ehr_broad_feature_selection_24h"
        / "features"
        / "selection_strict_v4"
    )

    steps = [
        (
            1,
            "Build prevalence-balanced patient cohort",
            [
                python,
                "-u",
                scripts / "build_cohort.py",
                "--repo_root",
                base,
                "--seed",
                args.seed,
                "--split_search_iterations",
                args.split_search_iterations,
                "--max_prevalence_gap",
                args.max_prevalence_gap,
                "--max_split_size_deviation",
                args.max_split_size_deviation,
            ],
            lambda: validate_cohort(
                base=base,
                seed=args.seed,
                max_prevalence_gap=(
                    args.max_prevalence_gap
                ),
                max_size_deviation=(
                    args.max_split_size_deviation
                ),
            ),
        ),
        (
            2,
            "Build initial 24-hour EHR tensor",
            [
                python,
                "-u",
                scripts / "build_ehr_tensor.py",
                "--root",
                base,
                "--cohort_csv",
                (
                    base
                    / "data/processed/cohorts"
                    / "cohort.csv"
                ),
                "--output_dir",
                (
                    base
                    / "data/processed/ehr"
                    / "ehr_feature_selection_24h"
                ),
                "--output_name",
                "ehr_24h_current_split.npz",
                "--window_hours",
                EHR_WINDOW_HOURS,
                "--chunksize",
                EHR_CHUNKSIZE,
                "--max_chart_features",
                MAX_CHART_FEATURES,
                "--max_lab_features",
                MAX_LAB_FEATURES,
            ],
            lambda: validate_candidate_tensor(
                base,
                cohort_npz,
                "Initial clinical EHR tensor",
            ),
        ),
        (
            3,
            "Remove features absent from training data",
            [
                python,
                "-u",
                scripts / "filter_ehr_features.py",
                "--input_npz",
                cohort_npz,
                "--output_npz",
                filtered_npz,
                "--output_features_csv",
                filtered_features,
                "--output_summary_json",
                filtered_summary,
            ],
            lambda: validate_filtered_tensor(
                base=base,
                input_npz=cohort_npz,
                output_npz=filtered_npz,
                features_csv=filtered_features,
                summary_json=filtered_summary,
            ),
        ),
        (
            4,
            "Run seeded clinical feature selection",
            [
                python,
                "-u",
                scripts / "select_ehr_features.py",
                "--npz_path",
                filtered_npz,
                "--output_dir",
                clinical_selection_dir,
                "--n_bootstraps",
                CLINICAL_BOOTSTRAPS,
                "--random_state",
                args.seed,
            ],
            lambda: validate_feature_selection(
                output_dir=clinical_selection_dir,
                consensus_name=(
                    "ehr_24h_current_split_"
                    "nonzero_train_consensus_"
                    "feature_evidence_train_only.csv"
                ),
                summary_stem=(
                    "ehr_24h_current_split_"
                    "nonzero_train"
                ),
                seed=args.seed,
                stage_name=(
                    "Clinical feature selection"
                ),
            ),
        ),
        (
            5,
            "Build broad 24-hour EHR tensor",
            [
                python,
                "-u",
                scripts / "build_broad_ehr_tensor.py",
                "--cohort_csv",
                (
                    base
                    / "data/processed/cohorts"
                    / "cohort.csv"
                ),
                "--min_train_sample_coverage",
                MIN_TRAIN_SAMPLE_COVERAGE,
                "--exclude_obvious_admin_or_leakage",
            ],
            lambda: validate_candidate_tensor(
                base,
                broad_npz,
                "Broad EHR tensor",
            ),
        ),
        (
            6,
            "Run seeded broad feature selection",
            [
                python,
                "-u",
                scripts / "select_broad_ehr_features.py",
                "--npz_path",
                broad_npz,
                "--output_dir",
                broad_selection_dir,
                "--n_bootstraps",
                BROAD_BOOTSTRAPS,
                "--elastic_top_k",
                BROAD_ELASTIC_TOP_K,
                "--random_state",
                args.seed,
            ],
            lambda: validate_feature_selection(
                output_dir=broad_selection_dir,
                consensus_name=(
                    "ehr_24h_broad_current_split_"
                    "consensus_feature_evidence_"
                    "train_only.csv"
                ),
                summary_stem=(
                    "ehr_24h_broad_current_split"
                ),
                seed=args.seed,
                stage_name=(
                    "Broad feature selection"
                ),
            ),
        ),
        (
            7,
            "Build final 30-feature EHR tensor",
            [
                python,
                "-u",
                scripts / "build_final_ehr_features.py",
                "--repo_root",
                base,
            ],
            lambda: validate_final_features(
                base
            ),
        ),
        (
            8,
            "Build final train-ready normalized EHR tensor",
            [
                python,
                "-u",
                scripts
                / "build_final_ehr_train_ready.py",
                "--input_npz",
                (
                    base
                    / "data/processed/ehr"
                    / "ehr_final_24h"
                    / "ehr_24h_final_current_split.npz"
                ),
                "--features_csv",
                (
                    base
                    / "data/processed/ehr"
                    / "ehr_final_24h"
                    / "ehr_24h_final_selected_features.csv"
                ),
                "--output_dir",
                (
                    base
                    / "data/processed/ehr"
                    / "ehr_final_24h_train_ready"
                ),
            ],
            lambda: validate_train_ready(
                base
            ),
        ),
    ]

    section(
        "RESPIRATORY PREPROCESSING PIPELINE"
    )

    print(
        "Project:",
        base,
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Maximum prevalence gap:",
        args.max_prevalence_gap,
    )

    print(
        "Maximum split-size deviation:",
        args.max_split_size_deviation,
    )

    for (
        number,
        name,
        command,
        validator,
    ) in steps:
        if number < int(
            args.start_step
        ):
            print(
                f"SKIP: STEP {number} - {name}"
            )
            continue

        run_step(
            number=number,
            name=name,
            command=command,
            base=base,
            environment=environment,
            validator=validator,
        )

    environment_path = (
        base
        / "data/processed/cohorts"
        / "preprocessing_environment.txt"
    )

    with open(
        environment_path,
        "w",
        encoding="utf-8",
    ) as file:
        subprocess.run(
            [
                python,
                "--version",
            ],
            stdout=file,
            stderr=subprocess.STDOUT,
            check=True,
            env=environment,
        )

        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "freeze",
            ],
            stdout=file,
            stderr=subprocess.STDOUT,
            check=True,
            env=environment,
        )

    section(
        "ALL PREPROCESSING STEPS PASSED"
    )

    print(
        "Final cohort:",
        (
            base
            / "data/processed/cohorts"
            / "cohort.csv"
        ),
    )

    print(
        "Final EHR NPZ:",
        (
            base
            / "data/processed/ehr"
            / "ehr_final_24h_train_ready"
            / "ehr_24h_final_train_ready_current_split.npz"
        ),
    )

    print(
        "Environment:",
        environment_path,
    )


if __name__ == "__main__":
    main()
