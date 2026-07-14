#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_build_ehr_tensor_from_source_24h_current_split.py

Clean 24h EHR candidate tensor builder.

This uses the strict source-feature idea from the old 03 script, but:
  - keeps 24h window
  - uses current final cohort and current split
  - does NOT use old NPZ files
  - does NOT manually force 26 features
  - does NOT create a new split
  - outputs candidate tensor for train-only feature selection

Input:
  final_resp48_stable72_no_prior_cxr_cohort.csv

Output:
  X_raw:     [N, 24, F], NaN where missing
  X:         [N, 24, F], zero-filled raw values
  mask:      [N, 24, F]
  y:         [N]
  sample_id: [N]
  split:     [N]
  variables: [F]
  labels:    [F]
  sources:   [F]
  itemids:   [F]
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# Utilities
# ============================================================

def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def normalize_label(x):
    return str(x).strip()


# ============================================================
# Strict clinical source filtering rules
# ============================================================

def should_keep_chartevent(row):
    """
    Strict chartevents filter.

    Keeps common bedside physiology/vitals.
    Excludes ventilator settings, device variables, alarms,
    scores, procedures, obvious treatment/proxy variables.
    """
    label = normalize_label(row.get("label", ""))
    param_type = normalize_label(row.get("param_type", ""))

    label_l = label.lower()
    param_l = param_type.lower()

    if "numeric" not in param_l:
        return False

    exclude_terms = [
        "alarm",
        "goal",
        "parameter",
        "apache",
        "score",
        "braden",
        "rass",
        "cam-icu",
        "pain",

        "height",
        "weight",
        "admission weight",
        "daily weight",
        "feeding weight",

        "vent",
        "ventilator",
        "ventilation",
        "peep",
        "tidal",
        "minute volume",
        "fio2",
        "inspired o2",
        "peak insp",
        "plateau",
        "airway",
        "paw",
        "apnea",
        "respiratory rate (set)",
        "respiratory rate (total)",
        "respiratory rate (spontaneous)",
        "inspiratory time",

        "pa line",
        "pulmonary artery",
        "aortic pressure",
        "rv ",
        "ra %",
        "pvr",
        "svr",
        "signal",
        "cco",
        "cerebral",

        "device",
        "mode",
        "tube",
        "ett",
        "trach",
        "intub",
        "extub",
    ]

    # CURRENT RUN:
    # Allow pre-index respiratory support intensity variables.
    # These are not future leakage if they occur before prediction_time,
    # but they are clinically close to the respiratory escalation outcome.
    support_allowed_terms = [
        "fio2",
        "inspired o2 fraction",
        "fraction inspired oxygen",
        "oxygen flow",
        "o2 flow",
    ]

    if any(term in label_l for term in support_allowed_terms):
        return True

    if any(term in label_l for term in exclude_terms):
        return False

    exact_allowed = {
        "heart rate",
        "respiratory rate",

        "o2 saturation pulseoxymetry",
        "arterial o2 saturation",

        "temperature celsius",
        "temperature fahrenheit",

        "arterial blood pressure systolic",
        "arterial blood pressure diastolic",
        "arterial blood pressure mean",

        "non invasive blood pressure systolic",
        "non invasive blood pressure diastolic",
        "non invasive blood pressure mean",

        "manual blood pressure systolic left",
        "manual blood pressure systolic right",
        "manual blood pressure diastolic left",
        "manual blood pressure diastolic right",

        "glucose (serum)",
        "glucose (whole blood)",
        "glucose (whole blood) (soft)",
        "glucose finger stick (range 70-100)",
    }

    return label_l in exact_allowed


def should_keep_labevent(row):
    """
    Strict labevents filter.

    Keeps common blood/serum/plasma ICU labs:
    blood gas, metabolic, CBC, electrolytes, renal, protein/liver,
    coagulation.

    Excludes urine/CSF/body-fluid variants and special tests.
    """
    label = normalize_label(row.get("label", ""))
    fluid = normalize_label(row.get("fluid", ""))
    category = normalize_label(row.get("category", ""))

    label_l = label.lower().strip()
    fluid_l = fluid.lower().strip()
    category_l = category.lower().strip()

    bad_fluid_terms = [
        "urine",
        "csf",
        "ascites",
        "pleural",
        "joint",
        "stool",
        "body fluid",
        "other body fluid",
        "peritoneal",
        "synovial",
        "sputum",
        "bronchial",
        "bile",
        "dialysis",
    ]

    fluid_text = f"{label_l} {fluid_l} {category_l}"

    if any(term in fluid_text for term in bad_fluid_terms):
        return False

    allowed_exact = {
        # Blood gas / respiratory physiology
        "ph",
        "po2",
        "pco2",
        "base excess",
        "bicarbonate",
        "calculated bicarbonate",
        "calculated total co2",
        "total co2",
        "oxygen",
        "oxygen saturation",
        "alveolar-arterial gradient",
        "carboxyhemoglobin",
        "methemoglobin",

        # Metabolic / severity
        "lactate",
        "glucose",
        "anion gap",

        # CBC
        "white blood cells",
        "wbc",
        "red blood cells",
        "rbc",
        "hemoglobin",
        "hematocrit",
        "platelet count",
        "platelets",
        "mch",
        "mchc",
        "mcv",
        "rdw",

        # Electrolytes
        "sodium",
        "potassium",
        "chloride",
        "calcium, total",
        "free calcium",
        "magnesium",
        "phosphate",

        # Renal
        "creatinine",
        "urea nitrogen",
        "bun",

        # Liver/protein
        "albumin",
        "bilirubin, total",
        "bilirubin, direct",

        # Coagulation
        "inr(pt)",
        "pt",
        "ptt",
        "fibrinogen",
    }

    aliases = {
        "inr": "inr(pt)",
        "international normalized ratio": "inr(pt)",
        "partial thromboplastin time": "ptt",
        "prothrombin time": "pt",
        "urea nitrogen": "urea nitrogen",
        "blood urea nitrogen": "bun",
        "white blood cell count": "white blood cells",
        "platelet": "platelets",
    }

    normalized = aliases.get(label_l, label_l)

    if normalized in allowed_exact:
        return True

    allowed_prefixes = [
        "neutrophils",
        "lymphocytes",
        "monocytes",
        "eosinophils",
        "basophils",
    ]

    if any(normalized == p or normalized.startswith(p + " ") for p in allowed_prefixes):
        return True

    return False


def lab_priority(label):
    l = str(label).lower().strip()

    priority = {
        "ph": 1,
        "po2": 2,
        "pco2": 3,
        "base excess": 4,
        "bicarbonate": 5,
        "calculated bicarbonate": 6,
        "calculated total co2": 7,
        "total co2": 8,
        "oxygen": 9,
        "oxygen saturation": 10,
        "alveolar-arterial gradient": 11,
        "carboxyhemoglobin": 12,
        "methemoglobin": 13,

        "lactate": 20,
        "anion gap": 21,
        "glucose": 22,

        "white blood cells": 30,
        "wbc": 31,
        "hemoglobin": 32,
        "hematocrit": 33,
        "platelet count": 34,
        "platelets": 35,
        "red blood cells": 36,
        "rbc": 37,
        "mcv": 38,
        "mch": 39,
        "mchc": 40,
        "rdw": 41,

        "sodium": 50,
        "potassium": 51,
        "chloride": 52,
        "calcium, total": 53,
        "free calcium": 54,
        "magnesium": 55,
        "phosphate": 56,
        "creatinine": 57,
        "urea nitrogen": 58,
        "bun": 59,

        "albumin": 70,
        "bilirubin, total": 71,
        "bilirubin, direct": 72,

        "inr(pt)": 80,
        "pt": 81,
        "ptt": 82,
        "fibrinogen": 83,
    }

    if l in priority:
        return priority[l]

    if l.startswith("neutrophils"):
        return 90
    if l.startswith("lymphocytes"):
        return 91
    if l.startswith("monocytes"):
        return 92
    if l.startswith("eosinophils"):
        return 93
    if l.startswith("basophils"):
        return 94

    return 999


def chart_priority(label):
    l = str(label).lower().strip()

    priority = {
        "heart rate": 1,
        "respiratory rate": 2,
        "o2 saturation pulseoxymetry": 3,
        "arterial o2 saturation": 4,

        "non invasive blood pressure systolic": 10,
        "non invasive blood pressure diastolic": 11,
        "non invasive blood pressure mean": 12,

        "arterial blood pressure systolic": 20,
        "arterial blood pressure diastolic": 21,
        "arterial blood pressure mean": 22,

        "temperature fahrenheit": 30,
        "temperature celsius": 31,

        "glucose finger stick (range 70-100)": 40,
        "glucose (serum)": 41,
        "glucose (whole blood)": 42,
        "glucose (whole blood) (soft)": 43,

        "manual blood pressure systolic left": 50,
        "manual blood pressure systolic right": 51,
        "manual blood pressure diastolic left": 52,
        "manual blood pressure diastolic right": 53,
    }

    return priority.get(l, 999)


def build_candidate_features(root, max_chart_features, max_lab_features):
    print_section("BUILD STRICT 24H CANDIDATE FEATURE LIST FROM SOURCE DICTIONARIES")

    root = Path(root)

    d_items_path = root / "data/raw/mimiciv/icu/d_items.csv.gz"
    d_labitems_path = root / "data/raw/mimiciv/hosp/d_labitems.csv.gz"

    if not d_items_path.exists():
        raise FileNotFoundError(d_items_path)

    if not d_labitems_path.exists():
        raise FileNotFoundError(d_labitems_path)

    d_items = pd.read_csv(d_items_path, compression="gzip")
    d_labs = pd.read_csv(d_labitems_path, compression="gzip")

    chart_rows = []

    for _, row in d_items.iterrows():
        if str(row.get("linksto", "")).lower() != "chartevents":
            continue

        if should_keep_chartevent(row):
            itemid = int(row["itemid"])
            label = normalize_label(row["label"])
            variable = f"chartevents::{itemid}::{label}::valuenum"

            chart_rows.append({
                "variable": variable,
                "source": "chartevents",
                "itemid": itemid,
                "label": label,
                "category": normalize_label(row.get("category", "")),
                "value_source": "valuenum",
            })

    chart_df = pd.DataFrame(chart_rows)

    if not chart_df.empty:
        chart_df = chart_df.drop_duplicates("itemid")
        chart_df["priority"] = chart_df["label"].apply(chart_priority)
        chart_df = chart_df.sort_values(["priority", "label", "itemid"]).drop_duplicates("label")

        if max_chart_features > 0:
            chart_df = chart_df.head(max_chart_features)

        chart_df = chart_df.drop(columns=["priority"])

    lab_rows = []

    for _, row in d_labs.iterrows():
        if should_keep_labevent(row):
            itemid = int(row["itemid"])
            label = normalize_label(row["label"])
            variable = f"labevents::{itemid}::{label}::valuenum"

            lab_rows.append({
                "variable": variable,
                "source": "labevents",
                "itemid": itemid,
                "label": label,
                "category": normalize_label(row.get("category", "")),
                "value_source": "valuenum",
            })

    lab_df = pd.DataFrame(lab_rows)

    if not lab_df.empty:
        lab_df = lab_df.drop_duplicates("itemid")
        lab_df["priority"] = lab_df["label"].apply(lab_priority)
        lab_df = lab_df.sort_values(["priority", "label", "itemid"]).drop_duplicates("label")

        if max_lab_features > 0:
            lab_df = lab_df.head(max_lab_features)

        lab_df = lab_df.drop(columns=["priority"])

    features = pd.concat([chart_df, lab_df], axis=0).reset_index(drop=True)

    if features.empty:
        raise ValueError("No candidate features selected. Check filtering rules.")

    features["feature_index"] = np.arange(len(features), dtype=np.int64)

    print("Chart features:", int((features["source"] == "chartevents").sum()))
    print("Lab features:", int((features["source"] == "labevents").sum()))
    print("Total candidate features:", len(features))

    print("\nCandidate features:")
    print(features[["feature_index", "source", "itemid", "label", "variable"]].to_string(index=False))

    return features


# ============================================================
# Cohort loading
# ============================================================

def load_current_cohort(cohort_csv, window_hours):
    print_section("LOAD CURRENT FINAL COHORT WITH CURRENT SPLIT")

    cohort_csv = Path(cohort_csv)

    if not cohort_csv.exists():
        raise FileNotFoundError(cohort_csv)

    df = pd.read_csv(cohort_csv)

    required = [
        "sample_id",
        "subject_id",
        "stay_id",
        "label",
        "split",
        "index_time",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Current cohort missing required columns: {missing}")

    df = df.copy()

    df["row_index"] = np.arange(len(df), dtype=np.int64)
    df["sample_id"] = df["sample_id"].astype(str)
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")

    if "hadm_id" in df.columns:
        df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")
    else:
        df["hadm_id"] = pd.NA

    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    df["prediction_time"] = df["index_time"]
    df["window_start"] = df["prediction_time"] - pd.to_timedelta(window_hours, unit="h")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["split"] = df["split"].astype(str).str.lower()

    df = df[
        df["sample_id"].notna()
        & df["subject_id"].notna()
        & df["stay_id"].notna()
        & df["prediction_time"].notna()
        & df["window_start"].notna()
        & df["label"].isin([0, 1])
        & df["split"].isin(["train", "val", "test"])
    ].copy()

    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    df["row_index"] = np.arange(len(df), dtype=np.int64)

    print("Cohort CSV:", cohort_csv)
    print("Cohort shape:", df.shape)
    print("Window hours:", window_hours)
    print("Label counts:", df["label"].value_counts().sort_index().to_dict())
    print("Split counts:", df["split"].value_counts().to_dict())
    print("\nSplit x label:")
    print(pd.crosstab(df["split"], df["label"]))

    return df


# ============================================================
# Tensor aggregation
# ============================================================

def apply_events_to_tensor(events, X_sum, X_count, feature_map, window_hours, time_col="charttime"):
    if events.empty:
        return 0

    events = events.copy()

    events[time_col] = pd.to_datetime(events[time_col], errors="coerce")
    events["window_start"] = pd.to_datetime(events["window_start"], errors="coerce")

    events = events[events[time_col].notna() & events["window_start"].notna()]
    if events.empty:
        return 0

    delta_hours = (events[time_col] - events["window_start"]).dt.total_seconds() / 3600.0
    events["hour_index"] = np.floor(delta_hours).astype("int64")

    events = events[
        (events["hour_index"] >= 0)
        & (events["hour_index"] < window_hours)
    ].copy()

    events["valuenum"] = pd.to_numeric(events["valuenum"], errors="coerce")
    events = events[np.isfinite(events["valuenum"])].copy()

    if events.empty:
        return 0

    events["feature_index"] = events["itemid"].map(feature_map)
    events = events[events["feature_index"].notna()].copy()

    if events.empty:
        return 0

    r = events["row_index"].to_numpy(dtype=np.int64)
    h = events["hour_index"].to_numpy(dtype=np.int64)
    f = events["feature_index"].to_numpy(dtype=np.int64)
    v = events["valuenum"].to_numpy(dtype=np.float32)

    np.add.at(X_sum, (r, h, f), v)
    np.add.at(X_count, (r, h, f), 1.0)

    return len(events)


def process_chartevents(root, cohort_df, features, X_sum, X_count, chunksize, window_hours):
    print_section("PROCESS CHARTEVENTS FROM SOURCE")

    root = Path(root)
    chartevents_path = root / "data/raw/mimiciv/icu/chartevents.csv.gz"

    if not chartevents_path.exists():
        raise FileNotFoundError(chartevents_path)

    chart_features = features[features["source"] == "chartevents"].copy()

    if chart_features.empty:
        print("No chartevents features selected.")
        return 0

    feature_map = dict(zip(chart_features["itemid"].astype(int), chart_features["feature_index"].astype(int)))
    itemids = set(chart_features["itemid"].astype(int).tolist())

    needed_stays = set(cohort_df["stay_id"].dropna().astype(int).tolist())

    cohort_small = cohort_df[["row_index", "stay_id", "window_start", "prediction_time"]].copy()
    cohort_small["stay_id"] = cohort_small["stay_id"].astype("int64")

    usecols = ["stay_id", "charttime", "itemid", "valuenum"]

    total_kept = 0
    chunk_no = 0

    reader = pd.read_csv(
        chartevents_path,
        compression="gzip",
        usecols=usecols,
        chunksize=chunksize,
    )

    for chunk in reader:
        chunk_no += 1

        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce").astype("Int64")
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")

        chunk = chunk[
            chunk["itemid"].isin(itemids)
            & chunk["stay_id"].isin(needed_stays)
        ].copy()

        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")

        chunk = chunk[
            chunk["stay_id"].notna()
            & chunk["charttime"].notna()
            & np.isfinite(chunk["valuenum"])
        ].copy()

        if chunk.empty:
            continue

        chunk["stay_id"] = chunk["stay_id"].astype("int64")

        merged = chunk.merge(cohort_small, on="stay_id", how="inner")

        if merged.empty:
            continue

        merged = merged[
            (merged["charttime"] >= merged["window_start"])
            & (merged["charttime"] < merged["prediction_time"])
        ].copy()

        if merged.empty:
            continue

        kept = apply_events_to_tensor(
            merged,
            X_sum=X_sum,
            X_count=X_count,
            feature_map=feature_map,
            window_hours=window_hours,
            time_col="charttime",
        )

        total_kept += kept

        if chunk_no % 20 == 0:
            print(f"chartevents chunks processed={chunk_no}, events added={total_kept}", flush=True)

    print("Total chartevents added:", total_kept)

    return total_kept


def process_labevents(root, cohort_df, features, X_sum, X_count, chunksize, window_hours):
    print_section("PROCESS LABEVENTS FROM SOURCE")

    root = Path(root)
    labevents_path = root / "data/raw/mimiciv/hosp/labevents.csv.gz"

    if not labevents_path.exists():
        raise FileNotFoundError(labevents_path)

    lab_features = features[features["source"] == "labevents"].copy()

    if lab_features.empty:
        print("No labevents features selected.")
        return 0

    feature_map = dict(zip(lab_features["itemid"].astype(int), lab_features["feature_index"].astype(int)))
    itemids = set(lab_features["itemid"].astype(int).tolist())

    if "hadm_id" in cohort_df.columns and cohort_df["hadm_id"].notna().any():
        match_mode = "hadm_id"

        cohort_lab = cohort_df[
            ["row_index", "subject_id", "hadm_id", "window_start", "prediction_time"]
        ].copy()

        cohort_lab = cohort_lab[cohort_lab["hadm_id"].notna()].copy()
        cohort_lab["hadm_id"] = cohort_lab["hadm_id"].astype("int64")

        needed_hadm = set(cohort_lab["hadm_id"].astype(int).tolist())

        usecols = ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]

        print("Lab matching mode: hadm_id + time window")
        print("Cohort rows with hadm_id:", len(cohort_lab))
        print("Unique hadm_id:", len(needed_hadm))

    else:
        match_mode = "subject_id"

        cohort_lab = cohort_df[
            ["row_index", "subject_id", "window_start", "prediction_time"]
        ].copy()

        cohort_lab = cohort_lab[cohort_lab["subject_id"].notna()].copy()
        cohort_lab["subject_id"] = cohort_lab["subject_id"].astype("int64")

        needed_subjects = set(cohort_lab["subject_id"].astype(int).tolist())

        usecols = ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]

        print("Lab matching mode: subject_id + time window")
        print("Reason: cohort has no usable hadm_id")
        print("Cohort rows:", len(cohort_lab))
        print("Unique subject_id:", len(needed_subjects))

    total_kept = 0
    chunk_no = 0

    reader = pd.read_csv(
        labevents_path,
        compression="gzip",
        usecols=usecols,
        chunksize=chunksize,
    )

    for chunk in reader:
        chunk_no += 1

        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce").astype("Int64")

        chunk = chunk[chunk["itemid"].isin(itemids)].copy()

        if chunk.empty:
            continue

        if match_mode == "hadm_id":
            chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["hadm_id"].isin(needed_hadm)].copy()

            if chunk.empty:
                continue

            merge_cols = ["hadm_id"]

        else:
            chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["subject_id"].isin(needed_subjects)].copy()

            if chunk.empty:
                continue

            merge_cols = ["subject_id"]

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")

        chunk = chunk[
            chunk["charttime"].notna()
            & np.isfinite(chunk["valuenum"])
        ].copy()

        if chunk.empty:
            continue

        if match_mode == "hadm_id":
            chunk = chunk[chunk["hadm_id"].notna()].copy()
            chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        else:
            chunk = chunk[chunk["subject_id"].notna()].copy()
            chunk["subject_id"] = chunk["subject_id"].astype("int64")

        merged = chunk.merge(cohort_lab, on=merge_cols, how="inner")

        if merged.empty:
            continue

        merged = merged[
            (merged["charttime"] >= merged["window_start"])
            & (merged["charttime"] < merged["prediction_time"])
        ].copy()

        if merged.empty:
            continue

        kept = apply_events_to_tensor(
            merged,
            X_sum=X_sum,
            X_count=X_count,
            feature_map=feature_map,
            window_hours=window_hours,
            time_col="charttime",
        )

        total_kept += kept

        if chunk_no % 20 == 0:
            print(f"labevents chunks processed={chunk_no}, events added={total_kept}", flush=True)

    print("Total labevents added:", total_kept)

    return total_kept


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2]
        ),
    )

    parser.add_argument(
        "--cohort_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2]
            / "data/processed/ehr/ehr_feature_selection_24h"
        ),
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="ehr_24h_current_split.npz",
    )

    parser.add_argument(
        "--window_hours",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=750000,
    )

    parser.add_argument(
        "--max_chart_features",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--max_lab_features",
        type=int,
        default=80,
    )

    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)

    out_tensor_dir = output_dir / "tensors"
    out_feature_dir = output_dir / "features"
    out_audit_dir = output_dir / "audits"

    out_tensor_dir.mkdir(parents=True, exist_ok=True)
    out_feature_dir.mkdir(parents=True, exist_ok=True)
    out_audit_dir.mkdir(parents=True, exist_ok=True)

    output_name = args.output_name
    output_stem = Path(output_name).stem

    features = build_candidate_features(
        root=root,
        max_chart_features=args.max_chart_features,
        max_lab_features=args.max_lab_features,
    )

    features_path = out_feature_dir / f"{output_stem}_candidate_features.csv"
    features.to_csv(features_path, index=False)

    cohort_df = load_current_cohort(
        cohort_csv=args.cohort_csv,
        window_hours=args.window_hours,
    )

    N = len(cohort_df)
    T = int(args.window_hours)
    F = len(features)

    print_section("ALLOCATE 24H TENSOR")
    print("N:", N)
    print("T:", T)
    print("F:", F)

    X_sum = np.zeros((N, T, F), dtype=np.float32)
    X_count = np.zeros((N, T, F), dtype=np.float32)

    n_chart = process_chartevents(
        root=root,
        cohort_df=cohort_df,
        features=features,
        X_sum=X_sum,
        X_count=X_count,
        chunksize=args.chunksize,
        window_hours=args.window_hours,
    )

    n_lab = process_labevents(
        root=root,
        cohort_df=cohort_df,
        features=features,
        X_sum=X_sum,
        X_count=X_count,
        chunksize=args.chunksize,
        window_hours=args.window_hours,
    )

    print_section("FINALIZE 24H CANDIDATE TENSOR")

    mask = (X_count > 0).astype(np.float32)

    X_raw = np.full_like(X_sum, np.nan, dtype=np.float32)
    observed = X_count > 0
    X_raw[observed] = X_sum[observed] / X_count[observed]

    # This X is raw zero-filled only. Feature selection uses X_raw + mask.
    X = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    y = cohort_df["label"].to_numpy(dtype=np.int64)
    sample_id = cohort_df["sample_id"].astype(str).to_numpy()
    split = cohort_df["split"].astype(str).to_numpy()

    variables = features["variable"].astype(str).to_numpy()
    labels = features["label"].astype(str).to_numpy()
    sources = features["source"].astype(str).to_numpy()
    itemids = features["itemid"].to_numpy(dtype=np.int64)
    value_sources = features["value_source"].astype(str).to_numpy()

    print("X_raw shape:", X_raw.shape)
    print("X shape:", X.shape)
    print("mask shape:", mask.shape)
    print("observed fraction:", float(mask.mean()))
    print("y counts:", dict(zip(*np.unique(y, return_counts=True))))
    print("split counts:", dict(zip(*np.unique(split, return_counts=True))))

    feature_coverage = mask.mean(axis=(0, 1))

    coverage_report = features.copy()
    coverage_report["tensor_coverage"] = feature_coverage
    coverage_report = coverage_report.sort_values("tensor_coverage", ascending=False)

    coverage_path = out_feature_dir / f"{output_stem}_feature_coverage.csv"
    coverage_report.to_csv(coverage_path, index=False)

    out_npz = out_tensor_dir / output_name

    np.savez_compressed(
        out_npz,
        X=X.astype(np.float32),
        X_raw=X_raw.astype(np.float32),
        mask=mask.astype(np.float32),
        y=y.astype(np.int64),
        sample_id=sample_id.astype(str),
        split=split.astype(str),
        variables=variables.astype(str),
        labels=labels.astype(str),
        sources=sources.astype(str),
        itemids=itemids.astype(np.int64),
        value_sources=value_sources.astype(str),
        cohort_csv=str(args.cohort_csv),
        source="clean_from_mimic_iv_csv_current_rules_24h_current_split",
        window_hours=np.array([args.window_hours], dtype=np.int64),
        n_bins=np.array([args.window_hours], dtype=np.int64),
        n_chartevents_added=np.array([n_chart], dtype=np.int64),
        n_labevents_added=np.array([n_lab], dtype=np.int64),
    )

    summary = {
        "output_npz": str(out_npz),
        "cohort_csv": str(args.cohort_csv),
        "features_csv": str(features_path),
        "coverage_csv": str(coverage_path),
        "shape": list(X.shape),
        "window_hours": int(args.window_hours),
        "observed_fraction": float(mask.mean()),
        "y_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "split_counts": {str(k): int(v) for k, v in zip(*np.unique(split, return_counts=True))},
        "n_chartevents_added": int(n_chart),
        "n_labevents_added": int(n_lab),
        "strict_feature_rules": False,
        "current_rules": True,
        "forced_feature_count": False,
    }

    summary_path = out_audit_dir / f"{output_stem}_summary.json"
    save_json(summary, summary_path)

    print_section("SAVED OUTPUTS")
    print("NPZ:", out_npz)
    print("candidate features:", features_path)
    print("coverage:", coverage_path)
    print("summary:", summary_path)

    print_section("TOP FEATURE COVERAGE")
    print(
        coverage_report[
            ["feature_index", "source", "itemid", "label", "tensor_coverage"]
        ].head(80).to_string(index=False)
    )


if __name__ == "__main__":
    main()
