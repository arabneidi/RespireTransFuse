from pathlib import Path
import argparse
import json
import gc
import time
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


BASE = Path("/content/drive/MyDrive/respire-transfuse")

COHORT_DEFAULT = BASE / "data/processed/cohorts/cohort.csv"

CHARTEVENTS = BASE / "data/raw/mimiciv/icu/chartevents.csv.gz"
D_ITEMS = BASE / "data/raw/mimiciv/icu/d_items.csv.gz"

LABEVENTS = BASE / "data/raw/mimiciv/hosp/labevents.csv.gz"
D_LABITEMS = BASE / "data/raw/mimiciv/hosp/d_labitems.csv.gz"

OUT_DIR = BASE / "data/processed/ehr/ehr_broad_feature_selection_24h"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIR / "ehr_24h_broad_current_split.npz"
OUT_FEATURES = OUT_DIR / "ehr_24h_broad_current_split_candidate_features.csv"
OUT_FEATURE_STATS_ALL = OUT_DIR / "ehr_24h_broad_current_split_feature_stats_all.csv"
OUT_MANIFEST = OUT_DIR / "ehr_candidates_manifest.json"

WINDOW_HOURS = 24
N_BINS = 24

CHUNKSIZE_CHARTEVENTS = 2_000_000
CHUNKSIZE_LABEVENTS = 1_000_000


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def require(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)


def normalize_id(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


def obvious_admin_or_leakage(label):
    text = clean_text(label)

    patterns = [
        r"\bventilator\b",
        r"\bventilation\b",
        r"\bvent mode\b",
        r"\bpeep\b",
        r"\btidal\b",
        r"\bminute volume\b",
        r"\bairway\b",
        r"\bpeak insp\b",
        r"\bplateau\b",
        r"\bintub",
        r"\bextub",
        r"\bett\b",
        r"\btrach",
        r"\btube\b",
        r"\bprocedure\b",
        r"\bdevice\b",
        r"\bmode\b",
        r"\balarm\b",
        r"\bgoal\b",
        r"\bparameter\b",
        r"\bscore\b",
        r"\bapache\b",
        r"\bheight\b",
        r"\bweight\b",
    ]

    return any(re.search(p, text) for p in patterns)


def build_item_metadata():
    d_items = pd.read_csv(D_ITEMS)
    d_labitems = pd.read_csv(D_LABITEMS)

    d_items = normalize_id(d_items, ["itemid"])
    d_labitems = normalize_id(d_labitems, ["itemid"])

    item_label_col = "label" if "label" in d_items.columns else "Label"
    lab_label_col = "label" if "label" in d_labitems.columns else "Label"

    chart_meta = d_items[["itemid", item_label_col]].copy()
    chart_meta = chart_meta.rename(columns={item_label_col: "label"})
    chart_meta["source"] = "chartevents"

    lab_meta = d_labitems[["itemid", lab_label_col]].copy()
    lab_meta = lab_meta.rename(columns={lab_label_col: "label"})
    lab_meta["source"] = "labevents"

    meta = pd.concat([chart_meta, lab_meta], ignore_index=True)
    meta["itemid"] = pd.to_numeric(meta["itemid"], errors="coerce").astype("Int64")
    meta["label"] = meta["label"].astype(str)
    meta["label_clean"] = meta["label"].apply(clean_text)
    meta["obvious_admin_or_leakage"] = meta["label"].apply(obvious_admin_or_leakage)

    return meta


def make_variable(source, itemid, label):
    return f"{source}::{int(itemid)}::{str(label)}::valuenum"


def update_counts_from_windowed_chunk(
    merged,
    source,
    label_map,
    measurement_counts,
    sample_pairs,
    train_sample_idx_set,
):
    if merged.empty:
        return

    merged["itemid_int"] = merged["itemid"].astype(int)
    merged["label"] = merged["itemid_int"].map(label_map).fillna("")
    merged["variable"] = (
        source
        + "::"
        + merged["itemid_int"].astype(str)
        + "::"
        + merged["label"].astype(str)
        + "::valuenum"
    )

    train_part = merged[merged["sample_idx"].astype(int).isin(train_sample_idx_set)].copy()
    if train_part.empty:
        return

    g = train_part.groupby(["variable", "itemid_int", "label"], as_index=False).agg(
        train_measurement_count=("value", "size")
    )

    for _, r in g.iterrows():
        key = str(r["variable"])
        measurement_counts[key] += int(r["train_measurement_count"])

    pairs = train_part[["variable", "sample_idx"]].drop_duplicates()
    sample_pairs.append(pairs)


def scan_chartevents_for_candidate_stats(
    cohort_stay,
    stay_ids_needed,
    train_sample_idx_set,
    label_map,
    bad_itemids,
    exclude_obvious,
):
    measurement_counts = defaultdict(int)
    sample_pairs = []

    total = 0
    after_stay = 0
    after_value = 0
    after_time = 0

    reader = pd.read_csv(
        CHARTEVENTS,
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        chunksize=CHUNKSIZE_CHARTEVENTS,
    )

    for chunk in tqdm(reader, desc="chartevents pass 1", unit="chunk"):
        total += len(chunk)

        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce").astype("Int64")

        chunk = chunk[chunk["stay_id"].isin(stay_ids_needed)].copy()
        after_stay += len(chunk)
        if chunk.empty:
            continue

        if exclude_obvious:
            chunk = chunk[~chunk["itemid"].isin(bad_itemids)].copy()
            if chunk.empty:
                continue

        chunk["value"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk[chunk["value"].notna()].copy()
        after_value += len(chunk)
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk[chunk["charttime"].notna()].copy()
        if chunk.empty:
            continue

        merged = chunk.merge(cohort_stay, on="stay_id", how="inner")
        merged = merged[
            (merged["charttime"] >= merged["window_start"])
            & (merged["charttime"] <= merged["window_end"])
        ].copy()

        after_time += len(merged)

        update_counts_from_windowed_chunk(
            merged=merged,
            source="chartevents",
            label_map=label_map,
            measurement_counts=measurement_counts,
            sample_pairs=sample_pairs,
            train_sample_idx_set=train_sample_idx_set,
        )

        del chunk, merged
        gc.collect()

    print("chartevents pass 1 scanned:", total)
    print("after stay:", after_stay)
    print("after value:", after_value)
    print("after time:", after_time)

    return measurement_counts, sample_pairs


def scan_labevents_for_candidate_stats(
    cohort_hadm,
    hadm_ids_needed,
    train_sample_idx_set,
    label_map,
    bad_itemids,
    exclude_obvious,
):
    measurement_counts = defaultdict(int)
    sample_pairs = []

    total = 0
    after_hadm = 0
    after_value = 0
    after_time = 0

    reader = pd.read_csv(
        LABEVENTS,
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=CHUNKSIZE_LABEVENTS,
    )

    for chunk in tqdm(reader, desc="labevents pass 1", unit="chunk"):
        total += len(chunk)

        chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce").astype("Int64")
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce").astype("Int64")

        chunk = chunk[chunk["hadm_id"].isin(hadm_ids_needed)].copy()
        after_hadm += len(chunk)
        if chunk.empty:
            continue

        if exclude_obvious:
            chunk = chunk[~chunk["itemid"].isin(bad_itemids)].copy()
            if chunk.empty:
                continue

        chunk["value"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk[chunk["value"].notna()].copy()
        after_value += len(chunk)
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk[chunk["charttime"].notna()].copy()
        if chunk.empty:
            continue

        merged = chunk.merge(cohort_hadm, on=["subject_id", "hadm_id"], how="inner")
        merged = merged[
            (merged["charttime"] >= merged["window_start"])
            & (merged["charttime"] <= merged["window_end"])
        ].copy()

        after_time += len(merged)

        update_counts_from_windowed_chunk(
            merged=merged,
            source="labevents",
            label_map=label_map,
            measurement_counts=measurement_counts,
            sample_pairs=sample_pairs,
            train_sample_idx_set=train_sample_idx_set,
        )

        del chunk, merged
        gc.collect()

    print("labevents pass 1 scanned:", total)
    print("after hadm:", after_hadm)
    print("after value:", after_value)
    print("after time:", after_time)

    return measurement_counts, sample_pairs


def build_feature_stats(
    chart_counts,
    chart_pairs,
    lab_counts,
    lab_pairs,
    total_train,
    min_train_sample_coverage,
):
    all_counts = {}

    for k, v in chart_counts.items():
        all_counts[k] = all_counts.get(k, 0) + int(v)

    for k, v in lab_counts.items():
        all_counts[k] = all_counts.get(k, 0) + int(v)

    all_pairs = []
    if chart_pairs:
        all_pairs.append(pd.concat(chart_pairs, ignore_index=True))
    if lab_pairs:
        all_pairs.append(pd.concat(lab_pairs, ignore_index=True))

    if all_pairs:
        pair_df = pd.concat(all_pairs, ignore_index=True).drop_duplicates()
        sample_counts = pair_df.groupby("variable")["sample_idx"].nunique().to_dict()
    else:
        sample_counts = {}

    rows = []

    for variable, meas_count in all_counts.items():
        parts = variable.split("::")
        source = parts[0]
        itemid = int(parts[1])
        label = parts[2] if len(parts) > 2 else ""

        sample_count = int(sample_counts.get(variable, 0))

        rows.append({
            "variable": variable,
            "source": source,
            "itemid": itemid,
            "label": label,
            "value_source": "valuenum",
            "train_measurement_count": int(meas_count),
            "train_sample_count": sample_count,
            "train_sample_coverage": sample_count / float(total_train),
        })

    stats = pd.DataFrame(rows)

    if stats.empty:
        raise RuntimeError("No candidate EHR variables found.")

    stats = stats.sort_values(
        ["train_sample_coverage", "train_measurement_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    selected = stats[stats["train_sample_coverage"] >= min_train_sample_coverage].copy()
    selected = selected.reset_index(drop=True)
    selected["feature_index"] = np.arange(len(selected), dtype=int)

    selected = selected[
        [
            "feature_index",
            "variable",
            "source",
            "itemid",
            "label",
            "value_source",
            "train_measurement_count",
            "train_sample_count",
            "train_sample_coverage",
        ]
    ].copy()

    return stats, selected


def fill_tensor_from_chartevents(
    X_raw,
    mask,
    cohort_stay,
    stay_ids_needed,
    selected_itemids,
    feature_to_idx,
    label_map,
):
    reader = pd.read_csv(
        CHARTEVENTS,
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        chunksize=CHUNKSIZE_CHARTEVENTS,
    )

    total_written = 0

    for chunk in tqdm(reader, desc="chartevents pass 2", unit="chunk"):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce").astype("Int64")

        chunk = chunk[
            chunk["stay_id"].isin(stay_ids_needed)
            & chunk["itemid"].isin(selected_itemids)
        ].copy()

        if chunk.empty:
            continue

        chunk["value"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk[chunk["value"].notna()].copy()
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk[chunk["charttime"].notna()].copy()
        if chunk.empty:
            continue

        merged = chunk.merge(cohort_stay, on="stay_id", how="inner")
        merged = merged[
            (merged["charttime"] >= merged["window_start"])
            & (merged["charttime"] <= merged["window_end"])
        ].copy()

        if merged.empty:
            continue

        merged["label"] = merged["itemid"].astype(int).map(label_map).fillna("")
        merged["variable"] = (
            "chartevents::"
            + merged["itemid"].astype(int).astype(str)
            + "::"
            + merged["label"].astype(str)
            + "::valuenum"
        )

        merged = merged[merged["variable"].isin(feature_to_idx)].copy()
        if merged.empty:
            continue

        merged["feature_idx"] = merged["variable"].map(feature_to_idx).astype(int)
        merged["hours_before_index"] = (
            (merged["index_time"] - merged["charttime"]).dt.total_seconds() / 3600.0
        )

        merged["bin_idx"] = (
            N_BINS - 1 - np.floor(merged["hours_before_index"]).astype(int)
        ).clip(0, N_BINS - 1).astype(int)

        agg = (
            merged
            .groupby(["sample_idx", "bin_idx", "feature_idx"], as_index=False)["value"]
            .mean()
        )

        si = agg["sample_idx"].to_numpy(dtype=np.int64)
        ti = agg["bin_idx"].to_numpy(dtype=np.int64)
        fi = agg["feature_idx"].to_numpy(dtype=np.int64)
        vv = agg["value"].to_numpy(dtype=np.float32)

        X_raw[si, ti, fi] = vv
        mask[si, ti, fi] = 1.0

        total_written += len(agg)

        del chunk, merged, agg
        gc.collect()

    print("chartevents tensor entries written:", total_written)


def fill_tensor_from_labevents(
    X_raw,
    mask,
    cohort_hadm,
    hadm_ids_needed,
    selected_itemids,
    feature_to_idx,
    label_map,
):
    reader = pd.read_csv(
        LABEVENTS,
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=CHUNKSIZE_LABEVENTS,
    )

    total_written = 0

    for chunk in tqdm(reader, desc="labevents pass 2", unit="chunk"):
        chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce").astype("Int64")
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce").astype("Int64")

        chunk = chunk[
            chunk["hadm_id"].isin(hadm_ids_needed)
            & chunk["itemid"].isin(selected_itemids)
        ].copy()

        if chunk.empty:
            continue

        chunk["value"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk[chunk["value"].notna()].copy()
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk[chunk["charttime"].notna()].copy()
        if chunk.empty:
            continue

        merged = chunk.merge(cohort_hadm, on=["subject_id", "hadm_id"], how="inner")
        merged = merged[
            (merged["charttime"] >= merged["window_start"])
            & (merged["charttime"] <= merged["window_end"])
        ].copy()

        if merged.empty:
            continue

        merged["label"] = merged["itemid"].astype(int).map(label_map).fillna("")
        merged["variable"] = (
            "labevents::"
            + merged["itemid"].astype(int).astype(str)
            + "::"
            + merged["label"].astype(str)
            + "::valuenum"
        )

        merged = merged[merged["variable"].isin(feature_to_idx)].copy()
        if merged.empty:
            continue

        merged["feature_idx"] = merged["variable"].map(feature_to_idx).astype(int)
        merged["hours_before_index"] = (
            (merged["index_time"] - merged["charttime"]).dt.total_seconds() / 3600.0
        )

        merged["bin_idx"] = (
            N_BINS - 1 - np.floor(merged["hours_before_index"]).astype(int)
        ).clip(0, N_BINS - 1).astype(int)

        agg = (
            merged
            .groupby(["sample_idx", "bin_idx", "feature_idx"], as_index=False)["value"]
            .mean()
        )

        si = agg["sample_idx"].to_numpy(dtype=np.int64)
        ti = agg["bin_idx"].to_numpy(dtype=np.int64)
        fi = agg["feature_idx"].to_numpy(dtype=np.int64)
        vv = agg["value"].to_numpy(dtype=np.float32)

        X_raw[si, ti, fi] = vv
        mask[si, ti, fi] = 1.0

        total_written += len(agg)

        del chunk, merged, agg
        gc.collect()

    print("labevents tensor entries written:", total_written)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cohort_csv", type=str, default=str(COHORT_DEFAULT))
    parser.add_argument("--min_train_sample_coverage", type=float, default=0.005)
    parser.add_argument("--exclude_obvious_admin_or_leakage", action="store_true")

    args = parser.parse_args()

    start = time.time()
    cohort_path = Path(args.cohort_csv)

    section("1. VERIFY PATHS")
    for p in [cohort_path, CHARTEVENTS, D_ITEMS, LABEVENTS, D_LABITEMS]:
        print(p, "| exists=", Path(p).exists())
        require(p)

    section("2. LOAD COHORT")

    cohort = pd.read_csv(cohort_path)

    required = ["sample_id", "subject_id", "hadm_id", "stay_id", "index_time", "label", "split"]
    missing = [c for c in required if c not in cohort.columns]
    if missing:
        raise ValueError(f"Missing cohort columns: {missing}")

    cohort = normalize_id(cohort, ["subject_id", "hadm_id", "stay_id"])
    cohort["index_time"] = pd.to_datetime(cohort["index_time"], errors="coerce")

    cohort = cohort[
        cohort["sample_id"].notna()
        & cohort["subject_id"].notna()
        & cohort["hadm_id"].notna()
        & cohort["stay_id"].notna()
        & cohort["index_time"].notna()
        & cohort["label"].notna()
        & cohort["split"].isin(["train", "val", "test"])
    ].copy()

    cohort = cohort.reset_index(drop=True)
    cohort["sample_idx"] = np.arange(len(cohort), dtype=np.int64)
    cohort["window_start"] = cohort["index_time"] - pd.to_timedelta(WINDOW_HOURS, unit="h")
    cohort["window_end"] = cohort["index_time"]

    print("cohort:", cohort.shape)
    print(pd.crosstab(cohort["split"], cohort["label"]))

    cohort_stay = cohort[
        ["sample_idx", "sample_id", "stay_id", "window_start", "window_end", "index_time"]
    ].copy()

    cohort_hadm = cohort[
        ["sample_idx", "sample_id", "subject_id", "hadm_id", "window_start", "window_end", "index_time"]
    ].copy()

    train_sample_idx_set = set(cohort.loc[cohort["split"] == "train", "sample_idx"].astype(int).tolist())
    total_train = len(train_sample_idx_set)

    stay_ids_needed = set(cohort_stay["stay_id"].dropna().astype(int).tolist())
    hadm_ids_needed = set(cohort_hadm["hadm_id"].dropna().astype(int).tolist())

    section("3. ITEM METADATA")

    item_meta = build_item_metadata()

    chart_label_map = dict(
        item_meta[item_meta["source"] == "chartevents"]
        .drop_duplicates("itemid")
        .set_index("itemid")["label"]
    )

    lab_label_map = dict(
        item_meta[item_meta["source"] == "labevents"]
        .drop_duplicates("itemid")
        .set_index("itemid")["label"]
    )

    chart_bad = set(
        item_meta[
            (item_meta["source"] == "chartevents")
            & (item_meta["obvious_admin_or_leakage"])
        ]["itemid"].dropna().astype(int).tolist()
    )

    lab_bad = set(
        item_meta[
            (item_meta["source"] == "labevents")
            & (item_meta["obvious_admin_or_leakage"])
        ]["itemid"].dropna().astype(int).tolist()
    )

    print("chartevents metadata items:", int((item_meta["source"] == "chartevents").sum()))
    print("labevents metadata items:", int((item_meta["source"] == "labevents").sum()))
    print("chartevents obvious admin/leakage:", len(chart_bad))
    print("labevents obvious admin/leakage:", len(lab_bad))
    print("exclude obvious admin/leakage:", bool(args.exclude_obvious_admin_or_leakage))

    section("4. PASS 1: TRAIN-ONLY CANDIDATE COVERAGE")

    chart_counts, chart_pairs = scan_chartevents_for_candidate_stats(
        cohort_stay=cohort_stay,
        stay_ids_needed=stay_ids_needed,
        train_sample_idx_set=train_sample_idx_set,
        label_map=chart_label_map,
        bad_itemids=chart_bad,
        exclude_obvious=args.exclude_obvious_admin_or_leakage,
    )

    lab_counts, lab_pairs = scan_labevents_for_candidate_stats(
        cohort_hadm=cohort_hadm,
        hadm_ids_needed=hadm_ids_needed,
        train_sample_idx_set=train_sample_idx_set,
        label_map=lab_label_map,
        bad_itemids=lab_bad,
        exclude_obvious=args.exclude_obvious_admin_or_leakage,
    )

    section("5. SELECT CANDIDATES BY TRAIN COVERAGE")

    all_stats, candidate_features = build_feature_stats(
        chart_counts=chart_counts,
        chart_pairs=chart_pairs,
        lab_counts=lab_counts,
        lab_pairs=lab_pairs,
        total_train=total_train,
        min_train_sample_coverage=args.min_train_sample_coverage,
    )

    all_stats.to_csv(OUT_FEATURE_STATS_ALL, index=False)
    candidate_features.to_csv(OUT_FEATURES, index=False)

    print("all observed variables:", len(all_stats))
    print("candidate variables after coverage filter:", len(candidate_features))
    print("min_train_sample_coverage:", args.min_train_sample_coverage)
    print(candidate_features.head(80).to_string(index=False))

    del chart_counts, chart_pairs, lab_counts, lab_pairs, all_stats
    gc.collect()

    section("6. PASS 2: BUILD CANDIDATE TENSOR")

    N = len(cohort)
    T = N_BINS
    F = len(candidate_features)

    X_raw = np.full((N, T, F), np.nan, dtype=np.float32)
    mask = np.zeros((N, T, F), dtype=np.float32)

    feature_to_idx = dict(zip(candidate_features["variable"], candidate_features["feature_index"]))

    selected_chart_itemids = set(
        candidate_features.loc[candidate_features["source"] == "chartevents", "itemid"].astype(int).tolist()
    )

    selected_lab_itemids = set(
        candidate_features.loc[candidate_features["source"] == "labevents", "itemid"].astype(int).tolist()
    )

    print("selected chartevents itemids:", len(selected_chart_itemids))
    print("selected labevents itemids:", len(selected_lab_itemids))

    if selected_chart_itemids:
        fill_tensor_from_chartevents(
            X_raw=X_raw,
            mask=mask,
            cohort_stay=cohort_stay,
            stay_ids_needed=stay_ids_needed,
            selected_itemids=selected_chart_itemids,
            feature_to_idx=feature_to_idx,
            label_map=chart_label_map,
        )

    if selected_lab_itemids:
        fill_tensor_from_labevents(
            X_raw=X_raw,
            mask=mask,
            cohort_hadm=cohort_hadm,
            hadm_ids_needed=hadm_ids_needed,
            selected_itemids=selected_lab_itemids,
            feature_to_idx=feature_to_idx,
            label_map=lab_label_map,
        )

    y = cohort["label"].astype(int).to_numpy()
    split = cohort["split"].astype(str).to_numpy()
    sample_id = cohort["sample_id"].astype(str).to_numpy()

    variables = candidate_features["variable"].astype(str).to_numpy()
    labels = candidate_features["label"].astype(str).to_numpy()
    sources = candidate_features["source"].astype(str).to_numpy()
    itemids = candidate_features["itemid"].astype(str).to_numpy()
    value_sources = candidate_features["value_source"].astype(str).to_numpy()

    print("X_raw:", X_raw.shape)
    print("mask:", mask.shape)
    print("y:", y.shape)
    print("candidate feature count:", F)
    print("observed entries:", int(mask.sum()))
    print("samples with zero EHR:", int((mask.sum(axis=(1, 2)) == 0).sum()))

    section("7. SAVE")

    np.savez_compressed(
        OUT_NPZ,
        X_raw=X_raw.astype(np.float32),
        mask=mask.astype(np.float32),
        y=y.astype(np.int64),
        split=split.astype(str),
        sample_id=sample_id.astype(str),
        variables=variables.astype(str),
        labels=labels.astype(str),
        sources=sources.astype(str),
        itemids=itemids.astype(str),
        value_sources=value_sources.astype(str),
        window_hours=np.array([WINDOW_HOURS], dtype=np.int64),
        n_bins=np.array([N_BINS], dtype=np.int64),
    )

    manifest = {
        "created_at": pd.Timestamp.now().isoformat(),
        "cohort_csv": str(cohort_path),
        "output_npz": str(OUT_NPZ),
        "candidate_features_csv": str(OUT_FEATURES),
        "all_feature_stats_csv": str(OUT_FEATURE_STATS_ALL),
        "window_hours": WINDOW_HOURS,
        "n_bins": N_BINS,
        "min_train_sample_coverage": args.min_train_sample_coverage,
        "exclude_obvious_admin_or_leakage": bool(args.exclude_obvious_admin_or_leakage),
        "shape": list(X_raw.shape),
        "rows": int(N),
        "candidate_features": int(F),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "observed_entries": int(mask.sum()),
        "samples_with_zero_ehr": int((mask.sum(axis=(1, 2)) == 0).sum()),
        "runtime_minutes": round((time.time() - start) / 60.0, 2),
    }

    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print("saved:", OUT_NPZ)
    print("saved:", OUT_FEATURES)
    print("saved:", OUT_FEATURE_STATS_ALL)
    print("saved:", OUT_MANIFEST)
    print("runtime minutes:", round((time.time() - start) / 60.0, 2))


if __name__ == "__main__":
    main()
