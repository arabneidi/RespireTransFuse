from pathlib import Path
import argparse
import re
import time
import gc
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ============================================================
# Build clean CXR-indexed respiratory support escalation cohort
# ============================================================

RANDOM_SEED = 42


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def normalize_int_id(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


def build_cxr_datetime(df):
    if "StudyDate" in df.columns and "StudyTime" in df.columns:
        date = df["StudyDate"].astype(str).str.replace(".0", "", regex=False)
        time_col = df["StudyTime"].astype(str).str.replace(".0", "", regex=False)
        time_col = time_col.str.extract(r"(\d+)")[0].fillna("0").str.zfill(6).str[:6]

        return pd.to_datetime(
            date + time_col,
            format="%Y%m%d%H%M%S",
            errors="coerce",
        )

    for c in ["study_datetime", "StudyDatetime", "charttime"]:
        if c in df.columns:
            return pd.to_datetime(df[c], errors="coerce")

    raise ValueError("No StudyDate/StudyTime or study_datetime column found in CXR metadata.")


def summarize_label(df, label_col, cohort_name):
    s = df[label_col]
    usable = s.notna()
    pos = s.eq(1)
    neg = s.eq(0)

    return {
        "cohort": cohort_name,
        "label_col": label_col,
        "total_rows_before_label_drop": int(len(df)),
        "usable_rows": int(usable.sum()),
        "positive": int(pos.sum()),
        "negative": int(neg.sum()),
        "missing_or_ambiguous": int(s.isna().sum()),
        "positive_rate": float(pos.sum() / max(usable.sum(), 1)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build clean CXR-indexed respiratory deterioration cohort."
    )

    parser.add_argument(
        "--repo_root",
        type=str,
        default="/content/drive/MyDrive/respire-transfuse",
        help="Clean repo root."
    )

    parser.add_argument(
        "--cxr_metadata",
        type=str,
        default=None,
        help="Path to mimic-cxr-2.0.0-metadata.csv.gz."
    )

    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root folder containing local MIMIC-CXR image files."
    )

    parser.add_argument(
        "--icu_dir",
        type=str,
        default=None,
        help="Folder containing MIMIC-IV ICU CSV files."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder for cohort files."
    )

    parser.add_argument(
        "--require_cxr_during_icu",
        action="store_true",
        default=True,
        help="Require CXR timestamp to be inside ICU stay."
    )

    parser.add_argument(
        "--allow_cxr_outside_icu",
        action="store_true",
        help="Disable CXR-during-ICU requirement."
    )

    parser.add_argument(
        "--exclude_prior_resp_event",
        action="store_true",
        default=True,
        help="Exclude CXR rows after prior respiratory support event."
    )

    parser.add_argument(
        "--include_prior_resp_event",
        action="store_true",
        help="Disable prior respiratory event exclusion."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    REPO_ROOT = Path(args.repo_root)

    CXR_METADATA = Path(args.cxr_metadata) if args.cxr_metadata else (
        REPO_ROOT / "data/raw/mimic_cxr/metadata/mimic-cxr-2.0.0-metadata.csv.gz"
    )

    IMAGE_ROOT = Path(args.image_root) if args.image_root else (
        REPO_ROOT / "data/raw/mimic_cxr/images"
    )

    ICU_DIR = Path(args.icu_dir) if args.icu_dir else (
        REPO_ROOT / "data/raw/mimiciv/icu"
    )

    ICUSTAYS = ICU_DIR / "icustays.csv.gz"
    D_ITEMS = ICU_DIR / "d_items.csv.gz"
    PROCEDUREEVENTS = ICU_DIR / "procedureevents.csv.gz"

    OUT_DIR = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "data/processed/cohorts"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_ALL_ROWS = OUT_DIR / "all_rows.csv"
    OUT_NO_PRIOR = OUT_DIR / "no_prior_rows.csv"
    OUT_COHORT = OUT_DIR / "cohort.csv"
    OUT_SUMMARY = OUT_DIR / "cohort_summary.csv"
    OUT_LABEL_COUNTS = OUT_DIR / "label_counts.csv"
    OUT_SPLIT_COUNTS = OUT_DIR / "split_counts.csv"
    OUT_RESP_ITEMIDS = OUT_DIR / "resp_itemids.csv"
    OUT_RESP_EVENTS = OUT_DIR / "resp_events.csv"
    OUT_MANIFEST = OUT_DIR / "cohort_manifest.json"

    REQUIRE_CXR_DURING_ICU = bool(args.require_cxr_during_icu) and not bool(args.allow_cxr_outside_icu)
    EXCLUDE_PRIOR_RESP_EVENT = bool(args.exclude_prior_resp_event) and not bool(args.include_prior_resp_event)

    section("1) Verify paths")

    for name, path in {
        "REPO_ROOT": REPO_ROOT,
        "CXR_METADATA": CXR_METADATA,
        "IMAGE_ROOT": IMAGE_ROOT,
        "ICUSTAYS": ICUSTAYS,
        "D_ITEMS": D_ITEMS,
        "PROCEDUREEVENTS": PROCEDUREEVENTS,
    }.items():
        print(f"{name}: {path} | exists={path.exists()}")
        require(path)

    section("2) Load source tables")

    meta = pd.read_csv(CXR_METADATA)
    icu = pd.read_csv(ICUSTAYS)
    d_items = pd.read_csv(D_ITEMS)
    proc = pd.read_csv(PROCEDUREEVENTS)

    print("CXR metadata:", meta.shape)
    print("ICU stays:", icu.shape)
    print("d_items:", d_items.shape)
    print("procedureevents:", proc.shape)

    section("3) Normalize IDs and times")

    meta = normalize_int_id(meta, ["subject_id", "study_id"])
    icu = normalize_int_id(icu, ["subject_id", "hadm_id", "stay_id"])
    proc = normalize_int_id(proc, ["subject_id", "hadm_id", "stay_id", "itemid"])
    d_items = normalize_int_id(d_items, ["itemid"])

    if "dicom_id" not in meta.columns:
        raise ValueError("CXR metadata must contain dicom_id.")

    meta["dicom_id"] = meta["dicom_id"].astype(str)
    meta["study_datetime"] = build_cxr_datetime(meta)

    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")

    if "starttime" in proc.columns:
        proc["event_time"] = pd.to_datetime(proc["starttime"], errors="coerce")
    elif "charttime" in proc.columns:
        proc["event_time"] = pd.to_datetime(proc["charttime"], errors="coerce")
    else:
        raise ValueError("procedureevents needs starttime or charttime.")

    print("Valid CXR datetime:", int(meta["study_datetime"].notna().sum()))
    print("Valid ICU intime/outtime:", int((icu["intime"].notna() & icu["outtime"].notna()).sum()))
    print("Valid procedure event_time:", int(proc["event_time"].notna().sum()))

    section("4) Index local CXR images")

    image_exts = {".jpg", ".jpeg", ".png"}

    def extract_dicom_id_from_image_path(p: Path):
        """
        Extract dicom_id from original or flattened image filenames.
        """
        stem = p.stem

        if "_" in stem:
            candidate = stem.split("_")[-1]
            if "-" in candidate:
                return candidate

        return stem

    image_files = []

    for p in tqdm(list(IMAGE_ROOT.iterdir()), desc="Scanning direct image folder"):
        if p.is_file() and p.suffix.lower() in image_exts:
            image_files.append(p)

    if len(image_files) == 0:
        print("Direct image scan found 0 files. Falling back to recursive scan.")
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            matches = list(tqdm(IMAGE_ROOT.rglob(ext), desc=f"Finding {ext} recursively"))
            print(ext, len(matches))
            image_files.extend(matches)

    dicom_to_path = {}
    duplicates = set()

    for p in tqdm(image_files, desc="Building image index"):
        key = extract_dicom_id_from_image_path(p)

        if key in dicom_to_path:
            duplicates.add(key)
        else:
            dicom_to_path[key] = str(p)

    meta["image_path"] = meta["dicom_id"].map(dicom_to_path)
    meta["image_exists"] = meta["image_path"].notna()

    print("Total local image files:", len(image_files))
    print("Unique dicom IDs from image files:", len(dicom_to_path))
    print("Duplicate dicom IDs:", len(duplicates))
    print("Metadata rows with local image:", int(meta["image_exists"].sum()))

    print("\nExample image-id mappings:")
    for i, (k, v) in enumerate(list(dicom_to_path.items())[:10]):
        print(f"{i:02d} | {k} -> {v}")

    section("5) Filter frontal local CXR")

    if "ViewPosition" in meta.columns:
        meta["ViewPosition_clean"] = meta["ViewPosition"].astype(str).str.upper().str.strip()

        print("ViewPosition counts:")
        print(meta["ViewPosition_clean"].value_counts(dropna=False).head(30))

        cxr = meta[
            meta["ViewPosition_clean"].isin(["AP", "PA"])
            & meta["image_exists"]
            & meta["study_datetime"].notna()
        ].copy()
    else:
        meta["ViewPosition_clean"] = ""
        cxr = meta[
            meta["image_exists"]
            & meta["study_datetime"].notna()
        ].copy()

    print("Filtered CXR rows:", len(cxr))
    print("Unique subjects:", cxr["subject_id"].nunique())
    print("Unique studies:", cxr["study_id"].nunique())
    print("Unique dicoms:", cxr["dicom_id"].nunique())

    section("6) Link CXR to ICU stays")

    icu_valid = icu[
        icu["subject_id"].notna()
        & icu["hadm_id"].notna()
        & icu["stay_id"].notna()
        & icu["intime"].notna()
        & icu["outtime"].notna()
    ].copy()

    cxr_icu = cxr.merge(
        icu_valid[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]],
        on="subject_id",
        how="inner",
    )

    print("CXR x ICU before time filter:", len(cxr_icu))

    if REQUIRE_CXR_DURING_ICU:
        cxr_icu = cxr_icu[
            (cxr_icu["study_datetime"] >= cxr_icu["intime"])
            & (cxr_icu["study_datetime"] <= cxr_icu["outtime"])
        ].copy()

    print("CXR x ICU after time filter:", len(cxr_icu))
    print("Unique subjects:", cxr_icu["subject_id"].nunique())
    print("Unique stays:", cxr_icu["stay_id"].nunique())
    print("Unique dicoms:", cxr_icu["dicom_id"].nunique())

    section("7) Select respiratory intervention events")

    item_label_col = None
    for c in ["label", "Label", "LABEL"]:
        if c in d_items.columns:
            item_label_col = c
            break

    if item_label_col is None:
        raise ValueError("Could not find label column in d_items.")

    d_items_small = d_items[["itemid", item_label_col]].copy()
    d_items_small = d_items_small.rename(columns={item_label_col: "item_label"})
    d_items_small["item_label_clean"] = d_items_small["item_label"].apply(clean_text)

    proc_labeled = proc.merge(d_items_small, on="itemid", how="left")

    resp_patterns = [
        r"\bintubat",
        r"mechanical ventilation",
        r"invasive ventilation",
        r"ventilator",
        r"\bventilation\b",
        r"\bcpap\b",
        r"\bbipap\b",
        r"non.?invasive",
        r"high flow",
        r"high-flow",
        r"\bhfnc\b",
        r"tracheostomy",
        r"\btrach\b",
    ]

    pattern = re.compile("|".join(resp_patterns), flags=re.IGNORECASE)

    proc_labeled["is_resp_event"] = proc_labeled["item_label_clean"].apply(
        lambda x: bool(pattern.search(str(x)))
    )

    resp_items = (
        proc_labeled.loc[proc_labeled["is_resp_event"], ["itemid", "item_label"]]
        .drop_duplicates()
        .sort_values(["item_label", "itemid"])
        .reset_index(drop=True)
    )

    resp_events = proc_labeled[
        proc_labeled["is_resp_event"]
        & proc_labeled["stay_id"].notna()
        & proc_labeled["event_time"].notna()
    ].copy()

    resp_events = resp_events[
        ["subject_id", "hadm_id", "stay_id", "itemid", "item_label", "event_time"]
    ].drop_duplicates()

    resp_items.to_csv(OUT_RESP_ITEMIDS, index=False)
    resp_events.to_csv(OUT_RESP_EVENTS, index=False)

    print("Selected respiratory itemids:")
    print(resp_items.to_string(index=False))
    print("Respiratory events:", len(resp_events))
    print("Top labels:")
    print(resp_events["item_label"].value_counts().head(30))

    section("8) Compute prior/future event timing")

    events_by_stay = {
        int(stay_id): g.sort_values("event_time")
        for stay_id, g in tqdm(resp_events.groupby("stay_id"), desc="Grouping events")
    }

    def get_event_timing(row):
        stay_id = row["stay_id"]
        t = row["study_datetime"]

        if pd.isna(stay_id) or pd.isna(t):
            return pd.Series({
                "had_prior_resp_event": False,
                "first_prior_resp_event_time": pd.NaT,
                "first_future_resp_event_time": pd.NaT,
                "hours_to_future_resp_event": np.nan,
            })

        ev = events_by_stay.get(int(stay_id), None)

        if ev is None or len(ev) == 0:
            return pd.Series({
                "had_prior_resp_event": False,
                "first_prior_resp_event_time": pd.NaT,
                "first_future_resp_event_time": pd.NaT,
                "hours_to_future_resp_event": np.nan,
            })

        prior = ev[ev["event_time"] <= t]
        future = ev[ev["event_time"] > t]

        prior_time = prior["event_time"].min() if len(prior) else pd.NaT
        future_time = future["event_time"].min() if len(future) else pd.NaT

        if pd.isna(future_time):
            hours = np.nan
        else:
            hours = (future_time - t).total_seconds() / 3600.0

        return pd.Series({
            "had_prior_resp_event": bool(len(prior) > 0),
            "first_prior_resp_event_time": prior_time,
            "first_future_resp_event_time": future_time,
            "hours_to_future_resp_event": hours,
        })

    timing = cxr_icu.progress_apply(get_event_timing, axis=1)

    cxr_icu = pd.concat(
        [cxr_icu.reset_index(drop=True), timing.reset_index(drop=True)],
        axis=1,
    )

    print("Rows:", len(cxr_icu))
    print("Prior respiratory event rows:", int(cxr_icu["had_prior_resp_event"].sum()))
    print("Future respiratory event rows:", int(cxr_icu["first_future_resp_event_time"].notna().sum()))
    print(cxr_icu["hours_to_future_resp_event"].describe())

    section("9) Create label variants")

    h = cxr_icu["hours_to_future_resp_event"]

    for window in [12, 24, 48, 72]:
        cxr_icu[f"label_event_{window}h"] = ((h > 0) & (h <= window)).astype(int)

    cxr_icu["label_48h_stable72h"] = np.nan
    cxr_icu.loc[(h > 0) & (h <= 48), "label_48h_stable72h"] = 1
    cxr_icu.loc[h.isna() | (h > 72), "label_48h_stable72h"] = 0

    cxr_icu["label_24h_stable72h"] = np.nan
    cxr_icu.loc[(h > 0) & (h <= 24), "label_24h_stable72h"] = 1
    cxr_icu.loc[h.isna() | (h > 72), "label_24h_stable72h"] = 0

    if EXCLUDE_PRIOR_RESP_EVENT:
        cxr_clean = cxr_icu[~cxr_icu["had_prior_resp_event"]].copy()
    else:
        cxr_clean = cxr_icu.copy()

    label_cols = [
        "label_event_12h",
        "label_event_24h",
        "label_event_48h",
        "label_event_72h",
        "label_24h_stable72h",
        "label_48h_stable72h",
    ]

    label_summary = []
    for label_col in label_cols:
        label_summary.append(summarize_label(cxr_icu, label_col, "all_cxr_icu"))
        label_summary.append(summarize_label(cxr_clean, label_col, "exclude_prior_resp_event"))

    label_summary = pd.DataFrame(label_summary)

    print("Rows before prior exclusion:", len(cxr_icu))
    print("Rows after prior exclusion:", len(cxr_clean))
    print(label_summary.to_string(index=False))

    section("10) Patient-level train/val/test split")

    rng = np.random.default_rng(RANDOM_SEED)

    subjects = np.array(sorted(cxr_clean["subject_id"].dropna().astype(int).unique()))
    rng.shuffle(subjects)

    n = len(subjects)
    train_subjects = set(subjects[: int(0.70 * n)])
    val_subjects = set(subjects[int(0.70 * n): int(0.85 * n)])
    test_subjects = set(subjects[int(0.85 * n):])

    def assign_split(subject_id):
        if pd.isna(subject_id):
            return "unknown"

        sid = int(subject_id)

        if sid in train_subjects:
            return "train"
        if sid in val_subjects:
            return "val"
        if sid in test_subjects:
            return "test"

        return "unknown"

    cxr_clean["split"] = cxr_clean["subject_id"].apply(assign_split)

    split_summary = []

    for label_col in label_cols:
        for split in ["train", "val", "test"]:
            d = cxr_clean[cxr_clean["split"] == split]
            s = d[label_col]
            usable = s.notna()

            split_summary.append({
                "label_col": label_col,
                "split": split,
                "rows": int(len(d)),
                "usable": int(usable.sum()),
                "positive": int(s.eq(1).sum()),
                "negative": int(s.eq(0).sum()),
                "missing_or_ambiguous": int(s.isna().sum()),
                "positive_rate": float(s.eq(1).sum() / max(usable.sum(), 1)),
            })

    split_summary = pd.DataFrame(split_summary)

    print("Unique subjects:", n)
    print("Train subjects:", len(train_subjects))
    print("Val subjects:", len(val_subjects))
    print("Test subjects:", len(test_subjects))
    print(split_summary.to_string(index=False))

    section("11) Create final cohort")

    label_source_col = "label_48h_stable72h"

    cohort = cxr_clean[cxr_clean[label_source_col].notna()].copy()
    cohort["label"] = cohort[label_source_col].astype(int)

    cohort["sample_id"] = (
        cohort["subject_id"].astype(str)
        + "_"
        + cohort["stay_id"].astype(str)
        + "_"
        + cohort["study_id"].astype(str)
        + "_"
        + cohort["dicom_id"].astype(str)
    )

    if cohort["sample_id"].duplicated().any():
        dup = cohort.loc[cohort["sample_id"].duplicated(), "sample_id"].head(20).tolist()
        raise ValueError(f"Duplicate sample_id found. Examples: {dup}")

    cohort["verified_image_path"] = cohort["image_path"]
    cohort["image_exists"] = True
    cohort["image_decode_ok"] = True

    cohort["index_time"] = cohort["study_datetime"]
    cohort["prediction_window_hours"] = 48
    cohort["stable_negative_window_hours"] = 72
    cohort["label_definition"] = (
        "1=selected_respiratory_intervention_within_48h_after_CXR; "
        "0=no_selected_respiratory_intervention_within_72h_after_CXR; "
        "prior_event_CXRs_excluded; "
        "events_between_48h_and_72h_excluded"
    )

    front_cols = [
        "sample_id",
        "subject_id",
        "hadm_id",
        "stay_id",
        "study_id",
        "dicom_id",
        "split",
        "label",
        "verified_image_path",
        "image_path",
        "image_exists",
        "image_decode_ok",
        "index_time",
        "study_datetime",
        "intime",
        "outtime",
        "hours_to_future_resp_event",
        "first_future_resp_event_time",
        "had_prior_resp_event",
        "ViewPosition",
        "ViewPosition_clean",
        "prediction_window_hours",
        "stable_negative_window_hours",
        "label_definition",
    ]

    front_cols = [c for c in front_cols if c in cohort.columns]
    other_cols = [c for c in cohort.columns if c not in front_cols]
    cohort = cohort[front_cols + other_cols].copy()

    summary_rows = []
    for split in ["train", "val", "test"]:
        d = cohort[cohort["split"] == split]
        summary_rows.append({
            "split": split,
            "rows": int(len(d)),
            "subjects": int(d["subject_id"].nunique()),
            "stays": int(d["stay_id"].nunique()),
            "studies": int(d["study_id"].nunique()),
            "dicoms": int(d["dicom_id"].nunique()),
            "positive": int((d["label"] == 1).sum()),
            "negative": int((d["label"] == 0).sum()),
            "positive_rate": float((d["label"] == 1).mean()) if len(d) else np.nan,
        })

    final_summary = pd.DataFrame(summary_rows)

    section("12) Save files")

    cxr_icu.to_csv(OUT_ALL_ROWS, index=False)
    cxr_clean.to_csv(OUT_NO_PRIOR, index=False)
    cohort.to_csv(OUT_COHORT, index=False)
    final_summary.to_csv(OUT_SUMMARY, index=False)
    label_summary.to_csv(OUT_LABEL_COUNTS, index=False)
    split_summary.to_csv(OUT_SPLIT_COUNTS, index=False)

    manifest = {
        "created_at": pd.Timestamp.now().isoformat(),
        "output_dir": str(OUT_DIR),
        "cohort_csv": str(OUT_COHORT),
        "all_rows_csv": str(OUT_ALL_ROWS),
        "no_prior_rows_csv": str(OUT_NO_PRIOR),
        "resp_itemids_csv": str(OUT_RESP_ITEMIDS),
        "resp_events_csv": str(OUT_RESP_EVENTS),
        "random_seed": RANDOM_SEED,
        "label": "label_48h_stable72h renamed to label",
        "positive": "selected respiratory intervention within 48h after CXR",
        "negative": "no selected respiratory intervention within 72h after CXR",
        "excluded": "prior-event CXR rows and ambiguous 48-72h future-event rows",
        "counts": {
            "cohort_rows": int(len(cohort)),
            "positive": int((cohort["label"] == 1).sum()),
            "negative": int((cohort["label"] == 0).sum()),
        },
        "split_summary": final_summary.to_dict(orient="records"),
    }

    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Saved:")
    print(OUT_ALL_ROWS)
    print(OUT_NO_PRIOR)
    print(OUT_COHORT)
    print(OUT_SUMMARY)
    print(OUT_LABEL_COUNTS)
    print(OUT_SPLIT_COUNTS)
    print(OUT_RESP_ITEMIDS)
    print(OUT_RESP_EVENTS)
    print(OUT_MANIFEST)

    print("\nFinal cohort:")
    print("rows:", len(cohort))
    print(cohort["label"].value_counts().sort_index())
    print(pd.crosstab(cohort["split"], cohort["label"]))

    print("\nRuntime minutes:", round((time.time() - t_start) / 60.0, 2))

    del meta, icu, d_items, proc, proc_labeled
    gc.collect()


if __name__ == "__main__":
    tqdm.pandas()
    main()
