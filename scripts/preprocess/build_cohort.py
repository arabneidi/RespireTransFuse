from pathlib import Path
import argparse
import time
import gc
import json
import hashlib

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DEFAULT_RANDOM_SEED = 42

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

def build_prevalence_balanced_patient_split(
    df,
    label_col,
    seed,
    search_iterations,
    max_prevalence_gap,
    max_size_deviation,
    existing_assignments_path=None,
):
    target_fractions = np.array(
        [0.70, 0.15, 0.15],
        dtype=np.float64,
    )

    split_names = np.array(
        ["train", "val", "test"],
        dtype=object,
    )

    source = df.loc[
        df[label_col].notna(),
        ["subject_id", label_col],
    ].copy()

    source = source.dropna(
        subset=["subject_id", label_col]
    )

    source["subject_id"] = pd.to_numeric(
        source["subject_id"],
        errors="raise",
    ).astype(np.int64)

    source[label_col] = pd.to_numeric(
        source[label_col],
        errors="raise",
    ).astype(np.int64)

    labels = set(
        source[label_col]
        .unique()
        .tolist()
    )

    if not labels.issubset({0, 1}):
        raise RuntimeError(
            f"{label_col} must contain only 0 and 1. "
            f"Found: {sorted(labels)}"
        )

    if source.empty:
        raise RuntimeError(
            "No usable final-label rows are available "
            "for splitting."
        )

    subject_stats = (
        source
        .groupby(
            "subject_id",
            sort=True,
        )
        .agg(
            usable_rows=(
                label_col,
                "size",
            ),
            positives=(
                label_col,
                "sum",
            ),
        )
        .sort_index()
    )

    subject_stats["usable_rows"] = (
        subject_stats["usable_rows"]
        .astype(np.int64)
    )

    subject_stats["positives"] = (
        subject_stats["positives"]
        .astype(np.int64)
    )

    if existing_assignments_path is not None and Path(existing_assignments_path).is_file():
        previous = pd.read_csv(existing_assignments_path)
        required = {"subject_id", "split"}

        if not required.issubset(previous.columns):
            raise RuntimeError("Existing patient split file is missing required columns.")

        previous = previous[["subject_id", "split"]].copy()
        previous["subject_id"] = pd.to_numeric(
            previous["subject_id"], errors="raise"
        ).astype(np.int64)
        previous["split"] = previous["split"].astype(str).str.strip().str.lower()

        if previous["subject_id"].duplicated().any():
            raise RuntimeError("Existing patient split file contains duplicate patients.")

        if not previous["split"].isin(["train", "val", "test"]).all():
            raise RuntimeError("Existing patient split file contains invalid splits.")

        split_map = dict(zip(previous["subject_id"], previous["split"]))
        new_subjects = sorted(set(subject_stats.index) - set(split_map))

        for subject_id in new_subjects:
            digest = hashlib.sha256(
                f"{seed}:{subject_id}".encode("utf-8")
            ).digest()

            value = int.from_bytes(digest[:8], "big") / 2**64

            split_map[int(subject_id)] = (
                "train" if value < 0.70
                else "val" if value < 0.85
                else "test"
            )

        assignments = subject_stats.reset_index().copy()
        assignments["negatives"] = (
            assignments["usable_rows"] - assignments["positives"]
        )
        assignments["subject_prevalence"] = (
            assignments["positives"] / assignments["usable_rows"]
        )
        assignments["split"] = assignments["subject_id"].map(split_map)

        if assignments["split"].isna().any():
            raise RuntimeError("Unassigned patient in current cohort.")

        summary = assignments.groupby("split").agg(
            rows=("usable_rows", "sum"),
            positives=("positives", "sum"),
            subjects=("subject_id", "size"),
        ).reindex(["train", "val", "test"])

        if summary.isna().any().any():
            raise RuntimeError("An existing split is empty after cohort selection.")

        summary["negatives"] = summary["rows"] - summary["positives"]
        summary["prevalence"] = summary["positives"] / summary["rows"]
        summary["row_fraction"] = summary["rows"] / summary["rows"].sum()

        prevalence_gap = float(
            summary["prevalence"].max() - summary["prevalence"].min()
        )

        size_deviation = float(
            np.abs(
                summary["row_fraction"].to_numpy() - target_fractions
            ).max()
        )

        if prevalence_gap > max_prevalence_gap:
            raise RuntimeError("Existing splits exceed the prevalence tolerance.")

        if size_deviation > max_size_deviation:
            raise RuntimeError("Existing splits exceed the size tolerance.")

        info = {
            "name": "patient_level_existing_assignments",
            "label_column": str(label_col),
            "seed": int(seed),
            "search_iterations": 0,
            "selected_search_iteration": 0,
            "new_subjects_assigned": int(len(new_subjects)),
            "overall_prevalence": float(source[label_col].mean()),
            "maximum_prevalence_gap": prevalence_gap,
            "maximum_row_size_deviation": size_deviation,
            "allowed_prevalence_gap": float(max_prevalence_gap),
            "allowed_row_size_deviation": float(max_size_deviation),
            "target_row_fractions": {
                "train": 0.70,
                "val": 0.15,
                "test": 0.15,
            },
        }

        return split_map, summary, assignments, info

    subjects = (
        subject_stats.index
        .to_numpy(dtype=np.int64)
    )

    subject_rows = (
        subject_stats["usable_rows"]
        .to_numpy(dtype=np.int64)
    )

    subject_positives = (
        subject_stats["positives"]
        .to_numpy(dtype=np.int64)
    )

    n_subjects = len(subjects)

    n_train = int(
        np.floor(
            0.70 * n_subjects
        )
    )

    n_val = int(
        np.floor(
            0.15 * n_subjects
        )
    )

    n_test = (
        n_subjects
        - n_train
        - n_val
    )

    if min(
        n_train,
        n_val,
        n_test,
    ) <= 0:
        raise RuntimeError(
            "Invalid patient split sizes."
        )

    total_rows = int(
        subject_rows.sum()
    )

    total_positives = int(
        subject_positives.sum()
    )

    total_negatives = (
        total_rows
        - total_positives
    )

    if (
        total_rows <= 0
        or total_positives <= 0
        or total_negatives <= 0
    ):
        raise RuntimeError(
            "The final cohort must contain "
            "both outcome classes."
        )

    overall_prevalence = (
        total_positives
        / total_rows
    )

    rng = np.random.default_rng(
        int(seed)
    )

    best = None

    for search_index in range(
        int(search_iterations)
    ):
        order = rng.permutation(
            n_subjects
        )

        split_indices = (
            order[:n_train],
            order[
                n_train:
                n_train + n_val
            ],
            order[
                n_train + n_val:
            ],
        )

        rows = np.array(
            [
                int(
                    subject_rows[
                        indices
                    ].sum()
                )
                for indices
                in split_indices
            ],
            dtype=np.int64,
        )

        positives = np.array(
            [
                int(
                    subject_positives[
                        indices
                    ].sum()
                )
                for indices
                in split_indices
            ],
            dtype=np.int64,
        )

        negatives = (
            rows
            - positives
        )

        if (
            np.any(rows <= 0)
            or np.any(positives <= 0)
            or np.any(negatives <= 0)
        ):
            continue

        prevalence = (
            positives
            / rows
        )

        row_fractions = (
            rows
            / total_rows
        )

        prevalence_gap = float(
            prevalence.max()
            - prevalence.min()
        )

        prevalence_target_error = float(
            np.abs(
                prevalence
                - overall_prevalence
            ).max()
        )

        size_deviation = float(
            np.abs(
                row_fractions
                - target_fractions
            ).max()
        )

        size_valid = (
            size_deviation
            <= float(
                max_size_deviation
            )
        )

        key = (
            0 if size_valid else 1,
            (
                prevalence_gap
                if size_valid
                else size_deviation
            ),
            (
                prevalence_target_error
                if size_valid
                else prevalence_gap
            ),
            (
                size_deviation
                if size_valid
                else prevalence_target_error
            ),
            search_index,
        )

        if (
            best is None
            or key < best["key"]
        ):
            best = {
                "key": key,
                "order": order.copy(),
                "search_index": int(
                    search_index
                ),
                "rows": rows.copy(),
                "positives": (
                    positives.copy()
                ),
                "negatives": (
                    negatives.copy()
                ),
                "prevalence": (
                    prevalence.copy()
                ),
                "row_fractions": (
                    row_fractions.copy()
                ),
                "prevalence_gap": (
                    prevalence_gap
                ),
                "prevalence_target_error": (
                    prevalence_target_error
                ),
                "size_deviation": (
                    size_deviation
                ),
            }

    if best is None:
        raise RuntimeError(
            "No valid patient-level split "
            "candidate was found."
        )

    if (
        best["size_deviation"]
        > float(max_size_deviation)
    ):
        raise RuntimeError(
            "No seeded split candidate satisfied "
            "the row-size tolerance. "
            f"Best={best['size_deviation']:.6f}; "
            f"allowed={float(max_size_deviation):.6f}."
        )

    if (
        best["prevalence_gap"]
        > float(max_prevalence_gap)
    ):
        raise RuntimeError(
            "No seeded split candidate satisfied "
            "the prevalence tolerance. "
            f"Best={best['prevalence_gap']:.6f}; "
            f"allowed={float(max_prevalence_gap):.6f}."
        )

    ordered_subjects = subjects[
        best["order"]
    ]

    train_subjects = ordered_subjects[
        :n_train
    ]

    val_subjects = ordered_subjects[
        n_train:
        n_train + n_val
    ]

    test_subjects = ordered_subjects[
        n_train + n_val:
    ]

    split_map = {}

    for name, ids in zip(
        split_names,
        [
            train_subjects,
            val_subjects,
            test_subjects,
        ],
    ):
        for subject_id in ids:
            split_map[
                int(subject_id)
            ] = str(name)

    assignments = (
        subject_stats
        .reset_index()
        .copy()
    )

    assignments["negatives"] = (
        assignments["usable_rows"]
        - assignments["positives"]
    )

    assignments[
        "subject_prevalence"
    ] = (
        assignments["positives"]
        / assignments["usable_rows"]
    )

    assignments["split"] = (
        assignments["subject_id"]
        .map(split_map)
    )

    assignments = (
        assignments
        .sort_values(
            "subject_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    if assignments[
        "split"
    ].isna().any():
        raise RuntimeError(
            "At least one eligible patient "
            "has no split assignment."
        )

    summary = pd.DataFrame(
        {
            "split": split_names,
            "rows": best["rows"],
            "positives": (
                best["positives"]
            ),
            "negatives": (
                best["negatives"]
            ),
            "subjects": [
                n_train,
                n_val,
                n_test,
            ],
            "prevalence": (
                best["prevalence"]
            ),
            "row_fraction": (
                best["row_fractions"]
            ),
        }
    ).set_index("split")

    info = {
        "name": "seeded_patient_level_prevalence_search_v1",
        "label_column": str(
            label_col
        ),
        "seed": int(seed),
        "search_iterations": int(
            search_iterations
        ),
        "selected_search_iteration": int(
            best["search_index"]
        ),
        "overall_prevalence": float(
            overall_prevalence
        ),
        "maximum_prevalence_gap": float(
            best["prevalence_gap"]
        ),
        "maximum_row_size_deviation": float(
            best["size_deviation"]
        ),
        "allowed_prevalence_gap": float(
            max_prevalence_gap
        ),
        "allowed_row_size_deviation": float(
            max_size_deviation
        ),
        "target_row_fractions": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
        },
    }

    return (
        split_map,
        summary,
        assignments,
        info,
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build clean CXR-indexed respiratory deterioration cohort."
    )

    parser.add_argument(
        "--repo_root",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2]
        ),
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

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Seed for reproducible patient-level split search."
    )

    parser.add_argument(
        "--split_search_iterations",
        type=int,
        default=50000,
        help="Number of seeded patient split candidates to evaluate."
    )

    parser.add_argument(
        "--max_prevalence_gap",
        type=float,
        default=0.0025,
        help="Maximum prevalence difference across train, val, and test."
    )

    parser.add_argument(
        "--max_split_size_deviation",
        type=float,
        default=0.01,
        help="Maximum absolute row-fraction deviation from 70/15/15."
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
    ADMISSIONS = REPO_ROOT / "data/raw/mimiciv/hosp/admissions.csv.gz"

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
    OUT_SUBJECT_SPLITS = OUT_DIR / "subject_split_assignments.csv"

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
        "ADMISSIONS": ADMISSIONS,
    }.items():
        print(f"{name}: {path} | exists={path.exists()}")
        require(path)

    section("2) Load source tables")

    meta = pd.read_csv(CXR_METADATA)
    icu = pd.read_csv(ICUSTAYS)
    d_items = pd.read_csv(D_ITEMS)
    proc = pd.read_csv(PROCEDUREEVENTS)
    adm = pd.read_csv(ADMISSIONS)

    print("CXR metadata:", meta.shape)
    print("ICU stays:", icu.shape)
    print("d_items:", d_items.shape)
    print("procedureevents:", proc.shape)
    print("admissions:", adm.shape)

    section("3) Normalize IDs and times")

    meta = normalize_int_id(meta, ["subject_id", "study_id"])
    icu = normalize_int_id(icu, ["subject_id", "hadm_id", "stay_id"])
    proc = normalize_int_id(proc, ["subject_id", "hadm_id", "stay_id", "itemid"])
    adm = normalize_int_id(adm, ["subject_id", "hadm_id"])
    d_items = normalize_int_id(d_items, ["itemid"])

    if "dicom_id" not in meta.columns:
        raise ValueError("CXR metadata must contain dicom_id.")

    meta["dicom_id"] = meta["dicom_id"].astype(str)
    meta["study_datetime"] = build_cxr_datetime(meta)

    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")

    for c in ["admittime", "dischtime", "deathtime"]:
        if c in adm.columns:
            adm[c] = pd.to_datetime(adm[c], errors="coerce")

    if "starttime" in proc.columns:
        proc["event_time"] = pd.to_datetime(proc["starttime"], errors="coerce")
    elif "charttime" in proc.columns:
        proc["event_time"] = pd.to_datetime(proc["charttime"], errors="coerce")
    else:
        raise ValueError("procedureevents needs starttime or charttime.")

    if "endtime" not in proc.columns:
        raise ValueError("procedureevents needs endtime.")

    proc["endtime"] = pd.to_datetime(proc["endtime"], errors="coerce")

    print("Valid CXR datetime:", int(meta["study_datetime"].notna().sum()))
    print("Valid ICU intime/outtime:", int((icu["intime"].notna() & icu["outtime"].notna()).sum()))
    print("Valid procedure event_time:", int(proc["event_time"].notna().sum()))

    section("4) Index local CXR images")

    image_exts = {".jpg", ".jpeg", ".png"}

    def extract_dicom_id_from_image_path(p: Path):

        stem = p.stem

        if "_" in stem:
            candidate = stem.split("_")[-1]
            if "-" in candidate:
                return candidate

        return stem

    image_files = []

    direct_candidates = sorted(
        IMAGE_ROOT.iterdir(),
        key=lambda path: path.as_posix(),
    )

    for p in tqdm(
        direct_candidates,
        desc="Scanning direct image folder",
    ):
        if (
            p.is_file()
            and p.suffix.lower() in image_exts
        ):
            image_files.append(p)

    if len(image_files) == 0:
        print(
            "Direct image scan found 0 files. "
            "Falling back to recursive scan."
        )

        for ext in [
            "*.jpg",
            "*.jpeg",
            "*.png",
        ]:
            matches = sorted(
                IMAGE_ROOT.rglob(ext),
                key=lambda path: path.as_posix(),
            )

            print(
                ext,
                len(matches),
            )

            image_files.extend(
                matches
            )

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

    adm_columns = [
        c
        for c in [
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "hospital_expire_flag",
            "discharge_location",
        ]
        if c in adm.columns
    ]

    adm_small = (
        adm[adm_columns]
        .drop_duplicates("hadm_id")
        .copy()
    )

    cxr_icu = cxr_icu.merge(
        adm_small,
        on="hadm_id",
        how="left",
        validate="many_to_one",
    )

    print(
        "CXR rows with hospital discharge time:",
        int(cxr_icu["dischtime"].notna().sum())
        if "dischtime" in cxr_icu.columns else 0,
    )

    valid_index = (
        (
            cxr_icu["admittime"].isna()
            | (cxr_icu["study_datetime"] >= cxr_icu["admittime"])
        )
        & (
            cxr_icu["dischtime"].isna()
            | (cxr_icu["study_datetime"] <= cxr_icu["dischtime"])
        )
        & (
            cxr_icu["deathtime"].isna()
            | (cxr_icu["study_datetime"] <= cxr_icu["deathtime"])
        )
    )

    print("X-rays outside recorded hospitalization:", int((~valid_index).sum()))

    cxr_icu = cxr_icu.loc[valid_index].copy()

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

    respiratory_itemids = {224385, 225792, 225794}

    proc_labeled["is_resp_event"] = proc_labeled["itemid"].isin(
        respiratory_itemids
    )

    matched_itemids = set(
        proc_labeled.loc[
            proc_labeled["is_resp_event"], "itemid"
        ].dropna().astype(int)
    )

    if matched_itemids != respiratory_itemids:
        raise RuntimeError("Expected respiratory intervention codes were not found.")

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
        for stay_id, g in tqdm(
            resp_events.groupby("stay_id"),
            desc="Grouping respiratory events by ICU stay",
        )
    }

    events_by_hadm = {
        int(hadm_id): g.sort_values("event_time")
        for hadm_id, g in tqdm(
            resp_events[
                resp_events["hadm_id"].notna()
            ].groupby("hadm_id"),
            desc="Grouping respiratory events by hospital admission",
        )
    }

    active_intervals_by_hadm = {
        int(hadm_id): g[["event_time", "endtime"]].copy()
        for hadm_id, g in proc_labeled.loc[
            proc_labeled["is_resp_event"]
            & proc_labeled["hadm_id"].notna()
            & proc_labeled["event_time"].notna()
        ].groupby("hadm_id")
    }

    def get_event_timing(row):
        stay_id = row["stay_id"]
        hadm_id = row["hadm_id"]
        t = row["study_datetime"]

        if pd.isna(t):
            return pd.Series({
                "had_prior_resp_event": False,
                "had_active_resp_support": False,
                "first_prior_resp_event_time": pd.NaT,
                "first_future_resp_event_time": pd.NaT,
                "hours_to_future_resp_event": np.nan,
            })

        prior_time = pd.NaT
        had_prior = False

        if pd.notna(stay_id):
            prior_events = events_by_stay.get(
                int(stay_id),
                None,
            )

            if prior_events is not None and len(prior_events) > 0:
                prior = prior_events[
                    prior_events["event_time"] <= t
                ]

                if len(prior) > 0:
                    prior_time = prior["event_time"].min()
                    had_prior = True

        future_time = pd.NaT
        had_active = False

        if pd.notna(hadm_id):
            intervals = active_intervals_by_hadm.get(int(hadm_id))

            if intervals is not None:
                had_active = bool((
                    (intervals["event_time"] <= t)
                    & (
                        intervals["endtime"].isna()
                        | (intervals["endtime"] > t)
                    )
                ).any())

            future_events = events_by_hadm.get(
                int(hadm_id),
                None,
            )

            if future_events is not None and len(future_events) > 0:
                future = future_events[
                    future_events["event_time"] > t
                ]

                if pd.notna(row["dischtime"]):
                    future = future[
                        future["event_time"] <= row["dischtime"]
                    ]

                if pd.notna(row["deathtime"]):
                    future = future[
                        future["event_time"] <= row["deathtime"]
                    ]

                if len(future) > 0:
                    future_time = future["event_time"].min()

        if pd.isna(future_time):
            hours = np.nan
        else:
            hours = (
                future_time - t
            ).total_seconds() / 3600.0

        return pd.Series({
            "had_prior_resp_event": bool(had_prior),
            "had_active_resp_support": bool(had_active),
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
    print("Active respiratory support rows:", int(cxr_icu["had_active_resp_support"].sum()))
    print("Future respiratory event rows:", int(cxr_icu["first_future_resp_event_time"].notna().sum()))
    print(cxr_icu["hours_to_future_resp_event"].describe())

    section("9) Create label variants")

    h = cxr_icu["hours_to_future_resp_event"]

    for window in [12, 24, 48, 72]:
        cxr_icu[f"label_event_{window}h"] = ((h > 0) & (h <= window)).astype(int)

    horizon_72h = (
        cxr_icu["study_datetime"]
        + pd.Timedelta(hours=72)
    )

    death_before_72h = (
        cxr_icu["deathtime"].notna()
        & (cxr_icu["deathtime"] > cxr_icu["study_datetime"])
        & (cxr_icu["deathtime"] < horizon_72h)
    )

    discharge_before_72h = (
        cxr_icu["dischtime"].notna()
        & (cxr_icu["dischtime"] > cxr_icu["study_datetime"])
        & (cxr_icu["dischtime"] < horizon_72h)
    )

    full_72h_observation = (
        cxr_icu["dischtime"].notna()
        & (cxr_icu["dischtime"] >= horizon_72h)
        & (~death_before_72h)
    )

    cxr_icu["horizon_72h"] = horizon_72h
    cxr_icu["death_before_72h"] = death_before_72h
    cxr_icu["discharge_before_72h"] = discharge_before_72h
    cxr_icu["full_72h_observation"] = full_72h_observation

    positive_48h = (
        (h > 0)
        & (h <= 48)
    )

    ambiguous_48_72h = (
        (h > 48)
        & (h <= 72)
    )

    no_event_through_72h = (
        h.isna()
        | (h > 72)
    )

    cxr_icu["label_48h_stable72h"] = np.nan

    cxr_icu.loc[
        positive_48h,
        "label_48h_stable72h",
    ] = 1

    cxr_icu.loc[
        no_event_through_72h
        & full_72h_observation,
        "label_48h_stable72h",
    ] = 0

    cxr_icu["label_24h_stable72h"] = np.nan

    cxr_icu.loc[
        (h > 0) & (h <= 24),
        "label_24h_stable72h",
    ] = 1

    cxr_icu.loc[
        no_event_through_72h
        & full_72h_observation,
        "label_24h_stable72h",
    ] = 0

    cxr_icu["followup_72h_status"] = np.select(
        [
            positive_48h,
            ambiguous_48_72h,
            death_before_72h,
            discharge_before_72h,
            no_event_through_72h & full_72h_observation,
        ],
        [
            "POSITIVE_EVENT_0_48H",
            "EXCLUDE_EVENT_48_72H",
            "EXCLUDE_DEATH_BEFORE_72H",
            "EXCLUDE_HOSPITAL_DISCHARGE_BEFORE_72H",
            "SOLID_NEGATIVE_FULL_72H",
        ],
        default="EXCLUDE_UNCERTAIN_72H_FOLLOWUP",
    )

    print("\nStrict 72-hour follow-up status:")
    print(
        cxr_icu["followup_72h_status"]
        .value_counts(dropna=False)
    )

    cxr_clean = cxr_icu.loc[
        ~cxr_icu["had_active_resp_support"]
    ].copy()

    if EXCLUDE_PRIOR_RESP_EVENT:
        cxr_clean = cxr_clean.loc[
            ~cxr_clean["had_prior_resp_event"]
        ].copy()

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

    section(
        "10) Seeded patient-level prevalence-balanced "
        "train/val/test split"
    )

    label_source_col = "label_48h_stable72h"

    (
        split_map,
        final_split_summary,
        subject_assignments,
        split_info,
    ) = build_prevalence_balanced_patient_split(
        df=cxr_clean,
        label_col=label_source_col,
        seed=int(args.seed),
        search_iterations=int(
            args.split_search_iterations
        ),
        max_prevalence_gap=float(
            args.max_prevalence_gap
        ),
        max_size_deviation=float(
            args.max_split_size_deviation
        ),
        existing_assignments_path=OUT_SUBJECT_SPLITS,
    )

    def assign_split(subject_id):
        if pd.isna(subject_id):
            return "excluded"

        return split_map.get(
            int(subject_id),
            "excluded",
        )

    cxr_clean["split"] = (
        cxr_clean["subject_id"]
        .apply(assign_split)
    )

    final_for_split = cxr_clean.loc[
        cxr_clean[
            label_source_col
        ].notna()
    ].copy()

    if not final_for_split[
        "split"
    ].isin(
        [
            "train",
            "val",
            "test",
        ]
    ).all():
        raise RuntimeError(
            "A final-cohort row was not assigned "
            "to train, val, or test."
        )

    patient_split_counts = (
        final_for_split
        .groupby("subject_id")[
            "split"
        ]
        .nunique()
    )

    leaking_patients = (
        patient_split_counts[
            patient_split_counts > 1
        ]
    )

    if len(leaking_patients) > 0:
        raise RuntimeError(
            f"Patient leakage found for "
            f"{len(leaking_patients)} patients."
        )

    split_summary_rows = []

    for label_col in label_cols:
        for split in [
            "train",
            "val",
            "test",
        ]:
            d = cxr_clean[
                cxr_clean[
                    "split"
                ] == split
            ]

            s = d[label_col]
            usable = s.notna()

            split_summary_rows.append(
                {
                    "label_col": (
                        label_col
                    ),
                    "split": split,
                    "rows": int(
                        len(d)
                    ),
                    "usable": int(
                        usable.sum()
                    ),
                    "positive": int(
                        s.eq(1).sum()
                    ),
                    "negative": int(
                        s.eq(0).sum()
                    ),
                    "missing_or_ambiguous": int(
                        s.isna().sum()
                    ),
                    "positive_rate": float(
                        s.eq(1).sum()
                        / max(
                            usable.sum(),
                            1,
                        )
                    ),
                }
            )

    split_summary = pd.DataFrame(
        split_summary_rows
    )

    print(
        "Split seed:",
        int(args.seed),
    )

    print(
        "Search iterations:",
        int(
            args.split_search_iterations
        ),
    )

    print(
        "Selected search iteration:",
        split_info[
            "selected_search_iteration"
        ],
    )

    print(
        "\nFinal target split summary:"
    )

    print(
        final_split_summary.to_string()
    )

    print(
        "\nMaximum prevalence gap:",
        f"{split_info['maximum_prevalence_gap']:.6f}",
        f"({split_info['maximum_prevalence_gap'] * 100:.3f} "
        "percentage points)",
    )

    print(
        "Maximum row-size deviation:",
        f"{split_info['maximum_row_size_deviation']:.6f}",
    )

    print(
        "\nAll label variants:"
    )

    print(
        split_summary.to_string(
            index=False
        )
    )

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
        "1=selected_respiratory_intervention_within_48h_after_CXR_across_same_hospitalization; "
        "0=no_selected_respiratory_intervention_through_72h_with_complete_hospital_observation; "
        "prior_event_in_index_ICU_stay_excluded; "
        "active_qualifying_support_at_CXR_excluded; "
        "events_between_48h_and_72h_excluded; "
        "early_death_or_discharge_without_qualifying_event_excluded; "
        "uncertain_72h_followup_excluded"
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
        "dischtime",
        "deathtime",
        "horizon_72h",
        "death_before_72h",
        "discharge_before_72h",
        "full_72h_observation",
        "followup_72h_status",
        "hours_to_future_resp_event",
        "first_future_resp_event_time",
        "had_prior_resp_event",
        "had_active_resp_support",
        "ViewPosition",
        "ViewPosition_clean",
        "prediction_window_hours",
        "stable_negative_window_hours",
        "label_definition",
    ]

    front_cols = [c for c in front_cols if c in cohort.columns]
    other_cols = [c for c in cohort.columns if c not in front_cols]
    cohort = cohort[front_cols + other_cols].copy()

    stable_sort_columns = [
        column
        for column in [
            "subject_id",
            "stay_id",
            "study_datetime",
            "study_id",
            "dicom_id",
            "sample_id",
        ]
        if column in cohort.columns
    ]

    cohort = (
        cohort
        .sort_values(
            stable_sort_columns,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    if not cohort["split"].isin(
        [
            "train",
            "val",
            "test",
        ]
    ).all():
        raise RuntimeError(
            "Final cohort contains an invalid split."
        )

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
    subject_assignments.to_csv(
        OUT_SUBJECT_SPLITS,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )

    with open(
        OUT_SUBJECT_SPLITS,
        "rb",
    ) as file:
        subject_split_sha256 = (
            hashlib.sha256(
                file.read()
            ).hexdigest()
        )

    manifest = {
        "created_at": pd.Timestamp.now().isoformat(),
        "output_dir": str(OUT_DIR),
        "cohort_csv": str(OUT_COHORT),
        "all_rows_csv": str(OUT_ALL_ROWS),
        "no_prior_rows_csv": str(OUT_NO_PRIOR),
        "resp_itemids_csv": str(OUT_RESP_ITEMIDS),
        "resp_events_csv": str(OUT_RESP_EVENTS),
        "random_seed": int(args.seed),
        "split_algorithm": {
            **split_info,
            "subject_assignments_csv": str(
                OUT_SUBJECT_SPLITS
            ),
            "subject_assignments_sha256": (
                subject_split_sha256
            ),
        },
        "label": "label_48h_stable72h renamed to label",
        "positive": "intubation, invasive ventilation, or non-invasive ventilation within 48h after CXR and before discharge or death in the same hospitalization",
        "negative": "no selected respiratory intervention through 72h after CXR with complete hospital observation through the full 72h horizon",
        "excluded": "prior respiratory intervention within the index ICU stay; active qualifying support at CXR; X-ray outside recorded hospitalization; first intervention 48-72h; early death or discharge without an earlier qualifying intervention; uncertain 72h follow-up",
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
    print(OUT_SUBJECT_SPLITS)

    print("\nFinal cohort:")
    print("rows:", len(cohort))
    print(cohort["label"].value_counts().sort_index())
    print(pd.crosstab(cohort["split"], cohort["label"]))

    print("\nRuntime minutes:", round((time.time() - t_start) / 60.0, 2))

    del meta, icu, adm, d_items, proc, proc_labeled
    gc.collect()

if __name__ == "__main__":
    tqdm.pandas()
    main()
