"""Verify that the dummy cohort, EHR tensor, images, and MedFuse code are available."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

COHORT_PATH = (
    ROOT
    / "data"
    / "dummy_100"
    / "cohort_dummy_100.csv"
)

EHR_PATH = (
    ROOT
    / "data"
    / "dummy_100"
    / "ehr_dummy_100.npz"
)

MEDFUSE_ROOT = (
    ROOT
    / "external"
    / "medfuse_original"
)

REQUIRED_PATHS = [
    COHORT_PATH,
    EHR_PATH,
    MEDFUSE_ROOT,
]


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main():
    errors = [
        display_path(path)
        for path in REQUIRED_PATHS
        if not path.exists()
    ]

    if COHORT_PATH.exists():
        cohort = pd.read_csv(COHORT_PATH)

        required_columns = {
            "sample_id",
            "label",
            "split",
        }

        missing_columns = sorted(
            required_columns.difference(cohort.columns)
        )

        if missing_columns:
            errors.append(
                "Missing cohort columns: "
                + ", ".join(missing_columns)
            )

        image_column = next(
            (
                column
                for column in [
                    "verified_image_path",
                    "image_path",
                    "cxr_path",
                    "path",
                ]
                if column in cohort.columns
            ),
            None,
        )

        if image_column is None:
            errors.append(
                "Missing cohort image-path column"
            )
        else:
            for raw_path in (
                cohort[image_column]
                .dropna()
                .astype(str)
            ):
                image_path = Path(raw_path)

                if not image_path.is_absolute():
                    image_path = ROOT / image_path

                if not image_path.exists():
                    errors.append(
                        display_path(image_path)
                    )
                    break

        if len(cohort) != 100:
            errors.append(
                f"Expected 100 cohort rows, found {len(cohort)}"
            )

        if EHR_PATH.exists() and not missing_columns:
            with np.load(EHR_PATH, allow_pickle=True) as ehr:
                required_arrays = {
                    "X",
                    "M",
                    "X_raw",
                    "mask",
                    "y",
                    "split",
                    "sample_id",
                    "labels",
                    "sources",
                    "itemids",
                }
                missing_arrays = sorted(
                    required_arrays.difference(ehr.files)
                )

                if missing_arrays:
                    errors.append(
                        "Missing EHR arrays: "
                        + ", ".join(missing_arrays)
                    )
                else:
                    if ehr["X"].shape != (100, 24, 26):
                        errors.append(
                            "Expected X shape (100, 24, 26), "
                            f"found {ehr['X'].shape}"
                        )

                    if ehr["M"].shape != ehr["X"].shape:
                        errors.append(
                            "EHR value and mask shapes do not match"
                        )

                    sample_ids = cohort["sample_id"].astype(str).to_numpy()
                    labels = cohort["label"].astype(int).to_numpy()
                    splits = cohort["split"].astype(str).to_numpy()

                    if not np.array_equal(
                        sample_ids,
                        ehr["sample_id"].astype(str),
                    ):
                        errors.append(
                            "Cohort and EHR sample order do not match"
                        )

                    if not np.array_equal(labels, ehr["y"].astype(int)):
                        errors.append(
                            "Cohort and EHR labels do not match"
                        )

                    if not np.array_equal(
                        splits,
                        ehr["split"].astype(str),
                    ):
                        errors.append(
                            "Cohort and EHR splits do not match"
                        )

                    if len(ehr["labels"]) != 26:
                        errors.append(
                            "Expected 26 EHR feature labels, "
                            f"found {len(ehr['labels'])}"
                        )

    if errors:
        print("Missing or invalid resources:")

        for error in errors:
            print(f"  - {error}")

        sys.exit(1)

    print("Dummy-run requirements verified.")


if __name__ == "__main__":
    main()
