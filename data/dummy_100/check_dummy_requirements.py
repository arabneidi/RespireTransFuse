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

EXPECTED_ITEMIDS = [
    "220045", "220210", "220277", "220179", "220180", "223761",
    "223834", "223835", "50813", "50820", "50821", "50818", "50882",
    "50811", "51301", "50912", "50960", "50971", "225638", "220224",
    "220235", "225698", "52172", "220632", "51214", "226536",
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
            "subject_id",
            "label",
            "split",
            "had_prior_resp_event",
            "had_active_resp_support",
            "full_72h_observation",
            "followup_72h_status",
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

        if not missing_columns:
            if cohort["had_prior_resp_event"].astype(bool).any():
                errors.append("Dummy cohort contains a prior respiratory event")

            if cohort["had_active_resp_support"].astype(bool).any():
                errors.append("Dummy cohort contains active respiratory support")

            positives = cohort["label"].astype(int).eq(1)
            negatives = cohort["label"].astype(int).eq(0)
            if not cohort.loc[
                positives, "followup_72h_status"
            ].eq("POSITIVE_EVENT_0_48H").all():
                errors.append("Dummy positive labels do not match follow-up status")

            if not cohort.loc[
                negatives, "followup_72h_status"
            ].eq("SOLID_NEGATIVE_FULL_72H").all():
                errors.append("Dummy negative labels do not match follow-up status")

            if not cohort.loc[
                negatives, "full_72h_observation"
            ].astype(bool).all():
                errors.append("Dummy negative rows lack complete 72-hour follow-up")

            if cohort.groupby("subject_id")["split"].nunique().gt(1).any():
                errors.append("Dummy cohort has patient overlap across splits")

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

                    itemids = ehr["itemids"].astype(str).tolist()
                    if itemids != EXPECTED_ITEMIDS:
                        errors.append("EHR item identifiers do not match the 26-feature registry")

    if errors:
        print("Missing or invalid resources:")

        for error in errors:
            print(f"  - {error}")

        sys.exit(1)

    print("Dummy-run requirements verified.")


if __name__ == "__main__":
    main()
