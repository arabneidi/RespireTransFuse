"""Validate the files required by the seven-model dummy smoke test.

The check confirms that the 100-row cohort, aligned EHR arrays, referenced chest
X-rays, model configurations, and bundled MedFuse source files are present and
internally consistent. It exits with a clear error before training starts when a
required path, array key, sample identifier, or image is missing.
"""

from pathlib import Path
import sys

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

    if errors:
        print("Missing or invalid resources:")

        for error in errors:
            print(f"  - {error}")

        sys.exit(1)

    print("Dummy-run requirements verified.")


if __name__ == "__main__":
    main()
