from pathlib import Path
import sys

import pandas as pd

root = Path(__file__).resolve().parents[2]

cohort_csv = root / "data" / "dummy_100" / "cohort_dummy_100.csv"
ehr_npz = root / "data" / "dummy_100" / "ehr_dummy_100.npz"

missing = []

for p in [cohort_csv, ehr_npz]:
    if not p.exists():
        missing.append(str(p.relative_to(root)))

if cohort_csv.exists():
    df = pd.read_csv(cohort_csv)
    image_col = None

    for col in ["verified_image_path", "image_path", "cxr_path", "path"]:
        if col in df.columns:
            image_col = col
            break

    if image_col is None:
        missing.append("dummy cohort image path column")
    else:
        for raw in df[image_col].astype(str).tolist():
            p = Path(raw)
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                missing.append(str(p.relative_to(root)))
                break

if missing:
    print("Missing required dummy-run files:")
    for p in missing:
        print(f"  - {p}")
    sys.exit(1)

print("Dummy requirements check passed.")
