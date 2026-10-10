from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from respire_transfuse.data.ehr_dataset import standardize_ehr_arrays, prepare_binary_labels

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_multimodal_splits(
    cohort_csv,
    ehr_npz,
    sample_col,
    label_col,
    split_col,
    image_col,
    require_image_exists=True,
    require_image_decode_ok=True,
):
    cohort_path = Path(cohort_csv)
    ehr_path = Path(ehr_npz)

    if not cohort_path.exists():
        raise FileNotFoundError(f"cohort_csv not found: {cohort_path}")

    if not ehr_path.exists():
        raise FileNotFoundError(f"ehr_npz not found: {ehr_path}")

    cohort = pd.read_csv(cohort_path)

    required_cols = [sample_col, label_col, split_col, image_col]
    missing_cols = [c for c in required_cols if c not in cohort.columns]

    if missing_cols:
        raise ValueError(f"Missing cohort columns: {missing_cols}. Available: {cohort.columns.tolist()}")

    cohort = cohort.dropna(subset=required_cols).copy()
    cohort = prepare_binary_labels(cohort, label_col)

    if require_image_exists and "image_exists" in cohort.columns:
        cohort = cohort[cohort["image_exists"].astype(bool)].copy()

    if require_image_decode_ok and "image_decode_ok" in cohort.columns:
        cohort = cohort[cohort["image_decode_ok"].astype(bool)].copy()

    cohort[sample_col] = cohort[sample_col].astype(str)
    cohort[split_col] = cohort[split_col].astype(str).str.lower().str.strip()

    data = np.load(ehr_path, allow_pickle=True)

    required_npz = ["X", "y", "sample_id", "split"]
    missing_npz = [k for k in required_npz if k not in data.files]

    if missing_npz:
        raise ValueError(f"EHR NPZ missing arrays: {missing_npz}. Available: {list(data.files)}")

    if "M" in data.files:
        mask_array = data["M"]
    elif "mask" in data.files:
        mask_array = data["mask"]
    else:
        raise ValueError(f"EHR NPZ missing mask array. Available: {list(data.files)}")

    feature_names = data["feature_names"].astype(str) if "feature_names" in data.files else None

    X, M = standardize_ehr_arrays(
        data["X"],
        mask_array,
        feature_names=feature_names,
    )

    y = data["y"].astype(np.int64)
    npz_sample_id = data["sample_id"].astype(str)
    npz_split = np.char.lower(np.char.strip(data["split"].astype(str)))

    npz_df = pd.DataFrame({
        sample_col: npz_sample_id,
        "npz_idx": np.arange(len(npz_sample_id), dtype=np.int64),
        "npz_label": y,
        "npz_split": npz_split,
    })

    df = cohort.merge(npz_df, on=sample_col, how="inner")
    df = df.sort_values("npz_idx").reset_index(drop=True)

    label_mismatch = int((df[label_col].astype(int).values != df["npz_label"].astype(int).values).sum())
    split_mismatch = int((df[split_col].astype(str).values != df["npz_split"].astype(str).values).sum())

    if label_mismatch > 0:
        bad = df[df[label_col].astype(int) != df["npz_label"].astype(int)].head(10)
        raise RuntimeError("Label mismatch between cohort CSV and EHR NPZ:\n" + bad.to_string(index=False))

    if split_mismatch > 0:
        bad = df[df[split_col].astype(str) != df["npz_split"].astype(str)].head(10)
        raise RuntimeError("Split mismatch between cohort CSV and EHR NPZ:\n" + bad.to_string(index=False))

    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col].isin(["val", "valid", "validation"])].copy()
    test_df = df[df[split_col] == "test"].copy()

    if len(train_df) == 0:
        raise RuntimeError("Train split is empty.")

    if len(val_df) == 0:
        raise RuntimeError("Validation split is empty.")

    if len(test_df) == 0:
        raise RuntimeError("Test split is empty.")

    summary = {
        "cohort_rows_after_image_filters": int(len(cohort)),
        "npz_rows": int(len(npz_df)),
        "merged_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_pos": int(train_df[label_col].sum()),
        "val_pos": int(val_df[label_col].sum()),
        "test_pos": int(test_df[label_col].sum()),
        "x_shape": list(X.shape),
        "m_shape": list(M.shape),
        "n_features": int(X.shape[-1]),
        "image_col": image_col,
        "sample_col": sample_col,
        "label_col": label_col,
        "split_col": split_col,
        "feature_names": feature_names.tolist() if feature_names is not None else None,
    }

    return train_df, val_df, test_df, X, M, feature_names, str(cohort_path.parent), summary


class MultimodalRespireDataset(Dataset):
    def __init__(
        self,
        df,
        X,
        M,
        image_col,
        sample_col,
        label_col,
        output_root,
        cohort_dir,
        transform,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.X = X
        self.M = M
        self.image_col = image_col
        self.sample_col = sample_col
        self.label_col = label_col
        self.output_root = Path(output_root)
        self.cohort_dir = Path(cohort_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, raw_path):
        raw_path = str(raw_path).replace("\\", "/")
        p = Path(raw_path)

        candidates = []

        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(self.output_root / p)
            candidates.append(self.cohort_dir / p)
            candidates.append(Path("/content") / p)

        for c in candidates:
            if c.exists():
                return c

        raise FileNotFoundError("Image path not found. Tried:\n" + "\n".join(str(c) for c in candidates[:10]))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_idx = int(row["npz_idx"])

        image_path = self._resolve_image_path(row[self.image_col])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        ehr_x = torch.as_tensor(self.X[npz_idx], dtype=torch.float32).contiguous()
        ehr_m = torch.as_tensor(self.M[npz_idx], dtype=torch.float32).contiguous()

        ehr_x = torch.nan_to_num(ehr_x, nan=0.0, posinf=0.0, neginf=0.0)
        ehr_m = torch.nan_to_num(ehr_m, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        return {
            "image": image,
            "ehr_x": ehr_x,
            "ehr_m": ehr_m,
            "label": torch.tensor(float(row[self.label_col]), dtype=torch.float32),
            "sample_id": str(row[self.sample_col]),
        }
