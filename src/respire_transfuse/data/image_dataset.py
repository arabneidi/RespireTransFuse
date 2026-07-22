"""Load chest X-ray cohorts and define image preprocessing for model training."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def prepare_binary_labels(df, label_col):
    df = df.copy()

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}. Columns: {df.columns.tolist()}")

    numeric = pd.to_numeric(df[label_col], errors="coerce")

    if numeric.isna().any():
        raise ValueError(f"Label column {label_col} must be numeric 0/1.")

    vals = sorted(numeric.unique().tolist())

    if not set(vals).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Label column {label_col} must contain only 0/1. Found: {vals}")

    df[label_col] = numeric.astype(int)

    return df


def load_image_splits(
    cohort_csv,
    image_col,
    label_col,
    split_col,
    sample_col,
    require_image_exists=True,
    require_image_decode_ok=True,
):
    cohort_csv = Path(cohort_csv)

    if not cohort_csv.exists():
        raise FileNotFoundError(cohort_csv)

    df = pd.read_csv(cohort_csv)

    required = [image_col, label_col, split_col, sample_col]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing cohort column '{col}'. Available columns: {df.columns.tolist()}")

    df = df.dropna(subset=required).copy()
    df = prepare_binary_labels(df, label_col)

    if require_image_exists and "image_exists" in df.columns:
        df = df[df["image_exists"].astype(bool)].copy()

    if require_image_decode_ok and "image_decode_ok" in df.columns:
        df = df[df["image_decode_ok"].astype(bool)].copy()

    df[sample_col] = df[sample_col].astype(str)
    df[split_col] = df[split_col].astype(str).str.lower().str.strip()

    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col].isin(["val", "valid", "validation"])].copy()
    test_df = df[df[split_col] == "test"].copy()

    if len(train_df) == 0:
        raise RuntimeError("Train split is empty.")
    if len(val_df) == 0:
        raise RuntimeError("Validation split is empty.")
    if len(test_df) == 0:
        raise RuntimeError("Test split is empty.")

    data_summary = {
        "cohort_csv": str(cohort_csv),
        "rows_after_filters": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_pos": int(train_df[label_col].sum()),
        "val_pos": int(val_df[label_col].sum()),
        "test_pos": int(test_df[label_col].sum()),
        "image_col": image_col,
        "label_col": label_col,
        "split_col": split_col,
        "sample_col": sample_col,
    }

    return train_df, val_df, test_df, str(cohort_csv.parent), data_summary


class CXRImageDataset(Dataset):
    def __init__(
        self,
        df,
        image_col,
        label_col,
        sample_col,
        output_root,
        cohort_dir,
        transform,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_col = image_col
        self.label_col = label_col
        self.sample_col = sample_col
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

        image_path = self._resolve_image_path(row[self.image_col])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        sample_id = str(row[self.sample_col]) if self.sample_col in self.df.columns else str(idx)

        return {
            "image": image,
            "label": label,
            "sample_id": sample_id,
        }


def build_image_transforms(image_size, hflip_p=0.0, affine_p=0.15, color_jitter_p=0.03):
    image_size = int(image_size)

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=4,
                    translate=(0.02, 0.02),
                    scale=(0.98, 1.02),
                    shear=0,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            ],
            p=float(affine_p),
        ),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                )
            ],
            p=float(color_jitter_p),
        ),
        transforms.RandomHorizontalFlip(p=float(hflip_p)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tf, eval_tf
