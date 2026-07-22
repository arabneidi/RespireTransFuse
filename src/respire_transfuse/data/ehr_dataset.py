"""Load and validate the sequential EHR inputs used by model training.

This module aligns cohort rows with NPZ value and observation-mask arrays,
standardizes supported tensor layouts, checks binary labels and split metadata,
and exposes PyTorch datasets for the 24-hour sequences. It also provides a
balanced binary batch sampler for training under the project's class imbalance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


def prepare_binary_labels(df, label_col):
    df = df.copy()

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}. Columns: {df.columns.tolist()}")

    numeric = pd.to_numeric(df[label_col], errors="coerce")

    if numeric.isna().any():
        bad = df.loc[numeric.isna(), label_col].head(10).tolist()
        raise ValueError(f"Label column {label_col} must be numeric 0/1. Bad examples: {bad}")

    vals = sorted(numeric.dropna().unique().tolist())

    if not set(vals).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Label column {label_col} must contain only 0/1. Found: {vals}")

    df[label_col] = numeric.astype(int)
    return df


def standardize_ehr_arrays(X, M, feature_names=None):
    X = X.astype(np.float32)
    M = M.astype(np.float32)

    if X.ndim != 3 or M.ndim != 3:
        raise ValueError(f"Expected X and M as 3D arrays. Got X={X.shape}, M={M.shape}")

    if X.shape != M.shape:
        raise ValueError(f"X and M shape mismatch. X={X.shape}, M={M.shape}")

    n_features = None
    if feature_names is not None:
        try:
            n_features = len(feature_names)
        except Exception:
            n_features = None

    if n_features is not None:
        if X.shape[-1] == n_features:
            return X, M
        if X.shape[1] == n_features:
            return np.transpose(X, (0, 2, 1)).copy(), np.transpose(M, (0, 2, 1)).copy()

    if X.shape[1] <= 64 and X.shape[2] > 64:
        return np.transpose(X, (0, 2, 1)).copy(), np.transpose(M, (0, 2, 1)).copy()

    return X, M


def load_ehr_splits(
    cohort_csv,
    ehr_npz,
    sample_col="sample_id",
    label_col="label",
    split_col="split",
):
    cohort_path = Path(cohort_csv)
    ehr_path = Path(ehr_npz)

    if not cohort_path.exists():
        raise FileNotFoundError(f"cohort_csv not found: {cohort_path}")

    if not ehr_path.exists():
        raise FileNotFoundError(f"ehr_npz not found: {ehr_path}")

    cohort = pd.read_csv(cohort_path)

    required_cols = [sample_col, label_col, split_col]
    missing_cols = [c for c in required_cols if c not in cohort.columns]
    if missing_cols:
        raise ValueError(f"Missing cohort columns: {missing_cols}. Available: {cohort.columns.tolist()}")

    cohort = cohort.dropna(subset=required_cols).copy()
    cohort = prepare_binary_labels(cohort, label_col)

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
        raise ValueError(f"EHR NPZ missing mask array. Expected 'M' or 'mask'. Available: {list(data.files)}")

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
        raise RuntimeError("Train split is empty after merge.")

    if len(val_df) == 0:
        raise RuntimeError("Validation split is empty after merge.")

    summary = {
        "cohort_rows": int(len(cohort)),
        "npz_rows": int(len(npz_df)),
        "merged_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "x_shape": list(X.shape),
        "m_shape": list(M.shape),
        "n_features": int(X.shape[-1]),
        "feature_names": feature_names.tolist() if feature_names is not None else None,
        "split_label_counts": pd.crosstab(df[split_col], df[label_col]).to_dict(),
    }

    return train_df, val_df, test_df, X, M, feature_names, summary


class EHRDataset(Dataset):
    def __init__(self, df, X, M, sample_col="sample_id", label_col="label"):
        self.df = df.reset_index(drop=True).copy()
        self.X = X
        self.M = M
        self.sample_col = sample_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_idx = int(row["npz_idx"])

        ehr_x = torch.as_tensor(self.X[npz_idx], dtype=torch.float32).contiguous()
        ehr_m = torch.as_tensor(self.M[npz_idx], dtype=torch.float32).contiguous()

        ehr_x = torch.nan_to_num(ehr_x, nan=0.0, posinf=0.0, neginf=0.0)
        ehr_m = torch.nan_to_num(ehr_m, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        return {
            "ehr_x": ehr_x,
            "ehr_m": ehr_m,
            "label": torch.tensor(float(row[self.label_col]), dtype=torch.float32),
            "sample_id": str(row[self.sample_col]),
        }


class BalancedBinaryBatchSampler(Sampler):
    def __init__(
        self,
        labels,
        batch_size,
        pos_fraction=0.35,
        batches_per_epoch=None,
        seed=42,
    ):
        self.labels = np.asarray(labels).astype(int)
        self.batch_size = int(batch_size)
        self.pos_fraction = float(pos_fraction)
        self.seed = int(seed)

        self.pos_idx = np.where(self.labels == 1)[0]
        self.neg_idx = np.where(self.labels == 0)[0]

        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise ValueError("BalancedBinaryBatchSampler requires both positive and negative samples.")

        self.pos_per_batch = max(1, int(round(self.batch_size * self.pos_fraction)))
        self.neg_per_batch = max(1, self.batch_size - self.pos_per_batch)

        if self.pos_per_batch + self.neg_per_batch != self.batch_size:
            self.neg_per_batch = self.batch_size - self.pos_per_batch

        if batches_per_epoch is None:
            self.batches_per_epoch = int(np.ceil(len(self.labels) / self.batch_size))
        else:
            self.batches_per_epoch = int(batches_per_epoch)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + np.random.randint(0, 10_000_000))

        pos_pool = rng.permutation(self.pos_idx)
        neg_pool = rng.permutation(self.neg_idx)

        pos_ptr = 0
        neg_ptr = 0

        for _ in range(self.batches_per_epoch):
            if pos_ptr + self.pos_per_batch > len(pos_pool):
                pos_pool = rng.permutation(self.pos_idx)
                pos_ptr = 0

            if neg_ptr + self.neg_per_batch > len(neg_pool):
                neg_pool = rng.permutation(self.neg_idx)
                neg_ptr = 0

            pos_batch = pos_pool[pos_ptr:pos_ptr + self.pos_per_batch]
            neg_batch = neg_pool[neg_ptr:neg_ptr + self.neg_per_batch]

            pos_ptr += self.pos_per_batch
            neg_ptr += self.neg_per_batch

            batch = np.concatenate([pos_batch, neg_batch])
            rng.shuffle(batch)

            yield batch.tolist()
