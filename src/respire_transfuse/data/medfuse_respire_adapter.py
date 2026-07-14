
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def standardize_ehr_arrays(X, M, feature_names=None):
    X = X.astype(np.float32)
    M = M.astype(np.float32)

    if X.ndim != 3 or M.ndim != 3:
        raise ValueError(f"Expected X/M as 3D arrays. Got X={X.shape}, M={M.shape}")

    if X.shape != M.shape:
        raise ValueError(f"X/M shape mismatch. X={X.shape}, M={M.shape}")

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


def prepare_binary_labels(df, label_col):
    df = df.copy()
    y = pd.to_numeric(df[label_col], errors="coerce")

    if y.isna().any():
        raise ValueError(f"Label column {label_col} contains non-numeric values.")

    values = set(y.unique().tolist())

    if not values.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Label column {label_col} must be binary 0/1. Found: {sorted(values)}")

    df[label_col] = y.astype(int)

    return df


def load_respire_medfuse_splits(
    cohort_csv,
    ehr_npz,
    sample_col="sample_id",
    image_col="verified_image_path",
    label_col="label",
    split_col="split",
    debug_n=0,
):
    cohort = pd.read_csv(cohort_csv)

    required = [sample_col, image_col, label_col, split_col]

    for col in required:
        if col not in cohort.columns:
            raise ValueError(f"Missing cohort column {col}. Available columns: {cohort.columns.tolist()}")

    cohort = cohort.dropna(subset=required).copy()
    cohort = prepare_binary_labels(cohort, label_col)

    if "image_exists" in cohort.columns:
        cohort = cohort[cohort["image_exists"] == True].copy()

    if "image_decode_ok" in cohort.columns:
        cohort = cohort[cohort["image_decode_ok"] == True].copy()

    cohort[sample_col] = cohort[sample_col].astype(str)
    cohort[split_col] = cohort[split_col].astype(str).str.lower().str.strip()

    data = np.load(ehr_npz, allow_pickle=True)

    for key in ["X", "M", "y", "sample_id", "split"]:
        if key not in data:
            raise ValueError(f"EHR NPZ missing key {key}. Available keys: {list(data.keys())}")

    feature_names = data["feature_names"].astype(str) if "feature_names" in data else None

    X, M = standardize_ehr_arrays(
        data["X"],
        data["M"],
        feature_names=feature_names,
    )

    npz_df = pd.DataFrame(
        {
            sample_col: data["sample_id"].astype(str),
            "npz_idx": np.arange(len(data["sample_id"]), dtype=np.int64),
            "npz_label": data["y"].astype(int),
            "npz_split": np.char.lower(np.char.strip(data["split"].astype(str))),
        }
    )

    df = cohort.merge(npz_df, on=sample_col, how="inner")
    df = df.sort_values("npz_idx").reset_index(drop=True)

    label_mismatch = int((df[label_col].astype(int).values != df["npz_label"].astype(int).values).sum())
    split_mismatch = int((df[split_col].astype(str).values != df["npz_split"].astype(str).values).sum())

    if label_mismatch > 0:
        raise RuntimeError(f"Label mismatch between cohort CSV and EHR NPZ: {label_mismatch}")

    if split_mismatch > 0:
        raise RuntimeError(f"Split mismatch between cohort CSV and EHR NPZ: {split_mismatch}")

    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col].isin(["val", "valid", "validation"])].copy()
    test_df = df[df[split_col] == "test"].copy()

    if int(debug_n) > 0:
        n = int(debug_n)

        def take_balanced(split_df):
            pos = split_df[split_df[label_col] == 1]
            neg = split_df[split_df[label_col] == 0]

            n_pos = min(len(pos), max(1, n // 2))
            n_neg = min(len(neg), max(1, n - n_pos))

            out = pd.concat([pos.head(n_pos), neg.head(n_neg)], axis=0)
            out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)

            return out

        train_df = take_balanced(train_df)
        val_df = take_balanced(val_df)
        test_df = take_balanced(test_df)

    summary = {
        "cohort_rows_after_image_filter": int(len(cohort)),
        "npz_rows": int(len(npz_df)),
        "merged_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_positives": int(train_df[label_col].sum()),
        "val_positives": int(val_df[label_col].sum()),
        "test_positives": int(test_df[label_col].sum()),
        "X_shape": list(X.shape),
        "M_shape": list(M.shape),
        "n_value_features": int(X.shape[-1]),
        "n_input_features": int(X.shape[-1] * 2),
        "n_features": int(X.shape[-1] * 2),
    }

    return train_df, val_df, test_df, X, M, feature_names, summary


def build_medfuse_image_transform(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class RespireMedFuseDataset(Dataset):
    CLASSES = ["respiratory_deterioration"]

    def __init__(
        self,
        df,
        X,
        M,
        output_root,
        sample_col="sample_id",
        image_col="verified_image_path",
        label_col="label",
        transform=None,
        load_images=True,
        image_size=224,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.X = X
        self.M = M
        self.output_root = Path(output_root)
        self.sample_col = sample_col
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform if transform is not None else build_medfuse_image_transform()
        self.load_images = bool(load_images)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.df)

    def resolve_image_path(self, raw):
        raw = str(raw).replace("\\", "/")
        p = Path(raw)

        candidates = []

        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(self.output_root / p)
            candidates.append(Path("/content") / p)
            candidates.append(Path("/content/drive/MyDrive") / p)

        for c in candidates:
            if c.exists():
                return c

        raise FileNotFoundError("Image path not found. Tried:\n" + "\n".join(str(c) for c in candidates))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npz_idx = int(row["npz_idx"])

        x = np.nan_to_num(
            self.X[npz_idx].astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        m = np.nan_to_num(
            self.M[npz_idx].astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        m = np.clip(m, 0.0, 1.0)

        seq_length = int(x.shape[0])

        x = np.concatenate([x, m], axis=-1).astype(np.float32)

        if self.load_images:
            image_path = self.resolve_image_path(row[self.image_col])
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        else:
            image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

        y = float(row[self.label_col])

        return {
            "x": x,
            "img": image,
            "y_ehr": np.float32(y),
            "y_cxr": np.float32(y),
            "seq_length": seq_length,
            "pair": bool(True),
            "sample_id": str(row[self.sample_col]),
        }



def medfuse_collate(batch):
    x = np.stack([b["x"] for b in batch], axis=0).astype(np.float32)
    img = torch.stack([b["img"] for b in batch], dim=0)

    y_ehr = np.asarray([float(np.asarray(b["y_ehr"]).reshape(-1)[0]) for b in batch], dtype=np.float32)
    y_cxr = np.asarray([float(np.asarray(b["y_cxr"]).reshape(-1)[0]) for b in batch], dtype=np.float32)

    seq_lengths = np.asarray([b["seq_length"] for b in batch], dtype=np.int64)
    pairs = np.asarray([b["pair"] for b in batch], dtype=bool)

    return x, img, y_ehr, y_cxr, seq_lengths, pairs


def build_respire_medfuse_loaders(args):
    train_df, val_df, test_df, X, M, feature_names, summary = load_respire_medfuse_splits(
        cohort_csv=args.cohort_csv,
        ehr_npz=args.ehr_npz,
        sample_col=args.sample_col,
        image_col=args.image_col,
        label_col=args.label_col,
        split_col=args.split_col,
        debug_n=args.debug_n,
    )

    transform = build_medfuse_image_transform(args.image_size)
    load_images = str(args.fusion_type) != "uni_ehr"

    train_ds = RespireMedFuseDataset(
        train_df,
        X,
        M,
        output_root=args.output_root,
        sample_col=args.sample_col,
        image_col=args.image_col,
        label_col=args.label_col,
        transform=transform,
        load_images=load_images,
        image_size=args.image_size,
    )

    val_ds = RespireMedFuseDataset(
        val_df,
        X,
        M,
        output_root=args.output_root,
        sample_col=args.sample_col,
        image_col=args.image_col,
        label_col=args.label_col,
        transform=transform,
        load_images=load_images,
        image_size=args.image_size,
    )

    test_ds = RespireMedFuseDataset(
        test_df,
        X,
        M,
        output_root=args.output_root,
        sample_col=args.sample_col,
        image_col=args.image_col,
        label_col=args.label_col,
        transform=transform,
        load_images=load_images,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
        collate_fn=medfuse_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
        collate_fn=medfuse_collate,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
        collate_fn=medfuse_collate,
    )

    return train_loader, val_loader, test_loader, feature_names, summary
