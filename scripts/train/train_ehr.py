#!/usr/bin/env python3
"""Train and evaluate the EHR-only Transformer risk model."""

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True.*norm_first was True",
    category=UserWarning,
)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respire_transfuse.utils.config import load_config, save_yaml
from respire_transfuse.data.ehr_dataset import (
    load_ehr_splits,
    EHRDataset,
    BalancedBinaryBatchSampler,
)
from respire_transfuse.models.ehr_only import EHRTransformerRiskModel
from respire_transfuse.training.seed import seed_everything
from respire_transfuse.training.metrics import (
    json_safe,
    sigmoid_np,
    choose_threshold_by_val_f1,
    summarize_probabilities,
)
from respire_transfuse.training.calibration import (
    fit_temperature_bias,
    calibrated_probabilities,
)
from respire_transfuse.training.plots import plot_history
from respire_transfuse.utils.epoch_metrics import save_epoch_artifacts
from respire_transfuse.training.ehr_engine import (
    train_one_epoch,
    evaluate,
    lr_factor_for_epoch,
    set_optimizer_lr,
    save_predictions,
)


def resolve_path(path):
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def limit_df_mixed(df, n, label_col):
    if n is None:
        return df

    n = int(n)
    if n <= 0 or len(df) <= n:
        return df

    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]

    n_pos = min(len(pos), max(1, n // 2))
    n_neg = min(len(neg), n - n_pos)

    out = pd.concat(
        [
            pos.head(n_pos),
            neg.head(n_neg),
        ],
        axis=0,
    )

    if len(out) < n:
        rest = df.drop(index=out.index, errors="ignore").head(n - len(out))
        out = pd.concat([out, rest], axis=0)

    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def build_loader(dataset, cfg, labels=None, train=False):
    train_cfg = cfg["training"]
    sampling_cfg = cfg["sampling"]

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])

    if train and bool(sampling_cfg["balanced_batches"]):
        sampler = BalancedBinaryBatchSampler(
            labels=labels,
            batch_size=batch_size,
            pos_fraction=float(sampling_cfg["pos_fraction"]),
            batches_per_epoch=sampling_cfg.get("batches_per_epoch", None),
            seed=int(train_cfg["seed"]),
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(train),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def make_criterion(train_df, label_col, loss_cfg, device):
    y = train_df[label_col].astype(int).values
    pos = int(y.sum())
    neg = int(len(y) - pos)

    if pos <= 0:
        raise RuntimeError("Training split has zero positives.")

    raw_pos_weight = neg / max(pos, 1)
    cap = float(loss_cfg.get("pos_weight_cap", 1.0))

    if cap > 0:
        pos_weight_value = min(raw_pos_weight, cap)
    else:
        pos_weight_value = raw_pos_weight

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), {
        "train_pos": pos,
        "train_neg": neg,
        "raw_pos_weight": float(raw_pos_weight),
        "used_pos_weight": float(pos_weight_value),
    }




def save_history_without_accuracy_brier(history, path):
    frame = pd.DataFrame(history)

    excluded_columns = [
        column
        for column in frame.columns
        if "accuracy" in column.lower()
        or "brier" in column.lower()
    ]

    if excluded_columns:
        frame = frame.drop(columns=excluded_columns)

    frame.to_csv(path, index=False)

def remove_unsaved_metrics(obj):
    if isinstance(obj, dict):
        return {
            key: remove_unsaved_metrics(value)
            for key, value in obj.items()
            if key not in {"accuracy", "brier"}
        }

    if isinstance(obj, list):
        return [remove_unsaved_metrics(value) for value in obj]

    return obj

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--paths", type=str, default="configs/paths.yaml")
    parser.add_argument("--config", type=str, default="configs/experiments/ehr_only.yaml")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_delta_auprc", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug_n", type=int, default=None)

    return parser.parse_args()

def main():
    args = parse_args()

    cfg = load_config(
        resolve_path(args.paths),
        resolve_path(args.config),
    )

    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)

    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)

    if args.num_workers is not None:
        cfg["training"]["num_workers"] = int(args.num_workers)

    if args.dropout is not None:
        cfg["model"]["dropout"] = float(args.dropout)

    if args.lr is not None:
        cfg["training"]["lr"] = float(args.lr)

    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = float(args.weight_decay)

    if args.patience is not None:
        cfg["training"]["patience"] = int(args.patience)

    if args.min_delta_auprc is not None:
        cfg["selection"]["min_delta_auprc"] = float(
            args.min_delta_auprc
        )

    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)

    seed_everything(int(cfg["training"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["training"].get("use_amp", False)) and device.type == "cuda"

    cols = cfg["columns"]

    train_df, val_df, test_df, X, M, feature_names, data_summary = load_ehr_splits(
        cohort_csv=cfg["data"]["cohort_csv"],
        ehr_npz=cfg["data"]["ehr_npz"],
        sample_col=cols["sample_col"],
        label_col=cols["label_col"],
        split_col=cols["split_col"],
    )

    if args.debug_n is not None:
        train_df = limit_df_mixed(train_df, args.debug_n, cols["label_col"])
        val_df = limit_df_mixed(val_df, max(16, args.debug_n // 2), cols["label_col"])
        test_df = limit_df_mixed(test_df, max(16, args.debug_n // 2), cols["label_col"])

    train_set = EHRDataset(
        train_df,
        X,
        M,
        sample_col=cols["sample_col"],
        label_col=cols["label_col"],
    )

    val_set = EHRDataset(
        val_df,
        X,
        M,
        sample_col=cols["sample_col"],
        label_col=cols["label_col"],
    )

    test_set = EHRDataset(
        test_df,
        X,
        M,
        sample_col=cols["sample_col"],
        label_col=cols["label_col"],
    ) if len(test_df) > 0 else None

    train_loader = build_loader(
        train_set,
        cfg,
        labels=train_df[cols["label_col"]].astype(int).values,
        train=True,
    )

    val_loader = build_loader(
        val_set,
        cfg,
        train=False,
    )

    test_loader = build_loader(
        test_set,
        cfg,
        train=False,
    ) if test_set is not None else None

    model = EHRTransformerRiskModel(
        n_features=int(X.shape[-1]),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        dim_feedforward=int(cfg["model"]["dim_feedforward"]),
        dropout=float(cfg["model"]["dropout"]),
        use_mask_channel=bool(cfg["model"]["use_mask_channel"]),
        use_cls_token=bool(cfg["model"]["use_cls_token"]),
        ehr_token_dim=int(cfg["model"].get("ehr_token_dim", 48)),
        local_scale_init=float(
            cfg["model"].get("local_scale_init", -1.1)
        ),
    ).to(device)

    model.set_fusion_adapters_trainable(False)

    criterion, criterion_info = make_criterion(
        train_df,
        cols["label_col"],
        cfg["loss"],
        device,
    )

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    if args.save_dir is not None:
        out_dir = Path(args.save_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        output_root = resolve_path(cfg["outputs"]["root"])
        out_dir = output_root / cfg["outputs"].get(
            "model_dir",
            "ehr_only",
        )

    cfg["outputs"]["resolved_save_dir"] = str(out_dir)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_path = model_dir / "best_by_auprc.pt"

    if best_path.exists():
        best_path.unlink()

    save_yaml(cfg, out_dir / "config_used.yaml")

    with open(out_dir / "data_summary.json", "w") as f:
        json.dump(json_safe(data_summary), f, indent=2)

    with open(out_dir / "criterion_info.json", "w") as f:
        json.dump(json_safe(criterion_info), f, indent=2)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_now = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    optimized_params_now = sum(
        p.numel()
        for group in optimizer.param_groups
        for p in group["params"]
    )

    print("=" * 100)
    print("EHR-only training")
    print("=" * 100)
    print("device:", device)
    print("use_amp:", use_amp)
    print("output:", out_dir)
    print("train/val/test:", len(train_df), len(val_df), len(test_df))
    print("train positives:", int(train_df[cols["label_col"]].sum()))
    print("val positives:", int(val_df[cols["label_col"]].sum()))
    print("test positives:", int(test_df[cols["label_col"]].sum()) if len(test_df) > 0 else 0)
    print("n_params:", int(total_params))
    print("n_trainable_now:", int(trainable_params_now))
    print("n_optimized_now:", int(optimized_params_now))

    if args.dry_run:
        batch = next(iter(train_loader))
        ehr_x = batch["ehr_x"].to(device)
        ehr_m = batch["ehr_m"].to(device)
        labels = batch["label"].to(device).float().view(-1)

        out = model(ehr_x, ehr_m)
        from respire_transfuse.training.ehr_losses import compute_ehr_loss

        losses = compute_ehr_loss(
            logits=out["logit"],
            labels=labels,
            criterion=criterion,
            loss_cfg=cfg["loss"],
            train=True,
        )

        losses["loss"].backward()

        print("\ndry run ok")
        print("batch ehr_x:", tuple(ehr_x.shape))
        print("batch ehr_m:", tuple(ehr_m.shape))
        print("batch labels:", tuple(labels.shape))
        print("logit:", tuple(out["logit"].shape))
        print("attn:", tuple(out["attn"].shape))
        print("loss:", float(losses["loss"].detach().cpu()))
        return

    history = []
    best_val_auprc = -1.0
    best_epoch = -1
    bad_epochs = 0

    total_points = int(cfg["training"]["epochs"])
    train_epochs = max(1, total_points)

    for stale_path in [
        out_dir / "calibration_bins_10_by_epoch.csv",
        out_dir / "adaptive_calibration_bins_10_by_epoch.csv",
    ]:
        if stale_path.exists():
            stale_path.unlink()

    epoch_prediction_dir = out_dir / "epoch_predictions"
    if epoch_prediction_dir.exists():
        shutil.rmtree(epoch_prediction_dir)

    for train_epoch in range(1, train_epochs + 1):
        epoch = train_epoch

        lr_factor = lr_factor_for_epoch(
            epoch=train_epoch,
            total_epochs=train_epochs,
            warmup_epochs=int(cfg["training"]["warmup_epochs"]),
            min_lr_factor=float(cfg["training"]["min_lr_factor"]),
        )

        current_lr = float(cfg["training"]["lr"]) * float(lr_factor)
        set_optimizer_lr(optimizer, current_lr)

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            loss_cfg=cfg["loss"],
            train_cfg=cfg["training"],
            device=device,
            use_amp=use_amp,
            epoch=epoch,
        )

        val_stats = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            loss_cfg=cfg["loss"],
            device=device,
            use_amp=use_amp,
            desc=f"EVAL val epoch {epoch}",
        )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "optim_train_loss": train_stats["loss"],
            "optim_train_bce": train_stats["bce"],
            "train_loss": train_stats["loss"],
            "train_bce": train_stats["bce"],
            "train_auroc": train_stats["auroc"],
            "train_auprc": train_stats["auprc"],
            "train_log_loss": train_stats["log_loss"],
            "val_loss": val_stats["loss"],
            "val_bce": val_stats["bce"],
            "val_auroc": val_stats["auroc"],
            "val_auprc": val_stats["auprc"],
            "val_log_loss": val_stats["log_loss"],
            "val_best_f1": val_stats["best_f1"],
            "val_best_f1_threshold": val_stats["best_f1_threshold"],
        }

        row.update(
            save_epoch_artifacts(
                save_dir=out_dir,
                epoch=epoch,
                split="train",
                sample_ids=None,
                y_true=train_stats["labels"],
                pred_values=train_stats["logits"],
                n_bins=10,
                save_predictions=False,
            )
        )

        row.update(
            save_epoch_artifacts(
                save_dir=out_dir,
                epoch=epoch,
                split="val",
                sample_ids=val_stats.get("sample_ids", None),
                y_true=val_stats["labels"],
                pred_values=val_stats["logits"] if "logits" in val_stats else val_stats["probs"],
                n_bins=10,
                save_predictions=True,
            )
        )

        history.append(row)
        save_history_without_accuracy_brier(history, out_dir / "history.csv")
        plot_history(out_dir / "history.csv", out_dir, title_prefix="EHR-only")

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.5f} | "
            f"train_auroc={row['train_auroc']:.5f} | "
            f"train_auprc={row['train_auprc']:.5f} | "
            f"val_loss={row['val_loss']:.5f} | "
            f"val_auroc={row['val_auroc']:.5f} | "
            f"val_auprc={row['val_auprc']:.5f}"
        )

        val_auprc = float(val_stats["auprc"])
        min_delta = float(cfg["selection"].get("min_delta_auprc", 0.0))

        if val_auprc > best_val_auprc + min_delta:
            best_val_auprc = val_auprc
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "trained_epochs": train_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "data_summary": data_summary,
                    "criterion_info": criterion_info,
                    "val_auprc": val_auprc,
                },
                model_dir / "best_by_auprc.pt",
            )
        else:
            bad_epochs += 1

        if bad_epochs >= int(cfg["training"]["patience"]):
            print(f"early stopping at epoch {epoch}")
            break

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved.")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_final = evaluate(
        model,
        val_loader,
        criterion,
        cfg["loss"],
        device,
        use_amp,
        desc="FINAL val",
    )

    if test_loader is None:
        raise RuntimeError(
            "The test split is empty or the test loader was not created."
        )

    test_final = evaluate(
        model,
        test_loader,
        criterion,
        cfg["loss"],
        device,
        use_amp,
        desc="FINAL test",
    )

    temperature, bias = fit_temperature_bias(
        val_logits=val_final["logits"],
        val_labels=val_final["labels"],
        device=device,
        max_iter=int(
            cfg["selection"]["calibration_max_iter"]
        ),
    )

    val_cal_prob = calibrated_probabilities(
        val_final["logits"],
        temperature,
        bias,
    )

    threshold, val_best_f1 = choose_threshold_by_val_f1(
        val_final["labels"],
        val_cal_prob,
    )

    test_cal_prob = (
        calibrated_probabilities(
            test_final["logits"],
            temperature,
            bias,
        )
        if test_final is not None
        else None
    )

    final_metrics = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_auprc": float(
            checkpoint["val_auprc"]
        ),
        "calibration": {
            "temperature": float(temperature),
            "bias": float(bias),
            "threshold": float(threshold),
            "val_best_f1": float(val_best_f1),
        },
        "raw": {
            "val": summarize_probabilities(
                val_final["labels"],
                sigmoid_np(val_final["logits"]),
            ),
            "test": (
                summarize_probabilities(
                    test_final["labels"],
                    sigmoid_np(test_final["logits"]),
                )
                if test_final is not None
                else None
            ),
        },
        "calibrated": {
            "val": summarize_probabilities(
                val_final["labels"],
                val_cal_prob,
                threshold=threshold,
            ),
            "test": (
                summarize_probabilities(
                    test_final["labels"],
                    test_cal_prob,
                    threshold=threshold,
                )
                if test_final is not None
                else None
            ),
        },
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            json_safe(
                remove_unsaved_metrics(
                    final_metrics
                )
            ),
            f,
            indent=2,
        )

    save_predictions(
        val_final,
        out_dir / "val_predictions.csv",
        threshold=threshold,
        temperature=temperature,
        bias=bias,
    )

    print("\nfinal EHR-only validation:")
    val_metrics = final_metrics["calibrated"]["val"]
    print(
        f"AUROC={val_metrics['auroc']:.5f} | "
        f"AUPRC={val_metrics['auprc']:.5f}"
    )

    save_predictions(
        test_final,
        out_dir / "test_predictions.csv",
        threshold=threshold,
        temperature=temperature,
        bias=bias,
    )

    print("\nfinal EHR-only test:")
    test_metrics = final_metrics["calibrated"]["test"]
    print(
        f"AUROC={test_metrics['auroc']:.5f} | "
        f"AUPRC={test_metrics['auprc']:.5f}"
    )

    print("\nsaved:")
    print(out_dir)


if __name__ == "__main__":
    main()
