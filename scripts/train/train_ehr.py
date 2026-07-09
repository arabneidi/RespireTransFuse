#!/usr/bin/env python3

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--paths", type=str, default="configs/paths.yaml")
    parser.add_argument("--config", type=str, default="configs/experiments/ehr_only_natural_sampling.yaml")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5.5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

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

    train_eval_loader = build_loader(
        train_set,
        cfg,
        train=False,
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
    ).to(device)

    criterion, criterion_info = make_criterion(
        train_df,
        cols["label_col"],
        cfg["loss"],
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    if args.save_dir is not None:
        out_dir = Path(args.save_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        output_root = resolve_path(cfg["outputs"]["root"])
        out_dir = output_root / "ehr_only"

    cfg["outputs"]["resolved_save_dir"] = str(out_dir)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(cfg, out_dir / "config_used.yaml")

    with open(out_dir / "data_summary.json", "w") as f:
        json.dump(json_safe(data_summary), f, indent=2)

    with open(out_dir / "criterion_info.json", "w") as f:
        json.dump(json_safe(criterion_info), f, indent=2)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_now = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    train_epochs = max(1, total_points - 1)

    for stale_path in [
        out_dir / "calibration_bins_10_by_epoch.csv",
        out_dir / "adaptive_calibration_bins_10_by_epoch.csv",
    ]:
        if stale_path.exists():
            stale_path.unlink()

    epoch_prediction_dir = out_dir / "epoch_predictions"
    if epoch_prediction_dir.exists():
        shutil.rmtree(epoch_prediction_dir)

    initial_train_stats = evaluate(
        model=model,
        loader=train_eval_loader,
        criterion=criterion,
        loss_cfg=cfg["loss"],
        device=device,
        use_amp=use_amp,
        desc="EVAL train epoch 1",
    )

    initial_val_stats = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        loss_cfg=cfg["loss"],
        device=device,
        use_amp=use_amp,
        desc="EVAL val epoch 1",
    )

    initial_row = {
        "epoch": 1,
        "lr": float(cfg["training"]["lr"]),
        "train_loss": initial_train_stats["loss"],
        "train_bce": initial_train_stats["bce"],
        "train_auroc": initial_train_stats["auroc"],
        "train_auprc": initial_train_stats["auprc"],
        "train_log_loss": initial_train_stats["log_loss"],
        "train_brier": initial_train_stats["brier"],
        "val_loss": initial_val_stats["loss"],
        "val_bce": initial_val_stats["bce"],
        "val_auroc": initial_val_stats["auroc"],
        "val_auprc": initial_val_stats["auprc"],
        "val_log_loss": initial_val_stats["log_loss"],
        "val_brier": initial_val_stats["brier"],
        "val_best_f1": initial_val_stats["best_f1"],
        "val_best_f1_threshold": initial_val_stats["best_f1_threshold"],
    }

    initial_row.update(
        save_epoch_artifacts(
            save_dir=out_dir,
            epoch=1,
            split="train",
            sample_ids=initial_train_stats.get("sample_ids", None),
            y_true=initial_train_stats["labels"],
            pred_values=initial_train_stats["logits"] if "logits" in initial_train_stats else initial_train_stats["probs"],
            n_bins=10,
            save_predictions=False,
        )
    )

    initial_row.update(
        save_epoch_artifacts(
            save_dir=out_dir,
            epoch=1,
            split="val",
            sample_ids=initial_val_stats.get("sample_ids", None),
            y_true=initial_val_stats["labels"],
            pred_values=initial_val_stats["logits"] if "logits" in initial_val_stats else initial_val_stats["probs"],
            n_bins=10,
            save_predictions=True,
        )
    )

    history.append(initial_row)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    plot_history(out_dir / "history.csv", out_dir, title_prefix="EHR-only")

    print(
        f"epoch 001 | "
        f"train_loss={initial_row['train_loss']:.5f} | "
        f"val_loss={initial_row['val_loss']:.5f} | "
        f"val_auroc={initial_row['val_auroc']:.5f} | "
        f"val_auprc={initial_row['val_auprc']:.5f}"
    )

    for train_epoch in range(1, train_epochs + 1):
        epoch = train_epoch + 1

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

        train_eval_stats = evaluate(
            model=model,
            loader=train_eval_loader,
            criterion=criterion,
            loss_cfg=cfg["loss"],
            device=device,
            use_amp=use_amp,
            desc=f"EVAL train epoch {epoch}",
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
            "train_loss": train_eval_stats["loss"],
            "train_bce": train_eval_stats["bce"],
            "train_auroc": train_eval_stats["auroc"],
            "train_auprc": train_eval_stats["auprc"],
            "train_log_loss": train_eval_stats["log_loss"],
            "train_brier": train_eval_stats["brier"],
            "val_loss": val_stats["loss"],
            "val_bce": val_stats["bce"],
            "val_auroc": val_stats["auroc"],
            "val_auprc": val_stats["auprc"],
            "val_log_loss": val_stats["log_loss"],
            "val_brier": val_stats["brier"],
            "val_best_f1": val_stats["best_f1"],
            "val_best_f1_threshold": val_stats["best_f1_threshold"],
        }

        row.update(
            save_epoch_artifacts(
                save_dir=out_dir,
                epoch=epoch,
                split="train",
                sample_ids=train_eval_stats.get("sample_ids", None),
                y_true=train_eval_stats["labels"],
                pred_values=train_eval_stats["logits"] if "logits" in train_eval_stats else train_eval_stats["probs"],
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
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        plot_history(out_dir / "history.csv", out_dir, title_prefix="EHR-only")

        print(f"epoch {epoch:03d} | train_loss={row['train_loss']:.5f} | val_loss={row['val_loss']:.5f} | val_auroc={row['val_auroc']:.5f} | val_auprc={row['val_auprc']:.5f}")

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

    best_path = model_dir / "best_by_auprc.pt"

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved.")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_final = evaluate(
        model,
        train_eval_loader,
        criterion,
        cfg["loss"],
        device,
        use_amp,
        desc="FINAL train",
    )

    val_final = evaluate(
        model,
        val_loader,
        criterion,
        cfg["loss"],
        device,
        use_amp,
        desc="FINAL val",
    )

    test_final = evaluate(
        model,
        test_loader,
        criterion,
        cfg["loss"],
        device,
        use_amp,
        desc="FINAL test",
    ) if test_loader is not None else None

    temperature, bias = fit_temperature_bias(
        val_logits=val_final["logits"],
        val_labels=val_final["labels"],
        device=device,
        max_iter=int(cfg["selection"]["calibration_max_iter"]),
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

    train_cal_prob = calibrated_probabilities(train_final["logits"], temperature, bias)
    test_cal_prob = calibrated_probabilities(test_final["logits"], temperature, bias) if test_final is not None else None

    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_auprc": best_val_auprc,
        "calibration": {
            "temperature": float(temperature),
            "bias": float(bias),
            "threshold": float(threshold),
            "val_best_f1": float(val_best_f1),
        },
        "raw": {
            "train": summarize_probabilities(train_final["labels"], sigmoid_np(train_final["logits"])),
            "val": summarize_probabilities(val_final["labels"], sigmoid_np(val_final["logits"])),
            "test": summarize_probabilities(test_final["labels"], sigmoid_np(test_final["logits"])) if test_final is not None else None,
        },
        "calibrated": {
            "train": summarize_probabilities(train_final["labels"], train_cal_prob, threshold=threshold),
            "val": summarize_probabilities(val_final["labels"], val_cal_prob, threshold=threshold),
            "test": summarize_probabilities(test_final["labels"], test_cal_prob, threshold=threshold) if test_final is not None else None,
        },
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(json_safe(final_metrics), f, indent=2)

    save_predictions(
        train_final,
        out_dir / "train_predictions.csv",
        threshold=threshold,
        temperature=temperature,
        bias=bias,
    )

    save_predictions(
        val_final,
        out_dir / "val_predictions.csv",
        threshold=threshold,
        temperature=temperature,
        bias=bias,
    )

    if test_final is not None:
        save_predictions(
            test_final,
            out_dir / "test_predictions.csv",
            threshold=threshold,
            temperature=temperature,
            bias=bias,
        )

    print("\nfinal EHR-only test:")
    if test_final is not None:
        test_metrics = final_metrics["calibrated"]["test"]
        print(f"AUROC={test_metrics['auroc']:.5f} | AUPRC={test_metrics['auprc']:.5f}")
    else:
        print("no test split")

    print("\nsaved:")
    print(out_dir)


if __name__ == "__main__":
    main()
