#!/usr/bin/env python3

import argparse
import json
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "ignore",
    message="Mapping deprecated model name .*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests to the HF Hub.*",
)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respire_transfuse.utils.config import load_config, save_yaml
from respire_transfuse.training.seed import seed_everything
from respire_transfuse.training.metrics import json_safe
from respire_transfuse.training.plots import plot_history
from respire_transfuse.utils.epoch_metrics import save_epoch_artifacts
from respire_transfuse.data.image_dataset import (
    load_image_splits,
    CXRImageDataset,
    build_image_transforms,
)
from respire_transfuse.models.image_only import (
    ConservativeImageModel,
    set_backbone_trainable,
    create_ema_model,
)
from respire_transfuse.training.engine import (
    train_image_one_epoch,
    evaluate_image,
    lr_factor_for_epoch,
    set_image_optimizer_lrs,
    metrics_at_threshold,
    save_image_predictions,
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

    out = pd.concat([pos.head(n_pos), neg.head(n_neg)], axis=0)

    if len(out) < n:
        rest = df.drop(index=out.index, errors="ignore").head(n - len(out))
        out = pd.concat([out, rest], axis=0)

    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def build_loader(dataset, batch_size, num_workers, train):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(train),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(num_workers) > 0,
        drop_last=False,
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
        used_pos_weight = min(raw_pos_weight, cap)
    else:
        used_pos_weight = raw_pos_weight

    pos_weight = torch.tensor([used_pos_weight], dtype=torch.float32, device=device)

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight), {
        "train_pos": pos,
        "train_neg": neg,
        "raw_pos_weight": float(raw_pos_weight),
        "used_pos_weight": float(used_pos_weight),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--paths", type=str, default="configs/paths.yaml")
    parser.add_argument("--config", type=str, default="configs/experiments/image_only.yaml")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--dropout", type=float, default=0.75)
    parser.add_argument("--lr_head", type=float, default=2e-5)
    parser.add_argument("--lr_backbone", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
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

    if args.lr_head is not None:
        cfg["training"]["lr_head"] = float(args.lr_head)

    if args.lr_backbone is not None:
        cfg["training"]["lr_backbone"] = float(args.lr_backbone)

    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = float(args.weight_decay)

    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)

    if args.pretrained is not None:
        cfg["model"]["pretrained"] = bool(args.pretrained)
    elif "dummy_100" in str(args.paths) or (args.save_dir is not None and "dummy_100" in str(args.save_dir)):
        cfg["model"]["pretrained"] = False

    seed_everything(int(cfg["training"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["training"].get("use_amp", False)) and device.type == "cuda"

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    loss_cfg = cfg["loss"]

    cohort_csv = resolve_path(data_cfg["cohort_csv"])

    train_df, val_df, test_df, cohort_dir, data_summary = load_image_splits(
        cohort_csv=cohort_csv,
        image_col=data_cfg["image_col"],
        label_col=data_cfg["label_col"],
        split_col=data_cfg["split_col"],
        sample_col=data_cfg["sample_col"],
        require_image_exists=bool(data_cfg.get("require_image_exists", True)),
        require_image_decode_ok=bool(data_cfg.get("require_image_decode_ok", True)),
    )

    if args.debug_n is not None:
        train_df = limit_df_mixed(train_df, args.debug_n, data_cfg["label_col"])
        val_df = limit_df_mixed(val_df, max(16, args.debug_n // 2), data_cfg["label_col"])
        test_df = limit_df_mixed(test_df, max(16, args.debug_n // 2), data_cfg["label_col"])

        data_summary["debug_n"] = int(args.debug_n)
        data_summary["train_rows"] = int(len(train_df))
        data_summary["val_rows"] = int(len(val_df))
        data_summary["test_rows"] = int(len(test_df))
        data_summary["train_pos"] = int(train_df[data_cfg["label_col"]].sum())
        data_summary["val_pos"] = int(val_df[data_cfg["label_col"]].sum())
        data_summary["test_pos"] = int(test_df[data_cfg["label_col"]].sum())

    train_tf, eval_tf = build_image_transforms(
        image_size=int(model_cfg["image_size"]),
        hflip_p=float(aug_cfg["hflip_p"]),
        affine_p=float(aug_cfg["affine_p"]),
        color_jitter_p=float(aug_cfg["color_jitter_p"]),
    )

    train_set = CXRImageDataset(
        train_df,
        image_col=data_cfg["image_col"],
        label_col=data_cfg["label_col"],
        sample_col=data_cfg["sample_col"],
        output_root=ROOT,
        cohort_dir=cohort_dir,
        transform=train_tf,
    )

    train_eval_set = CXRImageDataset(
        train_df,
        image_col=data_cfg["image_col"],
        label_col=data_cfg["label_col"],
        sample_col=data_cfg["sample_col"],
        output_root=ROOT,
        cohort_dir=cohort_dir,
        transform=eval_tf,
    )

    val_set = CXRImageDataset(
        val_df,
        image_col=data_cfg["image_col"],
        label_col=data_cfg["label_col"],
        sample_col=data_cfg["sample_col"],
        output_root=ROOT,
        cohort_dir=cohort_dir,
        transform=eval_tf,
    )

    test_set = CXRImageDataset(
        test_df,
        image_col=data_cfg["image_col"],
        label_col=data_cfg["label_col"],
        sample_col=data_cfg["sample_col"],
        output_root=ROOT,
        cohort_dir=cohort_dir,
        transform=eval_tf,
    )

    train_loader = build_loader(
        train_set,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        train=True,
    )

    train_eval_loader = build_loader(
        train_eval_set,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        train=False,
    )

    val_loader = build_loader(
        val_set,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        train=False,
    )

    test_loader = build_loader(
        test_set,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        train=False,
    )

    model = ConservativeImageModel(
        backbone_name=model_cfg["backbone"],
        pretrained=bool(model_cfg["pretrained"]),
        dropout=float(model_cfg["dropout"]),
        hidden_mult=float(model_cfg["hidden_mult"]),
        image_token_dim=int(model_cfg.get("image_token_dim", 48)),
        token_grid_size=int(model_cfg.get("token_grid_size", 2)),
    ).to(device)

    train_prevalence = float(train_df[data_cfg["label_col"]].mean())
    prior_info = {
        "prevalence": train_prevalence,
    }

    set_backbone_trainable(model, False)

    ema_model = create_ema_model(model).to(device) if bool(train_cfg["use_ema"]) else None

    criterion, criterion_info = make_criterion(
        train_df,
        data_cfg["label_col"],
        loss_cfg,
        device,
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "name": "backbone",
                "params": list(model.backbone.parameters()),
                "lr": float(train_cfg["lr_backbone"]),
                "weight_decay": float(train_cfg["weight_decay"]),
            },
            {
                "name": "head",
                "params": list(model.classifier.parameters()),
                "lr": float(train_cfg["lr_head"]),
                "weight_decay": float(train_cfg["weight_decay"]),
            },
        ]
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(use_amp),
    )

    if args.save_dir is not None:
        out_dir = Path(args.save_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        output_root = resolve_path(cfg["outputs"]["root"])
        out_dir = output_root / cfg["outputs"]["model_dir"]

    cfg["outputs"]["resolved_save_dir"] = str(out_dir)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(cfg, out_dir / "config_used.yaml")

    with open(out_dir / "data_summary.json", "w") as f:
        json.dump(json_safe(data_summary), f, indent=2)

    with open(out_dir / "criterion_info.json", "w") as f:
        json.dump(json_safe(criterion_info), f, indent=2)

    with open(out_dir / "prior_info.json", "w") as f:
        json.dump(json_safe(prior_info), f, indent=2)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_now = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 100)
    print("Image-only training")
    print("=" * 100)
    print("device:", device)
    print("use_amp:", use_amp)
    print("output:", out_dir)
    print("cohort_csv:", cohort_csv)
    print("train/val/test:", len(train_df), len(val_df), len(test_df))
    print("train positives:", int(train_df[data_cfg["label_col"]].sum()))
    print("val positives:", int(val_df[data_cfg["label_col"]].sum()))
    print("test positives:", int(test_df[data_cfg["label_col"]].sum()))
    print("n_params:", int(total_params))
    print("n_trainable_now:", int(trainable_params_now))

    if args.dry_run:
        batch = next(iter(train_loader))
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float().view(-1)

        out = model(images)

        from respire_transfuse.training.engine import compute_image_loss

        losses = compute_image_loss(
            logits=out["logit"],
            labels=labels,
            criterion=criterion,
            loss_cfg=loss_cfg,
            train_prevalence=train_prevalence,
            train=True,
        )

        losses["loss"].backward()

        print("\ndry run ok")
        print("batch image:", tuple(images.shape))
        print("batch labels:", tuple(labels.shape))
        print("logit:", tuple(out["logit"].shape))
        print("loss:", float(losses["loss"].detach().cpu()))
        return

    history = []
    best_val_auprc = -1.0
    best_epoch = -1
    bad_epochs = 0

    total_points = int(train_cfg["epochs"])
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

    initial_train_stats = evaluate_image(
        model=model,
        loader=train_eval_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        desc="EVAL train epoch 1",
    )

    initial_val_stats = evaluate_image(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        desc="EVAL val epoch 1",
    )

    initial_row = {
        "epoch": 1,
        "lr_head": float(train_cfg["lr_head"]),
        "lr_backbone": 0.0,
        "backbone_trainable": False,
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
            sample_ids=initial_train_stats["sample_ids"],
            y_true=initial_train_stats["labels"],
            pred_values=initial_train_stats["logits"],
            n_bins=10,
            save_predictions=False,
        )
    )

    initial_row.update(
        save_epoch_artifacts(
            save_dir=out_dir,
            epoch=1,
            split="val",
            sample_ids=initial_val_stats["sample_ids"],
            y_true=initial_val_stats["labels"],
            pred_values=initial_val_stats["logits"],
            n_bins=10,
            save_predictions=True,
        )
    )

    history.append(initial_row)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    plot_history(out_dir / "history.csv", out_dir, title_prefix="Image-only")

    print(
        f"epoch 001 | "
        f"train_loss={initial_row['train_loss']:.5f} | "
        f"val_loss={initial_row['val_loss']:.5f} | "
        f"val_auroc={initial_row['val_auroc']:.5f} | "
        f"val_auprc={initial_row['val_auprc']:.5f}"
    )

    prior_info = model.initialize_prior(train_prevalence)

    if ema_model is not None:
        ema_model.load_state_dict(model.state_dict())

    with open(out_dir / "prior_info.json", "w") as f:
        json.dump(json_safe(prior_info), f, indent=2)

    for train_epoch in range(1, train_epochs + 1):
        epoch = train_epoch + 1

        backbone_trainable = (
            train_epoch > int(train_cfg["freeze_backbone_epochs"])
            and float(train_cfg["lr_backbone"]) > 0.0
        )

        set_backbone_trainable(model, backbone_trainable)

        lr_factor = lr_factor_for_epoch(
            epoch=train_epoch,
            total_epochs=train_epochs,
            warmup_epochs=int(train_cfg["warmup_epochs"]),
            min_lr_factor=float(train_cfg["min_lr_factor"]),
        )

        current_lrs = set_image_optimizer_lrs(
            optimizer=optimizer,
            lr_head=float(train_cfg["lr_head"]),
            lr_backbone=float(train_cfg["lr_backbone"]),
            lr_factor=lr_factor,
            backbone_trainable=backbone_trainable,
        )

        train_stats = train_image_one_epoch(
            model=model,
            ema_model=ema_model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            loss_cfg=loss_cfg,
            train_prevalence=train_prevalence,
            grad_clip=float(train_cfg["grad_clip"]),
            ema_decay=float(train_cfg["ema_decay"]),
        )

        eval_model = ema_model if ema_model is not None else model

        train_eval_stats = evaluate_image(
            model=eval_model,
            loader=train_eval_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            desc=f"EVAL train epoch {epoch}",
        )

        val_stats = evaluate_image(
            model=eval_model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            desc=f"EVAL val epoch {epoch}",
        )

        row = {
            "epoch": epoch,
            "lr_head": float(current_lrs["head"]),
            "lr_backbone": float(current_lrs["backbone"]),
            "backbone_trainable": bool(backbone_trainable),
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
                sample_ids=train_eval_stats["sample_ids"],
                y_true=train_eval_stats["labels"],
                pred_values=train_eval_stats["logits"],
                n_bins=10,
                save_predictions=False,
            )
        )

        row.update(
            save_epoch_artifacts(
                save_dir=out_dir,
                epoch=epoch,
                split="val",
                sample_ids=val_stats["sample_ids"],
                y_true=val_stats["labels"],
                pred_values=val_stats["logits"],
                n_bins=10,
                save_predictions=True,
            )
        )

        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        plot_history(out_dir / "history.csv", out_dir, title_prefix="Image-only")

        print(f"epoch {epoch:03d} | train_loss={row['train_loss']:.5f} | val_loss={row['val_loss']:.5f} | val_auroc={row['val_auroc']:.5f} | val_auprc={row['val_auprc']:.5f}")

        val_auprc = float(val_stats["auprc"])
        min_delta = float(train_cfg["min_delta_auprc"])

        if val_auprc > best_val_auprc + min_delta:
            best_val_auprc = val_auprc
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "trained_epochs": train_epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "data_summary": data_summary,
                    "criterion_info": criterion_info,
                    "prior_info": prior_info,
                    "val_auprc": val_auprc,
                },
                model_dir / "best_by_auprc.pt",
            )
        else:
            bad_epochs += 1

        if bad_epochs >= int(train_cfg["patience"]):
            print(f"early stopping at epoch {epoch}")
            break

    best_path = model_dir / "best_by_auprc.pt"

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved.")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_final = evaluate_image(
        model=model,
        loader=train_eval_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        desc="FINAL train",
    )

    val_final = evaluate_image(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        desc="FINAL val",
    )

    test_final = evaluate_image(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        desc="FINAL test",
    )

    threshold = float(val_final["best_f1_threshold"])

    final_metrics = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_auprc": float(checkpoint["val_auprc"]),
        "threshold_selected_on_val_f1": threshold,
        "raw": {
            "train": metrics_at_threshold(train_final["labels"], train_final["probs"], threshold),
            "val": metrics_at_threshold(val_final["labels"], val_final["probs"], threshold),
            "test": metrics_at_threshold(test_final["labels"], test_final["probs"], threshold),
        },
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(json_safe(final_metrics), f, indent=2)

    save_image_predictions(train_final, out_dir / "train_predictions.csv", threshold=threshold)
    save_image_predictions(val_final, out_dir / "val_predictions.csv", threshold=threshold)
    save_image_predictions(test_final, out_dir / "test_predictions.csv", threshold=threshold)

    print("\nfinal image-only test:")
    test_metrics = final_metrics["raw"]["test"]
    print(f"AUROC={test_metrics['auroc']:.5f} | AUPRC={test_metrics['auprc']:.5f}")

    print("\nsaved:")
    print(out_dir)


if __name__ == "__main__":
    main()
