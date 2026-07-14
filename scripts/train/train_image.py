#!/usr/bin/env python3

import argparse
import json
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

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
    ImageOnlyModel,
    create_ema_model,
)
from respire_transfuse.training.engine import (
    train_image_one_epoch,
    evaluate_image,
    lr_factor_for_epoch,
    metrics_at_threshold,
    save_image_predictions,
)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


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
        [pos.head(n_pos), neg.head(n_neg)],
        axis=0,
    )

    if len(out) < n:
        rest = df.drop(
            index=out.index,
            errors="ignore",
        ).head(n - len(out))
        out = pd.concat([out, rest], axis=0)

    return out.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(drop=True)


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
    used_pos_weight = (
        min(raw_pos_weight, cap)
        if cap > 0
        else raw_pos_weight
    )

    pos_weight = torch.tensor(
        [used_pos_weight],
        dtype=torch.float32,
        device=device,
    )

    return nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    ), {
        "train_pos": pos,
        "train_neg": neg,
        "raw_pos_weight": float(raw_pos_weight),
        "used_pos_weight": float(used_pos_weight),
    }


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def save_history_without_accuracy_brier(history, path):
    frame = pd.DataFrame(history)
    excluded = [
        column
        for column in frame.columns
        if "accuracy" in column.lower()
        or "brier" in column.lower()
    ]

    if excluded:
        frame = frame.drop(columns=excluded)

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

    parser.add_argument(
        "--paths",
        type=str,
        default="configs/paths.yaml",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/image_only.yaml",
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_delta_auprc", type=float, default=None)
    parser.add_argument("--hflip_p", type=float, default=None)
    parser.add_argument("--affine_p", type=float, default=None)
    parser.add_argument("--color_jitter_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no_pretrained",
        dest="pretrained",
        action="store_false",
    )
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
    if args.hidden_dim is not None:
        cfg["model"]["hidden_dim"] = int(args.hidden_dim)
    if args.lr is not None:
        cfg["training"]["lr"] = float(args.lr)
    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = float(
            args.weight_decay
        )
    if args.patience is not None:
        cfg["training"]["patience"] = int(args.patience)
    if args.min_delta_auprc is not None:
        cfg["training"]["min_delta_auprc"] = float(
            args.min_delta_auprc
        )
    if args.hflip_p is not None:
        cfg["augmentation"]["hflip_p"] = float(args.hflip_p)
    if args.affine_p is not None:
        cfg["augmentation"]["affine_p"] = float(args.affine_p)
    if args.color_jitter_p is not None:
        cfg["augmentation"]["color_jitter_p"] = float(
            args.color_jitter_p
        )
    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)
    if args.pretrained is not None:
        cfg["model"]["pretrained"] = bool(args.pretrained)

    seed_everything(int(cfg["training"]["seed"]))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = (
        bool(cfg["training"].get("use_amp", False))
        and device.type == "cuda"
    )

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    loss_cfg = cfg["loss"]

    cohort_csv = resolve_path(data_cfg["cohort_csv"])

    (
        train_df,
        val_df,
        test_df,
        cohort_dir,
        data_summary,
    ) = load_image_splits(
        cohort_csv=cohort_csv,
        image_col=data_cfg["image_col"],
        label_col=data_cfg["label_col"],
        split_col=data_cfg["split_col"],
        sample_col=data_cfg["sample_col"],
        require_image_exists=bool(
            data_cfg.get("require_image_exists", True)
        ),
        require_image_decode_ok=bool(
            data_cfg.get("require_image_decode_ok", True)
        ),
    )

    if args.debug_n is not None:
        train_df = limit_df_mixed(
            train_df,
            args.debug_n,
            data_cfg["label_col"],
        )
        val_df = limit_df_mixed(
            val_df,
            max(16, args.debug_n // 2),
            data_cfg["label_col"],
        )
        test_df = limit_df_mixed(
            test_df,
            max(16, args.debug_n // 2),
            data_cfg["label_col"],
        )

        data_summary["debug_n"] = int(args.debug_n)
        data_summary["train_rows"] = int(len(train_df))
        data_summary["val_rows"] = int(len(val_df))
        data_summary["test_rows"] = int(len(test_df))
        data_summary["train_pos"] = int(
            train_df[data_cfg["label_col"]].sum()
        )
        data_summary["val_pos"] = int(
            val_df[data_cfg["label_col"]].sum()
        )
        data_summary["test_pos"] = int(
            test_df[data_cfg["label_col"]].sum()
        )

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
        train_cfg["batch_size"],
        train_cfg["num_workers"],
        train=True,
    )
    val_loader = build_loader(
        val_set,
        train_cfg["batch_size"],
        train_cfg["num_workers"],
        train=False,
    )
    test_loader = build_loader(
        test_set,
        train_cfg["batch_size"],
        train_cfg["num_workers"],
        train=False,
    )

    model = ImageOnlyModel(
        backbone_name=model_cfg["backbone"],
        pretrained=bool(model_cfg["pretrained"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    train_prevalence = float(
        train_df[data_cfg["label_col"]].mean()
    )

    prior_info = {
        "prevalence": train_prevalence,
        "initialization": "default_model_initialization",
        "prevalence_bias_initialization": False,
    }

    ema_model = (
        create_ema_model(model).to(device)
        if bool(train_cfg["use_ema"])
        else None
    )

    criterion, criterion_info = make_criterion(
        train_df,
        data_cfg["label_col"],
        loss_cfg,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
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
        out_dir = (
            resolve_path(cfg["outputs"]["root"])
            / cfg["outputs"]["model_dir"]
        )

    cfg["outputs"]["resolved_save_dir"] = str(out_dir)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_path = model_dir / "best_by_auprc.pt"
    if best_path.exists():
        best_path.unlink()

    save_yaml(cfg, out_dir / "config_used.yaml")

    with open(out_dir / "data_summary.json", "w") as file:
        json.dump(json_safe(data_summary), file, indent=2)
    with open(out_dir / "criterion_info.json", "w") as file:
        json.dump(json_safe(criterion_info), file, indent=2)
    with open(out_dir / "prior_info.json", "w") as file:
        json.dump(json_safe(prior_info), file, indent=2)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

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
    print("backbone_frozen:", True)
    print("n_params:", int(total_params))
    print("n_trainable_now:", int(trainable_params))

    if args.dry_run:
        batch = next(iter(train_loader))
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float().view(-1)

        out = model(images, return_all=True)

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

        print("dry run ok")
        print("image:", tuple(images.shape))
        print("feature_map:", tuple(out["feature_map"].shape))
        print("attention_map:", tuple(out["attention_map"].shape))
        print("image_features:", tuple(out["image_features"].shape))
        print("logit:", tuple(out["logit"].shape))
        print("loss:", float(losses["loss"].detach().cpu()))
        return

    history = []
    best_val_auprc = -1.0
    bad_epochs = 0
    train_epochs = max(1, int(train_cfg["epochs"]))

    for stale_path in [
        out_dir / "calibration_bins_10_by_epoch.csv",
        out_dir / "adaptive_calibration_bins_10_by_epoch.csv",
    ]:
        if stale_path.exists():
            stale_path.unlink()

    epoch_prediction_dir = out_dir / "epoch_predictions"
    if epoch_prediction_dir.exists():
        shutil.rmtree(epoch_prediction_dir)

    for epoch in range(1, train_epochs + 1):
        lr_factor = lr_factor_for_epoch(
            epoch=epoch,
            total_epochs=train_epochs,
            warmup_epochs=int(train_cfg["warmup_epochs"]),
            min_lr_factor=float(train_cfg["min_lr_factor"]),
        )
        current_lr = float(train_cfg["lr"]) * float(lr_factor)
        set_optimizer_lr(optimizer, current_lr)

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
            "lr": current_lr,
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
                sample_ids=train_stats["sample_ids"],
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
                sample_ids=val_stats["sample_ids"],
                y_true=val_stats["labels"],
                pred_values=val_stats["logits"],
                n_bins=10,
                save_predictions=True,
            )
        )

        history.append(row)
        save_history_without_accuracy_brier(
            history,
            out_dir / "history.csv",
        )
        plot_history(
            out_dir / "history.csv",
            out_dir,
            title_prefix="Image-only",
        )

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
        min_delta = float(train_cfg["min_delta_auprc"])

        if val_auprc > best_val_auprc + min_delta:
            best_val_auprc = val_auprc
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "data_summary": data_summary,
                    "criterion_info": criterion_info,
                    "prior_info": prior_info,
                    "val_auprc": val_auprc,
                },
                best_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= int(train_cfg["patience"]):
            print(f"early stopping at epoch {epoch}")
            break

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved.")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

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
            "val": metrics_at_threshold(
                val_final["labels"],
                val_final["probs"],
                threshold,
            ),
            "test": metrics_at_threshold(
                test_final["labels"],
                test_final["probs"],
                threshold,
            ),
        },
    }

    with open(out_dir / "metrics.json", "w") as file:
        json.dump(
            json_safe(remove_unsaved_metrics(final_metrics)),
            file,
            indent=2,
        )

    save_image_predictions(
        val_final,
        out_dir / "val_predictions.csv",
        threshold=threshold,
    )
    save_image_predictions(
        test_final,
        out_dir / "test_predictions.csv",
        threshold=threshold,
    )

    test_metrics = final_metrics["raw"]["test"]
    print("final image-only test:")
    print(
        f"AUROC={test_metrics['auroc']:.5f} | "
        f"AUPRC={test_metrics['auprc']:.5f}"
    )
    print("saved:")
    print(out_dir)


if __name__ == "__main__":
    main()
