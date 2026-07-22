#!/usr/bin/env python3
"""Train and evaluate the summary-level CXR and EHR fusion baseline.

The script loads the paired multimodal cohort, restores the configured unimodal
branch checkpoints, and concatenates projected image and EHR summaries before a
shared prediction head. It manages branch freezing and fine-tuning, chooses the
best model and classification threshold from validation performance, and saves
configuration snapshots, histories, plots, checkpoints, metrics, and test-set
predictions in the selected run directory.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respire_transfuse.utils.config import load_config, save_yaml
from respire_transfuse.training.seed import seed_everything
from respire_transfuse.training.metrics import json_safe
from respire_transfuse.training.plots import plot_history
from respire_transfuse.utils.epoch_metrics import save_epoch_artifacts
from respire_transfuse.data.image_dataset import build_image_transforms
from respire_transfuse.data.ehr_dataset import BalancedBinaryBatchSampler
from respire_transfuse.data.multimodal_dataset import (
    load_multimodal_splits,
    MultimodalRespireDataset,
)
from respire_transfuse.models.early_fusion import (
    build_early_fusion_from_config,
    configure_early_fusion_trainability,
)
from respire_transfuse.training.engine import (
    train_multimodal_one_epoch,
    evaluate_multimodal,
    lr_factor_for_epoch,
    metrics_at_threshold,
    save_multimodal_predictions,
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


def build_loader(dataset, batch_size, num_workers, train, sampling_cfg=None, labels=None, seed=42):
    if train and sampling_cfg is not None and bool(sampling_cfg.get("balanced_batches", False)):
        sampler = BalancedBinaryBatchSampler(
            labels=labels,
            batch_size=int(batch_size),
            pos_fraction=float(sampling_cfg.get("pos_fraction", 0.35)),
            batches_per_epoch=sampling_cfg.get("batches_per_epoch", None),
            seed=int(seed),
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=int(num_workers) > 0,
        )

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


def get_state_dict_from_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "model", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

    if isinstance(ckpt, dict):
        return ckpt

    raise RuntimeError("Unsupported checkpoint format.")


def load_checkpoint(module, path, strict=False):
    path = Path(path)

    if not path.exists():
        return {
            "path": str(path),
            "loaded": False,
            "reason": "file_not_found",
        }

    ckpt = torch.load(path, map_location="cpu")
    state = get_state_dict_from_checkpoint(ckpt)

    result = module.load_state_dict(state, strict=bool(strict))

    return {
        "path": str(path),
        "loaded": True,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def collect_trainable_params(items):
    params = []
    seen = set()

    for item in items:
        if item is None:
            continue

        if isinstance(item, torch.nn.Parameter):
            candidates = [item]
        else:
            candidates = list(item.parameters())

        for p in candidates:
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))

    return params


def build_parameter_groups(model, train_cfg):
    image_head = collect_trainable_params([
        model.image_branch.attention_score,
        model.image_branch.attention_mix_logit,
        model.image_branch.classifier,
    ])

    ehr = collect_trainable_params([
        model.ehr_branch,
    ])

    fusion = collect_trainable_params([
        model.image_fusion_proj,
        model.ehr_fusion_proj,
        model.fusion_head,
    ])

    candidates = [
        {
            "name": "image_head",
            "params": image_head,
            "lr": float(
                train_cfg["lr_image_head"]
            ),
            "weight_decay": float(
                train_cfg["weight_decay_image"]
            ),
        },
        {
            "name": "ehr",
            "params": ehr,
            "lr": float(
                train_cfg["lr_ehr"]
            ),
            "weight_decay": float(
                train_cfg["weight_decay_ehr"]
            ),
        },
        {
            "name": "fusion",
            "params": fusion,
            "lr": float(
                train_cfg["lr_fusion"]
            ),
            "weight_decay": float(
                train_cfg["weight_decay_fusion"]
            ),
        },
    ]

    groups = []
    seen = set()

    for group in candidates:
        params = []

        for parameter in group["params"]:
            parameter_id = id(parameter)

            if parameter_id not in seen:
                params.append(parameter)
                seen.add(parameter_id)

        if params:
            group["params"] = params
            group["base_lr"] = float(
                group["lr"]
            )
            groups.append(group)

    required = {
        "image_head",
        "ehr",
        "fusion",
    }

    found = {
        group["name"]
        for group in groups
    }

    if found != required:
        raise RuntimeError(
            f"Incorrect optimizer groups: {found}"
        )

    return groups


def set_group_lrs(optimizer, lr_factor):
    current = {}

    for group in optimizer.param_groups:
        base_lr = float(group.get("base_lr", group["lr"]))
        group["lr"] = base_lr * float(lr_factor)
        current[group.get("name", "group")] = float(group["lr"])

    return current


def _stats_value(stats, names):
    for name in names:
        if name in stats:
            return stats[name]
    raise KeyError(f"Missing required stats key. Tried: {names}")


def _stats_optional(stats, names):
    for name in names:
        if name in stats:
            return stats[name]
    return None


def _add_epoch_artifacts(row, save_dir, epoch, split, stats, save_predictions):
    labels = _stats_value(stats, ["labels", "y_true", "targets", "target"])
    pred_values = _stats_value(stats, ["logits", "fusion_logits", "probs", "probabilities", "preds", "predictions"])
    sample_ids = _stats_optional(stats, ["sample_ids", "ids", "sample_id"])

    row.update(
        save_epoch_artifacts(
            save_dir=save_dir,
            epoch=epoch,
            split=split,
            sample_ids=sample_ids,
            y_true=labels,
            pred_values=pred_values,
            n_bins=10,
            save_predictions=save_predictions,
        )
    )

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
        return [
            remove_unsaved_metrics(value)
            for value in obj
        ]

    return obj

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--paths", type=str, default="configs/paths.yaml")
    parser.add_argument("--config", type=str, default="configs/experiments/early_fusion.yaml")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--weight_decay_image", type=float, default=None)
    parser.add_argument("--weight_decay_ehr", type=float, default=None)
    parser.add_argument("--weight_decay_fusion", type=float, default=None)
    parser.add_argument("--lr_image_head", type=float, default=None)
    parser.add_argument("--lr_ehr", type=float, default=None)
    parser.add_argument("--lr_fusion", type=float, default=None)
    parser.add_argument("--fusion_dropout", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

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

    if args.lr is not None:
        cfg["training"]["lr_fusion"] = float(args.lr)

    if args.weight_decay is not None:
        cfg["training"]["weight_decay_image"] = float(args.weight_decay)
        cfg["training"]["weight_decay_ehr"] = float(args.weight_decay)
        cfg["training"]["weight_decay_fusion"] = float(args.weight_decay)

    if args.weight_decay_image is not None:
        cfg["training"]["weight_decay_image"] = float(
            args.weight_decay_image
        )

    if args.weight_decay_ehr is not None:
        cfg["training"]["weight_decay_ehr"] = float(
            args.weight_decay_ehr
        )

    if args.weight_decay_fusion is not None:
        cfg["training"]["weight_decay_fusion"] = float(
            args.weight_decay_fusion
        )

    if args.lr_image_head is not None:
        cfg["training"]["lr_image_head"] = float(args.lr_image_head)

    if args.lr_ehr is not None:
        cfg["training"]["lr_ehr"] = float(args.lr_ehr)

    if args.lr_fusion is not None:
        cfg["training"]["lr_fusion"] = float(args.lr_fusion)

    if args.fusion_dropout is not None:
        cfg["model"]["fusion_dropout"] = float(args.fusion_dropout)

    if args.grad_clip is not None:
        cfg["training"]["grad_clip"] = float(args.grad_clip)

    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)

    seed_everything(int(cfg["training"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["training"].get("use_amp", False)) and device.type == "cuda"

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]

    cohort_csv = resolve_path(data_cfg["cohort_csv"])
    ehr_npz = resolve_path(data_cfg["ehr_npz"])

    train_df, val_df, test_df, X, M, feature_names, cohort_dir, data_summary = load_multimodal_splits(
        cohort_csv=cohort_csv,
        ehr_npz=ehr_npz,
        sample_col=data_cfg["sample_col"],
        label_col=data_cfg["label_col"],
        split_col=data_cfg["split_col"],
        image_col=data_cfg["image_col"],
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
        image_size=int(cfg["image_branch"]["image_size"]),
        hflip_p=float(cfg["augmentation"]["hflip_p"]),
        affine_p=float(cfg["augmentation"]["affine_p"]),
        color_jitter_p=float(cfg["augmentation"]["color_jitter_p"]),
    )

    train_set = MultimodalRespireDataset(
        train_df,
        X,
        M,
        image_col=data_cfg["image_col"],
        sample_col=data_cfg["sample_col"],
        label_col=data_cfg["label_col"],
        output_root=ROOT,
        cohort_dir=cohort_dir,
        transform=train_tf,
    )

    val_set = MultimodalRespireDataset(
        val_df,
        X,
        M,
        image_col=data_cfg["image_col"],
        sample_col=data_cfg["sample_col"],
        label_col=data_cfg["label_col"],
        output_root=ROOT,
        cohort_dir=cohort_dir,
        transform=eval_tf,
    )

    test_set = MultimodalRespireDataset(
        test_df,
        X,
        M,
        image_col=data_cfg["image_col"],
        sample_col=data_cfg["sample_col"],
        label_col=data_cfg["label_col"],
        output_root=ROOT,
        cohort_dir=cohort_dir,
        transform=eval_tf,
    )

    train_loader = build_loader(
        train_set,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        train=True,
        sampling_cfg=cfg["sampling"],
        labels=train_df[data_cfg["label_col"]].astype(int).values,
        seed=int(train_cfg["seed"]),
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

    model = build_early_fusion_from_config(
        cfg,
        n_ehr_features=int(X.shape[-1]),
    )

    prior_info = {
        "initialization": "zero_final_bias",
        "prevalence_bias_initialization": False,
    }

    load_reports = {
        "mode": "scratch_early_fusion_no_checkpoint_loading",
        "ehr": {
            "loaded": False,
            "reason": "disabled_by_config",
        },
        "image": {
            "loaded": False,
            "reason": "disabled_by_config",
        },
    }

    if bool(cfg["checkpoints"].get("load_ehr", False)):
        load_reports["ehr"] = load_checkpoint(
            model.ehr_branch,
            resolve_path(cfg["checkpoints"]["ehr_path"]),
            strict=bool(cfg["checkpoints"].get("strict", False)),
        )

    if bool(cfg["checkpoints"].get("load_image", False)):
        load_reports["image"] = load_checkpoint(
            model.image_branch,
            resolve_path(cfg["checkpoints"]["image_path"]),
            strict=bool(cfg["checkpoints"].get("strict", False)),
        )

    configure_early_fusion_trainability(model, cfg["freeze"])
    model = model.to(device)

    criterion, criterion_info = make_criterion(
        train_df,
        data_cfg["label_col"],
        loss_cfg,
        device,
    )

    parameter_groups = build_parameter_groups(model, train_cfg)

    optimizer = torch.optim.AdamW(parameter_groups)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(use_amp),
    )

    output_root = resolve_path(cfg["outputs"]["root"])
    out_dir = output_root / cfg["outputs"]["model_dir"]

    if args.save_dir is not None:
        out_dir = Path(args.save_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir

    cfg["outputs"]["resolved_save_dir"] = str(out_dir)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_path = model_dir / "best_by_auprc.pt"

    if best_path.exists():
        best_path.unlink()

    for stale_name in [
        "calibration_bins_10_by_epoch.csv",
        "adaptive_calibration_bins_10_by_epoch.csv",
        "epoch_predictions",
    ]:
        stale_path = out_dir / stale_name
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        elif stale_path.exists():
            stale_path.unlink()

    save_yaml(cfg, out_dir / "config_used.yaml")

    with open(out_dir / "data_summary.json", "w") as f:
        json.dump(json_safe(data_summary), f, indent=2)

    with open(out_dir / "criterion_info.json", "w") as f:
        json.dump(json_safe(criterion_info), f, indent=2)

    with open(out_dir / "checkpoint_load_report.json", "w") as f:
        json.dump(json_safe(load_reports), f, indent=2)

    with open(out_dir / "prior_info.json", "w") as f:
        json.dump(json_safe(prior_info), f, indent=2)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_n = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 100)
    print("EarlyFusion training")
    print("=" * 100)
    print("device:", device)
    print("use_amp:", use_amp)
    print("output:", out_dir)
    print("cohort_csv:", cohort_csv)
    print("ehr_npz:", ehr_npz)
    print("image_col:", data_cfg["image_col"])
    print("train/val/test:", len(train_df), len(val_df), len(test_df))
    print("train positives:", int(train_df[data_cfg["label_col"]].sum()))
    print("val positives:", int(val_df[data_cfg["label_col"]].sum()))
    print("test positives:", int(test_df[data_cfg["label_col"]].sum()))
    print("n_params:", int(total_params))
    print("n_trainable:", int(trainable_params_n))

    if args.dry_run:
        batch = next(iter(train_loader))

        image = batch["image"].to(device)
        ehr_x = batch["ehr_x"].to(device)
        ehr_m = batch["ehr_m"].to(device)
        labels = batch["label"].to(device).float().view(-1)

        out = model(image=image, ehr_x=ehr_x, ehr_m=ehr_m, return_all=True)

        from respire_transfuse.training.engine import compute_multimodal_loss

        losses = compute_multimodal_loss(
            out=out,
            labels=labels,
            criterion=criterion,
            loss_cfg=loss_cfg,
        )

        losses["loss"].backward()

        print("\ndry run ok")
        print("image:", tuple(image.shape))
        print("ehr_x:", tuple(ehr_x.shape))
        print("ehr_m:", tuple(ehr_m.shape))
        print("label:", tuple(labels.shape))
        print("fusion_logit:", tuple(out["fusion_logit"].shape))
        print("ehr_logit:", tuple(out["ehr_logit"].shape))
        print("image_logit:", tuple(out["image_logit"].shape))
        print("loss:", float(losses["loss"].detach().cpu()))
        return

    history = []
    best_val_auprc = -1.0
    best_epoch = -1
    bad_epochs = 0

    total_points = int(train_cfg["epochs"])
    train_epochs = max(1, total_points)


    for train_epoch in range(1, train_epochs + 1):
        epoch = train_epoch

        lr_factor = lr_factor_for_epoch(
            epoch=train_epoch,
            total_epochs=train_epochs,
            warmup_epochs=int(train_cfg["warmup_epochs"]),
            min_lr_factor=float(train_cfg["min_lr_factor"]),
        )

        current_lrs = set_group_lrs(optimizer, lr_factor)


        train_stats = train_multimodal_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            loss_cfg=loss_cfg,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            grad_clip=float(train_cfg["grad_clip"]),
            epoch=epoch,
        )

        val_stats = evaluate_multimodal(
            model=model,
            loader=val_loader,
            criterion=criterion,
            loss_cfg=loss_cfg,
            device=device,
            use_amp=use_amp,
            desc=f"EVAL val epoch {epoch}",
        )

        row = {
            "epoch": epoch,
            "lr_image_head": float(current_lrs.get("image_head", 0.0)),
            "lr_ehr": float(current_lrs.get("ehr", 0.0)),
            "lr_fusion": float(current_lrs.get("fusion", 0.0)),
            "optim_train_loss": train_stats["loss"],
            "optim_train_bce": train_stats["fusion_bce"],
            "train_loss": train_stats["loss"],
            "train_bce": train_stats["fusion_bce"],
            "train_auroc": train_stats["auroc"],
            "train_auprc": train_stats["auprc"],
            "train_log_loss": train_stats["log_loss"],
            "val_loss": val_stats["loss"],
            "val_bce": val_stats["fusion_bce"],
            "val_auroc": val_stats["auroc"],
            "val_auprc": val_stats["auprc"],
            "val_log_loss": val_stats["log_loss"],
            "val_best_f1": val_stats["best_f1"],
            "val_best_f1_threshold": val_stats["best_f1_threshold"],
        }


        _add_epoch_artifacts(
            row,
            save_dir=out_dir,
            epoch=epoch,
            split="train",
            stats=train_stats,
            save_predictions=False,
        )

        _add_epoch_artifacts(
            row,
            save_dir=out_dir,
            epoch=epoch,
            split="val",
            stats=val_stats,
            save_predictions=True,
        )

        history.append(row)
        save_history_without_accuracy_brier(history, out_dir / "history.csv")
        plot_history(out_dir / "history.csv", out_dir, title_prefix="EarlyFusion")

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
        min_delta = float(cfg["selection"]["min_delta_auprc"])

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
                    "checkpoint_load_report": load_reports,
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

    val_final = evaluate_multimodal(
        model=model,
        loader=val_loader,
        criterion=criterion,
        loss_cfg=loss_cfg,
        device=device,
        use_amp=use_amp,
        desc="FINAL val",
    )

    test_final = evaluate_multimodal(
        model=model,
        loader=test_loader,
        criterion=criterion,
        loss_cfg=loss_cfg,
        device=device,
        use_amp=use_amp,
        desc="FINAL test",
    )

    threshold = float(
        val_final["best_f1_threshold"]
    )

    final_metrics = {
        "best_epoch": int(
            checkpoint["epoch"]
        ),
        "best_val_auprc": float(
            checkpoint["val_auprc"]
        ),
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

    with open(
        out_dir / "metrics.json",
        "w",
    ) as f:
        json.dump(
            json_safe(
                remove_unsaved_metrics(
                    final_metrics
                )
            ),
            f,
            indent=2,
        )

    save_multimodal_predictions(
        val_final,
        out_dir / "val_predictions.csv",
        threshold=threshold,
    )

    save_multimodal_predictions(
        test_final,
        out_dir / "test_predictions.csv",
        threshold=threshold,
    )

    print("\nfinal EarlyFusion test:")
    test_metrics = final_metrics["raw"]["test"]
    print(f"AUROC={test_metrics['auroc']:.5f} | AUPRC={test_metrics['auprc']:.5f}")


    print("\nsaved:")
    print(out_dir)


if __name__ == "__main__":
    main()
