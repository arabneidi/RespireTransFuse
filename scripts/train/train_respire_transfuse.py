#!/usr/bin/env python3
"""Train and evaluate RespireTransFuse with bidirectional cross-attention."""

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

from respire_transfuse.utils.config import (
    load_config,
    save_yaml,
)
from respire_transfuse.training.seed import (
    seed_everything,
)
from respire_transfuse.training.metrics import (
    json_safe,
)
from respire_transfuse.training.plots import (
    plot_history,
)
from respire_transfuse.utils.epoch_metrics import (
    save_epoch_artifacts,
)
from respire_transfuse.data.image_dataset import (
    build_image_transforms,
)
from respire_transfuse.data.multimodal_dataset import (
    load_multimodal_splits,
    MultimodalRespireDataset,
)
from respire_transfuse.models.respire_transfuse import (
    build_respire_transfuse_from_config,
    configure_respire_transfuse_trainability,
)
from respire_transfuse.training.engine import (
    train_multimodal_one_epoch,
    evaluate_multimodal,
    lr_factor_for_epoch,
    metrics_at_threshold,
    save_multimodal_predictions,
)


def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return ROOT / path


def limit_df_mixed(
    df,
    n,
    label_col,
):
    if n is None:
        return df

    n = int(n)

    if n <= 0 or len(df) <= n:
        return df

    pos = df[
        df[label_col] == 1
    ]

    neg = df[
        df[label_col] == 0
    ]

    n_pos = min(
        len(pos),
        max(1, n // 2),
    )

    n_neg = min(
        len(neg),
        n - n_pos,
    )

    out = pd.concat(
        [
            pos.head(n_pos),
            neg.head(n_neg),
        ],
        axis=0,
    )

    if len(out) < n:
        rest = df.drop(
            index=out.index,
            errors="ignore",
        ).head(
            n - len(out)
        )

        out = pd.concat(
            [
                out,
                rest,
            ],
            axis=0,
        )

    return out.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(
        drop=True,
    )


def build_loader(
    dataset,
    batch_size,
    num_workers,
    train,
):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(train),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(num_workers) > 0,
        drop_last=False,
    )


def make_criterion(
    train_df,
    label_col,
    loss_cfg,
):
    y = train_df[
        label_col
    ].astype(
        int
    ).values

    pos = int(
        y.sum()
    )

    neg = int(
        len(y) - pos
    )

    if pos <= 0:
        raise RuntimeError(
            "Training split has zero positives."
        )

    raw_pos_weight = (
        neg
        / max(pos, 1)
    )

    configured_cap = float(
        loss_cfg.get(
            "pos_weight_cap",
            1.0,
        )
    )

    if configured_cap != 1.0:
        raise RuntimeError(
            "Final scratch training requires "
            "loss.pos_weight_cap = 1.0."
        )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    info = {
        "train_pos": pos,
        "train_neg": neg,
        "raw_pos_weight": float(
            raw_pos_weight
        ),
        "used_pos_weight": 1.0,
        "criterion": (
            "BCEWithLogitsLoss"
        ),
        "class_weighting": False,
    }

    return criterion, info


def collect_trainable_params(
    items,
):
    params = []
    seen = set()

    for item in items:
        if item is None:
            continue

        if isinstance(
            item,
            torch.nn.Parameter,
        ):
            candidates = [
                item
            ]
        else:
            candidates = list(
                item.parameters()
            )

        for parameter in candidates:
            if (
                parameter.requires_grad
                and id(parameter) not in seen
            ):
                params.append(
                    parameter
                )

                seen.add(
                    id(parameter)
                )

    return params


def build_parameter_groups(
    model,
    train_cfg,
):
    image_backbone = (
        collect_trainable_params(
            [
                model.image_branch.backbone,
            ]
        )
    )

    image_head = (
        collect_trainable_params(
            [
                model.image_branch.attention_score,
                model.image_branch.attention_mix_logit,
                model.image_branch.classifier,
            ]
        )
    )

    ehr_core = (
        collect_trainable_params(
            [
                model.ehr_branch.cls_token,
                model.ehr_branch.input_proj,
                model.ehr_branch.local_block,
                model.ehr_branch.encoder,
                model.ehr_branch.attn_pool,
                model.ehr_branch.summary_mixer,
                model.ehr_branch.head,
            ]
        )
    )

    fusion = (
        collect_trainable_params(
            [
                model.image_token_proj,
                model.image_summary_proj,
                model.ehr_branch.fusion_token_proj,
                model.ehr_branch.fusion_summary_proj,
                model.cross_layers,
                model.fusion_head,
            ]
        )
    )

    candidates = [
        {
            "name": "image_backbone",
            "params": image_backbone,
            "lr": float(
                train_cfg[
                    "lr_image_backbone"
                ]
            ),
            "weight_decay": float(
                train_cfg[
                    "weight_decay_image"
                ]
            ),
        },
        {
            "name": "image_head",
            "params": image_head,
            "lr": float(
                train_cfg[
                    "lr_image_head"
                ]
            ),
            "weight_decay": float(
                train_cfg[
                    "weight_decay_image"
                ]
            ),
        },
        {
            "name": "ehr",
            "params": ehr_core,
            "lr": float(
                train_cfg[
                    "lr_ehr"
                ]
            ),
            "weight_decay": float(
                train_cfg[
                    "weight_decay_ehr"
                ]
            ),
        },
        {
            "name": (
                "cross_attention_fusion"
            ),
            "params": fusion,
            "lr": float(
                train_cfg[
                    "lr_fusion"
                ]
            ),
            "weight_decay": float(
                train_cfg[
                    "weight_decay_fusion"
                ]
            ),
        },
    ]

    groups = []

    for group in candidates:
        if len(
            group["params"]
        ) == 0:
            continue

        group[
            "base_lr"
        ] = float(
            group["lr"]
        )

        groups.append(
            group
        )

    if len(groups) == 0:
        raise RuntimeError(
            "No trainable parameter groups found."
        )

    return groups


def set_group_lrs(
    optimizer,
    lr_factor,
):
    current = {}

    for group in (
        optimizer.param_groups
    ):
        base_lr = float(
            group.get(
                "base_lr",
                group["lr"],
            )
        )

        group["lr"] = (
            base_lr
            * float(lr_factor)
        )

        current[
            group.get(
                "name",
                "group",
            )
        ] = float(
            group["lr"]
        )

    return current


def remove_unsaved_metrics(
    obj,
):
    if isinstance(
        obj,
        dict,
    ):
        cleaned = {}

        for key, value in (
            obj.items()
        ):
            key_lower = (
                str(key).lower()
            )

            if (
                "accuracy"
                in key_lower
                or "brier"
                in key_lower
            ):
                continue

            cleaned[key] = (
                remove_unsaved_metrics(
                    value
                )
            )

        return cleaned

    if isinstance(
        obj,
        list,
    ):
        return [
            remove_unsaved_metrics(
                value
            )
            for value in obj
        ]

    return obj


def save_history(
    history,
    path,
):
    frame = pd.DataFrame(
        history
    )

    excluded = [
        column
        for column in frame.columns
        if (
            "accuracy"
            in column.lower()
            or "brier"
            in column.lower()
        )
    ]

    if excluded:
        frame = frame.drop(
            columns=excluded
        )

    frame.to_csv(
        path,
        index=False,
    )


def parse_args():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--paths",
        type=str,
        default=(
            "configs/paths.yaml"
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "configs/experiments/"
            "respire_transfuse.yaml"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--weight_decay_image",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--weight_decay_ehr",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--weight_decay_fusion",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--lr_image_head",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--lr_ehr",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--lr_fusion",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--fusion_dropout",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=[
            "adamw",
            "adam",
        ],
        default=None,
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--cohort_csv",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--ehr_npz",
        type=str,
        default=None,
    )


    parser.add_argument(
        "--dry_run",
        action="store_true",
    )

    parser.add_argument(
        "--debug_n",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(
        resolve_path(
            args.paths
        ),
        resolve_path(
            args.config
        ),
    )

    if args.cohort_csv is not None:
        cfg["data"]["cohort_csv"] = str(
            args.cohort_csv
        )

    if args.ehr_npz is not None:
        cfg["data"]["ehr_npz"] = str(
            args.ehr_npz
        )

    if args.warmup_epochs is not None:
        cfg["training"]["warmup_epochs"] = int(
            args.warmup_epochs
        )

    if args.epochs is not None:
        cfg[
            "training"
        ][
            "epochs"
        ] = int(
            args.epochs
        )

    if args.batch_size is not None:
        cfg[
            "training"
        ][
            "batch_size"
        ] = int(
            args.batch_size
        )

    if args.num_workers is not None:
        cfg[
            "training"
        ][
            "num_workers"
        ] = int(
            args.num_workers
        )

    if args.lr is not None:
        cfg[
            "training"
        ][
            "lr_fusion"
        ] = float(
            args.lr
        )

    if args.weight_decay is not None:
        cfg[
            "training"
        ][
            "weight_decay_fusion"
        ] = float(
            args.weight_decay
        )

    if args.weight_decay_image is not None:
        cfg[
            "training"
        ][
            "weight_decay_image"
        ] = float(
            args.weight_decay_image
        )

    if args.weight_decay_ehr is not None:
        cfg[
            "training"
        ][
            "weight_decay_ehr"
        ] = float(
            args.weight_decay_ehr
        )

    if args.weight_decay_fusion is not None:
        cfg[
            "training"
        ][
            "weight_decay_fusion"
        ] = float(
            args.weight_decay_fusion
        )

    if args.lr_image_head is not None:
        cfg[
            "training"
        ][
            "lr_image_head"
        ] = float(
            args.lr_image_head
        )

    if args.lr_ehr is not None:
        cfg[
            "training"
        ][
            "lr_ehr"
        ] = float(
            args.lr_ehr
        )

    if args.lr_fusion is not None:
        cfg[
            "training"
        ][
            "lr_fusion"
        ] = float(
            args.lr_fusion
        )

    if args.fusion_dropout is not None:
        cfg[
            "model"
        ][
            "dropout"
        ] = float(
            args.fusion_dropout
        )

    if args.optimizer is not None:
        cfg[
            "training"
        ][
            "optimizer"
        ] = str(
            args.optimizer
        ).lower()

    if args.warmup_epochs is not None:
        cfg[
            "training"
        ][
            "warmup_epochs"
        ] = int(
            args.warmup_epochs
        )

    if args.seed is not None:
        cfg[
            "training"
        ][
            "seed"
        ] = int(
            args.seed
        )

    if bool(
        cfg[
            "sampling"
        ].get(
            "balanced_batches",
            False,
        )
    ):
        raise RuntimeError(
            "Final scratch training requires "
            "sampling.balanced_batches = false."
        )

    if bool(
        cfg[
            "checkpoints"
        ].get(
            "load_ehr",
            False,
        )
    ):
        raise RuntimeError(
            "Scratch multimodal training requires "
            "checkpoints.load_ehr = false."
        )

    if bool(
        cfg[
            "checkpoints"
        ].get(
            "load_image",
            False,
        )
    ):
        raise RuntimeError(
            "Scratch multimodal training requires "
            "checkpoints.load_image = false."
        )

    seed_everything(
        int(
            cfg[
                "training"
            ][
                "seed"
            ]
        )
    )

    device = torch.device(
        (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    use_amp = (
        bool(
            cfg[
                "training"
            ].get(
                "use_amp",
                False,
            )
        )
        and device.type == "cuda"
    )

    data_cfg = cfg[
        "data"
    ]

    train_cfg = cfg[
        "training"
    ]

    loss_cfg = cfg[
        "loss"
    ]

    cohort_csv = resolve_path(
        data_cfg[
            "cohort_csv"
        ]
    )

    ehr_npz = resolve_path(
        data_cfg[
            "ehr_npz"
        ]
    )

    (
        train_df,
        val_df,
        test_df,
        X,
        M,
        feature_names,
        cohort_dir,
        data_summary,
    ) = load_multimodal_splits(
        cohort_csv=cohort_csv,
        ehr_npz=ehr_npz,
        sample_col=data_cfg[
            "sample_col"
        ],
        label_col=data_cfg[
            "label_col"
        ],
        split_col=data_cfg[
            "split_col"
        ],
        image_col=data_cfg[
            "image_col"
        ],
        require_image_exists=bool(
            data_cfg.get(
                "require_image_exists",
                True,
            )
        ),
        require_image_decode_ok=bool(
            data_cfg.get(
                "require_image_decode_ok",
                True,
            )
        ),
    )

    if len(test_df) == 0:
        raise RuntimeError(
            "The test split is empty."
        )

    if args.debug_n is not None:
        train_df = limit_df_mixed(
            train_df,
            args.debug_n,
            data_cfg[
                "label_col"
            ],
        )

        val_df = limit_df_mixed(
            val_df,
            max(
                16,
                args.debug_n // 2,
            ),
            data_cfg[
                "label_col"
            ],
        )

        test_df = limit_df_mixed(
            test_df,
            max(
                16,
                args.debug_n // 2,
            ),
            data_cfg[
                "label_col"
            ],
        )

        data_summary[
            "debug_n"
        ] = int(
            args.debug_n
        )

        data_summary[
            "train_rows"
        ] = int(
            len(train_df)
        )

        data_summary[
            "val_rows"
        ] = int(
            len(val_df)
        )

        data_summary[
            "test_rows"
        ] = int(
            len(test_df)
        )

        data_summary[
            "train_pos"
        ] = int(
            train_df[
                data_cfg[
                    "label_col"
                ]
            ].sum()
        )

        data_summary[
            "val_pos"
        ] = int(
            val_df[
                data_cfg[
                    "label_col"
                ]
            ].sum()
        )

        data_summary[
            "test_pos"
        ] = int(
            test_df[
                data_cfg[
                    "label_col"
                ]
            ].sum()
        )

    train_tf, eval_tf = (
        build_image_transforms(
            image_size=int(
                cfg[
                    "image_branch"
                ][
                    "image_size"
                ]
            ),
            hflip_p=float(
                cfg[
                    "augmentation"
                ][
                    "hflip_p"
                ]
            ),
            affine_p=float(
                cfg[
                    "augmentation"
                ][
                    "affine_p"
                ]
            ),
            color_jitter_p=float(
                cfg[
                    "augmentation"
                ][
                    "color_jitter_p"
                ]
            ),
        )
    )

    train_set = (
        MultimodalRespireDataset(
            train_df,
            X,
            M,
            image_col=data_cfg[
                "image_col"
            ],
            sample_col=data_cfg[
                "sample_col"
            ],
            label_col=data_cfg[
                "label_col"
            ],
            output_root=ROOT,
            cohort_dir=cohort_dir,
            transform=train_tf,
        )
    )

    val_set = (
        MultimodalRespireDataset(
            val_df,
            X,
            M,
            image_col=data_cfg[
                "image_col"
            ],
            sample_col=data_cfg[
                "sample_col"
            ],
            label_col=data_cfg[
                "label_col"
            ],
            output_root=ROOT,
            cohort_dir=cohort_dir,
            transform=eval_tf,
        )
    )

    test_set = (
        MultimodalRespireDataset(
            test_df,
            X,
            M,
            image_col=data_cfg[
                "image_col"
            ],
            sample_col=data_cfg[
                "sample_col"
            ],
            label_col=data_cfg[
                "label_col"
            ],
            output_root=ROOT,
            cohort_dir=cohort_dir,
            transform=eval_tf,
        )
    )

    train_loader = build_loader(
        train_set,
        batch_size=train_cfg[
            "batch_size"
        ],
        num_workers=train_cfg[
            "num_workers"
        ],
        train=True,
    )

    val_loader = build_loader(
        val_set,
        batch_size=train_cfg[
            "batch_size"
        ],
        num_workers=train_cfg[
            "num_workers"
        ],
        train=False,
    )

    test_loader = build_loader(
        test_set,
        batch_size=train_cfg[
            "batch_size"
        ],
        num_workers=train_cfg[
            "num_workers"
        ],
        train=False,
    )

    model = (
        build_respire_transfuse_from_config(
            cfg,
            n_ehr_features=int(
                X.shape[-1]
            ),
        )
    )

    checkpoint_load_report = {
        "mode": (
            "scratch_multimodal_"
            "without_unimodal_checkpoints"
        ),
        "ehr": {
            "loaded": False,
            "reason": (
                "disabled_and_enforced"
            ),
        },
        "image": {
            "loaded": False,
            "reason": (
                "disabled_and_enforced"
            ),
        },
    }

    initialization_info = {
        "ehr_initialization": (
            "default_random_initialization"
        ),
        "image_head_initialization": (
            "default_random_initialization"
        ),
        "fusion_initialization": (
            "default_random_initialization"
        ),
        "prevalence_bias_initialization": False,
        "unimodal_checkpoint_loading": False,
        "image_backbone_pretrained": bool(
            cfg[
                "image_branch"
            ][
                "pretrained"
            ]
        ),
    }

    configure_respire_transfuse_trainability(
        model,
        cfg[
            "freeze"
        ],
    )

    model = model.to(
        device
    )

    criterion, criterion_info = (
        make_criterion(
            train_df,
            data_cfg[
                "label_col"
            ],
            loss_cfg,
        )
    )

    parameter_groups = (
        build_parameter_groups(
            model,
            train_cfg,
        )
    )

    optimizer_name = str(
        train_cfg.get(
            "optimizer",
            "adamw",
        )
    ).lower()

    optimizer_classes = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
    }

    if optimizer_name not in optimizer_classes:
        raise RuntimeError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    optimizer = optimizer_classes[
        optimizer_name
    ](
        parameter_groups
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=bool(
                use_amp
            ),
        )
    )

    if args.save_dir is not None:
        out_dir = Path(
            args.save_dir
        )

        if not out_dir.is_absolute():
            out_dir = (
                ROOT
                / out_dir
            )
    else:
        output_root = resolve_path(
            cfg[
                "outputs"
            ][
                "root"
            ]
        )

        out_dir = (
            output_root
            / cfg[
                "outputs"
            ][
                "model_dir"
            ]
        )

    cfg[
        "outputs"
    ][
        "resolved_save_dir"
    ] = str(
        out_dir
    )

    model_dir = (
        out_dir
        / "models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        model_dir
        / "best_by_auprc.pt"
    )

    if best_path.exists():
        best_path.unlink()

    save_yaml(
        cfg,
        out_dir
        / "config_used.yaml",
    )

    with open(
        out_dir
        / "data_summary.json",
        "w",
    ) as file:
        json.dump(
            json_safe(
                data_summary
            ),
            file,
            indent=2,
        )

    with open(
        out_dir
        / "criterion_info.json",
        "w",
    ) as file:
        json.dump(
            json_safe(
                criterion_info
            ),
            file,
            indent=2,
        )

    with open(
        out_dir
        / "checkpoint_load_report.json",
        "w",
    ) as file:
        json.dump(
            json_safe(
                checkpoint_load_report
            ),
            file,
            indent=2,
        )

    with open(
        out_dir
        / "initialization_info.json",
        "w",
    ) as file:
        json.dump(
            json_safe(
                initialization_info
            ),
            file,
            indent=2,
        )

    total_params = sum(
        parameter.numel()
        for parameter in (
            model.parameters()
        )
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in (
            model.parameters()
        )
        if parameter.requires_grad
    )

    optimized_params = sum(
        parameter.numel()
        for group in (
            optimizer.param_groups
        )
        for parameter in (
            group["params"]
        )
    )

    print(
        "=" * 100
    )

    print(
        "RespireTransFuse training"
    )

    print(
        "=" * 100
    )

    print(
        "device:",
        device,
    )

    print(
        "use_amp:",
        use_amp,
    )

    print(
        "optimizer:",
        optimizer_name,
    )

    print(
        "warmup_epochs:",
        int(
            train_cfg[
                "warmup_epochs"
            ]
        ),
    )

    print(
        "output:",
        out_dir,
    )

    print(
        "cohort_csv:",
        cohort_csv,
    )

    print(
        "ehr_npz:",
        ehr_npz,
    )

    print(
        "train/val/test:",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    print(
        "train positives:",
        int(
            train_df[
                data_cfg[
                    "label_col"
                ]
            ].sum()
        ),
    )

    print(
        "val positives:",
        int(
            val_df[
                data_cfg[
                    "label_col"
                ]
            ].sum()
        ),
    )

    print(
        "test positives:",
        int(
            test_df[
                data_cfg[
                    "label_col"
                ]
            ].sum()
        ),
    )

    print(
        "n_params:",
        int(
            total_params
        ),
    )

    print(
        "n_trainable_now:",
        int(
            trainable_params
        ),
    )

    if args.dry_run:
        batch = next(
            iter(
                train_loader
            )
        )

        image = batch[
            "image"
        ].to(
            device
        )

        ehr_x = batch[
            "ehr_x"
        ].to(
            device
        )

        ehr_m = batch[
            "ehr_m"
        ].to(
            device
        )

        labels = batch[
            "label"
        ].to(
            device
        ).float().view(
            -1
        )

        out = model(
            image=image,
            ehr_x=ehr_x,
            ehr_m=ehr_m,
            return_all=True,
        )

        from respire_transfuse.training.engine import (
            compute_multimodal_loss,
        )

        losses = (
            compute_multimodal_loss(
                out=out,
                labels=labels,
                criterion=criterion,
                loss_cfg=loss_cfg,
            )
        )

        losses[
            "loss"
        ].backward()

        print(
            "\ndry run ok"
        )

        print(
            "image:",
            tuple(
                image.shape
            ),
        )

        print(
            "ehr_x:",
            tuple(
                ehr_x.shape
            ),
        )

        print(
            "ehr_m:",
            tuple(
                ehr_m.shape
            ),
        )

        print(
            "label:",
            tuple(
                labels.shape
            ),
        )

        print(
            "fusion_logit:",
            tuple(
                out[
                    "fusion_logit"
                ].shape
            ),
        )

        print(
            "ehr_logit:",
            tuple(
                out[
                    "ehr_logit"
                ].shape
            ),
        )

        print(
            "image_logit:",
            tuple(
                out[
                    "image_logit"
                ].shape
            ),
        )

        print(
            "loss:",
            float(
                losses[
                    "loss"
                ].detach().cpu()
            ),
        )

        return

    history = []
    best_val_auprc = -1.0
    best_epoch = -1
    bad_epochs = 0

    total_epochs = int(
        train_cfg[
            "epochs"
        ]
    )

    if total_epochs <= 0:
        raise RuntimeError(
            "training.epochs must be positive."
        )

    for stale_path in [
        out_dir
        / "calibration_bins_10_by_epoch.csv",
        out_dir
        / "adaptive_calibration_bins_10_by_epoch.csv",
        out_dir
        / "history.csv",
        out_dir
        / "metrics.json",
        out_dir
        / "val_predictions.csv",
        out_dir
        / "test_predictions.csv",
        out_dir
        / "train_predictions.csv",
    ]:
        if stale_path.exists():
            stale_path.unlink()

    epoch_prediction_dir = (
        out_dir
        / "epoch_predictions"
    )

    if epoch_prediction_dir.exists():
        shutil.rmtree(
            epoch_prediction_dir
        )

    for epoch in range(
        1,
        total_epochs + 1,
    ):
        lr_factor = (
            lr_factor_for_epoch(
                epoch=epoch,
                total_epochs=total_epochs,
                warmup_epochs=int(
                    train_cfg[
                        "warmup_epochs"
                    ]
                ),
                min_lr_factor=float(
                    train_cfg[
                        "min_lr_factor"
                    ]
                ),
            )
        )

        current_lrs = (
            set_group_lrs(
                optimizer,
                lr_factor,
            )
        )

        train_stats = (
            train_multimodal_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                loss_cfg=loss_cfg,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                grad_clip=float(
                    train_cfg[
                        "grad_clip"
                    ]
                ),
                epoch=epoch,
            )
        )

        val_stats = (
            evaluate_multimodal(
                model=model,
                loader=val_loader,
                criterion=criterion,
                loss_cfg=loss_cfg,
                device=device,
                use_amp=use_amp,
                desc=(
                    f"EVAL val epoch {epoch}"
                ),
            )
        )

        row = {
            "epoch": epoch,
            "lr_image_backbone": float(
                current_lrs.get(
                    "image_backbone",
                    0.0,
                )
            ),
            "lr_image_head": float(
                current_lrs.get(
                    "image_head",
                    0.0,
                )
            ),
            "lr_ehr": float(
                current_lrs.get(
                    "ehr",
                    0.0,
                )
            ),
            "lr_fusion": float(
                current_lrs.get(
                    "cross_attention_fusion",
                    0.0,
                )
            ),
            "train_total_objective": float(
                train_stats[
                    "loss"
                ]
            ),
            "train_loss": float(
                train_stats[
                    "fusion_bce"
                ]
            ),
            "train_fusion_bce": float(
                train_stats[
                    "fusion_bce"
                ]
            ),
            "train_ehr_bce": float(
                train_stats[
                    "ehr_bce"
                ]
            ),
            "train_image_bce": float(
                train_stats[
                    "image_bce"
                ]
            ),
            "train_auroc": float(
                train_stats[
                    "auroc"
                ]
            ),
            "train_auprc": float(
                train_stats[
                    "auprc"
                ]
            ),
            "train_log_loss": float(
                train_stats[
                    "log_loss"
                ]
            ),
            "val_total_objective": float(
                val_stats[
                    "loss"
                ]
            ),
            "val_loss": float(
                val_stats[
                    "fusion_bce"
                ]
            ),
            "val_fusion_bce": float(
                val_stats[
                    "fusion_bce"
                ]
            ),
            "val_ehr_bce": float(
                val_stats[
                    "ehr_bce"
                ]
            ),
            "val_image_bce": float(
                val_stats[
                    "image_bce"
                ]
            ),
            "val_auroc": float(
                val_stats[
                    "auroc"
                ]
            ),
            "val_auprc": float(
                val_stats[
                    "auprc"
                ]
            ),
            "val_log_loss": float(
                val_stats[
                    "log_loss"
                ]
            ),
            "val_best_f1": float(
                val_stats[
                    "best_f1"
                ]
            ),
            "val_best_f1_threshold": float(
                val_stats[
                    "best_f1_threshold"
                ]
            ),
        }

        row.update(
            save_epoch_artifacts(
                save_dir=out_dir,
                epoch=epoch,
                split="train",
                sample_ids=train_stats.get(
                    "sample_ids",
                    None,
                ),
                y_true=train_stats[
                    "labels"
                ],
                pred_values=train_stats[
                    "logits"
                ],
                n_bins=10,
                save_predictions=False,
            )
        )

        row.update(
            save_epoch_artifacts(
                save_dir=out_dir,
                epoch=epoch,
                split="val",
                sample_ids=val_stats.get(
                    "sample_ids",
                    None,
                ),
                y_true=val_stats[
                    "labels"
                ],
                pred_values=val_stats[
                    "logits"
                ],
                n_bins=10,
                save_predictions=True,
            )
        )

        row = (
            remove_unsaved_metrics(
                row
            )
        )

        history.append(
            row
        )

        save_history(
            history,
            out_dir
            / "history.csv",
        )

        plot_history(
            out_dir
            / "history.csv",
            out_dir,
            title_prefix=(
                "RespireTransFuse"
            ),
        )

        print(
            f"epoch {epoch:03d} | "
            f"train_loss="
            f"{row['train_loss']:.5f} | "
            f"train_auroc="
            f"{row['train_auroc']:.5f} | "
            f"train_auprc="
            f"{row['train_auprc']:.5f} | "
            f"val_loss="
            f"{row['val_loss']:.5f} | "
            f"val_auroc="
            f"{row['val_auroc']:.5f} | "
            f"val_auprc="
            f"{row['val_auprc']:.5f}"
        )

        val_auprc = float(
            val_stats[
                "auprc"
            ]
        )

        min_delta = float(
            cfg[
                "selection"
            ].get(
                "min_delta_auprc",
                0.0,
            )
        )

        if (
            val_auprc
            > best_val_auprc
            + min_delta
        ):
            best_val_auprc = (
                val_auprc
            )

            best_epoch = (
                epoch
            )

            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "trained_epochs": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "config": cfg,
                    "data_summary": (
                        data_summary
                    ),
                    "criterion_info": (
                        criterion_info
                    ),
                    "checkpoint_load_report": (
                        checkpoint_load_report
                    ),
                    "initialization_info": (
                        initialization_info
                    ),
                    "val_auprc": (
                        val_auprc
                    ),
                    "val_auroc": float(
                        val_stats[
                            "auroc"
                        ]
                    ),
                },
                best_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= int(
            train_cfg[
                "patience"
            ]
        ):
            print(
                f"early stopping "
                f"at epoch {epoch}"
            )

            break

    if not best_path.exists():
        raise RuntimeError(
            "No best checkpoint was saved."
        )

    try:
        checkpoint = torch.load(
            best_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            best_path,
            map_location=device,
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    val_final = (
        evaluate_multimodal(
            model=model,
            loader=val_loader,
            criterion=criterion,
            loss_cfg=loss_cfg,
            device=device,
            use_amp=use_amp,
            desc="FINAL val",
        )
    )

    test_final = (
        evaluate_multimodal(
            model=model,
            loader=test_loader,
            criterion=criterion,
            loss_cfg=loss_cfg,
            device=device,
            use_amp=use_amp,
            desc="FINAL test",
        )
    )

    threshold = float(
        val_final[
            "best_f1_threshold"
        ]
    )

    val_metrics = (
        metrics_at_threshold(
            val_final[
                "labels"
            ],
            val_final[
                "probs"
            ],
            threshold,
        )
    )

    test_metrics = (
        metrics_at_threshold(
            test_final[
                "labels"
            ],
            test_final[
                "probs"
            ],
            threshold,
        )
    )

    final_metrics = {
        "best_epoch": int(
            checkpoint[
                "epoch"
            ]
        ),
        "best_val_auprc": float(
            checkpoint[
                "val_auprc"
            ]
        ),
        "best_val_auroc": float(
            checkpoint.get(
                "val_auroc",
                val_final[
                    "auroc"
                ],
            )
        ),
        "threshold_selected_on_val_f1": (
            threshold
        ),
        "model_variant": (
            "respire_transfuse"
        ),
        "training_mode": (
            "scratch_without_unimodal_checkpoints"
        ),
        "raw": {
            "val": val_metrics,
            "test": test_metrics,
        },
    }

    final_metrics = (
        remove_unsaved_metrics(
            final_metrics
        )
    )

    with open(
        out_dir
        / "metrics.json",
        "w",
    ) as file:
        json.dump(
            json_safe(
                final_metrics
            ),
            file,
            indent=2,
        )

    save_multimodal_predictions(
        val_final,
        out_dir
        / "val_predictions.csv",
        threshold=threshold,
    )

    save_multimodal_predictions(
        test_final,
        out_dir
        / "test_predictions.csv",
        threshold=threshold,
    )

    print(
        "\nfinal RespireTransFuse validation:"
    )

    print(
        f"AUROC="
        f"{val_metrics['auroc']:.5f} | "
        f"AUPRC="
        f"{val_metrics['auprc']:.5f}"
    )

    print(
        "\nfinal RespireTransFuse test:"
    )

    print(
        f"AUROC="
        f"{test_metrics['auroc']:.5f} | "
        f"AUPRC="
        f"{test_metrics['auprc']:.5f}"
    )

    print(
        "\nbest epoch:",
        int(
            checkpoint[
                "epoch"
            ]
        ),
    )

    print(
        "saved:",
        out_dir,
    )


if __name__ == "__main__":
    main()
