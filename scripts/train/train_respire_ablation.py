#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "scripts/train/train_respire_transfuse.py"

VARIANTS = {
    "concat_only": "ablation_a0_concat_only",
    "image_to_ehr": "ablation_a1_image_to_ehr",
    "ehr_to_image": "ablation_a2_ehr_to_image",
    "no_aux": "ablation_a3_no_aux",
}

class OneWayCrossAttentionLayer(nn.Module):

    def __init__(self, cross_layer, direction):
        super().__init__()
        self.direction = direction
        if direction == "image_to_ehr":
            self.active_block = cross_layer.image_queries_ehr
        elif direction == "ehr_to_image":
            self.active_block = cross_layer.ehr_queries_image
        else:
            raise ValueError(direction)

    def forward(
        self, image_tokens, ehr_tokens,
        ehr_padding_mask=None, need_weights=False,
    ):
        if self.direction == "image_to_ehr":
            new_image, weights = self.active_block(
                query_tokens=image_tokens,
                context_tokens=ehr_tokens,
                context_padding_mask=ehr_padding_mask,
                need_weights=need_weights,
            )
            return new_image, ehr_tokens, weights, None

        new_ehr, weights = self.active_block(
            query_tokens=ehr_tokens,
            context_tokens=image_tokens,
            context_padding_mask=None,
            need_weights=need_weights,
        )
        return image_tokens, new_ehr, None, weights

def apply_variant(model, variant):

    if len(model.cross_layers) != 1:
        raise RuntimeError("Expected exactly one cross-attention layer")

    if variant == "concat_only":

        model.cross_layers = nn.ModuleList([])

    elif variant in ("image_to_ehr", "ehr_to_image"):
        cross_layer = model.cross_layers[0]
        model.cross_layers = nn.ModuleList([
            OneWayCrossAttentionLayer(cross_layer, variant)
        ])

    elif variant == "no_aux":

        pass
    else:
        raise ValueError(f"Unknown ablation: {variant}")

    if model.fusion_dim != 48:
        raise RuntimeError(
            f"Expected fusion_dim=48; got {model.fusion_dim}"
        )
    if model.fusion_head[0].normalized_shape != (192,):
        raise RuntimeError("Expected a 192-D fusion head")
    return model

def load_trainer():
    if not TRAINER.is_file():
        raise FileNotFoundError(
            f"Missing trainer: {TRAINER}. "
            "Place this script in scripts/train/."
        )
    spec = importlib.util.spec_from_file_location(
        "respire_training_for_ablation", TRAINER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    ablation_parser = argparse.ArgumentParser(add_help=False)
    ablation_parser.add_argument(
        "--ablation", choices=tuple(VARIANTS), required=True
    )
    ablation_args, trainer_args = ablation_parser.parse_known_args()
    variant = ablation_args.ablation

    meta_parser = argparse.ArgumentParser(add_help=False)
    meta_parser.add_argument("--seed", type=int, default=40)
    meta_parser.add_argument("--save_dir", default=None)
    meta_parser.add_argument("--config", default=None)
    meta_parser.add_argument("--dry_run", action="store_true")
    meta_args, _ = meta_parser.parse_known_args(trainer_args)

    if meta_args.seed not in (40, 41, 42):
        raise ValueError("Use the matching full-model seeds: 40, 41, 42")

    if meta_args.config is None:
        trainer_args += [
            "--config",
            f"outputs/respire_transfuse{meta_args.seed}/config_used.yaml",
        ]

    if meta_args.save_dir is None:
        folder = VARIANTS[variant] + str(meta_args.seed)
        if meta_args.dry_run:
            folder += "_dryrun"
        trainer_args += ["--save_dir", f"outputs/{folder}"]
    else:
        resolved = Path(meta_args.save_dir)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        protected = {
            ROOT / "outputs" / f"respire_transfuse{seed}"
            for seed in (40, 41, 42)
        }
        if resolved.resolve() in {p.resolve() for p in protected}:
            raise ValueError("Refusing to overwrite a full-model run")

    trainer = load_trainer()
    model_builder = trainer.build_respire_transfuse_from_config
    config_loader = trainer.load_config

    def ablation_builder(cfg, n_ehr_features):
        return apply_variant(
            model_builder(cfg, n_ehr_features), variant
        )

    def ablation_load_config(paths_yaml, experiment_yaml):
        cfg = config_loader(paths_yaml, experiment_yaml)
        cfg.setdefault("ablation", {})["variant"] = variant
        cfg["ablation"]["baseline"] = "respire_transfuse"
        if variant == "no_aux":
            cfg["loss"]["image_aux_weight"] = 0.0
            cfg["loss"]["ehr_aux_weight"] = 0.0
        return cfg

    trainer.build_respire_transfuse_from_config = ablation_builder
    trainer.load_config = ablation_load_config

    print(f"\nABLATION: {variant}", flush=True)
    print("Trainer:", TRAINER, flush=True)
    print("Arguments:", trainer_args, flush=True)

    sys.argv = [str(TRAINER)] + trainer_args
    trainer.main()

    meta, _ = meta_parser.parse_known_args(trainer_args)
    out_dir = Path(meta.save_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open() as f:
            metrics = json.load(f)
        metrics["model_variant"] = VARIANTS[variant]
        metrics["ablation_variant"] = variant
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
