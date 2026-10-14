#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
MEDFUSE_CONFIG = ROOT / "configs/experiments/medfuse.yaml"
DEFAULT_SEEDS = (40, 41, 42)

MODEL_CHOICES = (
    "all",
    "ehr_only",
    "image_only",
    "early_fusion",
    "medfuse_uni_cxr",
    "medfuse_uni_ehr",
    "medfuse_multimodal_lstm",
    "respire_transfuse",
)

OUTPUT_DIRS = {
    "ehr_only": "ehr_only{seed}",
    "image_only": "image_only{seed}",
    "early_fusion": "early_fusion{seed}",
    "medfuse_uni_cxr": "medfuse_uni_cxr_{seed}",
    "medfuse_uni_ehr": "medfuse_ehr{seed}",
    "medfuse_multimodal_lstm": "medfuse_multimodal_lstm_{seed}",
    "respire_transfuse": "respire_transfuse{seed}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the configured RespireTransFuse experiments."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="all",
        help="Experiment to run; the default runs all seven in dependency order.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Training seeds; the reported experiments use 40, 41, and 42.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without starting training.",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Each seed may be listed only once.")
    return args


def output_dir(model: str, seed: int) -> str:
    return "outputs/" + OUTPUT_DIRS[model].format(seed=seed)


def yaml_command(run_name: str, model: str, seed: int) -> list[str]:
    with MEDFUSE_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    run = config["runs"][run_name]
    values = dict(run.get("args", {}))
    values["seed"] = seed
    values["save_dir"] = output_dir(model, seed)

    if run_name == "multimodal_lstm":
        values["load_state_ehr"] = (
            f"outputs/medfuse_ehr{seed}/best_checkpoint.pth.tar"
        )

    command = [sys.executable, "-u", run["script"]]
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        elif value is not None:
            command.extend([f"--{key}", str(value)])
    return command


def project_command(script: str, config: str, model: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        script,
        "--paths",
        "configs/paths.yaml",
        "--config",
        config,
        "--seed",
        str(seed),
        "--save_dir",
        output_dir(model, seed),
    ]


def configured_commands(seed: int) -> dict[str, tuple[str, list[str]]]:
    return {
        "ehr_only": (
            "EHR-Only Transformer",
            project_command(
                "scripts/train/train_ehr.py",
                "configs/experiments/ehr_only.yaml",
                "ehr_only",
                seed,
            ),
        ),
        "image_only": (
            "Image-Only CNN",
            project_command(
                "scripts/train/train_image.py",
                "configs/experiments/image_only.yaml",
                "image_only",
                seed,
            ),
        ),
        "early_fusion": (
            "Early Fusion",
            project_command(
                "scripts/train/train_early_fusion.py",
                "configs/experiments/early_fusion.yaml",
                "early_fusion",
                seed,
            ),
        ),
        "medfuse_uni_cxr": (
            "MedFuse Uni-CXR",
            yaml_command("uni_cxr", "medfuse_uni_cxr", seed),
        ),
        "medfuse_uni_ehr": (
            "MedFuse Uni-EHR",
            yaml_command("uni_ehr", "medfuse_uni_ehr", seed),
        ),
        "medfuse_multimodal_lstm": (
            "MedFuse Multimodal LSTM",
            yaml_command(
                "multimodal_lstm",
                "medfuse_multimodal_lstm",
                seed,
            ),
        ),
        "respire_transfuse": (
            "RespireTransFuse",
            project_command(
                "scripts/train/train_respire_transfuse.py",
                "configs/experiments/respire_transfuse.yaml",
                "respire_transfuse",
                seed,
            ),
        ),
    }


def main() -> int:
    args = parse_args()
    selected = MODEL_CHOICES[1:] if args.model == "all" else (args.model,)
    jobs = []
    for seed in args.seeds:
        commands = configured_commands(seed)
        for key in selected:
            name, command = commands[key]
            jobs.append((name, seed, command))

    print("Repository:", ROOT)
    print("Python:", sys.executable)

    for index, (name, seed, command) in enumerate(jobs, start=1):
        print()
        print(f"[{index}/{len(jobs)}] {name}, seed {seed}")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
