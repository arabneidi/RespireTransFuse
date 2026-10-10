#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
MEDFUSE_CONFIG = ROOT / "configs/experiments/medfuse.yaml"

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
        "--dry-run",
        action="store_true",
        help="Print the commands without starting training.",
    )
    return parser.parse_args()


def yaml_command(run_name: str) -> list[str]:
    with MEDFUSE_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    run = config["runs"][run_name]
    command = [sys.executable, "-u", run["script"]]

    for key, value in run.get("args", {}).items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        elif value is not None:
            command.extend([f"--{key}", str(value)])

    return command


def configured_commands() -> dict[str, tuple[str, list[str]]]:
    python = sys.executable
    paths = "configs/paths.yaml"

    return {
        "ehr_only": (
            "EHR-Only Transformer",
            [
                python,
                "-u",
                "scripts/train/train_ehr.py",
                "--paths",
                paths,
                "--config",
                "configs/experiments/ehr_only.yaml",
            ],
        ),
        "image_only": (
            "Image-Only CNN",
            [
                python,
                "-u",
                "scripts/train/train_image.py",
                "--paths",
                paths,
                "--config",
                "configs/experiments/image_only.yaml",
            ],
        ),
        "early_fusion": (
            "Early Fusion",
            [
                python,
                "-u",
                "scripts/train/train_early_fusion.py",
                "--paths",
                paths,
                "--config",
                "configs/experiments/early_fusion.yaml",
            ],
        ),
        "medfuse_uni_cxr": (
            "MedFuse Uni-CXR",
            yaml_command("uni_cxr"),
        ),
        "medfuse_uni_ehr": (
            "MedFuse Uni-EHR",
            yaml_command("uni_ehr"),
        ),
        "medfuse_multimodal_lstm": (
            "MedFuse Multimodal LSTM",
            yaml_command("multimodal_lstm"),
        ),
        "respire_transfuse": (
            "RespireTransFuse",
            [
                python,
                "-u",
                "scripts/train/train_respire_transfuse.py",
                "--paths",
                paths,
                "--config",
                "configs/experiments/respire_transfuse.yaml",
            ],
        ),
    }


def main() -> int:
    args = parse_args()
    commands = configured_commands()
    selected = list(commands) if args.model == "all" else [args.model]

    print("Repository:", ROOT)
    print("Python:", sys.executable)

    for index, key in enumerate(selected, start=1):
        name, command = commands[key]
        print()
        print(f"[{index}/{len(selected)}] {name}")
        print(" ".join(command))

        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
