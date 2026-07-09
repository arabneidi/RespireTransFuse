#!/usr/bin/env python3
"""Run the dummy 100 smoke-test training suite.

This file is intentionally pure Python so it can be run from PyCharm,
Windows, macOS, or Linux without requiring bash.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dummy 100 two-epoch model checks from the repository root."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it.",
    )
    return parser.parse_args()


def dummy_commands() -> list[list[str]]:
    python = sys.executable

    return [
        [
            python,
            "-u",
            "scripts/train/train_ehr.py",
            "--paths",
            "data/dummy_100/configs/paths_dummy_100.yaml",
            "--config",
            "configs/experiments/ehr_only_natural_sampling.yaml",
            "--save_dir",
            "outputs/dummy_100/ehr_only_2e",
            "--epochs",
            "2",
        ],
        [
            python,
            "-u",
            "scripts/train/train_image.py",
            "--paths",
            "data/dummy_100/configs/paths_dummy_100.yaml",
            "--config",
            "configs/experiments/image_only.yaml",
            "--save_dir",
            "outputs/dummy_100/image_only_2e",
            "--epochs",
            "2",
        ],
        [
            python,
            "-u",
            "scripts/train/train_respire_transfuse.py",
            "--paths",
            "data/dummy_100/configs/paths_dummy_100.yaml",
            "--config",
            "configs/experiments/respire_transfuse.yaml",
            "--save_dir",
            "outputs/dummy_100/respire_transfuse_2e",
            "--epochs",
            "2",
        ],
        [
            python,
            "-u",
            "scripts/train/train_early_fusion.py",
            "--paths",
            "data/dummy_100/configs/paths_dummy_100.yaml",
            "--config",
            "configs/experiments/early_fusion.yaml",
            "--save_dir",
            "outputs/dummy_100/early_fusion_2e",
            "--epochs",
            "2",
        ],
        [
            python,
            "-u",
            "scripts/train/run_medfuse_dummy_yaml.py",
            "--config",
            "data/dummy_100/configs/medfuse_dummy_2e.yaml",
            "--run",
            "uni_ehr",
        ],
        [
            python,
            "-u",
            "scripts/train/run_medfuse_dummy_yaml.py",
            "--config",
            "data/dummy_100/configs/medfuse_dummy_2e.yaml",
            "--run",
            "lstm",
        ],
    ]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    commands = dummy_commands()

    print("Repository:", repo_root)
    print("Python:", sys.executable)
    print()

    if args.dry_run:
        for index, command in enumerate(commands, start=1):
            print(f"[{index}/{len(commands)}]", " ".join(command))
        return 0

    for index, command in enumerate(commands, start=1):
        print("=" * 100)
        print(f"Running step {index}/{len(commands)}")
        print(" ".join(command))
        print("=" * 100)

        completed = subprocess.run(command, cwd=repo_root, check=False)

        if completed.returncode != 0:
            print(
                f"Step {index} failed with exit code {completed.returncode}.",
                file=sys.stderr,
            )
            return int(completed.returncode)

    print()
    print("Dummy 100 smoke test finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
