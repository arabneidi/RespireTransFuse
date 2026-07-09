#!/usr/bin/env python3
"""Run the full preprocessing shell pipeline from a local checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run run_preprocessing_before_training.sh with BASE set to the repository root."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Project root to pass as BASE. Defaults to this repository root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and BASE value without running it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    base = (args.base or repo_root).expanduser().resolve()
    script = repo_root / "run_preprocessing_before_training.sh"

    if not script.exists():
        print(f"Missing preprocessing script: {script}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["BASE"] = str(base)

    command = ["bash", str(script)]
    print("BASE:", env["BASE"])
    print("Running:", " ".join(command))

    if args.dry_run:
        return 0

    try:
        completed = subprocess.run(command, cwd=repo_root, env=env, check=False)
    except FileNotFoundError:
        print("Could not find bash. On Windows, run this from Git Bash or WSL.", file=sys.stderr)
        return 1

    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
