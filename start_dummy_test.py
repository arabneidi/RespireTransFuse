#!/usr/bin/env python3
"""Run the dummy 100 smoke-test training suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data/dummy_100/run_2epoch_6_models.sh from the repository root."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    script = repo_root / "data" / "dummy_100" / "run_2epoch_6_models.sh"

    if not script.exists():
        print(f"Missing dummy test script: {script}", file=sys.stderr)
        return 1

    command = ["bash", str(script)]
    print("Running:", " ".join(command))

    if args.dry_run:
        return 0

    try:
        completed = subprocess.run(command, cwd=repo_root, check=False)
    except FileNotFoundError:
        print("Could not find bash. On Windows, run this from Git Bash or WSL.", file=sys.stderr)
        return 1

    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
