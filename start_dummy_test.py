#!/usr/bin/env python3
"""Run the seven-model dummy-data smoke-test suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DUMMY_COHORT = "data/dummy_100/cohort_dummy_100.csv"
DUMMY_EHR = "data/dummy_100/ehr_dummy_100.npz"

DUMMY_PATHS_CONFIG = (
    "data/dummy_100/configs/paths_dummy_100.yaml"
)

MEDFUSE_CONFIG = (
    "data/dummy_100/configs/medfuse_dummy_2e.yaml"
)

DUMMY_REQUIREMENTS_SCRIPT = (
    "data/dummy_100/check_dummy_requirements.py"
)

MEDFUSE_RUNNER = (
    "data/dummy_100/run_medfuse_dummy_yaml.py"
)

MEDFUSE_CXR_CHECKPOINT = (
    "outputs/dummy_100/medfuse_cxr_2e/"
    "best_checkpoint.pth.tar"
)

MEDFUSE_EHR_CHECKPOINT = (
    "outputs/dummy_100/medfuse_ehr_2e/"
    "best_checkpoint.pth.tar"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two-epoch smoke tests for the seven "
            "evaluated models using the dummy 100 dataset."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )

    return parser.parse_args()


def model_commands() -> list[tuple[str, list[str]]]:
    python = sys.executable

    return [
        (
            "EHR-Only Transformer",
            [
                python,
                "-u",
                "scripts/train/train_ehr.py",
                "--paths",
                DUMMY_PATHS_CONFIG,
                "--config",
                "configs/experiments/ehr_only.yaml",
                "--save_dir",
                "outputs/dummy_100/ehr_only_2e",
                "--epochs",
                "2",
            ],
        ),
        (
            "Image-Only CNN",
            [
                python,
                "-u",
                "scripts/train/train_image.py",
                "--paths",
                DUMMY_PATHS_CONFIG,
                "--config",
                "configs/experiments/image_only.yaml",
                "--save_dir",
                "outputs/dummy_100/image_only_2e",
                "--epochs",
                "2",
            ],
        ),
        (
            "Early Fusion",
            [
                python,
                "-u",
                "scripts/train/train_early_fusion.py",
                "--paths",
                DUMMY_PATHS_CONFIG,
                "--config",
                "configs/experiments/early_fusion.yaml",
                "--save_dir",
                "outputs/dummy_100/early_fusion_2e",
                "--epochs",
                "2",
            ],
        ),
        (
            "RespireTransFuse",
            [
                python,
                "-u",
                "scripts/train/train_respire_transfuse.py",
                "--paths",
                DUMMY_PATHS_CONFIG,
                "--config",
                "configs/experiments/respire_transfuse.yaml",
                "--cohort_csv",
                DUMMY_COHORT,
                "--ehr_npz",
                DUMMY_EHR,
                "--save_dir",
                "outputs/dummy_100/respire_transfuse_2e",
                "--warmup_epochs",
                "0",
                "--epochs",
                "2",
            ],
        ),
        (
            "MedFuse Uni-CXR",
            [
                python,
                "-u",
                MEDFUSE_RUNNER,
                "--config",
                MEDFUSE_CONFIG,
                "--run",
                "uni_cxr",
            ],
        ),
        (
            "MedFuse Uni-EHR",
            [
                python,
                "-u",
                MEDFUSE_RUNNER,
                "--config",
                MEDFUSE_CONFIG,
                "--run",
                "uni_ehr",
            ],
        ),
        (
            "MedFuse Multimodal LSTM",
            [
                python,
                "-u",
                MEDFUSE_RUNNER,
                "--config",
                MEDFUSE_CONFIG,
                "--run",
                "multimodal_lstm",
            ],
        ),
    ]


def validate_required_files(repo_root: Path) -> None:
    required_files = [
        DUMMY_COHORT,
        DUMMY_EHR,
        DUMMY_PATHS_CONFIG,
        MEDFUSE_CONFIG,
        DUMMY_REQUIREMENTS_SCRIPT,
        MEDFUSE_RUNNER,
        "scripts/train/train_ehr.py",
        "scripts/train/train_image.py",
        "scripts/train/train_early_fusion.py",
        "scripts/train/train_respire_transfuse.py",
        "scripts/train/train_medfuse.py",
    ]

    missing = [
        relative_path
        for relative_path in required_files
        if not (repo_root / relative_path).exists()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Required files are missing:\n"
            f"{formatted}"
        )


def require_checkpoint(
    repo_root: Path,
    relative_path: str,
) -> None:
    checkpoint = repo_root / relative_path

    if not checkpoint.exists():
        raise FileNotFoundError(
            "Expected checkpoint was not created:\n"
            f"{checkpoint}"
        )

    print("Checkpoint verified:", checkpoint)


def run_command(
    command: list[str],
    repo_root: Path,
) -> int:
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
    )

    return int(completed.returncode)


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent

    commands = model_commands()

    print("Repository:", repo_root)
    print("Python:", sys.executable)
    print("Models:", len(commands))
    print()

    if args.dry_run:
        preflight_command = [
            sys.executable,
            "-u",
            DUMMY_REQUIREMENTS_SCRIPT,
        ]

        print(
            "[Preflight]",
            " ".join(preflight_command),
        )

        for index, (name, command) in enumerate(
            commands,
            start=1,
        ):
            print(
                f"[{index}/{len(commands)}] {name}"
            )
            print(" ".join(command))

        return 0

    try:
        validate_required_files(repo_root)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    preflight_command = [
        sys.executable,
        "-u",
        DUMMY_REQUIREMENTS_SCRIPT,
    ]

    print("=" * 100)
    print("Dummy-data requirements check")
    print("=" * 100)

    return_code = run_command(
        preflight_command,
        repo_root,
    )

    if return_code != 0:
        print(
            (
                "Dummy-data requirements check failed "
                f"with exit code {return_code}."
            ),
            file=sys.stderr,
        )
        return return_code

    for index, (name, command) in enumerate(
        commands,
        start=1,
    ):
        print()
        print("=" * 100)
        print(f"{index}/{len(commands)} {name}")
        print("=" * 100)
        print(" ".join(command))
        print()

        return_code = run_command(
            command,
            repo_root,
        )

        if return_code != 0:
            print(
                (
                    f"{name} failed with exit code "
                    f"{return_code}."
                ),
                file=sys.stderr,
            )
            return return_code

        try:
            if name == "MedFuse Uni-CXR":
                require_checkpoint(
                    repo_root,
                    MEDFUSE_CXR_CHECKPOINT,
                )

            elif name == "MedFuse Uni-EHR":
                require_checkpoint(
                    repo_root,
                    MEDFUSE_EHR_CHECKPOINT,
                )

        except FileNotFoundError as error:
            print(error, file=sys.stderr)
            return 1

    print()
    print("=" * 100)
    print("Seven-model dummy-data smoke test completed successfully.")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
