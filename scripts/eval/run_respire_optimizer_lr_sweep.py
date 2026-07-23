#!/usr/bin/env python3
"""Run the RespireTransFuse optimizer and fusion-learning-rate sweep."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TRAIN_SCRIPT = (
    ROOT
    / "scripts"
    / "train"
    / "train_respire_transfuse.py"
)

SWEEP_ROOT = (
    ROOT
    / "outputs"
    / "respire_transfuse_optimizer_lr_sweep"
)


EXPERIMENTS = [
    {
        "name": "adamw_lr_3p5e_5",
        "optimizer": "adamw",
        "lr_fusion": 3.5e-5,
    },
    {
        "name": "adamw_lr_5e_5",
        "optimizer": "adamw",
        "lr_fusion": 5.0e-5,
    },
    {
        "name": "adamw_lr_7e_5",
        "optimizer": "adamw",
        "lr_fusion": 7.0e-5,
    },
    {
        "name": "adamw_lr_1p4e_4",
        "optimizer": "adamw",
        "lr_fusion": 1.4e-4,
    },
    {
        "name": "adam_lr_3p5e_5",
        "optimizer": "adam",
        "lr_fusion": 3.5e-5,
    },
    {
        "name": "adam_lr_5e_5",
        "optimizer": "adam",
        "lr_fusion": 5.0e-5,
    },
    {
        "name": "adam_lr_7e_5",
        "optimizer": "adam",
        "lr_fusion": 7.0e-5,
    },
    {
        "name": "adam_lr_1p4e_4",
        "optimizer": "adam",
        "lr_fusion": 1.4e-4,
    },
]


SWEEP_CONTROLLED_OPTIONS = {
    "--optimizer",
    "--lr_fusion",
    "--save_dir",
    "--warmup_epochs",
    "--epochs",
    "--seed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run eight RespireTransFuse experiments comparing AdamW "
            "and Adam across four fusion learning rates."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Epochs per run. Kept at 20 to match the existing sweep.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rerun existing experiment folders.",
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue to later experiments after a failed run.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without starting training.",
    )

    args, train_args = parser.parse_known_args()
    args.train_args = train_args

    return args


def validate_passthrough_args(
    train_args: list[str],
) -> list[str]:
    supplied_options = {
        token.split("=", 1)[0]
        for token in train_args
        if token.startswith("--")
    }

    conflicts = sorted(
        supplied_options
        & SWEEP_CONTROLLED_OPTIONS
    )

    if conflicts:
        raise RuntimeError(
            "These arguments are controlled by the sweep: "
            + ", ".join(conflicts)
        )

    return list(train_args)


def validate_experiments() -> None:
    expected = {
        ("adamw", 3.5e-5),
        ("adamw", 5.0e-5),
        ("adamw", 7.0e-5),
        ("adamw", 1.4e-4),
        ("adam", 3.5e-5),
        ("adam", 5.0e-5),
        ("adam", 7.0e-5),
        ("adam", 1.4e-4),
    }

    actual = {
        (
            experiment["optimizer"],
            float(experiment["lr_fusion"]),
        )
        for experiment in EXPERIMENTS
    }

    names = [
        experiment["name"]
        for experiment in EXPERIMENTS
    ]

    if len(EXPERIMENTS) != 8:
        raise RuntimeError(
            "The sweep must contain exactly eight experiments."
        )

    if len(names) != len(set(names)):
        raise RuntimeError(
            "Experiment names must be unique."
        )

    if actual != expected:
        raise RuntimeError(
            "The sweep does not match the required two-optimizer "
            "by four-learning-rate design."
        )


def run_quiet(
    command: list[str],
    cwd: Path,
    log_path: Path,
) -> None:
    with log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_file:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
        )


def metric_value(
    mapping,
    names: list[str],
):
    if not isinstance(mapping, dict):
        return None

    for name in names:
        value = mapping.get(name)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def load_result(
    experiment: dict,
    output_dir: Path,
    status: str,
    duration_minutes: float,
) -> dict:
    row = {
        "experiment": experiment["name"],
        "optimizer": experiment["optimizer"],
        "lr_fusion": float(
            experiment["lr_fusion"]
        ),
        "status": status,
        "duration_minutes": float(
            duration_minutes
        ),
        "output_dir": str(output_dir),
        "best_epoch": None,
        "best_val_auroc": None,
        "best_val_auprc": None,
        "test_auroc": None,
        "test_auprc": None,
    }

    metrics_path = (
        output_dir
        / "metrics.json"
    )

    if metrics_path.exists():
        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metrics = json.load(file)

        raw = metrics.get(
            "raw",
            {},
        )

        validation_metrics = raw.get(
            "val",
            raw.get(
                "validation",
                {},
            ),
        )

        test_metrics = raw.get(
            "test",
            {},
        )

        row["best_epoch"] = metrics.get(
            "best_epoch"
        )

        row["best_val_auroc"] = metric_value(
            metrics,
            [
                "best_val_auroc",
                "val_auroc",
            ],
        )

        row["best_val_auprc"] = metric_value(
            metrics,
            [
                "best_val_auprc",
                "val_auprc",
            ],
        )

        if row["best_val_auroc"] is None:
            row["best_val_auroc"] = metric_value(
                validation_metrics,
                [
                    "auroc",
                    "val_auroc",
                ],
            )

        if row["best_val_auprc"] is None:
            row["best_val_auprc"] = metric_value(
                validation_metrics,
                [
                    "auprc",
                    "val_auprc",
                ],
            )

        row["test_auroc"] = metric_value(
            test_metrics,
            [
                "auroc",
                "test_auroc",
            ],
        )

        row["test_auprc"] = metric_value(
            test_metrics,
            [
                "auprc",
                "test_auprc",
            ],
        )

    history_path = (
        output_dir
        / "history.csv"
    )

    if history_path.exists():
        history = pd.read_csv(
            history_path
        )

        auprc_column = next(
            (
                column
                for column in [
                    "val_auprc",
                    "validation_auprc",
                ]
                if column in history.columns
            ),
            None,
        )

        auroc_column = next(
            (
                column
                for column in [
                    "val_auroc",
                    "validation_auroc",
                ]
                if column in history.columns
            ),
            None,
        )

        if auprc_column is not None:
            auprc_values = pd.to_numeric(
                history[auprc_column],
                errors="coerce",
            )

            if auprc_values.notna().any():
                best_index = (
                    auprc_values.idxmax()
                )

                row["best_val_auprc"] = float(
                    auprc_values.loc[
                        best_index
                    ]
                )

                if auroc_column is not None:
                    auroc_value = pd.to_numeric(
                        history.loc[
                            best_index,
                            auroc_column,
                        ],
                        errors="coerce",
                    )

                    if pd.notna(auroc_value):
                        row["best_val_auroc"] = float(
                            auroc_value
                        )

                if "epoch" in history.columns:
                    epoch_value = pd.to_numeric(
                        history.loc[
                            best_index,
                            "epoch",
                        ],
                        errors="coerce",
                    )

                    if pd.notna(epoch_value):
                        row["best_epoch"] = int(
                            epoch_value
                        )

    return row


def format_metric(value) -> str:
    if value is None:
        return "NA"

    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        return "NA"

    return f"{float(value):.5f}"


def save_summary(
    rows: list[dict],
) -> Path:
    summary = pd.DataFrame(rows)

    if (
        not summary.empty
        and "best_val_auprc"
        in summary.columns
    ):
        summary = summary.sort_values(
            [
                "best_val_auprc",
                "best_val_auroc",
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        ).reset_index(drop=True)

    summary_path = (
        SWEEP_ROOT
        / "sweep_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    return summary_path


def output_complete(
    output_dir: Path,
) -> bool:
    return (
        output_dir
        / "metrics.json"
    ).exists()


def build_command(
    experiment: dict,
    output_dir: Path,
    epochs: int,
    seed: int,
    extra_train_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--epochs",
        str(epochs),
        "--optimizer",
        experiment["optimizer"],
        "--lr_fusion",
        str(experiment["lr_fusion"]),
        "--warmup_epochs",
        "0",
        "--seed",
        str(seed),
        "--save_dir",
        str(output_dir),
    ]

    command.extend(
        extra_train_args
    )

    return command


def main() -> int:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    extra_train_args = (
        validate_passthrough_args(
            args.train_args
        )
    )

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            TRAIN_SCRIPT
        )

    validate_experiments()

    SWEEP_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Repository:", ROOT)
    print("Trainer:", TRAIN_SCRIPT)
    print("Output:", SWEEP_ROOT)
    print("Runs:", len(EXPERIMENTS))
    print("Epochs per run:", args.epochs)
    print("Seed:", args.seed)
    print("Warmup epochs: 0")
    print("Selection metric: validation AUPRC")
    print()

    rows = []

    for index, experiment in enumerate(
        EXPERIMENTS,
        start=1,
    ):
        output_dir = (
            SWEEP_ROOT
            / experiment["name"]
        )

        command = build_command(
            experiment=experiment,
            output_dir=output_dir,
            epochs=args.epochs,
            seed=args.seed,
            extra_train_args=extra_train_args,
        )

        if args.dry_run:
            print(
                f"[{index}/{len(EXPERIMENTS)}] "
                f"{experiment['name']}"
            )
            print(" ".join(command))
            print()
            continue

        if (
            output_complete(output_dir)
            and not args.force
        ):
            row = load_result(
                experiment=experiment,
                output_dir=output_dir,
                status="already_completed",
                duration_minutes=0.0,
            )

            rows.append(row)
            save_summary(rows)

            print(
                f"[{index}/{len(EXPERIMENTS)}] "
                f"{experiment['name']} already completed | "
                f"val AUPRC="
                f"{format_metric(row.get('best_val_auprc'))} | "
                f"val AUROC="
                f"{format_metric(row.get('best_val_auroc'))}",
                flush=True,
            )

            continue

        if output_dir.exists():
            shutil.rmtree(
                output_dir
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        log_path = (
            output_dir
            / "console.log"
        )

        start_time = time.time()

        try:
            run_quiet(
                command=command,
                cwd=ROOT,
                log_path=log_path,
            )

            duration_minutes = (
                time.time()
                - start_time
            ) / 60.0

            row = load_result(
                experiment=experiment,
                output_dir=output_dir,
                status="completed",
                duration_minutes=duration_minutes,
            )

            rows.append(row)
            save_summary(rows)

            print(
                f"[{index}/{len(EXPERIMENTS)}] "
                f"{experiment['name']} finished | "
                f"{duration_minutes:.1f} min | "
                f"val AUPRC="
                f"{format_metric(row.get('best_val_auprc'))} | "
                f"val AUROC="
                f"{format_metric(row.get('best_val_auroc'))}",
                flush=True,
            )

        except Exception:
            duration_minutes = (
                time.time()
                - start_time
            ) / 60.0

            row = load_result(
                experiment=experiment,
                output_dir=output_dir,
                status="failed",
                duration_minutes=duration_minutes,
            )

            rows.append(row)
            save_summary(rows)

            print(
                f"[{index}/{len(EXPERIMENTS)}] "
                f"{experiment['name']} failed | "
                f"log={log_path}",
                file=sys.stderr,
                flush=True,
            )

            if not args.continue_on_error:
                raise

    if args.dry_run:
        print(
            "Dry run completed. No training was started."
        )
        return 0

    summary_path = save_summary(rows)

    print()
    print("All eight runs processed.")
    print("Summary:", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )