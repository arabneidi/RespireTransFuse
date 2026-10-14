#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "scripts/train/train_respire_ablation.py"
OUTPUTS = ROOT / "outputs"

VARIANTS = {
    "concat_only": "ablation_a0_concat_only",
    "image_to_ehr": "ablation_a1_image_to_ehr",
    "ehr_to_image": "ablation_a2_ehr_to_image",
    "no_aux": "ablation_a3_no_aux",
}
IDS = {
    "concat_only": "A0",
    "image_to_ehr": "A1",
    "ehr_to_image": "A2",
    "no_aux": "A3",
}
REQUIRED = (
    "config_used.yaml",
    "metrics.json",
    "val_predictions.csv",
    "test_predictions.csv",
    "models/best_by_auprc.pt",
)

def cli():
    p = argparse.ArgumentParser(
        description="Run matched RespireTransFuse ablations."
    )
    p.add_argument("--variants", nargs="+", choices=tuple(VARIANTS),
                   default=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42])
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--dry-run", "--dry_run", action="store_true", dest="dry_run")
    p.add_argument("--archive-incomplete", action="store_true")

    for key in ("epochs", "batch_size", "num_workers", "warmup_epochs", "debug_n"):
        p.add_argument("--" + key, type=int, default=None)
    for key in (
        "fusion_dropout", "lr_image_head", "lr_ehr", "lr_fusion",
        "weight_decay_image", "weight_decay_ehr", "weight_decay_fusion",
    ):
        p.add_argument("--" + key, type=float, default=None)
    p.add_argument("--optimizer", choices=("adam", "adamw"), default=None)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= {40, 41, 42}:
        p.error("Seeds must be unique and drawn from 40, 41, and 42")
    if len(set(args.variants)) != len(args.variants):
        p.error("Each variant must appear only once")
    return args

def count_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))

def completed(folder, variant, seed):

    if not all((folder / name).is_file() for name in REQUIRED):
        return False
    try:
        with (folder / "config_used.yaml").open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        with (folder / "metrics.json").open(encoding="utf-8") as f:
            metrics = json.load(f)
        if not isinstance(cfg, dict) or not isinstance(metrics, dict):
            return False
        if cfg.get("ablation", {}).get("variant") != variant:
            return False
        if int(cfg.get("training", {}).get("seed", -1)) != seed:
            return False

        losses = cfg.get("loss", {})
        if variant == "no_aux":
            if float(losses.get("ehr_aux_weight", -1)) != 0.0:
                return False
            if float(losses.get("image_aux_weight", -1)) != 0.0:
                return False
        if count_rows(folder / "val_predictions.csv") != 1389:
            return False
        if count_rows(folder / "test_predictions.csv") != 1429:
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, csv.Error):
        return False

def output_folder(variant, seed, dry_run):
    suffix = "_dryrun" if dry_run else ""
    return OUTPUTS / f"{VARIANTS[variant]}{seed}{suffix}"

def config_for_seed(seed):
    cfg = OUTPUTS / f"respire_transfuse{seed}" / "config_used.yaml"
    if not cfg.is_file():
        raise FileNotFoundError(
            f"Missing full model's saved configuration: {cfg}"
        )
    return cfg

def assemble_command(variant, seed, args):
    out = output_folder(variant, seed, args.dry_run)
    cmd = [
        sys.executable, "-u", str(TRAINER),
        "--ablation", variant,
        "--seed", str(seed),
        "--config", str(config_for_seed(seed)),
        "--save_dir", str(out),
    ]
    if args.dry_run:
        cmd.append("--dry_run")
    for key in (
        "epochs", "batch_size", "num_workers", "warmup_epochs", "debug_n",
        "fusion_dropout", "lr_image_head", "lr_ehr", "lr_fusion",
        "weight_decay_image", "weight_decay_ehr", "weight_decay_fusion",
        "optimizer",
    ):
        value = getattr(args, key)
        if value is not None:
            cmd.extend(["--" + key, str(value)])
    return cmd, out

def archive(folder):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = folder.with_name(folder.name + "_incomplete_backup_" + stamp)
    i = 2
    while dest.exists():
        dest = folder.with_name(
            folder.name + "_incomplete_backup_" + stamp + f"_{i}"
        )
        i += 1
    folder.rename(dest)
    print(f"Archived incomplete directory to: {dest}", flush=True)

def write_metadata(folder, variant):

    path = folder / "metrics.json"
    with path.open(encoding="utf-8") as f:
        metrics = json.load(f)
    desired = {
        "ablation_id": IDS[variant],
        "ablation_variant": variant,
        "model_variant": VARIANTS[variant],
    }
    if any(metrics.get(k) != v for k, v in desired.items()):
        metrics.update(desired)
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

def main():
    args = cli()
    if not TRAINER.is_file():
        raise FileNotFoundError(
            f"Missing training script: {TRAINER}"
        )
    jobs = []
    for variant in args.variants:
        for seed in args.seeds:
            cmd, folder = assemble_command(variant, seed, args)
            jobs.append((variant, seed, cmd, folder))

    print(f"Planned jobs: {len(jobs)} | sequential", flush=True)
    for variant, seed, cmd, folder in jobs:
        status = "COMPLETE" if completed(folder, variant, seed) else (
            "EXISTS (needs review)" if folder.exists() else "NEW"
        )
        print(f"{IDS[variant]} seed {seed}: {status} -> {folder.name}", flush=True)
        if args.plan_only:
            print("  " + " ".join(cmd), flush=True)
    if args.plan_only:
        return

    reports = OUTPUTS / "ablation_launcher_reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "latest_manifest.json"
    records = []

    for num, (variant, seed, cmd, folder) in enumerate(jobs, 1):
        print(f"\n{'=' * 64}\n[{num}/{len(jobs)}] {IDS[variant]} "
              f"{variant} seed {seed}\n{'=' * 64}", flush=True)
        status, error = "failed", ""
        log = reports / f"{folder.name}.log"
        try:
            if folder.exists():
                if not args.dry_run and completed(folder, variant, seed):
                    write_metadata(folder, variant)
                    print("SKIPPED: completed output.", flush=True)
                    status = "skipped_completed"
                    continue
                if not args.archive_incomplete:
                    raise RuntimeError(
                        f"Output directory is incomplete: {folder}\nInspect it or use "
                        "--archive-incomplete to rename it before rerunning."
                    )
                archive(folder)

            print("COMMAND: " + " ".join(cmd), flush=True)

            with log.open("w", encoding="utf-8") as f:
                with subprocess.Popen(
                    cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                ) as proc:
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        f.write(line)
                    exit_code = proc.wait()
            if exit_code != 0:
                raise RuntimeError(f"Training exited with code {exit_code}; log: {log}")

            if not args.dry_run:
                if not completed(folder, variant, seed):
                    raise RuntimeError(
                        f"Training exited normally, but expected files, sample "
                        f"counts, or ablation identity failed validation: {folder}"
                    )
                write_metadata(folder, variant)
            status = "dry_run_passed" if args.dry_run else "completed"
            print(f"OK: {folder}", flush=True)
        except Exception as exc:
            error = str(exc)
            print(f"STOPPED: {error}", flush=True)
            raise
        finally:
            records.append({
                "ablation_id": IDS[variant], "variant": variant, "seed": seed,
                "status": status, "directory": str(folder),
                "log": str(log) if status != "skipped_completed" else None,
                "error": error,
            })
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

    print("\nAll selected jobs finished successfully or were skipped.", flush=True)
    print(f"Manifest: {report_path}", flush=True)
    print("Full-model runs: outputs/respire_transfuse{40,41,42}.", flush=True)

if __name__ == "__main__":
    main()
