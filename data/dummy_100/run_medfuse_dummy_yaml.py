"""Launch one MedFuse smoke-test configuration from the shared YAML file.

This adapter reads the requested Uni-CXR, Uni-EHR, or multimodal LSTM run,
translates its YAML settings into arguments accepted by ``train_medfuse.py``,
and executes that trainer from the repository root. Environment variables and
relative paths are preserved so the same command works on Windows, macOS, and
Linux through the Python dummy-test launcher.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def build_cli(script, args_dict):
    cmd = [sys.executable, "-W", "ignore", "-u", script]

    for key, value in args_dict.items():
        flag = "--" + str(key)

        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue

        if value is None:
            continue

        cmd.extend([flag, str(value)])

    return cmd


def keep_line(line):
    s = line.strip()

    if not s:
        return False

    drop = [
        "UserWarning:",
        "warnings.warn",
        "self.reduction:",
        "super().__init__",
        "The parameter 'pretrained'",
        "Arguments other than a weight enum",
        "size_average and reduce args",
        "trainable_model_params:",
        "LSTM(",
        "(layer0):",
        "(layer1):",
        "(do):",
        "(dense_layer):",
        "respiratory_deterioration",
        "validation loss:",
        "starting train epoch",
        "starting val epoch",
        "saving last checkpoint",
        "saving best checkpoint",
        "history saved:",
        "validating ...",
        "Loaded best checkpoint for final val/test:",
    ]

    if any(x in line for x in drop):
        return False

    if s in [")", "BCELoss()"]:
        return False

    if s.startswith("Original MedFuse training"):
        return True

    if s.startswith("running for fusion_type"):
        return True

    if s.startswith("epoch "):
        return True

    if " val epoch " in line and "val_auroc=" in line:
        return True

    if " test epoch " in line and "val_auroc=" in line:
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--show_full_log_on_error", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg.get("project_root", ".")).resolve()
    run_cfg = cfg["runs"][args.run]

    cmd = build_cli(run_cfg["script"], run_cfg.get("args", {}))

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("=" * 100)
    print(f"MedFuse dummy run: {args.run}")
    print("=" * 100)

    for line in proc.stdout.splitlines():
        if keep_line(line):
            print(line)

    if proc.returncode != 0:
        print()
        print(f"MedFuse dummy run failed with return code {proc.returncode}.")
        if args.show_full_log_on_error:
            print(proc.stdout)
        else:
            print("Rerun with --show_full_log_on_error to see the full raw log.")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
