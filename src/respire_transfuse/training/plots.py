from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_history(history_csv, out_dir, title_prefix="EHR-only"):
    history_csv = Path(history_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = pd.read_csv(history_csv)

    if len(hist) == 0:
        return

    hist = hist.copy()
    hist["epoch"] = pd.to_numeric(hist["epoch"], errors="coerce")
    hist = hist.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)
    hist["epoch"] = hist["epoch"].astype(int)

    epoch_ticks = hist["epoch"].tolist()

    plots = [
        (
            "loss_curve.png",
            "Evaluation loss",
            [
                ("train_loss", "Train loss"),
                ("val_loss", "Val loss"),
            ],
        ),
        (
            "auroc_curve.png",
            "AUROC",
            [
                ("train_auroc", "Train AUROC"),
                ("val_auroc", "Val AUROC"),
            ],
        ),
        (
            "auprc_curve.png",
            "AUPRC",
            [
                ("train_auprc", "Train AUPRC"),
                ("val_auprc", "Val AUPRC"),
            ],
        ),
        (
            "logloss_curve.png",
            "Log loss",
            [
                ("train_log_loss", "Train log loss"),
                ("val_log_loss", "Val log loss"),
            ],
        ),
    ]

    for filename, ylabel, columns in plots:
        plt.figure(figsize=(10, 5.5))

        for col, label in columns:
            if col in hist.columns:
                y = pd.to_numeric(hist[col], errors="coerce")
                plt.plot(hist["epoch"], y, marker="o", linewidth=2, label=label)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix}: {ylabel}")
        plt.ylim(0.0, 1.0)
        plt.xticks(epoch_ticks)

        if len(epoch_ticks) > 1:
            plt.xlim(min(epoch_ticks), max(epoch_ticks))

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    if "optim_train_loss" in hist.columns:
        plt.figure(figsize=(10, 5.5))
        y = pd.to_numeric(hist["optim_train_loss"], errors="coerce")
        plt.plot(hist["epoch"], y, marker="o", linewidth=2, label="Optimization train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Optimization train loss")
        plt.title(f"{title_prefix}: optimization train loss")
        plt.ylim(0.0, 1.0)
        plt.xticks(epoch_ticks)

        if len(epoch_ticks) > 1:
            plt.xlim(min(epoch_ticks), max(epoch_ticks))

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "optim_train_loss_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
