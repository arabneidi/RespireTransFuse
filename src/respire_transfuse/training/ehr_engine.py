from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from respire_transfuse.training.ehr_losses import compute_ehr_loss
from respire_transfuse.training.metrics import sigmoid_np, summarize_probabilities


def autocast_context(device, use_amp):
    if bool(use_amp) and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


def lr_factor_for_epoch(epoch, total_epochs, warmup_epochs, min_lr_factor):
    epoch = int(epoch)
    total_epochs = int(total_epochs)
    warmup_epochs = int(warmup_epochs)
    min_lr_factor = float(min_lr_factor)

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        alpha = epoch / max(warmup_epochs, 1)
        return min_lr_factor + alpha * (1.0 - min_lr_factor)

    if total_epochs <= warmup_epochs:
        return 1.0

    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))

    return min_lr_factor + cosine * (1.0 - min_lr_factor)


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def batch_to_device(batch, device):
    ehr_x = batch["ehr_x"].to(device, non_blocking=True)
    ehr_m = batch["ehr_m"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True).float().view(-1)
    sample_ids = list(batch["sample_id"])
    return ehr_x, ehr_m, labels, sample_ids


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    loss_cfg,
    train_cfg,
    device,
    use_amp,
    epoch,
):
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_n = 0

    all_labels = []
    all_logits = []

    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp) and device.type == "cuda")

    pbar = tqdm(loader, desc=f"TRAIN epoch {epoch}", leave=False)

    for batch in pbar:
        ehr_x, ehr_m, labels, _sample_ids = batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            out = model(ehr_x, ehr_m)
            logits = out["logit"]

            losses = compute_ehr_loss(
                logits=logits,
                labels=labels,
                criterion=criterion,
                loss_cfg=loss_cfg,
                train=True,
            )

            loss = losses["loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {float(loss.detach().cpu())}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_clip = float(train_cfg.get("grad_clip", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.numel()

        total_loss += float(loss.detach().cpu()) * batch_size
        total_bce += float(losses["bce"].detach().cpu()) * batch_size
        total_n += batch_size

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_logits.extend(logits.detach().float().cpu().numpy().tolist())

        pbar.set_postfix(
            loss=f"{float(loss.detach().cpu()):.4f}",
            bce=f"{float(losses['bce'].detach().cpu()):.4f}",
        )

    avg_loss = total_loss / max(total_n, 1)
    avg_bce = total_bce / max(total_n, 1)

    probs = sigmoid_np(all_logits)
    metrics = summarize_probabilities(all_labels, probs)

    return {
        "loss": float(avg_loss),
        "bce": float(avg_bce),
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "labels": all_labels,
        "logits": all_logits,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    loss_cfg,
    device,
    use_amp,
    desc,
):
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_n = 0

    all_labels = []
    all_logits = []
    all_sample_ids = []

    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        ehr_x, ehr_m, labels, sample_ids = batch_to_device(batch, device)

        with autocast_context(device, use_amp):
            out = model(ehr_x, ehr_m)
            logits = out["logit"]

            losses = compute_ehr_loss(
                logits=logits,
                labels=labels,
                criterion=criterion,
                loss_cfg=loss_cfg,
                train=False,
            )

            loss = losses["loss"]

        batch_size = labels.numel()

        total_loss += float(loss.detach().cpu()) * batch_size
        total_bce += float(losses["bce"].detach().cpu()) * batch_size
        total_n += batch_size

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_logits.extend(logits.detach().float().cpu().numpy().tolist())
        all_sample_ids.extend(sample_ids)

    avg_loss = total_loss / max(total_n, 1)
    avg_bce = total_bce / max(total_n, 1)

    probs = sigmoid_np(all_logits)
    metrics = summarize_probabilities(all_labels, probs)

    return {
        "loss": float(avg_loss),
        "bce": float(avg_bce),
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "best_f1_threshold": metrics["best_f1_threshold"],
        "best_f1": metrics["best_f1"],
        "labels": all_labels,
        "logits": all_logits,
        "sample_ids": all_sample_ids,
    }


def save_predictions(stats, out_path, threshold=None, temperature=None, bias=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logits = np.asarray(stats["logits"], dtype=np.float64)
    labels = np.asarray(stats["labels"], dtype=np.int64)

    raw_prob = sigmoid_np(logits)

    data = {
        "sample_id": stats["sample_ids"],
        "label": labels,
        "logit": logits,
        "prob": raw_prob,
    }

    if temperature is not None and bias is not None:
        cal_logits = logits / float(temperature) + float(bias)
        cal_prob = sigmoid_np(cal_logits)
        data["calibrated_logit"] = cal_logits
        data["calibrated_prob"] = cal_prob

        if threshold is not None:
            data["pred"] = (cal_prob >= float(threshold)).astype(int)

    elif threshold is not None:
        data["pred"] = (raw_prob >= float(threshold)).astype(int)

    pd.DataFrame(data).to_csv(out_path, index=False)
