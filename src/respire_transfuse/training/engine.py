"""Train and evaluate image-only and multimodal models with shared metrics."""

import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
    brier_score_loss,
)
from tqdm.auto import tqdm

from respire_transfuse.models.image_only import update_ema_model


def sigmoid_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.clip(logits, -50, 50)
    return 1.0 / (1.0 + np.exp(-logits))


def safe_auroc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.sum() == 0:
        return float("nan")

    return float(average_precision_score(y_true, y_prob))


def safe_logloss(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)

    return float(log_loss(y_true, y_prob, labels=[0, 1]))


def safe_brier(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    return float(brier_score_loss(y_true, y_prob))


def best_f1_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5, 0.0

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.nanargmax(f1))

    return float(thresholds[idx]), float(f1[idx])


def metrics_at_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)

    pred = (y_prob >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "auroc": safe_auroc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "log_loss": safe_logloss(y_true, y_prob),
        "brier": safe_brier(y_true, y_prob),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(tn / (tn + fp + 1e-12)),
        "npv": float(tn / (tn + fn + 1e-12)),
        "fpr": float(fp / (fp + tn + 1e-12)),
        "fnr": float(fn / (fn + tp + 1e-12)),
    }


def soft_average_precision_loss(logits, labels, tau=0.25):
    logits = logits.float().view(-1)
    labels = labels.float().view(-1)

    pos_mask = labels > 0.5

    if pos_mask.sum() == 0 or (~pos_mask).sum() == 0:
        return logits.new_tensor(0.0)

    s_i = logits.view(-1, 1)
    s_j = logits.view(1, -1)

    compare = torch.sigmoid((s_j - s_i) / float(tau))

    soft_rank = 1.0 + compare.sum(dim=1) - 0.5

    pos_compare = compare * labels.view(1, -1)
    soft_pos_rank = pos_compare.sum(dim=1) + labels * 0.5

    precision_at_i = soft_pos_rank / torch.clamp(soft_rank, min=1.0)
    soft_ap = precision_at_i[pos_mask].mean()

    return 1.0 - soft_ap


def compute_image_loss(logits, labels, criterion, loss_cfg, train_prevalence, train=True):
    logits = logits.float().view(-1)
    labels = labels.float().view(-1)

    bce = criterion(logits, labels)

    if train and float(loss_cfg["ap_loss_weight"]) > 0:
        ap_loss = soft_average_precision_loss(
            logits=logits,
            labels=labels,
            tau=float(loss_cfg["ap_loss_tau"]),
        )
    else:
        ap_loss = logits.new_tensor(0.0)

    logit_l2_loss = logits.pow(2).mean()

    if train and float(loss_cfg["prob_mean_weight"]) > 0:
        if float(loss_cfg["prob_mean_target"]) >= 0:
            prob_target = float(loss_cfg["prob_mean_target"])
        else:
            prob_target = float(train_prevalence)

        prob_mean_loss = (torch.sigmoid(logits).mean() - prob_target) ** 2
    else:
        prob_mean_loss = logits.new_tensor(0.0)

    if train and float(loss_cfg["logit_prior_weight"]) > 0:
        if float(loss_cfg["logit_prior_target"]) > -900:
            prior_logit = float(loss_cfg["logit_prior_target"])
        else:
            p = min(max(float(train_prevalence), 1e-5), 1.0 - 1e-5)
            prior_logit = math.log(p / (1.0 - p))

        logit_prior_loss = (logits.mean() - prior_logit) ** 2
    else:
        logit_prior_loss = logits.new_tensor(0.0)

    loss = (
        float(loss_cfg["bce_weight"]) * bce
        + float(loss_cfg["ap_loss_weight"]) * ap_loss
        + float(loss_cfg["logit_l2"]) * logit_l2_loss
        + float(loss_cfg["prob_mean_weight"]) * prob_mean_loss
        + float(loss_cfg["logit_prior_weight"]) * logit_prior_loss
    )

    return {
        "loss": loss,
        "bce": bce,
        "ap_loss": ap_loss,
        "logit_l2": logit_l2_loss,
        "prob_mean_loss": prob_mean_loss,
        "logit_prior_loss": logit_prior_loss,
    }


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
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_factor + cosine * (1.0 - min_lr_factor)


def set_image_optimizer_lrs(optimizer, lr_head, lr_backbone, lr_factor, backbone_trainable):
    current = {}

    for group in optimizer.param_groups:
        name = group.get("name", "")

        if name == "backbone":
            group["lr"] = float(lr_backbone) * float(lr_factor) if backbone_trainable else 0.0
            current["backbone"] = float(group["lr"])

        elif name == "head":
            group["lr"] = float(lr_head) * float(lr_factor)
            current["head"] = float(group["lr"])

    return current


def _model_logits(model, images):
    out = model(images)

    if isinstance(out, dict):
        return out["logit"].float()

    return out.float()


def train_image_one_epoch(
    model,
    ema_model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    use_amp,
    loss_cfg,
    train_prevalence,
    grad_clip,
    ema_decay,
):
    model.train()

    loss_sums = {}
    count = 0

    logits_all = []
    labels_all = []
    sample_ids_all = []

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    pbar = tqdm(loader, total=len(loader), desc="TRAIN", leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float().view(-1)
        sample_ids = batch["sample_id"]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            logits = _model_logits(model, images)
            loss_items = compute_image_loss(
                logits=logits,
                labels=labels,
                criterion=criterion,
                loss_cfg=loss_cfg,
                train_prevalence=train_prevalence,
                train=True,
            )
            loss = loss_items["loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {float(loss.detach().cpu())}")

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            optimizer.step()

        if ema_model is not None:
            update_ema_model(ema_model, model, decay=float(ema_decay))

        bs = int(labels.numel())
        count += bs

        for k, v in loss_items.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().cpu()) * bs

        logits_all.append(logits.detach().float().cpu().numpy())
        labels_all.append(labels.detach().float().cpu().numpy())
        sample_ids_all.extend(list(sample_ids))

        if step == 1 or step % 25 == 0:
            pbar.set_postfix({"loss": f"{float(loss.detach().cpu()):.4f}"})

    logits_np = np.concatenate(logits_all).astype(float)
    labels_np = np.concatenate(labels_all).astype(int)
    probs_np = sigmoid_np(logits_np)

    out = {
        "sample_ids": np.asarray(sample_ids_all).astype(str),
        "labels": labels_np,
        "logits": logits_np,
        "probs": probs_np,
        "auroc": safe_auroc(labels_np, probs_np),
        "auprc": safe_auprc(labels_np, probs_np),
        "log_loss": safe_logloss(labels_np, probs_np),
        "brier": safe_brier(labels_np, probs_np),
    }

    for k, v in loss_sums.items():
        out[k] = float(v / max(count, 1))

    return out


@torch.no_grad()
def evaluate_image(
    model,
    loader,
    criterion,
    device,
    use_amp,
    desc="EVAL",
):
    model.eval()

    total_bce = 0.0
    count = 0

    logits_all = []
    labels_all = []
    sample_ids_all = []

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    pbar = tqdm(loader, total=len(loader), desc=desc, leave=False, dynamic_ncols=True)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float().view(-1)
        sample_ids = batch["sample_id"]

        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            logits = _model_logits(model, images)
            bce = criterion(logits.float().view(-1), labels)

        bs = int(labels.numel())
        total_bce += float(bce.detach().cpu()) * bs
        count += bs

        logits_all.append(logits.detach().float().cpu().numpy())
        labels_all.append(labels.detach().float().cpu().numpy())
        sample_ids_all.extend(list(sample_ids))

    logits_np = np.concatenate(logits_all).astype(float)
    labels_np = np.concatenate(labels_all).astype(int)
    probs_np = sigmoid_np(logits_np)

    threshold, best_f1 = best_f1_threshold(labels_np, probs_np)

    return {
        "sample_ids": np.asarray(sample_ids_all).astype(str),
        "labels": labels_np,
        "logits": logits_np,
        "probs": probs_np,
        "loss": float(total_bce / max(count, 1)),
        "bce": float(total_bce / max(count, 1)),
        "auroc": safe_auroc(labels_np, probs_np),
        "auprc": safe_auprc(labels_np, probs_np),
        "log_loss": safe_logloss(labels_np, probs_np),
        "brier": safe_brier(labels_np, probs_np),
        "best_f1": float(best_f1),
        "best_f1_threshold": float(threshold),
    }


def save_image_predictions(stats, out_path, threshold=None):
    prob = np.asarray(stats["probs"], dtype=float)

    data = {
        "sample_id": np.asarray(stats["sample_ids"]).astype(str),
        "label": np.asarray(stats["labels"]).astype(int),
        "image_logit": np.asarray(stats["logits"]).astype(float),
        "image_prob": prob.astype(float),
    }

    if threshold is not None:
        data["prediction"] = (prob >= float(threshold)).astype(int)
        data["threshold"] = float(threshold)

    pd.DataFrame(data).to_csv(out_path, index=False)


def compute_multimodal_loss(out, labels, criterion, loss_cfg):
    labels = labels.float().view(-1)

    fusion_bce = criterion(out["fusion_logit"].float().view(-1), labels)
    ehr_bce = criterion(out["ehr_logit"].float().view(-1), labels)
    image_bce = criterion(out["image_logit"].float().view(-1), labels)

    loss = (
        float(loss_cfg["fusion_bce_weight"]) * fusion_bce
        + float(loss_cfg["ehr_aux_weight"]) * ehr_bce
        + float(loss_cfg["image_aux_weight"]) * image_bce
    )

    return {
        "loss": loss,
        "fusion_bce": fusion_bce,
        "ehr_bce": ehr_bce,
        "image_bce": image_bce,
    }


def batch_to_multimodal_device(batch, device):
    image = batch["image"].to(device, non_blocking=True)
    ehr_x = batch["ehr_x"].to(device, non_blocking=True)
    ehr_m = batch["ehr_m"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True).float().view(-1)
    sample_ids = list(batch["sample_id"])
    return image, ehr_x, ehr_m, labels, sample_ids


def train_multimodal_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    loss_cfg,
    scaler,
    device,
    use_amp,
    grad_clip,
    epoch,
):
    model.train()

    loss_sums = {}
    total_n = 0

    all_labels = []
    all_logits = []
    all_ehr_logits = []
    all_image_logits = []
    all_sample_ids = []

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    pbar = tqdm(loader, total=len(loader), desc=f"TRAIN epoch {epoch}", leave=False, dynamic_ncols=True)

    for batch in pbar:
        image, ehr_x, ehr_m, labels, sample_ids = batch_to_multimodal_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            out = model(image=image, ehr_x=ehr_x, ehr_m=ehr_m, return_all=True)
            losses = compute_multimodal_loss(
                out=out,
                labels=labels,
                criterion=criterion,
                loss_cfg=loss_cfg,
            )
            loss = losses["loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite multimodal training loss: {float(loss.detach().cpu())}")

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            optimizer.step()

        bs = int(labels.numel())
        total_n += bs

        for k, v in losses.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().cpu()) * bs

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_logits.extend(out["fusion_logit"].detach().float().cpu().numpy().tolist())
        all_ehr_logits.extend(out["ehr_logit"].detach().float().cpu().numpy().tolist())
        all_image_logits.extend(out["image_logit"].detach().float().cpu().numpy().tolist())
        all_sample_ids.extend(sample_ids)

        pbar.set_postfix(
            fusion_bce=(
                f"{float(losses['fusion_bce'].detach().cpu()):.4f}"
            )
        )

    logits_np = np.asarray(all_logits, dtype=float)
    labels_np = np.asarray(all_labels, dtype=int)
    probs_np = sigmoid_np(logits_np)

    out_stats = {
        "sample_ids": np.asarray(all_sample_ids).astype(str),
        "labels": labels_np,
        "logits": logits_np,
        "probs": probs_np,
        "ehr_logits": np.asarray(all_ehr_logits, dtype=float),
        "image_logits": np.asarray(all_image_logits, dtype=float),
        "auroc": safe_auroc(labels_np, probs_np),
        "auprc": safe_auprc(labels_np, probs_np),
        "log_loss": safe_logloss(labels_np, probs_np),
        "brier": safe_brier(labels_np, probs_np),
    }

    for k, v in loss_sums.items():
        out_stats[k] = float(v / max(total_n, 1))

    out_stats["loss"] = out_stats["loss"]
    return out_stats


@torch.no_grad()
def evaluate_multimodal(
    model,
    loader,
    criterion,
    loss_cfg,
    device,
    use_amp,
    desc="EVAL",
):
    model.eval()

    loss_sums = {}
    total_n = 0

    all_labels = []
    all_logits = []
    all_ehr_logits = []
    all_image_logits = []
    all_sample_ids = []

    amp_device = "cuda" if device.type == "cuda" else "cpu"

    pbar = tqdm(loader, total=len(loader), desc=desc, leave=False, dynamic_ncols=True)

    for batch in pbar:
        image, ehr_x, ehr_m, labels, sample_ids = batch_to_multimodal_device(batch, device)

        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            out = model(image=image, ehr_x=ehr_x, ehr_m=ehr_m, return_all=True)
            losses = compute_multimodal_loss(
                out=out,
                labels=labels,
                criterion=criterion,
                loss_cfg=loss_cfg,
            )

        bs = int(labels.numel())
        total_n += bs

        for k, v in losses.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().cpu()) * bs

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_logits.extend(out["fusion_logit"].detach().float().cpu().numpy().tolist())
        all_ehr_logits.extend(out["ehr_logit"].detach().float().cpu().numpy().tolist())
        all_image_logits.extend(out["image_logit"].detach().float().cpu().numpy().tolist())
        all_sample_ids.extend(sample_ids)

    logits_np = np.asarray(all_logits, dtype=float)
    labels_np = np.asarray(all_labels, dtype=int)
    probs_np = sigmoid_np(logits_np)

    threshold, best_f1 = best_f1_threshold(labels_np, probs_np)

    out_stats = {
        "sample_ids": np.asarray(all_sample_ids).astype(str),
        "labels": labels_np,
        "logits": logits_np,
        "probs": probs_np,
        "ehr_logits": np.asarray(all_ehr_logits, dtype=float),
        "image_logits": np.asarray(all_image_logits, dtype=float),
        "auroc": safe_auroc(labels_np, probs_np),
        "auprc": safe_auprc(labels_np, probs_np),
        "log_loss": safe_logloss(labels_np, probs_np),
        "brier": safe_brier(labels_np, probs_np),
        "best_f1": float(best_f1),
        "best_f1_threshold": float(threshold),
    }

    for k, v in loss_sums.items():
        out_stats[k] = float(v / max(total_n, 1))

    out_stats["loss"] = out_stats["loss"]
    return out_stats


def save_multimodal_predictions(stats, out_path, threshold=None):
    prob = np.asarray(stats["probs"], dtype=float)

    data = {
        "sample_id": np.asarray(stats["sample_ids"]).astype(str),
        "label": np.asarray(stats["labels"]).astype(int),
        "fusion_logit": np.asarray(stats["logits"]).astype(float),
        "fusion_prob": prob.astype(float),
        "ehr_logit": np.asarray(stats["ehr_logits"]).astype(float),
        "ehr_prob": sigmoid_np(np.asarray(stats["ehr_logits"], dtype=float)),
        "image_logit": np.asarray(stats["image_logits"]).astype(float),
        "image_prob": sigmoid_np(np.asarray(stats["image_logits"], dtype=float)),
    }

    if threshold is not None:
        data["prediction"] = (prob >= float(threshold)).astype(int)
        data["threshold"] = float(threshold)

    pd.DataFrame(data).to_csv(out_path, index=False)
