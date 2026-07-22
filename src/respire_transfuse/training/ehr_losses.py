"""Define the composite objectives available to the EHR training workflow.

The module implements differentiable average-precision, pairwise ranking, hard
negative ranking, and focal binary losses. ``compute_ehr_loss`` combines these
terms with the configured base criterion and weights, returning both the scalar
objective and component values for epoch-level reporting.
"""

import torch
import torch.nn.functional as F


def soft_average_precision_loss(logits, labels, tau=0.10):
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


def batch_pairwise_ranking_loss(logits, labels, max_pairs=65536, margin=0.0):
    logits = logits.float().view(-1)
    labels = labels.float().view(-1)

    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]

    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)

    total_pairs = int(pos.numel() * neg.numel())

    if total_pairs <= int(max_pairs):
        diff = pos[:, None] - neg[None, :]
        diff = diff.reshape(-1)
    else:
        pos_idx = torch.randint(0, pos.numel(), (int(max_pairs),), device=logits.device)
        neg_idx = torch.randint(0, neg.numel(), (int(max_pairs),), device=logits.device)
        diff = pos[pos_idx] - neg[neg_idx]

    return F.softplus(-(diff - float(margin))).mean()


def hard_pairwise_ranking_loss(
    logits,
    labels,
    hard_pos_k=32,
    hard_neg_k=64,
    margin=0.0,
):
    logits = logits.float().view(-1)
    labels = labels.float().view(-1)

    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]

    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)

    kp = min(int(hard_pos_k), pos.numel())
    kn = min(int(hard_neg_k), neg.numel())

    hard_pos = torch.topk(pos, k=kp, largest=False).values
    hard_neg = torch.topk(neg, k=kn, largest=True).values

    diff = hard_pos[:, None] - hard_neg[None, :]

    return F.softplus(-(diff - float(margin))).mean()


def focal_bce_loss(logits, labels, gamma=2.0, alpha=-1.0):
    logits = logits.float().view(-1)
    labels = labels.float().view(-1)

    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    p = torch.sigmoid(logits)
    pt = p * labels + (1.0 - p) * (1.0 - labels)

    focal = (1.0 - pt).pow(float(gamma)) * bce

    if float(alpha) >= 0.0:
        alpha_t = float(alpha) * labels + (1.0 - float(alpha)) * (1.0 - labels)
        focal = alpha_t * focal

    return focal.mean()


def compute_ehr_loss(
    logits,
    labels,
    criterion,
    loss_cfg,
    train=True,
):
    labels = labels.float().view(-1)
    logits = logits.float().view(-1)

    if float(loss_cfg["focal_gamma"]) > 0.0:
        bce = focal_bce_loss(
            logits=logits,
            labels=labels,
            gamma=float(loss_cfg["focal_gamma"]),
            alpha=float(loss_cfg["focal_alpha"]),
        )
    else:
        bce = criterion(logits, labels)

    if train and float(loss_cfg["ap_loss_weight"]) > 0.0:
        ap_loss = soft_average_precision_loss(
            logits=logits,
            labels=labels,
            tau=float(loss_cfg["ap_loss_tau"]),
        )
    else:
        ap_loss = logits.new_tensor(0.0)

    if train and float(loss_cfg["rank_loss_weight"]) > 0.0:
        rank_loss = batch_pairwise_ranking_loss(
            logits=logits,
            labels=labels,
            max_pairs=int(loss_cfg["rank_loss_max_pairs"]),
            margin=float(loss_cfg["rank_loss_margin"]),
        )
    else:
        rank_loss = logits.new_tensor(0.0)

    if train and float(loss_cfg["hard_rank_loss_weight"]) > 0.0:
        hard_rank_loss = hard_pairwise_ranking_loss(
            logits=logits,
            labels=labels,
            hard_pos_k=int(loss_cfg["hard_rank_pos_k"]),
            hard_neg_k=int(loss_cfg["hard_rank_neg_k"]),
            margin=float(loss_cfg["hard_rank_margin"]),
        )
    else:
        hard_rank_loss = logits.new_tensor(0.0)

    logit_l2 = logits.pow(2).mean()

    loss = (
        float(loss_cfg["bce_weight"]) * bce
        + float(loss_cfg["ap_loss_weight"]) * ap_loss
        + float(loss_cfg["rank_loss_weight"]) * rank_loss
        + float(loss_cfg["hard_rank_loss_weight"]) * hard_rank_loss
        + float(loss_cfg["logit_l2"]) * logit_l2
    )

    return {
        "loss": loss,
        "bce": bce,
        "ap_loss": ap_loss,
        "rank_loss": rank_loss,
        "hard_rank_loss": hard_rank_loss,
        "logit_l2": logit_l2,
    }
