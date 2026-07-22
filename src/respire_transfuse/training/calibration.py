"""Fit and apply validation-derived temperature and bias calibration."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from respire_transfuse.training.metrics import sigmoid_np


def inv_softplus(y):
    y = float(max(y, 1e-6))
    return math.log(math.exp(y) - 1.0)


def fit_temperature_bias(val_logits, val_labels, device, max_iter=200):
    logits = torch.tensor(val_logits, dtype=torch.float32, device=device)
    labels = torch.tensor(
        np.asarray(val_labels).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )

    raw_temp = torch.tensor(
        inv_softplus(1.0 - 0.05),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    bias = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    optimizer = torch.optim.LBFGS(
        [raw_temp, bias],
        lr=0.05,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temp = F.softplus(raw_temp) + 0.05
        cal_logits = logits / temp + bias
        loss = F.binary_cross_entropy_with_logits(cal_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        temp = F.softplus(raw_temp).item() + 0.05
        b = bias.item()

    return float(temp), float(b)


def apply_temperature_bias(logits, temperature, bias):
    logits = np.asarray(logits, dtype=np.float64)
    return logits / float(temperature) + float(bias)


def calibrated_probabilities(logits, temperature, bias):
    calibrated_logits = apply_temperature_bias(logits, temperature, bias)
    return sigmoid_np(calibrated_logits)
