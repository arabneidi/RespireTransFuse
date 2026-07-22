"""Configure repeatable random state for project experiments.

The public helper seeds Python, NumPy, and CPU or CUDA PyTorch generators and sets
the backend flags used for deterministic execution where supported. Training
entry points call it before constructing datasets, samplers, and model weights.
"""

import random

import numpy as np
import torch


def seed_everything(seed=42):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    return seed
