"""Small reproducibility helpers for the public demonstration."""

import os
import random

import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python and PyTorch using portable, best-effort settings."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
