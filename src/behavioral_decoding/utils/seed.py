"""Seed control.

Every reported number in this repo should be reproducible from a seed. Results
that move when only the seed moves are seed variance, not findings.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 0) -> int:
    """Seed Python, NumPy, and (if present) PyTorch. Returns the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
