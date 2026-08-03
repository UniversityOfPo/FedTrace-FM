"""Reproducibility utilities for seeding all relevant random number generators.

Note: enabling full determinism (``deterministic=True``) can noticeably
reduce performance, especially on GPU, because it disables certain fast
but non-deterministic cuDNN kernels and algorithm auto-tuning. Use it when
exact reproducibility matters more than raw throughput.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) random number generators.

    Args:
        seed: Seed value applied to all random number generators.
        deterministic: If True, additionally configure PyTorch/cuDNN for
            deterministic behavior. This can reduce performance, especially
            on GPU, since it disables non-deterministic (but faster) kernels
            and cuDNN's autotuner.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:  # pragma: no cover - depends on installed kernels
            logger.warning("Could not enable full deterministic algorithms: %s", exc)
    else:
        torch.backends.cudnn.benchmark = True

    logger.info("Random seed set to %d (deterministic=%s)", seed, deterministic)
