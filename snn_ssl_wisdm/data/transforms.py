"""AugPred-style transforms for self-supervised pretraining."""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn.functional as F


def arrow_of_time_pair(x: torch.Tensor, rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [C, T]
    Returns (view, label) where label in {0,1}: 0=original time, 1=reversed.
    """
    if rng.random() < 0.5:
        return x.flip(-1).clone(), torch.tensor(1, dtype=torch.long)
    return x.clone(), torch.tensor(0, dtype=torch.long)


def permutation_pair(x: torch.Tensor, rng: random.Random, num_chunks: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split time axis into num_chunks; with prob 0.5 apply random shuffle of chunks.
    label 0=identity order, 1=shuffled.
    """
    c, t = x.shape
    assert t % num_chunks == 0, f"Time length {t} must be divisible by num_chunks={num_chunks}"
    chunk = t // num_chunks
    chunks = x.view(c, num_chunks, chunk)
    if rng.random() < 0.5:
        perm = list(range(num_chunks))
        rng.shuffle(perm)
        y = torch.tensor(1, dtype=torch.long)
        shuffled = chunks[:, perm, :].contiguous().view(c, t)
        return shuffled, y
    return x.clone(), torch.tensor(0, dtype=torch.long)


def time_warp_pair(x: torch.Tensor, rng: random.Random, strength: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    With prob 0.5, apply random temporal stretch/compress via 1D interpolate then resize back to T.
    label 0=no warp, 1=warped.
    """
    c, t = x.shape
    if rng.random() < 0.5:
        scale = 1.0 + (rng.random() * 2 - 1) * strength
        new_len = max(8, int(round(t * scale)))
        y = torch.tensor(1, dtype=torch.long)
        xx = x.unsqueeze(0)
        warped = F.interpolate(xx, size=new_len, mode="linear", align_corners=False)
        out = F.interpolate(warped, size=t, mode="linear", align_corners=False).squeeze(0)
        return out, y
    return x.clone(), torch.tensor(0, dtype=torch.long)


def movement_std(x: torch.Tensor) -> float:
    """Scalar variability proxy for weighted sampling: std of L2 norm over time."""
    # x: [C, T]
    mag = torch.linalg.norm(x, dim=0)
    return float(mag.std().item() + 1e-6)
