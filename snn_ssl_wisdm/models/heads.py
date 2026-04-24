"""Classification and SSL heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class AugPredHeads(nn.Module):
    """Three binary logits: arrow-of-time, permutation, time-warp."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.head_aot = nn.Linear(in_dim, 1)
        self.head_perm = nn.Linear(in_dim, 1)
        self.head_tw = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns [B, 3] logits (one per task)."""
        return torch.cat(
            [self.head_aot(z), self.head_perm(z), self.head_tw(z)], dim=1
        )
