"""Classification and SSL heads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LinearClassifierHead(nn.Module):
    """Single linear layer — used for Cases 1 & 2 linear probing."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPClassifierHead(nn.Module):
    """Deeper MLP classifier for Case 3 (frozen backbone + non-linear probe).

    Uses LayerNorm instead of BatchNorm so small tail batches and probe-only
    training remain stable. Two hidden stages by default (in → h1 → h2 → classes).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        hidden_dim2: Optional[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        h2 = hidden_dim2 if hidden_dim2 is not None else max(128, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRProjector(nn.Module):
    """2-layer MLP projector for SimCLR (Chen et al.) on backbone features.

    Maps representation ``[B, in_dim]`` → latent ``[B, out_dim]`` (L2-normalised
    outside this module in the training loop).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AugPredHeads(nn.Module):
    """Four SSL prediction heads for AugPred pretraining.

    Tasks:
      head_aot  : Arrow-of-time    — binary (BCE), is the signal reversed?
      head_perm : Rotation/Perm    — 4-class (CE),  which of 4 chunk rotations was applied?
      head_tw   : Time-warp        — binary (BCE), was the signal time-warped?
      head_scale: Magnitude scale  — binary (BCE), was the amplitude scaled up?

    The 4-class rotation head provides a much stronger pretext signal than binary
    is/isn't-permuted, giving the backbone richer gradient information.
    """

    def __init__(self, in_dim: int, n_rotations: int = 4):
        super().__init__()
        self.head_aot   = nn.Linear(in_dim, 1)               # binary
        self.head_perm  = nn.Linear(in_dim, n_rotations)     # multi-class
        self.head_tw    = nn.Linear(in_dim, 1)               # binary
        self.head_scale = nn.Linear(in_dim, 1)               # binary

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns concatenated logits [B, 3+n_rotations] — seldom called directly."""
        return torch.cat(
            [self.head_aot(z), self.head_perm(z), self.head_tw(z), self.head_scale(z)], dim=1
        )
