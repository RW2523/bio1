"""Classification and SSL heads."""

from __future__ import annotations

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
    """Two-layer MLP head — used for Case 3 fine-tuning.

    Architecture: Linear → BN → ReLU → Dropout → Linear
    Provides more capacity for end-to-end fine-tuning.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
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
