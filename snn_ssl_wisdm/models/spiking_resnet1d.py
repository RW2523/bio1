"""1D Spiking ResNet backbone using SpikingJelly LIF (with local fallback).

Key design: chunk-based temporal processing.
  - Input [B, C, T] is split into `timesteps` temporal chunks of size T/timesteps.
  - Membrane state is reset ONCE per batch (not between timesteps) so LIF neurons
    accumulate charge across the full window — correct SNN temporal behaviour.
  - Rate-coded output = mean spike count across chunks → [B, feature_dim].

Bug fixed from v1: v1 called _reset_net() inside the loop (every timestep) AND fed
scaled copies of the full input, giving zero temporal memory and near-random features.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from snn_ssl_wisdm.train_utils import ensure_spikegpt_src_on_path


def _try_import_spikingjelly():
    ensure_spikegpt_src_on_path()
    try:
        from spikingjelly.clock_driven import functional, neuron, surrogate
        lif_cls   = neuron.LIFNode
        reset_net = functional.reset_net
        atan      = surrogate.ATan
        return lif_cls, reset_net, atan, False
    except Exception as e:
        print(f"[snn_ssl_wisdm] SpikingJelly import failed ({e}); using fallback LIF.")
        return None, None, None, True


# ---------------------------------------------------------------------------
# Surrogate gradient (ATan-like) used by the fallback LIF
# ---------------------------------------------------------------------------
class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_diff: torch.Tensor, alpha: float = 2.0):
        ctx.save_for_backward(v_diff)
        ctx.alpha = alpha
        return (v_diff >= 0).to(v_diff.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (v_diff,) = ctx.saved_tensors
        alpha = ctx.alpha
        sg = alpha / (1.0 + (alpha * v_diff) ** 2) * (1.0 / math.pi)
        return grad_output * sg, None


# ---------------------------------------------------------------------------
# Minimal stateful LIF — used when SpikingJelly import fails
# ---------------------------------------------------------------------------
class FallbackLIF(nn.Module):
    """Stateful LIF with ATan surrogate gradient.

    Matches SpikingJelly ``LIFNode`` with ``decay_input=False`` (direct current / DC drive):
    ``v = v * (1 - 1/tau) + x``. Reset is **soft** (subtract threshold after spike) or **hard**
    (clamp to ``v_reset``).
    """

    def __init__(
        self,
        tau: float = 10.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = True,
        soft_reset: bool = True,
    ):
        super().__init__()
        assert tau > 1.0, "tau must be > 1 for stable LIF dynamics"
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.soft_reset = soft_reset
        self.v: Optional[torch.Tensor] = None

    def reset(self):
        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_v = 0.0 if self.soft_reset else self.v_reset
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.full_like(x, init_v)
        self.v = self.v * (1.0 - 1.0 / self.tau) + x
        spike = _SurrogateSpike.apply(self.v - self.v_threshold, 2.0)
        sd = spike.detach() if self.detach_reset else spike
        if self.soft_reset:
            self.v = self.v - sd * self.v_threshold
        else:
            self.v = (1.0 - sd) * self.v + sd * self.v_reset
        return spike


def _fallback_reset_net(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "reset") and callable(m.reset):
            m.reset()


# ---------------------------------------------------------------------------
# Spiking residual block
# ---------------------------------------------------------------------------
class SpikingBasicBlock1D(nn.Module):
    """Pre-activation residual block: Conv → BN → LIF → Conv → BN → (+skip) → LIF."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        lif_factory,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.lif1 = lif_factory()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.lif2 = lif_factory()
        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.lif1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.lif2(out)
        return out


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class SpikingResNet1DBackbone(nn.Module):
    """
    Spiking 1-D ResNet18 backbone.

    Input  : [B, C, T]
    Output : [B, feature_dim]   (rate-coded mean spike count)

    Temporal processing (correct SNN behaviour):
        1.  Membrane state reset ONCE per batch.
        2.  Input window split into `timesteps` equal temporal chunks.
        3.  Each chunk fed sequentially; LIF state carries across chunks.
        4.  Output = mean of per-chunk feature vectors (rate coding).

    If T is not divisible by timesteps, the backbone falls back to rate coding
    (same full input at each step) which still benefits from LIF accumulation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        layers: Optional[List[int]] = None,
        feature_dim: int = 512,
        timesteps: int = 4,
        beta: float = 0.9,        # tau = 1/(1-beta); 0.9 → tau=10 (slow decay, good temporal memory)
        v_threshold: float = 1.0,
        detach_reset: bool = True,
        decay_input: bool = False,
        soft_reset: bool = True,
    ):
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]

        self.in_channels = in_channels
        self.timesteps   = timesteps
        self.feature_dim = feature_dim
        self.beta        = beta
        tau = max(1.01, 1.0 / max(1e-6, 1.0 - beta))

        lif_cls, sj_reset, atan_sf, self._fallback = _try_import_spikingjelly()

        if not self._fallback:
            def lif_factory():
                return lif_cls(
                    tau=tau,
                    decay_input=decay_input,
                    v_threshold=v_threshold,
                    v_reset=None if soft_reset else 0.0,
                    surrogate_function=atan_sf(),
                    detach_reset=detach_reset,
                )
            self._reset_net = lambda m: sj_reset(m)
        else:
            def lif_factory():
                return FallbackLIF(
                    tau=tau,
                    v_threshold=v_threshold,
                    v_reset=0.0,
                    detach_reset=detach_reset,
                    soft_reset=soft_reset,
                )
            self._reset_net = lambda m: _fallback_reset_net(m)

        # Stem
        self.conv1    = nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm1d(base_channels)
        self.lif_stem = lif_factory()
        self.maxpool  = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self._in_planes = base_channels
        self.layer1 = self._make_layer(lif_factory, base_channels,     layers[0], stride=1)
        self.layer2 = self._make_layer(lif_factory, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(lif_factory, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(lif_factory, base_channels * 8, layers[3], stride=2)

        self.final_ch = base_channels * 8
        assert self.final_ch == feature_dim, (
            f"feature_dim={feature_dim} must equal final stage channels={self.final_ch}. "
            "With the default ResNet-18 layout [2,2,2,2] and base_channels=64, feature_dim must be 512."
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, lif_factory, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(SpikingBasicBlock1D(self._in_planes, planes, s, lif_factory))
            self._in_planes = planes
        return nn.Sequential(*blocks)

    def _forward_single_step(self, x: torch.Tensor) -> torch.Tensor:
        """One SNN timestep: stem → 4 residual stages → GAP → [B, feature_dim]."""
        assert x.dim() == 3, f"Expected [B,C,T], got {tuple(x.shape)}"
        x = self.lif_stem(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chunk-based temporal SNN forward.

        Reset membrane ONCE per batch, then feed T//timesteps-sample chunks
        sequentially.  LIF state accumulates across chunks — neurons see the
        full 10-second context through their membrane potential history.
        Rate-coded output: mean spike count across timesteps.
        """
        assert x.dim() == 3, f"Expected [B,C,T], got {tuple(x.shape)}"
        B, C, T = x.shape

        use_chunks = (T % self.timesteps == 0)
        chunk = T // self.timesteps if use_chunks else T

        # ── single reset per sequence (not per timestep!) ──────────────────
        self._reset_net(self)

        spike_sum = torch.zeros(B, self.feature_dim, device=x.device, dtype=x.dtype)
        for t in range(self.timesteps):
            xt = x[:, :, t * chunk : (t + 1) * chunk] if use_chunks else x
            h  = self._forward_single_step(xt)
            assert h.shape == (B, self.feature_dim), \
                f"Feature shape mismatch: got {tuple(h.shape)}, expected ({B},{self.feature_dim})"
            spike_sum = spike_sum + h

        return spike_sum / self.timesteps

    def set_freeze(self, freeze: bool) -> None:
        """Freeze or unfreeze all backbone parameters."""
        for p in self.parameters():
            p.requires_grad = not freeze
