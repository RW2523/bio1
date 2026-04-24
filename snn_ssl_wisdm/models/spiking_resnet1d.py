"""1D Spiking ResNet backbone using SpikingJelly LIF (with local fallback)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from snn_ssl_wisdm.train_utils import ensure_spikegpt_src_on_path


def _try_import_spikingjelly():
    ensure_spikegpt_src_on_path()
    try:
        from spikingjelly.clock_driven import functional, neuron, surrogate

        lif_cls = neuron.LIFNode
        reset_net = functional.reset_net
        atan = surrogate.ATan
        return lif_cls, reset_net, atan, False
    except Exception as e:
        print(f"[snn_ssl_wisdm] SpikingJelly import failed ({e}); using fallback LIF.")
        return None, None, None, True


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


class FallbackLIF(nn.Module):
    """Minimal LIF with surrogate gradient (ATan-like), stateful."""

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        assert tau > 1.0
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.v: Optional[torch.Tensor] = None

    def reset(self):
        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        self.v = self.v + (x - (self.v - self.v_reset)) / self.tau
        spike = _SurrogateSpike.apply(self.v - self.v_threshold, 2.0)
        if self.detach_reset:
            sd = spike.detach()
        else:
            sd = spike
        self.v = (1.0 - sd) * self.v + sd * self.v_reset
        return spike


def _fallback_reset_net(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "reset") and callable(getattr(m, "reset")):
            m.reset()


class SpikingBasicBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        lif_factory,
        reset_ref,
    ):
        super().__init__()
        self.reset_ref = reset_ref
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.lif1 = lif_factory()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.lif2 = lif_factory()
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.float()
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.lif2(out)
        return out


class SpikingResNet1DBackbone(nn.Module):
    """
    Input [B, C, T]. Returns features [B, feature_dim].
    Uses T_in timesteps: independent full forwards with reset, then averages pooled features.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        layers: Optional[List[int]] = None,
        feature_dim: int = 512,
        timesteps: int = 4,
        beta: float = 0.5,
        v_threshold: float = 1.0,
        detach_reset: bool = True,
    ):
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        self.in_channels = in_channels
        self.timesteps = timesteps
        self.feature_dim = feature_dim
        self.beta = beta
        tau = max(1.01, 1.0 / max(1e-3, 1.0 - beta))

        lif_cls, sj_reset, atan_sf, self._fallback = _try_import_spikingjelly()
        if not self._fallback:

            def lif_factory():
                return lif_cls(
                    tau=tau,
                    v_threshold=v_threshold,
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
                )

            self._reset_net = lambda m: _fallback_reset_net(m)

        self.register_buffer(
            "time_scales",
            torch.linspace(0.25, 1.0, timesteps).view(timesteps, 1, 1, 1),
            persistent=False,
        )

        self.conv1 = nn.Conv1d(
            in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.lif_stem = lif_factory()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.in_planes = base_channels
        self.layer1 = self._make_layer(
            lif_factory, base_channels, layers[0], stride=1
        )
        self.layer2 = self._make_layer(
            lif_factory, base_channels * 2, layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            lif_factory, base_channels * 4, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            lif_factory, base_channels * 8, layers[3], stride=2
        )
        self.final_ch = base_channels * 8
        assert self.final_ch == feature_dim, (
            f"feature_dim {feature_dim} must equal last stage channels {self.final_ch} "
            "(ResNet18-1D layout); adjust config model.layers/base_channels."
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self._lif_factory = lif_factory

    def _make_layer(
        self, lif_factory, planes: int, blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers_list = []
        for s in strides:
            layers_list.append(
                SpikingBasicBlock1D(
                    self.in_planes, planes, s, lif_factory, self._reset_net
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers_list)

    def _forward_stem_trunk(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expected [B,C,T], got {tuple(x.shape)}"
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif_stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean pooled feature across timesteps."""
        assert x.dim() == 3
        feats = []
        scales = self.time_scales.to(dtype=x.dtype, device=x.device)
        for t in range(self.timesteps):
            self._reset_net(self)
            xt = x * scales[t]
            h = self._forward_stem_trunk(xt)
            h = self.avgpool(h).flatten(1)
            assert h.shape[1] == self.feature_dim, (
                f"feature dim mismatch {h.shape[1]} vs {self.feature_dim}"
            )
            feats.append(h)
        z = torch.stack(feats, dim=0).mean(dim=0)
        return z

    def set_freeze(self, freeze: bool) -> None:
        for p in self.parameters():
            p.requires_grad = not freeze
