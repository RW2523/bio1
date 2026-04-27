"""Latency and LIF spike-activity estimates for reporting."""

from __future__ import annotations

import time
from typing import List

import torch
import torch.nn as nn


def lif_mean_output_rate(module: nn.Module, x: torch.Tensor) -> float:
    """Mean forward output of all LIF-like modules (spike / surrogate activity), one forward."""
    rates: List[float] = []

    def hook(_m, _inp, out):
        if isinstance(out, torch.Tensor) and out.numel() > 0:
            rates.append(float(out.detach().float().mean().cpu()))

    hooks = []
    for m in module.modules():
        n = type(m).__name__
        if n in ("LIFNode", "ParametricLIFNode", "MultiStepLIFNode", "FallbackLIF"):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        module(x)
    for h in hooks:
        h.remove()
    if not rates:
        return 0.0
    return float(sum(rates) / len(rates))


def forward_latency_ms(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int = 3,
    repeats: int = 12,
) -> float:
    """Mean wall time per forward pass in milliseconds (GPU sync when CUDA)."""
    model.eval()
    x = x.to(device, non_blocking=True)
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if use_cuda:
                torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
            if use_cuda:
                torch.cuda.synchronize()
        t1 = time.perf_counter()
    return float((t1 - t0) / repeats * 1000.0)
