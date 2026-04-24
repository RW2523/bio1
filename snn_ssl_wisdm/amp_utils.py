"""AMP helpers compatible with PyTorch 1.x–2.x."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, ContextManager, Tuple

import torch


def amp_scaler_and_autocast(
    device: torch.device, amp: bool
) -> Tuple[object, Callable[[], ContextManager]]:
    """Returns (scaler, autocast_factory). On CPU, returns no-op scaler and null autocast."""
    use = bool(amp and device.type == "cuda")
    if not use:
        s, f = null_scaler()
        return s, f
    try:
        from torch.amp import GradScaler as GS
        from torch.amp import autocast as AC

        scaler = GS("cuda", enabled=True)

        def factory():
            return AC("cuda", enabled=True)

        return scaler, factory
    except Exception:
        from torch.cuda.amp import GradScaler as GS
        from torch.cuda.amp import autocast as AC

        scaler = GS(enabled=True)

        def factory():
            return AC(enabled=True)

        return scaler, factory


def null_scaler() -> Tuple[object, Callable[[], ContextManager]]:
    class _NS:
        def scale(self, x):
            return x

        def step(self, opt):
            opt.step()

        def update(self):
            pass

        def unscale_(self, opt):
            pass

    return _NS(), lambda: nullcontext()
