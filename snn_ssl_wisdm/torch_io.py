"""torch.load compatible across PyTorch versions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import torch


def torch_load(path: Union[str, Path], map_location: Any = None) -> Any:
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
