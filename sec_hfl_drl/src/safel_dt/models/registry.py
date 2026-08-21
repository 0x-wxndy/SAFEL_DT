"""Model registry: MLP factory + flat parameter helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """Simple feed-forward classifier for tabular IoT datasets."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden: Sequence[int] | int = (64, 64),
    ) -> None:
        super().__init__()
        if isinstance(hidden, int):
            widths = [hidden]
        else:
            widths = list(hidden)
        layers: list[nn.Module] = []
        prev = int(in_features)
        for h in widths:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model(
    name: str = "mlp",
    *,
    in_features: int,
    num_classes: int,
    hidden: Sequence[int] | int = (64, 64),
) -> nn.Module:
    """Build a model by name (currently only ``mlp``)."""
    if name != "mlp":
        raise ValueError(f"unknown model {name!r}; only 'mlp' is registered")
    return MLP(in_features=in_features, num_classes=num_classes, hidden=hidden)


def get_flat_params(model: nn.Module) -> np.ndarray:
    """Flatten all parameters into a contiguous float64 vector."""
    parts = [p.detach().cpu().reshape(-1).numpy().astype(np.float64) for p in model.parameters()]
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts, axis=0)


def set_flat_params(model: nn.Module, flat: np.ndarray) -> None:
    """Load a flat vector back into ``model`` parameters (in-place)."""
    flat64 = np.asarray(flat, dtype=np.float64).ravel()
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            chunk = flat64[offset : offset + n]
            if chunk.size != n:
                raise ValueError("flat vector is shorter than model parameter count")
            p.copy_(torch.from_numpy(chunk.reshape(tuple(p.shape))).to(dtype=p.dtype))
            offset += n
    if offset != flat64.size:
        raise ValueError("flat vector is longer than model parameter count")


def flat_param_size(model: nn.Module) -> int:
    """Number of scalar parameters in ``model``."""
    return int(sum(p.numel() for p in model.parameters()))
