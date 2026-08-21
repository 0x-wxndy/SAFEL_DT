"""Edge-IIoTset loader stub (Kaggle-gated secondary benchmark)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class EdgeIIoTsetMissing(FileNotFoundError):
    """Raised when the Edge-IIoTset CSV is not cached locally."""


class EdgeIIoTsetDataset(Dataset):
    """Minimal tensor-backed dataset wrapper."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self._y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx]


def _resolve_csv(data_dir: Path | str) -> Path:
    root = Path(data_dir)
    candidates = [
        root / "edge_iiotset" / "ML-EdgeIIoT-dataset.csv",
        root / "Edge-IIoTset" / "ML-EdgeIIoT-dataset.csv",
        root / "ML-EdgeIIoT-dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise EdgeIIoTsetMissing(
        "Edge-IIoTset CSV not found under "
        f"{root}. Place ML-EdgeIIoT-dataset.csv in results/data/edge_iiotset/."
    )


def load_edge_iiotset(
    data_dir: Path | str,
    **kwargs: Any,
) -> tuple[list[EdgeIIoTsetDataset], EdgeIIoTsetDataset, dict[str, object]]:
    """Load Edge-IIoTset if present; otherwise raise ``EdgeIIoTsetMissing``.

    Full CSV parsing is deferred -- this stub exists so imports and the
    conftest skip path work after the USB restore. Pass a pre-parsed
    cache or extend this loader when the raw CSV is available.
    """
    del kwargs
    _ = _resolve_csv(data_dir)
    raise NotImplementedError(
        "Edge-IIoTset full loader not restored yet; CSV was found but "
        "parsing code is still missing. Use N-BaIoT / synthetic for now."
    )
