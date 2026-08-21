"""Label-flip attack: permute the training labels.

Two strategies are supported:

* ``target_strategy="cyclic"`` (default): ``c -> (c + shift) % num_classes``.
  Deterministic; same mapping every record.
* ``target_strategy="random"``: each invocation of :meth:`transform_label`
  samples a new target class uniformly from the ``num_classes - 1``
  non-true classes. Per-record randomness; ``shift`` is ignored.

`LabelFlippedDataset` wraps any torch `Dataset` and applies the flip
on-the-fly when items are accessed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from safel_dt.attacks.base import Attack


class LabelFlipAttack:
    """Label flip with selectable target strategy.

    ``target_strategy="cyclic"`` keeps the original deterministic
    ``c -> (c + shift) % K`` behaviour (used by unit tests and the
    single-attack baseline). ``target_strategy="random"`` samples a
    uniform non-true target per record; the paper's mixed-attack
    scenario uses this mode.
    """

    name: str = "label_flip"

    def __init__(
        self,
        num_classes: int,
        shift: int = 1,
        *,
        target_strategy: str = "cyclic",
        rng_seed: int | None = None,
    ) -> None:
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        if target_strategy == "cyclic" and shift == 0:
            raise ValueError("shift must be non-zero for cyclic flip (else it's a no-op).")
        if target_strategy not in ("cyclic", "random"):
            raise ValueError(
                f"target_strategy must be 'cyclic' or 'random', got {target_strategy!r}"
            )
        self.num_classes = num_classes
        self.shift = shift
        self.target_strategy = target_strategy
        self._rng = np.random.default_rng(rng_seed)

    def transform_label(self, label: int) -> int:
        if self.target_strategy == "cyclic":
            return int((label + self.shift) % self.num_classes)
        candidates = [c for c in range(self.num_classes) if c != int(label)]
        return int(self._rng.choice(candidates))

    def transform_delta(self, delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return delta


class LabelFlippedDataset(Dataset):
    """Wrap a base `Dataset` and transform each label via `attack.transform_label`."""

    def __init__(self, base: Dataset, attack: Attack) -> None:
        self._base = base
        self._attack = attack

    def __len__(self) -> int:
        return len(self._base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._base[idx]
        if isinstance(y, torch.Tensor):
            y_new = self._attack.transform_label(int(y.item()))
            return x, torch.tensor(y_new, dtype=torch.long)
        y_new = self._attack.transform_label(int(y))
        return x, torch.tensor(y_new, dtype=torch.long)
