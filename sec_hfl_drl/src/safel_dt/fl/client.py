"""Local federated client: train, attack, encrypt, sign."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from safel_dt.attacks.base import Attack, NoAttack
from safel_dt.crypto.channel import SecureChannel
from safel_dt.crypto.signing import Signer
from safel_dt.fl.secure_aggregation import (
    EncryptedUpdate,
    SignedEncryptedUpdate,
    canonical_body,
)
from safel_dt.models.registry import get_flat_params, set_flat_params


@dataclass(frozen=True)
class LocalTrainConfig:
    """Hyper-parameters for one local SGD fit."""

    epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    device: str = "cpu"


class FederatedClient:
    """In-process FL client owning a local dataset + signing key."""

    def __init__(
        self,
        *,
        client_id: int,
        model_factory: Callable[[], nn.Module],
        train_set: Dataset,
        channel: SecureChannel,
        signer: Signer,
        config: LocalTrainConfig | None = None,
    ) -> None:
        self.client_id = int(client_id)
        self._model_factory = model_factory
        self._model = model_factory()
        self.train_set = train_set
        self._channel = channel
        self._signer = signer
        self.config = config if config is not None else LocalTrainConfig()

    def set_parameters(self, flat: np.ndarray) -> None:
        set_flat_params(self._model, flat)

    def evaluate(self, dataset: Dataset, *, batch_size: int = 128) -> tuple[float, float]:
        """Return ``(mean_loss, accuracy)`` on ``dataset``."""
        device = torch.device(self.config.device)
        self._model.to(device)
        self._model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device).long()
                logits = self._model(xb)
                total_loss += float(criterion(logits, yb).item())
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total

    def fit(
        self,
        *,
        round_idx: int,
        global_flat: np.ndarray,
        attack: Attack | None = None,
        rng: np.random.Generator | None = None,
    ) -> SignedEncryptedUpdate:
        """Local train -> delta -> optional attack -> encrypt -> sign."""
        att: Attack = attack if attack is not None else NoAttack()
        rng = rng if rng is not None else np.random.default_rng()
        device = torch.device(self.config.device)

        # Optional label-flip via dataset wrapper for label attacks.
        train_set = self.train_set
        if getattr(att, "name", "none") == "label_flip":
            from safel_dt.attacks.label_flip import LabelFlippedDataset

            train_set = LabelFlippedDataset(self.train_set, att)

        self.set_parameters(global_flat)
        self._model.to(device)
        self._model.train()
        opt = torch.optim.SGD(
            self._model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        last_loss = 0.0
        for _ in range(max(1, self.config.epochs)):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device).long()
                opt.zero_grad(set_to_none=True)
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                last_loss = float(loss.item())

        new_flat = get_flat_params(self._model)
        delta = new_flat - np.asarray(global_flat, dtype=np.float64)
        delta = att.transform_delta(delta, rng)
        payload = self._channel.encrypt_vector(delta)
        update = EncryptedUpdate(
            client_id=self.client_id,
            round_idx=int(round_idx),
            payload=payload,
            n_samples=len(train_set),  # type: ignore[arg-type]
            local_loss=last_loss,
        )
        body = canonical_body(update)
        signed = self._signer.sign(body)
        return SignedEncryptedUpdate(update=update, signed=signed, signer_body=body)
