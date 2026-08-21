"""Cloud-side global update + evaluation.

A `CloudServer`:

1. Broadcasts the current global parameter vector to each `FogServer`.
2. Receives per-client encrypted updates (and optionally a fog-level
   homomorphic aggregate).
3. Applies the configured aggregation strategy:
     * `"fedavg"` -- uses the fog-level homomorphic aggregates and combines
       them by total sample count. Privacy-preserving (cloud only sees
       fog-level decrypted aggregate).
     * Anything else from `STRATEGY_REGISTRY` (``krum``, ``trimmed_mean``,
       ``median``, ...) -- decrypts each client's delta and runs the
       robust strategy over the per-client list.
4. Applies the resulting delta and evaluates on the held-out test set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch import nn
from torch.utils.data import Dataset

from safel_dt.attacks.schedule import MaliciousSchedule, NoMalice
from safel_dt.crypto.channel import SecureChannel
from safel_dt.fl.fog_server import FogRoundResult, FogServer
from safel_dt.fl.secure_aggregation import canonical_body, verify_signed
from safel_dt.fl.strategies import get_strategy
from safel_dt.models.registry import flat_param_size, get_flat_params, set_flat_params


@dataclass
class CloudRoundOutcome:
    """Per-round metrics emitted by the cloud."""

    round_idx: int
    accuracy: float
    loss: float
    n_clients_accepted: int
    n_clients_rejected: int
    n_fogs_accepted: int
    aggregator: str
    # ``{global_client_id: local_loss}`` for the clients that participated.
    per_client_losses: dict[int, float] = field(default_factory=dict)
    # ``{fog_id: [global_client_ids that participated]}``.
    per_fog_participants: dict[int, list[int]] = field(default_factory=dict)
    # PR-14: per-fog adversary features (typed loosely to avoid import cycles).
    per_fog_adversary_features: dict[int, dict[int, object]] | None = None


class CloudServer:
    """Top-level orchestrator across all fogs."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module],
        fogs: list[FogServer],
        channel: SecureChannel,
        test_set: Dataset,
        aggregator: str = "fedavg",
        aggregator_options: dict[str, Any] | None = None,
        eval_batch_size: int = 128,
    ) -> None:
        if not channel.has_private_key:
            raise ValueError(
                "Cloud requires a channel with the private key (e.g. PaillierContext.generate)."
            )
        # validate aggregator name (raises if unknown)
        if aggregator != "fedavg":
            _ = get_strategy(aggregator)
        self._model_factory = model_factory
        self._global_model = model_factory()
        self._fogs = fogs
        self._channel = channel
        self._test_set = test_set
        self._eval_batch_size = eval_batch_size
        self._aggregator = aggregator
        self._aggregator_options: dict[str, Any] = dict(aggregator_options or {})
        self.global_param_count = flat_param_size(self._global_model)

    @property
    def aggregator(self) -> str:
        return self._aggregator

    @property
    def global_flat(self) -> np.ndarray:
        return get_flat_params(self._global_model)

    def evaluate(self) -> tuple[float, float]:
        """Evaluate the *global* model on the cloud-side test set."""
        from safel_dt.crypto.channel import PlaintextChannel
        from safel_dt.crypto.signing import Signer
        from safel_dt.fl.client import FederatedClient, LocalTrainConfig

        eval_client = FederatedClient(
            client_id=-1,
            model_factory=self._model_factory,
            train_set=self._test_set,
            channel=PlaintextChannel(),
            signer=Signer.generate(),
            config=LocalTrainConfig(),
        )
        eval_client.set_parameters(self.global_flat)
        return eval_client.evaluate(self._test_set, batch_size=self._eval_batch_size)

    def run_round(
        self,
        round_idx: int,
        *,
        schedule: MaliciousSchedule | None = None,
        rng: np.random.Generator | None = None,
        per_fog_selected_local: dict[int, list[int]] | None = None,
        aggregator_override: str | None = None,
    ) -> CloudRoundOutcome:
        """Dispatch -> collect -> aggregate (by mode) -> apply -> eval.

        ``per_fog_selected_local`` is keyed by ``fog_id`` and gives the
        local indices the fog policy chose. When ``None`` every client of
        every fog participates (the policy-less default).

        ``aggregator_override`` lets a cloud-level policy pick the
        aggregator per round (PR-6b). When ``None`` the server falls back
        to its constructor-time default. Names not in
        ``STRATEGY_REGISTRY`` raise.
        """
        sched: MaliciousSchedule = schedule if schedule is not None else NoMalice()
        if aggregator_override is not None:
            if aggregator_override != "fedavg":
                _ = get_strategy(aggregator_override)
            aggregator = aggregator_override
        else:
            aggregator = self._aggregator

        global_flat = self.global_flat
        fog_results: list[FogRoundResult] = [
            fog.run_round(
                round_idx=round_idx,
                global_flat=global_flat,
                schedule=sched,
                rng=rng,
                selected_local_indices=(
                    per_fog_selected_local.get(fog.fog_id)
                    if per_fog_selected_local is not None
                    else None
                ),
            )
            for fog in self._fogs
        ]

        if aggregator == "fedavg":
            new_delta, stats = self._aggregate_fedavg(fog_results)
        else:
            new_delta, stats = self._aggregate_robust(fog_results, round_idx, aggregator)

        if new_delta is not None:
            set_flat_params(self._global_model, global_flat + new_delta)

        loss, acc = self.evaluate()

        per_client_losses: dict[int, float] = {}
        per_fog_participants: dict[int, list[int]] = {}
        per_fog_adv: dict[int, dict[int, object]] = {}
        for fr in fog_results:
            participants: list[int] = []
            for u in fr.per_client_updates:
                cid = int(u.update.client_id)
                per_client_losses[cid] = float(u.update.local_loss)
                participants.append(cid)
            per_fog_participants[fr.fog_id] = participants
            if fr.client_features:
                per_fog_adv[fr.fog_id] = {
                    int(cid): feat for cid, feat in fr.client_features.items()
                }

        return CloudRoundOutcome(
            round_idx=round_idx,
            accuracy=acc,
            loss=loss,
            n_clients_accepted=stats["n_clients_accepted"],
            n_clients_rejected=stats["n_clients_rejected"],
            n_fogs_accepted=stats["n_fogs_accepted"],
            aggregator=aggregator,
            per_client_losses=per_client_losses,
            per_fog_participants=per_fog_participants,
            per_fog_adversary_features=per_fog_adv or None,
        )

    # --- aggregation branches ------------------------------------------------

    def _aggregate_fedavg(
        self, fog_results: list[FogRoundResult]
    ) -> tuple[np.ndarray | None, dict[str, int]]:
        weighted_ciphers: list[Any] = []
        total_samples = 0
        n_clients_accepted = 0
        n_clients_rejected = 0
        n_fogs_accepted = 0
        for fr in fog_results:
            agg = fr.aggregation
            if agg is None:
                continue
            unscaled = self._channel.scalar_mul_encrypted(
                agg.aggregated_payload, float(agg.total_samples)
            )
            weighted_ciphers.append(unscaled)
            total_samples += agg.total_samples
            n_clients_accepted += agg.n_accepted
            n_clients_rejected += agg.n_rejected
            n_fogs_accepted += 1
        if not weighted_ciphers or total_samples <= 0:
            return None, {
                "n_clients_accepted": n_clients_accepted,
                "n_clients_rejected": n_clients_rejected,
                "n_fogs_accepted": n_fogs_accepted,
            }
        summed = self._channel.sum_encrypted(weighted_ciphers)
        scaled = self._channel.scalar_mul_encrypted(summed, 1.0 / float(total_samples))
        delta = self._channel.decrypt_vector(scaled)
        return delta, {
            "n_clients_accepted": n_clients_accepted,
            "n_clients_rejected": n_clients_rejected,
            "n_fogs_accepted": n_fogs_accepted,
        }

    def _aggregate_robust(
        self, fog_results: list[FogRoundResult], round_idx: int, aggregator: str
    ) -> tuple[np.ndarray | None, dict[str, int]]:
        strategy = get_strategy(aggregator)
        deltas: list[np.ndarray] = []
        weights: list[int] = []
        n_accepted = 0
        n_rejected = 0
        n_fogs_accepted = 0
        for fr in fog_results:
            fog_had_one = False
            for u in fr.per_client_updates:
                if u.update.round_idx != round_idx:
                    n_rejected += 1
                    continue
                if u.signed.payload != u.signer_body:
                    n_rejected += 1
                    continue
                if canonical_body(u.update) != u.signer_body:
                    n_rejected += 1
                    continue
                if not verify_signed(u.signed):
                    n_rejected += 1
                    continue
                deltas.append(self._channel.decrypt_vector(u.update.payload))
                weights.append(u.update.n_samples)
                n_accepted += 1
                fog_had_one = True
            if fog_had_one:
                n_fogs_accepted += 1
        if not deltas:
            return None, {
                "n_clients_accepted": n_accepted,
                "n_clients_rejected": n_rejected,
                "n_fogs_accepted": n_fogs_accepted,
            }
        new_delta = strategy(deltas, weights, **self._aggregator_options)
        return new_delta, {
            "n_clients_accepted": n_accepted,
            "n_clients_rejected": n_rejected,
            "n_fogs_accepted": n_fogs_accepted,
        }
