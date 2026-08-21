"""Per-fog aggregation round.

A `FogServer`:

1. Receives the global parameter vector from the cloud.
2. Forwards it to each child `FederatedClient` and asks them to ``fit``
   under whatever attack the schedule prescribes for ``(client_id, round_idx)``.
3. Verifies signatures and (optionally) homomorphic-sums + scales the
   encrypted deltas (sample-weighted FedAvg inside one fog).
4. Returns a `FogRoundResult` containing both the per-client updates and
   (when computed) the fog-level aggregate.

The cloud picks whether to use the per-client list (for robust aggregation)
or the fog-level aggregate (for sample-weighted FedAvg).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safel_dt.attacks.schedule import MaliciousSchedule, NoMalice
from safel_dt.crypto.channel import SecureChannel
from safel_dt.fl.adversary_features import (
    ClientAdversaryFeatures,
    compute_client_adversary_features,
)
from safel_dt.fl.client import FederatedClient
from safel_dt.fl.secure_aggregation import (
    AggregationResult,
    SignedEncryptedUpdate,
    verify_and_sum,
)


@dataclass
class FogRoundResult:
    """Result of one fog-level FL round."""

    fog_id: int
    aggregation: AggregationResult | None
    per_client_updates: list[SignedEncryptedUpdate]
    client_features: dict[int, ClientAdversaryFeatures] | None = None


class FogServer:
    """In-process orchestrator for the clients attached to a single fog."""

    def __init__(
        self,
        *,
        fog_id: int,
        clients: list[FederatedClient],
        channel: SecureChannel,
        compute_homomorphic_aggregate: bool = True,
        compute_adversary_features: bool = False,
    ) -> None:
        self.fog_id = fog_id
        self.clients = clients
        self._channel = channel
        self._compute_aggregate = compute_homomorphic_aggregate
        self._compute_adversary_features = compute_adversary_features

    def run_round(
        self,
        *,
        round_idx: int,
        global_flat: np.ndarray,
        schedule: MaliciousSchedule | None = None,
        rng: np.random.Generator | None = None,
        selected_local_indices: list[int] | None = None,
    ) -> FogRoundResult:
        """Run a fog-level round.

        ``selected_local_indices`` is the list of *local* indices (into
        ``self.clients``) that the fog policy chose to include this round.
        When ``None`` every child participates (PR-3/PR-4 default).
        """
        sched: MaliciousSchedule = schedule if schedule is not None else NoMalice()
        if selected_local_indices is None:
            chosen = list(self.clients)
        else:
            chosen = [self.clients[i] for i in selected_local_indices]
        per_client: list[SignedEncryptedUpdate] = [
            c.fit(
                round_idx=round_idx,
                global_flat=global_flat,
                attack=sched.attack_for(c.client_id, round_idx),
                rng=rng,
            )
            for c in chosen
        ]
        agg: AggregationResult | None = None
        if self._compute_aggregate and per_client:
            agg = verify_and_sum(per_client, self._channel, expected_round_idx=round_idx)
            if agg.n_accepted == 0 or agg.aggregated_payload is None:
                agg = None

        features: dict[int, ClientAdversaryFeatures] | None = None
        if self._compute_adversary_features and per_client:
            deltas = {
                int(u.update.client_id): self._channel.decrypt_vector(u.update.payload)
                for u in per_client
            }
            losses = {int(u.update.client_id): float(u.update.local_loss) for u in per_client}
            features = compute_client_adversary_features(deltas=deltas, losses=losses)

        return FogRoundResult(
            fog_id=self.fog_id,
            aggregation=agg,
            per_client_updates=per_client,
            client_features=features,
        )
