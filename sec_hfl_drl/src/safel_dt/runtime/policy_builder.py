"""Builders that turn string policy names into concrete policy objects.

Both ``scripts/run_simulation.py`` (single run) and
``scripts/run_sweep.py`` (multi-seed sweep) need to translate a
``--policy heuristic --heur-mu-fog 3 --cloud-policy d3qn ...`` style
CLI into actual :class:`FogPolicy` / :class:`CloudPolicy` instances.
Keeping that mapping in one place avoids the two scripts drifting
apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from safel_dt.rl.binary_rl_policy import BinaryRLConfig, BinaryRLFogPolicy
from safel_dt.rl.cloud_env import CloudObsConfig
from safel_dt.rl.cloud_policy import (
    CloudPolicy,
    D3qnCloudPolicy,
    RoundRobinCloudPolicy,
    StaticCloudPolicy,
)
from safel_dt.rl.d3qn import D3qnConfig
from safel_dt.rl.heuristic_policy import HeuristicConfig, HeuristicFogPolicy
from safel_dt.rl.policy import FogPolicy, RandomPolicy, SacPolicy
from safel_dt.rl.sac_controller import SacControllerConfig
from safel_dt.rl.select_clients import SelectionConfig

FogPolicyName = Literal["all", "random", "heuristic", "binary_rl", "sac"]
CloudPolicyName = Literal["static", "round_robin", "d3qn"]

FOG_POLICY_NAMES: tuple[FogPolicyName, ...] = (
    "all", "random", "heuristic", "binary_rl", "sac",
)
CLOUD_POLICY_NAMES: tuple[CloudPolicyName, ...] = ("static", "round_robin", "d3qn")


@dataclass(frozen=True)
class FogPolicySpec:
    """Hyperparameters for whichever fog policy the caller asked for."""

    name: FogPolicyName = "all"
    mu_fog: int | None = None  # per-fog cohort cap; defaults to ``ceil(N/2)``
    m_min: int = 1
    tau: float = 0.5
    random_k: int | None = None
    heuristic_explore: float = 0.0
    binary_rl_alpha: float = 0.2
    binary_rl_use_fog_reward: bool = False
    sac_buffer_size: int = 2048
    sac_batch_size: int = 16
    sac_learning_starts: int = 8
    sac_gradient_steps: int = 1
    # A5: when True, the SAC observation grows by one scalar per client
    # equal to the heuristic policy's per-client score. Gives SAC a
    # strong prior on adversarial cohorts; defaults off so the existing
    # "vanilla" SAC numbers are reproducible.
    sac_heuristic_hint: bool = False


@dataclass(frozen=True)
class CloudPolicySpec:
    """Hyperparameters for whichever cloud policy the caller asked for."""

    name: CloudPolicyName = "static"
    aggregators: tuple[str, ...] = (
        "fedavg", "krum", "multi_krum", "trimmed_mean", "median",
    )
    d3qn_epsilon_decay_steps: int = 100
    d3qn_epsilon_end: float = 0.02
    d3qn_epsilon_exponential: bool = False
    static_aggregator: str = "fedavg"


def _default_mu_fog(n: int, spec_mu_fog: int | None) -> int:
    if spec_mu_fog is not None:
        return max(1, min(spec_mu_fog, n))
    return max(1, n // 2)


def build_fog_policies(
    *,
    spec: FogPolicySpec,
    client_to_fog: dict[int, list[int]],
    n_samples_per_client: Sequence[int],
    rounds_total: int,
    seed: int,
    adversary_features: bool = False,
) -> dict[int, FogPolicy] | None:
    """Return one :class:`FogPolicy` per fog, or ``None`` for "all clients".

    When ``adversary_features=True`` (PR-14) the heuristic and SAC fog
    policies are constructed with the larger 7-feature observation
    block + the matching cos-distance / norm-outlier score penalties.
    The flag is silently ignored by ``random`` and ``binary_rl`` since
    they don't consume the trace-buffer observation.
    """
    if spec.name == "all":
        return None

    out: dict[int, FogPolicy] = {}
    for fog_id, cids in client_to_fog.items():
        n_clients = len(cids)
        samples = [n_samples_per_client[c] for c in cids]
        mu_fog = _default_mu_fog(n_clients, spec.mu_fog)

        if spec.name == "random":
            k = spec.random_k if spec.random_k is not None else mu_fog
            k = max(1, min(k, n_clients))
            out[fog_id] = RandomPolicy(num_clients=n_clients, k=k, seed=seed + fog_id)
        elif spec.name == "heuristic":
            out[fog_id] = HeuristicFogPolicy(
                num_clients=n_clients,
                client_ids=list(cids),
                n_samples_per_client=samples,
                rounds_total=rounds_total,
                cfg=HeuristicConfig(
                    mu_fog=mu_fog,
                    m_min=min(spec.m_min, mu_fog),
                    explore_prob=spec.heuristic_explore,
                    seed=seed + fog_id,
                ),
                adversary_features=adversary_features,
            )
        elif spec.name == "binary_rl":
            out[fog_id] = BinaryRLFogPolicy(
                num_clients=n_clients,
                client_ids=list(cids),
                n_samples_per_client=samples,
                rounds_total=rounds_total,
                cfg=BinaryRLConfig(
                    mu_fog=mu_fog,
                    m_min=min(spec.m_min, mu_fog),
                    alpha=spec.binary_rl_alpha,
                    use_fog_reward=spec.binary_rl_use_fog_reward,
                    seed=seed + fog_id,
                ),
            )
        elif spec.name == "sac":
            out[fog_id] = SacPolicy(
                num_clients=n_clients,
                client_ids=list(cids),
                n_samples_per_client=samples,
                rounds_total=rounds_total,
                selection=SelectionConfig(
                    tau=spec.tau, mu_fog=mu_fog, m_min=min(spec.m_min, mu_fog)
                ),
                sac_cfg=SacControllerConfig(
                    buffer_size=spec.sac_buffer_size,
                    batch_size=spec.sac_batch_size,
                    learning_starts=spec.sac_learning_starts,
                    gradient_steps=spec.sac_gradient_steps,
                    seed=seed + fog_id,
                ),
                adversary_features=adversary_features,
                heuristic_hint=spec.sac_heuristic_hint,
            )
        else:
            raise ValueError(
                f"Unknown fog policy {spec.name!r}; expected one of {FOG_POLICY_NAMES}"
            )
    return out


def build_cloud_policy(
    *,
    spec: CloudPolicySpec,
    fog_ids: Sequence[int],
    seed: int,
) -> CloudPolicy | None:
    """Return a cloud policy or ``None`` for "static fedavg" semantics."""
    if spec.name == "static":
        if spec.static_aggregator == "fedavg":
            return None
        return StaticCloudPolicy(name=spec.static_aggregator, aggregators=spec.aggregators)
    if spec.name == "round_robin":
        return RoundRobinCloudPolicy(aggregators=spec.aggregators)
    if spec.name == "d3qn":
        obs_cfg = CloudObsConfig(
            fog_ids=tuple(sorted(fog_ids)),
            aggregators=spec.aggregators,
        )
        d3qn = D3qnCloudPolicy(
            obs_cfg=obs_cfg,
            d3qn_cfg=D3qnConfig(
                epsilon_decay_steps=spec.d3qn_epsilon_decay_steps,
                epsilon_end=spec.d3qn_epsilon_end,
                epsilon_exponential=spec.d3qn_epsilon_exponential,
                seed=seed,
            ),
        )
        return cast(CloudPolicy, d3qn)
    raise ValueError(
        f"Unknown cloud policy {spec.name!r}; expected one of {CLOUD_POLICY_NAMES}"
    )


@dataclass(frozen=True)
class SweepCombo:
    """A single experiment cell in the sweep grid."""

    policy: FogPolicyName
    cloud_policy: CloudPolicyName
    seed: int
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def trace_filename(self) -> str:
        bits = [self.policy, self.cloud_policy, f"seed{self.seed:04d}"]
        for k, v in sorted(self.extras.items()):
            bits.append(f"{k}-{v}")
        return "__".join(bits) + ".jsonl"


__all__ = [
    "CLOUD_POLICY_NAMES",
    "FOG_POLICY_NAMES",
    "CloudPolicyName",
    "CloudPolicySpec",
    "FogPolicyName",
    "FogPolicySpec",
    "SweepCombo",
    "build_cloud_policy",
    "build_fog_policies",
]
