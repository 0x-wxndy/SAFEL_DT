"""Unit tests for `safel_dt.runtime.policy_builder`."""

from __future__ import annotations

import pytest

from safel_dt.rl.binary_rl_policy import BinaryRLFogPolicy
from safel_dt.rl.cloud_policy import (
    D3qnCloudPolicy,
    RoundRobinCloudPolicy,
    StaticCloudPolicy,
)
from safel_dt.rl.heuristic_policy import HeuristicFogPolicy
from safel_dt.rl.policy import RandomPolicy, SacPolicy
from safel_dt.runtime.policy_builder import (
    CloudPolicySpec,
    FogPolicySpec,
    SweepCombo,
    build_cloud_policy,
    build_fog_policies,
)


def _client_map() -> dict[int, list[int]]:
    return {0: [0, 1, 2], 1: [3, 4, 5]}


def _samples(n: int = 6) -> list[int]:
    return [100] * n


def test_build_fog_policies_all_returns_none() -> None:
    out = build_fog_policies(
        spec=FogPolicySpec(name="all"),
        client_to_fog=_client_map(),
        n_samples_per_client=_samples(),
        rounds_total=10,
        seed=0,
    )
    assert out is None


def test_build_fog_policies_random_per_fog() -> None:
    out = build_fog_policies(
        spec=FogPolicySpec(name="random", mu_fog=2),
        client_to_fog=_client_map(),
        n_samples_per_client=_samples(),
        rounds_total=10,
        seed=0,
    )
    assert out is not None
    assert set(out.keys()) == {0, 1}
    for p in out.values():
        assert isinstance(p, RandomPolicy)


def test_build_fog_policies_heuristic() -> None:
    out = build_fog_policies(
        spec=FogPolicySpec(name="heuristic", mu_fog=2),
        client_to_fog=_client_map(),
        n_samples_per_client=_samples(),
        rounds_total=10,
        seed=0,
    )
    assert out is not None
    for p in out.values():
        assert isinstance(p, HeuristicFogPolicy)
        chosen = p.select(round_idx=0)
        assert len(chosen) == 2


def test_build_fog_policies_binary_rl() -> None:
    out = build_fog_policies(
        spec=FogPolicySpec(name="binary_rl", mu_fog=2),
        client_to_fog=_client_map(),
        n_samples_per_client=_samples(),
        rounds_total=10,
        seed=0,
    )
    assert out is not None
    for p in out.values():
        assert isinstance(p, BinaryRLFogPolicy)


def test_build_fog_policies_sac() -> None:
    out = build_fog_policies(
        spec=FogPolicySpec(name="sac", mu_fog=2),
        client_to_fog=_client_map(),
        n_samples_per_client=_samples(),
        rounds_total=10,
        seed=0,
    )
    assert out is not None
    for p in out.values():
        assert isinstance(p, SacPolicy)


def test_build_fog_policies_invalid_name_rejected() -> None:
    with pytest.raises(ValueError):
        build_fog_policies(
            spec=FogPolicySpec(name="banana"),  # type: ignore[arg-type]
            client_to_fog=_client_map(),
            n_samples_per_client=_samples(),
            rounds_total=10,
            seed=0,
        )


def test_build_fog_policies_mu_fog_clamped_to_cohort() -> None:
    """mu_fog larger than cohort size must be clamped, not crash."""
    out = build_fog_policies(
        spec=FogPolicySpec(name="heuristic", mu_fog=99),
        client_to_fog={0: [0, 1, 2]},
        n_samples_per_client=_samples(3),
        rounds_total=5,
        seed=0,
    )
    assert out is not None


def test_build_cloud_policy_static_fedavg_returns_none() -> None:
    out = build_cloud_policy(
        spec=CloudPolicySpec(name="static", static_aggregator="fedavg"),
        fog_ids=(0, 1),
        seed=0,
    )
    assert out is None


def test_build_cloud_policy_static_named_returns_static() -> None:
    out = build_cloud_policy(
        spec=CloudPolicySpec(name="static", static_aggregator="krum"),
        fog_ids=(0, 1),
        seed=0,
    )
    assert isinstance(out, StaticCloudPolicy)
    assert out.select(round_idx=0) == "krum"


def test_build_cloud_policy_round_robin() -> None:
    out = build_cloud_policy(
        spec=CloudPolicySpec(name="round_robin"),
        fog_ids=(0, 1),
        seed=0,
    )
    assert isinstance(out, RoundRobinCloudPolicy)


def test_build_cloud_policy_d3qn() -> None:
    out = build_cloud_policy(
        spec=CloudPolicySpec(name="d3qn"),
        fog_ids=(0, 1, 2),
        seed=0,
    )
    assert isinstance(out, D3qnCloudPolicy)


def test_sweep_combo_trace_filename() -> None:
    combo = SweepCombo(policy="sac", cloud_policy="d3qn", seed=7)
    assert combo.trace_filename == "sac__d3qn__seed0007.jsonl"


def test_sweep_combo_trace_filename_with_extras_sorted() -> None:
    combo = SweepCombo(
        policy="sac",
        cloud_policy="static",
        seed=3,
        extras={"attack": "labelflip", "mu_fog": "2"},
    )
    assert combo.trace_filename == "sac__static__seed0003__attack-labelflip__mu_fog-2.jsonl"
