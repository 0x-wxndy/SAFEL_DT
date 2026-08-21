"""Unit tests for `safel_dt.runtime.cost_accounting`."""

from __future__ import annotations

import numpy as np

from safel_dt.dt.profiles import sample_profile_params
from safel_dt.runtime.cost_accounting import (
    PrivacyConfig,
    TimingCoefficients,
    compute_fog_cost_breakdown,
)
from safel_dt.types import DeviceDTState, FogDTState


def _make_device(client_id: int, fog_id: int, n_samples: int, profile: str) -> DeviceDTState:
    rng = np.random.default_rng(client_id)
    params = sample_profile_params(profile, rng)  # type: ignore[arg-type]
    return DeviceDTState(
        client_id=client_id,
        fog_id=fog_id,
        n_samples=n_samples,
        profile=profile,  # type: ignore[arg-type]
        record_size_kb=params["record_size_kb"],
        c_enc=params["c_enc"],
        c_auth=params["c_auth"],
        c_verify=params["c_verify"],
        lambda_i=params["lambda"],
        battery=params["battery"],
        cpu=params["cpu"],
        mem=params["mem"],
        link_quality=params["link_quality"],
        packet_loss=params["packet_loss"],
    )


def test_breakdown_zero_selection_yields_zero_costs() -> None:
    fog = FogDTState(fog_id=0, mu_fog=50.0, delta=10.0)
    devs = [_make_device(i, 0, 100, "good") for i in range(3)]
    b = compute_fog_cost_breakdown(
        fog_state=fog,
        devices=devs,
        selected_local_indices=[],
    )
    assert b.n_selected == 0
    assert b.cost_comm == 0.0
    assert b.cost_sec == 0.0
    assert b.workload == 0.0
    assert b.g_lat == 0.0
    assert b.g_cap == 0.0


def test_breakdown_costs_scale_with_selection_count() -> None:
    fog = FogDTState(fog_id=0, mu_fog=200.0, delta=100.0)
    devs = [_make_device(i, 0, 100, "good") for i in range(4)]
    b1 = compute_fog_cost_breakdown(fog_state=fog, devices=devs, selected_local_indices=[0])
    b3 = compute_fog_cost_breakdown(fog_state=fog, devices=devs, selected_local_indices=[0, 1, 2])
    assert b3.cost_comm > b1.cost_comm
    assert b3.cost_sec > b1.cost_sec
    assert b3.workload > b1.workload
    assert b3.n_selected == 3
    assert b1.n_selected == 1


def test_capacity_violation_triggers_g_cap() -> None:
    fog = FogDTState(fog_id=0, mu_fog=2.0, delta=100.0)
    devs = [_make_device(i, 0, 50, "bad") for i in range(4)]
    b = compute_fog_cost_breakdown(
        fog_state=fog,
        devices=devs,
        selected_local_indices=[0, 1, 2, 3],
    )
    assert b.g_cap > 0.0


def test_latency_violation_triggers_g_lat() -> None:
    fog = FogDTState(fog_id=0, mu_fog=1000.0, delta=0.001)
    devs = [_make_device(i, 0, 200, "bad") for i in range(3)]
    b = compute_fog_cost_breakdown(
        fog_state=fog,
        devices=devs,
        selected_local_indices=[0, 1, 2],
        timing=TimingCoefficients(t_comm_per_kb=0.1, c_ml_per_sample=1e-3),
    )
    assert b.g_lat > 0.0


def test_privacy_proxy_finite_and_nonneg() -> None:
    fog = FogDTState(fog_id=0, mu_fog=200.0, delta=100.0)
    devs = [_make_device(i, 0, 100, "good") for i in range(3)]
    b = compute_fog_cost_breakdown(
        fog_state=fog,
        devices=devs,
        selected_local_indices=[0, 1, 2],
        privacy=PrivacyConfig(epsilon=1.0, eta=1e6),
    )
    assert b.mi_estimate >= 0.0
    assert np.isfinite(b.mi_estimate)


def test_privacy_eta_triggers_g_priv_when_exceeded() -> None:
    """With a small eta the MI bound dominates -> g_priv must be > 0."""
    fog = FogDTState(fog_id=0, mu_fog=200.0, delta=100.0)
    devs = [_make_device(i, 0, 100, "good") for i in range(3)]
    b = compute_fog_cost_breakdown(
        fog_state=fog,
        devices=devs,
        selected_local_indices=[0, 1, 2],
        privacy=PrivacyConfig(epsilon=1.0, eta=1.0),
    )
    assert b.g_priv > 0.0


def test_privacy_eta_large_keeps_g_priv_zero() -> None:
    fog = FogDTState(fog_id=0, mu_fog=200.0, delta=100.0)
    devs = [_make_device(i, 0, 100, "good") for i in range(3)]
    b = compute_fog_cost_breakdown(
        fog_state=fog,
        devices=devs,
        selected_local_indices=[0, 1, 2],
        privacy=PrivacyConfig(epsilon=0.01, eta=1e6),
    )
    assert b.g_priv == 0.0


def test_per_client_time_is_per_client() -> None:
    """``per_client_time`` is independent of cohort size."""
    from safel_dt.runtime.cost_accounting import TimingCoefficients, per_client_time

    timing = TimingCoefficients(t_comm_per_kb=0.01, c_ml_per_sample=1e-4)
    d_small = _make_device(0, 0, 100, "good")
    d_big = _make_device(1, 0, 10_000, "good")
    t_small = per_client_time(d_small, timing=timing)
    t_big = per_client_time(d_big, timing=timing)
    assert t_big > t_small
    # ml term should dominate for the big client: ~1.0s
    assert 0.5 < t_big < 5.0
