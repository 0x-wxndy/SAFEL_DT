"""Unit tests for PR-9b: stragglers / dropouts.

We exercise the ``_apply_drops`` helper directly instead of running a
full simulation: the helper is pure, deterministic given an RNG, and
returns enough information to assert both the random-Bernoulli and the
deadline-late branches behave as expected.
"""

from __future__ import annotations

import numpy as np

from safel_dt.runtime.cost_accounting import TimingCoefficients
from safel_dt.runtime.simulator import _apply_drops
from safel_dt.types import DeviceDTState, FogDTState


def _device(cid: int, *, drop_prob: float = 0.0, n_samples: int = 100) -> DeviceDTState:
    return DeviceDTState(
        client_id=cid,
        fog_id=0,
        n_samples=n_samples,
        profile="medium",
        record_size_kb=1.0,
        c_enc=0.01,
        c_auth=0.005,
        c_verify=0.005,
        lambda_i=1.0,
        battery=1.0,
        cpu=1.0,
        mem=1.0,
        link_quality=1.0,
        packet_loss=0.0,
        data_fraction=1.0,
        label_noise=0.0,
        drop_prob=drop_prob,
    )


def _fog(fid: int = 0, *, delta: float = 5.0, device_ids: list[int] | None = None) -> FogDTState:
    return FogDTState(fog_id=fid, device_ids=device_ids or [0, 1, 2], delta=delta)


def test_no_drops_when_disabled_and_no_late() -> None:
    devs = {i: _device(i, drop_prob=0.5) for i in range(3)}
    fogs = {0: _fog()}
    rng = np.random.default_rng(0)
    survivors, dr, dl = _apply_drops(
        per_fog_selected={0: [0, 1, 2]},
        client_to_fog={0: [0, 1, 2]},
        device_states=devs,
        fog_states=fogs,
        timing=TimingCoefficients(),
        enable_random=False,
        drop_late=False,
        rng=rng,
    )
    assert survivors == {0: [0, 1, 2]}
    assert dr == {}
    assert dl == {}


def test_bernoulli_drop_rate_matches_drop_prob() -> None:
    """With drop_prob=0.5 and many trials, ~50% of clients should be dropped."""
    n_clients = 1000
    devs = {i: _device(i, drop_prob=0.5) for i in range(n_clients)}
    fogs = {0: _fog(device_ids=list(range(n_clients)))}
    rng = np.random.default_rng(42)
    survivors, dr, dl = _apply_drops(
        per_fog_selected={0: list(range(n_clients))},
        client_to_fog={0: list(range(n_clients))},
        device_states=devs,
        fog_states=fogs,
        timing=TimingCoefficients(),
        enable_random=True,
        drop_late=False,
        rng=rng,
    )
    n_kept = len(survivors[0])
    n_dropped = len(dr[0])
    assert n_kept + n_dropped == n_clients
    assert dl[0] == []
    assert 0.45 * n_clients < n_dropped < 0.55 * n_clients


def test_drop_prob_zero_keeps_everyone() -> None:
    devs = {i: _device(i, drop_prob=0.0) for i in range(5)}
    fogs = {0: _fog(device_ids=list(range(5)))}
    rng = np.random.default_rng(0)
    survivors, dr, _ = _apply_drops(
        per_fog_selected={0: [0, 1, 2, 3, 4]},
        client_to_fog={0: [0, 1, 2, 3, 4]},
        device_states=devs,
        fog_states=fogs,
        timing=TimingCoefficients(),
        enable_random=True,
        drop_late=False,
        rng=rng,
    )
    assert survivors == {0: [0, 1, 2, 3, 4]}
    assert dr == {0: []}


def test_drop_late_drops_slow_clients() -> None:
    """Client 2 has n_samples=10_000 -> t_ml dominates and breaks delta=1.0."""
    devs = {
        0: _device(0, n_samples=10),
        1: _device(1, n_samples=10),
        2: _device(2, n_samples=10_000),
    }
    fogs = {0: _fog(delta=1.0, device_ids=[0, 1, 2])}
    rng = np.random.default_rng(0)
    survivors, dr, dl = _apply_drops(
        per_fog_selected={0: [0, 1, 2]},
        client_to_fog={0: [0, 1, 2]},
        device_states=devs,
        fog_states=fogs,
        timing=TimingCoefficients(),
        enable_random=False,
        drop_late=True,
        rng=rng,
    )
    assert survivors == {0: [0, 1]}
    assert dr == {0: []}
    assert dl == {0: [2]}


def test_random_drop_takes_precedence_over_late() -> None:
    """A client dropped randomly is not double-counted as late."""
    devs = {
        0: _device(0, drop_prob=1.0, n_samples=10_000),  # always dropped randomly
    }
    fogs = {0: _fog(delta=0.001, device_ids=[0])}
    rng = np.random.default_rng(0)
    survivors, dr, dl = _apply_drops(
        per_fog_selected={0: [0]},
        client_to_fog={0: [0]},
        device_states=devs,
        fog_states=fogs,
        timing=TimingCoefficients(),
        enable_random=True,
        drop_late=True,
        rng=rng,
    )
    assert survivors == {0: []}
    assert dr == {0: [0]}
    assert dl == {0: []}


def test_apply_drops_passes_through_when_both_disabled() -> None:
    """If drops are off, the input cohort is returned untouched (None stays None)."""
    rng = np.random.default_rng(0)
    survivors, dr, dl = _apply_drops(
        per_fog_selected=None,
        client_to_fog={0: [0]},
        device_states={0: _device(0)},
        fog_states={0: _fog()},
        timing=TimingCoefficients(),
        enable_random=False,
        drop_late=False,
        rng=rng,
    )
    assert survivors is None
    assert dr == {}
    assert dl == {}


def test_apply_drops_materialises_implicit_cohort_when_enabled() -> None:
    """When ``per_fog_selected`` is None but drops are on, we drop against
    the implicit "select all clients in the fog" cohort, so traces are
    consistent whether or not a fog policy is in use.
    """
    devs = {i: _device(i, drop_prob=1.0) for i in range(3)}
    fogs = {0: _fog(device_ids=[0, 1, 2])}
    rng = np.random.default_rng(0)
    survivors, dr, _ = _apply_drops(
        per_fog_selected=None,
        client_to_fog={0: [0, 1, 2]},
        device_states=devs,
        fog_states=fogs,
        timing=TimingCoefficients(),
        enable_random=True,
        drop_late=False,
        rng=rng,
    )
    assert survivors == {0: []}
    assert sorted(dr[0]) == [0, 1, 2]
