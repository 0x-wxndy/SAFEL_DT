"""PR-9a: per-round multiplicative lognormal noise on device parameters."""

from __future__ import annotations

import numpy as np

from safel_dt.runtime.simulator import _jittered_states
from safel_dt.types import DeviceDTState


def _device(cid: int, *, lam: float = 4.0, size: float = 1.0) -> DeviceDTState:
    return DeviceDTState(
        client_id=cid,
        fog_id=0,
        n_samples=100,
        profile="medium",
        record_size_kb=size,
        c_enc=0.01,
        c_auth=0.005,
        c_verify=0.005,
        lambda_i=lam,
        battery=1.0,
        cpu=1.0,
        mem=1.0,
        link_quality=1.0,
        packet_loss=0.0,
    )


def test_zero_sigma_returns_same_dict_identity() -> None:
    """sigma=0 must short-circuit -- no allocation, no jitter."""
    base = {0: _device(0), 1: _device(1)}
    rng = np.random.default_rng(0)
    out = _jittered_states(base=base, sigma=0.0, rng=rng)
    assert out is base


def test_jitter_changes_lambda_and_record_size() -> None:
    base = {0: _device(0, lam=4.0, size=2.0)}
    rng = np.random.default_rng(0)
    out = _jittered_states(base=base, sigma=0.2, rng=rng)
    assert out[0] is not base[0]
    # Bothlam and size jittered; equal to original would be vanishingly unlikely
    assert out[0].lambda_i != 4.0
    assert out[0].record_size_kb != 2.0
    # Other fields unchanged
    assert out[0].n_samples == base[0].n_samples
    assert out[0].c_enc == base[0].c_enc
    assert out[0].drop_prob == base[0].drop_prob


def test_jitter_mean_is_approx_nominal_in_log_space() -> None:
    """Over many draws the geometric mean of the multiplier ~ 1.0."""
    base = {0: _device(0, lam=4.0)}
    rng = np.random.default_rng(123)
    sigma = 0.3
    n_trials = 5000
    log_ratios = []
    for _ in range(n_trials):
        out = _jittered_states(base=base, sigma=sigma, rng=rng)
        log_ratios.append(np.log(out[0].lambda_i / 4.0))
    mean = float(np.mean(log_ratios))
    std = float(np.std(log_ratios))
    assert abs(mean) < 3 * sigma / np.sqrt(n_trials)
    assert abs(std - sigma) < 0.05


def test_jitter_is_per_device_independent() -> None:
    """Two clients should get *different* multipliers from the same RNG."""
    base = {0: _device(0, lam=4.0), 1: _device(1, lam=4.0)}
    rng = np.random.default_rng(7)
    out = _jittered_states(base=base, sigma=0.2, rng=rng)
    assert out[0].lambda_i != out[1].lambda_i


def test_jitter_keeps_positivity() -> None:
    """Lognormal multiplier is always > 0; even huge sigma can't flip sign."""
    base = {0: _device(0, lam=1.0, size=1.0)}
    rng = np.random.default_rng(0)
    for _ in range(100):
        out = _jittered_states(base=base, sigma=2.0, rng=rng)
        assert out[0].lambda_i > 0.0
        assert out[0].record_size_kb > 0.0
