"""Per-round, per-fog cost & constraint accounting.

Aggregates the closed-form costs from :mod:`safel_dt.costs` into a single
:class:`CostBreakdown` per fog, given:

* the per-client device profiles (``DeviceDTState``) attached to the fog,
* the selection mask the fog policy produced this round,
* the fog's static configuration (capacity ``mu_fog``, deadline ``delta``,
  privacy budget ``eta``).

The breakdown feeds the Lagrangian reward and the dual-ascent multiplier
update implemented in :mod:`safel_dt.runtime.lagrangian`.

Time stages are derived from the device profile in a coarse,
physically-plausible way (the paper's eq. (7) is agnostic about how
``T_comm`` / ``T_sec`` / ``T_ml`` decompose):

* ``T_comm  = sum_i s_i * record_size_kb_i * t_comm_per_kb``
* ``T_sec   = sum_i s_i * (c_enc_i + c_auth_i + c_verify_i)``
* ``T_ml    = c_ml_per_sample * sum_i s_i * n_samples_i``

The privacy proxy uses the loose ``epsilon * |D_s|`` bound -- caller
can swap it for the tighter Renyi form via :class:`PrivacyConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safel_dt.costs.capacity import current_workload, g_cap
from safel_dt.costs.comm import comm_cost
from safel_dt.costs.latency import g_lat, t_round
from safel_dt.costs.privacy import g_priv, mi_upper_bound
from safel_dt.costs.sec import per_device_sec_cost, total_sec_cost
from safel_dt.costs.train import train_cost
from safel_dt.types import DeviceDTState, FogDTState


@dataclass(frozen=True)
class CostBreakdown:
    """Single round of per-fog cost / constraint accounting."""

    fog_id: int
    n_selected: int

    cost_comm: float
    cost_train: float
    cost_sec: float

    t_round: float
    workload: float
    mi_estimate: float

    g_lat: float
    g_cap: float
    g_priv: float


@dataclass(frozen=True)
class TimingCoefficients:
    """Convert per-device profile fields to time-stage contributions."""

    t_comm_per_kb: float = 0.01  # s / KB
    c_ml_per_sample: float = 1e-4  # s / sample


@dataclass(frozen=True)
class PrivacyConfig:
    """Privacy-cost configuration for ``compute_fog_cost_breakdown``.

    The MI proxy follows :func:`safel_dt.costs.privacy.mi_upper_bound`:

    * loose bound (``use_tight_bound=False``): ``I_est = epsilon * n_selected``
    * tight bound (``use_tight_bound=True``):  ``I_est = 0.5 * epsilon^2``

    ``eta`` is the *configured* privacy budget the experimenter wants to
    keep below; ``g_priv = max(0, I_est - eta)`` is the hinge violation
    fed into the Lagrangian penalty. A small ``eta`` (default 1.0)
    means even a few selected clients can violate the budget; raise it
    to relax the constraint.
    """

    epsilon: float = 0.5
    eta: float = 1.0
    use_tight_bound: bool = False


_DEFAULT_TIMING: TimingCoefficients = TimingCoefficients()
_DEFAULT_PRIVACY: PrivacyConfig = PrivacyConfig()


def _selection_vector(local_indices: list[int], n: int) -> np.ndarray:
    s = np.zeros(n, dtype=np.float64)
    for i in local_indices:
        if 0 <= i < n:
            s[i] = 1.0
    return s


def per_client_time(
    device: DeviceDTState, *, timing: TimingCoefficients = _DEFAULT_TIMING,
) -> float:
    """Estimated end-to-end round time for a single client (parallel-client model).

    Useful for straggler / deadline-drop logic in the simulator: while
    :func:`compute_fog_cost_breakdown` sums times across the selected
    cohort (serial-fog model from the paper), the *deadline* a real client
    has to meet is its own ``T_comm + T_sec + T_ml``. This helper returns
    that individual time, decoupled from cohort size.
    """
    t_comm = device.record_size_kb * timing.t_comm_per_kb
    t_sec = device.c_enc + device.c_auth + device.c_verify
    t_ml = timing.c_ml_per_sample * device.n_samples
    return float(t_comm + t_sec + t_ml)


def compute_fog_cost_breakdown(
    *,
    fog_state: FogDTState,
    devices: list[DeviceDTState],
    selected_local_indices: list[int],
    timing: TimingCoefficients = _DEFAULT_TIMING,
    privacy: PrivacyConfig = _DEFAULT_PRIVACY,
) -> CostBreakdown:
    """Aggregate per-client costs into a single per-fog breakdown."""
    n = len(devices)
    if n == 0:
        return CostBreakdown(
            fog_id=fog_state.fog_id,
            n_selected=0,
            cost_comm=0.0,
            cost_train=0.0,
            cost_sec=0.0,
            t_round=0.0,
            workload=0.0,
            mi_estimate=0.0,
            g_lat=0.0,
            g_cap=0.0,
            g_priv=0.0,
        )
    s = _selection_vector(selected_local_indices, n)
    n_samples = np.array([d.n_samples for d in devices], dtype=np.float64)
    sigma = np.array([d.record_size_kb for d in devices], dtype=np.float64)
    lambdas = np.array([d.lambda_i for d in devices], dtype=np.float64)
    per_dev_sec = np.array(
        [per_device_sec_cost(d.c_enc, d.c_auth, d.c_verify) for d in devices],
        dtype=np.float64,
    )

    c_comm = comm_cost(s, n_samples, sigma)
    c_sec = total_sec_cost(s, per_dev_sec)
    workload = current_workload(s, lambdas)

    t_comm = float(np.sum(s * sigma) * timing.t_comm_per_kb)
    t_sec = float(c_sec)  # treat sec cost as seconds for the timing budget
    t_ml = float(timing.c_ml_per_sample * np.sum(s * n_samples))
    t_total = t_round(t_comm, t_sec, t_ml)

    c_ml_total = float(timing.c_ml_per_sample * np.sum(s * n_samples))
    c_train = train_cost(
        c_sec=c_sec, c_ml=c_ml_total, workload=workload, mu_fog=fog_state.mu_fog
    )

    n_selected_clients = int(np.sum(s > 0))
    mi_est = mi_upper_bound(
        privacy.epsilon, n_selected_clients, tight=privacy.use_tight_bound
    )

    return CostBreakdown(
        fog_id=fog_state.fog_id,
        n_selected=int(np.sum(s > 0)),
        cost_comm=c_comm,
        cost_train=c_train,
        cost_sec=c_sec,
        t_round=t_total,
        workload=workload,
        mi_estimate=mi_est,
        g_lat=g_lat(t_total, fog_state.delta),
        g_cap=g_cap(workload, fog_state.mu_fog),
        g_priv=g_priv(mi_est, eta=privacy.eta),
    )
