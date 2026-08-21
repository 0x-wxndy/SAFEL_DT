"""Training cost and CPU saturation factor (paper eqs. (5)-(6)).

The CPU response-time scaling follows the standard M/M/1 form::

    phi_cpu(lambda) = 1 / (1 - lambda / mu_fog),   0 <= lambda < mu_fog.

Training cost is::

    C_train(W, lambda) = phi_cpu(lambda) * ( C_sec + C_ml(W) ).
"""

from __future__ import annotations

_PHI_CAP_DEFAULT: float = 100.0


def phi_cpu(workload: float, mu_fog: float, *, cap: float = _PHI_CAP_DEFAULT) -> float:
    """Compute paper eq. (6) with a finite saturation cap.

    The raw ``1 / (1 - workload/mu_fog)`` diverges as the workload
    approaches capacity and is ``+inf`` past it. To keep ``cost_train``
    a smooth, comparable signal we cap the response factor at ``cap``
    (default 100, i.e. behaviour matches the unclipped form up to ~99%
    utilisation). Capacity *overflow* itself is captured separately by
    :func:`safel_dt.costs.capacity.g_cap` and penalised through the
    Lagrangian multiplier ``nu_cap``.
    """
    if mu_fog <= 0.0:
        raise ValueError(f"mu_fog must be > 0, got {mu_fog}")
    if workload < 0.0:
        raise ValueError(f"workload must be >= 0, got {workload}")
    if cap <= 0.0:
        raise ValueError(f"cap must be > 0, got {cap}")
    if workload >= mu_fog:
        return cap
    raw = 1.0 / (1.0 - workload / mu_fog)
    return min(raw, cap)


def train_cost(
    c_sec: float, c_ml: float, workload: float, mu_fog: float,
    *, phi_cap: float = _PHI_CAP_DEFAULT,
) -> float:
    """Compute paper eq. (5) with phi_cpu cap (see :func:`phi_cpu`)."""
    if c_sec < 0.0 or c_ml < 0.0:
        raise ValueError("c_sec and c_ml must both be >= 0.")
    return phi_cpu(workload, mu_fog, cap=phi_cap) * (c_sec + c_ml)
