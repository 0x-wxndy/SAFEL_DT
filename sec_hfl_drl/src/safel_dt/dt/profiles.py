"""Device quality-profile draws for DT heterogeneity."""

from __future__ import annotations

from typing import Any

import numpy as np

_PROFILE_MEANS: dict[str, dict[str, float]] = {
    "good": {
        "record_size_kb": 0.5,
        "c_enc": 0.005,
        "c_auth": 0.002,
        "c_verify": 0.002,
        "lambda": 1.0,
        "battery": 1.0,
        "cpu": 1.0,
        "mem": 1.0,
        "link_quality": 1.0,
        "packet_loss": 0.0,
        "data_fraction": 1.0,
        "label_noise": 0.0,
        "drop_prob": 0.0,
    },
    "medium": {
        "record_size_kb": 1.0,
        "c_enc": 0.01,
        "c_auth": 0.005,
        "c_verify": 0.005,
        "lambda": 2.0,
        "battery": 0.8,
        "cpu": 0.8,
        "mem": 0.8,
        "link_quality": 0.9,
        "packet_loss": 0.01,
        "data_fraction": 1.0,
        "label_noise": 0.0,
        "drop_prob": 0.02,
    },
    "bad": {
        "record_size_kb": 2.0,
        "c_enc": 0.03,
        "c_auth": 0.01,
        "c_verify": 0.01,
        "lambda": 5.0,
        "battery": 0.4,
        "cpu": 0.4,
        "mem": 0.5,
        "link_quality": 0.6,
        "packet_loss": 0.05,
        "data_fraction": 0.8,
        "label_noise": 0.05,
        "drop_prob": 0.1,
    },
}


def sample_profile_params(profile: str, rng: np.random.Generator) -> dict[str, Any]:
    """Draw a slightly noisy parameter dict for ``profile``."""
    key = profile if profile in _PROFILE_MEANS else "medium"
    base = _PROFILE_MEANS[key]
    out: dict[str, Any] = {}
    for name, mean in base.items():
        if name in ("drop_prob", "label_noise", "packet_loss", "data_fraction"):
            # keep probability-like fields near the mean, clipped to [0, 1]
            noise = float(rng.normal(0.0, 0.02))
            out[name] = float(np.clip(mean + noise, 0.0, 1.0))
        else:
            mult = float(np.exp(rng.normal(0.0, 0.05)))
            out[name] = float(max(1e-9, mean * mult))
    return out
