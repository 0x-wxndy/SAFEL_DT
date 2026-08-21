"""Shared DT state types used across the simulator and cost accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

QualityProfile = Literal["good", "medium", "bad"]


@dataclass
class DeviceDTState:
    """Per-client digital-twin snapshot for one simulation seed / round base."""

    client_id: int
    fog_id: int
    n_samples: int
    profile: str
    record_size_kb: float
    c_enc: float
    c_auth: float
    c_verify: float
    lambda_i: float
    battery: float
    cpu: float
    mem: float
    link_quality: float
    packet_loss: float
    data_fraction: float = 1.0
    label_noise: float = 0.0
    drop_prob: float = 0.0


@dataclass
class FogDTState:
    """Per-fog digital-twin configuration."""

    fog_id: int
    device_ids: list[int] = field(default_factory=list)
    mu_fog: float = 50.0
    delta: float = 5.0
