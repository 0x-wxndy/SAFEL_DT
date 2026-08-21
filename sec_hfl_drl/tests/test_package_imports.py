"""Smoke test: every safel_dt subpackage must import cleanly."""

from __future__ import annotations

import importlib

import pytest

PACKAGES = [
    "safel_dt",
    "safel_dt.config",
    "safel_dt.seeds",
    "safel_dt.types",
    "safel_dt.dt",
    "safel_dt.dt.device",
    "safel_dt.dt.fog_dt",
    "safel_dt.dt.cloud_dt",
    "safel_dt.dt.profiles",
    "safel_dt.data",
    "safel_dt.data.datasets",
    "safel_dt.data.partition",
    "safel_dt.data.nbaiot",
    "safel_dt.data.edge_iiotset",
    "safel_dt.data.toniot",
    "safel_dt.data.tabular",
    "safel_dt.models",
    "safel_dt.models.mlp",
    "safel_dt.models.registry",
    "safel_dt.crypto",
    "safel_dt.crypto.paillier",
    "safel_dt.crypto.signing",
    "safel_dt.crypto.tls",
    "safel_dt.transport",
    "safel_dt.transport.mqtt_real",
    "safel_dt.transport.mqtt_sim",
    "safel_dt.transport.replay",
    "safel_dt.transport.timing",
    "safel_dt.fl",
    "safel_dt.fl.client",
    "safel_dt.fl.fog_server",
    "safel_dt.fl.cloud_server",
    "safel_dt.fl.secure_aggregation",
    "safel_dt.fl.strategies",
    "safel_dt.fl.strategies.fedavg",
    "safel_dt.fl.strategies.trimmed_mean",
    "safel_dt.fl.strategies.krum",
    "safel_dt.fl.strategies.fednova",
    "safel_dt.fl.strategies.adaptive",
    "safel_dt.costs",
    "safel_dt.costs.comm",
    "safel_dt.costs.train",
    "safel_dt.costs.sec",
    "safel_dt.costs.latency",
    "safel_dt.costs.privacy",
    "safel_dt.costs.capacity",
    "safel_dt.costs.reward",
    "safel_dt.attacks",
    "safel_dt.attacks.base",
    "safel_dt.attacks.label_flip",
    "safel_dt.attacks.model_scale",
    "safel_dt.attacks.gaussian",
    "safel_dt.attacks.schedule",
    "safel_dt.rl",
    "safel_dt.rl.fog_env",
    "safel_dt.rl.policy",
    "safel_dt.rl.sac_controller",
    "safel_dt.rl.select_clients",
    "safel_dt.rl.state_extraction",
    "safel_dt.rl.multipliers",
    "safel_dt.rl.trainers",
    "safel_dt.runtime",
    "safel_dt.runtime.simulator",
    "safel_dt.runtime.round_context",
    "safel_dt.runtime.checkpointing",
    "safel_dt.runtime.tracing",
    "safel_dt.eval",
    "safel_dt.eval.metrics",
    "safel_dt.eval.reporting",
]


@pytest.mark.parametrize("module_name", PACKAGES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_version_is_string() -> None:
    import safel_dt

    assert isinstance(safel_dt.__version__, str)
    assert safel_dt.__version__
