"""SacController smoke test.

We don't measure SAC quality here -- that's the integration test's job.
This unit only verifies that:

* ``act(obs)`` returns a valid action vector,
* ``remember(...)`` / ``learn(...)`` round-trip and increment the buffer,
* ``learn(...)`` past ``learning_starts`` actually invokes SB3's train step
  without raising.
"""

from __future__ import annotations

import numpy as np

from safel_dt.rl.sac_controller import SacController, SacControllerConfig
from safel_dt.rl.state_extraction import obs_dim


def _ctl() -> SacController:
    cfg = SacControllerConfig(
        buffer_size=64,
        batch_size=4,
        learning_starts=2,
        gradient_steps=1,
        verbose=0,
        seed=0,
    )
    return SacController(num_clients=3, cfg=cfg)


def test_act_returns_valid_action() -> None:
    ctl = _ctl()
    obs = np.zeros(obs_dim(3), dtype=np.float32)
    a = ctl.act(obs)
    assert a.shape == (3,)
    assert ((a >= 0.0) & (a <= 1.0)).all()


def test_remember_increments_buffer() -> None:
    ctl = _ctl()
    obs = np.zeros(obs_dim(3), dtype=np.float32)
    a = ctl.act(obs)
    assert ctl.n_transitions == 0
    ctl.remember(obs=obs, action=a, reward=0.1, next_obs=obs, done=False)
    assert ctl.n_transitions == 1


def test_learn_runs_after_warmup() -> None:
    ctl = _ctl()
    obs = np.zeros(obs_dim(3), dtype=np.float32)
    for r in range(5):
        a = ctl.act(obs)
        ctl.learn(obs=obs, action=a, reward=float(r), next_obs=obs, done=False)
    assert ctl.n_transitions == 5
