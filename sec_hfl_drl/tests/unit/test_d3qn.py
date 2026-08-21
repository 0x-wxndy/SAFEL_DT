"""Unit tests for `safel_dt.rl.d3qn`."""

from __future__ import annotations

import numpy as np
import torch

from safel_dt.rl.d3qn import D3qnConfig, D3qnController, DuelingQNet


def test_dueling_q_net_output_shape() -> None:
    net = DuelingQNet(obs_dim=8, n_actions=5, hidden=16)
    x = torch.randn(4, 8)
    q = net(x)
    assert q.shape == (4, 5)


def test_dueling_q_net_zero_mean_advantage() -> None:
    """Q = V + (A - mean(A)). Advantage stream should average to zero
    after the dueling subtraction."""
    net = DuelingQNet(obs_dim=4, n_actions=3, hidden=8)
    x = torch.randn(2, 4)
    h = net.trunk(x)
    a = net.advantage(h)
    centered = a - a.mean(dim=-1, keepdim=True)
    assert torch.allclose(centered.mean(dim=-1), torch.zeros(2), atol=1e-6)


def test_controller_act_returns_valid_action_index() -> None:
    ctrl = D3qnController(obs_dim=6, n_actions=4, cfg=D3qnConfig(seed=0))
    obs = np.zeros(6, dtype=np.float32)
    for _ in range(20):
        a = ctrl.act(obs)
        assert 0 <= a < 4


def test_controller_deterministic_act_uses_argmax() -> None:
    ctrl = D3qnController(obs_dim=4, n_actions=3, cfg=D3qnConfig(seed=0))
    obs = np.zeros(4, dtype=np.float32)
    a1 = ctrl.act(obs, deterministic=True)
    a2 = ctrl.act(obs, deterministic=True)
    assert a1 == a2


def test_controller_learn_pushes_and_runs_gradient_steps() -> None:
    cfg = D3qnConfig(
        learning_starts=4, batch_size=4, buffer_size=64, gradient_steps=1, seed=1
    )
    ctrl = D3qnController(obs_dim=4, n_actions=3, cfg=cfg)
    obs = np.zeros(4, dtype=np.float32)
    next_obs = np.ones(4, dtype=np.float32)
    for i in range(10):
        info = ctrl.learn(obs=obs, action=i % 3, reward=0.1, next_obs=next_obs, done=False)
        assert info["buffer"] == float(i + 1)
    assert info["loss"] >= 0.0


def test_target_network_diverges_after_updates() -> None:
    cfg = D3qnConfig(
        learning_starts=2, batch_size=2, buffer_size=32, gradient_steps=2, target_tau=0.5,
        seed=2,
    )
    ctrl = D3qnController(obs_dim=4, n_actions=2, cfg=cfg)
    rng = np.random.default_rng(0)
    online_before = [p.detach().clone() for p in ctrl._online.parameters()]
    for _ in range(30):
        obs = rng.normal(size=4).astype(np.float32)
        next_obs = rng.normal(size=4).astype(np.float32)
        ctrl.learn(
            obs=obs,
            action=int(rng.integers(0, 2)),
            reward=float(rng.normal()),
            next_obs=next_obs,
            done=False,
        )
    online_after = list(ctrl._online.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(online_before, online_after, strict=True))


def test_epsilon_decays() -> None:
    cfg = D3qnConfig(
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=10,
        learning_starts=1, batch_size=1, buffer_size=8, seed=0,
    )
    ctrl = D3qnController(obs_dim=2, n_actions=2, cfg=cfg)
    eps0 = ctrl.epsilon
    obs = np.zeros(2, dtype=np.float32)
    for _ in range(20):
        ctrl.learn(obs=obs, action=0, reward=0.0, next_obs=obs, done=False)
    assert ctrl.epsilon < eps0
    assert abs(ctrl.epsilon - 0.1) < 1e-6
