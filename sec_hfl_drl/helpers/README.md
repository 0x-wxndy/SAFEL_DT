# SAFEL-DT — Secure Adaptive Federated Learning for Hierarchical Digital Twins

Reference implementation of the **SAFEL-DT** framework: hierarchical
federated learning (Client → Fog → Cloud) with continuous client
selection at fog nodes (Soft Actor–Critic), discrete fog-and-aggregator
selection at the cloud (Double-Dueling DQN), Paillier secure
aggregation, and a utility-augmented Lagrangian reward over
communication / training / security costs and latency / capacity /
privacy constraints.

This repository replaces the prior `hfl_phd/src` prototype. No DRL
algorithms or FL aggregators are reimplemented from scratch; we build on
**Flower** for federated learning, **Stable-Baselines3** for SAC, and
**Tianshou** for the Double-Dueling DQN agent.

## Status

**Through PR-4 + IoT pivot.** Working end-to-end FL pipeline with
Paillier secure aggregation, four robust aggregators (Krum / Multi-Krum /
Trimmed-Mean / Median), and three adversarial models (label flip,
model-scale, Gaussian noise) on **three real IoT IDS datasets**:

* **N-BaIoT** (UCI, 9 real IoT devices, 115 features, binary/11-class
  IDS) — primary, auto-downloaded.
* **TON_IoT** (UNSW Canberra 2020, ~40 features after preprocessing,
  binary/10-class IDS network flows) — secondary, ~30 MB.
* **Edge-IIoTset** (Ferrag et al. 2022, ~46 features, 15-class IDS) —
  optional third, Kaggle-gated.

A deterministic `SyntheticTabularDataset` keeps unit tests hermetic.
See `docs/ARCHITECTURE.md` for the plan and `docs/PAPER_MAPPING.md` for
the paper-equation / algorithm → code-path crosswalk.

## Quickstart (development)

```powershell
# from e:\hfl_phd\sec_hfl_drl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]

ruff check .
pytest -q

# end-to-end FL on the offline synthetic fixture (no download)
python scripts/run_simulation.py --dataset synthetic --rounds 3

# end-to-end FL on real N-BaIoT (downloads ~600 MB on first run)
python scripts/run_simulation.py --dataset nbaiot --rounds 5 --fogs 3

# Edge-IIoTset requires a one-time Kaggle download:
python scripts/download_edge_iiotset.py
python scripts/run_simulation.py --dataset edge_iiotset --rounds 5 --fogs 3
```

## Layout

```text
configs/        # YAML scenarios (datasets, attacks, policies, seeds)
src/safel_dt/
  dt/           # device / fog / cloud Digital Twins
  data/         # datasets + Dirichlet / IID partitioning
  models/       # PyTorch backbones (tabular MLP; IoT-focused)
  crypto/       # Paillier, ECDSA, TLS helpers
  transport/    # MQTT (real + sim) + replay protection + timing
  fl/           # Flower clients, fog/cloud servers, strategies
  costs/        # paper §III equations, one file per cost/constraint
  attacks/      # label-flip, model-scale, Gaussian, schedule
  rl/           # Gymnasium envs + SB3 / Tianshou trainers
  runtime/      # round simulator + tracing + checkpointing
  eval/         # metrics + reporting
scripts/        # CLI entry points
analysis/       # plotting tools (port from old analysis/)
tests/          # pytest (unit + integration)
docs/           # architecture, paper mapping, threat model, ...
infra/          # optional docker-compose Mosquitto for real-MQTT runs
results/        # outputs (gitignored)
```

## Library choices

| Concern | Library | Why |
|---|---|---|
| Federated learning + strategies | [Flower](https://flower.ai/) | Customisable strategies (FedAvg, Krum, …); cloud uses a *dispatcher* strategy chosen by D3QN. |
| Continuous fog RL agent | [Stable-Baselines3 `SAC`](https://stable-baselines3.readthedocs.io/) | Matches paper §IV-C: squashed Gaussian actor, twin critics, auto-α. |
| Discrete cloud RL agent | [Tianshou](https://tianshou.org/) | Drop-in Double + Dueling DQN. |
| Homomorphic encryption | [`phe`](https://python-paillier.readthedocs.io/) | Per-coordinate Paillier with fixed-point quantization. |
| Signing | [`eth_account`](https://eth-account.readthedocs.io/) | ECDSA signing/verification (same as the SMART-BUILDING-DT codebase we draw device design from). |
| MQTT + TLS | [`paho-mqtt`](https://eclipse.dev/paho/) | Used both in the sim bus and in the real-broker path. |

## Paper

`docs/PAPER_MAPPING.md` maps every equation / table / algorithm in the
paper to the code path that implements it.

## License

TBD.
