# Experiments

How to reproduce the paper tables once the codebase is fleshed out
(PR-3 onwards).

## Per-PR readiness

- **PR-0**: scaffolding only. `pytest` passes against import smoke tests;
  nothing runs end-to-end.
- **PR-1**: Digital-Twin states + IID/Dirichlet partition +
  cost/constraint/reward equations + dual-ascent multiplier update.
- **PR-2**: Paillier wrapper + ECDSA signer + in-process MQTT bus + sim
  clock / measure context.
- **PR-3**: end-to-end FL pipeline -- `FederatedClient`, `FogServer`,
  `CloudServer`, `SecureChannel`/`PlaintextChannel`, `JsonlWriter`, and a
  `run_simulation` CLI.
- **PR-4**: adversarial layer -- `NoAttack`, `LabelFlipAttack`,
  `ModelScaleAttack`, `GaussianNoiseAttack`, plus `FixedAdversary` /
  `PeriodicAdversary` schedules; robust aggregators `krum`, `multi_krum`,
  `trimmed_mean`, `median` exposed through a `STRATEGY_REGISTRY` and
  selectable via `SimulatorConfig.aggregator` + `aggregator_options`.
  Integration test shows a sign-flipped `gamma=-10` attacker destroying
  FedAvg while Multi-Krum / Trimmed-Mean / Median continue to learn.
- **PR-3.5 (IoT pivot, this commit)**: dropped MNIST / F-MNIST / CIFAR-10
  loaders and the CNN/ResNet-8 backbones; the framework is now
  IoT-tabular end-to-end. Adds two real benchmarks and one synthetic
  test fixture (see below). All 202 unit + integration tests still pass.
- **PR-5**: SAC fog-level client selection. Each fog runs
  an SB3 ``SAC`` instance with observation = per-client (loss_ema,
  participated_last, samples_norm, round_progress); action ∈ [0,1]^N.
  ``SelectClients(weights, tau, mu_fog, m_min)`` (paper Algorithm 1)
  turns the continuous action into the participation cohort.
  Pluggable ``FogPolicy`` protocol with ``AllPolicy`` / ``RandomPolicy``
  / ``SacPolicy``; selectable from the CLI via ``--policy {all,random,sac}``.
  Integration test confirms SAC's adversary participation rate strictly
  decreases from early to late rounds.
- **PR-6a**: utility-augmented Lagrangian reward (paper
  eq. 5) + dual-ascent multiplier loop (paper eq. 6) wired into the
  per-round simulator. New ``runtime/cost_accounting.py`` aggregates
  per-fog ``CostBreakdown`` (paper eqs. 3-9) from per-client device
  profiles (good/medium/bad). New ``runtime/lagrangian.py`` holds the
  mutable ``LagrangianState`` and updates ``(nu_lat, nu_cap, nu_priv)``
  by projected, clipped dual ascent over averaged constraint
  violations. Selectable from the CLI via ``--reward {simple,lagrangian}``;
  per-fog costs and global multipliers are persisted to the JSONL
  trace.
- **PR-6b**: cloud-level Dueling + Double DQN (D3QN)
  that picks the aggregator each round.  New
  ``rl/d3qn.py`` (``DuelingQNet`` + per-step ``D3qnController`` with
  replay buffer, soft-updated target net, and epsilon-greedy action),
  ``rl/cloud_env.py`` (observation = per-fog ``(loss, rejection_rate,
  participation, last_reward)`` + multipliers + previous-aggregator
  one-hot), ``rl/cloud_policy.py`` (``CloudPolicy`` protocol with
  ``StaticCloudPolicy`` / ``D3qnCloudPolicy``).  ``CloudServer`` now
  accepts an ``aggregator_override`` per round so the policy can
  switch among ``{fedavg, krum, multi_krum, trimmed_mean, median}``.
  Selectable from the CLI via ``--cloud-policy {static,d3qn}``.
  Integration test shows that under a sign-flipped ``ModelScaleAttack``
  D3QN tapers off FedAvg over training and the final accuracy strictly
  beats a FedAvg-only baseline.
- **PR-7**: comparison baselines.  Two new fog policies
  -- ``HeuristicFogPolicy`` (deterministic linear scoring over
  ``samples_norm``, ``loss_ema`` and divergence proxy) and
  ``BinaryRLFogPolicy`` (per-client epsilon-greedy bandit on negated
  loss, with optional fog-reward update) -- plus
  ``RoundRobinCloudPolicy`` that cycles through the aggregator menu.
  All conform to the existing ``FogPolicy`` / ``CloudPolicy``
  protocols; selectable from the CLI via
  ``--policy {all,random,heuristic,binary_rl,sac}`` and
  ``--cloud-policy {static,round_robin,d3qn}``.  Integration test runs
  all five fog policies head-to-head under a label-flip adversary and
  verifies (a) every policy completes 30 rounds without exception and
  (b) SAC + Heuristic each match or beat the random baseline within
  noise.
- **PR-8 (this commit)**: end-to-end sweep + analysis pipeline that
  fills the paper's main comparison table.  New
  ``safel_dt.runtime.policy_builder`` factors policy / cloud-policy
  construction out of the single-run script into reusable
  ``build_fog_policies()`` and ``build_cloud_policy()`` helpers
  governed by ``FogPolicySpec`` / ``CloudPolicySpec`` dataclasses, so
  the same code path serves both single runs and sweeps.
  ``scripts/run_sweep.py`` walks the cross-product of ``--fog-policies``
  × ``--cloud-policies`` × ``--seeds`` (with per-seed dataset caching
  to amortise N-BaIoT's 150 s load), writes one
  ``<fog>__<cloud>__seed<NNNN>.jsonl`` per cell, and truncates stale
  traces so re-runs are idempotent.  New
  ``safel_dt.eval.aggregate`` (and its thin
  ``analysis/aggregate_runs.py`` CLI) parses the filename convention,
  reduces each trace to a :class:`RunSummary` (final/max accuracy,
  final/min loss, mean per-fog cost components, final multipliers),
  and emits ``summary.csv`` (per-run rows) + ``summary_by_policy.csv``
  (mean / std / min / max across seeds for each policy combo), plus
  optional ``accuracy.png`` / ``loss.png`` learning curves.
  Integration test exercises the full sweep -> aggregate round-trip
  on a synthetic 4-cell mini-sweep.

## Datasets (IoT-only)

| Dataset | Source | Loader | Auto-download | Shape |
|---|---|---|---|---|
| **N-BaIoT** (primary) | [UCI](https://archive.ics.uci.edu/dataset/442) | `safel_dt.data.nbaiot.load_nbaiot_per_device` | yes (HTTP, ~600 MB) | 9 devices × 115 features × {benign, 10 attacks} |
| **TON_IoT** (secondary) | [UNSW](https://research.unsw.edu.au/projects/toniot-datasets) | `safel_dt.data.toniot.load_toniot` | no (manual, ~30 MB) | 211k flows × ~40 features × {benign, 9 attacks} |
| **Edge-IIoTset** (optional) | [Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) | `safel_dt.data.edge_iiotset.load_edge_iiotset` | no (Kaggle API; run `scripts/download_edge_iiotset.py`) | ~46 features × {benign, 14 attacks} |
| **Synthetic tabular** (tests only) | -- | `safel_dt.data.tabular.SyntheticTabularDataset` | -- | configurable; defaults to 32 features × 4 classes |

### Per-device FL split

N-BaIoT has a **natural** 9-client split (one client per device); the
paper's 3-fog topology maps to `assign_clients_to_fogs(9, 3,
strategy="contiguous")`. Edge-IIoTset is shipped as a single CSV; we
default to a uniform random split across 9 clients to keep the topology
comparable. Dirichlet skew is available via `data.partition` if a
stronger non-IID setting is wanted.

## Reproduction recipe (target state)

```powershell
# 1. environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]

# 2. dataset (one-time)
# N-BaIoT auto-downloads on first --dataset nbaiot run.
# Edge-IIoTset: configure ~/.kaggle/kaggle.json then:
python scripts/download_edge_iiotset.py

# 3. smoke (single seed, 5 rounds on N-BaIoT)
python scripts/run_simulation.py --dataset nbaiot --rounds 5 \
    --fogs 3 --seed 101

# 3b. SAC + Lagrangian reward (paper utility-augmented objective)
python scripts/run_simulation.py --dataset nbaiot --rounds 30 \
    --policy sac --reward lagrangian --sac-mu-fog 3 \
    --lag-delta 5 --lag-mu-fog 50 --lag-eta 200

# 3c. Two-agent DRL stack: SAC fog selection + D3QN cloud aggregator
python scripts/run_simulation.py --dataset nbaiot --rounds 50 \
    --policy sac --reward lagrangian --sac-mu-fog 3 \
    --cloud-policy d3qn --d3qn-eps-decay 30 --mode multi

# 3d. Comparison baselines (PR-7): heuristic + binary RL + round-robin cloud
python scripts/run_simulation.py --dataset nbaiot --rounds 30 \
    --policy heuristic --heur-mu-fog 3 --mode multi
python scripts/run_simulation.py --dataset nbaiot --rounds 30 \
    --policy binary_rl --binary-rl-mu-fog 3 --mode multi
python scripts/run_simulation.py --dataset nbaiot --rounds 30 \
    --policy sac --sac-mu-fog 3 --cloud-policy round_robin --mode multi

# 4. multi-seed comparison sweep (PR-8)
python scripts/run_sweep.py --dataset nbaiot --mode multi --rounds 50 \
    --seeds 0,1,2,3,4 \
    --fog-policies all,random,heuristic,binary_rl,sac \
    --cloud-policies static \
    --fogs 3 --mu-fog 2 \
    --out results/runs/sweep_nbaiot

# 5. aggregate the sweep into the paper's main table + learning curves
python analysis/aggregate_runs.py results/runs/sweep_nbaiot --plot
```

All runs emit JSONL + CSV under `results/runs/`; aggregation lands in
`results/reports/`.
