"""Multi-seed x multi-policy sweep runner.

For each ``(fog_policy, cloud_policy, seed)`` cell in the cross-product
of the CLI ranges, this script:

1. Builds the same simulator config the single-run script uses
   (:mod:`safel_dt.runtime.policy_builder` does the heavy lifting).
2. Runs ``cfg.rounds`` rounds in-process.
3. Writes a JSONL trace to
   ``<out>/<fog>__<cloud>__seed<NNNN>.jsonl``.

The output folder is the input ``analysis/aggregate_runs.py`` expects.
Defaults are tuned for a fast smoke (synthetic / 2 seeds / 2 policies /
5 rounds); turn it up for the paper run.

Example::

    python scripts/run_sweep.py \\
        --dataset nbaiot --mode multi --rounds 30 \\
        --fog-policies all,random,heuristic,binary_rl,sac \\
        --cloud-policies static \\
        --seeds 0,1,2 --fogs 3 --mu-fog 2 \\
        --out results/runs/sweep_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from safel_dt.costs.reward import CostMaxes, PenaltyConstraints, RewardWeights
from safel_dt.data.datasets import load_dataset
from safel_dt.data.nbaiot import _resolve_root as _resolve_nbaiot_root
from safel_dt.data.partition import assign_clients_to_fogs, iid_partition
from safel_dt.data.tabular import SyntheticTabularDataset
from safel_dt.fl.client import LocalTrainConfig
from safel_dt.models.registry import make_model
from safel_dt.runtime.attack_builder import (
    AttackSpec,
    build_schedule,
    parse_stepwise_breakpoints,
)
from safel_dt.runtime.cost_accounting import PrivacyConfig
from safel_dt.runtime.lagrangian import DualStepConfig, LagrangianConfig
from safel_dt.runtime.policy_builder import (
    CLOUD_POLICY_NAMES,
    FOG_POLICY_NAMES,
    CloudPolicySpec,
    FogPolicySpec,
    SweepCombo,
    build_cloud_policy,
    build_fog_policies,
)
from safel_dt.runtime.simulator import EncryptionConfig, SimulatorConfig, run_simulation
from safel_dt.types import FogDTState


def _build_synthetic(num_clients: int, samples_per_client: int, seed: int):
    in_features = 32
    num_classes = 4
    full = SyntheticTabularDataset(
        n_samples=num_clients * samples_per_client,
        in_features=in_features,
        num_classes=num_classes,
        seed=seed,
        projection_seed=1234,
    )
    rng = np.random.default_rng(seed + 7)
    parts = iid_partition(len(full), num_clients, rng)
    train_sets = [Subset(full, idx.tolist()) for idx in parts]
    test_set = SyntheticTabularDataset(
        n_samples=400,
        in_features=in_features,
        num_classes=num_classes,
        seed=seed + 99,
        projection_seed=1234,
    )
    return train_sets, test_set, {"in_features": in_features, "num_classes": num_classes}


def _comma_list(arg: str, allowed: Sequence[str], kind: str) -> list[str]:
    items = [a.strip() for a in arg.split(",") if a.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"--{kind} cannot be empty.")
    for it in items:
        if it not in allowed:
            raise argparse.ArgumentTypeError(
                f"unknown {kind} {it!r}; expected one of {tuple(allowed)}"
            )
    return items


def _trace_is_complete(trace_path: Path, expected_rounds: int) -> bool:
    """Return True if ``trace_path`` already contains ``round_idx`` 0..expected_rounds-1.

    Used by ``--skip-existing`` so a sweep can resume after a crash without
    redoing combos that already finished. We require contiguous coverage from
    round 0 onward so a truncated trace (e.g. power-cut mid-write) is treated
    as incomplete and re-run from scratch.
    """
    if not trace_path.exists() or expected_rounds <= 0:
        return False
    seen: set[int] = set()
    try:
        with trace_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return False
                if isinstance(row, dict) and "round_idx" in row:
                    seen.add(int(row["round_idx"]))
    except OSError:
        return False
    return all(r in seen for r in range(expected_rounds))


def _seed_list(arg: str) -> list[int]:
    try:
        return [int(s.strip()) for s in arg.split(",") if s.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad --seeds value: {arg!r} ({exc})") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        choices=("synthetic", "nbaiot", "edge_iiotset", "toniot"),
        default="synthetic",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SAFEL_DT_DATA_DIR", ".")),
    )
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seeds", type=_seed_list, default=[0, 1])
    p.add_argument(
        "--fog-policies",
        type=lambda s: _comma_list(s, FOG_POLICY_NAMES, "fog-policies"),
        default=["all", "random"],
    )
    p.add_argument(
        "--cloud-policies",
        type=lambda s: _comma_list(s, CLOUD_POLICY_NAMES, "cloud-policies"),
        default=["static"],
    )
    p.add_argument("--clients", type=int, default=9)
    p.add_argument("--fogs", type=int, default=3)
    p.add_argument("--samples-per-client", type=int, default=200, help="synthetic only")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--mode", choices=("binary", "multi"), default="binary")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--max-per-class", type=int, default=5000, help="N-BaIoT only")
    p.add_argument("--max-samples", type=int, default=200_000)
    p.add_argument(
        "--mu-fog", type=int, default=None,
        help="Per-fog cohort cap shared by heuristic / binary_rl / sac.",
    )
    p.add_argument("--m-min", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument(
        "--aggregator", type=str, default="fedavg",
        help="Cloud aggregator when --cloud-policies contains 'static'.",
    )
    p.add_argument(
        "--cloud-aggregators",
        type=str, default="fedavg,krum,multi_krum,trimmed_mean,median",
        help="Aggregator menu for round_robin / d3qn cloud policies.",
    )
    p.add_argument(
        "--d3qn-eps-decay", type=int, default=100,
        help="Number of rounds over which D3QN epsilon decays.",
    )
    p.add_argument(
        "--d3qn-eps-end", type=float, default=0.02,
        help="Final exploration rate after the decay window.",
    )
    p.add_argument(
        "--d3qn-eps-exponential", action="store_true",
        help="Use geometric (exponential) epsilon decay. Recommended for "
             "short runs (<=200 rounds).",
    )
    p.add_argument("--fog-capacity", type=float, default=None,
                   help="Per-fog mu_fog (updates/s). Default 50.")
    p.add_argument("--fog-deadline", type=float, default=None,
                   help="Per-fog deadline (s). Default 5.")
    p.add_argument("--privacy-epsilon", type=float, default=0.5)
    p.add_argument("--privacy-eta", type=float, default=1.0,
                   help="Privacy budget; g_priv = max(0, I_est - eta).")
    p.add_argument("--reward", choices=("simple", "lagrangian"), default="simple",
                   help="Reward shaping for fog policies.")
    p.add_argument("--lag-omega", type=float, default=1.0,
                   help="Utility weight in the Lagrangian reward.")
    p.add_argument("--lag-alpha", type=float, default=1.0,
                   help="Weight on comm cost.")
    p.add_argument("--lag-beta", type=float, default=1.0,
                   help="Weight on training cost.")
    p.add_argument("--lag-gamma", type=float, default=1.0,
                   help="Weight on security cost.")
    p.add_argument("--lag-delta", type=float, default=5.0,
                   help="Lagrangian latency budget (s).")
    p.add_argument("--lag-mu-fog", type=float, default=50.0,
                   help="Lagrangian fog capacity bound (updates/s).")
    p.add_argument("--lag-eta", type=float, default=1.0,
                   help="Lagrangian privacy budget (MI bound).")
    p.add_argument("--lag-eta-dual", type=float, default=0.05,
                   help="Dual-ascent step size for nu_lat / nu_cap / nu_priv.")
    p.add_argument("--lag-clip", type=float, default=10.0,
                   help="Symmetric reward clip.")
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip combos whose trace file already exists with >= --rounds round_idx entries. "
        "Lets a partially-completed sweep resume after a crash / power loss.",
    )
    p.add_argument("--enable-random-drops", action="store_true",
                   help="Enable per-client Bernoulli dropouts (drop_prob from profile).")
    p.add_argument("--drop-late", action="store_true",
                   help="Drop clients whose per-client time exceeds the fog deadline.")
    p.add_argument("--device-noise", type=float, default=0.0,
                   help="Per-round lognormal jitter sigma on lambda_i + record_size_kb. "
                   "0 = deterministic; 0.1 = ~+/-10%% per round.")
    p.add_argument("--attack",
                   choices=("none", "label_flip", "model_scale", "gaussian", "mixed"),
                   default="none",
                   help="Adversarial behaviour applied to every run in the sweep. "
                        "'mixed' assigns each malicious client one of "
                        "{label_flip, model_scale, gaussian} at cohort "
                        "construction (paper's mixed-attack scenario); requires "
                        "--attack-stepwise or --attack-ramp.")
    p.add_argument("--attack-frac", type=float, default=0.0,
                   help="Fraction of clients made malicious (sampled per-seed).")
    p.add_argument("--attack-start-round", type=int, default=0)
    p.add_argument("--attack-end-round", type=int, default=-1,
                   help="-1 = until end of run.")
    p.add_argument("--attack-shift", type=int, default=1)
    p.add_argument("--attack-gamma", type=float, default=10.0)
    p.add_argument("--attack-sigma", type=float, default=1.5,
                   help="gaussian: stddev of noise added to the delta. "
                        "Default 1.5; the original 0.5 was too weak to dent "
                        "FedAvg on N-BaIoT.")
    p.add_argument(
        "--attack-ramp",
        type=float,
        nargs=2,
        metavar=("EARLY_FRAC", "LATE_FRAC"),
        default=None,
        help="(PR-16) Escalating attack across rounds: cohort grows "
        "linearly from EARLY_FRAC*N clients to LATE_FRAC*N. Overrides "
        "--attack-frac and the start/end-round window.",
    )
    p.add_argument(
        "--attack-stepwise",
        type=str,
        default=None,
        metavar="'r0:f0,r1:f1,...'",
        help="Stepwise malicious-fraction schedule (paper). "
        "Comma-separated round:fraction pairs starting at round 0, e.g. "
        "'0:0.10,100:0.15,200:0.20,250:0.25'. Overrides --attack-frac, "
        "--attack-ramp, and the start/end-round window.",
    )
    p.add_argument(
        "--mixed-gamma-range",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=(10.0, 50.0),
        help="model_scale gamma range when --attack mixed; each model-scale "
        "attacker samples gamma ~ U[LO, HI] at cohort construction. "
        "Default (10, 50) matches the paper.",
    )
    p.add_argument(
        "--encryption",
        choices=("plain", "paillier"),
        default="plain",
        help="Secure channel: 'plain' (no-op) or 'paillier' (additively HE, "
        "calibrated against the real model size at boot).",
    )
    p.add_argument(
        "--paillier-keybits",
        type=int,
        default=1024,
        help="Paillier key length when --encryption paillier (default 1024).",
    )
    p.add_argument(
        "--sig-alg",
        choices=("hmac", "ecdsa", "mldsa"),
        default="hmac",
        help="Update authenticity: hmac | ecdsa | mldsa (ML-DSA-65).",
    )
    p.add_argument(
        "--sac-heuristic-hint",
        action="store_true",
        help="(A5) Append the heuristic policy's per-client score to the "
             "SAC observation. Gives SAC the heuristic's strong prior on "
             "adversarial cohorts; helpful when --adversary-features is "
             "also on. No-op for non-SAC fog policies.",
    )
    p.add_argument(
        "--sac-gradient-steps",
        type=int,
        default=1,
        help="Number of SAC gradient steps per FL round (per fog). "
             "SB3 default is 1; FL rounds are expensive so more steps per "
             "transition can dramatically improve sample efficiency. "
             "Try 4-8 for 300-round sweeps. No-op for non-SAC fog policies.",
    )
    p.add_argument(
        "--adversary-features",
        action="store_true",
        help="(PR-14) Compute per-client adversary-detection features and "
        "feed them into the fog policy observation (SAC + heuristic only).",
    )
    p.add_argument("--out", type=Path, default=Path("results/runs/sweep"))
    return p.parse_args(argv)


_WORLD_CACHE: dict[tuple[str, int], tuple] = {}


def _load_world(args: argparse.Namespace, seed: int):
    """Load (or fetch cached) train splits, test set, and metadata.

    Cache keyed by ``(dataset, seed)``: across fog-/cloud-policy choices
    the world is identical, so reloading the IoT CSVs five times per
    seed is pure waste. Cache is process-local; sweeps that span
    machines share nothing.
    """
    key = (args.dataset, seed)
    cached = _WORLD_CACHE.get(key)
    if cached is not None:
        return cached

    if args.dataset == "synthetic":
        train_sets, test_set, meta = _build_synthetic(
            args.clients, args.samples_per_client, seed
        )
        world = (train_sets, test_set, meta, args.clients)
    elif args.dataset == "nbaiot":
        resolved = _resolve_nbaiot_root(args.data_dir)
        if resolved is not None and seed == args.seeds[0]:
            print(f"[run_sweep] using cached N-BaIoT at {resolved}")
        # ``--clients`` is honoured for N-BaIoT via intra-device IID splits
        # (each of the 9 physical devices is shuffled and partitioned). If
        # the caller leaves the default value (9), we keep the legacy
        # "one client per device" behaviour.
        nbaiot_kwargs: dict[str, object] = dict(
            mode=args.mode, max_per_class=args.max_per_class, seed=seed,
        )
        if args.clients > 0:
            nbaiot_kwargs["num_clients"] = args.clients
        train_sets, test_set, meta = load_dataset(
            "nbaiot", args.data_dir, **nbaiot_kwargs,
        )
        world = (train_sets, test_set, meta, len(train_sets))
    elif args.dataset == "edge_iiotset":
        train_sets, test_set, meta = load_dataset(
            "edge_iiotset", args.data_dir,
            mode=args.mode, num_clients=args.clients,
            max_samples=args.max_samples, seed=seed,
        )
        world = (train_sets, test_set, meta, len(train_sets))
    elif args.dataset == "toniot":
        train_sets, test_set, meta = load_dataset(
            "toniot", args.data_dir,
            mode=args.mode, num_clients=args.clients,
            max_samples=args.max_samples, seed=seed,
        )
        world = (train_sets, test_set, meta, len(train_sets))
    else:
        raise ValueError(f"unknown dataset {args.dataset!r}")

    _WORLD_CACHE[key] = world
    return world


def _run_one(
    *,
    args: argparse.Namespace,
    combo: SweepCombo,
    out_dir: Path,
) -> float:
    train_sets, test_set, meta, num_clients = _load_world(args, combo.seed)
    in_features = int(meta["in_features"])
    num_classes = int(meta["num_classes"])
    client_to_fog = assign_clients_to_fogs(num_clients, args.fogs)
    fog_spec = FogPolicySpec(
        name=combo.policy,
        mu_fog=args.mu_fog,
        sac_heuristic_hint=args.sac_heuristic_hint,
        sac_gradient_steps=args.sac_gradient_steps,
        m_min=args.m_min,
        tau=args.tau,
    )
    fog_policies = build_fog_policies(
        spec=fog_spec,
        client_to_fog=client_to_fog,
        n_samples_per_client=[len(s) for s in train_sets],
        rounds_total=args.rounds,
        seed=combo.seed,
        adversary_features=args.adversary_features,
    )
    aggregators = tuple(a.strip() for a in args.cloud_aggregators.split(",") if a.strip())
    cloud_spec = CloudPolicySpec(
        name=combo.cloud_policy,
        aggregators=aggregators,
        static_aggregator=args.aggregator,
        d3qn_epsilon_decay_steps=args.d3qn_eps_decay,
        d3qn_epsilon_end=args.d3qn_eps_end,
        d3qn_epsilon_exponential=args.d3qn_eps_exponential,
    )
    cloud_policy = build_cloud_policy(
        spec=cloud_spec,
        fog_ids=tuple(sorted(client_to_fog.keys())),
        seed=combo.seed,
    )

    ramp_early, ramp_late = (
        (float(args.attack_ramp[0]), float(args.attack_ramp[1]))
        if args.attack_ramp is not None
        else (None, None)
    )
    stepwise_bps = (
        parse_stepwise_breakpoints(args.attack_stepwise)
        if args.attack_stepwise
        else None
    )
    mixed_gamma_range = (
        float(args.mixed_gamma_range[0]),
        float(args.mixed_gamma_range[1]),
    )
    attack_spec = AttackSpec(
        name=args.attack,
        frac=args.attack_frac,
        start_round=max(0, args.attack_start_round),
        end_round=None if args.attack_end_round < 0 else args.attack_end_round,
        label_shift=args.attack_shift,
        model_gamma=args.attack_gamma,
        gaussian_sigma=args.attack_sigma,
        ramp_early_frac=ramp_early,
        ramp_late_frac=ramp_late,
        stepwise_breakpoints=stepwise_bps,
        mixed_gamma_range=mixed_gamma_range,
    )
    schedule, _malicious = build_schedule(
        spec=attack_spec,
        num_classes=num_classes,
        client_ids=list(range(num_clients)),
        seed=combo.seed,
        rounds_total=args.rounds,
    )

    trace_path = out_dir / combo.trace_filename
    if trace_path.exists():
        trace_path.unlink()
    fog_states = None
    if args.fog_capacity is not None or args.fog_deadline is not None:
        mu_fog = args.fog_capacity if args.fog_capacity is not None else 50.0
        delta = args.fog_deadline if args.fog_deadline is not None else 5.0
        fog_states = {
            fid: FogDTState(fog_id=fid, device_ids=list(cids), mu_fog=mu_fog, delta=delta)
            for fid, cids in client_to_fog.items()
        }
    lagrangian = (
        LagrangianConfig(
            cost_maxes=CostMaxes(comm_max=1_000_000.0, train_max=100.0, sec_max=10.0),
            constraints=PenaltyConstraints(
                delta=args.lag_delta, mu_fog=args.lag_mu_fog, eta=args.lag_eta
            ),
            weights=RewardWeights(
                omega=args.lag_omega,
                alpha=args.lag_alpha,
                beta=args.lag_beta,
                gamma=args.lag_gamma,
            ),
            dual_steps=DualStepConfig(
                eta_lat=args.lag_eta_dual,
                eta_cap=args.lag_eta_dual,
                eta_priv=args.lag_eta_dual,
            ),
            reward_clip=args.lag_clip,
        )
        if args.reward == "lagrangian"
        else None
    )
    cfg = SimulatorConfig(
        seed=combo.seed,
        rounds=args.rounds,
        model_factory=lambda: make_model(
            "mlp",
            in_features=in_features,
            hidden=args.hidden,
            num_classes=num_classes,
        ),
        client_train_sets=train_sets,
        client_to_fog=client_to_fog,
        test_set=test_set,
        train_cfg=LocalTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
        ),
        aggregator=args.aggregator,
        trace_path=trace_path,
        fog_policies=fog_policies,
        cloud_policy=cloud_policy,
        fog_states=fog_states,
        privacy=PrivacyConfig(epsilon=args.privacy_epsilon, eta=args.privacy_eta),
        lagrangian=lagrangian,
        enable_random_drops=args.enable_random_drops,
        drop_late=args.drop_late,
        device_noise_sigma=args.device_noise,
        malicious_schedule=schedule,
        encryption=EncryptionConfig(
            mode=args.encryption,
            paillier_keybits=args.paillier_keybits,
        ),
        sig_alg=args.sig_alg,
        adversary_features=args.adversary_features,
    )
    outcomes = run_simulation(cfg)
    return float(outcomes[-1].accuracy) if outcomes else float("nan")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    _WORLD_CACHE.clear()

    combos: list[SweepCombo] = [
        SweepCombo(
            policy=fp,
            cloud_policy=cp,
            seed=seed,
        )
        for fp in args.fog_policies
        for cp in args.cloud_policies
        for seed in args.seeds
    ]
    print(
        f"[run_sweep] {len(combos)} runs = {len(args.fog_policies)} fog * "
        f"{len(args.cloud_policies)} cloud * {len(args.seeds)} seeds; "
        f"out_dir={out_dir}"
    )
    t0 = time.time()
    for i, combo in enumerate(combos, start=1):
        trace_path = out_dir / combo.trace_filename
        if args.skip_existing and _trace_is_complete(trace_path, args.rounds):
            print(
                f"[run_sweep] {i:>3}/{len(combos)} "
                f"{combo.policy:<10} {combo.cloud_policy:<11} "
                f"seed={combo.seed:<3} {'SKIP (resume)':<22} (0.0s)"
            )
            continue
        run_t0 = time.time()
        try:
            final_acc = _run_one(args=args, combo=combo, out_dir=out_dir)
            status = f"acc={final_acc:.3f}"
        except Exception as exc:
            status = f"FAILED: {exc}"
            final_acc = float("nan")
        dt = time.time() - run_t0
        print(
            f"[run_sweep] {i:>3}/{len(combos)} "
            f"{combo.policy:<10} {combo.cloud_policy:<11} "
            f"seed={combo.seed:<3} {status:<22} ({dt:.1f}s)"
        )
    print(f"[run_sweep] done in {(time.time() - t0):.1f}s")
    print(
        "[run_sweep] aggregate with: "
        f"python analysis/aggregate_runs.py {out_dir} --plot"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
