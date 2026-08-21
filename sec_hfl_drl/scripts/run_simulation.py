"""Single-seed FL run over an IoT IDS dataset (N-BaIoT or Edge-IIoTset).

Examples
--------
Quick smoke (no download, fully synthetic tabular fixture)::

    python scripts/run_simulation.py --dataset synthetic --rounds 3

Real N-BaIoT (auto-downloads ~600 MB from UCI on first run)::

    python scripts/run_simulation.py --dataset nbaiot --rounds 5 \
        --data-dir results/data --fogs 3

Edge-IIoTset (requires one-time ``scripts/download_edge_iiotset.py``)::

    python scripts/run_simulation.py --dataset edge_iiotset --rounds 5 \
        --data-dir results/data --fogs 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import cast

import numpy as np
from torch.utils.data import Subset

from safel_dt.costs.reward import CostMaxes, PenaltyConstraints, RewardWeights
from safel_dt.data.datasets import load_dataset
from safel_dt.data.nbaiot import _resolve_root as _resolve_nbaiot_root
from safel_dt.data.partition import assign_clients_to_fogs, iid_partition
from safel_dt.data.tabular import SyntheticTabularDataset
from safel_dt.fl.client import LocalTrainConfig
from safel_dt.models.registry import make_model
from safel_dt.rl.binary_rl_policy import BinaryRLConfig, BinaryRLFogPolicy
from safel_dt.rl.cloud_env import CloudObsConfig
from safel_dt.rl.cloud_policy import (
    CloudPolicy,
    D3qnCloudPolicy,
    RoundRobinCloudPolicy,
    StaticCloudPolicy,
)
from safel_dt.rl.d3qn import D3qnConfig
from safel_dt.rl.heuristic_policy import HeuristicConfig, HeuristicFogPolicy
from safel_dt.rl.policy import FogPolicy, RandomPolicy, SacPolicy
from safel_dt.rl.sac_controller import SacControllerConfig
from safel_dt.rl.select_clients import SelectionConfig
from safel_dt.runtime.attack_builder import (
    AttackSpec,
    build_schedule,
    parse_stepwise_breakpoints,
)
from safel_dt.runtime.cost_accounting import PrivacyConfig, TimingCoefficients
from safel_dt.runtime.lagrangian import DualStepConfig, LagrangianConfig
from safel_dt.runtime.simulator import EncryptionConfig, SimulatorConfig, run_simulation


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=("synthetic", "nbaiot", "edge_iiotset", "toniot"),
        default="synthetic",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("SAFEL_DT_DATA_DIR", ".")),
        help=(
            "Where to look for / cache IoT datasets. Loader walks subdirs "
            "and recognises common N-BaIoT folder names (e.g. ``N-baIOT``). "
            "Defaults to the SAFEL_DT_DATA_DIR env var, then to the current "
            "directory."
        ),
    )
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--clients", type=int, default=9)
    p.add_argument("--fogs", type=int, default=3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--mode", choices=("binary", "multi"), default="binary")
    p.add_argument("--samples-per-client", type=int, default=200, help="synthetic only")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--max-per-class", type=int, default=5000, help="N-BaIoT only")
    p.add_argument("--max-samples", type=int, default=200_000, help="Edge-IIoTset only")
    p.add_argument("--trace", type=Path, default=Path("results/runs/sim.jsonl"))
    p.add_argument(
        "--policy",
        choices=("all", "random", "heuristic", "binary_rl", "sac"),
        default="all",
        help="Per-fog participation policy.",
    )
    p.add_argument("--sac-mu-fog", type=int, default=None, help="SAC capacity per fog.")
    p.add_argument("--sac-tau", type=float, default=0.5)
    p.add_argument("--sac-m-min", type=int, default=1)
    p.add_argument("--random-k", type=int, default=None, help="random policy cohort size.")
    p.add_argument("--heur-mu-fog", type=int, default=None, help="Heuristic capacity per fog.")
    p.add_argument("--heur-explore", type=float, default=0.0, help="Heuristic explore prob.")
    p.add_argument(
        "--binary-rl-mu-fog", type=int, default=None, help="Binary-RL capacity per fog."
    )
    p.add_argument("--binary-rl-alpha", type=float, default=0.2)
    p.add_argument(
        "--binary-rl-use-fog-reward",
        action="store_true",
        help="Update Q-values with the fog-level reward instead of per-client loss.",
    )
    p.add_argument(
        "--reward",
        choices=("simple", "lagrangian"),
        default="simple",
        help="Reward function for fog policies (no-op for --policy all).",
    )
    p.add_argument("--lag-omega", type=float, default=1.0, help="Lagrangian utility weight.")
    p.add_argument("--lag-alpha", type=float, default=0.3, help="Lagrangian comm-cost weight.")
    p.add_argument("--lag-beta", type=float, default=0.3, help="Lagrangian train-cost weight.")
    p.add_argument("--lag-gamma", type=float, default=0.2, help="Lagrangian sec-cost weight.")
    p.add_argument("--lag-delta", type=float, default=5.0, help="Latency deadline (s).")
    p.add_argument("--lag-mu-fog", type=float, default=50.0, help="Fog capacity (updates/s).")
    p.add_argument("--lag-eta", type=float, default=200.0, help="Privacy budget (MI cap).")
    p.add_argument("--lag-eta-dual", type=float, default=0.05, help="Dual step size.")
    p.add_argument("--lag-clip", type=float, default=10.0, help="Reward clip.")
    p.add_argument(
        "--cloud-policy",
        choices=("static", "round_robin", "d3qn"),
        default="static",
        help="Cloud-level policy: static, round-robin, or D3QN-driven switching.",
    )
    p.add_argument(
        "--cloud-aggregators",
        type=str,
        default="fedavg,krum,multi_krum,trimmed_mean,median",
        help="Comma-separated aggregator menu for the D3QN cloud policy.",
    )
    p.add_argument(
        "--d3qn-eps-decay",
        type=int,
        default=100,
        help="Epsilon decay steps for D3QN exploration.",
    )
    p.add_argument(
        "--d3qn-eps-end",
        type=float,
        default=0.02,
        help="Final exploration rate after decay (was 0.05; lower keeps "
             "the late-round trace de-noised when total rounds are short).",
    )
    p.add_argument(
        "--d3qn-eps-exponential",
        action="store_true",
        help="Use geometric (exponential) epsilon decay instead of linear. "
             "Recommended for short runs (<=200 rounds): epsilon drops below "
             "0.2 within the first ~20%% of decay steps.",
    )
    p.add_argument(
        "--fog-capacity",
        type=float,
        default=None,
        help="Per-fog capacity (mu_fog, updates/s) used in g_cap. Default 50.",
    )
    p.add_argument(
        "--fog-deadline",
        type=float,
        default=None,
        help="Per-fog round deadline (delta, s) used in g_lat. Default 5.",
    )
    p.add_argument(
        "--privacy-epsilon",
        type=float,
        default=0.5,
        help="DP epsilon used in the MI proxy.",
    )
    p.add_argument(
        "--privacy-eta",
        type=float,
        default=1.0,
        help="Privacy budget (eta); g_priv = max(0, I_est - eta).",
    )
    p.add_argument(
        "--enable-random-drops",
        action="store_true",
        help="Enable per-client Bernoulli dropouts using drop_prob from the device profile.",
    )
    p.add_argument(
        "--drop-late",
        action="store_true",
        help="Drop clients whose per-client estimated time exceeds the fog deadline.",
    )
    p.add_argument(
        "--device-noise",
        type=float,
        default=0.0,
        help="Per-round multiplicative lognormal jitter on lambda_i + record_size_kb. "
        "0 = deterministic profile; 0.1 = ~+/-10%% per round.",
    )
    p.add_argument(
        "--attack",
        choices=("none", "label_flip", "model_scale", "gaussian", "mixed"),
        default="none",
        help="Adversarial behaviour. 'mixed' assigns each malicious client one "
        "of {label_flip, model_scale, gaussian} at cohort construction "
        "(paper's mixed-attack scenario); requires --attack-stepwise or "
        "--attack-ramp to define activation.",
    )
    p.add_argument("--attack-frac", type=float, default=0.0,
                   help="Fraction of clients made malicious (sampled once per seed).")
    p.add_argument("--attack-start-round", type=int, default=0,
                   help="First round adversaries are active (inclusive).")
    p.add_argument("--attack-end-round", type=int, default=-1,
                   help="Last round adversaries are active (exclusive); -1 = until end.")
    p.add_argument("--attack-shift", type=int, default=1,
                   help="label_flip: cyclic shift (c -> (c+shift) %% num_classes).")
    p.add_argument("--attack-gamma", type=float, default=10.0,
                   help="model_scale: multiplier on the delta (use <0 for sign-flip).")
    p.add_argument("--attack-sigma", type=float, default=1.5,
                   help="gaussian: stddev of noise added to the delta. "
                        "Default 1.5 (was 0.5; too weak to dent FedAvg on N-BaIoT).")
    p.add_argument(
        "--attack-ramp",
        type=float,
        nargs=2,
        metavar=("EARLY_FRAC", "LATE_FRAC"),
        default=None,
        help="(PR-16) Escalating attack: cohort grows linearly from "
        "EARLY_FRAC*N clients in round 0 to LATE_FRAC*N in the final "
        "round. Overrides --attack-frac and --attack-start-round / "
        "--attack-end-round. Use 0.0 0.5 for a 0%% -> 50%% ramp.",
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
        help="Secure channel: 'plain' (no-op, fast) or 'paillier' "
        "(additively HE). When 'paillier' the simulator runs a one-shot "
        "calibration of encrypt/sign/verify wall-times and uses the "
        "measured values for cost_sec, instead of synthetic profile draws.",
    )
    p.add_argument(
        "--paillier-keybits",
        type=int,
        default=1024,
        help="Paillier key length when --encryption paillier. Default 1024 "
        "(fast for smoke tests); paper canonical is 2048.",
    )
    p.add_argument(
        "--sig-alg",
        choices=("hmac", "ecdsa", "mldsa"),
        default="hmac",
        help="Update authenticity: hmac (default), ecdsa (P-256 classical), "
        "mldsa (ML-DSA-65 via liboqs; requires pip install -e '.[pqc]').",
    )
    p.add_argument(
        "--sac-heuristic-hint",
        action="store_true",
        help="(A5) Append the heuristic policy's per-client score to the "
             "SAC observation. Gives SAC the heuristic's strong prior on "
             "adversarial cohorts; pair with --adversary-features.",
    )
    p.add_argument(
        "--adversary-features",
        action="store_true",
        help="(PR-14) Compute per-client adversary-detection features "
        "(delta_norm_ratio, cos_dist_to_mean, loss_zscore) and feed them "
        "into the fog policy observation. Lets SAC/heuristic learn to "
        "avoid label-flip / model-scale adversaries. Adds one decryption "
        "per accepted update per round (cheap under plaintext channel, "
        "non-trivial under Paillier).",
    )
    args = p.parse_args()

    if args.dataset == "synthetic":
        client_train_sets, test_set, meta = _build_synthetic(
            args.clients, args.samples_per_client, args.seed
        )
        num_clients = args.clients
    elif args.dataset == "nbaiot":
        resolved = _resolve_nbaiot_root(args.data_dir)
        if resolved is not None:
            print(f"[run_simulation] using cached N-BaIoT at {resolved}")
        else:
            print(
                f"[run_simulation] no N-BaIoT under {args.data_dir!s}; "
                "the loader will download ~600 MB. Re-run with "
                "--data-dir <parent_of_N-baIOT> or set SAFEL_DT_DATA_DIR "
                "to use a local copy."
            )
        nbaiot_kwargs: dict[str, object] = dict(
            mode=args.mode,
            max_per_class=args.max_per_class,
            seed=args.seed,
        )
        if args.clients and args.clients > 0:
            nbaiot_kwargs["num_clients"] = args.clients
        client_train_sets, test_set, meta = load_dataset(
            "nbaiot", args.data_dir, **nbaiot_kwargs,
        )
        num_clients = len(client_train_sets)
    elif args.dataset == "edge_iiotset":
        client_train_sets, test_set, meta = load_dataset(
            "edge_iiotset",
            args.data_dir,
            mode=args.mode,
            num_clients=args.clients,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        num_clients = len(client_train_sets)
    else:  # toniot
        client_train_sets, test_set, meta = load_dataset(
            "toniot",
            args.data_dir,
            mode=args.mode,
            num_clients=args.clients,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        num_clients = len(client_train_sets)

    in_features = int(meta["in_features"])
    num_classes = int(meta["num_classes"])
    client_to_fog = assign_clients_to_fogs(num_clients, args.fogs)

    fog_policies: dict[int, FogPolicy] | None
    if args.policy == "all":
        fog_policies = None
    elif args.policy == "random":
        k = args.random_k if args.random_k is not None else max(
            1, len(client_to_fog[next(iter(client_to_fog))]) // 2
        )
        fog_policies = {
            fid: RandomPolicy(num_clients=len(cids), k=min(k, len(cids)), seed=args.seed + fid)
            for fid, cids in client_to_fog.items()
        }
    elif args.policy == "heuristic":
        def _heur_mu_fog(cids: list[int]) -> int:
            return args.heur_mu_fog or max(1, len(cids) // 2)

        fog_policies = {
            fid: HeuristicFogPolicy(
                num_clients=len(cids),
                client_ids=list(cids),
                n_samples_per_client=[len(client_train_sets[c]) for c in cids],
                rounds_total=args.rounds,
                cfg=HeuristicConfig(
                    mu_fog=min(_heur_mu_fog(cids), len(cids)),
                    explore_prob=args.heur_explore,
                    seed=args.seed + fid,
                ),
                adversary_features=args.adversary_features,
            )
            for fid, cids in client_to_fog.items()
        }
    elif args.policy == "binary_rl":
        def _br_mu_fog(cids: list[int]) -> int:
            return args.binary_rl_mu_fog or max(1, len(cids) // 2)

        fog_policies = {
            fid: BinaryRLFogPolicy(
                num_clients=len(cids),
                client_ids=list(cids),
                n_samples_per_client=[len(client_train_sets[c]) for c in cids],
                rounds_total=args.rounds,
                cfg=BinaryRLConfig(
                    mu_fog=min(_br_mu_fog(cids), len(cids)),
                    alpha=args.binary_rl_alpha,
                    use_fog_reward=args.binary_rl_use_fog_reward,
                    seed=args.seed + fid,
                ),
            )
            for fid, cids in client_to_fog.items()
        }
    else:  # sac
        sel = SelectionConfig(tau=args.sac_tau, mu_fog=args.sac_mu_fog, m_min=args.sac_m_min)
        sac_cfg = SacControllerConfig(seed=args.seed)
        fog_policies = {
            fid: SacPolicy(
                num_clients=len(cids),
                client_ids=list(cids),
                n_samples_per_client=[len(client_train_sets[c]) for c in cids],
                rounds_total=args.rounds,
                selection=sel,
                sac_cfg=sac_cfg,
                adversary_features=args.adversary_features,
                heuristic_hint=args.sac_heuristic_hint,
            )
            for fid, cids in client_to_fog.items()
        }

    cloud_policy: CloudPolicy | None
    if args.cloud_policy == "d3qn":
        aggregators = tuple(a.strip() for a in args.cloud_aggregators.split(",") if a.strip())
        obs_cfg = CloudObsConfig(
            fog_ids=tuple(sorted(client_to_fog.keys())),
            aggregators=aggregators,
        )
        cloud_policy = cast(
            CloudPolicy,
            D3qnCloudPolicy(
                obs_cfg=obs_cfg,
                d3qn_cfg=D3qnConfig(
                    epsilon_decay_steps=args.d3qn_eps_decay,
                    epsilon_end=args.d3qn_eps_end,
                    epsilon_exponential=args.d3qn_eps_exponential,
                    seed=args.seed,
                ),
            ),
        )
    elif args.cloud_policy == "round_robin":
        aggregators = tuple(a.strip() for a in args.cloud_aggregators.split(",") if a.strip())
        cloud_policy = RoundRobinCloudPolicy(aggregators=aggregators)
    elif args.cloud_policy == "static":
        cloud_policy = None
    else:
        cloud_policy = StaticCloudPolicy(name="fedavg")

    lagrangian: LagrangianConfig | None = None
    if args.reward == "lagrangian":
        lagrangian = LagrangianConfig(
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

    fog_states = None
    if args.fog_capacity is not None or args.fog_deadline is not None:
        from safel_dt.types import FogDTState
        mu_fog = args.fog_capacity if args.fog_capacity is not None else 50.0
        delta = args.fog_deadline if args.fog_deadline is not None else 5.0
        fog_states = {
            fid: FogDTState(fog_id=fid, device_ids=list(cids), mu_fog=mu_fog, delta=delta)
            for fid, cids in client_to_fog.items()
        }

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
    schedule, malicious = build_schedule(
        spec=attack_spec,
        num_classes=num_classes,
        client_ids=list(range(len(client_train_sets))),
        seed=args.seed,
        rounds_total=args.rounds,
    )
    if malicious:
        if attack_spec.is_ramped:
            print(
                f"[run_simulation] attack={attack_spec.name} ramp "
                f"{ramp_early:.2f}->{ramp_late:.2f} over {args.rounds} rounds "
                f"(cohort_ids={malicious})"
            )
        else:
            print(
                f"[run_simulation] attack={attack_spec.name} on {len(malicious)} "
                f"clients (ids={malicious}), rounds[{attack_spec.start_round},"
                f"{attack_spec.end_round if attack_spec.end_round is not None else 'end'})"
            )

    cfg = SimulatorConfig(
        seed=args.seed,
        rounds=args.rounds,
        model_factory=lambda: make_model(
            "mlp",
            in_features=in_features,
            hidden=args.hidden,
            num_classes=num_classes,
        ),
        client_train_sets=client_train_sets,
        client_to_fog=client_to_fog,
        test_set=test_set,
        train_cfg=LocalTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
        ),
        trace_path=args.trace,
        fog_policies=fog_policies,
        lagrangian=lagrangian,
        timing=TimingCoefficients(),
        privacy=PrivacyConfig(epsilon=args.privacy_epsilon, eta=args.privacy_eta),
        fog_states=fog_states,
        cloud_policy=cloud_policy,
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
    print(
        f"[run_simulation] dataset={args.dataset} clients={num_clients} fogs={args.fogs} "
        f"in_features={in_features} num_classes={num_classes} policy={args.policy} "
        f"reward={args.reward} cloud_policy={args.cloud_policy} "
        f"encryption={args.encryption}"
        + (f":{args.paillier_keybits}b" if args.encryption == "paillier" else "")
        + f" sig_alg={args.sig_alg}"
        + (" adversary_features=on" if args.adversary_features else "")
    )
    outcomes = run_simulation(cfg)
    rewards_by_round: dict[int, float] = {}
    if args.trace is not None and args.trace.exists():
        from safel_dt.eval.aggregate import iter_jsonl

        for row in iter_jsonl(args.trace):
            rewards_by_round[int(row.get("round_idx", -1))] = float(row.get("cloud_reward", 0.0))
    for o in outcomes:
        r = rewards_by_round.get(o.round_idx)
        reward_str = f"reward={r:+.3f}  " if r is not None else ""
        print(
            f"round={o.round_idx:>3}  acc={o.accuracy:.3f}  loss={o.loss:.3f}  "
            f"{reward_str}clients_ok={o.n_clients_accepted}  fogs_ok={o.n_fogs_accepted}  "
            f"agg={o.aggregator}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
