"""Emit paper-ready summary tables for cloud reward and per-round costs.

For each of the three credible-scale sweeps (label_flip, model_scale,
gaussian) we compute, per (fog × cloud) policy, the seed-aggregated mean
and standard deviation of:

* ``cloud_reward`` — per-round and final-round
* ``cost_comm`` (bytes) — per-round mean
* ``cost_train`` (s) — per-round mean
* ``cost_sec`` (s) — per-round mean

Outputs land in ``results/figures/paper/``:

* ``rewards_table.md``     — markdown summary, paste into the paper
* ``rewards_table.csv``    — same data, CSV
* ``costs_table.md``       — markdown summary
* ``costs_table.csv``      — same data, CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from safel_dt.eval.aggregate import parse_filename

ATTACK_DIRS: dict[str, str] = {
    "label_flip": "sweep_credible_label_flip",
    "model_scale": "sweep_credible_model_scale",
    "gaussian": "sweep_credible_gaussian",
}

POLICY_KEYS: list[tuple[str, str, str]] = [
    ("all", "static", "All clients + FedAvg (no defence)"),
    ("heuristic", "static", "Heuristic + adversary-aware features"),
    ("sac", "d3qn", "Full RL (SAC + D3QN) + adversary-aware features"),
]


def _iter_rounds(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _collect(path: Path) -> dict[str, list[float]]:
    rewards: list[float] = []
    cost_comm: list[float] = []
    cost_train: list[float] = []
    cost_sec: list[float] = []
    for r in _iter_rounds(path):
        rewards.append(float(r.get("cloud_reward", math.nan)))
        cd = r.get("costs") or {}
        c_comm = c_train = c_sec = 0.0
        for v in cd.values():
            c_comm += float(v.get("comm", 0.0))
            c_train += float(v.get("train", 0.0))
            c_sec += float(v.get("sec", 0.0))
        cost_comm.append(c_comm)
        cost_train.append(c_train)
        cost_sec.append(c_sec)
    return {
        "rewards": rewards,
        "cost_comm": cost_comm,
        "cost_train": cost_train,
        "cost_sec": cost_sec,
    }


def _collect_attack(run_dir: Path) -> dict[tuple[str, str], list[dict[str, list[float]]]]:
    out: dict[tuple[str, str], list[dict[str, list[float]]]] = defaultdict(list)
    for path in sorted(run_dir.glob("*.jsonl")):
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        fog, cloud, _seed = parsed
        out[(fog, cloud)].append(_collect(path))
    return out


def _stat(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _fmt_num(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    if abs(v) >= 1e4:
        return f"{v:.2e}"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    if abs(v) >= 1e-3:
        return f"{v:.4f}"
    return f"{v:.2e}"


def build_rewards_table(
    attack_to_by_policy: dict[str, dict[tuple[str, str], list[dict[str, list[float]]]]],
) -> tuple[list[dict[str, str | float]], str]:
    rows: list[dict[str, str | float]] = []
    md_lines = [
        "| Attack | Policy | Mean per-round reward | Final-round reward |",
        "|---|---|---|---|",
    ]
    for attack, by_policy in attack_to_by_policy.items():
        for fog, cloud, label in POLICY_KEYS:
            seeds = by_policy.get((fog, cloud), [])
            if not seeds:
                continue
            per_seed_mean = [float(np.mean(s["rewards"])) for s in seeds]
            per_seed_final = [float(s["rewards"][-1]) for s in seeds]
            mean_m, mean_s = _stat(per_seed_mean)
            final_m, final_s = _stat(per_seed_final)
            rows.append(
                {
                    "attack": attack,
                    "fog_policy": fog,
                    "cloud_policy": cloud,
                    "label": label,
                    "mean_reward_mean": mean_m,
                    "mean_reward_std": mean_s,
                    "final_reward_mean": final_m,
                    "final_reward_std": final_s,
                }
            )
            md_lines.append(
                f"| {attack} | {label} | "
                f"{_fmt_num(mean_m)} ± {_fmt_num(mean_s)} | "
                f"{_fmt_num(final_m)} ± {_fmt_num(final_s)} |"
            )
    return rows, "\n".join(md_lines) + "\n"


def build_costs_table(
    attack_to_by_policy: dict[str, dict[tuple[str, str], list[dict[str, list[float]]]]],
) -> tuple[list[dict[str, str | float]], str]:
    rows: list[dict[str, str | float]] = []
    md_lines = [
        "| Attack | Policy | Comm (bytes/round) | Train (s/round) | Security (s/round) |",
        "|---|---|---|---|---|",
    ]
    for attack, by_policy in attack_to_by_policy.items():
        for fog, cloud, label in POLICY_KEYS:
            seeds = by_policy.get((fog, cloud), [])
            if not seeds:
                continue
            comm_means = [float(np.mean(s["cost_comm"])) for s in seeds]
            train_means = [float(np.mean(s["cost_train"])) for s in seeds]
            sec_means = [float(np.mean(s["cost_sec"])) for s in seeds]
            comm_m, comm_s = _stat(comm_means)
            train_m, train_s = _stat(train_means)
            sec_m, sec_s = _stat(sec_means)
            rows.append(
                {
                    "attack": attack,
                    "fog_policy": fog,
                    "cloud_policy": cloud,
                    "label": label,
                    "comm_mean": comm_m,
                    "comm_std": comm_s,
                    "train_mean": train_m,
                    "train_std": train_s,
                    "sec_mean": sec_m,
                    "sec_std": sec_s,
                }
            )
            md_lines.append(
                f"| {attack} | {label} | "
                f"{_fmt_num(comm_m)} ± {_fmt_num(comm_s)} | "
                f"{_fmt_num(train_m)} ± {_fmt_num(train_s)} | "
                f"{_fmt_num(sec_m)} ± {_fmt_num(sec_s)} |"
            )
    return rows, "\n".join(md_lines) + "\n"


def _write_csv(rows: list[dict[str, str | float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root", type=Path, default=Path("results/runs"),
        help="Parent directory containing the sweep_credible_* folders.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("results/figures/paper"),
        help="Where to write the .md / .csv tables.",
    )
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    attack_to_by_policy: dict[
        str, dict[tuple[str, str], list[dict[str, list[float]]]]
    ] = {}
    for attack, dirname in ATTACK_DIRS.items():
        run_dir = args.runs_root / dirname
        if not run_dir.is_dir():
            print(f"[paper_tables] skipping missing dir: {run_dir}")
            continue
        attack_to_by_policy[attack] = _collect_attack(run_dir)

    rewards_rows, rewards_md = build_rewards_table(attack_to_by_policy)
    costs_rows, costs_md = build_costs_table(attack_to_by_policy)

    (args.out_dir / "rewards_table.md").write_text(
        "# Cloud reward table\n\n"
        "Mean per-round and final-round `cloud_reward`, averaged over 3 seeds.\n\n"
        + rewards_md,
        encoding="utf-8",
    )
    (args.out_dir / "costs_table.md").write_text(
        "# System cost breakdown table\n\n"
        "Mean per-round cost by category, averaged over 3 seeds. "
        "Units differ across categories (bytes for comm, seconds for "
        "train and security).\n\n" + costs_md,
        encoding="utf-8",
    )
    _write_csv(rewards_rows, args.out_dir / "rewards_table.csv")
    _write_csv(costs_rows, args.out_dir / "costs_table.csv")

    print(f"[paper_tables] wrote tables -> {args.out_dir}")
    for f in sorted(args.out_dir.glob("*table*")):
        print(f"  {f}")
    print()
    print("### rewards_table.md preview")
    print(rewards_md)
    print("### costs_table.md preview")
    print(costs_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
