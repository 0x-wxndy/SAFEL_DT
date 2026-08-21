"""Importable library for aggregating sweep JSONL traces.

Used by ``analysis/aggregate_runs.py`` (the CLI) and by the tests.
Kept separate from the script so tests don't need ``argparse``-level
plumbing.

Per-run files follow the naming convention emitted by
``scripts/run_sweep.py``::

    <fog_policy>__<cloud_policy>__seed<NNNN>[__<extras>].jsonl
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

_FILENAME_RE: re.Pattern[str] = re.compile(
    r"^(?P<fog>[a-z0-9_]+)__(?P<cloud>[a-z0-9_]+)__seed(?P<seed>\d+)"
    r"(?:__.*)?\.jsonl$"
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "fog_policy",
    "cloud_policy",
    "seed",
    "rounds_completed",
    "final_acc",
    "max_acc",
    "final_loss",
    "min_loss",
    "mean_cost_comm",
    "mean_cost_train",
    "mean_cost_sec",
    "mean_cloud_reward",
    "final_cloud_reward",
    "mean_g_lat",
    "mean_g_cap",
    "mean_g_priv",
    "final_nu_lat",
    "final_nu_cap",
    "final_nu_priv",
    "mean_dropped_random",
    "mean_dropped_late",
    "attack_name",
    "n_malicious",
)

BY_POLICY_METRICS: tuple[str, ...] = (
    "final_acc",
    "max_acc",
    "final_loss",
    "min_loss",
    "mean_cost_comm",
    "mean_cost_train",
    "mean_cost_sec",
    "mean_cloud_reward",
    "final_cloud_reward",
    "mean_g_lat",
    "mean_g_cap",
    "mean_g_priv",
    "final_nu_lat",
    "final_nu_cap",
    "final_nu_priv",
    "mean_dropped_random",
    "mean_dropped_late",
)


@dataclass(frozen=True)
class RunSummary:
    """Per-run reduction over the JSONL trace."""

    fog_policy: str
    cloud_policy: str
    seed: int
    rounds_completed: int
    final_acc: float
    max_acc: float
    final_loss: float
    min_loss: float
    mean_cost_comm: float
    mean_cost_train: float
    mean_cost_sec: float
    mean_cloud_reward: float
    final_cloud_reward: float
    mean_g_lat: float
    mean_g_cap: float
    mean_g_priv: float
    final_nu_lat: float
    final_nu_cap: float
    final_nu_priv: float
    mean_dropped_random: float
    mean_dropped_late: float
    attack_name: str
    n_malicious: int

    def as_row(self) -> dict[str, str]:
        return {
            "fog_policy": self.fog_policy,
            "cloud_policy": self.cloud_policy,
            "seed": str(self.seed),
            "rounds_completed": str(self.rounds_completed),
            "final_acc": f"{self.final_acc:.6f}",
            "max_acc": f"{self.max_acc:.6f}",
            "final_loss": f"{self.final_loss:.6f}",
            "min_loss": f"{self.min_loss:.6f}",
            "mean_cost_comm": f"{self.mean_cost_comm:.6f}",
            "mean_cost_train": f"{self.mean_cost_train:.6f}",
            "mean_cost_sec": f"{self.mean_cost_sec:.6f}",
            "mean_cloud_reward": f"{self.mean_cloud_reward:.6f}",
            "final_cloud_reward": f"{self.final_cloud_reward:.6f}",
            "mean_g_lat": f"{self.mean_g_lat:.6f}",
            "mean_g_cap": f"{self.mean_g_cap:.6f}",
            "mean_g_priv": f"{self.mean_g_priv:.6f}",
            "final_nu_lat": f"{self.final_nu_lat:.6f}",
            "final_nu_cap": f"{self.final_nu_cap:.6f}",
            "final_nu_priv": f"{self.final_nu_priv:.6f}",
            "mean_dropped_random": f"{self.mean_dropped_random:.6f}",
            "mean_dropped_late": f"{self.mean_dropped_late:.6f}",
            "attack_name": self.attack_name,
            "n_malicious": str(self.n_malicious),
        }


def parse_filename(name: str) -> tuple[str, str, int] | None:
    """Pull ``(fog, cloud, seed)`` out of the sweep filename convention."""
    m = _FILENAME_RE.match(name)
    if m is None:
        return None
    return m.group("fog"), m.group("cloud"), int(m.group("seed"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield one parsed dict per non-empty line of a JSONL file."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _safe_mean(xs: Sequence[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def _round_cost_total(field: str, row: dict[str, Any]) -> float:
    costs = row.get("costs")
    if not isinstance(costs, dict) or not costs:
        return 0.0
    return float(sum(float(v.get(field, 0.0)) for v in costs.values()))


def _round_violation_mean(field: str, row: dict[str, Any]) -> float:
    """Mean of per-fog ``g_lat`` / ``g_cap`` / ``g_priv`` across fogs."""
    costs = row.get("costs")
    if not isinstance(costs, dict) or not costs:
        return 0.0
    vals = [float(v.get(field, 0.0)) for v in costs.values()]
    return float(sum(vals) / max(len(vals), 1))


def _round_drop_count(field: str, row: dict[str, Any]) -> int:
    """Total number of clients dropped this round under ``field``."""
    drops = row.get(field)
    if not isinstance(drops, dict):
        return 0
    return int(sum(len(v) for v in drops.values() if isinstance(v, list)))


def summarise_run(
    *,
    fog_policy: str,
    cloud_policy: str,
    seed: int,
    rows: list[dict[str, Any]],
) -> RunSummary:
    """Reduce one JSONL trace (list of round records) to a :class:`RunSummary`."""
    if not rows:
        return RunSummary(
            fog_policy=fog_policy,
            cloud_policy=cloud_policy,
            seed=seed,
            rounds_completed=0,
            final_acc=0.0,
            max_acc=0.0,
            final_loss=math.inf,
            min_loss=math.inf,
            mean_cost_comm=0.0,
            mean_cost_train=0.0,
            mean_cost_sec=0.0,
            mean_cloud_reward=0.0,
            final_cloud_reward=0.0,
            mean_g_lat=0.0,
            mean_g_cap=0.0,
            mean_g_priv=0.0,
            final_nu_lat=0.0,
            final_nu_cap=0.0,
            final_nu_priv=0.0,
            mean_dropped_random=0.0,
            mean_dropped_late=0.0,
            attack_name="none",
            n_malicious=0,
        )
    accs = [float(r.get("accuracy", 0.0)) for r in rows]
    losses = [float(r.get("loss", math.inf)) for r in rows]
    comms = [_round_cost_total("comm", r) for r in rows]
    trains = [_round_cost_total("train", r) for r in rows]
    secs = [_round_cost_total("sec", r) for r in rows]
    rewards = [float(r.get("cloud_reward", 0.0)) for r in rows]
    g_lats = [_round_violation_mean("g_lat", r) for r in rows]
    g_caps = [_round_violation_mean("g_cap", r) for r in rows]
    g_privs = [_round_violation_mean("g_priv", r) for r in rows]
    drop_r = [_round_drop_count("dropped_random", r) for r in rows]
    drop_l = [_round_drop_count("dropped_late", r) for r in rows]
    last_mults = rows[-1].get("multipliers") or {}
    return RunSummary(
        fog_policy=fog_policy,
        cloud_policy=cloud_policy,
        seed=seed,
        rounds_completed=len(rows),
        final_acc=accs[-1],
        max_acc=max(accs),
        final_loss=losses[-1],
        min_loss=min(losses),
        mean_cost_comm=_safe_mean(comms),
        mean_cost_train=_safe_mean(trains),
        mean_cost_sec=_safe_mean(secs),
        mean_cloud_reward=_safe_mean(rewards),
        final_cloud_reward=rewards[-1],
        mean_g_lat=_safe_mean(g_lats),
        mean_g_cap=_safe_mean(g_caps),
        mean_g_priv=_safe_mean(g_privs),
        final_nu_lat=float(last_mults.get("lat", 0.0) or 0.0),
        final_nu_cap=float(last_mults.get("cap", 0.0) or 0.0),
        final_nu_priv=float(last_mults.get("priv", 0.0) or 0.0),
        mean_dropped_random=_safe_mean([float(x) for x in drop_r]),
        mean_dropped_late=_safe_mean([float(x) for x in drop_l]),
        attack_name=str(rows[-1].get("attack_name", "none")),
        n_malicious=len(rows[-1].get("malicious_ids") or []),
    )


def aggregate_run_dir(run_dir: Path) -> list[RunSummary]:
    """Read every ``*.jsonl`` under ``run_dir`` and return per-run summaries."""
    summaries: list[RunSummary] = []
    for path in sorted(run_dir.glob("*.jsonl")):
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        fog, cloud, seed = parsed
        rows = list(iter_jsonl(path))
        summaries.append(
            summarise_run(fog_policy=fog, cloud_policy=cloud, seed=seed, rows=rows)
        )
    return summaries


def write_summary_csv(summaries: Sequence[RunSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SUMMARY_COLUMNS))
        w.writeheader()
        for s in summaries:
            w.writerow(s.as_row())


def write_by_policy_csv(summaries: Sequence[RunSummary], out_path: Path) -> None:
    """Aggregate per-policy mean / std / min / max across seeds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[RunSummary]] = defaultdict(list)
    for s in summaries:
        groups[(s.fog_policy, s.cloud_policy)].append(s)

    fieldnames = ["fog_policy", "cloud_policy", "n_seeds"]
    for m in BY_POLICY_METRICS:
        fieldnames.extend([f"{m}_mean", f"{m}_std", f"{m}_min", f"{m}_max"])

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for (fog, cloud), runs in sorted(groups.items()):
            row: dict[str, str] = {
                "fog_policy": fog,
                "cloud_policy": cloud,
                "n_seeds": str(len(runs)),
            }
            for m in BY_POLICY_METRICS:
                vals = [float(getattr(r, m)) for r in runs]
                row[f"{m}_mean"] = f"{mean(vals):.6f}"
                row[f"{m}_std"] = (
                    f"{pstdev(vals):.6f}" if len(vals) > 1 else "0.000000"
                )
                row[f"{m}_min"] = f"{min(vals):.6f}"
                row[f"{m}_max"] = f"{max(vals):.6f}"
            w.writerow(row)


def plot_learning_curves(run_dir: Path, out_dir: Path) -> None:
    """Dump ``accuracy.png`` and ``loss.png`` (mean ± std across seeds)."""
    import matplotlib.pyplot as plt
    import numpy as np

    series: dict[tuple[str, str], dict[str, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in sorted(run_dir.glob("*.jsonl")):
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        fog, cloud, _seed = parsed
        accs: list[float] = []
        losses: list[float] = []
        for row in iter_jsonl(path):
            accs.append(float(row.get("accuracy", 0.0)))
            losses.append(float(row.get("loss", math.nan)))
        if accs:
            series[(fog, cloud)]["accuracy"].append(accs)
        if losses:
            series[(fog, cloud)]["loss"].append(losses)

    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("accuracy", "loss"):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for (fog, cloud), per_metric in sorted(series.items()):
            runs = per_metric.get(metric)
            if not runs:
                continue
            min_len = min(len(r) for r in runs)
            arr = np.asarray([r[:min_len] for r in runs], dtype=np.float64)
            x = np.arange(min_len)
            m = arr.mean(axis=0)
            s = arr.std(axis=0) if arr.shape[0] > 1 else np.zeros_like(m)
            label = f"{fog}/{cloud}"
            ax.plot(x, m, label=label, linewidth=1.8)
            ax.fill_between(x, m - s, m + s, alpha=0.15)
        ax.set_xlabel("round")
        ax.set_ylabel(metric)
        ax.set_title(f"Per-policy {metric} (mean ± std across seeds)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=140)
        plt.close(fig)


__all__ = [
    "BY_POLICY_METRICS",
    "SUMMARY_COLUMNS",
    "RunSummary",
    "aggregate_run_dir",
    "iter_jsonl",
    "parse_filename",
    "plot_learning_curves",
    "summarise_run",
    "write_by_policy_csv",
    "write_summary_csv",
]
