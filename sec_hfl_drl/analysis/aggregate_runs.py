"""CLI wrapper around :mod:`safel_dt.eval.aggregate`.

Reads a folder of per-run JSONL traces produced by
``scripts/run_sweep.py`` and writes ``summary.csv``,
``summary_by_policy.csv``, and (optionally) ``accuracy.png`` /
``loss.png``.

Example::

    python analysis/aggregate_runs.py results/runs/sweep_2026-05-26 --plot
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from safel_dt.eval.aggregate import (
    aggregate_run_dir,
    plot_learning_curves,
    write_by_policy_csv,
    write_summary_csv,
)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="Folder containing the *.jsonl traces.")
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write summary.csv / summary_by_policy.csv. Defaults to run_dir.",
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Also dump accuracy.png / loss.png learning-curve PNGs.",
    )
    args = p.parse_args(argv)

    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir {run_dir!s} is not a directory.")
    out_dir = args.out_dir or run_dir

    summaries = aggregate_run_dir(run_dir)
    if not summaries:
        print(f"[aggregate_runs] no recognisable *.jsonl files under {run_dir}")
        return 1
    write_summary_csv(summaries, out_dir / "summary.csv")
    write_by_policy_csv(summaries, out_dir / "summary_by_policy.csv")
    print(
        f"[aggregate_runs] wrote {len(summaries)} run summaries -> "
        f"{out_dir / 'summary.csv'}"
    )
    if args.plot:
        plot_learning_curves(run_dir, out_dir)
        print(f"[aggregate_runs] wrote learning-curve plots -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
