"""End-to-end smoke for the multi-seed sweep + aggregation pipeline.

Runs a tiny synthetic sweep through ``scripts/run_sweep.main`` and
verifies:

* The output folder contains the expected JSONL files (one per
  ``(fog_policy, cloud_policy, seed)`` cell).
* ``analysis/aggregate_runs.main`` consumes them and writes
  ``summary.csv`` + ``summary_by_policy.csv`` with the right row
  cardinality.

Plot generation is *not* exercised here (matplotlib import is slow and
not the point of this test).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import scripts.run_sweep as run_sweep

from safel_dt.eval.aggregate import (
    aggregate_run_dir,
    write_by_policy_csv,
    write_summary_csv,
)


@pytest.mark.slow
def test_sweep_then_aggregate_round_trip(tmp_path: Path) -> None:
    out_dir = tmp_path / "sweep_pipeline"
    rc = run_sweep.main([
        "--dataset", "synthetic",
        "--rounds", "2",
        "--seeds", "0,1",
        "--fog-policies", "all,random",
        "--cloud-policies", "static",
        "--clients", "4",
        "--fogs", "2",
        "--samples-per-client", "60",
        "--epochs", "1",
        "--out", str(out_dir),
    ])
    assert rc == 0

    jsonls = sorted(p.name for p in out_dir.glob("*.jsonl"))
    assert jsonls == [
        "all__static__seed0000.jsonl",
        "all__static__seed0001.jsonl",
        "random__static__seed0000.jsonl",
        "random__static__seed0001.jsonl",
    ]
    for p in out_dir.glob("*.jsonl"):
        with p.open(encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        assert len(lines) == 2, f"{p.name} should have 2 rounds, got {len(lines)}"

    summaries = aggregate_run_dir(out_dir)
    assert len(summaries) == 4
    summary_csv = out_dir / "summary.csv"
    bypolicy_csv = out_dir / "summary_by_policy.csv"
    write_summary_csv(summaries, summary_csv)
    write_by_policy_csv(summaries, bypolicy_csv)

    with summary_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert {(r["fog_policy"], r["seed"]) for r in rows} == {
        ("all", "0"), ("all", "1"), ("random", "0"), ("random", "1"),
    }
    with bypolicy_csv.open(encoding="utf-8") as fh:
        bp = list(csv.DictReader(fh))
    assert len(bp) == 2
    for r in bp:
        assert int(r["n_seeds"]) == 2
        assert 0.0 <= float(r["final_acc_mean"]) <= 1.0
