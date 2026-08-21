"""Unit tests for `safel_dt.eval.aggregate`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from safel_dt.eval.aggregate import (
    BY_POLICY_METRICS,
    SUMMARY_COLUMNS,
    aggregate_run_dir,
    parse_filename,
    summarise_run,
    write_by_policy_csv,
    write_summary_csv,
)


def _row(round_idx: int, accuracy: float, loss: float, **extras) -> dict:
    base = {"round_idx": round_idx, "accuracy": accuracy, "loss": loss}
    base.update(extras)
    return base


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_parse_filename_canonical() -> None:
    assert parse_filename("sac__d3qn__seed0007.jsonl") == ("sac", "d3qn", 7)


def test_parse_filename_with_extras() -> None:
    assert parse_filename(
        "heuristic__static__seed0042__mu_fog-3.jsonl"
    ) == ("heuristic", "static", 42)


def test_parse_filename_garbage_rejected() -> None:
    assert parse_filename("hello.jsonl") is None
    assert parse_filename("sac_static_seed1.jsonl") is None


def test_summarise_run_empty_returns_zero_summary() -> None:
    s = summarise_run(fog_policy="x", cloud_policy="y", seed=0, rows=[])
    assert s.rounds_completed == 0
    assert s.final_acc == 0.0
    assert s.max_acc == 0.0


def test_summarise_run_picks_finals_and_extrema() -> None:
    rows = [
        _row(0, 0.5, 1.0,
             costs={"0": {"comm": 100.0, "train": 1.0, "sec": 0.5,
                          "g_lat": 0.0, "g_cap": 0.0, "g_priv": 0.0}},
             cloud_reward=0.1),
        _row(1, 0.7, 0.8,
             costs={"0": {"comm": 200.0, "train": 2.0, "sec": 0.5,
                          "g_lat": 0.2, "g_cap": 0.0, "g_priv": 0.0}},
             cloud_reward=0.3),
        _row(2, 0.6, 0.9,
             costs={"0": {"comm": 150.0, "train": 1.5, "sec": 0.5,
                          "g_lat": 0.4, "g_cap": 0.1, "g_priv": 0.05}},
             cloud_reward=-0.05,
             multipliers={"lat": 0.4, "cap": 0.2, "priv": 0.0}),
    ]
    s = summarise_run(fog_policy="sac", cloud_policy="d3qn", seed=3, rows=rows)
    assert s.rounds_completed == 3
    assert s.final_acc == 0.6
    assert s.max_acc == 0.7
    assert s.final_loss == 0.9
    assert s.min_loss == 0.8
    assert abs(s.mean_cost_comm - 150.0) < 1e-6
    assert abs(s.mean_cost_train - 1.5) < 1e-6
    assert s.mean_cost_sec == 0.5
    assert s.final_nu_lat == 0.4
    assert s.final_nu_cap == 0.2
    assert s.final_nu_priv == 0.0
    assert abs(s.mean_cloud_reward - (0.1 + 0.3 - 0.05) / 3) < 1e-6
    assert s.final_cloud_reward == -0.05
    assert abs(s.mean_g_lat - 0.2) < 1e-6
    assert abs(s.mean_g_cap - (0.1 / 3)) < 1e-6
    assert abs(s.mean_g_priv - (0.05 / 3)) < 1e-6
    assert s.mean_dropped_random == 0.0
    assert s.mean_dropped_late == 0.0


def test_summarise_run_counts_dropped_clients() -> None:
    rows = [
        _row(0, 0.5, 1.0, dropped_random={"0": [3, 5], "1": []}, dropped_late={"0": []}),
        _row(1, 0.6, 0.9, dropped_random={"0": [], "1": [7]}, dropped_late={"0": [1]}),
    ]
    s = summarise_run(fog_policy="all", cloud_policy="static", seed=0, rows=rows)
    assert s.mean_dropped_random == 1.5  # (2 + 1) / 2
    assert s.mean_dropped_late == 0.5    # (0 + 1) / 2


def test_summarise_run_costs_sum_across_fogs() -> None:
    rows = [
        _row(0, 0.1, 1.0, costs={
            "0": {"comm": 100.0, "train": 1.0, "sec": 0.5},
            "1": {"comm": 200.0, "train": 2.0, "sec": 0.5},
        }),
    ]
    s = summarise_run(fog_policy="all", cloud_policy="static", seed=0, rows=rows)
    assert s.mean_cost_comm == 300.0
    assert s.mean_cost_train == 3.0
    assert s.mean_cost_sec == 1.0


def test_aggregate_run_dir_skips_unrecognised_files(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "all__static__seed0000.jsonl", [_row(0, 0.5, 1.0)])
    _write_jsonl(tmp_path / "random__static__seed0001.jsonl", [_row(0, 0.6, 0.9)])
    (tmp_path / "garbage.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "weird.jsonl").write_text("not_matching\n", encoding="utf-8")
    out = aggregate_run_dir(tmp_path)
    assert {(s.fog_policy, s.cloud_policy, s.seed) for s in out} == {
        ("all", "static", 0),
        ("random", "static", 1),
    }


def test_write_summary_csv_header_and_rows(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "sac__static__seed0000.jsonl", [_row(0, 0.5, 1.0)])
    summaries = aggregate_run_dir(tmp_path)
    out = tmp_path / "summary.csv"
    write_summary_csv(summaries, out)
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == list(SUMMARY_COLUMNS)
    assert rows[0]["fog_policy"] == "sac"
    assert rows[0]["seed"] == "0"


def test_write_by_policy_csv_groups_and_aggregates(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "all__static__seed0000.jsonl", [_row(0, 0.4, 1.0)])
    _write_jsonl(tmp_path / "all__static__seed0001.jsonl", [_row(0, 0.6, 1.0)])
    _write_jsonl(tmp_path / "sac__static__seed0000.jsonl", [_row(0, 0.8, 0.5)])
    summaries = aggregate_run_dir(tmp_path)
    out = tmp_path / "summary_by_policy.csv"
    write_by_policy_csv(summaries, out)
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_key = {(r["fog_policy"], r["cloud_policy"]): r for r in rows}
    assert ("all", "static") in by_key and ("sac", "static") in by_key
    all_row = by_key[("all", "static")]
    assert all_row["n_seeds"] == "2"
    assert abs(float(all_row["final_acc_mean"]) - 0.5) < 1e-6
    sac_row = by_key[("sac", "static")]
    assert sac_row["n_seeds"] == "1"
    assert float(sac_row["final_acc_mean"]) == 0.8


def test_by_policy_metric_columns_all_present(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "all__static__seed0000.jsonl", [_row(0, 0.4, 1.0)])
    out = tmp_path / "summary_by_policy.csv"
    write_by_policy_csv(aggregate_run_dir(tmp_path), out)
    with out.open(encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    for m in BY_POLICY_METRICS:
        for stat in ("mean", "std", "min", "max"):
            assert f"{m}_{stat}" in header, f"missing column {m}_{stat}"
