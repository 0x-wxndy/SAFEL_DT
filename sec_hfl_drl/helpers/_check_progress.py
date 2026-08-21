"""Quick health check on completed cells of the headline sweep."""
from __future__ import annotations
import json
from pathlib import Path

sweep_dir = Path("results/runs/sweep_headline")
cells = sorted(sweep_dir.glob("*.jsonl"))
print(f"cells found: {len(cells)}")
print()
print(f"{'cell':<32s}  {'rounds':>6s}  {'final_acc':>9s}  {'final_loss':>10s}  {'mean_acc_last50':>14s}  {'mal_max':>7s}")
for cell in cells:
    lines = [json.loads(l) for l in cell.read_text(encoding="utf-8").splitlines()]
    if not lines:
        print(f"{cell.stem:<32s}  EMPTY")
        continue
    n = len(lines)
    final = lines[-1]
    last50 = lines[-50:] if n >= 50 else lines
    mean_acc = sum(l["accuracy"] for l in last50) / len(last50)
    mal_max = max(len(l.get("active_malicious_ids", [])) for l in lines)
    print(
        f"{cell.stem:<32s}  {n:>6d}  {final['accuracy']:>9.3f}  {final['loss']:>10.3f}  "
        f"{mean_acc:>14.3f}  {mal_max:>7d}"
    )

# Aggregator pattern for d3qn cells specifically
d3qn_cells = [c for c in cells if "__d3qn__" in c.name]
if d3qn_cells:
    print()
    print("D3QN aggregator share (round >=100, after exploration):")
    for cell in d3qn_cells:
        lines = [json.loads(l) for l in cell.read_text(encoding="utf-8").splitlines()]
        if len(lines) < 100:
            print(f"  {cell.stem}: still in exploration (n={len(lines)})")
            continue
        post = lines[100:]
        agg_counts: dict[str, int] = {}
        for l in post:
            agg_counts[l["cloud_aggregator"]] = agg_counts.get(l["cloud_aggregator"], 0) + 1
        total = sum(agg_counts.values())
        share = {k: f"{v / total:.2%}" for k, v in sorted(agg_counts.items())}
        print(f"  {cell.stem}: {share}")
