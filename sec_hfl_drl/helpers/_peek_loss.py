import json, numpy as np
from pathlib import Path

losses = []
for s in (0, 1, 2):
    p = Path(f"results/runs/sweep_headline/sac__d3qn__seed{s:04d}.jsonl")
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    losses.append(float(np.mean([r["loss"] for r in rows[-10:]])))
print(f"tail loss per seed: {[f'{l:.3f}' for l in losses]}")
print(f"mean: {np.mean(losses):.3f}  std: {np.std(losses, ddof=1):.3f}")
