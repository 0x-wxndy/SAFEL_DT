import json
from pathlib import Path

for s in (1, 2):
    p = Path(f"results/runs/sweep_headline/sac__d3qn__seed{s:04d}.jsonl")
    if not p.exists():
        print(f"seed {s}: not present"); continue
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    print(f"\n=== seed {s} ({len(rows)} rounds) ===")
    print("  last 5 rounds:")
    for r in rows[-5:]:
        m = r["multipliers"]
        print(
            f"    r={r['round_idx']:3d}  acc={r['accuracy']:.3f}  "
            f"agg={r['aggregator']:>14s}  "
            f"nu=(lat={m['lat']:.2f}, cap={m['cap']:.2f}, priv={m['priv']:.2f})"
        )
    print(f"  full-run nu_priv max: {max(r['multipliers']['priv'] for r in rows):.2f}")
    print(f"  full-run accuracy at r=99/199/end: "
          f"{rows[99]['accuracy']:.3f} / "
          f"{rows[199]['accuracy']:.3f} / "
          f"{rows[-1]['accuracy']:.3f}" if len(rows) >= 200 else
          f"  full-run accuracy at r={len(rows)-1}: {rows[-1]['accuracy']:.3f}")
