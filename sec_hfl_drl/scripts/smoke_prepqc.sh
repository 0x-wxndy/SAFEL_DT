#!/usr/bin/env bash
# Minimal pre-PQC smoke sweep: paper topology, short horizon.
#
# Usage (from sec_hfl_drl/, venv active):
#   bash scripts/smoke_prepqc.sh
#   bash scripts/smoke_prepqc.sh --dataset nbaiot --rounds 20
#   SMOKE_DATASET=nbaiot bash scripts/smoke_prepqc.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

ROUNDS="${SMOKE_ROUNDS:-20}"
DATASET="${SMOKE_DATASET:-synthetic}"
SEED="${SMOKE_SEED:-0}"
DATA_DIR="${SMOKE_DATA_DIR:-results/data}"
OUT="${SMOKE_OUT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rounds) ROUNDS="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUT" ]]; then
  OUT="results/runs/smoke_prepqc_${DATASET}"
fi

# Compress the paper's 300-round stepwise schedule onto [0, ROUNDS).
r1=$(( ROUNDS * 100 / 300 ))
r2=$(( ROUNDS * 200 / 300 ))
r3=$(( ROUNDS * 250 / 300 ))
[[ "$r1" -lt 1 ]] && r1=1
[[ "$r2" -le "$r1" ]] && r2=$(( r1 + 1 ))
[[ "$r3" -le "$r2" ]] && r3=$(( r2 + 1 ))
[[ "$r3" -ge "$ROUNDS" ]] && r3=$(( ROUNDS - 1 ))
STEPWISE="0:0.10,${r1}:0.15,${r2}:0.20,${r3}:0.25"

echo "[smoke_prepqc] dataset=${DATASET} rounds=${ROUNDS} out=${OUT} data_dir=${DATA_DIR}"
echo "[smoke_prepqc] stepwise=${STEPWISE}"

mkdir -p "$OUT"

EXTRA_DS=(--data-dir "$DATA_DIR")
if [[ "$DATASET" == "nbaiot" ]]; then
  EXTRA_DS+=(--mode multi --max-per-class 500 --clients 30)
elif [[ "$DATASET" == "synthetic" ]]; then
  EXTRA_DS+=(--clients 30 --samples-per-client 80)
fi

python -u -m scripts.run_sweep \
  --dataset "$DATASET" \
  --rounds "$ROUNDS" \
  --seeds "$SEED" \
  --fog-policies all,sac \
  --cloud-policies static,d3qn \
  --fogs 3 \
  --mu-fog 5 \
  --m-min 2 \
  --epochs 1 \
  --batch-size 32 \
  --lr 0.005 \
  --momentum 0.9 \
  --reward lagrangian \
  --lag-omega 1.0 --lag-alpha 0.3 --lag-beta 0.3 --lag-gamma 0.2 \
  --lag-delta 5 --lag-mu-fog 50 --lag-eta 200 \
  --lag-eta-dual 0.05 --lag-clip 10 \
  --fog-capacity 50 --fog-deadline 5 \
  --privacy-epsilon 0.5 --privacy-eta 200 \
  --d3qn-eps-decay "$ROUNDS" --d3qn-eps-end 0.02 \
  --attack mixed \
  --attack-stepwise "$STEPWISE" \
  --mixed-gamma-range 10 50 \
  --encryption plain \
  --adversary-features \
  --sac-heuristic-hint \
  --sac-gradient-steps 2 \
  --out "$OUT" \
  "${EXTRA_DS[@]}"

echo "[smoke_prepqc] sweep done → ${OUT}"
echo "[smoke_prepqc] checking paper invariants..."
python -u -m scripts.check_prepqc_vs_paper --trace-dir "$OUT" --rounds "$ROUNDS"
