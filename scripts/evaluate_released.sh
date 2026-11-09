#!/usr/bin/env bash
# Evaluate the released base models and ARISE checkpoints on the test split and compare with Table II.
#   scripts/evaluate_released.sh [gpu] [column ...]      (default: all 12 columns)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${1:-0}; shift || true
COLS=("$@"); (( ${#COLS[@]} )) || COLS=($(ls configs | sed 's/\.yaml$//'))
for COL in "${COLS[@]}"; do
  OUT=$ROOT/results/$COL/released; mkdir -p $OUT
  echo "[$COL] $(date -Iseconds)"
  ARISE_PRESET=$COL $PY -u -m arise.test --base --out $OUT/base > $OUT/base.log 2>&1 \
    || { tail -20 $OUT/base.log; exit 1; }
  ARISE_PRESET=$COL $PY -u -m arise.test --ckpt $ROOT/checkpoints/$COL.pkl --out $OUT/arise > $OUT/arise.log 2>&1 \
    || { tail -20 $OUT/arise.log; exit 1; }
done
$PY scripts/summarize.py released "${COLS[@]}"
