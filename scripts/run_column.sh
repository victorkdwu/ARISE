#!/usr/bin/env bash
# Full ARISE pipeline for one Table II column:
#   behaviour clustering -> anchor selection -> fine-tuning grid (benign-only + anchor configs) -> summary.
#   scripts/run_column.sh <column> [gpu]        e.g. scripts/run_column.sh horuseye_iot 0
set -euo pipefail
COL=$1
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
PY=${PYTHON:-python}
export ARISE_PRESET=$COL CUDA_VISIBLE_DEVICES=${2:-0}
LOG=results/$COL/logs; mkdir -p $LOG
echo "[$COL] cluster  $(date -Iseconds)"; $PY -u -m arise.cluster > $LOG/cluster.log 2>&1
echo "[$COL] anchor   $(date -Iseconds)"; $PY -u -m arise.anchor  > $LOG/anchor.log 2>&1
echo "[$COL] train    $(date -Iseconds)"
$PY -u -m arise.train --anchor_type all --prefix t3_ --test --grid $(cat scripts/grids/$COL.txt) > $LOG/train.log 2>&1
$PY scripts/summarize.py runs $COL
