#!/bin/bash
# Base segmentation of one 4096 px chunk of the full section, C500 settings (identical to run_500.sh C500).
# usage: run_chunk.sh <name> <x0 y0 x1 y1>; all chunks: xargs -P 7 -L 1 run_chunk.sh < chunks.txt  (make_chunks.py writes chunks.txt)
set -uo pipefail
NAME=$1; X0=$2; Y0=$3; X1=$4; Y1=$5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=~/miniforge3/envs/spatial/bin/python; CODE=$(cd "$(dirname "$0")/../scripts" && pwd)
ROOT=${ATERA_ROOT:-"/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"}; B="$ROOT/Cerebellum_sample"; DENS="$ROOT/work/density/tx_density.zarr"
FULL=${ATERA_FULL:-./full}; OUT=$FULL/base/$NAME; mkdir -p $FULL/logs
[ -f "$OUT/cells.parquet" ] && { echo "skip $NAME"; exit 0; }
$PY "$CODE/run_fullsample.py" --bundle "$B" --sample Cerebellum_sample --out "$OUT" --workers 1 --window $X0 $Y0 $X1 $Y1 --margin 400 \
  --stats "$(dirname "$0")/../config/stats_density.json" --density "$DENS" --density-tissue-thr 10 --param glia_soma_max_um=8 \
  --combined "48,941,0.459,5.815,177.18,0.793" --stat thr_alt=0.6 --param thr_value=0.5 --param glia_soma_thr_value=0.2 \
  > $FULL/logs/$NAME.log 2>&1 && echo "done $NAME" || echo "FAILED $NAME"
