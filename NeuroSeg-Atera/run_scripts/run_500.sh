#!/bin/bash
# Base segmentation of one 500 um window (reconstructed from the run used on 2026-10-01).
# usage: run_500.sh <variant: C500 (final) | Bt60_500 (density only)> <window name> <x0 y0 x1 y1 in level-0 px>
set -uo pipefail
V=$1; CROP=$2; X0=$3; Y0=$4; X1=$5; Y1=$6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=~/miniforge3/envs/spatial/bin/python; CODE=$(cd "$(dirname "$0")/../scripts" && pwd)
ROOT=${ATERA_ROOT:-"/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"}; B="$ROOT/work/local_bundle"; DENS="$ROOT/work/density/tx_density.zarr"
W=${ATERA_RUNS:-./atera_local}; STATS="$(dirname "$0")/../config/stats_density.json"; OUT="$W/crops_$V/$CROP"
if [ "$V" = C500 ]; then
  SIG=(--combined "48,941,0.459,5.815,177.18,0.793" --stat thr_alt=0.6 --param thr_value=0.5 --param glia_soma_thr_value=0.2)
else
  SIG=(--density-bg 5.815 --param glia_soma_thr_value=0.3162)
fi
$PY "$CODE/run_fullsample.py" --bundle "$B" --sample Cerebellum_sample --out "$OUT" --workers 1 --window $X0 $Y0 $X1 $Y1 --margin 400 \
  --stats "$STATS" --density "$DENS" --density-tissue-thr 10 --param glia_soma_max_um=8 "${SIG[@]}" || { echo "FAILED $V $CROP"; exit 1; }
$PY "$CODE/compare_crop.py" --bundle "$B" --run "$OUT" --out "$ROOT/compare/option_$V/$CROP" > /dev/null || { echo "FAILED compare $V $CROP"; exit 1; }
