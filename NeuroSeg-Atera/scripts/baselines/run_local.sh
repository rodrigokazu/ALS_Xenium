#!/bin/bash
# Local (Mac) runner of the Atera crop baselines; same drivers as run_baseline.sbatch (SCG), local paths via
# ATERA_LOCAL=1 (atera_common.py). At most ~6 cores: every BLAS / OpenMP pool is pinned to 1 thread.
#   bash run_local.sh <method> [crop ...]    method = voronoi | watershed | cellpose | baysor | segger | score | stats_cp
set -uo pipefail
export ATERA_LOCAL=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BL=$REPO/paper_reseg_pipeline/scg/baselines
PY=~/miniforge3/envs/spatial/bin/python
WORK=${ATERA_WORK:-/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local/baselines}
LOGS=$WORK/_logs; mkdir -p "$LOGS"
M=$1; shift
CROPS=("$@"); [ ${#CROPS[@]} -gt 0 ] || CROPS=(spot1_purkinje_left spot2_gl_wm_centre spot3_purkinje_right)
run() {  # method crop
  local m=$1 c=$2 T0=$(date +%s) rc
  echo "=== $m $c start $(date)"
  case $m in
    voronoi)   $PY $HERE/voronoi_crop.py --crop "$c" ;;
    watershed) $PY $HERE/watershed_crop.py --crop "$c" ;;
    cellpose)  $PY $HERE/cellpose_crop.py --crop "$c" --stage prep && \
               ~/miniforge3/envs/cellpose_env/bin/python $HERE/cellpose_crop.py --crop "$c" --stage segment && \
               $PY $HERE/cellpose_crop.py --crop "$c" --stage finish ;;
    baysor)    $PY $HERE/baysor_crop.py --crop "$c" --baysor ~/opt/Baysor/build/user/baysor \
                 --config ~/opt/Baysor/configs/xenium.toml ;;
    segger)
      MB=$WORK/_minibundle/$c
      [ -f "$MB/WINDOW_UM" ] || $PY $HERE/make_minibundle.py --crop "$c" || return 1
      OUT=$WORK/segger/$c
      ~/miniforge3/envs/segger_env/bin/python $BL/segger/run_segger_fullsample.py --bundle "$MB" --sample "Atera_$c" \
        --out "$OUT" --work "$OUT/_work" --workers 4 --window $(cat "$MB/WINDOW_UM") && $PY $HERE/segger_post.py --crop "$c" ;;
    *) echo "unknown method $m"; return 2 ;;
  esac
  rc=$?
  echo "=== $m $c end $(date) rc=$rc wall $(( $(date +%s) - T0 )) s"
  return $rc
}
case $M in
  score)    $PY $HERE/score_crops.py 2>&1 | tee "$LOGS/score.log" ;;
  stats_cp) $PY $HERE/stats_cellpose_norm.py --workers 6 2>&1 | tee "$LOGS/stats_cp.log" ;;
  *) for c in "${CROPS[@]}"; do run "$M" "$c" > "$LOGS/${M}_${c}.log" 2>&1; tail -1 "$LOGS/${M}_${c}.log"; done ;;
esac
