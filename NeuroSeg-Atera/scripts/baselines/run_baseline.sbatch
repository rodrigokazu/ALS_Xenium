#!/bin/bash
#SBATCH --job-name=atera_bl
#SBATCH --output=/home/mw28/spatial_dapi_pipeline/logs/%x_%A_%a.out
#SBATCH --error=/home/mw28/spatial_dapi_pipeline/logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#
# Paper baselines on the Atera cerebellum crops (atera_reseg/baselines/, see README.md). One crop per array task
# (line of crops_marcel.tsv); METHOD selects the driver. Resources are given on the sbatch command line per method.
#   sbatch --job-name=atera_voronoi --array=0-2 --export=ALL,METHOD=voronoi run_baseline.sbatch
#   METHOD = stats_ws | stats_cp | voronoi | watershed | cellpose | baysor | minibundle | segger | score
set -uo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE=/home/mw28/spatial_dapi_pipeline/atera_reseg/baselines
BL=/home/mw28/spatial_dapi_pipeline/paper_reseg_pipeline/scg/baselines
PY=/home/mw28/miniforge3/envs/spatial/bin/python3
CROPS=/home/mw28/spatial_dapi_pipeline/atera_reseg/crops_marcel.tsv
WORK=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/_work_reseg_atera/baselines
CROP=""
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then CROP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CROPS" | cut -f1); fi
echo "=== $METHOD ${CROP} : start $(date) on $(hostname) job ${SLURM_JOB_ID:-} ==="
T0=$(date +%s)
case "$METHOD" in
  stats_ws)  $PY $HERE/stats_watershed_threshold.py ;;
  stats_cp)  $PY $HERE/stats_cellpose_norm.py --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 1 )) ;;
  voronoi)   $PY $HERE/voronoi_crop.py --crop "$CROP" ;;
  watershed) $PY $HERE/watershed_crop.py --crop "$CROP" ;;
  cellpose)
    export CELLPOSE_LOCAL_MODELS_PATH=$BL/models
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
    /home/mw28/miniforge3/envs/cellpose_env/bin/python $HERE/cellpose_crop.py --crop "$CROP" ;;
  baysor)
    export LD_LIBRARY_PATH=/home/mw28/tools/baysor/env/lib:${LD_LIBRARY_PATH:-}
    $PY $HERE/baysor_crop.py --crop "$CROP" --baysor /home/mw28/tools/baysor/install/bin/baysor \
      --config /home/mw28/tools/baysor/src/configs/xenium.toml ;;
  minibundle) $PY $HERE/make_minibundle.py --crop "$CROP" ;;
  segger)
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    MB=$WORK/_minibundle/$CROP
    [ -f "$MB/WINDOW_UM" ] || $PY $HERE/make_minibundle.py --crop "$CROP" || exit 1
    OUT=$WORK/segger/$CROP
    # the benchmark's driver, unchanged, on the crop-sized bundle with its own --window (crop +- 25 um)
    /home/mw28/miniforge3/envs/segger_env/bin/python $BL/segger/run_segger_fullsample.py --bundle "$MB" \
      --sample "Atera_$CROP" --out "$OUT" --work "$OUT/_work" --workers "${SLURM_CPUS_PER_TASK:-6}" \
      --window $(cat "$MB/WINDOW_UM") && $PY $HERE/segger_post.py --crop "$CROP" ;;
  score)     $PY $HERE/score_crops.py ;;
  *) echo "unknown METHOD $METHOD"; exit 2 ;;
esac
rc=$?
echo "=== $METHOD ${CROP} : end $(date) rc=$rc wall $(( $(date +%s) - T0 )) s ==="
exit $rc
