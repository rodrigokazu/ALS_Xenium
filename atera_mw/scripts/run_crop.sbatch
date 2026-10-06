#!/bin/bash
#SBATCH --job-name=reseg_atera_crop
#SBATCH --output=/home/mw28/spatial_dapi_pipeline/logs/%x_%A_%a.out
#SBATCH --error=/home/mw28/spatial_dapi_pipeline/logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#
# New segmentation (paper_reseg_pipeline/splitmerge/rules.py, unchanged, via atera_reseg/run_fullsample.py --window) on crop windows of the Atera cerebellum,
# with the WHOLE-SAMPLE statistics from the --stats-only run (never crop-local thresholds), before any
# full-sample run. One crop per array task, from a TSV: name x0_px y0_px x1_px y1_px (level-0 pixels).
# Output: _work_reseg_atera/crops/<name>/ (final_labels.npy is full-size but sparse outside the window).
# Submit: sbatch --array=0-2 --export=CROPS=<tsv> atera_reseg/run_crop.sbatch
set -uo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/mw28/miniforge3/envs/spatial/bin/python3
CODE=/home/mw28/spatial_dapi_pipeline/atera_reseg
SPATIAL_ROOT=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial
SAMPLE=${SAMPLE:-Cerebellum_sample}
BUNDLE=$SPATIAL_ROOT/Atera_CB/$SAMPLE
# SIGNAL=density: segment on the transcript-density image (option B) with its own whole-sample stats
SIGNAL=${SIGNAL:-18s}
if [ "$SIGNAL" = density ]; then
  DENS=$SPATIAL_ROOT/_work_reseg_atera/density/tx_density.zarr
  STATS=$SPATIAL_ROOT/_work_reseg_atera/_stats_density/$SAMPLE/stats.json; DENS_ARG="--density $DENS"; SUB=crops_density
else
  STATS=$SPATIAL_ROOT/_work_reseg_atera/_stats/$SAMPLE/stats.json; DENS_ARG=""; SUB=crops
fi
[ -f "$STATS" ] || { echo "FAILED: no whole-sample stats at $STATS (run --stats-only first)"; exit 1; }
read -r NAME X0 Y0 X1 Y1 < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CROPS")
OUT=$SPATIAL_ROOT/_work_reseg_atera/$SUB/$NAME
echo "=== $SAMPLE crop $NAME ($X0,$Y0)-($X1,$Y1) px : start $(date) on $(hostname) ==="
$PY "$CODE/run_fullsample.py" --bundle "$BUNDLE" --sample "$SAMPLE" --out "$OUT" --workers $(( ${SLURM_CPUS_PER_TASK:-4} - 1 )) \
  --window $X0 $Y0 $X1 $Y1 --stats "$STATS" $DENS_ARG || { echo "FAILED run_fullsample: $NAME"; exit 1; }
echo "=== $NAME done $(date) ==="
