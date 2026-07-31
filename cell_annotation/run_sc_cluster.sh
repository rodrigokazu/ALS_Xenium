#!/bin/bash
#SBATCH --job-name=sc_cluster
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/CellAnnot/logs/sc_cluster_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/CellAnnot/logs/sc_cluster_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | cell_annotation/run_sc_cluster.sh
#
# Clustering job: 16 CPUs, 256G, 12 hours. Sized for the full 1.88M cell object; the neighbour
# graph is the peak.
# ========================================================================================

#
# QC -> normalize -> PCA -> neighbors -> Leiden on the ALS spinal-cord Xenium concat.
# Everything runs as a job: login-node imports time out.
#
# FULL run (1.88M cells): submit with no --sample.
# FAST test (~94-100k cells): pass a sample, e.g.
#   sbatch --cpus-per-task=8 --mem=64G --time=1:00:00 run_sc_cluster.sh SD_CODE
# where $1 (optional) = a sample/sd_code to subset for the smoke test.

set -euo pipefail

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/cell_annotator_env/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=/home/rodrigok/SLURM_jobs/Spatial/CellAnnot/logs
OUTDIR=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/SC_MNcorrected_h5ads
mkdir -p "$LOGDIR"

INPUT=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/SC_MNcorrected_h5ads/ALS_SCXenium_MNcorrected_gausss_concatenated_offset_allsamples_preQC_20260706.h5ad
RESOLUTION=${RESOLUTION:-1.0}

SAMPLE_ARG=""
TAG="full"
if [[ $# -ge 1 && -n "${1:-}" ]]; then
    SAMPLE_ARG="--sample $1"
    TAG="sample_$1"
fi

OUT="$OUTDIR/ALS_SCXenium_clustered_${TAG}_r${RESOLUTION}.h5ad"

echo "[wrapper] host=$(hostname) date=$(date)"
echo "[wrapper] python=$PY"
echo "[wrapper] input=$INPUT"
echo "[wrapper] out=$OUT"
echo "[wrapper] resolution=$RESOLUTION sample_arg='$SAMPLE_ARG' cpus=${SLURM_CPUS_PER_TASK:-NA}"

"$PY" -u "$SCRIPT_DIR/sc_cluster.py" \
    --input "$INPUT" \
    --out "$OUT" \
    --resolution "$RESOLUTION" \
    $SAMPLE_ARG

echo "[wrapper] done: $OUT"
