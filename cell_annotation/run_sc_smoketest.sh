#!/bin/bash
#SBATCH --job-name=sc_smoketest
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/CellAnnot/logs/sc_smoketest_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/CellAnnot/logs/sc_smoketest_%j.err
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
# ========================================================================================
# ALS_Xenium repo commentary | cell_annotation/run_sc_smoketest.sh
#
# Single-sample end-to-end check, clustering then figures, with no language model call: 8
# CPUs, 96G, 3 hours.
#
# This is the one that caught the uniformly-False qc_flag bug before it reached the full
# cohort. Run it after any change to sc_cluster.py.
# ========================================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/cell_annotator_env/bin/python
D=/home/rodrigok/SLURM_jobs/Spatial/CellAnnot
SAMPLE=${1:-SD03522_BG}
RES=${RESOLUTION:-0.5}
OUTBASE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/CellAnnot
CLUST=$OUTBASE/clustered_${SAMPLE}_res${RES}.h5ad
FIGDIR=$OUTBASE/figs/${SAMPLE}_res${RES}
mkdir -p "$OUTBASE" "$FIGDIR"
echo "START $(date) sample=$SAMPLE res=$RES on $(hostname)"
echo "=== CLUSTER ==="
$PY $D/sc_cluster.py --sample "$SAMPLE" --out "$CLUST" --resolution "$RES"
echo "=== FIGURES ==="
$PY $D/sc_figs.py --input "$CLUST" --outdir "$FIGDIR" --tag "$SAMPLE"
echo "DONE $(date). Figures in: $FIGDIR"
