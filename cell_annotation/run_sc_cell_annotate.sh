#!/bin/bash
#SBATCH --job-name=sc_annotate
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/CellAnnot/logs/sc_annotate_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/CellAnnot/logs/sc_annotate_%j.err
# ========================================================================================
# ALS_Xenium repo commentary | cell_annotation/run_sc_cell_annotate.sh
#
# Annotation job: 8 CPUs, 128G, 4 hours. Modest, because the work is a handful of API calls
# and some bookkeeping rather than computation.
#
# Needs ANTHROPIC_API_KEY in the environment. Run with --dry-run first to see the marker lists
# that would be sent.
# ========================================================================================

#
# Run cell-annotator (LLM cluster labelling, Anthropic/Claude backend) on the
# clustered .h5ad produced by run_sc_cluster.sh.
#
# Usage:
#   sbatch run_sc_cell_annotate.sh <clustered.h5ad> [--dry-run]
# Any extra args after the input path are forwarded to sc_cell_annotate.py.
#
# ANTHROPIC_API_KEY must be in the environment or in ~/.env (or CELL_ANNOTATOR_ENV_FILE);
# the script fails loudly if it is absent. Only external egress = per-cluster marker
# lists + expected-types prompt to Anthropic.

set -euo pipefail

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/cell_annotator_env/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=/home/rodrigok/SLURM_jobs/Spatial/CellAnnot/logs
mkdir -p "$LOGDIR"

if [[ $# -lt 1 ]]; then
    echo "FATAL: need the clustered .h5ad path as first argument." >&2
    echo "Usage: sbatch run_sc_cell_annotate.sh <clustered.h5ad> [--dry-run]" >&2
    exit 1
fi

INPUT="$1"; shift
MODEL=${MODEL:-claude-haiku-4-5}

echo "[wrapper] host=$(hostname) date=$(date)"
echo "[wrapper] python=$PY"
echo "[wrapper] input=$INPUT model=$MODEL extra_args=$*"

"$PY" -u "$SCRIPT_DIR/sc_cell_annotate.py" \
    --input "$INPUT" \
    --cluster-key leiden \
    --sample-key sample \
    --species human \
    --tissue "spinal cord" \
    --provider anthropic \
    --model "$MODEL" \
    "$@"

echo "[wrapper] done."
