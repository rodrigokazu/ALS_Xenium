#!/bin/bash
#SBATCH --job-name=gene_coloc_FINAL
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --array=0-2%3
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_gene_coloc_FINAL.sh
#
# Runs the co-localisation quantification as a three-task array, one gene each for MNX1, BCL6
# and STMN2: 4 CPUs, 110G, 5 hours per task.
#
# The genes come from small index files sitting next to the script, so the array is 0-2
# instead of being parameterised on the command line.
#
# Reads transcripts.parquet. On _final that file is complete and in the integer namespace. So
# the co-localisation numbers are valid right now even though the polygon figures are not; the
# boundary problem affects only the filled-polygon rendering.
# ========================================================================================

# ^ one array task per MN-coloc gene (MNX1, BCL6, STMN2); all tasks write the SAME OUT dir.
#   Direct _FINAL analog of the _gausss run_gene_coloc.sh (which had NO staged _FINAL twin).
#   Env/logs/caches/exit-code mirror run_overlays_FINAL.sh.
#
#   AFTER all 3 array tasks finish, build the comparison figure (light; run interactively):
#     PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/xenium_vistools/bin/python
#     PYTHONNOUSERSITE=1 $PY gene_coloc_compare_fig_FINAL.py <OUT> \
#         --lowq <_final dual-pass-QC REMOVE/REVIEW labels>   # gauss-era SD01620_BI does NOT carry
#
#   DO NOT SUBMIT until BOTH hold, else the run is a clean no-op / zero-gated:
#     (a) the Novae _FINAL per-sample h5ads exist under NOVAE_JOINT_DIR/per_sample
#         (missing -> every sample SKIPs, no stats written); AND
#     (b) Marcel's _final is_MN annotation has landed (missing -> is_MN.sum()==0 everywhere ->
#         stats CSV written all-zero, is_MN_pending=1, NO heroes rendered; see gene_coloc_hero_FINAL.py).

set -uo pipefail
CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/xenium_vistools/bin/python

GENES=(MNX1 BCL6 STMN2)
GENE="${GENES[${SLURM_ARRAY_TASK_ID:-0}]}"
# OUT overridable as $1; default = a gene_coloc subdir of the Novae JOINT _FINAL output.
OUT="${1:-/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_niches_FINAL/gene_coloc}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
# route all caches to node-local scratch (never $HOME/OAK -- quota is full)
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$OUT"

echo "Job ${SLURM_JOB_NAME} array task ${SLURM_ARRAY_TASK_ID:-0} (gene=${GENE}) started: $(date) on $(hostname)"
echo "OUT=${OUT}"
$PY "$CODE/gene_coloc_hero_FINAL.py" --gene "$GENE" --out "$OUT" "${@:2}"
rc=$?
echo "Job finished: $(date) (exit $rc)"
exit $rc
