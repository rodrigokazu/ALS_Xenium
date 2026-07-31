#!/bin/bash
#SBATCH --job-name=ct_tsne_FINAL
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_ct_tsne_FINAL.sh
#
# Wrapper for the cohort tSNE: 16 CPUs, 128G, 6 hours, and it needs PYTHONNOUSERSITE=1 plus
# LD_LIBRARY_PATH pointed at the environment's lib directory.
#
# Naming trap. This wraps run_opentsne_FINAL.py. It is a completely different job from
# run_tsne_sweep_FINAL.sh, the astrocyte parameter sweep. The two were confused once already
# during staging and the launch failed on missing filenames.
# ========================================================================================

# openTSNE (FFT-accelerated, CPU) of the whole _FINAL cohort from X_pca.npy (run_opentsne_FINAL.py).
# Adapted from CellTyping_coarse/code/run_tsne.sh. KEEP: opentsne_gpu env + LD_LIBRARY_PATH=$ENV/lib
# (the proven env gotcha). F4 deltas: oak %x_%j logs; caches -> $TMPDIR; PYTHONDONTWRITEBYTECODE;
# exit-code propagated. DO NOT SUBMIT until X_pca.npy exists (written by run_ct_pipe_FINAL.sh);
# run AFTER the pipeline (afterok) so X_pca.npy and X_tsne.npy stay row-aligned.
set -uo pipefail   # NOT -e: capture the python exit code explicitly
ENV=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/opentsne_gpu
CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PY="$ENV/bin/python"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"   # KEEP: opentsne_gpu env lib gotcha
# route all caches to node-local scratch (never $HOME/OAK -- quota is full)
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "start $(date) $(hostname)"
$PY "$CODE/run_opentsne_FINAL.py"
rc=$?
echo "end $(date) exit $rc"
exit $rc
