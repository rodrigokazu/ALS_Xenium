#!/bin/bash
#SBATCH --job-name=ct_heroes_FINAL
#SBATCH --output=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.out
#SBATCH --error=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_ct_heroes_FINAL.sh
#
# Renders the cell-type hero crops: 8 CPUs, 110G, 3 hours.
#
# It reads the morphology OME-TIFFs, hence the memory. Those images reach the _final tree
# through a symlink chain into Ranger_procd_mw_gauss and then into the original Ranger_procd
# re-segmentation directories, so this job breaks if either of those raw trees is deleted.
# That deletion was proposed once; see the repo README before agreeing to any cleanup of
# Ranger_procd_mw_gauss.
#
# Expect centroid fallback output until the boundary re-export lands.
# ========================================================================================

# Cell-type hero close-ups for the _FINAL cohort (hero_celltype_FINAL.py). Env: xenium_vistools.
# Adapted from CellTyping_coarse/code/run_heroes.sh. F4 deltas: oak %x_%j logs; caches -> $TMPDIR;
# PYTHONDONTWRITEBYTECODE; exit-code propagated.
#
# *** HOLD -- BLOCKED by F1: the _final cell/nucleus boundaries are stale/partial (~10%
# coverage, old gm namespace), so the hero script CANNOT render filled-polygon "Xenium-Explorer"
# heroes yet. It degrades cleanly to a valid cell-type CENTROID map over the real DAPI+18S image
# (no crash). Filled-polygon heroes auto-enable with zero code change once Marcel re-exports
# full-coverage integer-id boundaries. Run only when polygon heroes are needed AND boundaries are
# fixed; the centroid-fallback figures can be produced now if useful. Requires obs_celltype.csv
# (written by run_ct_pipe_FINAL.sh) to exist. ***
set -uo pipefail   # NOT -e: capture the python exit code explicitly
CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/xenium_vistools/bin/python
OUT=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/CellTyping_coarse_FINAL/celltype_heroes

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
# route all caches to node-local scratch (never $HOME/OAK -- quota is full)
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$OUT"

echo "start $(date) $(hostname)"
$PY "$CODE/hero_celltype_FINAL.py" --out "$OUT" "$@"
rc=$?
echo "end $(date) exit $rc"
exit $rc
