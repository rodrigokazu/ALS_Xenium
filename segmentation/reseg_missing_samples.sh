#!/bin/bash
#SBATCH --job-name=xenium_reseg_missing
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/Xenium_ranger/logs/%x_%A_%a.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/Xenium_ranger/logs/%x_%A_%a.err
# ========================================================================================
# ALS_Xenium repo commentary | segmentation/reseg_missing_samples.sh
#
# A two-task repair job for the two samples that fell out of the original array: SD006/14 BG
# and SD028/18 BG.
#
# They were missing for different reasons and neither was data loss. SD028/18 BG had never
# been re-segmented with large cells at all. SD006/14 BG had been, but the output sat under a
# directory name that did not start with ALS, so the concatenation loop skipped it silently.
# That second failure mode is the more instructive one; the pipeline was quietly working on 19
# samples and reporting success.
#
# Self-contained by design, with the bundle paths embedded and the xeniumranger path baked in,
# because the .bashrc PATH entries had gone stale and only export for interactive shells
# anyway. It runs in scratch and rsyncs the finished pipestance to OAK, preserving the scratch
# copy if the copy out fails. That was the December quota failure mode.
#
# Both tasks completed in about four hours each and closed the cohort at 22 re-segmented
# directories.
# ========================================================================================


set -euo pipefail

# ============================================================
# Re-segment the two samples that were never large-cell re-segmented.
#   index 0 -> SD006/14 BG (ALS1, slide 0027679)
#   index 1 -> SD028/18 BG (ALS4, slide 0027849)
# Output lands in Ranger_procd/ following the existing naming convention:
#   ALS<n>_<SAMPLE>__largecells_reseg_<stamp>_job<jobid>_<arrayidx>/
# Self-contained: bundle paths are embedded, no manifest needed.
# Mirrors run_xenium_array.sh for params and name derivation.
# ============================================================
SEGMENT_LARGE_CELLS="true"
LOCALMEM_GB=256
DEST_ROOT="/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Ranger_procd"

# xeniumranger install — current location (the .bashrc PATH entries are stale and
# point at a dir that no longer exists, so put it on PATH explicitly here).
XENIUMRANGER_DIR="/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Xenium_ranger/xeniumranger-xenium4.0"
export PATH="${XENIUMRANGER_DIR}:${PATH}"

INPUTS=(
  "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/Xenium_ALS/Extracted_dataset/20241002__225213__20241002_JCK_ALS_1/output-XETG00102__0027679__SD00614_BG__20241002__225236"
  "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/Xenium_ALS/Extracted_dataset/20241216__235004__20241216_JCK_ALS_4/output-XETG00102__0027849__SD02818_BG__20241216__235029"
)

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

command -v xeniumranger >/dev/null || die "xeniumranger not found"
command -v rsync        >/dev/null || die "rsync not found"

# ------------------------------------------------------------
# SELECT INPUT FOR THIS ARRAY TASK
# ------------------------------------------------------------
IN="${INPUTS[$SLURM_ARRAY_TASK_ID]:-}"
IN="${IN%/}"
[[ -n "${IN}" ]] || die "No input for array index ${SLURM_ARRAY_TASK_ID}"
[[ -d "${IN}" ]] || die "Input bundle missing: ${IN}"

# ------------------------------------------------------------
# SAMPLE NAME DERIVATION (identical logic to run_xenium_array.sh)
# ------------------------------------------------------------
BUNDLE_NAME="$(basename "${IN}")"
DATASET_DIR="$(basename "$(dirname "${IN}")")"

SAMPLE_CODE="$(echo "${BUNDLE_NAME}" | awk -F'__' '{print $3}')"
ALS_TAG=""
[[ "${DATASET_DIR}" =~ ALS_([0-9]+) ]] && ALS_TAG="ALS${BASH_REMATCH[1]}"

SAMPLE="${ALS_TAG}_${SAMPLE_CODE}"
SAMPLE="$(echo "${SAMPLE}" | tr -cs 'A-Za-z0-9._+' '_')"

RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
# NOTE: double underscore before 'largecells' to match existing Ranger_procd dirs.
ID="${SAMPLE}__largecells_reseg_${RUNSTAMP}_job${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

LOCALCORES="${SLURM_CPUS_PER_TASK:-16}"

# ------------------------------------------------------------
# SCRATCH (run on fast local disk, then publish to Oak)
# ------------------------------------------------------------
SCRATCH_ROOT="${TMPDIR:-${SLURM_TMPDIR:-/tmp/${USER}}}"
SCRATCH="$(mktemp -d "${SCRATCH_ROOT%/}/xenium_${ID}_XXXXXX")"
WORKDIR="${SCRATCH}/work"
mkdir -p "${WORKDIR}"

log "------------------------------------------------------------"
log "ARRAY TASK:   ${SLURM_ARRAY_TASK_ID}"
log "INPUT BUNDLE: ${IN}"
log "RUN ID:       ${ID}"
log "SCRATCH:      ${SCRATCH}"
log "DEST:         ${DEST_ROOT}/${ID}"
log "NODE:         $(hostname)"
log "------------------------------------------------------------"

cd "${WORKDIR}"

# ------------------------------------------------------------
# RUN XENIUM RANGER RESEGMENT
# ------------------------------------------------------------
xeniumranger resegment \
  --xenium-bundle "${IN}" \
  --id "${ID}" \
  --segment-large-cells="${SEGMENT_LARGE_CELLS}" \
  --localcores "${LOCALCORES}" \
  --localmem "${LOCALMEM_GB}"

OUT_SCRATCH="${WORKDIR}/${ID}"
[[ -f "${OUT_SCRATCH}/outs/experiment.xenium" ]] || die "Validation failed: ${OUT_SCRATCH}/outs/experiment.xenium missing"

# ------------------------------------------------------------
# PUBLISH TO Ranger_procd (full pipestance dir, like the other ALS*_ runs)
# ------------------------------------------------------------
DEST="${DEST_ROOT}/${ID}"
log "Publishing pipestance to ${DEST} ..."
mkdir -p "${DEST_ROOT}"
if rsync -a "${OUT_SCRATCH}/" "${DEST}/"; then
  [[ -f "${DEST}/outs/experiment.xenium" ]] || die "Copy incomplete at ${DEST}"
  log "------------------------------------------------------------"
  log "DONE. Saved: ${DEST}"
  log "Scratch retained at ${SCRATCH} (safe to remove manually once verified)."
  log "------------------------------------------------------------"
else
  die "rsync to Oak FAILED (likely disk quota). Data preserved in scratch: ${OUT_SCRATCH}"
fi
