#!/bin/bash
#SBATCH --job-name=xenium_array
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/rodrigok/SLURM_jobs/Xenium_ranger/logs/%x_%A_%a.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Xenium_ranger/logs/%x_%A_%a.err
# ========================================================================================
# ALS_Xenium repo commentary | segmentation/run_xenium_array.sh
#
# The Xenium Ranger re-segmentation array that produced the large-cell segmentation everything
# downstream was originally built on. Six tasks, 16 CPUs, 256G, 72 hours each.
#
# --segment-large-cells=true is the whole point. Motor neuron somata are large enough that
# default segmentation cuts them up or loses them, and recovering those cells is the
# difference between having a motor neuron population and not.
#
# Input bundles are listed in the xenium_inputs files kept alongside this on SCG. Output
# pipestances are around 12 to 22G each and land in Ranger_procd with an ALS-prefixed
# directory name, which is what the older concatenation loop globbed for.
#
# Historical note that still matters. Those original Ranger_procd directories hold the
# physical DAPI and 18S OME-TIFFs. Marcel's later _final and _gauss trees only symlink to
# them, so deleting Ranger_procd breaks the morphology images for every current overlay
# script.
# ========================================================================================


set -euo pipefail

# ============================================================
# CONFIG
# ============================================================
MANIFEST="/home/rodrigok/SLURM_jobs/Xenium_ranger/xenium_inputs56.txt"
SEGMENT_LARGE_CELLS="true"
LOCALMEM_GB=256
# ============================================================

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

command -v xeniumranger >/dev/null || die "xeniumranger not found"
command -v tar          >/dev/null || die "tar not found"
command -v awk          >/dev/null || die "awk not found"
command -v mktemp       >/dev/null || die "mktemp not found"

# ------------------------------------------------------------
# SELECT INPUT FOR THIS ARRAY TASK
# ------------------------------------------------------------
IN="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")"
IN="${IN%/}"

[[ -n "${IN}" ]] || die "No input for array index ${SLURM_ARRAY_TASK_ID}"
[[ -d "${IN}" ]] || die "Input bundle missing: ${IN}"

# ------------------------------------------------------------
# SAMPLE NAME DERIVATION
# ------------------------------------------------------------
BUNDLE_NAME="$(basename "${IN}")"
DATASET_DIR="$(basename "$(dirname "${IN}")")"

SAMPLE_CODE="$(echo "${BUNDLE_NAME}" | awk -F'__' '{print $3}')"
ALS_TAG=""
[[ "${DATASET_DIR}" =~ ALS_([0-9]+) ]] && ALS_TAG="ALS${BASH_REMATCH[1]}"

SAMPLE="${ALS_TAG}_${SAMPLE_CODE}"
SAMPLE="$(echo "${SAMPLE}" | tr -cs 'A-Za-z0-9._+' '_' )"

RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
JOBTAG="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
ID="${SAMPLE}_largecells_reseg_${RUNSTAMP}_job${JOBTAG}"

LOCALCORES="${SLURM_CPUS_PER_TASK:-16}"

# ------------------------------------------------------------
# SCRATCH
# ------------------------------------------------------------
SCRATCH_ROOT="${TMPDIR:-${SLURM_TMPDIR:-/tmp/${USER}}}"
SCRATCH="$(mktemp -d "${SCRATCH_ROOT%/}/xenium_${ID}_XXXXXX")"
WORKDIR="${SCRATCH}/work"
mkdir -p "${WORKDIR}"

log "------------------------------------------------------------"
log "ARRAY TASK:     ${SLURM_ARRAY_TASK_ID}"
log "INPUT BUNDLE:   ${IN}"
log "RUN ID:         ${ID}"
log "SCRATCH:        ${SCRATCH}"
log "NODE:           $(hostname)"
log "------------------------------------------------------------"

cd "${WORKDIR}"

# ------------------------------------------------------------
# RUN XENIUM
# ------------------------------------------------------------
xeniumranger resegment \
  --xenium-bundle "${IN}" \
  --id "${ID}" \
  --segment-large-cells="${SEGMENT_LARGE_CELLS}" \
  --localcores "${LOCALCORES}" \
  --localmem "${LOCALMEM_GB}"

OUT_SCRATCH="${WORKDIR}/${ID}"

[[ -f "${OUT_SCRATCH}/outs/experiment.xenium" ]] || die "Validation failed"

# ------------------------------------------------------------
# TAR (STAYS IN SCRATCH)
# ------------------------------------------------------------
tar -cf "${ID}.tar" "${ID}"

TAR_PATH="${WORKDIR}/${ID}.tar"

log "------------------------------------------------------------"
log "XENIUM RUN COMPLETE — JOB PAUSED"
log
log "OUTPUT BUNDLE:"
log "  ${OUT_SCRATCH}"
log
log "TARBALL:"
log "  ${TAR_PATH}"
log
log "COPY EXAMPLES:"
log "  scp rodrigok@$(hostname):${TAR_PATH} ./"
log "  rsync -av rodrigok@$(hostname):${OUT_SCRATCH}/ ./"
log
log "WHEN DONE:"
log "  scancel ${SLURM_JOB_ID}"
log "------------------------------------------------------------"

sleep infinity
