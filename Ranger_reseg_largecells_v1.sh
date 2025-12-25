#!/bin/bash
#SBATCH --job-name=sleepy_ranger
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=/home/rodrigok/SLURM_jobs/Xenium_ranger/logs/slurm_%x_%j.out
#SBATCH --error=/home/rodrigok/SLURM_jobs/Xenium_ranger/logs/slurm_%x_%j.err

set -euo pipefail

# ============================================================
# USER EDIT SECTION
# ============================================================
IN="/oak/stanford/scg/lab_mpsnyder/johnck/Projects/Xenium_ALS/Extrated_dataset/20241121__195935__20241121_JCK_ALS_2/output-XETG00102__0028096__SD01922_BI__20241121__195958"

SEGMENT_LARGE_CELLS="true"
LOCALMEM_GB=256
SAMPLE_OVERRIDE=""
# ============================================================

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

command -v xeniumranger >/dev/null || die "xeniumranger not found"
command -v tar          >/dev/null || die "tar not found"
command -v awk          >/dev/null || die "awk not found"
command -v df           >/dev/null || die "df not found"
command -v du           >/dev/null || die "du not found"
command -v mktemp       >/dev/null || die "mktemp not found"

IN="${IN%/}"
[[ -d "${IN}" ]] || die "Input bundle does not exist"

[[ -f "${IN}/experiment.xenium" ]] || die "Missing experiment.xenium"
[[ -f "${IN}/gene_panel.json" ]]     || die "Missing gene_panel.json"
[[ -f "${IN}/cells.parquet" ]]       || die "Missing cells.parquet"
[[ -f "${IN}/transcripts.parquet" ]] || die "Missing transcripts.parquet"

# ------------------------------------------------------------
# SAMPLE NAME
# ------------------------------------------------------------
BUNDLE_NAME="$(basename "${IN}")"
DATASET_DIR="$(basename "$(dirname "${IN}")")"

SAMPLE_CODE="$(echo "${BUNDLE_NAME}" | awk -F'__' '{print $3}')"
[[ -n "${SAMPLE_CODE}" ]] || die "Failed to parse sample code"

ALS_TAG=""
[[ "${DATASET_DIR}" =~ ALS_([0-9]+) ]] && ALS_TAG="ALS${BASH_REMATCH[1]}"

AUTO_SAMPLE="${SAMPLE_CODE}"
[[ -n "${ALS_TAG}" ]] && AUTO_SAMPLE="${ALS_TAG}_${SAMPLE_CODE}"

SAMPLE="${SAMPLE_OVERRIDE:-${AUTO_SAMPLE}}"
SAMPLE="$(echo "${SAMPLE}" | tr -cs 'A-Za-z0-9._+' '_' )"

RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
JOBTAG="${SLURM_JOB_ID:-manual_$$}"
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
log "Running Xenium Ranger (PAUSE-AFTER-SUCCESS MODE)"
log "Job ID:        ${SLURM_JOB_ID}"
log "Run ID:        ${ID}"
log "Scratch root:  ${SCRATCH}"
log "------------------------------------------------------------"

cd "${WORKDIR}"

# ------------------------------------------------------------
# RUN XENIUM
# ------------------------------------------------------------
log "Launching xeniumranger resegment..."
xeniumranger resegment \
  --xenium-bundle "${IN}" \
  --id "${ID}" \
  --segment-large-cells="${SEGMENT_LARGE_CELLS}" \
  --localcores "${LOCALCORES}" \
  --localmem "${LOCALMEM_GB}"

OUT_SCRATCH="${WORKDIR}/${ID}"

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------
[[ -f "${OUT_SCRATCH}/outs/experiment.xenium" ]] || die "Missing experiment.xenium"
log "Validation PASSED"

# ------------------------------------------------------------
# TAR OUTPUT (STAYS IN SCRATCH)
# ------------------------------------------------------------
cd "${WORKDIR}"
log "Creating tarball in scratch..."
tar -cf "${ID}.tar" "${ID}"

TAR_PATH="${WORKDIR}/${ID}.tar"

log "------------------------------------------------------------"
log "XENIUM RUN COMPLETE — JOB IS NOW PAUSED"
log
log "DATA LOCATIONS:"
log "  Output bundle: ${OUT_SCRATCH}"
log "  Tarball:       ${TAR_PATH}"
log
log "NODE:"
log "  $(hostname)"
log
log "EXAMPLE COPY COMMANDS:"
log "  scp rodrigok@$(hostname):${TAR_PATH} ./"
log "  rsync -av rodrigok@$(hostname):${OUT_SCRATCH}/ ./"
log
log "WHEN YOU ARE DONE COPYING:"
log "  scancel ${SLURM_JOB_ID}"
log "------------------------------------------------------------"

# ------------------------------------------------------------
# PAUSE FOREVER (UNTIL YOU CANCEL)
# ------------------------------------------------------------
sleep infinity
