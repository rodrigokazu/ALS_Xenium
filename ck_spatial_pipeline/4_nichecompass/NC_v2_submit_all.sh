#!/bin/bash
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/NC_v2_submit_all.sh
#
# Submission driver for the four v2 configurations. Only C3, integrated_whole, feeds the
# canonical lineage. The per-sample and ventral-horn configurations are superseded.
# ========================================================================================
# ------------------------------------------------------------------------------ #
# NC_v2_submit_all.sh — submission driver for the four NicheCompass v2           #
# configurations (C1 persample_whole, C2 persample_vh, C3 integrated_whole,       #
# C4 integrated_vh) under the fixed runstamp 20260902_ncv2.                       #
#                                                                                 #
#   preflight (CPU/batch, writes the tag manifest)                                #
#     └── afterok ──> integrated_whole   (1 GPU job)                              #
#     └── afterok ──> integrated_vh      (1 GPU job)                              #
#     └── afterok ──> persample_whole    (chunked array, NCHUNKS GPU jobs)        #
#     └── afterok ──> persample_vh       (chunked array, NCHUNKS GPU jobs)        #
#                                                                                 #
# GPU PROFILES (--profile-all P / --profile-<config> P). #SBATCH lines cannot     #
# read the environment, so a profile is rendered as sbatch COMMAND-LINE overrides #
# (--partition/--gres/--mem/--time beat the directives in the files) plus         #
# --export=ALL,NC_GPU_PROFILE=<name>, which every GPU sbatch body reads to pick   #
# NC_EDGE_BATCH and the H200 OpenSSL shim.                                        #
#   h200      gpu_normal  gpu:h200:1      MaxTime 1 d  %2  draws on the QOS budget #
#             DEFAULT. sbatch --test-only 2026-09-02 10:30: starts IMMEDIATELY on  #
#             the H200 node while gpu_long projected 2026-09-08 (6 days).          #
#   gpu_long  gpu_long    gpu:h200:1      MaxTime 3 d  %2  draws on the QOS budget #
#             opt-in only: another user queued six 3-day GPU jobs on 09-02 and     #
#             the projected start went from "now" to 6-12 days out.               #
#   qrtx      batch       gpu:qrtx4000:1  MaxTime 14 d %3  QOS 'normal' — no cap   #
#             fallback: 8 GB card (NC_EDGE_BATCH=2048), --mem <= 120G because the  #
#             380G qrtx nodes are shared (the 08-25 stall was a 128G ask vs 124G   #
#             free); only 2 of 4 qrtx nodes alive on 09-02 (d16-35, d16-37         #
#             drained), projected start 23:59 that night.                          #
#                                                                                 #
# THE BINDING CONSTRAINT (verified on this cluster 2026-09-02):                    #
#   gpu_long / gpu_normal / gpu_short QOS allow at most 6 SUBMITTED jobs per user  #
#   (MaxSubmitPU=6) and 2 concurrent GPUs (MaxTRESPU gres/gpu=2, repeated on the   #
#   partition QOS gpu_shared). Array TASKS count individually against the 6:       #
#   --array=0-6 is refused outright with QOSMaxSubmitJobPerUserLimit, --array=0-5  #
#   is accepted. With the default NCHUNKS=2 the all-H200 plan costs exactly        #
#   2+2+1+1 = 6 slots and fits in ONE submission. The driver tallies EVERY gpu_*   #
#   job it is about to submit — a gpu_* preflight included — against what is       #
#   already in the gpu_* queue and REFUSES an over-cap plan. The three QOSes are   #
#   separate objects, so the real cap may be per partition; the driver counts      #
#   across all gpu_* partitions on purpose (conservative). qrtx jobs ride `batch`  #
#   (QOS 'normal', no caps) and cost no slot.                                      #
# ------------------------------------------------------------------------------ #

set -eo pipefail

# WORKDIR is the SAME constant every .sbatch body hardcodes: they cd there, invoke
# NC_v2_train.py / NC_v2_downstream.py by RELATIVE name, and their #SBATCH -o/-e point
# at its logs/. The driver therefore refuses to run from anywhere else — a copy in another
# directory would pass its own deployment gate while the jobs it submits run against the
# deployed tree (demonstrated live 2026-09-02: a scratch copy planned six GPU jobs while
# NicheCompasso_v2/ held only logs/; every one would have died at `python NC_v2_train.py`).
WORKDIR=/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso_v2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
BASE_DIR=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC
RUNS_V2="${BASE_DIR}/runs_v2"

GPU_PARTITIONS=gpu_long,gpu_normal,gpu_short,gpu_admin
QOS_MAX_SUBMIT=6          # MaxSubmitPU on every gpu_* QOS (sacctmgr, verified)
QOS_MAX_GPUS=2            # MaxTRESPU gres/gpu on every gpu_* QOS (verified)

RUNSTAMP=20260902_ncv2
NCHUNKS=2
THROTTLE=""               # empty = per-profile default (2 on h200/gpu_long, 3 on qrtx)
PREFLIGHT_PARTITION=batch
CONFIGS="integrated_whole,integrated_vh,persample_whole,persample_vh"
DEFAULT_PROFILE=h200
PROFILE_persample_whole=""
PROFILE_persample_vh=""
PROFILE_integrated_whole=""
PROFILE_integrated_vh=""
FORCE=""
DRY_RUN=""
DO_PREFLIGHT=1
PREFLIGHT_ONLY=""

# ---- profile tables ------------------------------------------------------------ #
valid_profile     () { case "$1" in h200|gpu_long|qrtx) return 0 ;; *) return 1 ;; esac; }
profile_partition () { case "$1" in h200) echo gpu_normal ;; gpu_long) echo gpu_long ;; qrtx) echo batch ;; esac; }
profile_gres      () { case "$1" in h200|gpu_long) echo gpu:h200:1 ;; qrtx) echo gpu:qrtx4000:1 ;; esac; }
profile_throttle  () { case "$1" in h200|gpu_long) echo 2 ;; qrtx) echo 3 ;; esac; }   # 2 = QOS GPU cap; 3 = qrtx node count
profile_counts_qos () { case "$1" in h200|gpu_long) return 0 ;; *) return 1 ;; esac; }  # draws on MaxSubmitPU=6?
profile_mem () {  # profile_mem <profile> <config>
    case "$1:$2" in
        h200:persample_whole|gpu_long:persample_whole)   echo 128G ;;   # per-section peak ~1-2G; proven class
        h200:persample_vh|gpu_long:persample_vh)         echo 64G  ;;   # 931-10,194 cells per section
        h200:integrated_whole|gpu_long:integrated_whole) echo 256G ;;   # 08-26 peaks 10.6 / 14.5 GiB; deliberate slack
        h200:integrated_vh|gpu_long:integrated_vh)       echo 128G ;;   # 75,229 cells, ~15x smaller than C3
        qrtx:persample_whole)   echo 64G  ;;   # <= 120G on the shared 380G qrtx nodes (08-25: 128G ask stalled vs 124G free)
        qrtx:persample_vh)      echo 48G  ;;
        qrtx:integrated_whole)  echo 120G ;;   # the ceiling; measured peak 14.5 GiB
        qrtx:integrated_vh)     echo 64G  ;;
    esac
}
profile_time () {  # profile_time <profile> <config>
    case "$1:$2" in
        h200:*)                    echo 1-00:00:00 ;;   # gpu_normal MaxTime; every config fits (C3 = 2h05m proven)
        gpu_long:persample_whole)  echo 1-00:00:00 ;;   # 10 sections/task, worst case ~12 h
        gpu_long:persample_vh)     echo 06:00:00   ;;   # 9 sections/task, minutes each
        gpu_long:integrated_whole) echo 3-00:00:00 ;;   # partition ceiling; no mid-training checkpointing
        gpu_long:integrated_vh)    echo 1-00:00:00 ;;
        qrtx:integrated_whole)     echo 3-00:00:00 ;;   # batch MaxTime is 14 d; 3 d is plenty for a slower card
        qrtx:*)                    echo 2-00:00:00 ;;   # per-sample chunks and C4 on the slower card
    esac
}
profile_for () {  # profile_for <config> -> effective profile name
    local v=""
    case "$1" in
        persample_whole)  v="${PROFILE_persample_whole}" ;;
        persample_vh)     v="${PROFILE_persample_vh}" ;;
        integrated_whole) v="${PROFILE_integrated_whole}" ;;
        integrated_vh)    v="${PROFILE_integrated_vh}" ;;
    esac
    echo "${v:-${DEFAULT_PROFILE}}"
}
describe () {  # describe <config> -> "h200 -> gpu_normal / gpu:h200:1 / 128G / 1-00:00:00"
    local prof; prof=$(profile_for "$1")
    echo "${prof} -> $(profile_partition "${prof}") / $(profile_gres "${prof}") / $(profile_mem "${prof}" "$1") / $(profile_time "${prof}" "$1")"
}
array_spec () {  # array_spec <config> -> "0-<NCHUNKS-1>%<throttle>"
    local prof thr; prof=$(profile_for "$1"); thr="${THROTTLE:-$(profile_throttle "${prof}")}"
    echo "0-$((NCHUNKS - 1))%${thr}"
}

usage () {
    cat <<'USAGE'
Usage: NC_v2_submit_all.sh [options]      (run it FROM /home/rodrigok/SLURM_jobs/Spatial/NicheCompasso_v2)

  --runstamp STAMP         run tree + manifest stamp                 (default 20260902_ncv2)
  --nchunks N              chunks per per-sample array               (default 2)
  --throttle N             array concurrency (%N) for BOTH arrays    (default: per profile — 2 on h200/gpu_long, 3 on qrtx)
  --configs LIST           comma list, submitted in this order
                           (default integrated_whole,integrated_vh,persample_whole,persample_vh)
  --profile-all P          GPU profile for every config              (default h200)
  --profile-<config> P     GPU profile for ONE config, beats --profile-all for it; <config> is one of
                           persample_whole  persample_vh  integrated_whole  integrated_vh
  --preflight-partition P  partition for the CPU preflight           (default batch; a gpu_* value costs
                           one of the 6 QOS slots and is charged as such)
  --no-preflight           do not submit the preflight; requires an existing manifest
                           and submits the configs with no dependency
  --preflight-only         submit only the preflight and stop
  --force                  submit even though a manifest for this runstamp already exists
  --dry-run                print every sbatch command, submit nothing
  -h, --help               this text

Profiles (P) — rendered as sbatch --partition/--gres/--mem/--time overrides + --export=ALL,NC_GPU_PROFILE=P
  h200      gpu_normal  gpu:h200:1      throttle %2  NC_EDGE_BATCH=4096  OpenSSL shim  1 QOS slot per job   [DEFAULT: immediate start on 09-02]
  gpu_long  gpu_long    gpu:h200:1      throttle %2  NC_EDGE_BATCH=4096  OpenSSL shim  1 QOS slot per job   [opt-in: 6-12 day queue on 09-02]
  qrtx      batch       gpu:qrtx4000:1  throttle %3  NC_EDGE_BATCH=2048  no shim       0 QOS slots          [fallback: 8 GB card, 2 of 4 nodes alive]
  --mem / --time per (profile, config):
                   persample_whole    persample_vh      integrated_whole   integrated_vh
    h200           128G  1-00:00:00   64G  1-00:00:00   256G  1-00:00:00   128G  1-00:00:00
    gpu_long       128G  1-00:00:00   64G  06:00:00     256G  3-00:00:00   128G  1-00:00:00
    qrtx            64G  2-00:00:00   48G  2-00:00:00   120G  3-00:00:00    64G  2-00:00:00

Examples
  ./NC_v2_submit_all.sh                                    # preflight + all four on h200 (6 GPU slots)
  ./NC_v2_submit_all.sh --dry-run                          # see the plan first
  ./NC_v2_submit_all.sh --profile-all qrtx                 # everything on the qrtx4000 fallback (0 GPU slots)
  ./NC_v2_submit_all.sh --profile-all qrtx --profile-integrated_whole h200
                                                           # C3 on the H200, C1/C2/C4 on qrtx (1 GPU slot)
  ./NC_v2_submit_all.sh --configs integrated_whole,integrated_vh
  ./NC_v2_submit_all.sh --no-preflight --configs persample_vh
  ./NC_v2_submit_all.sh --runstamp 20260903_ncv2b --nchunks 6 --configs persample_whole
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --runstamp)             RUNSTAMP="$2"; shift 2 ;;
        --nchunks)              NCHUNKS="$2"; shift 2 ;;
        --throttle)             THROTTLE="$2"; shift 2 ;;
        --configs)              CONFIGS="$2"; shift 2 ;;
        --profile-all)          DEFAULT_PROFILE="$2"; shift 2 ;;
        --profile-persample_whole|--profile-persample-whole)   PROFILE_persample_whole="$2"; shift 2 ;;
        --profile-persample_vh|--profile-persample-vh)         PROFILE_persample_vh="$2"; shift 2 ;;
        --profile-integrated_whole|--profile-integrated-whole) PROFILE_integrated_whole="$2"; shift 2 ;;
        --profile-integrated_vh|--profile-integrated-vh)       PROFILE_integrated_vh="$2"; shift 2 ;;
        --preflight-partition)  PREFLIGHT_PARTITION="$2"; shift 2 ;;
        --no-preflight)         DO_PREFLIGHT=0; shift ;;
        --preflight-only)       PREFLIGHT_ONLY=1; shift ;;
        --force)                FORCE=1; shift ;;
        --dry-run)              DRY_RUN=1; shift ;;
        -h|--help)              usage; exit 0 ;;
        *) echo "ERROR: unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
done

TAG_MANIFEST="${RUNS_V2}/${RUNSTAMP}_tag_manifest.json"

case "${NCHUNKS}" in ''|*[!0-9]*) echo "ERROR: --nchunks must be a positive integer" >&2; exit 2 ;; esac
[ "${NCHUNKS}" -ge 1 ] || { echo "ERROR: --nchunks must be >= 1" >&2; exit 2; }
if [ -n "${THROTTLE}" ]; then
    case "${THROTTLE}" in *[!0-9]*) echo "ERROR: --throttle must be a positive integer" >&2; exit 2 ;; esac
    [ "${THROTTLE}" -ge 1 ] || { echo "ERROR: --throttle must be >= 1" >&2; exit 2; }
fi
for p in "${DEFAULT_PROFILE}" ${PROFILE_persample_whole} ${PROFILE_persample_vh} \
         ${PROFILE_integrated_whole} ${PROFILE_integrated_vh}; do
    valid_profile "${p}" || { echo "ERROR: unknown GPU profile '${p}' (h200 | gpu_long | qrtx)" >&2; exit 2; }
done
for cfg in $(echo "${CONFIGS}" | tr ',' ' '); do
    case "${cfg}" in
        persample_whole|persample_vh|integrated_whole|integrated_vh) : ;;
        *) echo "ERROR: unknown config '${cfg}' in --configs" >&2; exit 2 ;;
    esac
done

# 0) Location gate — this driver must be the copy deployed in WORKDIR (see the note above).
WORKDIR_P=$( { cd "${WORKDIR}" 2>/dev/null && pwd -P; } || true)
if [ -z "${WORKDIR_P}" ] || [ "${SCRIPT_DIR}" != "${WORKDIR_P}" ]; then
    echo "ERROR: run this driver from ${WORKDIR}" >&2
    if [ -z "${WORKDIR_P}" ]; then
        echo "       ${WORKDIR} does not exist; this copy lives in ${SCRIPT_DIR}." >&2
    else
        echo "       This copy lives in ${SCRIPT_DIR}; the deployed tree is ${WORKDIR_P}." >&2
    fi
    echo "       The sbatch files cd to ${WORKDIR} and invoke NC_v2_*.py relative to it, and their" >&2
    echo "       #SBATCH -o/-e point at its logs/, so submitting from another directory would launch" >&2
    echo "       jobs against a different (possibly empty or stale) tree. Deploy there, then resubmit." >&2
    exit 2
fi

# 1) Log dir must pre-exist: SLURM opens the -o/-e files at job start and a missing
#    dir kills the job before the script body runs.
mkdir -p "${WORKDIR}/logs"
mkdir -p "${RUNS_V2}"

echo "=============================================================================="
echo "NicheCompass v2 submission | runstamp ${RUNSTAMP} | $(date)"
echo "  scripts   : ${WORKDIR}"
echo "  manifest  : ${TAG_MANIFEST}"
echo "  run tree  : ${RUNS_V2}/<config>/${RUNSTAMP}"
echo "  geometry  : NCHUNKS=${NCHUNKS}, array throttle ${THROTTLE:+%${THROTTLE} (explicit)}${THROTTLE:-per profile (%2 h200/gpu_long, %3 qrtx)}"
echo "  configs   : ${CONFIGS}"
echo "  profiles  : (profile -> partition / gres / mem / time; NC_GPU_PROFILE is exported to the job)"
for cfg in $(echo "${CONFIGS}" | tr ',' ' '); do
    printf '     %-17s %s\n' "${cfg}" "$(describe "${cfg}")"
done
echo "=============================================================================="

# 2) Deployment gate — every file this plan invokes must be present in WORKDIR.
#    Cheaper to fail here than to have six jobs pend for hours and then die.
MISSING=""
for f in NC_v2_preflight.sbatch NC_v2_persample_whole.sbatch NC_v2_persample_vh.sbatch \
         NC_v2_integrated_whole.sbatch NC_v2_integrated_vh.sbatch \
         NC_v2_preflight.py NC_v2_train.py NC_v2_downstream.py; do
    [ -f "${WORKDIR}/${f}" ] || MISSING="${MISSING} ${f}"
done
if [ -n "${MISSING}" ]; then
    echo "ERROR: missing from ${WORKDIR}:${MISSING}" >&2
    echo "       Deploy the whole NicheCompasso_v2 set before submitting." >&2
    exit 2
fi
echo "-- deployment gate PASSED (5 sbatch + 3 python present)"

# 3) Idempotency gate — the manifest is the ONLY source of truth for index -> TAG.
#    Re-running the preflight over an existing one would move the ground under any
#    run already built from it.
if [ "${DO_PREFLIGHT}" = "1" ] && [ -f "${TAG_MANIFEST}" ] && [ -z "${FORCE}" ]; then
    echo "ERROR: a tag manifest for runstamp ${RUNSTAMP} already exists:" >&2
    echo "         ${TAG_MANIFEST}" >&2
    echo "       Resubmitting the preflight would overwrite the source of truth that the" >&2
    echo "       existing runs under ${RUNS_V2}/*/${RUNSTAMP}/ were built from. Pick one:" >&2
    echo "         --no-preflight            submit configs against the existing manifest" >&2
    echo "         --force                   overwrite the manifest anyway" >&2
    echo "         --runstamp <new stamp>    open a parallel run tree" >&2
    exit 2
fi
if [ "${DO_PREFLIGHT}" = "0" ] && [ ! -f "${TAG_MANIFEST}" ]; then
    echo "ERROR: --no-preflight given but no manifest at ${TAG_MANIFEST}" >&2
    exit 2
fi

# 4) Existing-run warning (not a refusal — finished sections are legitimately skipped
#    via results/.NC_V2_DONE, which is how a truncated array task resumes).
for cfg in $(echo "${CONFIGS}" | tr ',' ' '); do
    d="${RUNS_V2}/${cfg}/${RUNSTAMP}"
    if [ -d "${d}" ]; then
        # `|| true` inside the braces so a find hiccup cannot trip pipefail
        n_done=$( { find "${d}" -name .NC_V2_DONE 2>/dev/null || true; } | wc -l | tr -d ' ')
        echo "-- NOTE: ${d} already exists (${n_done} completed run(s)); those are SKIPPED unless NC_V2_FORCE=1"
    fi
done

# 5) QOS submit-cap arithmetic ------------------------------------------------- #
#    Only jobs that land on a gpu_* partition draw on the 6-slot MaxSubmitPU budget;
#    qrtx jobs ride `batch` (QOS 'normal', no caps) and are tallied separately.
NEEDED_QOS=0
NEEDED_QRTX=0
for cfg in $(echo "${CONFIGS}" | tr ',' ' '); do
    case "${cfg}" in
        persample_whole|persample_vh)      n=${NCHUNKS} ;;
        integrated_whole|integrated_vh)    n=1 ;;
    esac
    if profile_counts_qos "$(profile_for "${cfg}")"; then
        NEEDED_QOS=$((NEEDED_QOS + n))
    else
        NEEDED_QRTX=$((NEEDED_QRTX + n))
    fi
done
if [ -n "${PREFLIGHT_ONLY}" ]; then NEEDED_QOS=0; NEEDED_QRTX=0; fi
# A preflight moved onto a gpu_* partition is itself a submitted GPU job: charge it.
if [ "${DO_PREFLIGHT}" = "1" ]; then
    case "${PREFLIGHT_PARTITION}" in gpu_*) NEEDED_QOS=$((NEEDED_QOS + 1)); echo "-- NOTE: preflight on ${PREFLIGHT_PARTITION} costs 1 of the ${QOS_MAX_SUBMIT} GPU submit slots" ;; esac
fi

# squeue counts PENDING+RUNNING+COMPLETING by default, which is exactly what
# MaxSubmitPU charges; -r expands array tasks, which count individually.
if SQ_OUT=$(squeue -u "${USER}" -h -r -p "${GPU_PARTITIONS}" 2>/dev/null); then
    IN_QUEUE=$(printf '%s\n' "${SQ_OUT}" | grep -c . || true)
else
    echo "-- WARNING: squeue failed; the GPU submit-cap check below is not reliable" >&2
    IN_QUEUE=0
fi
echo "-- GPU QOS budget: ${IN_QUEUE} in queue + ${NEEDED_QOS} requested vs MaxSubmitPU=${QOS_MAX_SUBMIT} (concurrency ceiling ${QOS_MAX_GPUS} GPUs); qrtx/batch jobs requested: ${NEEDED_QRTX} (no cap)"
if [ "${NEEDED_QOS}" -gt 0 ] && [ "$((IN_QUEUE + NEEDED_QOS))" -gt "${QOS_MAX_SUBMIT}" ]; then
    echo "ERROR: this plan needs ${NEEDED_QOS} GPU submit slots but only $((QOS_MAX_SUBMIT - IN_QUEUE)) are free." >&2
    echo "       sbatch would reject the overflow with QOSMaxSubmitJobPerUserLimit." >&2
    echo "       Submit in waves, e.g." >&2
    echo "         ./NC_v2_submit_all.sh --configs integrated_whole,integrated_vh" >&2
    echo "         # wait for those to leave the queue, then" >&2
    echo "         ./NC_v2_submit_all.sh --no-preflight --configs persample_whole" >&2
    echo "         ./NC_v2_submit_all.sh --no-preflight --configs persample_vh" >&2
    echo "       or lower --nchunks (concurrency is capped at ${QOS_MAX_GPUS} GPUs either way)," >&2
    echo "       or move configs to the qrtx fallback, which costs no slot:" >&2
    echo "         ./NC_v2_submit_all.sh --profile-all qrtx --profile-integrated_whole h200" >&2
    exit 2
fi
if [ "${NEEDED_QRTX}" -gt 0 ]; then
    # Informational only: the qrtx pool is small and partly drained (09-02: 2 of 4 alive).
    QRTX_NODES=$( { sinfo -h -p batch -o "%n %G %t %e" 2>/dev/null || true; } | grep -i qrtx || true)
    if [ -n "${QRTX_NODES}" ]; then
        echo "-- qrtx4000 nodes on batch (node / gres / state / free MB):"
        printf '%s\n' "${QRTX_NODES}" | sed 's/^/     /'
    else
        echo "-- NOTE: could not list qrtx4000 nodes (sinfo unavailable here); on 09-02 only 2 of 4 were alive"
    fi
fi

# 6) Submit -------------------------------------------------------------------- #
# Every job inherits the runstamp/manifest/geometry through the environment
# (--export=ALL, plus the per-job NC_GPU_PROFILE), so the five scripts cannot disagree.
export NC_V2_RUNSTAMP="${RUNSTAMP}"
export NC_V2_TAG_MANIFEST="${TAG_MANIFEST}"
export NC_V2_NCHUNKS="${NCHUNKS}"

submit () {
    # submit <sbatch file> [sbatch args...] -> echoes the job id on stdout
    local script="$1"; shift
    if [ -n "${DRY_RUN}" ]; then
        echo "DRYRUN"
        echo "   sbatch --parsable $* ${WORKDIR}/${script}" >&2
        return 0
    fi
    local raw
    raw=$(sbatch --parsable "$@" "${WORKDIR}/${script}")
    echo "${raw%%;*}"   # strip any ";cluster" suffix from a federated controller
}

submit_gpu () {
    # submit_gpu <config> <sbatch file> [extra sbatch args...] -> job id on stdout
    # Renders the config's profile as command-line overrides (they beat the #SBATCH
    # lines in the file) and hands the profile name to the body via --export.
    local cfg="$1" script="$2"; shift 2
    local prof; prof=$(profile_for "${cfg}")
    submit "${script}" \
        --partition="$(profile_partition "${prof}")" \
        --gres="$(profile_gres "${prof}")" \
        --mem="$(profile_mem "${prof}" "${cfg}")" \
        --time="$(profile_time "${prof}" "${cfg}")" \
        --export="ALL,NC_GPU_PROFILE=${prof}" \
        "$@"
}

PRE_ID=""
IDS=""
DEP=()

# If a later sbatch is rejected (QOS, syntax, controller), say plainly what already landed.
on_submit_error () {
    local rc=$?
    echo "" >&2
    echo "!! submission aborted (exit ${rc}). Already submitted: preflight=${PRE_ID:-none}${IDS:- none}" >&2
    echo "   scancel those job ids if you want a clean slate before retrying." >&2
}
trap on_submit_error ERR

if [ "${DO_PREFLIGHT}" = "1" ]; then
    PRE_ID=$(submit NC_v2_preflight.sbatch --partition "${PREFLIGHT_PARTITION}")
    echo "-- preflight        : ${PRE_ID}  (partition ${PREFLIGHT_PARTITION}, writes the tag manifest)"
    if [ -n "${DRY_RUN}" ]; then
        DEP=(--dependency="afterok:<preflight>")   # placeholder so the printed plan shows the shape
    else
        DEP=(--dependency="afterok:${PRE_ID}")
    fi
else
    echo "-- preflight        : SKIPPED (--no-preflight); using the existing manifest"
fi

if [ -n "${PREFLIGHT_ONLY}" ]; then
    echo "-- --preflight-only: stopping here"
    echo "=============================================================================="
    echo "Submitted: preflight=${PRE_ID}"
    echo "Next: ./NC_v2_submit_all.sh --no-preflight   (once the manifest exists)"
    echo "=============================================================================="
    exit 0
fi

for cfg in $(echo "${CONFIGS}" | tr ',' ' '); do
    case "${cfg}" in
        integrated_whole)
            jid=$(submit_gpu "${cfg}" NC_v2_integrated_whole.sbatch "${DEP[@]}")
            echo "-- integrated_whole : ${jid}  (1 job, 1 GPU, ~2h05m proven on H200; $(describe "${cfg}"))" ;;
        integrated_vh)
            jid=$(submit_gpu "${cfg}" NC_v2_integrated_vh.sbatch "${DEP[@]}")
            echo "-- integrated_vh    : ${jid}  (1 job, 1 GPU, 75,229 cells; $(describe "${cfg}"))" ;;
        persample_whole)
            spec=$(array_spec "${cfg}")
            jid=$(submit_gpu "${cfg}" NC_v2_persample_whole.sbatch --array="${spec}" "${DEP[@]}")
            echo "-- persample_whole  : ${jid}  (array ${spec}, 20 sections over ${NCHUNKS} chunks; $(describe "${cfg}"))" ;;
        persample_vh)
            spec=$(array_spec "${cfg}")
            jid=$(submit_gpu "${cfg}" NC_v2_persample_vh.sbatch --array="${spec}" "${DEP[@]}")
            echo "-- persample_vh     : ${jid}  (array ${spec}, 17 sections over ${NCHUNKS} chunks; $(describe "${cfg}"))" ;;
    esac
    IDS="${IDS} ${cfg}=${jid}"
done

echo "=============================================================================="
if [ -n "${DRY_RUN}" ]; then
    echo "DRY RUN — nothing was submitted."
else
    echo "Submitted: preflight=${PRE_ID:-none}${IDS}"
    echo ""
    echo "Monitor:"
    echo "  squeue -u ${USER} -r                                   # -r expands array tasks"
    echo "  squeue --start -u ${USER}                              # projected starts (conservative)"
    echo "  sacct -j <jobid> --format=JobID,JobName%18,Partition,State,Elapsed,MaxRSS,ExitCode"
    echo "  tail -f ${WORKDIR}/logs/NC_v2_int_whole_<jobid>.out"
    echo "  tail -f ${WORKDIR}/logs/NC_v2_ps_whole_<arrayid>_<task>.out"
    echo ""
    echo "A failed parent leaves dependents PENDING with DependencyNeverSatisfied —"
    echo "clear them with scancel <jobid>. A task killed by the wall clock loses only its"
    echo "in-flight section: resubmit with the same command (finished sections skip)."
fi
echo "=============================================================================="
