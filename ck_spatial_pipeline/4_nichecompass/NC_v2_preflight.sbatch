#!/bin/bash

#SBATCH --job-name=NC_v2_preflight                                                    # Job name (%x in log paths)
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso_v2/logs/%x_%j.out    # STDOUT
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso_v2/logs/%x_%j.err     # STDERR
#SBATCH --partition=batch                                                             # CPU-only gate; QOS 'normal' has NO submit/TRES caps
#SBATCH --cpus-per-task=4                                                             # obs-only reads of 20 h5ads, light
#SBATCH --mem=32G                                                                     # obs frames only, generous headroom
#SBATCH --time=01:00:00                                                               # measured ~10 s for all 20 files; 1 h is pure slack
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/NC_v2_preflight.sbatch
#
# Wrapper for the preflight.
# ========================================================================================

# ------------------------------------------------------------------------------ #
# NicheCompass v2 PREFLIGHT — writes the TAG MANIFEST that is the ONLY source of #
# truth for array-index -> TAG mapping in all four v2 configurations.            #
#                                                                                #
# Reads obs only (no X, no layers) from the 20 QCed per-sample files under       #
# Ranger_procd/, verifies the post-canonicalisation label contract               #
# (is_MN_v2 == is_MN bit-for-bit; in_VH_v2/is_MN_v2 all <NA> on the 3            #
# vh_v2_source=='excluded' sections) and emits                                   #
#   <runs_v2>/<runstamp>_tag_manifest.json                                       #
# with keys tags_all20 (C1/C3) and tags_vh17 (C2/C4) plus per-tag counts.        #
#                                                                                #
# Exits non-zero on any contract failure, which blocks the afterok chain and so  #
# stops all 40 section-runs before a single GPU minute is spent.                 #
# Deliberately CPU/batch: every gpu_* QOS allows only 6 submitted jobs per user, #
# and the all-H200 plan needs every one of those slots for C1-C4. If the driver  #
# is told to move it (--preflight-partition gpu_normal, e.g. when `batch` is      #
# backed up), the driver charges it against that 6-slot budget and block 3 below #
# switches the H200 OpenSSL shim on by itself.                                   #
# ------------------------------------------------------------------------------ #

set -eo pipefail

WORKDIR=/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso_v2
PYBIN=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_env/bin/python
BASE_DIR=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC
DATA_DIR=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Ranger_procd
RUNS_V2="${BASE_DIR}/runs_v2"

# Runstamp is FIXED for this exercise; override only to open a parallel tree.
RUNSTAMP="${NC_V2_RUNSTAMP:-20260902_ncv2}"
TAG_MANIFEST="${NC_V2_TAG_MANIFEST:-${RUNS_V2}/${RUNSTAMP}_tag_manifest.json}"

# 1) Ensure the log dir exists (SLURM opens %x_%j files at job start; keep for reruns)
mkdir -p "${WORKDIR}/logs"
mkdir -p "${RUNS_V2}"

# 2) Environment init — the venv rides on the python/3.11.1 module; gcc/13.3.0 pairs with it
module load python/3.11.1 gcc/13.3.0

# 3) Exports — PYTHONNOUSERSITE keeps ~/.local out of the venv's import path
# The H200 node image has no OS libssl1.1 (python/3.11.1 _ssl needs it; batch nodes have
# it). This job is meant for `batch`, but if it was submitted to a gpu_* partition
# (driver --preflight-partition) prepend the shared-apps OpenSSL 1.1.1g build exactly as
# the GPU wrappers do (verified IMPORT_OK on that node, srun 52418731). Keyed on the
# partition SLURM actually placed the job in, so it is a no-op on batch nodes.
case "${SLURM_JOB_PARTITION:-}" in
    gpu_*)
        export LD_LIBRARY_PATH=/scg/apps/software/openssl/1.1.1g/lib:${LD_LIBRARY_PATH}
        echo "-- gpu_* partition (${SLURM_JOB_PARTITION}) — H200 OpenSSL 1.1.1g shim exported" ;;
esac
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NC_V2_RUNSTAMP="${RUNSTAMP}"
export NC_V2_TAG_MANIFEST="${TAG_MANIFEST}"
export NC_V2_BASE_DIR="${BASE_DIR}"
export NC_V2_DATA_DIR="${DATA_DIR}"
export NC_V2_RUNS_V2="${RUNS_V2}"

# 4) Run from the pipeline dir, never /tmp (stray /tmp/struct.py shadows stdlib struct)
cd "${WORKDIR}"

echo "== NC_v2_preflight | job ${SLURM_JOB_ID:-NA} | partition ${SLURM_JOB_PARTITION:-unknown} | node $(hostname) | $(date) =="
echo "   runstamp     : ${RUNSTAMP}"
echo "   data dir     : ${DATA_DIR}"
echo "   tag manifest : ${TAG_MANIFEST}"

# 5) Build the tag manifest + run the data-contract gates
"${PYBIN}" NC_v2_preflight.py \
    --runstamp "${RUNSTAMP}" \
    --data-dir "${DATA_DIR}" \
    --base-dir "${BASE_DIR}" \
    --manifest "${TAG_MANIFEST}"

# 6) Gate the manifest itself before any GPU job is allowed to consume it.
#    Cheap, and it is the one artefact all 40 section-runs index into.
echo "== manifest gate =="
"${PYBIN}" - "${TAG_MANIFEST}" <<'PYGATE'
import json
import sys

path = sys.argv[1]
with open(path) as fh:
    m = json.load(fh)

problems = []
for key, n_expected in (("tags_all20", 20), ("tags_vh17", 17)):
    tags = m.get(key)
    if not isinstance(tags, list):
        problems.append(f"{key}: missing or not a list")
        continue
    if len(tags) != n_expected:
        problems.append(f"{key}: {len(tags)} tags, expected {n_expected}")
    if len(set(tags)) != len(tags):
        problems.append(f"{key}: duplicate tags")
    print(f"   {key:<11} n={len(tags):>2}  {' '.join(map(str, tags))}", flush=True)

all20 = set(m.get("tags_all20") or [])
vh17 = set(m.get("tags_vh17") or [])
if not vh17 <= all20:
    problems.append(f"tags_vh17 is not a subset of tags_all20 (extra: {sorted(vh17 - all20)})")
print(f"   excluded (whole-section only): {' '.join(sorted(all20 - vh17))}", flush=True)

if problems:
    print("*** MANIFEST GATE FAILED ***", flush=True)
    for p in problems:
        print(f"    - {p}", flush=True)
    sys.exit(1)
print("   manifest gate PASSED", flush=True)
PYGATE

echo "== NC_v2_preflight done | $(date) =="
