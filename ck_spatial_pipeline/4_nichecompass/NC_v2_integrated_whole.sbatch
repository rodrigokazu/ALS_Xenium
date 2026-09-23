#!/bin/bash

#SBATCH --job-name=NC_v2_int_whole                                                    # Job name (%x in log paths)
#SBATCH --output=/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso_v2/logs/%x_%j.out    # STDOUT
#SBATCH --error=/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso_v2/logs/%x_%j.err     # STDERR
#SBATCH --partition=gpu_normal                                                        # PROFILE DEFAULT h200: H200 node, MaxTime 1 d, started immediately on 09-02 (gpu_long projected 6 d). The driver overrides partition/gres/mem/time per NC_GPU_PROFILE (see block 3).
#SBATCH --gres=gpu:h200:1                                                             # 141GB H200; NicheCompass 0.3.3 is single-device. qrtx profile: gpu:qrtx4000:1 on batch
#SBATCH --cpus-per-task=8                                                             # BLAS/OMP + sklearn kNN, then UMAP/Leiden on CPU (same on every profile)
#SBATCH --mem=256G                                                                    # 08-26 peaks were 10.6 GiB (train) / 14.5 GiB (downstream); 256G is deliberate slack. qrtx profile: 120G (shared 380G nodes)
#SBATCH --time=1-00:00:00                                                             # gpu_normal ceiling; the proven run is ~2h05m (see NOTE 1). gpu_long profile: 3-00:00:00; qrtx: 3-00:00:00
#SBATCH --account=mpsnyder
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/NC_v2_integrated_whole.sbatch
#
# The 20260902 base run, integrated across all 20 sections on the whole section. Job 52474115.
# Its Leiden object carries the v1 niche numbering and is the input to nc3_step1.
# ========================================================================================

# ------------------------------------------------------------------------------ #
# C3 integrated_whole — ONE integrated NicheCompass model over all 20 WHOLE       #
# sections (1,097,669 cells x 480 genes), per-sample sklearn 4-NN graphs          #
# assembled block-diagonally, cat_covariates_keys=['sample'] (20 categories).     #
# This is the 2026-08-26 design (job 52418732) RE-RUN under the new labels:       #
# is_MN_v2 is the sole "Motor neurons" class and the coarse transcriptional        #
# cluster is demoted to "Other neurons".                                          #
#                                                                                 #
# Train (GPU) then downstream (same node, CPU) in ONE job — no afterok handoff.    #
# Outputs: runs_v2/integrated_whole/<runstamp>/{model,figures,results}/            #
#                                                                                 #
# NOTE 1 (proven footprint): 08-26 integrated training on this exact input took    #
#   1h15m wall on an H200 (early stop epoch 44/400, best 36, val edge-recon acc    #
#   0.949 / F1 0.974) and the CPU downstream took 45m50s (UMAP alone 32 min).      #
#   Budget ~2h05m, which is why the whole job fits gpu_normal's 1-day cap with a   #
#   10x margin; the gpu_long profile asks that partition's 3-day ceiling instead.  #
# NOTE 2 (no checkpointing): NicheCompass 0.3.3 writes nothing until model.save(),  #
#   so a wall-clock kill loses the whole train. Hence the generous asks.            #
# NOTE 3 (never touches the 08-26 tree): this writes only under runs_v2/. The       #
#   artifacts/sample_integration/latest symlink is NOT created, moved or read —     #
#   four existing scripts default to it and repointing it would silently retarget   #
#   them.                                                                            #
# NOTE 4 (GPU profile): the #SBATCH lines above are the MANUAL-submission defaults   #
#   (= profile h200). NC_v2_submit_all.sh passes --partition/--gres/--mem/--time on  #
#   the command line (they beat the directives) and exports NC_GPU_PROFILE; block 3  #
#   below turns that into the per-node-type settings. On the 8 GB qrtx4000 the      #
#   trainer runs the 5.38M-nnz cohort graph at NC_EDGE_BATCH=2048 (the 08-25 batch  #
#   variant's value) — slower, not different.                                        #
# ------------------------------------------------------------------------------ #

set -eo pipefail

WORKDIR=/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso_v2
PYBIN=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_env/bin/python
BASE_DIR=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC
DATA_DIR=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Ranger_procd
RUNS_V2="${BASE_DIR}/runs_v2"

CONFIG=integrated_whole
MANIFEST_KEY=tags_all20
RUNSTAMP="${NC_V2_RUNSTAMP:-20260902_ncv2}"
TAG_MANIFEST="${NC_V2_TAG_MANIFEST:-${RUNS_V2}/${RUNSTAMP}_tag_manifest.json}"
RUN_DIR="${RUNS_V2}/${CONFIG}/${RUNSTAMP}"
MANIFEST="${RUN_DIR}/results/run_manifest.json"

# 1) Ensure the log dir exists (SLURM opens %x_%j files at job start; keep for reruns)
mkdir -p "${WORKDIR}/logs"

# 2) Environment init — the venv rides on the python/3.11.1 module; gcc/13.3.0 pairs with it
module load python/3.11.1 gcc/13.3.0

# 3) GPU PROFILE — the ONLY block that differs per node type.
#    The driver submits with --export=ALL,NC_GPU_PROFILE=<name>; a manual submission that
#    leaves it unset gets the profile inferred from the partition SLURM placed the job in.
#      h200     -> gpu_normal (MaxTime 1 d), gpu:h200:1,     NC_EDGE_BATCH=4096, OpenSSL shim   [driver default]
#      gpu_long -> gpu_long   (MaxTime 3 d), gpu:h200:1,     NC_EDGE_BATCH=4096, OpenSSL shim   [opt-in; 6-day queue on 09-02]
#      qrtx     -> batch,                    gpu:qrtx4000:1, NC_EDGE_BATCH=2048, no shim        [fallback; 8 GB card]
if [ -z "${NC_GPU_PROFILE}" ]; then
    case "${SLURM_JOB_PARTITION:-}" in
        batch)    NC_GPU_PROFILE=qrtx ;;
        gpu_long) NC_GPU_PROFILE=gpu_long ;;
        *)        NC_GPU_PROFILE=h200 ;;
    esac
    echo "-- NC_GPU_PROFILE unset — inferred '${NC_GPU_PROFILE}' from partition '${SLURM_JOB_PARTITION:-unknown}'"
fi
export NC_GPU_PROFILE
case "${NC_GPU_PROFILE}" in
    h200|gpu_long)
        # H200 node image has no OS libssl1.1 (python/3.11.1 _ssl needs it; batch nodes have it) —
        # prepend the shared-apps OpenSSL 1.1.1g build (verified IMPORT_OK on this node, srun 52418731).
        # Needed for the WHOLE job: the CPU downstream half runs on the same H200 node instead of
        # the batch partition, where the 08-26 downstream ran.
        export LD_LIBRARY_PATH=/scg/apps/software/openssl/1.1.1g/lib:${LD_LIBRARY_PATH}
        export NC_EDGE_BATCH=4096   # the Sanger/H200-proven batch for the 5.38M-nnz cohort graph
        SSL_SHIM=yes
        EXPECT_PARTITION='gpu_*' ;;
    qrtx)
        export NC_EDGE_BATCH=2048   # the 8 GB qrtx4000 value from the 08-25 batch variant; batch nodes have OS libssl1.1
        SSL_SHIM=no
        EXPECT_PARTITION='batch' ;;
    *)
        echo "ERROR: unknown NC_GPU_PROFILE '${NC_GPU_PROFILE}' (expected h200 | gpu_long | qrtx)" >&2
        exit 2 ;;
esac
# Profile/partition cross-check: a qrtx profile on the H200 node (or the reverse) would train
# with the wrong edge batch and, on the H200, without the OpenSSL shim. Only checkable inside
# SLURM (SLURM_JOB_PARTITION is unset in the local unit harness).
if [ -n "${SLURM_JOB_PARTITION:-}" ]; then
    case "${SLURM_JOB_PARTITION}" in
        ${EXPECT_PARTITION}) : ;;
        *) echo "ERROR: NC_GPU_PROFILE=${NC_GPU_PROFILE} expects partition ${EXPECT_PARTITION} but this job runs in '${SLURM_JOB_PARTITION}'" >&2
           exit 2 ;;
    esac
fi

# 4) Exports common to every profile
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# numba JIT cache: without a persistent, writable cache dir UMAP/pynndescent recompile on every
# run — the downstream drafter measured a 30-min stall. Sharing one dir across jobs is safe
# (numba writes its index files atomically).
export NUMBA_CACHE_DIR=/home/rodrigok/.numba_cache_ncv2
mkdir -p "${NUMBA_CACHE_DIR}"

# --- warm the page cache: importing nichecompass/torch from /oak page-faults the .so files in at ~100 KB/s
#     on the H200 node (Lustre sync I/O, no read-ahead); a sequential pre-read runs at ~175 MB/s (5.8 GB ~90 s)
#     and makes the import take the normal ~2.5 min instead of stalling for 20+ min (verified 2026-09-02).
VENV_SP=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_env/lib/python3.11/site-packages
echo "== warming venv .so page cache | $(date) =="
find "${VENV_SP}" -name '*.so*' -size +1M -print0 2>/dev/null | xargs -0 cat > /dev/null 2>&1 || true
echo "== warm done | $(date) =="
export NC_V2_RUNSTAMP="${RUNSTAMP}"
export NC_V2_CONFIG="${CONFIG}"
export NC_V2_TAG_MANIFEST="${TAG_MANIFEST}"
export NC_V2_BASE_DIR="${BASE_DIR}"
export NC_V2_DATA_DIR="${DATA_DIR}"
export NC_V2_RUNS_V2="${RUNS_V2}"
# NC_ALLOW_CPU deliberately NOT set: a missing GPU must abort this job loudly.
# NC_MIN_GPS deliberately NOT set: the trainer's default 20 is the right gate (93 survive).

# 5) Run from the pipeline dir, never /tmp (stray /tmp/struct.py shadows stdlib struct)
cd "${WORKDIR}"

echo "== ${SLURM_JOB_NAME:-NC_v2_int_whole} | job ${SLURM_JOB_ID:-NA} | node $(hostname) | $(date) =="
echo "-- gpu profile: ${NC_GPU_PROFILE} | partition ${SLURM_JOB_PARTITION:-unknown} | NC_EDGE_BATCH=${NC_EDGE_BATCH} | OpenSSL shim: ${SSL_SHIM} | NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}"
echo "== nvidia-smi =="
nvidia-smi

# 6) Manifest gate — the tag list is resolved from the manifest, never hardcoded here
if [ ! -f "${TAG_MANIFEST}" ]; then
    echo "ERROR: tag manifest not found: ${TAG_MANIFEST}" >&2
    echo "       Run NC_v2_preflight.sbatch first — it is the ONLY source of truth for the tag lists." >&2
    exit 2
fi
TAGS=$("${PYBIN}" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(" ".join(list(m[sys.argv[2]])))' \
       "${TAG_MANIFEST}" "${MANIFEST_KEY}")
if [ -z "${TAGS}" ]; then
    echo "ERROR: manifest key ${MANIFEST_KEY} is empty in ${TAG_MANIFEST}" >&2
    exit 2
fi
echo "-- config     : ${CONFIG}"
echo "-- runstamp   : ${RUNSTAMP}"
echo "-- manifest   : ${TAG_MANIFEST} (key ${MANIFEST_KEY})"
echo "-- run dir    : ${RUN_DIR}"
echo "-- sections   : ${TAGS}"

if [ -f "${RUN_DIR}/results/.NC_V2_DONE" ] && [ -z "${NC_V2_FORCE}" ]; then
    echo "== SKIP: ${RUN_DIR}/results/.NC_V2_DONE present (export NC_V2_FORCE=1 to recompute) =="
    exit 0
fi

# results/run_manifest.json is the trainer's "train finished" signal ONLY when it parses:
# a kill mid-write leaves a truncated file, and trusting its mere existence would skip
# training forever and lose the run's provenance. Hence the parse test.
manifest_ok () {
    [ -f "$1" ] && "${PYBIN}" -c 'import json,sys; json.load(open(sys.argv[1]))' "$1" 2>/dev/null
}

# 7) Train — 400-epoch ceiling; 0.3.3 early stopping is ON by default.
#    `set -eo pipefail` means a non-zero exit here aborts the job BEFORE downstream.
#    The Trainer's per-epoch monitor line is KEPT here (unlike the arrays): on a
#    2-hour single job it is the primary progress signal and matches the 08-26 log.
#    A completed train whose downstream failed leaves a PARSEABLE run_manifest.json but
#    no .NC_V2_DONE — NC_v2_train.py would itself SKIP it (exit 0) after its ~2.5-min
#    import, so skip straight to downstream here. A manifest that exists but does NOT
#    parse is a truncated write: retrain with --overwrite (the trainer also treats a
#    corrupt manifest as unfinished; belt and braces). NC_V2_FORCE=1 retrains regardless.
TRAIN_EXTRA=""
if [ -n "${NC_V2_FORCE}" ]; then
    TRAIN_EXTRA="--overwrite"
elif [ -f "${MANIFEST}" ] && ! manifest_ok "${MANIFEST}"; then
    echo ""
    echo "== [train] ${MANIFEST} present but UNPARSEABLE (truncated write?) — retraining with --overwrite =="
    TRAIN_EXTRA="--overwrite"
fi
if [ -z "${NC_V2_FORCE}" ] && manifest_ok "${MANIFEST}"; then
    echo ""
    echo "== [train] SKIPPED — ${MANIFEST} already present and parseable;"
    echo "                     re-running downstream only (NC_V2_FORCE=1 to retrain) =="
else
    echo ""
    echo "== [train] ${CONFIG} | $(date) =="
    "${PYBIN}" NC_v2_train.py \
        --config "${CONFIG}" \
        --runstamp "${RUNSTAMP}" \
        --manifest "${TAG_MANIFEST}" ${TRAIN_EXTRA}
fi

# 8) Path gate: the trainer derives the run dir from config/runstamp; make any
#    disagreement with the path computed here fail loudly instead of silently.
#    The manifest must PARSE, not merely exist.
if [ ! -d "${RUN_DIR}/model" ] || ! manifest_ok "${MANIFEST}"; then
    echo "ERROR: run dir gate FAILED — expected ${RUN_DIR}/model and a parseable ${MANIFEST}" >&2
    exit 3
fi

# 9) Downstream on the same node's CPUs (neighbors -> UMAP -> Leiden 0.3 ->
#    differential GP tests -> niche x cell_type_mn / is_MN_v2 / in_VH_v2 surfaces)
echo ""
echo "== [downstream] ${CONFIG} | $(date) =="
"${PYBIN}" NC_v2_downstream.py \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}"

touch "${RUN_DIR}/results/.NC_V2_DONE"

echo ""
echo "=============================================================================="
echo "ALL DONE — ${CONFIG} | job ${SLURM_JOB_ID:-NA} | profile ${NC_GPU_PROFILE} | $(date)"
echo "  run dir  : ${RUN_DIR}"
echo "  model    : ${RUN_DIR}/model"
echo "  figures  : ${RUN_DIR}/figures"
echo "  results  : ${RUN_DIR}/results"
echo "  logs     : ${WORKDIR}/logs/${SLURM_JOB_NAME:-NC_v2_int_whole}_${SLURM_JOB_ID:-NA}.{out,err}"
echo "=============================================================================="
