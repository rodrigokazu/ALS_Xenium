#!/bin/bash
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/SUBMIT_ALL_FINAL.sh
#
# The whole pipeline as one dependency graph. Read this before submitting anything by hand,
# because the ordering constraints are real and several stages will produce confident nonsense
# if fed a half-written input.
#
# Shape of the DAG: concat first, then in parallel the Novae joint fit, the Novae independent
# array, the astrocyte isolation with its tSNE sweep, and coarse cell typing; then the
# overlays and colocalisation off the joint run, and the dual-pass QC off the independent run
# plus cell typing; then the PDF and audio reports.
#
# Flags: --domains-only runs the half that does not need motor-neuron labels. That is what was
# launched on 2026-07-22 as jobs 52199989 to 52200003. --skip-celltyping and --no-astro-tsne
# trim the expensive optional arms. --dry-run prints the submission plan without touching the
# queue and is worth running every time.
#
# A full launch refuses to proceed unless check_annotation_ready.sh passes. That guard is the
# point; do not route around it because Marcel said the annotation is coming.
#
# Job ids land in SUBMITTED_JOBS.txt. If you edit stage script names, edit them here too. The
# two names drifted apart once already when the cell typing runners were renamed to run_ct_*
# and the launch failed on missing files.
# ========================================================================================

# =============================================================================================
# SUBMIT_ALL_FINAL.sh -- one-command submission of the ENTIRE MN-corrected _FINAL pipeline on
# Marcel's "_final" segmentation, with SLURM dependencies + arrays, in this order:
#
#   [0] concat                                          run_VH_concatenate_FINAL.sh
#         |
#         +-- afterok --> (parallel fan-out) -----------------------------------------------.
#         |                                                                                  |
#   [1a] novae JOINT      run_novae_JOINT_FINAL.sh            (GPU; per_sample/*__niches.h5ad)
#   [1b] novae INDEP      run_novae_INDEPENDENT_FINAL.sh      (array 0-19%5, GPU)
#            +-- afterany --> novae INDEP aggregate           (--wrap; stitch summary)
#   [1c] astro            run_astro_sanity_FINAL.sh           (clustered_full.h5ad)
#            +-- afterok  --> tsne prep --> afterok --> tsne sweep (array 0-23%4)  [--no-astro-tsne skips]
#   [1d] celltyping       run_ct_pipe_FINAL.sh                (obs_celltype.csv, X_pca.npy)
#            +-- afterok  --> ct_tsne --> afterok --> ct_plots
#            +-- afterok  --> ct_dotplots
#            +-- afterok  --> ct_heroes   [FULL mode only; is_MN + boundary(C2)-gated]
#         |
#   [2 ] novae JOINT -- afterok --> overlays  run_overlays_FINAL.sh   (array 0-19%6)  [both modes]
#                    -- afterok --> coloc     run_gene_coloc_FINAL.sh (array 0-2%3)   [FULL only; is_MN]
#                                      +-- afterok --> gene-comparison (--wrap)        [FULL only; is_MN]
#   [2 ] dpqc      afterok(novae INDEP + celltyping) run_dpqc_array_FINAL.sh (array 0-19%6)
#   [3 ] dpqc aggregate  afterany(dpqc) run_dpqc_aggregate_FINAL.sh   = Cohort/Meeting/Audio/Strategy PDFs
#
# GUARD: refuses to submit unless check_annotation_ready.sh PASSES, OR --domains-only is given.
#   --domains-only  : skip the is_MN-gated stages (coloc, gene-comparison, MN + celltype heroes)
#                     and do NOT require the annotation. Everything else (concat, both novae,
#                     astro, celltyping, domain overlays, dpqc + PDFs/audio) still runs.
#
# FLAGS
#   --domains-only     skip is_MN-gated stages; no annotation required (see GUARD above)
#   --skip-celltyping  drop the celltyping branch; dpqc then depends on novae-INDEP only and its
#                      composition sub-score degrades to Untyped with a loud warning (dpqc review G1)
#   --no-astro-tsne    do not submit the astro tSNE parameter sweep (prep + array)
#   --dry-run          print the exact sbatch commands + the resolved DAG, submit NOTHING
#   -h | --help        show this help
#
# *** STAGING: do NOT run this now. It SUBMITS SLURM jobs. Run it (on the SCG login node, with a
#     live controller) only after: (full) Marcel's _final MN annotation has landed for all 20 AND
#     the _final inputs are group-readable + the concat inputs exist; or (--domains-only) for the
#     annotation-independent domain/QC pipeline. Preview any time with --dry-run. ***
# =============================================================================================
set -uo pipefail

# ---- fixed locations (mission OUTPUT NAMING; match final_config single-source-of-truth) -------
CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
LOGS="$CODE/logs"
SPATIAL=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial
NOVAE_JOINT="$SPATIAL/Novae_persample_niches_FINAL"
COLOC_OUT="$NOVAE_JOINT/gene_coloc"                 # == run_gene_coloc_FINAL.sh default OUT
NOVAE_ENV=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/novae_env
VIS_PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/xenium_vistools/bin/python
ACCT="--account=mpsnyder --partition=batch"
SUBMITTED="$CODE/SUBMITTED_JOBS.txt"

# ---- flags ------------------------------------------------------------------------------------
FULL=true; RUN_CT=true; RUN_ASTRO_TSNE=true; DRYRUN=false
usage(){ sed -n '2,60p' "$0"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --domains-only)    FULL=false ;;
    --skip-celltyping) RUN_CT=false ;;
    --no-astro-tsne)   RUN_ASTRO_TSNE=false ;;
    --dry-run)         DRYRUN=true ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "unknown argument: $1  (see --help)"; exit 2 ;;
  esac
  shift
done
$DRYRUN && SUBMITTED="$SUBMITTED.dryrun"     # never clobber the real manifest during a preview

# ---- preflight: SLURM present + code dir + logs -----------------------------------------------
command -v sbatch  >/dev/null 2>&1 || { echo "ABORT: sbatch not on PATH (is the SLURM controller reachable?)"; exit 6; }
command -v squeue  >/dev/null 2>&1 || { echo "ABORT: squeue not on PATH"; exit 6; }
[ -d "$CODE" ] || { echo "ABORT: code dir missing: $CODE"; exit 6; }
mkdir -p "$LOGS"

# ---- preflight: required staged scripts exist for this mode -----------------------------------
REQ=(run_VH_concatenate_FINAL.sh run_novae_JOINT_FINAL.sh run_novae_INDEPENDENT_FINAL.sh
     novae_INDEPENDENT_FINAL.py run_astro_sanity_FINAL.sh run_overlays_FINAL.sh
     run_dpqc_array_FINAL.sh run_dpqc_aggregate_FINAL.sh check_annotation_ready.sh)
$RUN_ASTRO_TSNE && REQ+=(run_tsne_sweep_prep_FINAL.sh run_tsne_sweep_FINAL.sh)
$FULL && REQ+=(run_gene_coloc_FINAL.sh gene_coloc_compare_fig_FINAL.py)

MISSING=()
for f in "${REQ[@]}"; do [ -e "$CODE/$f" ] || MISSING+=("$f"); done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "PREFLIGHT FAILED: missing staged script(s) in $CODE:"
  printf '  - %s\n' "${MISSING[@]}"
  exit 3
fi

# ---- preflight: celltyping stage (separately staged; may be absent today -- CRITIC C1) --------
if $RUN_CT; then
  CT_SCRIPTS=(run_ct_pipe_FINAL.sh run_ct_tsne_FINAL.sh run_ct_plots_FINAL.sh run_ct_dotpersample_FINAL.sh run_opentsne_FINAL.py)
  $FULL && CT_SCRIPTS+=(run_ct_heroes_FINAL.sh)
  CT_MISSING=()
  for f in "${CT_SCRIPTS[@]}"; do [ -e "$CODE/$f" ] || CT_MISSING+=("$f"); done
  if [ ${#CT_MISSING[@]} -gt 0 ]; then
    echo "PREFLIGHT: coarse cell-typing _FINAL scripts are NOT staged yet in $CODE:"
    printf '  - %s\n' "${CT_MISSING[@]}"
    echo
    echo "  Per CRITIC-RULING 2026-07-19 (C1) they are a separate staging item: adapt"
    echo "    celltype_pipeline / celltype_plots(+plots_fix) / celltype_dotplots_persample / hero_celltype"
    echo "    + run_pipe/tsne/plots/dotpersample/heroes + run_opentsne  ->  *_FINAL (import final_config;"
    echo "    OUT=CellTyping_coarse_FINAL; logs->final_rerun_code/logs; caches->\$TMPDIR)."
    echo
    echo "  To launch the rest NOW without typing, re-run with --skip-celltyping"
    echo "  (dpqc will depend on novae-INDEP only; its composition sub-score degrades to Untyped"
    echo "   with a loud warning -- see the dpqc staging review G1)."
    exit 3
  fi
fi

# ---- GUARD: annotation readiness gates the is_MN-dependent stages ------------------------------
echo "================================================================"
echo "SUBMIT_ALL_FINAL   $(date)"
echo "  mode         : $([ "$FULL" = true ] && echo 'FULL (is_MN stages ON)' || echo 'domains-only (is_MN stages OFF)')"
echo "  celltyping   : $RUN_CT      astro tSNE sweep: $RUN_ASTRO_TSNE      dry-run: $DRYRUN"
echo "  code / logs  : $CODE"
echo "================================================================"
if $FULL; then
  echo ">>> FULL mode: checking MN annotation readiness (gate for is_MN stages)..."
  if bash "$CODE/check_annotation_ready.sh"; then
    echo ">>> annotation READY -- is_MN stages (coloc, gene-comparison, MN/celltype heroes) WILL be submitted."
  else
    echo
    echo "ABORT: MN annotation is NOT READY (see the table above)."
    echo "  Wait for Marcel's _final annotation for all 20 samples, then re-run; OR run with"
    echo "  --domains-only to submit the annotation-independent domain/QC pipeline now."
    exit 4
  fi
else
  echo ">>> --domains-only: SKIPPING is_MN stages (coloc, gene-comparison, MN/celltype heroes);"
  echo "    annotation readiness NOT required for the domain/QC pipeline."
fi
echo
echo "  NOTE (boundary, CRITIC C2): overlays / coloc heroes / celltype heroes DRAW cell polygons."
echo "  On _final today the boundary parquets are ~10% coverage + carry the old gm- id namespace,"
echo "  so boundary figures may render EMPTY until Marcel re-exports full-coverage integer-id"
echo "  boundaries. The scripts degrade (SKIP / [ERR], no crash); domain/gene overlays that need"
echo "  no polygons are unaffected."
echo

# ---- submit helper (retries a flaky controller; --parsable jobid on stdout; honors --dry-run) -
# dry-run uses a FILE-backed counter: submit() is called via $(...) command substitution, which
# runs in a subshell, so a plain shell var would not persist across calls (every id would repeat).
_DRYCNT="${TMPDIR:-/tmp}/.submit_all_dryid.$$"
$DRYRUN && echo 1000 > "$_DRYCNT"
submit(){   # usage: JID=$(submit <sbatch args...>)  ; prints jobid, returns nonzero on failure
  if $DRYRUN; then
    local n; n=$(( $(cat "$_DRYCNT") + 1 )); echo "$n" > "$_DRYCNT"
    { printf '  [dry-run] sbatch --parsable'; printf ' %q' "$@"; printf '\n'; } >&2
    echo "$n"; return 0
  fi
  local out i
  for i in $(seq 1 30); do
    out=$(sbatch --parsable "$@" 2>&1)
    if echo "$out" | grep -qE '^[0-9]+'; then echo "${out%%;*}"; return 0; fi
    echo "  (controller busy, retry $i) $out" >&2; sleep 10
  done
  echo "  SUBMIT FAILED after retries: sbatch $*" >&2
  return 1
}

# ---- manifest ---------------------------------------------------------------------------------
[ -f "$SUBMITTED" ] && cp "$SUBMITTED" "$SUBMITTED.prev" 2>/dev/null || true
{
  echo "# SUBMIT_ALL_FINAL manifest  $(date)"
  echo "# mode=$([ "$FULL" = true ] && echo FULL || echo domains-only)  celltyping=$RUN_CT  astro_tsne=$RUN_ASTRO_TSNE  dry_run=$DRYRUN"
  printf '# %-20s %-12s %-30s %s\n' STAGE JOBID DEPENDENCY NOTES
} > "$SUBMITTED"
record(){ # record STAGE JOBID DEP NOTES  (echo + append to manifest)
  printf '  %-20s %-12s dep=%-26s %s\n' "$1" "$2" "${3:-none}" "${4:-}"
  printf '%-22s %-12s %-30s %s\n' "$1" "$2" "${3:-none}" "${4:-}" >> "$SUBMITTED"
}
J_CT_PIPE=""   # referenced conditionally in the dpqc dependency

# =============================================================================================
echo "========================= STAGE 0: CONCAT ========================="
J_CONCAT=$(submit run_VH_concatenate_FINAL.sh) || { echo "ABORT: concat submit failed"; exit 5; }
record CONCAT "$J_CONCAT" none "run_VH_concatenate_FINAL.sh -> combined + per-sample preQC h5ads"

echo
echo "===== STAGE 1: novae JOINT | novae INDEP | astro | celltyping (parallel, afterok concat) ====="
J_JOINT=$(submit --dependency=afterok:$J_CONCAT run_novae_JOINT_FINAL.sh) \
  || { echo "ABORT: novae JOINT submit failed"; exit 5; }
record NOVAE_JOINT "$J_JOINT" "afterok:$J_CONCAT" "GPU; writes per_sample/*__niches.h5ad"

J_INDEP=$(submit --dependency=afterok:$J_CONCAT run_novae_INDEPENDENT_FINAL.sh) \
  || { echo "ABORT: novae INDEP submit failed"; exit 5; }
record NOVAE_INDEP "$J_INDEP" "afterok:$J_CONCAT" "array 0-19%5 (GPU); per_sample/*__niches_independent.h5ad"

J_INDEP_AGG=$(submit --dependency=afterany:$J_INDEP --job-name=Novae_INDEP_FINAL_agg \
    --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err" \
    --time=00:20:00 --mem=8G --cpus-per-task=1 $ACCT \
    --wrap "source /home/rodrigok/.bashrc; module load python/3.11.1 gcc/13.3.0; source $NOVAE_ENV/bin/activate; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; cd $CODE; python novae_INDEPENDENT_FINAL.py aggregate; rc=\$?; exit \$rc") \
  || { echo "ABORT: novae INDEP aggregate submit failed"; exit 5; }
record NOVAE_INDEP_AGG "$J_INDEP_AGG" "afterany:$J_INDEP" "stitch per-sample summary (novae_env)"

J_ASTRO=$(submit --dependency=afterok:$J_CONCAT run_astro_sanity_FINAL.sh) \
  || { echo "ABORT: astro submit failed"; exit 5; }
record ASTRO "$J_ASTRO" "afterok:$J_CONCAT" "run_astro_sanity_FINAL.sh -> clustered_full.h5ad"

if $RUN_ASTRO_TSNE; then
  J_TPREP=$(submit --dependency=afterok:$J_ASTRO run_tsne_sweep_prep_FINAL.sh) \
    || { echo "ABORT: astro tsne prep submit failed"; exit 5; }
  record ASTRO_TSNE_PREP "$J_TPREP" "afterok:$J_ASTRO" "200k subsample for the sweep"
  J_TSWEEP=$(submit --dependency=afterok:$J_TPREP run_tsne_sweep_FINAL.sh) \
    || { echo "ABORT: astro tsne sweep submit failed"; exit 5; }
  record ASTRO_TSNE_SWEEP "$J_TSWEEP" "afterok:$J_TPREP" "array 0-23%4 (GPU); param sweep PNGs"
fi

if $RUN_CT; then
  J_CT_PIPE=$(submit --dependency=afterok:$J_CONCAT run_ct_pipe_FINAL.sh) \
    || { echo "ABORT: celltyping pipe submit failed"; exit 5; }
  record CT_PIPE "$J_CT_PIPE" "afterok:$J_CONCAT" "obs_celltype.csv + X_pca.npy + celltyped h5ad"
  J_CT_TSNE=$(submit --dependency=afterok:$J_CT_PIPE run_ct_tsne_FINAL.sh) \
    || { echo "ABORT: celltyping tsne submit failed"; exit 5; }
  record CT_TSNE "$J_CT_TSNE" "afterok:$J_CT_PIPE" "X_tsne.npy (opentsne_gpu)"
  J_CT_DOT=$(submit --dependency=afterok:$J_CT_PIPE run_ct_dotpersample_FINAL.sh) \
    || { echo "ABORT: celltyping dotplots submit failed"; exit 5; }
  record CT_DOTPLOTS "$J_CT_DOT" "afterok:$J_CT_PIPE" "per-sample dotplots"
  J_CT_PLOTS=$(submit --dependency=afterok:$J_CT_TSNE run_ct_plots_FINAL.sh) \
    || { echo "ABORT: celltyping plots submit failed"; exit 5; }
  record CT_PLOTS "$J_CT_PLOTS" "afterok:$J_CT_TSNE" "cohort composition + tSNE plots"
  if $FULL; then
    J_CT_HERO=$(submit --dependency=afterok:$J_CT_PIPE run_ct_heroes_FINAL.sh) \
      || { echo "ABORT: celltyping heroes submit failed"; exit 5; }
    record CT_HEROES "$J_CT_HERO" "afterok:$J_CT_PIPE" "is_MN + boundary(C2)-gated MN heroes"
  fi
fi

echo
echo "===== STAGE 2: overlays | coloc[FULL] | dpqc ====="
J_OVERLAYS=$(submit --dependency=afterok:$J_JOINT run_overlays_FINAL.sh) \
  || { echo "ABORT: overlays submit failed"; exit 5; }
record OVERLAYS "$J_OVERLAYS" "afterok:$J_JOINT" "array 0-19%6; domain/gene overlays (boundary C2 caveat)"

if $FULL; then
  J_COLOC=$(submit --dependency=afterok:$J_JOINT run_gene_coloc_FINAL.sh) \
    || { echo "ABORT: coloc submit failed"; exit 5; }
  record COLOC "$J_COLOC" "afterok:$J_JOINT" "array 0-2%3 (MNX1/BCL6/STMN2); is_MN-gated"
  J_GENECMP=$(submit --dependency=afterok:$J_COLOC --job-name=gene_coloc_cmp_FINAL \
      --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err" \
      --time=00:20:00 --mem=16G --cpus-per-task=2 $ACCT \
      --wrap "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; cd $CODE; $VIS_PY gene_coloc_compare_fig_FINAL.py $COLOC_OUT; rc=\$?; exit \$rc") \
    || { echo "ABORT: gene-comparison submit failed"; exit 5; }
  record GENE_COMPARISON "$J_GENECMP" "afterok:$J_COLOC" "compare fig (self-skips if is_MN-pending)"
fi

# dpqc depends on novae-INDEP (afterok) + celltyping pipe (afterok) when typing is in the run
DPQC_DEP="afterok:$J_INDEP"
$RUN_CT && DPQC_DEP="afterok:$J_INDEP:$J_CT_PIPE"
J_DPQC_ARR=$(submit --dependency=$DPQC_DEP run_dpqc_array_FINAL.sh) \
  || { echo "ABORT: dpqc array submit failed"; exit 5; }
record DPQC_ARRAY "$J_DPQC_ARR" "$DPQC_DEP" "array 0-19%6; per-sample dual-pass QC"

echo
echo "===== STAGE 3: dpqc aggregate = Cohort/Meeting/Audio/Strategy PDFs (listening.io audio) ====="
J_DPQC_AGG=$(submit --dependency=afterany:$J_DPQC_ARR run_dpqc_aggregate_FINAL.sh) \
  || { echo "ABORT: dpqc aggregate submit failed"; exit 5; }
record DPQC_AGGREGATE "$J_DPQC_AGG" "afterany:$J_DPQC_ARR" "cohort PDFs + audio .txt/.pdf"

# =============================================================================================
echo
echo "================================================================"
if $DRYRUN; then
  echo "DRY RUN COMPLETE -- nothing was submitted. Preview manifest -> $SUBMITTED"
else
  echo "ALL SUBMITTED. Manifest -> $SUBMITTED"
fi
echo "----------------------------------------------------------------"
cat "$SUBMITTED"
echo "----------------------------------------------------------------"
if ! $DRYRUN; then
  echo "Current queue:"
  squeue -u "${USER:-rodrigok}" -o "%.14i %.24j %.9T %.10M %R" 2>/dev/null || true
fi

if $FULL; then
  echo
  echo "MANUAL post-steps (need human input / Marcel's boundary re-export -- NOT auto-submitted):"
  echo "  * matched_triple_panel_FINAL.py --sample <LABEL> --out <dir> [--cx <um> --cy <um>]"
  echo "      hand-placed 3-gene (MNX1/STMN2/CE_STMN2) hero -- pick the crop centre by eye."
  echo "  * All hero/boundary figures (overlays MN heroes, coloc heroes, celltype heroes) render"
  echo "      real polygons only after Marcel re-exports full-coverage integer-id boundaries (C2)."
fi
echo "================================================================"
$DRYRUN && rm -f "$_DRYCNT"
exit 0
