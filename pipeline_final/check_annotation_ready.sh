#!/bin/bash
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/check_annotation_ready.sh
#
# The gate in front of a full run. Exits 0 only when all 20 samples have motor-neuron
# annotation, either as a per-sample *_cell_classification_MN.csv in the _final delivery or as
# a CellAnnot clustered h5ad. Otherwise it exits 1 and prints exactly which samples are
# missing.
#
# It reads its sample list from final_config so it cannot disagree with the pipeline about
# what the cohort is.
#
# As of the last check it returns 1, with only the SD03522 prototype present, and that
# prototype is built on the older gm-id segmentation rather than _final. Passing this script
# is necessary but not sufficient: also confirm the join reports cell_index+1, an overlap near
# 1.0, is_MN_provenance='row-count-ok', roughly 150 to 500 motor neurons per sample, and get
# Marcel's written confirmation that the annotation was built on _final.
# ========================================================================================

# check_annotation_ready.sh -- is Marcel's _final MN annotation present for the WHOLE cohort?
# ---------------------------------------------------------------------------------------------
# READY iff EVERY discovered sample (the authoritative final_config.discover_samples() 20-sample
# cohort -- same dedup as the concat builder) has an MN annotation in EITHER accepted form:
#     (a) CellAnnot/clustered_<LABEL>_res0.5.h5ad            (h5ad prototype form), OR
#     (b) <_final sample dir>/*_cell_classification_MN.csv   (per-sample CSV form).
# Prints a per-sample table + the missing list; exits 0 = READY, 1 = NOT READY, 2 = enumerate error.
#
# Used two ways:
#   * by SUBMIT_ALL_FINAL.sh as the launch gate for the is_MN-dependent stages (coloc / gene-
#     comparison / MN + celltype heroes);
#   * standalone by the PI to poll status: bash check_annotation_ready.sh
#
# STAGING NOTE: this ONLY READS the filesystem (dir listing + file existence). It runs nothing,
#   submits nothing, and writes nothing. It is a readiness probe, not an analysis.
#
# CAVEAT: presence != correctness. The EXACT _final is_MN JOIN KEY (integer cell_id vs the old
#   'xenium_cell_id'/gm-namespace) is only settled once Marcel regenerates the annotation ON the
#   _final segmentation. Confirm that with Marcel at launch; VH_concatenate_FINAL's join has a
#   row-count provenance guard, but this probe checks presence only.
set -uo pipefail

CODE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code
PY=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/pertpy_env/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

# ---- Enumerate the cohort via the single source of truth (final_config) -----------------------
# Emits: "META<TAB>expected_count<TAB>cellannot_dir"
#   then "S<TAB>label<TAB>cellannot_h5ad<TAB>final_sample_dir" per discovered sample.
MAP=$("$PY" - "$CODE" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
try:
    import final_config as fc
except Exception as e:
    sys.stderr.write("FATAL: cannot import final_config from %s: %s: %s\n"
                     % (sys.argv[1], type(e).__name__, e))
    sys.exit(9)
ca = fc.SPATIAL_ROOT / "CellAnnot"
try:
    exp = int(fc.EXPECTED_SAMPLE_COUNT)
except Exception:
    exp = 20
pairs = sorted(fc.discover_samples(verbose=False))
print("META\t%d\t%s" % (exp, ca))
for lab, p in pairs:
    print("S\t%s\t%s\t%s" % (lab, ca / ("clustered_%s_res0.5.h5ad" % lab), p))
PYEOF
)
rc=$?
if [ $rc -ne 0 ] || [ -z "$MAP" ]; then
  echo "check_annotation_ready: ERROR enumerating the cohort via final_config (rc=$rc)." >&2
  echo "  (is $PY OK, and does $CODE/final_config.py import + RANGER_FINAL exist?)" >&2
  echo "NOT READY"
  exit 2
fi

EXPECT=20
CA_DIR="?"
declare -a LABELS H5S DIRS
while IFS=$'\t' read -r tag a b c; do
  case "$tag" in
    META) EXPECT="$a"; CA_DIR="$b" ;;
    S)    LABELS+=("$a"); H5S+=("$b"); DIRS+=("$c") ;;
  esac
done <<< "$MAP"

N=${#LABELS[@]}
echo "================================================================"
echo "MN ANNOTATION READINESS   ($(date))"
echo "  CellAnnot dir : $CA_DIR"
echo "  discovered    : $N sample(s)   (expected $EXPECT)"
echo "  forms accepted: CellAnnot/clustered_<LABEL>_res0.5.h5ad   OR"
echo "                  <_final sample dir>/*_cell_classification_MN.csv"
echo "----------------------------------------------------------------"

ready=0
declare -a MISSING=()
for i in "${!LABELS[@]}"; do
  lab="${LABELS[$i]}"; h5="${H5S[$i]}"; d="${DIRS[$i]}"
  form=""
  if [ -f "$h5" ]; then
    form="h5ad"
  elif compgen -G "$d/*_cell_classification_MN.csv" >/dev/null 2>&1; then
    form="csv"
  fi
  if [ -n "$form" ]; then
    ready=$((ready+1)); printf "  [READY %-4s] %s\n" "$form" "$lab"
  else
    MISSING+=("$lab"); printf "  [MISSING  ] %s\n" "$lab"
  fi
done

echo "----------------------------------------------------------------"
echo "  ready $ready / $N"
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "  MISSING (${#MISSING[@]}): ${MISSING[*]}"
fi
echo "================================================================"

# Cohort-integrity: if discovery != expected, the annotation question is moot -- resolve first.
if [ "$N" -ne "$EXPECT" ]; then
  echo "NOT READY  (discovered $N != expected $EXPECT -- cohort-integrity issue upstream; fix before launch)"
  exit 1
fi
if [ ${#MISSING[@]} -eq 0 ] && [ "$ready" -eq "$N" ]; then
  echo "READY  (all $N samples carry an MN annotation)"
  echo "  REMINDER: presence != correctness -- confirm the regenerated _final is_MN JOIN KEY with"
  echo "            Marcel (integer cell_id vs xenium_cell_id) at launch. This is the standing gate."
  exit 0
fi
echo "NOT READY  (${#MISSING[@]} sample(s) still lack an MN annotation)"
exit 1
