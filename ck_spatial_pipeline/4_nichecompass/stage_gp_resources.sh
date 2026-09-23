#!/bin/bash
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/stage_gp_resources.sh
#
# Stages the gene-programme resources once into NicheCompass_SC/gene_programs/: NicheNet
# networks, OmniPath and the MEBOCOST metabolite tables at GitHub tag 0.3.3. The v2 trainer
# requires this cache (NC_REQUIRE_GP_CACHE is on by default) and never fetches anything
# live. Idempotent.
# ========================================================================================

# ------------------------------------------------------------------------------ #
# stage_gp_resources.sh — idempotent GP-resource staging for the NicheCompass     #
# ALS spinal-cord Xenium run (SCG port of the Sanger LSF pipeline).               #
#                                                                                 #
# What it does:                                                                   #
#   1) mkdir -p the oak output scaffold (gene_programs/, gene_annotations/,       #
#      artifacts/sample_integration/, results/) and the SLURM logs dir.           #
#   2) Stages the MEBOCOST enzyme/sensor TSVs into                                #
#      gene_programs/metabolite_enzyme_sensor_gps/. These are LOCAL FILES that    #
#      ship in the NicheCompass GitHub repo (NOT in the wheel), pinned to the     #
#      "0.3.3" git tag matching the installed package (the "v0.3.3" tag does      #
#      not exist). Files already present, non-empty AND verified are skipped;     #
#      missing/empty/corrupt files are (re-)downloaded atomically                 #
#      (.part.$$ -> mv).                                                          #
#                                                                                 #
# NOT staged here: OmniPath / NicheNet networks. NC_SC_gp_precheck.py fetches     #
# them live on its first run (compute nodes have outbound HTTPS) and saves CSVs   #
# under gene_programs/ for load_from_disk=True reruns.                            #
#                                                                                 #
# Runs ON SCG (login or batch node). Plain bash + curl/coreutils; no pip, no      #
# module loads. Verification gate = exact line counts + first header field,      #
# both pinned to the 0.3.3 tag content. STAGE_GP_SKIP_LINECOUNT=1 relaxes the     #
# line-count check (header + non-empty checks always apply).                      #
# ------------------------------------------------------------------------------ #

set -euo pipefail

# ==== Config ================================================================== #
NC_BASE=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC
GP_DIR="${NC_BASE}/gene_programs"
MEBOCOST_DIR="${GP_DIR}/metabolite_enzyme_sensor_gps"
GA_DIR="${NC_BASE}/gene_annotations"
ARTIFACTS_DIR="${NC_BASE}/artifacts/sample_integration"
RESULTS_DIR="${NC_BASE}/results"
LOGS_DIR=/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso/logs

# MEBOCOST TSVs pinned to the git tag that matches the installed nichecompass 0.3.3
RAW_BASE="https://raw.githubusercontent.com/Lotfollahi-lab/nichecompass/0.3.3/data/gene_programs/metabolite_enzyme_sensor_gps"
RAW_BASE_FALLBACK="https://raw.githubusercontent.com/Lotfollahi-lab/nichecompass/main/data/gene_programs/metabolite_enzyme_sensor_gps"

ENZYMES_TSV="human_metabolite_enzymes.tsv"
ENZYMES_LINES=6882          # wc -l on the 0.3.3 tag content (verified from SCG)
ENZYMES_HDR="metabolite"    # first tab-separated header field

SENSORS_TSV="human_metabolite_sensors.tsv"
SENSORS_LINES=441
SENSORS_HDR="ID"

CURL_OPTS=(-fSL --retry 3 --retry-delay 5 --max-time 120)

echo "== stage_gp_resources | host $(hostname) | $(date) =="

# ==== 1) Output scaffold (idempotent) ========================================= #
echo "-- 1) output dirs --"
for d in "${MEBOCOST_DIR}" "${GA_DIR}" "${ARTIFACTS_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"; do
    mkdir -p "${d}"
    echo "dir ok: ${d}"
done

# Clean stale partial downloads left by a crashed earlier run
find "${MEBOCOST_DIR}" -maxdepth 1 -name '*.part.*' -print -delete 2>/dev/null || true

# ==== 2) MEBOCOST TSV staging ================================================= #
verify_tsv () {
    # verify_tsv <path> <expected_lines> <expected_first_header_field>
    local path="$1" want_lines="$2" want_hdr="$3"
    local got_lines got_hdr
    if [[ ! -s "${path}" ]]; then
        echo "  verify: ${path} missing or empty"
        return 1
    fi
    got_hdr=$(head -1 "${path}" | cut -f1)
    if [[ "${got_hdr}" != "${want_hdr}" ]]; then
        echo "  verify: ${path} header field1='${got_hdr}' (expected '${want_hdr}')"
        return 1
    fi
    got_lines=$(wc -l < "${path}")
    if [[ "${STAGE_GP_SKIP_LINECOUNT:-0}" != "1" && "${got_lines}" -ne "${want_lines}" ]]; then
        echo "  verify: ${path} has ${got_lines} lines (expected ${want_lines};" \
             "set STAGE_GP_SKIP_LINECOUNT=1 to override)"
        return 1
    fi
    return 0
}

stage_tsv () {
    # stage_tsv <filename> <expected_lines> <expected_first_header_field>
    local fname="$1" want_lines="$2" want_hdr="$3"
    local dest="${MEBOCOST_DIR}/${fname}"
    local url="${RAW_BASE}/${fname}"
    local part="${dest}.part.$$"

    if verify_tsv "${dest}" "${want_lines}" "${want_hdr}" > /dev/null 2>&1; then
        echo "SKIP (already staged + verified): ${dest}"
        return 0
    fi

    if [[ -s "${dest}" ]]; then
        echo "RESTAGE: ${dest} exists but failed verification — re-downloading:"
        verify_tsv "${dest}" "${want_lines}" "${want_hdr}" || true
    else
        echo "STAGE: ${dest} (missing or empty) <- ${url}"
    fi

    if ! curl "${CURL_OPTS[@]}" -o "${part}" "${url}"; then
        rm -f "${part}"
        echo "ERROR: download failed: ${url}" >&2
        echo "       Manual fallback (verify content first — unpinned branch):" >&2
        echo "       ${RAW_BASE_FALLBACK}/${fname}" >&2
        return 1
    fi

    if ! verify_tsv "${part}" "${want_lines}" "${want_hdr}"; then
        rm -f "${part}"
        echo "ERROR: downloaded file failed verification, not installed: ${url}" >&2
        return 1
    fi

    mv -f "${part}" "${dest}"
    echo "STAGED: ${dest}"
    return 0
}

echo "-- 2) MEBOCOST enzyme/sensor TSVs --"
stage_tsv "${ENZYMES_TSV}" "${ENZYMES_LINES}" "${ENZYMES_HDR}"
stage_tsv "${SENSORS_TSV}" "${SENSORS_LINES}" "${SENSORS_HDR}"

# ==== 3) Summary ============================================================== #
echo "-- 3) staged file summary --"
wc -l "${MEBOCOST_DIR}/${ENZYMES_TSV}" "${MEBOCOST_DIR}/${SENSORS_TSV}"
ls -la "${MEBOCOST_DIR}"
echo "NOTE: OmniPath/NicheNet network CSVs are created by NC_SC_gp_precheck.py"
echo "      on its first run under: ${GP_DIR}"
echo "== stage_gp_resources done (all staged + verified) | $(date) =="
