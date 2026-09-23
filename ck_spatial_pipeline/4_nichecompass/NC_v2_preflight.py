#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/NC_v2_preflight.py
#
# A CPU-only data contract check before any GPU time: labels, the cell_type_mn contract and
# section discovery. It writes 20260902_ncv2_tag_manifest.json, the only source of the
# section list. The v3 retrain pins to that manifest.
# ========================================================================================

"""
NC_v2_preflight.py — go/no-go data-contract gate and TAG-MANIFEST generator for
the four NicheCompass v2 configurations (runstamp 20260902_ncv2) over the 20
QCed ALS spinal-cord Xenium pass2 sections.

WHY THIS EXISTS
    40 section-runs (C1 20 + C2 17) plus 2 integrated runs will be fired at a
    shared H200 node under a QOS that allows only 6 submitted GPU jobs and 2
    concurrent GPUs. Every avoidable abort is expensive, and a silently wrong
    label column would poison all 40 downstream surfaces. So every contract the
    trainer and the downstream depend on is checked HERE, once, on CPU, in
    seconds — before a single GPU second is spent.

    It is also the ONLY source of truth for array-index -> TAG mapping. The
    sbatch files and the trainer read the manifest it writes; nothing else
    hardcodes a tag list.

WHAT IT READS (h5py, read-only)
    <pass2-dir>/ALS_SCXenium_<TAG>_pass2.h5ad  for the 20 canonical TAGs.
    X data is NEVER read — only its `shape` and `encoding-type` attributes, to
    cross-check n_obs and to assert it is sparse (csr/csc). The counts layer's
    `data` buffer IS read in full, in 1M-element chunks (<= 17 MB per section
    today), so the integer/non-negativity gate is exhaustive, not a sample.

WHAT IT WRITES
    <runs-base>/<runstamp>_tag_manifest.json          (the tag manifest; the
                                                       path is overridable with
                                                       --manifest)
    <manifest-copy>/<runstamp>_tag_manifest.json      (identical copy, scripts dir)
    <runs-base>/<config>/<runstamp>/                  (the four run roots, mkdir -p)
    Nothing else. It never touches Ranger_procd/, NicheCompasso/, or
    artifacts/sample_integration/latest. The two manifests are written in one
    two-phase step: <path>.tmp.<pid> is staged next to EACH target first, then
    both are os.replace()d (a pre-existing manifest is held as
    <path>.prev.<pid> during the swap and removed on success). A leftover
    *.tmp.<pid> / *.prev.<pid> therefore means a hard kill mid-write, never a
    handled failure.

EXIT CONTRACT (drives every sbatch that follows)
    exit 0 — every gate passed; a one-screen summary table is printed and the
             manifest is on disk at BOTH targets. The gate ledger inside the
             manifest is the complete ledger: its gates_summary is the same
             string the final GO line prints.
    exit 1 — at least one gate FAILED, or the manifest could not be written to
             BOTH targets. Every failure is listed loudly at the end. NO
             manifest is left behind by a failed run: gate failures never reach
             the write, and a failed write rolls back every temp, every target
             it had already replaced and every backup, so the filesystem is
             exactly as it was (a stale manifest from an earlier run is
             reported, never silently reused, never deleted).

HOW THE SBATCH FILES CONSUME THE MANIFEST (tested verbatim, 2026-09-02)
    MANIFEST=/oak/.../NicheCompass_SC/runs_v2/20260902_ncv2_tag_manifest.json

    # one TAG per array task (C1 reads tags_all20, C2 reads tags_vh17)
    TAG=$("${PYBIN}" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m[sys.argv[2]][int(sys.argv[3])])' "${MANIFEST}" tags_all20 "${SLURM_ARRAY_TASK_ID}")

    # chunked array (the QOS-legal shape: 6 tasks, each looping over its slice)
    TAGS=$("${PYBIN}" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(" ".join(m[sys.argv[2]][int(sys.argv[3])::int(sys.argv[4])]))' "${MANIFEST}" tags_all20 "${SLURM_ARRAY_TASK_ID}" 6)
    #   -> task 0 gets SD00614BG SD01620BI SD02622BI SD04219BI, task 2 gets 3 tags, etc.

    # per-config metadata, so nothing downstream re-derives a tag list:
    #   manifest["configs"]["persample_vh"] -> {"tag_list_key", "n_runs",
    #       "n_cells_total", "run_dir", "per_tag_run_dir_pattern",
    #       "cat_covariates_keys", "cat_covariates_embeds_nums"}
    #   manifest["label_contract"]["cell_type_mn_classes"] -> the fixed 10-class
    #       list, in the order every legend/crosstab must use
    #   manifest["per_tag"][TAG] -> n_obs, n_in_vh_v2, n_is_mn_v2, n_na,
    #       cell_type_mn_counts_whole, cell_type_mn_counts_vh, fingerprint, ...
    #       (so a run's gates can be derived here instead of hardcoded there)

CPU-only. No matplotlib (this script draws nothing), no torch, no nichecompass
import — so it runs in seconds and needs no GPU, no CUDA and no H200 openssl
shim. Only h5py + numpy + pandas are imported.
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version

import h5py
import numpy as np
import pandas as pd

# SLURM logs are block-buffered files: force line buffering so progress is
# visible in logs/*.out while the job runs (NC_SC_gp_precheck.py convention).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

SCRIPT_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def section(title: str) -> None:
    print("", flush=True)
    log("#" * 78)
    log(f"### {title}")
    log("#" * 78)


# --------------------------------------------------------------------------- #
# Gate ledger (pattern ported from NC_SC_downstream_v2_ismn.py)                #
# --------------------------------------------------------------------------- #
class Gates:
    """Collect-all-then-verdict ledger.

    A preflight must report EVERY broken contract in one run, not die on the
    first one, so failures are accumulated and the verdict is deferred.
    """

    def __init__(self):
        self.rows = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        verdict = "PASS" if ok else "*** FAIL ***"
        log(f"GATE {name:<24} {verdict:<12} | {detail}")
        self.rows.append({"gate": name, "passed": ok, "detail": detail})
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if not r["passed"]]

    @property
    def all_passed(self) -> bool:
        return len(self.failed) == 0

    def summary(self) -> str:
        return (f"{sum(1 for r in self.rows if r['passed'])}/{len(self.rows)} "
                "gates passed")

    def as_list(self):
        return list(self.rows)


# --------------------------------------------------------------------------- #
# Contract constants                                                           #
# --------------------------------------------------------------------------- #
# The 20 QCed sections. File TAG != obs label: 'SD00614BG' on disk,
# 'SD00614_BG' in obs['sample'] — labels are ALWAYS taken from obs, never files.
# Verbatim from NC_SC_training_all.py so the two scripts cannot drift.
SAMPLE_TAGS = (
    "SD00614BG", "SD01015BG", "SD01115BG", "SD01320BG", "SD01413BA",
    "SD01616BI", "SD01620BI", "SD01623BI", "SD01915BG", "SD01922BI",
    "SD01923BI", "SD02022BI", "SD02622BI", "SD02818BG", "SD02913BA",
    "SD03522BG", "SD03614BG", "SD03914BG", "SD04219BI", "SD05413BG",
)

TAG_PATTERN = re.compile(r"^(SD\d+)([A-Z]+)$")  # 'SD00614BG' -> 'SD00614_BG'

# obs columns the four configurations and every downstream surface depend on
REQUIRED_OBS_COLS = (
    "sample", "status", "cell_type",
    "is_MN", "is_MN_v1", "is_MN_v2",
    "in_VH_v2", "has_vh_v2", "vh_v2_source",
)

# HARD cohort gates (post-canonicalisation audit, 2026-09-02; every one of these
# was re-derived from the files today and the arithmetic closes)
EXPECTED_N_CELLS = 1_097_669
EXPECTED_N_GENES = 480
EXPECTED_IS_MN_V2_TRUE = 844
EXPECTED_IN_VH_V2_TRUE = 75_229
EXPECTED_NA_CELLS = 122_902          # = the 3 excluded sections, exactly
EXPECTED_EXCLUDED_TAGS = ("SD01015BG", "SD01616BI", "SD02913BA")

# Warn-only derived cohort figures (useful, but not part of the hard contract)
SNAPSHOT_IS_MN_V1_TRUE = 774
SNAPSHOT_COARSE_MN = 101_660         # transcriptional "Motor neurons" class
SNAPSHOT_PROMOTED_FROM_COARSE = 472  # is_MN_v2 True that WERE coarse MN
SNAPSHOT_PROMOTED_FROM_OTHER = 372   # is_MN_v2 True that were glia/other

# The 9 coarse transcriptional classes, byte-identical and in this ORDER in all
# 20 files (h5py-verified 2026-09-02). FIXED_CLASSES_10 is derived from it.
CELL_TYPE_CATEGORIES = (
    "Astrocytes", "Endothelial", "Excitatory neurons", "Inhibitory neurons",
    "Macrophages", "Microglia", "Motor neurons", "OPCs", "Oligodendrocytes",
)
TRANSCRIPTIONAL_MN_CAT = "Motor neurons"  # the ~101k-cell coarse cluster
OTHER_NEURONS = "Other neurons"           # what it is renamed to
MN_CLASS = "Motor neurons"                # what is_MN_v2 True cells become

# THE LABEL CONTRACT's class order. Motor neurons LAST so the 844-cell sliver
# draws on top of every stacked bar (NC_SC_downstream_v2_ismn.py precedent).
FIXED_CLASSES_10 = tuple(
    [c for c in CELL_TYPE_CATEGORIES if c != TRANSCRIPTIONAL_MN_CAT]
    + [OTHER_NEURONS, MN_CLASS]
)

VH_SOURCE_VOCAB = ("manual", "robin", "excluded")
STATUS_VOCAB = ("control", "sporadic", "c9")

# The four configurations and which tag list each one runs over
CONFIGS = {
    "persample_whole": {
        "tag_list_key": "tags_all20", "scope": "whole",
        "mode": "persample",
        "what": "one model per section, whole section",
    },
    "persample_vh": {
        "tag_list_key": "tags_vh17", "scope": "vh",
        "mode": "persample",
        "what": "one model per section, in_VH_v2 == True cells only",
    },
    "integrated_whole": {
        "tag_list_key": "tags_all20", "scope": "whole",
        "mode": "integrated",
        "what": "one integrated model, 20 whole sections, cat_covariates ['sample']",
    },
    "integrated_vh": {
        "tag_list_key": "tags_vh17", "scope": "vh",
        "mode": "integrated",
        "what": "one integrated model, 17 VH subsets, cat_covariates ['sample']",
    },
}

# GP resources: every one must already be cached so all four configs take the
# load_from_disk branch with ZERO downloads. Paths relative to gene_programs/.
GP_CACHE_RELPATHS = (
    "omnipath_lr_network.csv",
    "nichenet_lr_network.csv",
    "nichenet_ligand_target_matrix.csv",
    "metabolite_enzyme_sensor_gps/human_metabolite_enzymes.tsv",
    "metabolite_enzyme_sensor_gps/human_metabolite_sensors.tsv",
)

# A VH subset must survive kneighbors_graph(n_neighbors=4); the trainer raises
# if n_obs <= n_neighbors. Smallest real VH subset today is 931 cells.
N_NEIGHBORS = 4
VH_MIN_CELLS_HARD = N_NEIGHBORS + 1
VH_MIN_CELLS_WARN = 500

# Warn-only per-tag snapshot of the 2026-09-02 audit. Its job is diagnostic: if
# a cohort total fails, this tells you WHICH section drifted instead of leaving
# you with "844 != 806". Never a hard gate — upstream may legitimately
# re-annotate, and the hard gates above are the contract.
#         TAG:        (n_obs,  vh_true, mn_v2, coarse_mn, status,     vh_source)
SNAPSHOT_PER_TAG = {
    "SD00614BG": (50432,  6493,  77,  5092, "control",  "manual"),
    "SD01015BG": (28955,     0,   0,  2010, "control",  "excluded"),
    "SD01115BG": (52650,  2783,  40,  5356, "control",  "manual"),
    "SD01320BG": (86552,  8095,  65,  5089, "c9",       "manual"),
    "SD01413BA": (54685,  3813,  64,  3859, "control",  "manual"),
    "SD01616BI": (43437,     0,   0,  4541, "sporadic", "excluded"),
    "SD01620BI": (66667,  5207, 109, 10660, "c9",       "manual"),
    "SD01623BI": (57332,  7772,  53,  4738, "sporadic", "manual"),
    "SD01915BG": (87520, 10194,  85,  7914, "control",  "manual"),
    "SD01922BI": (25962,   941,  15,  2530, "sporadic", "manual"),
    "SD01923BI": (17912,   931,  16,  2587, "sporadic", "robin"),
    "SD02022BI": (54460,  4700,  73,  8459, "c9",       "manual"),
    "SD02622BI": (41467,  1474,   5,  3059, "sporadic", "manual"),
    "SD02818BG": (58349,  3955,  41,  5776, "control",  "manual"),
    "SD02913BA": (50510,     0,   0,  5633, "control",  "excluded"),
    "SD03522BG": (106698, 5410,  47,  6660, "sporadic", "manual"),
    "SD03614BG": (39379,  1460,  31,  3384, "control",  "manual"),
    "SD03914BG": (74235,  5860,  65,  5789, "control",  "robin"),
    "SD04219BI": (57026,  3932,  44,  4755, "c9",       "robin"),
    "SD05413BG": (43441,  2209,  14,  3769, "control",  "manual"),
}

# The counts `data` buffer is scanned EXHAUSTIVELY in chunks of this many
# elements (float32 -> 4 MB per chunk; the largest section today has 4.4M nnz,
# i.e. 5 chunks / 17 MB). A head slice of a CSC matrix only ever sees the first
# few GENES, and a strided sample misses any isolated non-integer, so neither
# can certify the buffer — and certifying it is this gate's whole job.
COUNTS_CHUNK_ELEMS = 1_000_000

# nichecompass 0.3.3 needs a SPARSE X (models/nichecompass.py computes
# features_scale_factors_ from adata.X.sum(0)[0], which only works on the
# np.matrix row a sparse sum returns) and reads layers['counts'] as its
# reconstruction target; both must be one of these anndata encodings.
SPARSE_ENCODINGS = ("csr_matrix", "csc_matrix")

DEFAULT_PASS2_DIR = ("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/"
                     "Spatial/Ranger_procd")
DEFAULT_BASE_FOLDER = ("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/"
                       "Spatial/NicheCompass_SC")
DEFAULT_RUNS_BASE = f"{DEFAULT_BASE_FOLDER}/runs_v2"
DEFAULT_GP_DIR = f"{DEFAULT_BASE_FOLDER}/gene_programs"
DEFAULT_MANIFEST_COPY = "/home/rodrigok/SLURM_jobs/Spatial/NicheCompasso_v2"
DEFAULT_RUNSTAMP = "20260902_ncv2"


# --------------------------------------------------------------------------- #
# h5py helpers (idioms verified live on these files 2026-09-02)                 #
# --------------------------------------------------------------------------- #
def _as_str(value, default=""):
    """h5py attrs come back as bytes on some writers and str on others."""
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return _as_str(value.ravel()[0] if value.size else None, default)
    return str(value)


def read_categorical(obj):
    """Read a label column -> (categories list[str], codes np.ndarray).

    All ten label columns in these files are anndata `categorical` groups
    (h5py-verified today), but a plain string dataset is accepted too so that a
    benign upstream encoding change degrades into a normal read rather than
    taking out the whole probe of that section.
    """
    if isinstance(obj, h5py.Group) and "categories" in obj:
        raw = obj["categories"][:]
        cats = [c.decode("utf-8") if isinstance(c, bytes) else str(c)
                for c in raw]
        return cats, np.asarray(obj["codes"][:])
    arr = np.asarray(obj[:])
    values = [v.decode("utf-8") if isinstance(v, bytes) else str(v)
              for v in arr]
    cats = sorted(set(values))
    lookup = {c: i for i, c in enumerate(cats)}
    return cats, np.asarray([lookup[v] for v in values], dtype=np.int64)


def read_bool_column(obs, name):
    """Read a boolean obs column -> (values, na_mask, kind).

    Handles BOTH encodings present in these files: `nullable-boolean` groups
    (values+mask; is_MN, is_MN_v1, is_MN_v2, in_VH_v2) and plain bool datasets
    (has_vh_v2). na_mask True == pd.NA.

    Nothing here ever coerces pd.NA: on this env (anndata 0.12 / pandas 2.3)
    np.asarray(series).astype(bool) RAISES on the all-NA excluded sections, and
    the only silent-and-correct pandas paths are .fillna(False) /
    to_numpy(na_value=False). Working on the raw values+mask arrays sidesteps
    the whole class of bug.
    """
    obj = obs[name]
    if isinstance(obj, h5py.Group) and "values" in obj and "mask" in obj:
        values = np.asarray(obj["values"][:]).astype(bool)
        mask = np.asarray(obj["mask"][:]).astype(bool)
        return values, mask, "nullable-boolean"
    arr = np.asarray(obj[:])
    if arr.dtype.kind == "b":
        return arr.astype(bool), np.zeros(arr.shape[0], dtype=bool), "plain-bool"
    raise TypeError(
        f"obs['{name}'] is neither a nullable-boolean group nor a bool dataset "
        f"(dtype {arr.dtype}) — refusing to guess its truth semantics.")


def fingerprint(path):
    """size+mtime, so a later job can prove the inputs did not change."""
    st = os.stat(path)
    return {
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def cell_type_mn_counts(cats, codes, mn_true, subset=None):
    """Apply THE LABEL CONTRACT and count the resulting 10 classes.

    Recipe (deterministic, never written back into the pass2 files):
        1. the coarse transcriptional class "Motor neurons" is renamed
           "Other neurons";
        2. every cell with is_MN_v2 == True becomes "Motor neurons", which wins
           over EVERY base class.
    Cells with is_MN_v2 <NA> are simply not "Motor neurons" — they keep their
    (possibly demoted) base class, so the 10 classes tile every cell.

    `subset` restricts the accounting to a boolean row mask (used for the
    in_VH_v2 composition that C2/C4 will actually see). Returns a dict in
    FIXED_CLASSES_10 order.
    """
    if subset is None:
        subset = np.ones(codes.shape[0], dtype=bool)
    out = {c: 0 for c in FIXED_CLASSES_10}
    for idx, cat in enumerate(cats):
        selected = (codes == idx) & subset
        n_total = int(selected.sum())
        n_promoted = int((selected & mn_true).sum())
        if cat == TRANSCRIPTIONAL_MN_CAT:
            out[OTHER_NEURONS] = n_total - n_promoted
        else:
            out[cat] = n_total - n_promoted
    out[MN_CLASS] = int((mn_true & subset).sum())
    return out


# --------------------------------------------------------------------------- #
# Per-file probe                                                               #
# --------------------------------------------------------------------------- #
def probe_section(tag, path):
    """Structural + label probe of one pass2 file. X data is never read.

    Returns a record dict. `open_ok=False` means nothing else in the record can
    be trusted; the caller degrades gracefully instead of crashing so that ONE
    run of the preflight reports every broken file.
    """
    rec = {"tag": tag, "path": path, "open_ok": False, "problems": []}

    if not os.path.isfile(path):
        rec["problems"].append("file missing")
        return rec
    rec["fingerprint"] = fingerprint(path)

    with h5py.File(path, "r") as fh:
        obs = fh["obs"]

        # ---- n_obs from the obs index; n_vars from the var index
        obs_index_key = _as_str(obs.attrs.get("_index"), "_index")
        n_obs = int(obs[obs_index_key].shape[0])
        var = fh["var"]
        var_index_key = _as_str(var.attrs.get("_index"), "_index")
        var_names_raw = var[var_index_key][:]
        var_names = [g.decode("utf-8") if isinstance(g, bytes) else str(g)
                     for g in var_names_raw]
        rec["n_obs"] = n_obs
        rec["n_vars"] = len(var_names)
        rec["var_names_head"] = var_names[:3]
        rec["var_names_tail"] = var_names[-3:]
        rec["_var_names"] = var_names          # popped before the manifest
        rec["n_vars_ok"] = (len(var_names) == EXPECTED_N_GENES)

        # ---- required obs columns
        present = {c: (c in obs) for c in REQUIRED_OBS_COLS}
        rec["missing_obs_cols"] = sorted(c for c, ok in present.items() if not ok)
        rec["obs_cols_ok"] = not rec["missing_obs_cols"]
        if not rec["obs_cols_ok"]:
            rec["problems"].append(
                "missing obs columns: " + ", ".join(rec["missing_obs_cols"]))
            rec["open_ok"] = True
            return rec

        # ---- counts layer: present, sparse (csr/csc), integer-valued,
        #      non-negative, shape (n_obs, 480)
        rec["counts_present"] = ("layers" in fh and "counts" in fh["layers"])
        rec["counts_encoding"] = None
        rec["counts_sparse_ok"] = False
        rec["counts_int_ok"] = False
        rec["counts_nonneg_ok"] = False
        rec["counts_shape_ok"] = False
        if rec["counts_present"]:
            counts = fh["layers"]["counts"]
            if isinstance(counts, h5py.Group):
                rec["counts_encoding"] = (
                    _as_str(counts.attrs.get("encoding-type")) or "group")
            else:
                rec["counts_encoding"] = "array"       # a dense Dataset
            rec["counts_sparse_ok"] = rec["counts_encoding"] in SPARSE_ENCODINGS
            if isinstance(counts, h5py.Group) and "data" in counts:
                shape_attr = np.asarray(counts.attrs.get("shape", [0, 0])).ravel()
                rec["counts_shape"] = [int(x) for x in shape_attr]
                data = counts["data"]
                nnz = int(data.shape[0])
                rec["counts_nnz"] = nnz
                rec["counts_dtype"] = str(data.dtype)
                if "indptr" in counts:
                    rec["counts_indptr_len"] = int(counts["indptr"].shape[0])
                # EXHAUSTIVE chunked pass over the WHOLE data buffer (not a
                # head slice, not a strided sample). Cheap: the whole cohort is
                # 28.7M float32 (~115 MB), 0.2-1.6 s per section on oak when
                # the filesystem is not stalling (measured 2026-09-02; the
                # largest buffer is SD03522BG, 4.43M nnz = 17 MB). Short-
                # circuits only once BOTH defects have been seen, because both
                # flags are then already decided.
                bad_int = bad_neg = False
                n_scanned = 0
                cmin = cmax = None
                for lo in range(0, nnz, COUNTS_CHUNK_ELEMS):
                    chunk = np.asarray(data[lo:lo + COUNTS_CHUNK_ELEMS])
                    if chunk.size == 0:
                        continue
                    n_scanned += int(chunk.size)
                    lo_v, hi_v = float(chunk.min()), float(chunk.max())
                    cmin = lo_v if cmin is None else min(cmin, lo_v)
                    cmax = hi_v if cmax is None else max(cmax, hi_v)
                    if not bad_int:      # NaN != rint(NaN), so NaN fails too
                        bad_int = bool(np.any(chunk != np.rint(chunk)))
                    if not bad_neg:
                        bad_neg = bool(np.any(chunk < 0))
                    if bad_int and bad_neg:
                        break
                rec["counts_scanned_n"] = n_scanned
                rec["counts_scan_exhaustive"] = (n_scanned == nnz)
                rec["counts_int_ok"] = (nnz > 0) and not bad_int
                rec["counts_nonneg_ok"] = (nnz > 0) and not bad_neg
                rec["counts_min"] = cmin
                rec["counts_max"] = cmax
                rec["counts_shape_ok"] = (
                    len(rec["counts_shape"]) == 2
                    and rec["counts_shape"][0] == n_obs
                    and rec["counts_shape"][1] == EXPECTED_N_GENES)
            else:
                rec["problems"].append(
                    "layers['counts'] is not a sparse csr/csc group (encoding "
                    f"{rec['counts_encoding']!r}) — the trainer's counts_key "
                    "target must be sparse")
        else:
            rec["problems"].append(
                "layers['counts'] absent — NicheCompass must NOT fall back to X "
                "(X is log1p-normalised)")

        # ---- X: n_obs cross-check from its shape attribute AND the sparsity
        # gate from its encoding-type attribute (attrs only — X data is never
        # read). A dense X passes every other gate here and then dies inside
        # nichecompass 0.3.3's model init (adata.X.sum(0)[0]) AFTER the full
        # load + block-diagonal graph phase, inside a scarce GPU slot.
        rec["x_shape"] = None
        rec["x_encoding"] = None
        if "X" in fh:
            xobj = fh["X"]
            if isinstance(xobj, h5py.Group):
                rec["x_shape"] = [int(x) for x in
                                  np.asarray(xobj.attrs.get("shape", [0, 0])).ravel()]
                rec["x_encoding"] = (
                    _as_str(xobj.attrs.get("encoding-type")) or "group")
            else:
                rec["x_shape"] = [int(x) for x in xobj.shape]
                rec["x_encoding"] = "array"            # a dense Dataset
        rec["x_sparse_ok"] = rec["x_encoding"] in SPARSE_ENCODINGS
        rec["n_obs_consistent"] = bool(
            rec["x_shape"] is not None
            and len(rec["x_shape"]) == 2
            and rec["x_shape"][0] == n_obs
            and rec["x_shape"][1] == EXPECTED_N_GENES)

        # ---- obsm['spatial']: the kNN input for all four configs. Read in full
        # (26 MB cohort-wide) so a degenerate section aborts here in seconds
        # instead of inside a GPU job's [3/7] load phase. NOTE obsm carries
        # THREE coordinate arrays and only 'spatial' is the per-section micron
        # geometry — 'spatial_grid_offset' and 'spatial_orig' must be dropped.
        rec["obsm_keys"] = sorted(fh["obsm"].keys()) if "obsm" in fh else []
        rec["spatial_present"] = ("obsm" in fh and "spatial" in fh["obsm"])
        rec["spatial_shape_ok"] = False
        rec["spatial_finite_ok"] = False
        if rec["spatial_present"]:
            coords = np.asarray(fh["obsm"]["spatial"][:], dtype=np.float64)
            rec["spatial_shape"] = list(coords.shape)
            rec["spatial_ndim"] = int(coords.ndim)
            rec["spatial_shape_ok"] = bool(
                coords.ndim == 2 and coords.shape[0] == n_obs
                and coords.shape[1] >= 2)
            rec["spatial_finite_ok"] = bool(np.all(np.isfinite(coords)))
            if rec["spatial_shape_ok"]:
                rec["spatial_bbox"] = {
                    "min": [float(x) for x in coords.min(axis=0)],
                    "max": [float(x) for x in coords.max(axis=0)],
                }
                rec["spatial_n_cols"] = int(coords.shape[1])
            del coords
        else:
            rec["problems"].append("obsm['spatial'] absent — no kNN graph input")

        # ---- sample label (authoritative: obs, never the filename). A -1 code
        # is pd.NA: the `0 <= i < len(cats)` filter below DROPS it, so a section
        # with NaN samples would still show "exactly one label". NA codes are
        # therefore counted and ANDed into the flag, exactly as cell_type and
        # vh_v2_source already do (this project has shipped this bug class
        # before: Milo Sex='nan' x71).
        sample_cats, sample_codes = read_categorical(obs["sample"])
        sample_present = sorted({sample_cats[i] for i in np.unique(sample_codes)
                                 if 0 <= i < len(sample_cats)})
        rec["sample_values"] = sample_present
        rec["sample_nan_codes"] = int((sample_codes < 0).sum())
        rec["sample_unique_ok"] = (len(sample_present) == 1
                                   and rec["sample_nan_codes"] == 0)
        rec["sample"] = sample_present[0] if sample_present else None
        match = TAG_PATTERN.match(tag)
        rec["sample_from_tag"] = f"{match.group(1)}_{match.group(2)}" if match else None
        rec["sample_matches_tag"] = (rec["sample"] == rec["sample_from_tag"])

        # ---- status (same -1 code gate)
        status_cats, status_codes = read_categorical(obs["status"])
        status_present = sorted({status_cats[i] for i in np.unique(status_codes)
                                 if 0 <= i < len(status_cats)})
        rec["status_values"] = status_present
        rec["status"] = status_present[0] if len(status_present) == 1 else None
        rec["status_nan_codes"] = int((status_codes < 0).sum())
        rec["status_ok"] = (len(status_present) == 1
                            and status_present[0] in STATUS_VOCAB
                            and rec["status_nan_codes"] == 0)

        # ---- cell_type: vocabulary, order, no NaN codes
        ct_cats, ct_codes = read_categorical(obs["cell_type"])
        rec["cell_type_categories"] = ct_cats
        rec["cell_type_vocab_ok"] = (tuple(ct_cats) == CELL_TYPE_CATEGORIES)
        rec["cell_type_nan_codes"] = int((ct_codes < 0).sum())
        rec["cell_type_no_nan_ok"] = (rec["cell_type_nan_codes"] == 0)
        rec["n_coarse_motor_neurons"] = (
            int((ct_codes == ct_cats.index(TRANSCRIPTIONAL_MN_CAT)).sum())
            if TRANSCRIPTIONAL_MN_CAT in ct_cats else None)

        # ---- vh_v2_source
        src_cats, src_codes = read_categorical(obs["vh_v2_source"])
        src_present = sorted({src_cats[i] for i in np.unique(src_codes)
                              if 0 <= i < len(src_cats)})
        rec["vh_v2_source_values"] = src_present
        rec["vh_v2_source"] = src_present[0] if len(src_present) == 1 else None
        rec["vh_v2_source_nan_codes"] = int((src_codes < 0).sum())
        rec["vh_source_ok"] = (
            len(src_present) == 1
            and src_present[0] in VH_SOURCE_VOCAB
            and rec["vh_v2_source_nan_codes"] == 0)

        # ---- the boolean label columns
        mn_v2_vals, mn_v2_na, mn_v2_kind = read_bool_column(obs, "is_MN_v2")
        mn_vals, mn_na, mn_kind = read_bool_column(obs, "is_MN")
        v1_vals, v1_na, v1_kind = read_bool_column(obs, "is_MN_v1")
        vh_vals, vh_na, vh_kind = read_bool_column(obs, "in_VH_v2")
        has_vals, has_na, has_kind = read_bool_column(obs, "has_vh_v2")
        rec["dtypes"] = {"is_MN_v2": mn_v2_kind, "is_MN": mn_kind,
                         "is_MN_v1": v1_kind, "in_VH_v2": vh_kind,
                         "has_vh_v2": has_kind}

        mn_true = mn_v2_vals & ~mn_v2_na
        vh_true = vh_vals & ~vh_na
        rec["n_is_mn_v2"] = int(mn_true.sum())
        rec["n_is_mn_v2_false"] = int((~mn_v2_vals & ~mn_v2_na).sum())
        rec["n_na"] = int(mn_v2_na.sum())
        rec["n_in_vh_v2"] = int(vh_true.sum())
        rec["n_in_vh_v2_false"] = int((~vh_vals & ~vh_na).sum())
        rec["n_in_vh_v2_na"] = int(vh_na.sum())
        rec["n_is_mn_v1"] = int((v1_vals & ~v1_na).sum())
        rec["n_is_mn_v1_na"] = int(v1_na.sum())
        rec["n_has_vh_v2_true"] = int(has_vals.sum())

        # ---- CANONICALISATION: is_MN must equal is_MN_v2 cell-for-cell.
        # The NA mask and the values at non-NA positions are the semantic
        # content, so those two are the HARD gate. Bit-equality of the raw
        # `values` buffer at MASKED positions is undefined in the
        # nullable-boolean encoding (pandas keeps whatever was in the buffer),
        # so it is reported and warned about, never failed on.
        rec["is_mn_mask_equal"] = bool(np.array_equal(mn_na, mn_v2_na))
        rec["is_mn_values_equal_effective"] = bool(
            rec["is_mn_mask_equal"]
            and np.array_equal(mn_vals[~mn_na], mn_v2_vals[~mn_v2_na]))
        rec["is_mn_values_equal_bitwise"] = bool(np.array_equal(mn_vals, mn_v2_vals))
        rec["is_mn_equals_v2"] = bool(rec["is_mn_mask_equal"]
                                      and rec["is_mn_values_equal_effective"])
        rec["n_is_mn"] = int((mn_vals & ~mn_na).sum())
        rec["n_is_mn_na"] = int(mn_na.sum())

        # ---- NA patterns: in_VH_v2 and is_MN_v2 must be NA on the same cells,
        # otherwise a "VH-annotated" section could silently contribute zero MNs.
        rec["na_mask_vh_eq_mn"] = bool(np.array_equal(vh_na, mn_v2_na))

        # ---- excluded/annotated coherence: the three independent markers of an
        # excluded section must agree (source == 'excluded', all-NA labels,
        # has_vh_v2 all False). tags_vh17 is derived from this, by manifest —
        # never discovered by a crash inside a GPU job.
        src_says_excluded = (rec["vh_v2_source"] == "excluded")
        labels_all_na = bool(mn_v2_na.all() and vh_na.all())
        has_all_false = (rec["n_has_vh_v2_true"] == 0)
        has_all_true = (rec["n_has_vh_v2_true"] == n_obs)
        rec["excluded"] = src_says_excluded
        rec["vh_annotated"] = not src_says_excluded
        rec["excluded_triple_agree"] = bool(
            (src_says_excluded and labels_all_na and has_all_false)
            or ((not src_says_excluded) and (not labels_all_na) and has_all_true))
        # For an annotated section there must be NO NA at all
        rec["annotated_zero_na_ok"] = bool(
            src_says_excluded or (rec["n_na"] == 0 and rec["n_in_vh_v2_na"] == 0))

        # ---- is_MN_v2 True must lie inside in_VH_v2 True
        rec["n_mn_outside_vh"] = int((mn_true & ~vh_true).sum())
        rec["mn_subset_of_vh_ok"] = (rec["n_mn_outside_vh"] == 0)

        # ---- VH input size (this is literally C2's n_obs for this section)
        rec["vh_knn_ok"] = bool(
            src_says_excluded or rec["n_in_vh_v2"] >= VH_MIN_CELLS_HARD)
        rec["vh_small_warn"] = bool(
            (not src_says_excluded) and rec["n_in_vh_v2"] < VH_MIN_CELLS_WARN)

        # ---- THE LABEL CONTRACT, whole section and VH subset
        if rec["cell_type_vocab_ok"]:
            whole = cell_type_mn_counts(ct_cats, ct_codes, mn_true)
            vh_only = cell_type_mn_counts(ct_cats, ct_codes, mn_true,
                                          subset=vh_true)
            rec["cell_type_mn_counts_whole"] = whole
            rec["cell_type_mn_counts_vh"] = vh_only
            rec["partition_tiles_ok"] = bool(
                sum(whole.values()) == n_obs
                and sum(vh_only.values()) == rec["n_in_vh_v2"]
                and whole[MN_CLASS] == rec["n_is_mn_v2"]
                and vh_only[MN_CLASS] == rec["n_is_mn_v2"])
            coarse_idx = ct_cats.index(TRANSCRIPTIONAL_MN_CAT)
            coarse_mask = (ct_codes == coarse_idx)
            rec["n_promoted_from_coarse_MN"] = int((mn_true & coarse_mask).sum())
            rec["n_promoted_from_other_classes"] = int((mn_true & ~coarse_mask).sum())
            rec["n_demoted_to_other_neurons"] = whole[OTHER_NEURONS]
        else:
            rec["partition_tiles_ok"] = False
            rec["problems"].append(
                "cell_type category vocabulary/order differs from the contract — "
                "the fixed 10-class partition cannot be built")

        rec["open_ok"] = True
    return rec


# --------------------------------------------------------------------------- #
# Manifest writer                                                              #
# --------------------------------------------------------------------------- #
def write_manifest_two_phase(payload: str, targets) -> list:
    """Write `payload` to every path in `targets`, all-or-nothing.

    Phase 1 stages a pid-suffixed temp NEXT TO each target (same filesystem,
    so the later rename is atomic and needs no extra quota) and fsyncs it. This
    is where a full or inode-maxed filesystem fails — before anything has been
    replaced anywhere. The two targets live on different quota-limited
    filesystems on SCG (oak vs home), so this case is realistic.

    Phase 2 moves any pre-existing target aside as <target>.prev.<pid> and
    os.replace()s the temp in, target by target.

    If ANY step raises OSError, everything this call did is undone — temps
    removed, replaced targets restored from their backup (or removed when
    nothing was there before), backups consumed — and the OSError is
    re-raised. The failure path therefore leaves the filesystem exactly as it
    was: no sbatch can find a complete primary manifest under a NO-GO verdict.
    On success the backups are deleted and the written targets are returned.
    """
    targets = list(dict.fromkeys(targets))      # the same path twice = once
    pid = os.getpid()
    staged = []       # (tmp, target) written in phase 1
    replaced = []     # (target, backup_or_None) touched in phase 2
    try:
        for target in targets:
            if os.path.exists(target) and not os.path.isfile(target):
                raise OSError(f"{target} exists and is not a regular file")
            tmp = f"{target}.tmp.{pid}"
            with open(tmp, "w") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            staged.append((tmp, target))
        for tmp, target in staged:
            backup = None
            if os.path.isfile(target):
                backup = f"{target}.prev.{pid}"
                os.replace(target, backup)
            replaced.append((target, backup))   # recorded BEFORE the swap
            os.replace(tmp, target)
    except OSError:
        for target, backup in reversed(replaced):
            try:
                if backup is not None and os.path.isfile(backup):
                    os.replace(backup, target)  # put the previous file back
                elif os.path.isfile(target):
                    os.remove(target)           # nothing was there before
            except OSError as exc:
                log(f"ROLLBACK FAILED for {target}: {exc} — inspect by hand")
        for tmp, _ in staged:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError as exc:
                log(f"ROLLBACK FAILED for temp {tmp}: {exc} — remove by hand")
        raise
    for target, backup in replaced:
        if backup is not None:
            try:
                os.remove(backup)
            except OSError as exc:
                log(f"could not remove backup {backup}: {exc} — remove by hand")
    return [target for target, _ in replaced]


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=("Data-contract gate + TAG-manifest generator for the four "
                     "NicheCompass v2 configurations. Exit 0 = go, exit 1 = "
                     "no-go (no manifest written)."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --data-dir / --base-dir / --manifest are accepted aliases so that the
    # script runs unmodified under either calling convention (the sbatch wrapper
    # uses those names; the pipeline spec uses --pass2-dir/--runs-base).
    parser.add_argument("--pass2-dir", "--data-dir", dest="pass2_dir",
                        default=DEFAULT_PASS2_DIR,
                        help="dir holding ALS_SCXenium_<TAG>_pass2.h5ad")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_FOLDER,
                        help="NicheCompass_SC root; --runs-base and --gp-dir "
                             "default to <base-dir>/runs_v2 and "
                             "<base-dir>/gene_programs")
    parser.add_argument("--runs-base", default=None,
                        help="runs_v2 base; the four <config>/<runstamp>/ dirs "
                             "and (by default) the manifest land here "
                             "[default: <base-dir>/runs_v2]")
    parser.add_argument("--runstamp", default=DEFAULT_RUNSTAMP,
                        help="fixed runstamp for this whole v2 sweep")
    parser.add_argument("--manifest", default=None,
                        help="explicit path for the tag manifest "
                             "[default: <runs-base>/<runstamp>_tag_manifest.json]")
    parser.add_argument("--manifest-copy", default=DEFAULT_MANIFEST_COPY,
                        help="scripts dir (or explicit *.json path) to receive "
                             "an identical copy of the manifest")
    parser.add_argument("--gp-dir", default=None,
                        help="cached gene-program resources dir "
                             "[default: <base-dir>/gene_programs]")
    args = parser.parse_args(argv)
    if args.runs_base is None:
        args.runs_base = os.path.join(args.base_dir, "runs_v2")
    if args.gp_dir is None:
        args.gp_dir = os.path.join(args.base_dir, "gene_programs")
    return args


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    gates = Gates()
    warnings_seen = []

    def warn(msg):
        warnings_seen.append(msg)
        log(f"WARNING: {msg}")

    manifest_path = args.manifest or os.path.join(
        args.runs_base, f"{args.runstamp}_tag_manifest.json")
    if args.manifest_copy.endswith(".json"):
        manifest_copy_path = args.manifest_copy
    else:
        manifest_copy_path = os.path.join(
            args.manifest_copy, f"{args.runstamp}_tag_manifest.json")

    # ----------------------------------------------------------------------- #
    section("[1/6] Environment / inputs")
    # ----------------------------------------------------------------------- #
    log(f"python {sys.version.split()[0]} @ {sys.executable}")
    log(f"script {os.path.abspath(__file__)}")
    log(f"SLURM job {os.environ.get('SLURM_JOB_ID', 'n/a')} on "
        f"{os.uname().nodename}")
    versions = {}
    for pkg in ("h5py", "numpy", "pandas"):
        try:
            versions[pkg] = pkg_version(pkg)
        except Exception:
            versions[pkg] = None
        log(f"  {pkg} {versions[pkg]}")
    log(f"pass2 dir     : {args.pass2_dir}")
    log(f"runs base     : {args.runs_base}")
    log(f"runstamp      : {args.runstamp}")
    log(f"gp cache dir  : {args.gp_dir}")
    log(f"manifest      : {manifest_path}")
    log(f"manifest copy : {manifest_copy_path}")
    log("CPU-only, read-only on the pass2 files; X data is never read.")

    tags = sorted(SAMPLE_TAGS)
    gates.check("tag_list_sorted", tags == sorted(set(tags)) and len(tags) == 20,
                f"{len(tags)} canonical TAGs, de-duplicated and sorted")

    paths = {t: os.path.join(args.pass2_dir, f"ALS_SCXenium_{t}_pass2.h5ad")
             for t in tags}
    missing = [t for t in tags if not os.path.isfile(paths[t])]
    gates.check("pass2_files_present", not missing,
                "all 20 present" if not missing
                else f"MISSING: {', '.join(missing)}")

    # Glob-drift warning (NC_SC_training_all.py convention): flag pass2 files on
    # disk that are NOT in the 20-section manifest and will be ignored.
    try:
        on_disk = {
            fn for fn in os.listdir(args.pass2_dir)
            if fn.startswith("ALS_SCXenium_") and fn.endswith("_pass2.h5ad")}
        unexpected = sorted(on_disk - {os.path.basename(p) for p in paths.values()})
        if unexpected:
            warn(f"{len(unexpected)} pass2 file(s) on disk are NOT in the "
                 f"20-section manifest and will be IGNORED: "
                 f"{', '.join(unexpected)}")
    except OSError as exc:
        warn(f"could not list {args.pass2_dir}: {exc}")

    # ----------------------------------------------------------------------- #
    section(f"[2/6] Per-section probe ({len(tags)} files, h5py, read-only)")
    # ----------------------------------------------------------------------- #
    records = []
    for tag in tags:
        t0 = time.time()
        try:
            rec = probe_section(tag, paths[tag])
        except Exception:
            rec = {"tag": tag, "path": paths[tag], "open_ok": False,
                   "problems": ["probe raised:\n" + traceback.format_exc()]}
        records.append(rec)
        if rec.get("open_ok"):
            vh = (f"{rec.get('n_in_vh_v2'):>6,}" if rec.get("vh_annotated")
                  else "   n/a")
            mn = (f"{rec.get('n_is_mn_v2'):>4}" if rec.get("vh_annotated")
                  else " n/a")
            log(f"  {tag}  n_obs={rec.get('n_obs'):>7,}  vars={rec.get('n_vars')}"
                f"  vh={vh}  mn_v2={mn}  na={rec.get('n_na', 0):>6,}"
                f"  src={rec.get('vh_v2_source')}"
                f"  status={rec.get('status')}  ({time.time() - t0:.1f}s)")
        else:
            log(f"  {tag}  PROBE FAILED: {'; '.join(rec['problems'])}")
        for problem in rec.get("problems", []):
            if rec.get("open_ok"):
                warn(f"{tag}: {problem}")

    ok_records = [r for r in records if r.get("open_ok")]
    gates.check("h5ad_opened", len(ok_records) == len(tags),
                f"{len(ok_records)}/{len(tags)} files probed cleanly")

    def aggregate(gate_name, flag_key, ok_detail):
        """One gate per contract, naming the offending TAGs (not 20x noise).

        Zero probed sections FAILS rather than passing vacuously — an empty or
        wrong --pass2-dir must never produce a screen of green gate lines.
        """
        if not ok_records:
            return gates.check(gate_name, False,
                               "no section was probed — cannot evaluate")
        bad = [r["tag"] for r in ok_records if not r.get(flag_key, False)]
        detail = (f"{len(ok_records) - len(bad)}/{len(ok_records)} {ok_detail}"
                  if not bad else f"OFFENDING TAGS: {', '.join(bad)}")
        return gates.check(gate_name, not bad, detail)

    aggregate("obs_columns_present", "obs_cols_ok",
              f"sections carry all {len(REQUIRED_OBS_COLS)} required obs columns")
    aggregate("n_vars_480", "n_vars_ok", f"sections have {EXPECTED_N_GENES} vars")
    aggregate("counts_layer_present", "counts_present",
              "sections carry layers['counts']")
    aggregate("counts_layer_sparse", "counts_sparse_ok",
              f"sections store layers['counts'] as one of {SPARSE_ENCODINGS}")
    aggregate("counts_integer_valued", "counts_int_ok",
              "sections have integer-valued counts (exhaustive scan of the "
              "whole data buffer)")
    aggregate("counts_nonnegative", "counts_nonneg_ok",
              "sections have non-negative counts (exhaustive scan)")
    aggregate("counts_shape_matches", "counts_shape_ok",
              "counts shapes == (n_obs, 480)")
    aggregate("n_obs_consistent", "n_obs_consistent",
              "X shape attrs agree with obs index length")
    aggregate("x_is_sparse", "x_sparse_ok",
              f"sections store X as one of {SPARSE_ENCODINGS} (0.3.3 indexes "
              "adata.X.sum(0)[0], which needs a sparse X)")
    aggregate("spatial_key_present", "spatial_present",
              "sections carry obsm['spatial']")
    aggregate("spatial_shape_matches", "spatial_shape_ok",
              "obsm['spatial'] is (n_obs, >=2)")
    aggregate("spatial_all_finite", "spatial_finite_ok",
              "obsm['spatial'] has no NaN/Inf coordinate")
    aggregate("sample_label_unique", "sample_unique_ok",
              "sections carry exactly one obs['sample'] value and zero -1 "
              "(NaN) codes")
    aggregate("status_vocabulary", "status_ok",
              f"sections have one status in {STATUS_VOCAB} and zero -1 (NaN) "
              "codes")
    aggregate("cell_type_vocabulary", "cell_type_vocab_ok",
              "sections share the 9-class cell_type vocabulary AND order")
    aggregate("cell_type_no_nan_codes", "cell_type_no_nan_ok",
              "sections have zero -1 (NaN) cell_type codes")
    aggregate("vh_source_vocabulary", "vh_source_ok",
              f"sections have one vh_v2_source in {VH_SOURCE_VOCAB}")
    aggregate("is_MN_eq_is_MN_v2", "is_mn_equals_v2",
              "sections have is_MN == is_MN_v2 (NA mask + values)")
    aggregate("na_mask_vh_eq_mn", "na_mask_vh_eq_mn",
              "sections have identical in_VH_v2 / is_MN_v2 NA masks")
    aggregate("excluded_triple_agree", "excluded_triple_agree",
              "sections agree across vh_v2_source / all-NA / has_vh_v2")
    aggregate("annotated_zero_na", "annotated_zero_na_ok",
              "annotated sections carry zero NA labels")
    aggregate("mn_v2_subset_of_vh", "mn_subset_of_vh_ok",
              "sections have every is_MN_v2 True cell inside in_VH_v2")
    aggregate("partition_tiles_cells", "partition_tiles_ok",
              "sections' 10-class partition tiles every cell (whole + VH)")
    aggregate("vh_min_cells_for_knn", "vh_knn_ok",
              f"annotated sections have > {N_NEIGHBORS} VH cells")

    # Non-fatal observations
    for rec in ok_records:
        if not rec.get("sample_matches_tag", True):
            warn(f"{rec['tag']}: obs['sample']={rec.get('sample')!r} != "
                 f"tag-derived {rec.get('sample_from_tag')!r} — TRUSTING obs, "
                 "as the trainer does")
        if not rec.get("is_mn_values_equal_bitwise", True) \
                and rec.get("is_mn_equals_v2"):
            warn(f"{rec['tag']}: is_MN / is_MN_v2 raw `values` buffers differ at "
                 "MASKED positions only — semantically identical (mask + non-NA "
                 "values match), so this is not a contract breach")
        if rec.get("spatial_n_cols", 2) != 2:
            warn(f"{rec['tag']}: obsm['spatial'] has "
                 f"{rec['spatial_n_cols']} columns — kneighbors_graph would use "
                 "ALL of them; confirm the extra column is really a coordinate")
        if rec.get("vh_small_warn"):
            warn(f"{rec['tag']}: only {rec['n_in_vh_v2']:,} VH cells "
                 f"(< {VH_MIN_CELLS_WARN}) — C2 will train a tiny model here; "
                 "expect 2-4 Leiden niches and size the edge batch off the graph")

    # is_MN_v1 asymmetry: SD02913BA is 'excluded' (v2 labels 100% NA) yet its
    # is_MN_v1 column is fully non-NA with 12 True. Reported, never gated —
    # any "v1 vs v2" table must show 12 v1 MNs and 'n/a' v2 MNs there.
    v1_asym = [r["tag"] for r in ok_records
               if r.get("excluded") and r.get("n_is_mn_v1_na", 0) == 0]
    if v1_asym:
        log(f"NOTE: excluded section(s) {', '.join(v1_asym)} carry a fully "
            "non-NA is_MN_v1 while is_MN_v2 is 100% NA — expected asymmetry; "
            "'v1 vs v2' tables must print 'n/a' for v2 there, not 0.")

    # Panel identity across sections: the claim "93 GPs survive identically for
    # every configuration" is gene-based, so it only holds if the panels match.
    panels = {r["tag"]: r.get("_var_names") for r in ok_records if r.get("_var_names")}
    panel_ok = True
    if panels:
        ref_tag = tags[0] if tags[0] in panels else sorted(panels)[0]
        ref = panels[ref_tag]
        drifted = [t for t, vn in panels.items() if vn != ref]
        panel_ok = not drifted
        gates.check("gene_panel_identical", panel_ok,
                    f"{len(panels)} sections share the {len(ref)}-gene panel "
                    f"(order included), ref {ref_tag}" if panel_ok
                    else f"PANEL DRIFT in: {', '.join(drifted)}")
    else:
        gates.check("gene_panel_identical", False, "no panel could be read")

    # Cross-section uniqueness of obs['sample']. The per-section gate above only
    # proves each file carries ONE label; nothing yet proves the 20 labels are
    # distinct. The integrated configs derive the 'sample' covariate's levels
    # from obs['sample'].unique() on the concat (models/nichecompass.py), so
    # two sections sharing a label collapse into one level, and the
    # cat_covariates_embeds_nums recorded below would be a lie. A tag mismatch
    # stays warn-only (obs is trusted, as the trainer does); a duplicate is not.
    sample_labels = [r.get("sample") for r in ok_records]
    unlabelled = [r["tag"] for r in ok_records if r.get("sample") is None]
    duplicated = sorted({lab for lab in sample_labels
                         if lab is not None and sample_labels.count(lab) > 1})
    if not ok_records:
        gates.check("sample_labels_unique", False,
                    "no section was probed — cannot evaluate")
    elif duplicated or unlabelled:
        parts = []
        if duplicated:
            owners = {lab: [r["tag"] for r in ok_records if r.get("sample") == lab]
                      for lab in duplicated}
            parts.append("DUPLICATED LABELS: " + "; ".join(
                f"{lab!r} in {', '.join(owners[lab])}" for lab in duplicated))
        if unlabelled:
            parts.append(f"NO SINGLE LABEL: {', '.join(unlabelled)}")
        gates.check("sample_labels_unique", False, " | ".join(parts))
    else:
        gates.check("sample_labels_unique", True,
                    f"{len(set(sample_labels))} distinct obs['sample'] labels "
                    f"across {len(ok_records)} sections (one covariate level "
                    "each)")

    # ----------------------------------------------------------------------- #
    section("[3/6] Cohort gates")
    # ----------------------------------------------------------------------- #
    n_obs_total = int(sum(r.get("n_obs", 0) for r in ok_records))
    n_mn_total = int(sum(r.get("n_is_mn_v2", 0) for r in ok_records))
    n_vh_total = int(sum(r.get("n_in_vh_v2", 0) for r in ok_records))
    n_na_total = int(sum(r.get("n_na", 0) for r in ok_records))
    n_v1_total = int(sum(r.get("n_is_mn_v1", 0) for r in ok_records))
    n_coarse_total = int(sum(r.get("n_coarse_motor_neurons", 0) or 0
                             for r in ok_records))
    n_promo_coarse = int(sum(r.get("n_promoted_from_coarse_MN", 0)
                             for r in ok_records))
    n_promo_other = int(sum(r.get("n_promoted_from_other_classes", 0)
                            for r in ok_records))

    gates.check("cohort_n_obs", n_obs_total == EXPECTED_N_CELLS,
                f"{n_obs_total:,} cells (expected {EXPECTED_N_CELLS:,})")
    gates.check("cohort_is_MN_v2", n_mn_total == EXPECTED_IS_MN_V2_TRUE,
                f"{n_mn_total:,} is_MN_v2 True (expected "
                f"{EXPECTED_IS_MN_V2_TRUE:,})")
    gates.check("cohort_in_VH_v2", n_vh_total == EXPECTED_IN_VH_V2_TRUE,
                f"{n_vh_total:,} in_VH_v2 True (expected "
                f"{EXPECTED_IN_VH_V2_TRUE:,})")
    gates.check("cohort_na_cells", n_na_total == EXPECTED_NA_CELLS,
                f"{n_na_total:,} <NA> cells (expected {EXPECTED_NA_CELLS:,})")

    excluded_tags = sorted(r["tag"] for r in ok_records if r.get("excluded"))
    tags_vh17 = sorted(r["tag"] for r in ok_records if r.get("vh_annotated"))
    gates.check("excluded_tag_set",
                tuple(excluded_tags) == tuple(sorted(EXPECTED_EXCLUDED_TAGS)),
                f"{len(excluded_tags)} excluded: {', '.join(excluded_tags)} "
                f"(expected {', '.join(sorted(EXPECTED_EXCLUDED_TAGS))})")
    gates.check("tags_vh17_count", len(tags_vh17) == 17,
                f"{len(tags_vh17)} VH-annotated sections")

    # Warn-only derived figures
    for label, got, want in (
            ("is_MN_v1 True", n_v1_total, SNAPSHOT_IS_MN_V1_TRUE),
            ("coarse transcriptional MN", n_coarse_total, SNAPSHOT_COARSE_MN),
            ("promoted from coarse MN", n_promo_coarse,
             SNAPSHOT_PROMOTED_FROM_COARSE),
            ("promoted from other classes", n_promo_other,
             SNAPSHOT_PROMOTED_FROM_OTHER)):
        if got != want:
            warn(f"cohort {label} = {got:,}, 2026-09-02 snapshot says {want:,} "
                 "(warn-only: derived figure, not part of the hard contract)")
        else:
            log(f"  cohort {label}: {got:,} (matches snapshot)")

    if n_promo_coarse + n_promo_other != n_mn_total:
        gates.check("promotion_split_closes", False,
                    f"{n_promo_coarse} + {n_promo_other} != {n_mn_total}")
    else:
        gates.check("promotion_split_closes", True,
                    f"{n_promo_coarse:,} from the coarse MN class + "
                    f"{n_promo_other:,} from other classes = {n_mn_total:,}")

    # Per-tag drift diagnostics (warn-only; tells you WHICH section moved)
    for rec in ok_records:
        snap = SNAPSHOT_PER_TAG.get(rec["tag"])
        if snap is None:
            continue
        s_obs, s_vh, s_mn, s_coarse, s_status, s_src = snap
        for field, got, want in (
                ("n_obs", rec.get("n_obs"), s_obs),
                ("n_in_vh_v2", rec.get("n_in_vh_v2"), s_vh),
                ("n_is_mn_v2", rec.get("n_is_mn_v2"), s_mn),
                ("n_coarse_motor_neurons", rec.get("n_coarse_motor_neurons"),
                 s_coarse),
                ("status", rec.get("status"), s_status),
                ("vh_v2_source", rec.get("vh_v2_source"), s_src)):
            if got != want:
                warn(f"{rec['tag']}: {field} = {got!r}, snapshot says {want!r}")

    # Cohort-level label-contract roll-ups (these become the trainer's and the
    # downstream's EXPECTED_* values — derived here, never hardcoded there)
    cohort_whole = {c: 0 for c in FIXED_CLASSES_10}
    cohort_vh = {c: 0 for c in FIXED_CLASSES_10}
    for rec in ok_records:
        for cls, n in (rec.get("cell_type_mn_counts_whole") or {}).items():
            cohort_whole[cls] += int(n)
        for cls, n in (rec.get("cell_type_mn_counts_vh") or {}).items():
            cohort_vh[cls] += int(n)
    gates.check("cohort_partition_closes",
                sum(cohort_whole.values()) == n_obs_total
                and sum(cohort_vh.values()) == n_vh_total
                and cohort_whole[MN_CLASS] == n_mn_total,
                f"whole {sum(cohort_whole.values()):,} == {n_obs_total:,}; "
                f"VH {sum(cohort_vh.values()):,} == {n_vh_total:,}; "
                f"MN class {cohort_whole[MN_CLASS]:,}")
    log("  cohort cell_type_mn (whole sections, C3): "
        + ", ".join(f"{c}={cohort_whole[c]:,}" for c in FIXED_CLASSES_10))
    log("  cohort cell_type_mn (VH cells only, C4):  "
        + ", ".join(f"{c}={cohort_vh[c]:,}" for c in FIXED_CLASSES_10))

    status_sections, status_cells = {}, {}
    for rec in ok_records:
        st = rec.get("status")
        status_sections[st] = status_sections.get(st, 0) + 1
        status_cells[st] = status_cells.get(st, 0) + int(rec.get("n_obs", 0))
    log("  status: " + ", ".join(
        f"{k}={status_sections[k]} sections / {status_cells[k]:,} cells"
        for k in sorted(status_sections, key=lambda s: (s is None, s))))

    # ----------------------------------------------------------------------- #
    section("[4/6] Gene-program cache (every config must load_from_disk)")
    # ----------------------------------------------------------------------- #
    gp_cache_files = {}
    gp_missing = []
    for rel in GP_CACHE_RELPATHS:
        path = os.path.join(args.gp_dir, rel)
        ok = os.path.isfile(path) and os.path.getsize(path) > 0
        entry = {"path": path, "relpath": rel, "present": bool(ok)}
        if ok:
            entry.update(fingerprint(path))
            log(f"  {rel:<52} {entry['size_bytes'] / 1e6:>9.1f} MB")
        else:
            gp_missing.append(rel)
            log(f"  {rel:<52} MISSING or EMPTY")
        gp_cache_files[os.path.basename(rel)] = entry
    gates.check("gp_cache_present", not gp_missing,
                f"all {len(GP_CACHE_RELPATHS)} cached resources present and "
                "non-empty" if not gp_missing
                else f"MISSING/EMPTY: {', '.join(gp_missing)}")
    # The trainer os.chdir()s into gene_programs/ (the NicheNet extractor drops
    # temp .rds into the CWD), so read+execute on that dir is load-bearing.
    gates.check("gp_dir_readable",
                os.path.isdir(args.gp_dir)
                and os.access(args.gp_dir, os.R_OK | os.X_OK),
                f"{args.gp_dir} is readable and chdir-able")
    if not os.access(args.gp_dir, os.W_OK):
        warn(f"{args.gp_dir} is not writable — harmless while every GP resource "
             "is cached (load_from_disk), fatal if a cache file ever goes away")

    # ----------------------------------------------------------------------- #
    section("[5/6] Output tree")
    # ----------------------------------------------------------------------- #
    # Only create anything once the data contract has passed: a no-go run must
    # not litter the tree, and must never leave a manifest behind.
    config_run_dirs = {cfg: os.path.join(args.runs_base, cfg, args.runstamp)
                       for cfg in CONFIGS}
    if not gates.all_passed:
        log("SKIPPING directory creation and manifest write — "
            f"{len(gates.failed)} gate(s) already failed.")
    else:
        made = []
        try:
            for cfg, run_dir in config_run_dirs.items():
                os.makedirs(run_dir, exist_ok=True)
                made.append(run_dir)
                log(f"  {cfg:<17} -> {run_dir}")
            os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(manifest_copy_path) or ".",
                        exist_ok=True)
            gates.check("output_dirs_created", len(made) == len(CONFIGS),
                        f"{len(made)} run roots under {args.runs_base}")
        except OSError as exc:
            gates.check("output_dirs_created", False, f"{exc}")

        # A real write probe, not just os.access: oak quota here is
        # INODE-enforced and has been at its ceiling before, and a job that
        # discovers this after 75 minutes of H200 time is an expensive way to
        # find out.
        probe_failures = []
        probe_targets = list(dict.fromkeys(
            list(config_run_dirs.values())
            + [args.runs_base,
               os.path.dirname(manifest_path) or ".",
               os.path.dirname(manifest_copy_path) or "."]))
        for target in probe_targets:
            probe = os.path.join(target, f".nc_v2_preflight_probe_{os.getpid()}")
            try:
                with open(probe, "w") as fh:
                    fh.write("ok\n")
                os.remove(probe)
            except OSError as exc:
                probe_failures.append(f"{target} ({exc})")
        gates.check("output_dirs_writable", not probe_failures,
                    "write probe passed on every output dir"
                    if not probe_failures
                    else "; ".join(probe_failures))

    # ----------------------------------------------------------------------- #
    section("[6/6] Tag manifest + summary")
    # ----------------------------------------------------------------------- #
    per_tag = {}
    for rec in ok_records:
        clean = {k: v for k, v in rec.items() if not k.startswith("_")}
        clean.pop("problems", None)
        clean.pop("var_names_head", None)
        clean.pop("var_names_tail", None)
        per_tag[rec["tag"]] = clean

    tag_lists = {"tags_all20": tags, "tags_vh17": tags_vh17}
    configs_block = {}
    for cfg, meta in CONFIGS.items():
        cfg_tags = tag_lists[meta["tag_list_key"]]
        if meta["scope"] == "vh":
            n_cells = sum(per_tag[t]["n_in_vh_v2"] for t in cfg_tags
                          if t in per_tag)
        else:
            n_cells = sum(per_tag[t]["n_obs"] for t in cfg_tags if t in per_tag)
        configs_block[cfg] = {
            "what": meta["what"],
            "mode": meta["mode"],
            "scope": meta["scope"],
            "tag_list_key": meta["tag_list_key"],
            "n_tags": len(cfg_tags),
            "n_runs": len(cfg_tags) if meta["mode"] == "persample" else 1,
            "n_cells_total": int(n_cells),
            "cat_covariates_keys": (None if meta["mode"] == "persample"
                                    else ["sample"]),
            "cat_covariates_embeds_nums": (None if meta["mode"] == "persample"
                                           else [len(cfg_tags)]),
            "run_dir": config_run_dirs[cfg],
            "per_tag_run_dir_pattern": (
                os.path.join(config_run_dirs[cfg], "<TAG>")
                if meta["mode"] == "persample" else None),
        }

    manifest = {
        "runstamp": args.runstamp,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_local": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.abspath(__file__),
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version.split()[0],
        "versions": versions,
        "pass2_dir": args.pass2_dir,
        "runs_base": args.runs_base,
        "gp_dir": args.gp_dir,
        "manifest_path": manifest_path,
        "manifest_copy_path": manifest_copy_path,
        "tags_all20": tags,
        "tags_vh17": tags_vh17,
        "excluded": excluded_tags,
        "per_tag": per_tag,
        "configs": configs_block,
        "cohort": {
            "n_cells": n_obs_total,
            "n_genes": EXPECTED_N_GENES,
            "n_sections": len(ok_records),
            "n_sections_vh_annotated": len(tags_vh17),
            "n_in_vh_v2_true": n_vh_total,
            "n_is_mn_v2_true": n_mn_total,
            "n_is_mn_v1_true": n_v1_total,
            "n_na_cells": n_na_total,
            "n_coarse_motor_neurons": n_coarse_total,
            "status_sections": status_sections,
            "status_cells": status_cells,
        },
        "label_contract": {
            "rule": ("cell_type_mn = cell_type with the coarse transcriptional "
                     f"'{TRANSCRIPTIONAL_MN_CAT}' class renamed "
                     f"'{OTHER_NEURONS}'; then every cell with is_MN_v2 == True "
                     f"becomes '{MN_CLASS}', which wins over every base class. "
                     "is_MN_v2 <NA> cells are simply not motor neurons and keep "
                     "their (possibly demoted) base class."),
            "mn_flag_column": "is_MN_v2",
            "cell_type_categories": list(CELL_TYPE_CATEGORIES),
            "cell_type_mn_classes": list(FIXED_CLASSES_10),
            "class_order_note": (
                f"'{MN_CLASS}' is LAST so the {n_mn_total}-cell sliver draws on "
                "top of every stacked bar; always build the Categorical with "
                "this explicit list, otherwise an excluded section silently "
                "loses the Motor neurons level entirely."),
            "cohort_counts_whole": cohort_whole,
            "cohort_counts_vh": cohort_vh,
            "n_promoted_from_coarse_MN": n_promo_coarse,
            "n_promoted_from_other_classes": n_promo_other,
            "n_demoted_to_other_neurons": cohort_whole[OTHER_NEURONS],
            "na_handling": ("obs['is_MN_v2'] is a pandas nullable boolean; the "
                            "ONLY safe masks on this env are "
                            "s.astype('boolean').fillna(False).to_numpy() or "
                            "s.to_numpy(dtype=bool, na_value=False). "
                            "np.asarray(s).astype(bool) RAISES."),
        },
        "gp_cache_files": gp_cache_files,
        "graph_recipe": {
            "spatial_key": "spatial",
            "obsm_keys_to_drop": ["spatial_grid_offset", "spatial_orig"],
            "n_neighbors": N_NEIGHBORS,
            "method": ("sklearn.neighbors.kneighbors_graph(mode='connectivity', "
                       "include_self=False, n_jobs=-1) then A.maximum(A.T)"),
            "adj_key": "spatial_connectivities",
            "stored_in": "obsp",
        },
        "gates": gates.as_list(),
        "gates_summary": gates.summary(),
        "all_gates_passed": gates.all_passed,
        "warnings": warnings_seen,
    }

    # ---- the one-screen summary table
    rows = []
    for rec in ok_records:
        annotated = rec.get("vh_annotated")
        whole = rec.get("cell_type_mn_counts_whole") or {}
        rows.append({
            "TAG": rec["tag"],
            "sample": rec.get("sample"),
            "status": rec.get("status"),
            "n_obs": f"{rec.get('n_obs', 0):,}",
            "vh_true": f"{rec['n_in_vh_v2']:,}" if annotated else "n/a",
            "mn_v2": f"{rec['n_is_mn_v2']:,}" if annotated else "n/a",
            "mn_v1": f"{rec.get('n_is_mn_v1', 0):,}",
            "na": f"{rec.get('n_na', 0):,}",
            "coarseMN": f"{rec.get('n_coarse_motor_neurons') or 0:,}",
            "promo_in": (f"{rec.get('n_promoted_from_coarse_MN', 0):,}"
                         if annotated else "n/a"),
            "promo_out": (f"{rec.get('n_promoted_from_other_classes', 0):,}"
                          if annotated else "n/a"),
            "otherNeu": f"{whole.get(OTHER_NEURONS, 0):,}",
            "src": rec.get("vh_v2_source"),
        })
    if rows:
        table = pd.DataFrame(rows)
        print("", flush=True)
        with pd.option_context("display.max_rows", None,
                               "display.max_columns", None,
                               "display.width", 220):
            print(table.to_string(index=False), flush=True)
        print("", flush=True)
    log(f"cohort: {n_obs_total:,} cells | {len(ok_records)} sections "
        f"({len(tags_vh17)} VH-annotated, {len(excluded_tags)} excluded) | "
        f"in_VH_v2 {n_vh_total:,} | is_MN_v2 {n_mn_total:,} | "
        f"<NA> {n_na_total:,}")
    log(f"label contract: Motor neurons {cohort_whole[MN_CLASS]:,} "
        f"({n_promo_coarse:,} promoted from the coarse MN class + "
        f"{n_promo_other:,} from other classes) | "
        f"Other neurons {cohort_whole[OTHER_NEURONS]:,} demoted")
    log("per-config runs: " + " | ".join(
        f"{cfg} {configs_block[cfg]['n_runs']} run(s), "
        f"{configs_block[cfg]['n_cells_total']:,} cells"
        for cfg in ("persample_whole", "persample_vh",
                    "integrated_whole", "integrated_vh")))

    # ---- verdict
    def report_stale_manifest():
        for path in dict.fromkeys((manifest_path, manifest_copy_path)):
            if os.path.isfile(path):
                stale = fingerprint(path)
                log(f"  A manifest ALREADY EXISTS at {path} (written "
                    f"{stale['mtime_iso']}). It is STALE relative to this "
                    "failed check and was left untouched — remove it by hand "
                    "before any sbatch reads it.")

    print("", flush=True)
    log("=" * 78)
    if not gates.all_passed:
        log(f"NO-GO: {len(gates.failed)} of {len(gates.rows)} gates FAILED. "
            "No manifest written; do NOT submit any v2 job.")
        for row in gates.failed:
            log(f"  FAILED GATE  {row['gate']:<24} | {row['detail']}")
        report_stale_manifest()
        if warnings_seen:
            log(f"  plus {len(warnings_seen)} warning(s) above.")
        log("=" * 78)
        return 1

    # The ledger is FROZEN from here on: the manifest carries the complete gate
    # list, so its gates_summary is the very string the GO line below prints.
    # Nothing may call gates.check() after this point (the write itself is not a
    # gate — a row saying "manifest written" inside the manifest is tautological
    # and, worse, it made the file say 38/38 while the log said 39/39).
    manifest["gates"] = gates.as_list()
    manifest["gates_summary"] = gates.summary()
    manifest["all_gates_passed"] = gates.all_passed
    payload = json.dumps(manifest, indent=2, default=str) + "\n"

    # Two-phase atomic write to BOTH targets (stage both temps, then replace
    # both; see write_manifest_two_phase). An array task can never read a
    # half-written manifest, and a failure at either target leaves NO manifest
    # behind — previously a copy-target failure returned 1 with a complete,
    # valid primary manifest already sitting at the path every sbatch reads.
    try:
        written = write_manifest_two_phase(
            payload, (manifest_path, manifest_copy_path))
    except OSError:
        log("MANIFEST WRITE FAILED:\n" + traceback.format_exc())
        log("=" * 78)
        log(f"NO-GO: the data contract passed ({gates.summary()}) but the "
            "manifest could not be written to BOTH targets. Everything this "
            "run staged or replaced was rolled back, so no manifest from this "
            "run is on disk anywhere. Do NOT submit any v2 job.")
        report_stale_manifest()
        log("=" * 78)
        return 1
    for target in written:
        log(f"manifest written: {target} "
            f"({os.path.getsize(target) / 1024:.1f} KB, two-phase atomic)")

    log(f"GO: {gates.summary()}; {len(warnings_seen)} warning(s).")
    if warnings_seen:
        for msg in warnings_seen:
            log(f"  warning: {msg}")
    log("ALL DONE — NC v2 preflight")
    log(f"  manifest      : {manifest_path}")
    log(f"  manifest copy : {manifest_copy_path}")
    for cfg in ("persample_whole", "persample_vh", "integrated_whole",
                "integrated_vh"):
        log(f"  {cfg:<17} : {config_run_dirs[cfg]}")
    log(f"  tags_all20    : {len(tags)} tags")
    log(f"  tags_vh17     : {len(tags_vh17)} tags "
        f"(excluded {', '.join(excluded_tags)})")
    log(f"  elapsed       : {time.time() - SCRIPT_T0:.1f}s")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
