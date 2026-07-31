#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/final_config.py
#
# The single place every _FINAL script gets its paths from. Nothing here runs an analysis. It
# answers four questions for the rest of the pipeline: where the raw data lives, which 20 of
# the 22 delivered directories we keep, how to turn one of those directories into an AnnData,
# and where results are written. A path that is wrong here is wrong everywhere, which is why
# it lives in one file.
#
# Read the dedup rules twice. Marcel's delivery has 22 directories for 20 samples. SD01620_BI
# and SD02022 are superseded originals and we keep the ALS6 redos instead; Region_2 is really
# SD05413_BG; two control directories carry a trailing underscore. discover_samples() encodes
# all of that and raises instead of guessing when the count comes out wrong. SD01616 is
# recorded as C9ORF in status.csv but analysed as sporadic, applied here as an explicit
# override so the deviation stays visible to a reviewer instead of hiding in someone's head.
#
# load_final_sample() reads a 10x MTX directory instead of an .h5, which is what changed with
# the _final delivery, then cuts 541 features down to the 480 Gene Expression probes and
# discards the 41 negative codewords and 20 negative probes. Cell ids are plain integers on
# _final, so the matrix-to-cells join is 1:1 and the 3 to 20 percent silent cell loss that
# dogged the _gausss generation is gone.
#
# One trap. obsm['spatial'] is a ten column grid offset that exists so 20 sections can be
# drawn on one canvas. True microns live in obsm['spatial_orig'], and that is what any spatial
# graph has to use. Swap them and you get maps that look perfectly reasonable sitting on top
# of neighbourhoods that are garbage.
#
# Deployed at /oak/.../RK/Spatial/final_rerun_code/. The absolute paths below are SCG paths,
# so this repo copy documents the run rather than reproducing it locally.
# ========================================================================================

"""
final_config.py -- single source of truth for the ALS SC Xenium MN-corrected
_FINAL re-run (Marcel's Ranger_procd_mw_final segmentation, delivered 2026-07-19).

EVERY staged _FINAL script imports this module for: the raw-data root, the
20-sample discovery + dedup, per-sample loading (10x MTX dir -> 480-Gene-Expression
AnnData with a clean integer cell_id join), the grid-offset convention, and all
_FINAL output paths. This module runs NO analysis; it is pure config + loaders.

Adapted from the proven _gausss concat recipe
(/home/rodrigok/Notebooks/Spatial_MN_correct/build_mn_notebook.py). The ONLY
behavioural deltas vs _gausss (verified by recon 2026-07-19), each flagged inline:

  D1  matrix source is a 10x MTX DIRECTORY (cell_feature_matrix/ = barcodes.tsv.gz
      + features.tsv.gz + matrix.mtx.gz), NOT cell_feature_matrix.h5. Load with
      sc.read_10x_mtx(dir, gex_only=False), then subset var to Gene Expression.
  D2  cell ids are PLAIN INTEGERS 1..N in BOTH the matrix barcodes and cells.parquet
      cell_id -> the matrix<->cells join is a clean 1:1 (overlap ~1.0). The old
      gm-ID / hex namespace mismatch AND the 3-20% date-skew drop are GONE.
  D3  obs_names are '<LABEL>:<cell_id>' using the resolved LABEL (e.g.
      'SD05413_BG:1' for the Region_2 dir), NOT the raw run-dir name.
  D4  panel = 541 features (480 Gene Expression + 41 Negative Control Codeword
      + 20 Negative Control Probe); keep the 480 GE panel. seg method string in
      cells.parquet is 'Imported Cell Segmentation'.

All the proven discovery / dedup / gate logic and the spatial-offset contract are
preserved verbatim from _gausss.

Kernel/env: run under pertpy_env (or novae_env). Runners MUST export
PYTHONNOUSERSITE=1; the sys.path strip below is belt-and-braces so a stale
~/.local anndata (rejects zarr>2) never shadows the env's own anndata.
"""
import sys
# Drop user-site paths so the env's own anndata/scanpy are used (see docstring).
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ======================================================================
# RAW DATA
# ======================================================================
SPATIAL_ROOT = Path("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial")
RANGER_FINAL = SPATIAL_ROOT / "Ranger_procd_mw_final"          # Marcel's "_final" segmentation
METADATA_CSV = SPATIAL_ROOT / "ALS_Xenium_sample_metadata.csv"  # demographics (join by sd_code)
STATUS_CSV   = SPATIAL_ROOT / "status.csv"                      # disease-group sections

SEG_PIPELINE = "mw_final"   # provenance tag written to obs['seg_pipeline'] (was 'mw_gausss')

# ======================================================================
# DISCOVERY / DEDUP   (ported verbatim from the proven _gausss recipe)
# ======================================================================
EXPECTED_SAMPLE_COUNT = 20
EXPECTED_PANEL_SIZE   = 480                                    # Gene Expression genes (of 541 total)
EXPECTED_DONOR_SPLIT  = {"control": 10, "sporadic": 6, "c9": 4}

# Dirs that are NOT samples. IGNORE_SUBSTRINGS matched on the raw name; EXCLUDE_DIR_NAMES
# matched after stripping a trailing ' Kopie'. '_reseg_results' is scratch.
IGNORE_SUBSTRINGS = ("Kopie",)          # any dir whose name contains this is skipped
IGNORE_EXACT      = {"_reseg_results"}

# Superseded duplicate-donor originals; their ALS6 redos are kept. NOT a QC filter.
EXCLUDE_DIR_NAMES = {
    "SD01620_BI",   # superseded by redo dir SD016_20_BI  (C9 donor SD016/20)
    "SD02022",      # superseded by redo dir SD020_22_BI  (C9 donor SD020/22)
}

# Samples that MUST survive discovery (reseg-refresh sanity gate).
REQUIRED_SD_CODES = {"SD00614", "SD02818"}

# SD-code-free / label-fix directory names -> canonical sample label.
SAMPLE_LABEL_OVERRIDES = {
    "Region_2": "SD05413_BG",   # ALS5 Region_2 = SD054/13 BG, Control F51 cervical (QC report s4.6)
}

# status.csv corrections (donor-level, keyed by sd_code).
STATUS_OVERRIDES = {
    "SD01616": "sporadic",      # listed under C9ORF in status.csv; project override C9->sporadic
}

# ======================================================================
# SPATIAL OFFSET   (grid tiling for whole-cohort PLOTTING ONLY)
# ======================================================================
# CONTRACT: obsm['spatial_orig'] = TRUE microns (never shifted) -> use for ALL spatial
#   GRAPHS (Novae/squidpy delaunay, Moran's I) with slide_key='run_id'. obsm['spatial'] =
#   grid-offset copy for whole-cohort PLOTTING ONLY. NEVER build a neighbour graph on
#   obsm['spatial']: the 2e6-um inter-slide gap fragments it.
SPATIAL_OFFSET    = 2_000_000.0
SPATIAL_GRID_COLS = 10

# ======================================================================
# _FINAL OUTPUT PATHS   (see OUTPUT NAMING; _FINAL suffix never clobbers _gausss)
# ======================================================================
STAGED_CODE_DIR = SPATIAL_ROOT / "final_rerun_code"     # all staged code + this module
LOG_DIR         = STAGED_CODE_DIR / "logs"              # SLURM --output/--error target

H5AD_DIR         = SPATIAL_ROOT / "SC_MNcorrected_FINAL_h5ads"
COMBINED_H5AD    = H5AD_DIR / "ALS_SCXenium_MNcorrected_FINAL_concatenated_offset_allsamples_preQC_20260719.h5ad"
PERSAMPLE_SUFFIX = "__MNcorrected_FINAL.h5ad"           # per-sample: <LABEL>__MNcorrected_FINAL.h5ad in H5AD_DIR

NOVAE_JOINT_DIR   = SPATIAL_ROOT / "Novae_persample_niches_FINAL"
NOVAE_INDEP_DIR   = SPATIAL_ROOT / "Novae_persample_INDEPENDENT_FINAL"
ASTRO_DIR         = SPATIAL_ROOT / "astro_isolation_FINAL"
COARSE_TYPING_DIR = SPATIAL_ROOT / "CellTyping_coarse_FINAL"
DUALPASS_QC_DIR   = NOVAE_INDEP_DIR / "dualpass_qc"     # dual-pass QC lives under the INDEP out


def persample_h5ad(label):
    """Path of the per-sample PRE-QC h5ad for a resolved sample label."""
    return H5AD_DIR / f"{label}{PERSAMPLE_SUFFIX}"


# ======================================================================
# ID / LABEL HELPERS   (ported from the proven recipe)
# ======================================================================
def _norm_cell_id(idx):
    """Normalize cell ids to a canonical string array.

    _final ids are plain integers ('1','2',...) in BOTH the matrix barcodes and
    cells.parquet cell_id (delta D2). Cast to str, strip whitespace, and drop a
    stray '.0' if a float ever crept in. NO hyphen/'cell' stripping is needed --
    the old gm/hex namespace (which required it) is gone.
    """
    s = pd.Index(np.asarray(idx)).astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def choose_id_column(df):
    """Pick the cell-id column of a cells table / CSV (case-insensitive fallback)."""
    for c in ["cell_id", "barcode", "cell_barcode", "cell", "id", "Cell", "CellID", "cell_ID"]:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for cand in ["cell_id", "barcode", "cell", "id"]:
        if cand in lower:
            return lower[cand]
    raise ValueError(f"No cell ID column found. Columns (first 60): {list(df.columns[:60])}")


def extract_sd_code_anywhere(x):
    """Extract a canonical SDxxxxx donor code from any string form."""
    s = str(x)
    m = re.search(r"(SD\d{5})", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(SD\d{3})[_\-](\d{2})", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"
    m = re.search(r"(SD)\s*0*(\d+)\s*/\s*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2)):03d}{int(m.group(3)):02d}"
    return None


def resolve_label(dir_name):
    """Resolve a raw run-dir name to its canonical sample label.

    Order: (1) SAMPLE_LABEL_OVERRIDES (Region_2 -> SD05413_BG); (2) strip trailing
    underscores (SD00614_BG_ -> SD00614_BG, SD02818_BG_ -> SD02818_BG); (3) regex-
    extract the SD*_<block> label -- this also collapses the redo dirs
    (SD016_20_BI -> SD01620_BI, SD020_22_BI -> SD02022_BI), whose superseded
    originals are dropped via EXCLUDE_DIR_NAMES so there is no label collision.
    """
    for alias, ov in SAMPLE_LABEL_OVERRIDES.items():
        if dir_name == alias or alias in dir_name:
            return ov
    base = dir_name.rstrip("_")
    m = re.search(r"(SD\d{5}_[A-Za-z0-9]+)", base)
    if m:
        return m.group(1)
    m = re.search(r"(SD\d{3})[_\-](\d{2})[_\-]([A-Za-z0-9]+)", base)
    if m:
        return f"{m.group(1)}{m.group(2)}_{m.group(3)}"
    sd = extract_sd_code_anywhere(base)
    return sd if sd else base


# ======================================================================
# DISCOVERY
# ======================================================================
def _has_matrix(d):
    """True if dir d carries a complete 10x MTX directory (delta D1)."""
    m = d / "cell_feature_matrix"
    return m.is_dir() and all((m / f).exists() for f in
                              ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"))


def skip_reason(d):
    """None if d is a real sample dir, else a human-readable WHY it is skipped."""
    if not d.is_dir():
        return "not a directory"
    name = d.name
    if name in IGNORE_EXACT:
        return "reseg scratch dir (IGNORE_EXACT)"
    if any(sub in name for sub in IGNORE_SUBSTRINGS):
        return "partial German-named copy (' Kopie')"
    base = name.replace(" Kopie", "").strip()
    if base in EXCLUDE_DIR_NAMES:
        return "superseded duplicate-donor original (EXCLUDE_DIR_NAMES; redo is kept)"
    if name.startswith("_"):
        return "underscore-prefixed non-sample dir"
    if not _has_matrix(d):
        return "no complete cell_feature_matrix/ MTX dir"
    return None


def discover_samples(verbose=True, strict=False):
    """Discover the kept sample dirs on RANGER_FINAL.

    Returns [(label, Path), ...] for the 20 kept samples (label = resolve_label),
    printing every skipped dir with its reason and a count / sd_code sanity summary.

    strict=False (default) only WARNS on the three cohort-integrity conditions -- kept
    count != EXPECTED_SAMPLE_COUNT, a duplicate sd_code across kept samples, or a missing
    REQUIRED_SD_CODE -- which is correct on _final TODAY (exactly 20 kept, all sd_codes
    unique, both reseg samples present). strict=True (M1 enabler) turns each of those into
    a HARD ValueError: the concat builder should call discover_samples(strict=True) so a
    future _final dir change (a stray reseg dir that resolves to a colliding label, or a
    missing redo) HALTS the build instead of silently poisoning every downstream _FINAL
    analysis. The donor-SPLIT invariant (EXPECTED_DONOR_SPLIT = control 10 / sporadic 6 /
    c9 4) needs the status.csv join and so is enforced downstream in the concat builder,
    NOT here.
    """
    root = RANGER_FINAL
    all_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    kept, skipped = [], []
    for p in all_dirs:
        r = skip_reason(p)
        if r is None:
            kept.append((resolve_label(p.name), p))
        else:
            skipped.append((p.name, r))

    # Cohort-integrity checks (computed regardless of verbose so strict can enforce them).
    sds = [extract_sd_code_anywhere(lab) for lab, _ in kept]
    dup = sorted({c for c in sds if c and sds.count(c) > 1})
    missing = sorted(REQUIRED_SD_CODES - set(sds))
    wrong_count = len(kept) != EXPECTED_SAMPLE_COUNT

    if verbose:
        print("=" * 72)
        print(f"DISCOVERY  |  {root}")
        print("=" * 72)
        print(f"total dirs={len(all_dirs)}  kept={len(kept)}  skipped={len(skipped)}  "
              f"(expect {EXPECTED_SAMPLE_COUNT} kept)")
        print(f"skipped ({len(skipped)}):")
        for n, why in skipped:
            print(f"  - {n:<24}: {why}")
        print(f"kept ({len(kept)}, dir -> label):")
        for lab, p in kept:
            tag = ""
            if p.name in SAMPLE_LABEL_OVERRIDES or any(a in p.name for a in SAMPLE_LABEL_OVERRIDES):
                tag = "   [LABEL OVERRIDE]"
            elif p.name.endswith("_"):
                tag = "   [trailing-underscore strip]"
            print(f"  + {p.name:<24} -> {lab}{tag}")
        if dup:
            print(f"  WARNING: duplicate sd_code across kept samples: {dup} (a superseded "
                  f"original was not excluded, or a label collision)")
        if missing:
            print(f"  WARNING: required reseg samples missing from kept set: {missing}")
        if wrong_count:
            print(f"  WARNING: kept {len(kept)} != EXPECTED_SAMPLE_COUNT {EXPECTED_SAMPLE_COUNT} "
                  f"-- inspect the skip reasons above.")
        print("=" * 72, flush=True)

    if strict:
        problems = []
        if wrong_count:
            problems.append(f"kept {len(kept)} != EXPECTED_SAMPLE_COUNT {EXPECTED_SAMPLE_COUNT}")
        if dup:
            problems.append(f"duplicate sd_code across kept samples: {dup}")
        if missing:
            problems.append(f"required reseg samples missing: {missing}")
        if problems:
            raise ValueError(
                f"discover_samples(strict=True) cohort-integrity FAIL on {root}: "
                + "; ".join(problems)
                + ". Inspect the skip reasons (run with verbose=True) before any _FINAL build.")

    return kept


# ======================================================================
# PER-SAMPLE LOADER
# ======================================================================
def load_final_sample(label, sample_dir, verbose=True):
    """Load one _final sample into a PRE-QC AnnData (480-GE, integer-id joined).

    Steps (deltas D1-D4):
      * read the 10x MTX dir with sc.read_10x_mtx(gex_only=False)          [D1]
      * subset var to the 480 Gene Expression features                     [D4]
      * join cells.parquet 1:1 by plain-integer cell_id, LOG the overlap   [D2]
      * set obs_names '<LABEL>:<cell_id>'; xenium_cell_id = the integer str [D3]
      * obsm['spatial'] = obsm['spatial_orig'] = TRUE microns (centroids).
        The grid offset is applied later by apply_grid_offset(...) in the
        concat loop -- per-sample files keep true microns.

    Returns the AnnData. Raises loudly on a missing MTX/cells file, a panel-size
    mismatch, non-unique ids, or a zero-overlap join.
    """
    import warnings
    import scanpy as sc  # deferred so path-only consumers don't pay the import

    sample_dir = Path(sample_dir)
    mtx_dir = sample_dir / "cell_feature_matrix"
    cells_parquet = sample_dir / "cells.parquet"
    cells_csv_gz = sample_dir / "cells.csv.gz"

    if not _has_matrix(sample_dir):
        raise FileNotFoundError(f"Missing/incomplete 10x MTX dir: {mtx_dir}")
    if not (cells_parquet.exists() or cells_csv_gz.exists()):
        raise FileNotFoundError(f"Missing cell metadata (cells.parquet / cells.csv.gz) in {sample_dir}")

    # --- matrix (10x MTX DIR; _final has NO .h5) ---------------------- [D1]
    # read_10x_mtx parses the integer barcodes and emits a benign UserWarning that the
    # obs index is not string; we fix it on the next line, so silence just this read.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.read_10x_mtx(str(mtx_dir), gex_only=False)
    adata.var_names_make_unique()
    # read_10x_mtx parses the plain-integer barcodes ('1','2',...) into an int64 obs
    # index; cast to canonical strings BEFORE any var-slice -- anndata refuses to slice
    # a non-string / non-range integer obs index (TypeError in _normalize_index).  [D2]
    adata.obs_names = _norm_cell_id(adata.obs_names)
    if not adata.obs_names.is_unique:
        _nd = int(pd.Index(adata.obs_names).duplicated().sum())
        raise ValueError(f"{label}: matrix obs_names not unique after normalization ({_nd} dups)")
    if verbose:
        print(f"  [load] {label}: raw matrix n_obs={adata.n_obs}, n_vars={adata.n_vars}", flush=True)

    # --- subset to 480 Gene Expression ------------------------------- [D4]
    if "feature_types" not in adata.var.columns:
        raise RuntimeError(f"{label}: read_10x_mtx returned no 'feature_types' var column; "
                           f"cannot isolate Gene Expression from control probes.")
    n_all = adata.n_vars
    adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
    if verbose:
        print(f"  [filter] {label}: vars {n_all} -> {adata.n_vars} (Gene Expression only)")
    if adata.n_vars != EXPECTED_PANEL_SIZE:
        raise RuntimeError(f"{label}: panel size {adata.n_vars} != EXPECTED_PANEL_SIZE "
                           f"{EXPECTED_PANEL_SIZE}. Check the feature_types filter / gene_panel.")
    # Belt-and-braces: no control/blank probe names survived the GE filter.
    _ctrl = re.compile(r"NegControl|BLANK|antisense|Deprecated|UnassignedCodeword|Intergenic",
                       re.IGNORECASE)
    _leaked = sorted(g for g in adata.var_names if _ctrl.search(str(g)))
    if _leaked:
        raise RuntimeError(f"{label}: control/blank probe names leaked into the panel: {_leaked[:10]}")

    # --- cells.parquet (fallback csv.gz), integer cell_id 1:1 join --- [D2]
    # (matrix obs_names already normalized to canonical strings above)
    cells, src = None, None
    if cells_parquet.exists():
        try:
            cells = pd.read_parquet(cells_parquet); src = "cells.parquet"
        except Exception as e:
            if verbose:
                print(f"  [load] {label}: parquet failed ({type(e).__name__}); trying csv.gz")
    if cells is None:
        cells = pd.read_csv(cells_csv_gz, compression="gzip", low_memory=False); src = "cells.csv.gz"
    idc = choose_id_column(cells)
    cells[idc] = _norm_cell_id(cells[idc])
    cells = cells.set_index(idc)
    if not cells.index.is_unique:
        raise ValueError(f"{label}: cells metadata index not unique after normalization "
                         f"({int(cells.index.duplicated().sum())} dups)")

    mask = adata.obs_names.isin(cells.index)
    n_common = int(mask.sum())
    overlap = float(n_common / adata.n_obs) if adata.n_obs else 0.0
    print(f"  [join] {label}: matrix<->cells overlap {n_common}/{adata.n_obs} "
          f"= {overlap:.4f}  (src={src})  [expect ~1.0000; delta D2]", flush=True)
    if n_common == 0:
        raise ValueError(f"{label}: NO overlapping cell IDs between matrix and cells metadata "
                         f"-- the integer-id assumption (delta D2) is violated; inspect namespaces.")
    if overlap < 0.99:
        print(f"  [join] {label}: WARNING overlap {overlap:.4f} < 0.99 -- _final was expected to "
              f"be a clean 1:1; investigate before trusting this sample.", flush=True)

    common = adata.obs_names[mask]
    adata = adata[common].copy()
    adata.obs = adata.obs.join(cells.loc[common], how="left")

    # N2 (guardrail G15 downstream dependency): X keeps RAW counts and holds ONLY the 480
    # Gene Expression genes -- the 61 control probes are dropped. So neg-control QC
    # (neg_ctrl_rate, neg-ctrl FDR) MUST be sourced from the cells-metadata columns now
    # joined into obs (above), NOT from X. This joins ALL cells.parquet columns; the list
    # below is the EXACT _final cells.parquet neg-ctrl/QC schema (verified on live data
    # 2026-07-19, SD03614_BG) so the concat builder + dual-pass QC can confirm they are in
    # obs. NB (G15): per-region qv>20 / Q20 are TRANSCRIPT-level metrics that live in
    # transcripts.parquet, NOT in cells.parquet -- they are intentionally absent from this
    # cell-level list and MUST be computed downstream from transcripts.parquet.
    _qc_metric_cols = [c for c in (
        "transcript_counts", "control_probe_counts", "genomic_control_counts",
        "control_codeword_counts", "unassigned_codeword_counts",
        "deprecated_codeword_counts", "total_counts", "cell_area", "nucleus_area",
        "nucleus_count", "segmentation_method") if c in adata.obs.columns]
    if verbose:
        print(f"  [qc-cols] {label}: neg-ctrl/QC metric cols preserved in obs: "
              f"{_qc_metric_cols if _qc_metric_cols else 'NONE FOUND -- inspect cells schema'}",
              flush=True)

    # --- spatial (TRUE microns) -------------------------------------------
    for xcol, ycol in [("x_centroid", "y_centroid"), ("centroid_x", "centroid_y"),
                       ("x", "y"), ("x_location", "y_location"), ("x_center", "y_center")]:
        if xcol in adata.obs.columns and ycol in adata.obs.columns:
            xy = adata.obs[[xcol, ycol]].to_numpy(dtype=float)
            adata.obsm["spatial"] = xy
            adata.obsm["spatial_orig"] = xy.copy()
            adata.uns["spatial_columns"] = {"x": xcol, "y": ycol, "source": src}
            break
    else:
        print(f"  [spatial] {label}: WARNING no recognized centroid columns; obsm['spatial'] not set")

    # --- provenance obs / obs_names '<LABEL>:<cell_id>' -------------- [D3]
    sd_code = extract_sd_code_anywhere(label)
    adata.obs["xenium_cell_id"] = adata.obs_names.astype(str)      # plain integer string
    adata.obs_names = pd.Index([f"{label}:{c}" for c in adata.obs["xenium_cell_id"]])
    adata.obs["sample"] = label
    adata.obs["run_id"] = sample_dir.name                          # RAW dir (holds any MN CSVs)
    adata.obs["sd_code"] = sd_code if sd_code else ""
    adata.obs["seg_pipeline"] = SEG_PIPELINE
    adata.uns["sample_label"] = label
    adata.uns["sample_dir"] = str(sample_dir)
    adata.uns["xenium_outs_path"] = str(sample_dir)
    adata.uns["cell_join_overlap"] = overlap

    if verbose:
        print(f"  [ok] {label}: n_obs={adata.n_obs}, n_vars={adata.n_vars}, "
              f"sd_code={sd_code}, run_id={sample_dir.name}", flush=True)
    return adata


def apply_grid_offset(adata, kept_index):
    """Apply the cohort plotting grid offset to obsm['spatial'] (aliasing-safe).

    kept_index = 0-based rank among KEPT samples. Sets obsm['spatial_orig'] = TRUE
    microns (unshifted) and obsm['spatial'] = grid-offset copy (PLOTTING ONLY).
    xy is copied before shifting so the two arrays never alias. Call this in the
    concat loop; per-sample files should keep obsm['spatial'] == true microns.
    """
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    gx, gy = kept_index % SPATIAL_GRID_COLS, kept_index // SPATIAL_GRID_COLS
    shifted = xy.copy()
    shifted[:, 0] += gx * SPATIAL_OFFSET
    shifted[:, 1] += gy * SPATIAL_OFFSET
    adata.obsm["spatial_orig"] = xy       # true microns, unshifted
    adata.obsm["spatial"] = shifted       # grid-offset, plotting only
    adata.uns["spatial_offset"] = {"offset": float(SPATIAL_OFFSET),
                                   "grid_cols": int(SPATIAL_GRID_COLS),
                                   "gx": int(gx), "gy": int(gy), "index": int(kept_index)}
    return adata


if __name__ == "__main__":
    # Discovery-only self-check (no matrix load, no writes).
    kept = discover_samples(verbose=True)
    print(f"\ndiscover_samples() -> {len(kept)} kept samples")
    for lab, p in kept:
        print(f"  {lab:<14} <- {p.name}")
