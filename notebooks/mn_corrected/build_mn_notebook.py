#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | notebooks/mn_corrected/build_mn_notebook.py
#
# Builds the MN-corrected concatenation notebook by writing nbformat v4 JSON directly. Run it,
# then copy the resulting .ipynb to SCG.
#
# The reason it is a builder and not a notebook under version control is practical. jupytext
# is not installed in the environment, and a notebook edited in place accumulates execution
# counts and output blobs that make every diff unreadable. Generating it from a script keeps
# the logic reviewable.
#
# This is the _gausss generation of the concatenation, so it is the direct ancestor of
# pipeline_final/VH_concatenate_FINAL.py, which ports its logic one to one. The gates, the
# prints, the grid offset and the h5py sanitisation all originate here.
#
# MN_LABEL_SPEC is pinned to MN_motorneuron with 'yes' as the positive value, and it has to be
# pinned. A heuristic fallback would have selected MN_mn_shape instead. That column flags
# around 10,500 morphology candidates against roughly 150 real calls.
#
# The bool-before-numeric ordering in the sanitisation loops is the fix for is_MN being
# stringified into truthy "True" and "False", because pandas reports bool as a numeric dtype.
#
# Superseded for new work. Kept because the _gausss objects it produced are still referenced
# and because it documents where the current gates came from.
# ========================================================================================

"""
build_mn_notebook.py
Assembles the .ipynb for the MN-corrected (Gaussian reseg) concatenation pipeline
directly as nbformat-v4 JSON (no jupytext needed). Each cell 'source' is a single
string, which is valid nbformat.

Run locally, then scp the produced .ipynb to SCG.
"""
import json
import sys

CELLS = []

def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text})

# ============================================================
md(r"""# ALS SC Xenium — MN-corrected (Gaussian reseg) concatenation

Concatenates **Marcel's Gaussian-resegmented** Xenium samples
(`Ranger_procd_mw_gausss`) into one PRE-QC AnnData, and also writes **one h5ad per
sample**. This is the segmentation Rodrigo judged much better than the old
`Ranger_procd` run; it supersedes `ALS_SCXenium_concatenated_offset_allsamples_reseg_20260617.h5ad`.

Recipe is the proven `VH_concatenate_allsamples.py`, adapted for this folder:

* **Naming is `SD*` / `Region_2`, not `ALS*`** — sample selection is by "does this dir
  contain a `cell_feature_matrix.h5`", not by prefix.
* **Flat layout** — the Xenium outputs sit *directly* in each sample dir, there is no
  `outs/` subfolder, so `find_xenium_outs` checks the dir itself first.
* **`… Kopie` copies are ignored** (partial German-named duplicates) and so is
  `_reseg_results`.
* **Duplicate-donor exclusion kept**: originals `SD01620_BI` and `SD02022` are dropped;
  the ALS6 redos `SD016_20_BI` / `SD020_22_BI` are kept. NOT a QC filter.
* Worst-QC `SD01922_BI` (SD019/22) is KEPT — output is PRE-QC by design; QC is downstream.
* **MN labels**: if Marcel's `*_cell_classification_MN.csv` / `*_MN_TDP.csv` are readable,
  they are joined per cell into `obs`, and a boolean `is_MN` is derived. As of build time
  these files are mode `0700` (owner `mw28`) and NOT readable by `rodrigok` — the notebook
  will WARN and set `uns['mn_labels_joined']=False`. **Ask Marcel to `chmod g+r` them, then
  re-run**; MN labels then land in both the per-sample and concatenated objects.

**Outputs** (to `/oak/.../RK/Spatial/SC_MNcorrected_h5ads/`):
* per sample: `<sample_label>__MNcorrected.h5ad`
* combined:   `ALS_SCXenium_MNcorrected_gausss_concatenated_offset_allsamples_preQC_20260706.h5ad`

**Kernel**: run in `pertpy_env`. The first cell strips `~/.local` from `sys.path` to avoid
the stale user-site `anndata` that raises `zarr-python major version > 2 is not supported`.

Expected result (same donors as the 2026-06-17 run): **20 samples**, control 10 / sporadic 6 / c9 4.
""")

# ------------------------------------------------------------
md(r"""## Sample note — `Region_2` = SD054/13 BG (Control)

The Xenium run directory **`Region_2`** (batch ALS5, run 2025-01-21, slides 0027749/0027752) has no
`SD` donor code in its name. It is **SD054/13 BG — Control, Female, 51 y, Cervical Spinal Cord (BG
block), fresh-frozen**.

* **Provenance:** the ALS5 QC report / sample inventory (`ALS_Xenium_SampleInventory_2026-04-02.pdf`,
  Run-5 overview: *Region_2 (=SD054/13 BG)*), reconciled with the metadata cross-correlation
  (`ALS_SCXenium_Metadata_CrossCorrelation_2026-03-31.pdf`, where SD054/13 was earlier flagged
  *selected but not run*). SD054/13 was not un-run — it was run under the generic label `Region_2`.
* **How it flows through this notebook:** `SAMPLE_LABEL_OVERRIDES["Region_2"] = "SD05413_BG"` renames
  it, so `sd_code` resolves to `SD05413`; `status.csv` lists SD054/13 in the CONTROLS section, so
  `status = control`; and `ALS_Xenium_sample_metadata.csv` (row `SD05413`, `sample_ranger_procd =
  Region_2`) supplies its demographics. It is therefore **one of the 10 controls** in the 10 / 6 / 4 split.
* **Caveat:** this is a records-based reconciliation, not a genotype/barcode match. If Region_2's
  disease label ever becomes load-bearing for an ALS-vs-control claim, confirm it against slide records.
""")

# ------------------------------------------------------------
code(r"""# --- Environment guard -------------------------------------------------------
# pertpy_env's scanpy import breaks because a stale user-site (~/.local) anndata
# rejects zarr>2. Drop user-site paths so the env's own anndata is used.
import sys
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]

import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

print("scanpy", sc.__version__)
import anndata as ad
print("anndata", ad.__version__, "| path:", ad.__file__)
""")

# ------------------------------------------------------------
code(r"""# --- CONFIG ------------------------------------------------------------------
ranger_root = Path("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Ranger_procd_mw_gausss")
out_dir     = Path("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/SC_MNcorrected_h5ads")
metadata_csv = Path("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/ALS_Xenium_sample_metadata.csv")
status_csv   = Path("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/status.csv")

out_dir.mkdir(parents=True, exist_ok=True)
combined_h5ad = out_dir / "ALS_SCXenium_MNcorrected_gausss_concatenated_offset_allsamples_preQC_20260706.h5ad"

SPATIAL_OFFSET   = 2_000_000.0
SPATIAL_GRID_COLS = 10
VERBOSE = True
MAX_EXAMPLES = 5

SEG_PIPELINE = "mw_gausss"   # provenance tag written to obs['seg_pipeline']

# Directories that are NOT samples (matched after stripping a trailing ' Kopie').
# Copies contain only partial parquet/zip files; _reseg_results is scratch.
IGNORE_SUBSTRINGS = ("Kopie",)          # any dir whose name contains this is skipped
IGNORE_EXACT      = {"_reseg_results"}  # exact dir names to skip

# Superseded original runs (duplicate donors); their ALS6 redos are kept.
# NOTE: NOT a QC filter. Matched on the exact (Kopie-stripped) dir name.
EXCLUDE_DIR_NAMES = {
    "SD01620_BI",   # superseded by SD016_20_BI redo
    "SD02022",      # superseded by SD020_22_BI redo
}

# Samples that MUST be present (sanity gate on the reseg refresh).
REQUIRED_SD_CODES = {"SD00614", "SD02818"}

# Exact final sample count expected (fail loud if a sample is silently dropped).
EXPECTED_SAMPLE_COUNT = 20

# Exact Gene-Expression panel size expected (neg-ctrl/blank probes already dropped).
EXPECTED_PANEL_SIZE = 480

# Status corrections (errors in status.csv).
STATUS_OVERRIDES = {
    "SD01616": "sporadic",   # listed under C9ORF in status.csv; correct is Sporadic
}

# Sample-label overrides (SD-code-free directory names).
SAMPLE_LABEL_OVERRIDES = {
    "Region_2": "SD05413_BG",  # ALS5 Region_2 = SD054/13 BG, Control F51 (QC report s4.6)
}

# --- MN-label join spec ------------------------------------------------------
# Marcel exports per-sample MN classifications as:
#   <name>_cell_classification_MN.csv       -> columns prefixed MN_
#   <name>_cell_classification_MN_TDP.csv   -> columns prefixed MNTDP_
# Schema is unknown at build time (files are 0700). The notebook joins ALL columns
# verbatim (prefixed) and prints value_counts so the label column can be pinned.
# Once known, set MN_LABEL_SPEC to derive a clean boolean obs['is_MN'], e.g.
#   MN_LABEL_SPEC = {"column": "MN_cell_type", "positive_values": ["MN", "MNs", "motor neuron"]}
MN_JOIN_ENABLED = True
MN_LABEL_SPEC = {"column": "MN_motorneuron", "positive_values": ["yes"]}   # PINNED 2026-07-06: Marcel's *_cell_classification_MN.csv col 'motorneuron' (yes/no) joins as MN_motorneuron; is_MN=(=="yes"). ~149 MNs/sample (SD01922_BI). Pinning is required so the heuristic does NOT grab MN_mn_shape (~10.5k morphology candidates) instead of the stricter MN call.
# Fallback auto-detection vocabulary (used only if MN_LABEL_SPEC is None):
MN_POSITIVE_TOKENS = {"mn", "mns", "motor neuron", "motorneuron", "motor_neuron",
                      "alpha-mn", "alpha mn", "amn", "true", "1", "yes", "positive"}
""")

# ------------------------------------------------------------
code(r"""# --- HELPERS (adapted from the proven VH_concatenate_allsamples.py) ----------
def dprint(msg):
    if VERBOSE:
        print(msg, flush=True)

def normalize_cell_ids(idx):
    s = pd.Index(idx).astype(str)
    s = s.str.replace(r"-\d+$", "", regex=True)
    s = s.str.replace(r"^cell[_\-]?", "", regex=True)
    return pd.Index(s)

def choose_id_column(df):
    for c in ["cell_id", "barcode", "cell_barcode", "cell", "id", "Cell", "CellID", "cell_ID"]:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower = {str(c).lower(): c for c in df.columns}
    for cand in ["cell_id", "barcode", "cell", "id"]:
        if cand in lower:
            return lower[cand]
    raise ValueError(f"No cell ID column found. Columns (first 60): {list(df.columns[:60])}")

def find_xenium_outs(sample_dir):
    # mw_gausss layout: files live directly in the sample dir.
    if (sample_dir / "cell_feature_matrix.h5").exists():
        return sample_dir
    # fallbacks for classic layouts
    direct = sample_dir / "outs"
    if (direct / "cell_feature_matrix.h5").exists():
        return direct
    nested = sample_dir / sample_dir.name / "outs"
    if (nested / "cell_feature_matrix.h5").exists():
        return nested
    cands = []
    for p in sample_dir.rglob("cell_feature_matrix.h5"):
        try:
            rel = p.relative_to(sample_dir)
            if len(rel.parts) <= 6:
                cands.append(p.parent)
        except Exception:
            continue
    if cands:
        cands = sorted(cands, key=lambda x: (len(x.relative_to(sample_dir).parts), str(x)))
        return cands[0]
    return None

def extract_sd_code_anywhere(x):
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

def extract_sample_label(sample_dir, outs_dir):
    s = str(outs_dir)
    m = re.search(r"(SD\d{5}_[A-Za-z0-9]+)", s)
    if m:
        return m.group(1)
    m = re.search(r"(SD\d{3})[_\-](\d{2})[_\-]([A-Za-z0-9]+)", s)
    if m:
        return f"{m.group(1)}{m.group(2)}_{m.group(3)}"
    sd = extract_sd_code_anywhere(s)
    if sd:
        return sd
    return sample_dir.name

def _minmax2(xy):
    return (float(np.nanmin(xy[:, 0])), float(np.nanmax(xy[:, 0]))), \
           (float(np.nanmin(xy[:, 1])), float(np.nanmax(xy[:, 1])))

def parse_status_mapping_from_csv(csv_path):
    if not csv_path.exists():
        dprint(f"WARNING: status CSV not found: {csv_path}. obs['status'] will be 'unknown'.")
        return {}
    df = pd.read_csv(csv_path)
    cols_lower = {c: str(c).strip().lower() for c in df.columns}
    status_col = next((c for c, lc in cols_lower.items() if lc in {"status", "group", "genotype"}), None)
    sample_col = next((c for c, lc in cols_lower.items() if lc in {"sample", "sample_id", "id", "case number", "case_number"}), None)
    mapping = {}
    if status_col is not None and sample_col is not None:
        dprint(f"[status.csv] explicit columns: sample='{sample_col}', status='{status_col}'")
        for _, row in df.iterrows():
            sd = extract_sd_code_anywhere(row.get(sample_col))
            if not sd:
                continue
            st = str(row.get(status_col)).strip().lower()
            if not st or st == "nan":
                continue
            if "control" in st:   mapping[sd] = "control"
            elif "c9" in st:      mapping[sd] = "c9"
            elif "sporadic" in st: mapping[sd] = "sporadic"
            else:                 mapping[sd] = st
        return mapping
    # section-header parsing (status.csv format)
    section_map = {"SPORADIC": "sporadic", "C9ORF": "c9", "CONTROL": "control", "CONTROLS": "control"}
    first_col = df.columns[0]
    current = None
    dprint(f"[status.csv] section-header parsing from first column '{first_col}'")
    for _, row in df.iterrows():
        v = row.get(first_col)
        if pd.isna(v):
            continue
        txt = str(v).strip()
        if not txt:
            continue
        up = txt.upper()
        if up in {"CASE NUMBER", "CASENUMBER"}:
            continue
        if up in section_map:
            current = section_map[up]
            dprint(f"[status.csv] section -> {up} = '{current}'")
            continue
        sd = extract_sd_code_anywhere(up)
        if sd and current is not None:
            mapping[sd] = current
    return mapping
""")

# ------------------------------------------------------------
code(r"""# --- MN-label join helper ----------------------------------------------------
def join_mn_labels(adata, sample_dir):
    # Join Marcel's per-cell MN classifications if readable. Robust to unknown schema
    # and to permission errors (files are 0700 at build time).
    adata.uns["mn_labels_joined"] = False
    adata.uns["mn_source_files"] = []
    adata.uns["mn_columns_added"] = []
    if not MN_JOIN_ENABLED:
        return adata

    specs = [("MN_", sorted(sample_dir.glob("*_cell_classification_MN.csv"))),
             ("MNTDP_", sorted(sample_dir.glob("*_cell_classification_MN_TDP.csv")))]
    added_cols = []
    any_joined = False
    for prefix, files in specs:
        # MN_TDP.csv also matches *_MN.csv glob? No: *_MN.csv requires ending _MN.csv,
        # *_MN_TDP.csv ends _MN_TDP.csv. Guard anyway:
        files = [f for f in files if (prefix == "MNTDP_") == f.name.endswith("_MN_TDP.csv")]
        for f in files:
            try:
                mn = pd.read_csv(f, low_memory=False)
            except PermissionError:
                dprint(f"  [MN] PERMISSION DENIED: {f.name} (ask Marcel to chmod g+r); skipping")
                continue
            except Exception as e:
                dprint(f"  [MN] WARN: could not read {f.name}: {type(e).__name__}: {e}")
                continue
            try:
                idc = choose_id_column(mn)
            except ValueError as e:
                dprint(f"  [MN] WARN: {f.name} has no id column ({e}); skipping")
                continue
            mn[idc] = normalize_cell_ids(mn[idc]).astype(str)
            mn = mn[~mn[idc].duplicated(keep="first")].set_index(idc)
            # cell ids in adata.obs are 'run_id:xenium_cell_id'; match on xenium_cell_id
            key = adata.obs["xenium_cell_id"].astype(str)
            label_cols = [c for c in mn.columns if c != idc]
            ren = {c: f"{prefix}{c}" for c in label_cols}
            joined = mn[label_cols].rename(columns=ren).reindex(key.values)
            joined.index = adata.obs_names
            n_match = int(joined.notna().any(axis=1).sum())
            for c in ren.values():
                adata.obs[c] = joined[c].values
                added_cols.append(c)
            dprint(f"  [MN] joined {f.name}: {len(ren)} cols, {n_match}/{adata.n_obs} cells matched")
            adata.uns["mn_source_files"].append(f.name)
            any_joined = True
    adata.uns["mn_labels_joined"] = bool(any_joined)
    adata.uns["mn_columns_added"] = added_cols
    if not any_joined:
        dprint("  [MN] no MN labels joined for this sample (unreadable or absent)")
    return adata

def derive_is_MN(adata):
    # Add boolean obs['is_MN'] from the joined MN columns.
    # Uses MN_LABEL_SPEC if provided, else a heuristic over MN_ columns.
    adata.obs["is_MN"] = False
    if not adata.uns.get("mn_labels_joined", False):
        adata.uns["is_MN_source"] = "none (MN labels not joined)"
        return adata
    if MN_LABEL_SPEC is not None:
        col = MN_LABEL_SPEC["column"]
        if col not in adata.obs.columns:
            dprint(f"  [is_MN] WARN: spec column '{col}' not in obs; leaving is_MN=False")
            adata.uns["is_MN_source"] = f"spec column '{col}' missing"
            return adata
        pos = {str(v).strip().lower() for v in MN_LABEL_SPEC["positive_values"]}
        vals = adata.obs[col].astype(str).str.strip().str.lower()
        adata.obs["is_MN"] = vals.isin(pos).values
        adata.uns["is_MN_source"] = f"spec:{col}"
        return adata
    # Heuristic: scan MN_ columns for MN-positive tokens.
    mn_cols = [c for c in adata.obs.columns if c.startswith("MN_")]
    for c in mn_cols:
        vals = adata.obs[c].astype(str).str.strip().str.lower()
        hits = vals.isin(MN_POSITIVE_TOKENS)
        if hits.any():
            adata.obs["is_MN"] = (adata.obs["is_MN"].values | hits.values)
            adata.uns["is_MN_source"] = f"heuristic:{c}"
            dprint(f"  [is_MN] heuristic matched {int(hits.sum())} cells in '{c}'")
            return adata
    adata.uns["is_MN_source"] = "heuristic: no MN-positive tokens found (pin MN_LABEL_SPEC)"
    dprint("  [is_MN] heuristic found nothing; set MN_LABEL_SPEC after inspecting value_counts")
    return adata
""")

# ------------------------------------------------------------
code(r"""# --- Per-sample loader (adapted; flat mw_gausss layout) ----------------------
def load_sample_from_outs(outs_dir):
    h5_path = outs_dir / "cell_feature_matrix.h5"
    cells_parquet = outs_dir / "cells.parquet"
    cells_csv_gz = outs_dir / "cells.csv.gz"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing expression H5: {h5_path}")
    if not (cells_parquet.exists() or cells_csv_gz.exists()):
        raise FileNotFoundError(f"Missing cell metadata (cells.parquet / cells.csv.gz) in {outs_dir}")

    adata = sc.read_10x_h5(str(h5_path), gex_only=False)
    adata.var_names_make_unique()
    dprint(f"  [load] raw matrix: n_obs={adata.n_obs}, n_vars={adata.n_vars}")

    if "feature_types" in adata.var.columns:
        n_all = adata.n_vars
        adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
        dprint(f"  [filter] vars: {n_all} -> {adata.n_vars} (Gene Expression only)")
    else:
        dprint("  [filter] WARNING: no 'feature_types'; skipping GE filter")

    adata.obs_names = normalize_cell_ids(adata.obs_names)
    # Symmetric uniqueness guard: the cells-table side is checked below; the matrix
    # obs side must be too. If stripping '-\d+$' ever collapsed two matrix cell ids,
    # the join and adata[common] slice would corrupt silently.
    if not adata.obs_names.is_unique:
        _nd = int(pd.Index(adata.obs_names).duplicated().sum())
        raise ValueError(f"matrix obs_names not unique after normalization ({_nd} dups); "
                         f"the '-\\d+$' strip collided cell ids for {h5_path}")

    cells, src = None, None
    if cells_parquet.exists():
        try:
            cells = pd.read_parquet(cells_parquet); src = "cells.parquet"
        except Exception as e:
            dprint(f"  [load] parquet failed ({type(e).__name__}); trying csv.gz")
    if cells is None:
        cells = pd.read_csv(cells_csv_gz, compression="gzip", low_memory=False); src = "cells.csv.gz"

    idc = choose_id_column(cells)
    cells[idc] = normalize_cell_ids(cells[idc].astype(str)).astype(str)
    cells = cells.set_index(idc)
    if not cells.index.is_unique:
        raise ValueError(f"cells metadata index not unique after normalization ({int(cells.index.duplicated().sum())} dups)")

    mask = adata.obs_names.isin(cells.index)
    n_common = int(mask.sum())
    dprint(f"  [join] overlap: {n_common}/{adata.n_obs} cells have metadata (src={src})")
    if n_common == 0:
        raise ValueError("No overlapping cell IDs between matrix and cell metadata.")

    common = adata.obs_names[mask]
    adata = adata[common].copy()
    adata.obs = adata.obs.join(cells.loc[common], how="left")

    for xcol, ycol in [("x_centroid", "y_centroid"), ("centroid_x", "centroid_y"),
                       ("x", "y"), ("x_location", "y_location"), ("x_center", "y_center")]:
        if xcol in adata.obs.columns and ycol in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[[xcol, ycol]].to_numpy()
            adata.uns["spatial_columns"] = {"x": xcol, "y": ycol, "source": src}
            (xmn, xmx), (ymn, ymx) = _minmax2(adata.obsm["spatial"].astype(float))
            dprint(f"  [spatial] ({xcol},{ycol}) x=[{xmn:.1f},{xmx:.1f}] y=[{ymn:.1f},{ymx:.1f}]")
            break
    else:
        dprint("  [spatial] WARNING: no recognized spatial columns; obsm['spatial'] not set")

    adata.uns["xenium_outs_path"] = str(outs_dir)
    return adata
""")

# ------------------------------------------------------------
code(r"""# --- Load status + sample metadata -------------------------------------------
status_map = parse_status_mapping_from_csv(status_csv)
dprint(f"Loaded {len(status_map)} status mappings from {status_csv.name}")
for sd, ns in STATUS_OVERRIDES.items():
    dprint(f"[STATUS_OVERRIDE] {sd}: '{status_map.get(sd, 'unknown')}' -> '{ns}'")

sample_meta = None
if metadata_csv.exists():
    _m = pd.read_csv(metadata_csv).drop_duplicates(subset="sd_code", keep="last")
    sample_meta = _m.set_index("sd_code")
    dprint(f"Loaded sample metadata: {sample_meta.shape[0]} rows x {sample_meta.shape[1]} cols")
else:
    dprint(f"NOTE: {metadata_csv.name} not found — demographics absent; status still set")
""")

# ------------------------------------------------------------
code(r"""# --- Discover sample directories ---------------------------------------------
# skip_reason returns None if d is a real sample dir, else a human-readable WHY.
# is_candidate is defined in terms of it so the exclusion logic is not duplicated.
def skip_reason(d):
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
    return None

def is_candidate(d):
    return skip_reason(d) is None

all_dirs = sorted(p for p in ranger_root.iterdir() if p.is_dir())
excluded = sorted(((p.name, skip_reason(p)) for p in all_dirs if not is_candidate(p)),
                  key=lambda t: t[0])
sample_dirs = [p for p in all_dirs if is_candidate(p)]
print("=" * 72)
print("DISCOVERY PLAN  |  which dirs are kept as samples and which are skipped, and WHY")
print("=" * 72)
dprint(f"Total dirs: {len(all_dirs)} | candidate samples: {len(sample_dirs)} | skipped: {len(excluded)}")
dprint(f"Skipped dirs ({len(excluded)}):")
for n, why in excluded:
    dprint(f"  - {n:<28} : {why}")
dprint(f"Candidate sample dirs ({len(sample_dirs)}, expect {EXPECTED_SAMPLE_COUNT} to load):")
for p in sample_dirs:
    _ov = next((ov for al, ov in SAMPLE_LABEL_OVERRIDES.items() if al in p.name), None)
    if _ov:
        dprint(f"  + {p.name:<24} -> {_ov}  (SD-code-free dir; label override applied at load)")
    else:
        dprint(f"  + {p.name}")
print("=" * 72, flush=True)
""")

# ------------------------------------------------------------
code(r"""# --- MAIN LOOP: load, MN-join, offset, write per-sample h5ad -----------------
adatas = []
kept, skipped = 0, 0
loaded_sd_codes = []
offset_records = {}
persample_written = []

n_total = len(sample_dirs)
print("=" * 72)
print("RUN HEADER  |  ALS SC Xenium MN-corrected (Gaussian reseg) concatenation")
print("=" * 72)
print(f"  ranger_root       : {ranger_root}")
print(f"  out_dir           : {out_dir}")
print(f"  seg_pipeline      : {SEG_PIPELINE}")
print(f"  candidate dirs    : {n_total}")
print(f"  expected samples  : {EXPECTED_SAMPLE_COUNT}   (control 10 / sporadic 6 / c9 4)")
print(f"  expected panel    : {EXPECTED_PANEL_SIZE} Gene Expression genes")
print(f"  spatial offset    : {SPATIAL_OFFSET:g} um, grid {SPATIAL_GRID_COLS} cols  (plotting only)")
print(f"  MN join enabled   : {MN_JOIN_ENABLED}   (label spec pinned: {MN_LABEL_SPEC is not None})")
print("=" * 72, flush=True)

for i_dir, sample_dir in enumerate(sample_dirs, start=1):
    dprint(f"\n[ {i_dir:>2}/{n_total} ] === {sample_dir.name} ===")
    outs = find_xenium_outs(sample_dir)
    if outs is None:
        skipped += 1
        dprint(f"[ {i_dir:>2}/{n_total} ] [SKIP] {sample_dir.name}: no cell_feature_matrix.h5")
        continue
    try:
        adata = load_sample_from_outs(outs)

        sample_label = extract_sample_label(sample_dir, outs)
        for alias, ov in SAMPLE_LABEL_OVERRIDES.items():
            if alias in sample_dir.name:
                dprint(f"[LABEL_OVERRIDE] '{alias}' -> '{ov}'")
                sample_label = ov
                break
        sd_code = extract_sd_code_anywhere(sample_label)
        status = status_map.get(sd_code, "unknown") if sd_code else "unknown"
        if sd_code in STATUS_OVERRIDES:
            dprint(f"[STATUS_OVERRIDE] {sd_code}: '{status}' -> '{STATUS_OVERRIDES[sd_code]}'")
            status = STATUS_OVERRIDES[sd_code]

        run_id = sample_dir.name
        adata.obs["xenium_cell_id"] = adata.obs_names.astype(str)
        adata.obs_names = pd.Index([f"{run_id}:{c}" for c in adata.obs["xenium_cell_id"]])
        adata.obs["sample"] = sample_label
        adata.obs["status"] = status
        adata.obs["run_id"] = run_id
        adata.obs["sd_code"] = sd_code if sd_code else ""
        adata.obs["seg_pipeline"] = SEG_PIPELINE
        adata.uns["sample_dir"] = str(sample_dir)
        adata.uns["sample_label"] = sample_label

        # MN labels (if readable) + is_MN
        adata = join_mn_labels(adata, sample_dir)
        adata = derive_is_MN(adata)

        # spatial grid offset (same scheme as the proven recipe)
        # CONTRACT: obsm['spatial_orig'] = TRUE microns (never shifted) -> use for ALL
        #   spatial GRAPHS (Novae/squidpy delaunay, Moran's I) with slide_key='run_id'.
        #   obsm['spatial'] = grid-offset copy for whole-cohort PLOTTING ONLY (tiles the
        #   slides on a 10-col grid so they don't overlap). Never build a neighbour graph
        #   on obsm['spatial']: the 2e6-um inter-slide gap would fragment it.
        # xy is copied before shifting so spatial_orig and spatial do not alias.
        if "spatial" in adata.obsm:
            xy = adata.obsm["spatial"].astype(float)
            idx = kept   # 0-based rank among KEPT samples; failed samples never consume a slot
            gx, gy = idx % SPATIAL_GRID_COLS, idx // SPATIAL_GRID_COLS
            shifted = xy.copy()
            shifted[:, 0] += gx * SPATIAL_OFFSET
            shifted[:, 1] += gy * SPATIAL_OFFSET
            adata.obsm["spatial_orig"] = xy       # true microns, unshifted
            adata.obsm["spatial"] = shifted       # grid-offset, plotting only
            adata.uns["spatial_offset"] = {"offset": float(SPATIAL_OFFSET),
                "grid_cols": int(SPATIAL_GRID_COLS), "gx": int(gx), "gy": int(gy), "index": int(idx)}
            offset_records[run_id] = {"sample": sample_label, "sd_code": sd_code, "status": status,
                "offset": float(SPATIAL_OFFSET), "grid_cols": int(SPATIAL_GRID_COLS),
                "gx": int(gx), "gy": int(gy), "index": int(idx)}
        else:
            adata.uns["spatial_offset"] = None

        dprint(f"[ {i_dir:>2}/{n_total} ] [OK] {sample_label} (dir={sample_dir.name}): "
               f"n_obs={adata.n_obs}, n_vars={adata.n_vars}, sd_code={sd_code}, status={status}, "
               f"MN_joined={adata.uns['mn_labels_joined']}, is_MN={int(adata.obs['is_MN'].sum())} "
               f"| running kept={kept + 1}/{EXPECTED_SAMPLE_COUNT}")

        # write PRE-QC per-sample h5ad (sanitize obs for h5py)
        # A single-slide file has no need for the cohort grid offset, and a grid-shifted
        # obsm['spatial'] is a footgun for naive per-sample squidpy/plotting. So in the
        # per-sample file set obsm['spatial'] back to TRUE microns (== spatial_orig).
        # The COMBINED object keeps the grid offset for whole-cohort plotting.
        aw = adata.copy()
        if "spatial_orig" in aw.obsm:
            aw.obsm["spatial"] = aw.obsm["spatial_orig"].copy()
            aw.uns["spatial_offset"] = None
            aw.uns["spatial_note"] = "per-sample file: obsm['spatial']==spatial_orig (true microns); grid offset applied only in the combined object"
        _san = []
        for col in list(aw.obs.columns):
            s = aw.obs[col]
            if isinstance(s.dtype, pd.CategoricalDtype):
                continue
            # bool (is_MN) and numeric write cleanly to HDF5 — keep their dtype.
            # NOTE: pd.api.types.is_numeric_dtype(bool_series) is TRUE, so bool must be
            # checked FIRST or is_MN gets stringified to 'True'/'False' (a truthy-string bug).
            if s.dtype == bool:
                continue
            if pd.api.types.is_numeric_dtype(s):
                continue
            aw.obs[col] = s.astype(object).where(s.notna(), "NA").astype(str)
            _san.append(col)
        ps_path = out_dir / f"{sample_label}__MNcorrected.h5ad"
        aw.write_h5ad(str(ps_path), compression="gzip")
        persample_written.append(ps_path.name)
        dprint(f"[WRITE] per-sample -> {ps_path.name}")

        adatas.append(adata)
        loaded_sd_codes.append(sd_code)
        kept += 1
    except Exception as e:
        skipped += 1
        dprint(f"[ {i_dir:>2}/{n_total} ] [FAIL] {sample_dir.name}: {type(e).__name__}: {e}")
        continue

print("\n" + "-" * 72)
print(f"MAIN LOOP DONE  |  loaded {kept}  |  skipped/failed {skipped}  |  of {n_total} candidate dirs")
print("-" * 72, flush=True)
""")

# ------------------------------------------------------------
code(r"""# --- GATES: required samples, exact count, panel consistency -----------------
missing = REQUIRED_SD_CODES - set(loaded_sd_codes)
if missing:
    raise RuntimeError(f"Required reseg samples missing: {sorted(missing)}; loaded={sorted(set(loaded_sd_codes))}")
dprint(f"[gate] required reseg samples present: {sorted(REQUIRED_SD_CODES)}")

if kept != EXPECTED_SAMPLE_COUNT:
    raise RuntimeError(f"Expected {EXPECTED_SAMPLE_COUNT} samples, got {kept}. "
                       f"A sample was silently dropped — inspect [FAIL]/[SKIP] above.")
dprint(f"[gate] sample count OK: {kept}")

# Guard against a silent sd_code collision (e.g. a superseded original NOT excluded would
# collide with its redo). set(loaded_sd_codes) would hide it; count the raw list instead.
_dupes = sorted({c for c in loaded_sd_codes if c and loaded_sd_codes.count(c) > 1})
if _dupes:
    raise RuntimeError(f"Duplicate sd_code across kept samples (superseded original not "
                       f"excluded, or label collision): {_dupes}. loaded={loaded_sd_codes}")
dprint(f"[gate] sd_codes unique across {kept} kept samples")

# Donor-split gate: the count can be 20 while a status is misassigned (wrong genotype or
# 'unknown'), silently corrupting the design. Gate the expected per-donor split.
EXPECTED_DONOR_SPLIT = {"control": 10, "sporadic": 6, "c9": 4}
_donor_status = {}   # sample_label -> status (donor-level, not cell-level)
for a in adatas:
    _donor_status[a.uns["sample_label"]] = str(a.obs["status"].iloc[0])
_split = {}
for st in _donor_status.values():
    _split[st] = _split.get(st, 0) + 1
if _split != EXPECTED_DONOR_SPLIT:
    raise RuntimeError(f"Donor split {_split} != expected {EXPECTED_DONOR_SPLIT}. "
                       f"A status is misassigned or 'unknown' — check status.csv parsing / "
                       f"STATUS_OVERRIDES / SAMPLE_LABEL_OVERRIDES. Per-donor: {_donor_status}")
dprint(f"[gate] donor split OK: {_split}")

var_sets = [set(a.var_names) for a in adatas]
common = set.intersection(*var_sets)
union = set.union(*var_sets)
dprint(f"[panel] per-sample n_vars: {[a.n_vars for a in adatas]}")
if len(common) != len(union):
    raise RuntimeError(f"Gene panel MISMATCH: {len(union)-len(common)} gene(s) not shared by all samples: "
                       f"{sorted(union - common)[:15]}")
dprint(f"[panel] OK: all {kept} samples share identical {len(common)}-gene panel")

# Assert the exact Gene-Expression panel size (neg-ctrl/blank probes already dropped upstream).
if len(common) != EXPECTED_PANEL_SIZE:
    raise RuntimeError(f"Panel size {len(common)} != expected {EXPECTED_PANEL_SIZE} Gene Expression genes. "
                       f"Check the 'feature_types == Gene Expression' filter and that no control "
                       f"probes (NegControlProbe/NegControlCodeword/BLANK/antisense/Deprecated) leaked in.")
dprint(f"[panel] OK: panel size == {EXPECTED_PANEL_SIZE} Gene Expression genes (assertion passed)")

# Belt-and-braces: verify no control/blank probe names survived the GE filter.
_ctrl_pat = re.compile(r"NegControl|BLANK|antisense|Deprecated|UnassignedCodeword|Intergenic", re.IGNORECASE)
_leaked = sorted(g for g in common if _ctrl_pat.search(str(g)))
if _leaked:
    raise RuntimeError(f"Control/blank probes leaked into the panel: {_leaked[:15]}")
dprint("[panel] OK: no control/blank probe names present in the panel")
""")

# ------------------------------------------------------------
code(r"""# --- Concatenate + restore uns + metadata join + write -----------------------
adata_all = sc.concat(adatas, axis=0, join="outer", merge="same", label=None, index_unique=None)
adata_all.obs_names_make_unique()
adata_all.obs["sample"] = adata_all.obs["sample"].astype("category")
adata_all.obs["status"] = adata_all.obs["status"].astype("category")

adata_all.uns["spatial_offset_table"] = offset_records
adata_all.uns["concat_provenance"] = {
    "script": "Spatial_MN_correct concatenation notebook",
    "ranger_root": str(ranger_root),
    "seg_pipeline": SEG_PIPELINE,
    "excluded_superseded": sorted(EXCLUDE_DIR_NAMES),
    "spatial_offset": float(SPATIAL_OFFSET),
    "spatial_grid_cols": int(SPATIAL_GRID_COLS),
    "coords": "obsm['spatial']=grid-offset (plotting); obsm['spatial_orig']=true microns (use for graphs, slide_key='run_id')",
    "qc": "PRE-QC (no cell/gene filtering)",
    "mn_labels_joined_any": any(a.uns.get("mn_labels_joined", False) for a in adatas),
}

# demographics join by sd_code
if sample_meta is not None:
    skip = {"sample_ranger_procd", "sd_code_raw", "disease", "run", "batch"}
    meta_cols = [c for c in sample_meta.columns if c not in skip]
    adata_all.obs = adata_all.obs.join(sample_meta[meta_cols], on="sd_code", how="left")
    n_match = int(adata_all.obs[meta_cols[0]].notna().sum()) if meta_cols else 0
    dprint(f"Metadata join: {n_match}/{adata_all.n_obs} cells matched ({len(meta_cols)} cols)")

# sanitize obs for h5py write
_san = []
for col in list(adata_all.obs.columns):
    s = adata_all.obs[col]
    if isinstance(s.dtype, pd.CategoricalDtype):
        continue
    # bool (is_MN) and numeric write cleanly to HDF5 — keep their dtype.
    # bool MUST be tested before is_numeric_dtype (which is True for bool) so is_MN
    # is not stringified to 'True'/'False' (both truthy -> silently breaks masking).
    if s.dtype == bool:
        continue
    if pd.api.types.is_numeric_dtype(s):
        continue
    adata_all.obs[col] = s.astype(object).where(s.notna(), "NA").astype(str)
    _san.append(col)
if _san:
    dprint(f"[sanitize] coerced {len(_san)} obs cols to str: {_san}")

dprint(f"\nCombined: n_obs={adata_all.n_obs}, n_vars={adata_all.n_vars}")
dprint("Status counts (cells):\n" + str(adata_all.obs["status"].value_counts(dropna=False)))
_ds = adata_all.obs[["sample", "status"]].drop_duplicates()
dprint("Donors per status:\n" + str(_ds.groupby("status")["sample"].nunique()))

dprint(f"\nWriting combined -> {combined_h5ad}")
adata_all.write_h5ad(str(combined_h5ad), compression="gzip")
dprint("Done.")

# --- FINAL SUMMARY TABLE (per-sample) ---------------------------------------
print("\n" + "=" * 72)
print("FINAL SUMMARY  |  per-sample (PRE-QC)")
print("=" * 72)
_summ = []
for a in adatas:
    _orig = a.obsm.get("spatial_orig")
    if _orig is not None:
        _x = (float(np.nanmin(_orig[:, 0])), float(np.nanmax(_orig[:, 0])))
        _y = (float(np.nanmin(_orig[:, 1])), float(np.nanmax(_orig[:, 1])))
        _ext = f"x[{_x[0]:.0f},{_x[1]:.0f}] y[{_y[0]:.0f},{_y[1]:.0f}]"
    else:
        _ext = "no spatial"
    _summ.append({"sample": a.uns["sample_label"], "sd_code": a.obs["sd_code"].iloc[0],
                  "status": a.obs["status"].iloc[0], "n_obs": a.n_obs, "n_vars": a.n_vars,
                  "MN_joined": a.uns.get("mn_labels_joined", False),
                  "n_is_MN": int(a.obs["is_MN"].sum()),
                  "spatial_orig_um": _ext})
_summ_df = pd.DataFrame(_summ).sort_values(["status", "sample"]).reset_index(drop=True)
print(_summ_df.to_string(index=True))
print("-" * 72)
print(f"TOTAL cells : {adata_all.n_obs:,}   |   samples : {len(adatas)}   |   panel : {adata_all.n_vars}")
print("Donors per status :", dict(_summ_df.groupby("status")["sample"].nunique()))
print("Per-sample h5ads written :", len(persample_written))
print("=" * 72, flush=True)
""")

# ------------------------------------------------------------
code(r"""# --- MN validation report (per-sample + combined) ----------------------------
print("MN labels joined (any sample):", adata_all.uns['concat_provenance']['mn_labels_joined_any'])
print()
rows = []
for a in adatas:
    rows.append({"sample": a.uns["sample_label"], "status": a.obs["status"].iloc[0],
                 "n_cells": a.n_obs, "MN_joined": a.uns.get("mn_labels_joined", False),
                 "n_is_MN": int(a.obs["is_MN"].sum()), "is_MN_source": a.uns.get("is_MN_source", "")})
mn_summary = pd.DataFrame(rows).sort_values("sample")
print(mn_summary.to_string(index=False))

if adata_all.uns['concat_provenance']['mn_labels_joined_any']:
    print("\nCombined is_MN by status:")
    print(adata_all.obs.groupby("status")["is_MN"].agg(["sum", "size"]))
    # show the MN columns present + their value_counts so MN_LABEL_SPEC can be pinned
    mn_cols = [c for c in adata_all.obs.columns if c.startswith("MN_") or c.startswith("MNTDP_")]
    print("\nJoined MN columns:", mn_cols)
    for c in mn_cols[:12]:
        vc = adata_all.obs[c].astype(str).value_counts().head(8)
        print(f"\n[{c}] top values:\n{vc}")
else:
    print("\nMN labels were NOT joined (files unreadable). To enable:")
    print("  1) Ask Marcel to: chmod g+r <sample>/*_cell_classification_MN*.csv (group scg_lab_mpsnyder)")
    print("  2) Re-run this notebook. MN columns + is_MN will populate; then pin MN_LABEL_SPEC")
    print("     from the printed value_counts and re-run once more for a clean is_MN.")
""")

# ------------------------------------------------------------
md(r"""## After running

**Check the log for:**
* `Loaded 20 samples, skipped/failed 0` and `[gate] sample count OK: 20`
* `[panel] OK: all 20 samples share identical 480-gene panel`
* Donors per status = control 10 / sporadic 6 / c9 4
* MN report: whether `MN_joined` is True for all 20

**MN labels**: if the report says MN labels were not joined, that is the `chmod`
blocker — nothing else is wrong. Once Marcel opens the files, re-run; inspect the printed
`MN_` column value_counts; set `MN_LABEL_SPEC` (column + positive values) in the CONFIG cell;
re-run once more so `is_MN` is derived from the real label rather than the heuristic.

**Next**: build the dual-pass QC notebook on
`ALS_SCXenium_MNcorrected_gausss_concatenated_offset_allsamples_preQC_20260706.h5ad`
(this object is PRE-QC by design). Use `obsm['spatial_orig']` (true microns) for any spatial
graph with `slide_key='run_id'`; `obsm['spatial']` is grid-offset for plotting only.
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "pertpy_env", "language": "python", "name": "pertpy_env"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = sys.argv[1] if len(sys.argv) > 1 else "ALS_SCXenium_MNcorrected_concatenation.ipynb"
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "with", len(CELLS), "cells")
