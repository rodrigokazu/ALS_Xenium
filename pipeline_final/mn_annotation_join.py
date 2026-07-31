#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/mn_annotation_join.py
#
# Attaches Marcel's motor-neuron calls to our cells. It is written defensively because the
# annotation has arrived late, in more than one shape, and once on the wrong segmentation.
#
# Three sources are tried per sample, in order: the per-sample *_cell_classification_MN.csv
# (plus the _MN_TDP.csv companion), then a CellAnnot clustered h5ad, and if neither is there
# the module gives up honestly by setting is_MN=False and stamping uns['is_MN_source']='none
# (annotation pending)'. Downstream scripts test that flag and degrade instead of inventing
# motor neurons.
#
# is_MN is derived strictly as MN_motorneuron == 'yes'. It is not MN_mn_shape. That looser
# morphology field flags around 10,500 candidate cells per section against roughly 150 to 500
# real calls, so a heuristic that grabs the wrong column inflates the MN population by more
# than an order of magnitude and every downstream statistic silently changes.
#
# The join key cost us real time and is now pinned. Marcel's predictions are keyed on a
# 0-based cell_index; _final cells.parquet uses a 1-based string cell_id. The relationship is
# cell_id == cell_index + 1, confirmed by Marcel across all 22 samples against the features
# table, and verified here on Region_2 by centroid match. _candidate_source_keys emits the +1
# form first and also offers the raw value so a future 1-based delivery self-corrects. It
# deliberately does not offer an index_core+1 variant, because a bare 0-based RangeIndex would
# produce an overlap near 1.0 purely by row position and mis-align everything while looking
# like a clean join.
#
# There is a second guard behind the overlap check. A source whose row count differs from the
# authoritative _final per-sample n_obs by more than one percent raises. That is what stops an
# annotation built on the older segmentation from spoofing a perfect join. The old prototype
# was 7.3 percent off and got caught. Set require_final_provenance=False only if you know why
# you are doing it.
# ========================================================================================

"""
mn_annotation_join.py -- robust is_MN + MN_*/MNTDP_* join for the _FINAL re-run.

Marcel's MN annotation on the _final segmentation is PENDING ("I also try to fix
the annotation ... I let you know"). This module is READY so that the moment it
lands -- in EITHER accepted form -- every staged script gets is_MN and the
MN_*/MNTDP_* covariates by calling join_is_MN(adata, label).

Per sample it tries, IN ORDER:
  (a) <RANGER_FINAL>/<run_dir>/*_cell_classification_MN.csv (+ *_MN_TDP.csv)
      -> columns prefixed MN_ / MNTDP_ (the old CSV export form)
  (b) CellAnnot/clustered_<LABEL>_res0.5.h5ad  (obs read via h5py)
      -> is_MN + MN_*/MNTDP_* already carried in obs
  (c) NEITHER, or best join overlap < MIN_OVERLAP_FRAC
      -> is_MN = False (real bool) + uns['is_MN_source'] = 'none (annotation pending)'

is_MN is derived STRICTLY:  is_MN = (MN_motorneuron == 'yes')   [MN_LABEL_SPEC].
NOT MN_mn_shape (the looser morphology candidate, ~14x more positives: 4573 vs 323
in the SD03522 prototype). Confirmed against the prototype obs on 2026-07-19.

Join keys are tried robustly (integer cell_id / xenium_cell_id / obs-name core-id);
the best-overlap key wins and the overlap fraction is printed LOUDLY. is_MN is
preserved as a REAL numpy bool so it survives the h5py write sanitize loop (which
tests dtype==bool BEFORE is_numeric_dtype, else bool -> 'True'/'False' strings).

*** OPEN ITEM -- confirm at launch. The EXACT _final join key is TBD until Marcel
regenerates the annotation on _final. The ONLY annotation that exists today, the
prototype CellAnnot/clustered_SD03522_BG_res0.5.h5ad, is built on the OLDER
segmentation: its obs_names are 'SD03522_BG:gm#####' and its xenium_cell_id is an
8-char hex, NEITHER of which matches the _final plain-integer cell ids ('1','2',...).
So on _final it yields overlap ~0 and this module CORRECTLY degrades (c) today.
When Marcel drops the regenerated annotation, verify the join reports overlap ~1.0
and is_MN counts land near the prototype's ~150-320 MNs/sample. ***
"""
import sys
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]

from pathlib import Path

import numpy as np
import pandas as pd

import final_config as cfg
from final_config import RANGER_FINAL, choose_id_column

# ======================================================================
# CONFIG
# ======================================================================
CELLANNOT_DIR = cfg.SPATIAL_ROOT / "CellAnnot"

# Strict is_MN spec (PINNED 2026-07-06, re-confirmed on the _final-era prototype
# 2026-07-19). The CSV form exports a 'motorneuron' col -> prefixed MN_motorneuron;
# the h5ad form already stores MN_motorneuron -- both converge on this column.
MN_LABEL_SPEC = {"column": "MN_motorneuron", "positive_values": ["yes"]}

# Accept a join only when the best candidate-key overlap is at least this. A correct
# _final 1:1 join is ~1.0; a namespace mismatch (old-seg prototype vs _final integers)
# is ~0 -> 0.5 decisively separates them. Anything accepted below 0.95 also prints a
# loud CAUTION (partial join -> confirm at launch).
MIN_OVERLAP_FRAC = 0.5

# ROW-COUNT PROVENANCE GUARD (N1 launch gate, guardrails G14+G5). A high integer-id
# overlap is NECESSARY BUT NOT SUFFICIENT proof the annotation was built on _final:
# EVERY segmentation emits integer cell ids 1..N, so an old/other-seg annotation with
# integer ids would ALSO overlap ~1.0 (seg-A cell#1 != seg-B cell#1 -- a SPOOFED join).
# MIN_OVERLAP_FRAC only separates a namespace mismatch (~0) from a join; it CANNOT tell
# correct-_final from wrong-seg-with-integer-ids. The cleanest programmatic discriminator
# is the annotation's ROW COUNT: a correct _final table has one row per _final cell
# (e.g. SD03522 == 119958), whereas the old-seg prototype has 111168 (7.3% off). Require
# the source row count to match the authoritative _final per-sample cell count within
# this fraction; a high-overlap join that FAILS this is a HARD provenance error (raise),
# not a silent degrade. 1% tolerates a within-seg drop of a few unassigned cells while
# still decisively rejecting the 7.3%-off old seg. NOTE: row-count is itself necessary-
# not-sufficient (a different seg with the SAME count is not caught) -- Marcel's WRITTEN
# confirmation that the annotation was built on _final remains the required human backstop.
ROW_COUNT_TOL_FRAC = 0.01


# ======================================================================
# ID NORMALIZATION
# ======================================================================
def _norm_id(values):
    """Canonical string ids: str, strip whitespace, drop a stray trailing '.0'."""
    s = pd.Series(np.asarray(values), dtype="object").astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.to_numpy()


def _strip_bundle_suffix(values):
    """Strip a trailing 10x/Xenium bundle suffix '-<digits>' (N3).

    e.g. 'gm00001-1' -> 'gm00001', '1-1' -> '1'; suffix-free ids are returned unchanged.
    Marcel's _final CSV id column may follow the bundle convention ('gmNNNNN-1' / '1-1');
    stripping recovers the plain integer id that matches the _final obs ids.
    """
    s = pd.Series(np.asarray(values), dtype="object").astype(str).str.strip()
    return s.str.replace(r"-\d+$", "", regex=True).to_numpy()


# ======================================================================
# _FINAL PROVENANCE  (row-count discriminator; see ROW_COUNT_TOL_FRAC)
# ======================================================================
def _final_cell_count(run_dir):
    """Authoritative _final per-sample cell count from the RAW run dir, or None.

    Reads the row count WITHOUT loading the matrix: cells.parquet metadata num_rows
    (cheap), else the barcodes.tsv.gz line count. Used as the reference for the row-count
    provenance guard so it stays correct even if join_is_MN is ever called on a QC-
    filtered AnnData (whose n_obs would be < the full _final count). Best-effort: any
    read failure returns None and the caller falls back to adata.n_obs.
    """
    if run_dir is None:
        return None
    run_dir = Path(run_dir)
    cells_parquet = run_dir / "cells.parquet"
    if cells_parquet.exists():
        try:
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(str(cells_parquet)).metadata.num_rows)
        except Exception:
            try:
                return int(pd.read_parquet(cells_parquet, columns=[]).shape[0])
            except Exception:
                pass
    barcodes = run_dir / "cell_feature_matrix" / "barcodes.tsv.gz"
    if barcodes.exists():
        try:
            import gzip
            with gzip.open(barcodes, "rt") as fh:
                return sum(1 for _ in fh)
        except Exception:
            pass
    return None


# ======================================================================
# H5AD OBS READER  (env-agnostic; avoids anndata choking on /uns/log1p base 'null')
# ======================================================================
def _read_h5ad_obs(path):
    """Reconstruct the /obs DataFrame of an .h5ad via h5py (index preserved).

    Handles: the index (_index attr), categorical groups (categories+codes, -1->NaN),
    nullable groups (values+mask), and plain datasets (bool/numeric/bytes->str).
    """
    import h5py

    def _decode(a):
        a = np.asarray(a)
        if a.dtype.kind in ("S", "O"):
            return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else x for x in a],
                            dtype=object)
        return a

    with h5py.File(str(path), "r") as f:
        if "obs" not in f:
            raise ValueError(f"{path}: no /obs group")
        g = f["obs"]
        idxkey = g.attrs.get("_index", "_index")
        if isinstance(idxkey, bytes):
            idxkey = idxkey.decode()
        index = _decode(g[idxkey][:]) if idxkey in g else np.arange(next(iter(g.values())).shape[0])

        data = {}
        for k in g.keys():
            if k == idxkey:
                continue
            node = g[k]
            if isinstance(node, h5py.Group):
                keys = set(node.keys())
                if "categories" in keys and "codes" in keys:
                    cats = _decode(node["categories"][:])
                    codes = np.asarray(node["codes"][:])
                    data[k] = np.array([cats[c] if c >= 0 else np.nan for c in codes], dtype=object)
                elif "values" in keys:                     # nullable int/bool
                    vals = np.asarray(node["values"][:]).astype(object)
                    if "mask" in keys:
                        vals[np.asarray(node["mask"][:]).astype(bool)] = np.nan
                    data[k] = _decode(vals)
                # else: unsupported group encoding -> skip
            else:
                try:
                    data[k] = _decode(node[:])
                except Exception:
                    pass
    return pd.DataFrame(data, index=pd.Index(index))


# ======================================================================
# SOURCE LOADING  (a) _final CSVs  ->  (b) CellAnnot h5ad
# ======================================================================
def _load_csv_source(run_dir, verbose=True):
    """Load Marcel's per-cell CSV classifications from a _final run dir, if present.

    <name>_cell_classification_MN.csv     -> MN_    prefixed columns
    <name>_cell_classification_MN_TDP.csv -> MNTDP_ prefixed columns
    Returns a DataFrame indexed by the (normalized string) cell id, or None.
    """
    specs = [("MN_", sorted(run_dir.glob("*_cell_classification_MN.csv")), "_MN.csv"),
             ("MNTDP_", sorted(run_dir.glob("*_cell_classification_MN_TDP.csv")), "_MN_TDP.csv")]
    frames = []
    for prefix, files, endswith in specs:
        # '*_cell_classification_MN.csv' would also glob-match '*_MN_TDP.csv'? No -- it
        # requires a literal '_MN.csv' ending; guard anyway so TDP never lands in MN_.
        files = [f for f in files if f.name.endswith(endswith)]
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
            except PermissionError:
                if verbose:
                    print(f"  [MN] PERMISSION DENIED: {f.name} (ask Marcel to chmod g+r); skipping")
                continue
            except Exception as e:
                if verbose:
                    print(f"  [MN] WARN: could not read {f.name}: {type(e).__name__}: {e}")
                continue
            try:
                idc = choose_id_column(df)
            except ValueError as e:
                if verbose:
                    print(f"  [MN] WARN: {f.name} has no id column ({e}); skipping")
                continue
            df["__idraw"] = df[idc].astype(str)
            df = df[~df["__idraw"].duplicated(keep="first")].set_index("__idraw")
            label_cols = [c for c in df.columns if c != idc]
            df = df[label_cols].rename(columns={c: f"{prefix}{c}" for c in label_cols})
            # M2: the raw id is carried by the DataFrame INDEX (set above to __idraw =
            # df[idc]) and is matched downstream via the 'index_core' candidate key (+ its
            # '-nosuffix' variant, N3). The id COLUMN itself is intentionally NOT kept as a
            # covariate, so it is dropped from label_cols here; the bare
            # 'cell_id'/'xenium_cell_id' candidate branch therefore never fires for a CSV
            # source (index_core carries it), which is correct.
            frames.append(df)
            if verbose:
                print(f"  [MN] read {f.name}: {df.shape[1]} cols, {df.shape[0]} rows "
                      f"(id col '{idc}')")
    if not frames:
        return None
    out = frames[0]
    for extra in frames[1:]:
        out = out.join(extra, how="outer")
    return out


def _load_source(label, run_dir, verbose=True):
    """Return (source_df, kind, desc) or (None, 'none', reason).

    (a) _final CSVs in run_dir  ->  (b) CellAnnot/clustered_<LABEL>_res0.5.h5ad.
    """
    if run_dir is not None:
        run_dir = Path(run_dir)
        if run_dir.is_dir():
            csv_df = _load_csv_source(run_dir, verbose=verbose)
            if csv_df is not None:
                return csv_df, "csv", f"{run_dir.name}/*_cell_classification_MN*.csv"

    h5 = CELLANNOT_DIR / f"clustered_{label}_res0.5.h5ad"
    if h5.exists():
        try:
            obs = _read_h5ad_obs(h5)
            return obs, "h5ad", h5.name
        except Exception as e:
            if verbose:
                print(f"  [MN] WARN: could not read {h5.name}: {type(e).__name__}: {e}")

    return None, "none", "no _final CSVs and no CellAnnot h5ad"


# ======================================================================
# JOIN-KEY CANDIDATES
# ======================================================================
def _candidate_source_keys(source_df):
    """Ordered [(name, values)] of candidate id representations on the SOURCE side.

    For each raw candidate we ALSO emit a bundle-suffix-stripped variant when any value
    carries a trailing '-<digits>' (N3: Marcel's _final CSV id col may follow the 10x
    bundle convention 'gmNNNNN-1' / '1-1'; stripping recovers the plain '1' that matches
    the _final integer obs ids). best-overlap picks whichever variant actually joins, so a
    stripped variant that does not help is simply never selected -- and a spurious win is
    still caught by the row-count provenance guard.

    OFF-BY-ONE (verified 2026-07-22): Marcel's annotation is keyed on a 0-BASED
    'cell_index' (same convention as annotation/neuron_predictions.parquet, and the
    future mn_prob/is_MN layer he'll add on the SAME sample+cell_index key), while the
    _final adata side keys on the 1-BASED matrix barcode ('xenium_cell_id' = '1','2',...;
    obs_names '<LABEL>:<cell_id>'). Confirmed empirically on Region_2: cell_index + 1 ==
    cells.parquet cell_id, centroid match |dx|+|dy| = 0.00000 with the +1 vs 252 without.
    So when a source carries 'cell_index' we emit a 'cell_index+1' candidate (0-based ->
    1-based) FIRST so it wins on ties; the raw 0-based 'cell_index' is kept as a fallback
    that self-corrects (scores strictly higher) should a future delivery already be
    1-based. Without this every label would land on the NEIGHBOURING cell -- a silent
    shift that still reports overlap ~1.0 and passes the row-count guard.
    """
    raw = []
    # 0-based cell_index -> 1-based adata namespace (see OFF-BY-ONE note above)
    if "cell_index" in source_df.columns:
        ci = pd.to_numeric(source_df["cell_index"], errors="coerce")
        if ci.notna().any():
            ci = ci.to_numpy()
            raw.append(("cell_index+1", ci + 1))   # verified-correct mapping; first -> wins ties
            raw.append(("cell_index", ci))         # fallback if a future delivery is 1-based
    for col in ("cell_id", "xenium_cell_id"):
        if col in source_df.columns:
            raw.append((col, source_df[col].to_numpy()))
    # obs/index core-id: the part after the last ':' ('SD:gm00002' -> 'gm00002',
    # plain '1' -> '1'); for _final CSVs the index IS already the integer id.
    # NOTE: we deliberately DO NOT emit an 'index_core+1' variant here. A bare 0-based
    # RangeIndex (the default for any parquet/CSV read without an explicit id index)
    # would then produce overlap ~1.0 purely from ROW COUNT -- a false friend that
    # reindexes labels by row POSITION, silently mis-aligning them if the source row
    # order differs from the adata. The only trustworthy 0-based signal is an EXPLICIT
    # 'cell_index' column (handled above with +1); that is exactly how Marcel's
    # neuron_predictions.parquet and the coming mn_prob/is_MN layer carry the key.
    idx_core = np.array([str(i).split(":")[-1] for i in source_df.index], dtype=object)
    raw.append(("index_core", idx_core))

    out = []
    for name, vals in raw:
        out.append((name, vals))
        has_suffix = pd.Series(np.asarray(vals), dtype="object").astype(str).str.contains(
            r"-\d+$", regex=True).any()
        if has_suffix:
            out.append((f"{name}-nosuffix", _strip_bundle_suffix(vals)))
    return out


def _adata_key(adata):
    """Canonical adata id series (index = obs_names): the plain-integer xenium_cell_id
    if present, else the obs-name core-id (part after the last ':')."""
    if "xenium_cell_id" in adata.obs.columns:
        vals = adata.obs["xenium_cell_id"].to_numpy()
    else:
        vals = np.array([str(n).split(":")[-1] for n in adata.obs_names], dtype=object)
    return pd.Series(_norm_id(vals), index=adata.obs_names)


# ======================================================================
# PUBLIC API
# ======================================================================
def join_is_MN(adata, label, run_dir=None, verbose=True,
               require_final_provenance=True, final_n_cells=None):
    """Join is_MN + MN_*/MNTDP_* onto adata for one sample; degrade gracefully.

    Always sets:
      obs['is_MN']                real numpy bool (all-False when degraded)
      uns['is_MN_source']         provenance string
      uns['is_MN_overlap_frac']   best candidate-key overlap fraction
      uns['is_MN_join_key']       the winning candidate key name (or None)
      uns['mn_columns_added']     list of MN_/MNTDP_ obs cols added (or [])
      uns['is_MN_provenance']     'row-count-ok' / 'RAISED' / 'BYPASSED-degraded' /
                                  'n/a (no confirmed join)'
      uns['is_MN_source_nrows']   annotation source row count (0 when no source)
      uns['is_MN_final_ncells']   authoritative _final per-sample cell count (or None)
      uns['is_MN_rowcount_ratio'] source_nrows / _final cell count (or None)

    run_dir defaults to RANGER_FINAL/<obs['run_id']> so the RAW dir (which holds any
    _final MN CSVs, e.g. 'Region_2', 'SD00614_BG_') is searched, not the label dir.

    require_final_provenance (default True) makes the row-count provenance check a HARD
      error (raise) per N1: a source that joins with high overlap but whose row count
      does NOT match the _final per-sample cell count within ROW_COUNT_TOL_FRAC is almost
      certainly from a DIFFERENT segmentation (a spoofed integer-id overlap) and must
      HALT the build rather than silently poison MN positioning. Set False ONLY to
      explicitly accept an annotation you have human-confirmed is on _final despite a
      row-count difference (it then LOUD-degrades to is_MN=False instead of raising).
    final_n_cells overrides the reference count; otherwise it is read from the RAW _final
      run dir (cells.parquet / barcodes), else falls back to adata.n_obs.
    """
    n = adata.n_obs
    adata.obs["is_MN"] = np.zeros(n, dtype=bool)
    adata.uns["is_MN_source"] = "none (annotation pending)"
    adata.uns["is_MN_overlap_frac"] = 0.0
    adata.uns["is_MN_join_key"] = None
    adata.uns["mn_columns_added"] = []
    adata.uns["is_MN_provenance"] = "n/a (no confirmed join)"
    adata.uns["is_MN_source_nrows"] = 0
    adata.uns["is_MN_final_ncells"] = None
    adata.uns["is_MN_rowcount_ratio"] = None

    if run_dir is None:
        if "run_id" in adata.obs.columns and len(adata.obs["run_id"]):
            run_dir = RANGER_FINAL / str(adata.obs["run_id"].iloc[0])
        else:
            run_dir = RANGER_FINAL / label

    source_df, kind, desc = _load_source(label, run_dir, verbose=verbose)
    if source_df is None:
        print(f"  [MN] {label}: NO annotation source ({desc}) -> is_MN=False (annotation pending)",
              flush=True)
        return adata

    adata_key = _adata_key(adata)
    best = None   # (keyname, overlap, aligned_source_df)
    for src_name, src_vals in _candidate_source_keys(source_df):
        sdf = source_df.copy()
        sdf["__k"] = _norm_id(src_vals)
        sdf = sdf[~sdf["__k"].duplicated(keep="first")].set_index("__k")
        overlap = float(adata_key.isin(sdf.index).mean())
        if verbose:
            print(f"  [MN] {label}: candidate key '{src_name}' overlap={overlap:.4f}")
        if best is None or overlap > best[1]:
            best = (src_name, overlap, sdf.reindex(adata_key.to_numpy()))

    keyname, overlap, aligned = best
    adata.uns["is_MN_overlap_frac"] = overlap
    adata.uns["is_MN_join_key"] = keyname

    if overlap < MIN_OVERLAP_FRAC:
        msg = (f"{kind}:{desc} present but best overlap {overlap:.4f} < {MIN_OVERLAP_FRAC} "
               f"via '{keyname}' (namespace mismatch -- annotation pending regen on _final)")
        adata.uns["is_MN_source"] = msg
        print(f"  [MN] {label}: DEGRADE -> is_MN=False. {msg}", flush=True)
        return adata

    # --- ROW-COUNT PROVENANCE GUARD (N1) -----------------------------------------
    # High integer-id overlap is necessary but NOT sufficient (see ROW_COUNT_TOL_FRAC).
    # Compare the annotation's row count to the authoritative _final per-sample cell
    # count; a high-overlap join with a mismatched row count is a wrong-segmentation
    # spoof and is a HARD error (unless the caller has human-confirmed _final provenance).
    n_source = int(source_df.shape[0])
    final_count = final_n_cells if final_n_cells is not None else _final_cell_count(run_dir)
    ref_n = int(final_count) if final_count is not None else int(adata.n_obs)
    ref_src = ("final_n_cells arg" if final_n_cells is not None
               else ("raw _final run dir" if final_count is not None else "adata.n_obs (fallback)"))
    ratio = (n_source / ref_n) if ref_n else 0.0
    adata.uns["is_MN_source_nrows"] = n_source
    adata.uns["is_MN_final_ncells"] = int(final_count) if final_count is not None else None
    adata.uns["is_MN_rowcount_ratio"] = float(ratio)
    print(f"  [MN] {label}: provenance -- source rows={n_source} vs _final cells={ref_n} "
          f"({ref_src}); ratio={ratio:.4f} (tol +/-{ROW_COUNT_TOL_FRAC:.0%})", flush=True)
    if abs(1.0 - ratio) > ROW_COUNT_TOL_FRAC:
        msg = (f"{label}: PROVENANCE MISMATCH -- annotation ({kind}:{desc}) row count "
               f"{n_source} != _final cell count {ref_n} (ratio {ratio:.4f}, tol "
               f"+/-{ROW_COUNT_TOL_FRAC:.0%}). A high-overlap integer-id join with a wrong "
               f"row count is almost certainly a DIFFERENT segmentation (spoofed overlap); "
               f"regenerate the annotation ON _final. If you have Marcel's written "
               f"confirmation it IS on _final, re-run with require_final_provenance=False.")
        if require_final_provenance:
            adata.uns["is_MN_provenance"] = "RAISED"
            raise RuntimeError(msg)
        adata.uns["is_MN_provenance"] = "BYPASSED-degraded"
        adata.uns["is_MN_source"] = "PROVENANCE BYPASSED -> " + msg
        print(f"  [MN] {label}: PROVENANCE BYPASSED (require_final_provenance=False) -> "
              f"is_MN=False. {msg}", flush=True)
        return adata
    adata.uns["is_MN_provenance"] = "row-count-ok"

    if overlap < 0.95:
        n_unmatched = int(round((1.0 - overlap) * n))
        print(f"  [MN] {label}: CAUTION partial join overlap={overlap:.4f} via '{keyname}' "
              f"(< 0.95) -- {n_unmatched} unmatched cells get NaN in the added MN_/MNTDP_ "
              f"covariates (is_MN itself is NaN->False safe); the downstream h5py write-"
              f"sanitize MUST handle NaN in these object/category cols (N4). Confirm at "
              f"launch that the annotation covers all cells.", flush=True)

    # add MN_/MNTDP_ covariate columns (aligned to adata order)
    add_cols = [c for c in aligned.columns if str(c).startswith("MN_") or str(c).startswith("MNTDP_")]
    for c in add_cols:
        adata.obs[c] = aligned[c].to_numpy()
    adata.uns["mn_columns_added"] = add_cols

    # derive strict is_MN (real bool)
    col = MN_LABEL_SPEC["column"]
    pos = {str(v).strip().lower() for v in MN_LABEL_SPEC["positive_values"]}
    if col in aligned.columns:
        vals = aligned[col].astype("object").astype(str).str.strip().str.lower()
        is_mn = np.asarray(vals.isin(pos).to_numpy(), dtype=bool)   # NaN->'nan'->False
        adata.obs["is_MN"] = is_mn
        adata.uns["is_MN_source"] = (f"{kind}:{desc}:spec[{col}=={sorted(pos)}] "
                                     f"overlap={overlap:.4f} key={keyname}")
    elif "is_MN" in aligned.columns:
        is_mn = np.asarray(aligned["is_MN"].fillna(False).to_numpy(), dtype=bool)
        adata.obs["is_MN"] = is_mn
        adata.uns["is_MN_source"] = (f"{kind}:{desc}:source is_MN bool (no '{col}') "
                                     f"overlap={overlap:.4f} key={keyname}")
    else:
        adata.uns["is_MN_source"] = (f"{kind}:{desc} joined but neither '{col}' nor 'is_MN' "
                                     f"present -> is_MN=False (overlap={overlap:.4f})")
        print(f"  [MN] {label}: WARNING joined but no MN label column found -> is_MN=False", flush=True)
        return adata

    n_mn = int(adata.obs["is_MN"].sum())
    print(f"  [MN] {label}: JOINED via '{keyname}' overlap={overlap:.4f}; "
          f"is_MN True={n_mn}/{n}; +{len(add_cols)} MN_/MNTDP_ cols  ({kind})", flush=True)
    print(f"  [MN] {label}: NOTE row-count provenance PASSED but is necessary-not-"
          f"sufficient (a same-count wrong seg is not caught) -- still require Marcel's "
          f"WRITTEN confirmation the annotation was built on _final (not _gauss) before "
          f"trusting MN positioning.", flush=True)
    return adata


if __name__ == "__main__":
    # Degraded-path self-check on the prototype label (no _final CSVs today -> the
    # old-seg h5ad is found but overlap ~0 -> DEGRADE). Requires an AnnData; only
    # exercised via the smoke test, so here we just print the config.
    print("MN_LABEL_SPEC   :", MN_LABEL_SPEC)
    print("MIN_OVERLAP_FRAC:", MIN_OVERLAP_FRAC)
    print("CELLANNOT_DIR   :", CELLANNOT_DIR)
    print("prototype exists:", (CELLANNOT_DIR / 'clustered_SD03522_BG_res0.5.h5ad').exists())
