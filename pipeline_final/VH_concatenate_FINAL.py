#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/VH_concatenate_FINAL.py
#
# Builds the cohort object. Everything downstream reads what this produces, so it is the one
# script where failing loudly matters more than finishing.
#
# It walks Marcel's _final delivery, keeps the 20 real samples, loads each as an AnnData,
# joins metadata and motor-neuron labels, applies the grid offset, and writes both a combined
# pre-QC h5ad and one h5ad per sample. Discovery, dedup, loading and offset all come from
# final_config, so this file holds the orchestration and the gates rather than the rules.
#
# The gates raise, they do not warn: the two required controls SD00614 and SD02818 must be
# present, the sample count must be 20, the donor split must be 10 control / 6 sporadic / 4
# C9, and the panel must be 480 genes. That looks paranoid until you remember the _gausss
# generation ran to completion twice on a quietly wrong expression matrix.
#
# Pre-QC by design. No cells are filtered here beyond what loading implies. QC decisions
# belong downstream where they can be inspected and reversed.
#
# Two bugs are fixed in here and both are easy to reintroduce. Boolean obs columns must be
# tested with dtype == bool before the numeric branch, because pandas reports bool as numeric
# and the sanitiser will happily stringify is_MN into the truthy strings "True" and "False",
# at which point every cell is a motor neuron. And the h5ad writer chokes on object/bool
# columns carrying NaN, so obs is sanitised to strings before the write.
#
# Expect roughly 1.85 million cells across 20 samples with join_overlap 1.0 on every one. The
# 2026-07-22 run gave 1,854,568.
# ========================================================================================

"""
VH_concatenate_FINAL.py -- concatenate Marcel's "_final" segmentation
(Ranger_procd_mw_final, delivered 2026-07-19) into ONE PRE-QC AnnData, and write
one PRE-QC h5ad per sample. This is the _FINAL re-run of the proven MN-corrected
_gausss concat pipeline (build_mn_notebook.py) -- SAME gates, prints, offset, and
h5py sanitize; only the raw-data deltas (10x MTX dir, integer-id 1:1 join, 480-GE
subset) change, and those are ALL handled inside final_config.load_final_sample.

*** STAGING JOB. This script RUNS the concat when submitted, but nothing is
submitted until Marcel's _final MN annotation lands. It DEGRADES GRACEFULLY today
(no _final MN CSVs exist yet -> is_MN=False, annotation pending). ***

Design (vs the _gausss build_mn_notebook.py, which it ports 1:1 in logic):
  * discovery/dedup/loading/offset  -> final_config  (discover_samples(strict=True),
    load_final_sample, apply_grid_offset). Deltas D1-D4 live there.
  * is_MN + MN_*/MNTDP_* join        -> mn_annotation_join.join_is_MN  (ready for
    Marcel's _final annotation in either accepted form; degrades to is_MN=False now).
  * status parsing + STATUS_OVERRIDES, demographics join, the HARD GATES, the h5py
    bool-first sanitize, and the summary/MN reports are ported here verbatim.

HARD GATES (fail loud, per the mission):
  kept == 20  |  donor split control10/sporadic6/c9 4  |  panel == 480 GE, identical
  across samples, no control-probe leak  |  is_MN stays a real bool through the
  sanitize + write + READBACK (the combined file is re-opened and is_MN dtype asserted
  == bool -- catches the classic is_numeric_dtype-stringifies-bool regression).

Outputs (see final_config):
  per-sample : H5AD_DIR/<LABEL>__MNcorrected_FINAL.h5ad   (single-slide: spatial==
               spatial_orig == true microns; uns['spatial_offset']=None)
  combined   : COMBINED_H5AD  (obsm['spatial']=grid-offset for cohort PLOTTING ONLY;
               obsm['spatial_orig']=true microns for graphs; uns['spatial_offset_table'])

Kernel/env: pertpy_env. The runner exports PYTHONNOUSERSITE=1; the sys.path strip
below is belt-and-braces against a stale ~/.local anndata.
"""
import os
import sys

# Drop user-site paths so the env's own anndata/scanpy win (pertpy_env import guard).
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]
# Import the shared _FINAL foundation modules from the staged code dir.
_CODE_DIR = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code"
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import re
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import final_config as cfg
import mn_annotation_join

# ----------------------------------------------------------------------
# RUN-TIME KNOBS
# ----------------------------------------------------------------------
VERBOSE = True

# Provenance guard for the MN join (N1 launch gate). Default True: at launch a source
# that joins with high integer-id overlap but whose ROW COUNT does not match the _final
# per-sample cell count (within ROW_COUNT_TOL_FRAC) is almost certainly a DIFFERENT
# segmentation (a spoofed integer-id overlap) and HALTS the build. Set the env var
# MN_REQUIRE_FINAL_PROVENANCE=0 ONLY to accept an annotation you have human-confirmed is
# on _final despite a row-count difference (it then loud-degrades to is_MN=False). Today
# (no _final annotation exists) the guard never fires: every sample degrades cleanly.
REQUIRE_FINAL_PROVENANCE = os.environ.get("MN_REQUIRE_FINAL_PROVENANCE", "1") not in (
    "0", "false", "False", "no", "NO")


def dprint(msg):
    if VERBOSE:
        print(msg, flush=True)


def _minmax2(xy):
    return (float(np.nanmin(xy[:, 0])), float(np.nanmax(xy[:, 0]))), \
           (float(np.nanmin(xy[:, 1])), float(np.nanmax(xy[:, 1])))


# ----------------------------------------------------------------------
# STATUS PARSING  (ported verbatim from build_mn_notebook.py; uses cfg helpers)
# ----------------------------------------------------------------------
def parse_status_mapping_from_csv(csv_path):
    """sd_code -> {control,c9,sporadic} from status.csv (explicit cols or section headers)."""
    csv_path = cfg.Path(csv_path)
    if not csv_path.exists():
        dprint(f"WARNING: status CSV not found: {csv_path}. obs['status'] will be 'unknown'.")
        return {}
    df = pd.read_csv(csv_path)
    cols_lower = {c: str(c).strip().lower() for c in df.columns}
    status_col = next((c for c, lc in cols_lower.items() if lc in {"status", "group", "genotype"}), None)
    sample_col = next((c for c, lc in cols_lower.items()
                       if lc in {"sample", "sample_id", "id", "case number", "case_number"}), None)
    mapping = {}
    if status_col is not None and sample_col is not None:
        dprint(f"[status.csv] explicit columns: sample='{sample_col}', status='{status_col}'")
        for _, row in df.iterrows():
            sd = cfg.extract_sd_code_anywhere(row.get(sample_col))
            if not sd:
                continue
            st = str(row.get(status_col)).strip().lower()
            if not st or st == "nan":
                continue
            if "control" in st:    mapping[sd] = "control"
            elif "c9" in st:       mapping[sd] = "c9"
            elif "sporadic" in st: mapping[sd] = "sporadic"
            else:                  mapping[sd] = st
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
        sd = cfg.extract_sd_code_anywhere(up)
        if sd and current is not None:
            mapping[sd] = current
    return mapping


# ----------------------------------------------------------------------
# H5PY SANITIZE  (bool-first -- is_MN MUST NOT be stringified; see gotcha)
# ----------------------------------------------------------------------
def sanitize_obs_for_h5ad(adata, prefix="[sanitize]"):
    """Coerce write-unsafe obs cols to str for the h5py write.

    ORDER IS LOAD-BEARING. bool (is_MN) is tested BEFORE is_numeric_dtype, because
    pd.api.types.is_numeric_dtype(bool_series) is True -- the older
    VH_concatenate_allsamples.py ordering would stringify is_MN to 'True'/'False'
    (both truthy -> silently breaks every is_MN mask downstream). Categoricals and
    numeric cols (incl. numeric MN_/MNTDP_ covariates, NaN-safe as float) are left
    untouched; object/string cols (incl. object MN_/MNTDP_ covariates with NaN from a
    partial join, N4) are coerced with NaN -> 'NA'.
    """
    _san = []
    for col in list(adata.obs.columns):
        s = adata.obs[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            continue
        if s.dtype == bool:                       # is_MN -- keep the real bool
            continue
        if pd.api.types.is_numeric_dtype(s):      # numeric (incl. NaN-safe float)
            continue
        adata.obs[col] = s.astype(object).where(s.notna(), "NA").astype(str)
        _san.append(col)
    if _san:
        dprint(f"{prefix} coerced {len(_san)} obs cols to str: {_san}")
    return _san


# ======================================================================
# MAIN
# ======================================================================
def main():
    cfg.H5AD_DIR.mkdir(parents=True, exist_ok=True)

    # --- status + demographics -----------------------------------------------
    status_map = parse_status_mapping_from_csv(cfg.STATUS_CSV)
    dprint(f"Loaded {len(status_map)} status mappings from {cfg.STATUS_CSV.name}")
    for sd, ns in cfg.STATUS_OVERRIDES.items():
        dprint(f"[STATUS_OVERRIDE] {sd}: '{status_map.get(sd, 'unknown')}' -> '{ns}'")

    sample_meta = None
    if cfg.METADATA_CSV.exists():
        _m = pd.read_csv(cfg.METADATA_CSV)
        if "sd_code" in _m.columns:
            _m = _m.drop_duplicates(subset="sd_code", keep="last")
            sample_meta = _m.set_index("sd_code")
            dprint(f"Loaded sample metadata: {sample_meta.shape[0]} rows x "
                   f"{sample_meta.shape[1]} cols from {cfg.METADATA_CSV.name}")
        else:
            dprint(f"NOTE: {cfg.METADATA_CSV.name} has no 'sd_code' column; demographics skipped")
    else:
        dprint(f"NOTE: {cfg.METADATA_CSV.name} not found -- demographics absent; status still set")

    # --- discover (strict: HARD-halt on kept!=20 / dup sd_code / missing reseg) ---
    # discover_samples(verbose=True) prints the full DISCOVERY plan (every skipped dir +
    # reason, every kept dir -> label). strict=True turns the three cohort-integrity
    # conditions into a ValueError (M1) so a future _final dir change cannot silently
    # poison the cohort.
    kept_list = cfg.discover_samples(verbose=True, strict=True)

    # --- run header ----------------------------------------------------------
    n_total = len(kept_list)
    print("=" * 72)
    print("RUN HEADER  |  ALS SC Xenium MN-corrected (_final segmentation) concatenation")
    print("=" * 72)
    print(f"  ranger_root       : {cfg.RANGER_FINAL}")
    print(f"  out h5ad dir      : {cfg.H5AD_DIR}")
    print(f"  combined h5ad     : {cfg.COMBINED_H5AD.name}")
    print(f"  seg_pipeline      : {cfg.SEG_PIPELINE}")
    print(f"  kept sample dirs  : {n_total}   (expect {cfg.EXPECTED_SAMPLE_COUNT})")
    print(f"  expected split    : control {cfg.EXPECTED_DONOR_SPLIT['control']} / "
          f"sporadic {cfg.EXPECTED_DONOR_SPLIT['sporadic']} / c9 {cfg.EXPECTED_DONOR_SPLIT['c9']}")
    print(f"  expected panel    : {cfg.EXPECTED_PANEL_SIZE} Gene Expression genes")
    print(f"  spatial offset    : {cfg.SPATIAL_OFFSET:g} um, grid {cfg.SPATIAL_GRID_COLS} cols "
          f"(plotting only)")
    print(f"  MN provenance gate: require_final_provenance={REQUIRE_FINAL_PROVENANCE}  "
          f"(env MN_REQUIRE_FINAL_PROVENANCE)")
    print("=" * 72, flush=True)

    # --- main loop -----------------------------------------------------------
    adatas = []
    kept, skipped = 0, 0
    loaded_sd_codes = []
    offset_records = {}      # run_id -> offset params (survives merge='same')
    mn_prov_records = {}     # label   -> MN join provenance (survives merge='same')
    persample_written = []

    for i_dir, (label, sample_dir) in enumerate(kept_list, start=1):
        dprint(f"\n[ {i_dir:>2}/{n_total} ] === {sample_dir.name}  ->  {label} ===")

        # LOAD is the only step with real per-sample data variability: collect its
        # failures (so ALL are visible) and let the kept==20 gate catch the shortfall.
        try:
            adata = cfg.load_final_sample(label, sample_dir, verbose=VERBOSE)
        except Exception as e:
            skipped += 1
            dprint(f"[ {i_dir:>2}/{n_total} ] [FAIL load] {sample_dir.name}: {type(e).__name__}: {e}")
            continue

        # status (load_final_sample sets sample/run_id/sd_code/seg_pipeline, NOT status)
        sd_code = str(adata.obs["sd_code"].iloc[0]) if "sd_code" in adata.obs.columns else \
            (cfg.extract_sd_code_anywhere(label) or "")
        status = status_map.get(sd_code, "unknown") if sd_code else "unknown"
        if sd_code in cfg.STATUS_OVERRIDES:
            dprint(f"[STATUS_OVERRIDE] {sd_code}: '{status}' -> '{cfg.STATUS_OVERRIDES[sd_code]}'")
            status = cfg.STATUS_OVERRIDES[sd_code]
        adata.obs["status"] = status

        # MN join. Degrades to is_MN=False today (no _final annotation). A PROVENANCE
        # MISMATCH at launch (RuntimeError) is a COHORT-level correctness halt and is
        # intentionally NOT swallowed -- it must stop the build with its own message,
        # not be masked as a per-sample [FAIL]/count shortfall.
        mn_annotation_join.join_is_MN(
            adata, label, run_dir=sample_dir, verbose=VERBOSE,
            require_final_provenance=REQUIRE_FINAL_PROVENANCE)

        # spatial grid offset (0-based rank among KEPT; failed loads never consume a slot)
        run_id = str(adata.obs["run_id"].iloc[0]) if "run_id" in adata.obs.columns else sample_dir.name
        if "spatial" in adata.obsm:
            cfg.apply_grid_offset(adata, kept)
            rec = dict(adata.uns.get("spatial_offset", {}))
            rec.update({"sample": label, "sd_code": sd_code, "status": status})
            offset_records[run_id] = rec
        else:
            dprint(f"[offset] {label}: NOTE no obsm['spatial']; skipping offset")
            adata.uns["spatial_offset"] = None

        # per-donor MN provenance audit (survives concat merge='same')
        mn_prov_records[label] = {
            "sd_code": sd_code, "status": status,
            "is_MN_source": str(adata.uns.get("is_MN_source", "")),
            "is_MN_overlap_frac": float(adata.uns.get("is_MN_overlap_frac", 0.0)),
            "is_MN_join_key": adata.uns.get("is_MN_join_key", None),
            "is_MN_provenance": str(adata.uns.get("is_MN_provenance", "")),
            "is_MN_source_nrows": int(adata.uns.get("is_MN_source_nrows", 0)),
            "is_MN_final_ncells": adata.uns.get("is_MN_final_ncells", None),
            "n_is_MN": int(adata.obs["is_MN"].sum()),
            "n_mn_cols": len(adata.uns.get("mn_columns_added", [])),
        }

        dprint(f"[ {i_dir:>2}/{n_total} ] [OK] {label} (dir={sample_dir.name}): "
               f"n_obs={adata.n_obs}, n_vars={adata.n_vars}, sd_code={sd_code}, status={status}, "
               f"is_MN={int(adata.obs['is_MN'].sum())}, MN_prov={adata.uns.get('is_MN_provenance')} "
               f"| running kept={kept + 1}/{cfg.EXPECTED_SAMPLE_COUNT}")

        # --- write PRE-QC per-sample h5ad (single-slide: spatial == true microns) ---
        aw = adata.copy()
        if "spatial_orig" in aw.obsm:
            aw.obsm["spatial"] = aw.obsm["spatial_orig"].copy()
            aw.uns["spatial_offset"] = None
            aw.uns["spatial_note"] = ("per-sample file: obsm['spatial']==spatial_orig (true "
                                      "microns); grid offset applied only in the combined object")
        sanitize_obs_for_h5ad(aw, prefix=f"[sanitize {label}]")
        ps_path = cfg.persample_h5ad(label)
        aw.write_h5ad(str(ps_path), compression="gzip")
        persample_written.append(ps_path.name)
        dprint(f"[WRITE] per-sample -> {ps_path.name}")
        del aw

        adatas.append(adata)
        loaded_sd_codes.append(sd_code)
        kept += 1

    print("\n" + "-" * 72)
    print(f"MAIN LOOP DONE  |  loaded {kept}  |  skipped/failed {skipped}  |  of {n_total} kept dirs")
    print("-" * 72, flush=True)

    if not adatas:
        raise RuntimeError("No samples loaded; nothing to concatenate.")

    # ================================================================
    # HARD GATES
    # ================================================================
    # (1) required reseg samples present
    missing = cfg.REQUIRED_SD_CODES - set(loaded_sd_codes)
    if missing:
        raise RuntimeError(f"Required reseg samples missing: {sorted(missing)}; "
                           f"loaded={sorted(set(loaded_sd_codes))}")
    dprint(f"[gate] required reseg samples present: {sorted(cfg.REQUIRED_SD_CODES)}")

    # (2) exact sample count
    if kept != cfg.EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(f"Expected {cfg.EXPECTED_SAMPLE_COUNT} samples, got {kept}. "
                           f"A sample was silently dropped -- inspect [FAIL load] above.")
    dprint(f"[gate] sample count OK: {kept}")

    # (3) sd_codes unique (count the raw list; set() would hide a collision)
    _dupes = sorted({c for c in loaded_sd_codes if c and loaded_sd_codes.count(c) > 1})
    if _dupes:
        raise RuntimeError(f"Duplicate sd_code across kept samples (superseded original not "
                           f"excluded, or label collision): {_dupes}. loaded={loaded_sd_codes}")
    dprint(f"[gate] sd_codes unique across {kept} kept samples")

    # (4) donor split (count can be 20 while a status is misassigned/'unknown')
    _donor_status = {a.uns["sample_label"]: str(a.obs["status"].iloc[0]) for a in adatas}
    _split = {}
    for st in _donor_status.values():
        _split[st] = _split.get(st, 0) + 1
    if _split != cfg.EXPECTED_DONOR_SPLIT:
        raise RuntimeError(f"Donor split {_split} != expected {cfg.EXPECTED_DONOR_SPLIT}. "
                           f"A status is misassigned or 'unknown' -- check status.csv parsing / "
                           f"STATUS_OVERRIDES / SAMPLE_LABEL_OVERRIDES. Per-donor: {_donor_status}")
    dprint(f"[gate] donor split OK: {_split}")

    # (5) panel identity across samples (join='outer' would silently zero-pad a mismatch)
    var_sets = [set(a.var_names) for a in adatas]
    common = set.intersection(*var_sets)
    union = set.union(*var_sets)
    dprint(f"[panel] per-sample n_vars: {[a.n_vars for a in adatas]}")
    if len(common) != len(union):
        raise RuntimeError(f"Gene panel MISMATCH: {len(union) - len(common)} gene(s) not shared by "
                           f"all samples: {sorted(union - common)[:15]}")
    dprint(f"[panel] OK: all {kept} samples share identical {len(common)}-gene panel")

    # (6) exact GE panel size
    if len(common) != cfg.EXPECTED_PANEL_SIZE:
        raise RuntimeError(f"Panel size {len(common)} != expected {cfg.EXPECTED_PANEL_SIZE} Gene "
                           f"Expression genes. Check the feature_types filter / that no control "
                           f"probes leaked in.")
    dprint(f"[panel] OK: panel size == {cfg.EXPECTED_PANEL_SIZE} Gene Expression genes")

    # (7) no control/blank probe leak
    _ctrl = re.compile(r"NegControl|BLANK|antisense|Deprecated|UnassignedCodeword|Intergenic",
                       re.IGNORECASE)
    _leaked = sorted(g for g in common if _ctrl.search(str(g)))
    if _leaked:
        raise RuntimeError(f"Control/blank probes leaked into the panel: {_leaked[:15]}")
    dprint("[panel] OK: no control/blank probe names present in the panel")

    # ================================================================
    # CONCATENATE + provenance + demographics + write
    # ================================================================
    dprint("\nConcatenating...")
    adata_all = sc.concat(adatas, axis=0, join="outer", merge="same", label=None, index_unique=None)
    adata_all.obs_names_make_unique()
    adata_all.obs["sample"] = adata_all.obs["sample"].astype("category")
    adata_all.obs["status"] = adata_all.obs["status"].astype("category")

    # per-sample audit trails that merge='same' strips from uns
    adata_all.uns["spatial_offset_table"] = offset_records
    adata_all.uns["mn_provenance_table"] = mn_prov_records
    _prov_states = sorted({r["is_MN_provenance"] for r in mn_prov_records.values()})
    adata_all.uns["concat_provenance"] = {
        "script": "VH_concatenate_FINAL.py",
        "ranger_root": str(cfg.RANGER_FINAL),
        "seg_pipeline": cfg.SEG_PIPELINE,
        "excluded_superseded": sorted(cfg.EXCLUDE_DIR_NAMES),
        "spatial_offset": float(cfg.SPATIAL_OFFSET),
        "spatial_grid_cols": int(cfg.SPATIAL_GRID_COLS),
        "coords": ("obsm['spatial']=grid-offset (plotting only); obsm['spatial_orig']=true "
                   "microns (use for graphs, slide_key='run_id')"),
        "qc": "PRE-QC (no cell/gene filtering)",
        "panel_size": int(len(common)),
        "sample_count": int(kept),
        "donor_split": _split,
        "mn_provenance_states": _prov_states,
        "mn_annotation_pending": bool(
            any(r["is_MN_source"].startswith("none") or "pending" in r["is_MN_source"]
                for r in mn_prov_records.values())),
        "require_final_provenance": bool(REQUIRE_FINAL_PROVENANCE),
    }

    # demographics join by sd_code (drop cols already in obs so a name clash can't abort)
    if sample_meta is not None:
        skip = {"sample_ranger_procd", "sd_code_raw", "disease", "run", "batch"}
        meta_cols = [c for c in sample_meta.columns if c not in skip]
        _clash = [c for c in meta_cols if c in adata_all.obs.columns]
        if _clash:
            dprint(f"[meta] NOTE dropping {len(_clash)} metadata col(s) already in obs "
                   f"(no clobber): {_clash}")
            meta_cols = [c for c in meta_cols if c not in adata_all.obs.columns]
        if meta_cols:
            adata_all.obs = adata_all.obs.join(sample_meta[meta_cols], on="sd_code", how="left")
            n_match = int(adata_all.obs[meta_cols[0]].notna().sum())
            dprint(f"Metadata join: {n_match}/{adata_all.n_obs} cells matched ({len(meta_cols)} cols)")
            _unmatched = sorted(adata_all.obs.loc[adata_all.obs[meta_cols[0]].isna(), "sd_code"]
                                .astype(str).unique())
            if _unmatched:
                dprint(f"  WARNING: unmatched sd_codes (no demographic row): {_unmatched}")

    # sanitize + write combined
    sanitize_obs_for_h5ad(adata_all, prefix="[sanitize combined]")
    dprint(f"\nCombined: n_obs={adata_all.n_obs}, n_vars={adata_all.n_vars}")
    dprint("Status counts (cells):\n" + str(adata_all.obs["status"].value_counts(dropna=False)))
    _ds = adata_all.obs[["sample", "status"]].drop_duplicates()
    dprint("Donors per status:\n" + str(_ds.groupby("status")["sample"].nunique()))
    dprint(f"\nWriting combined -> {cfg.COMBINED_H5AD}")
    adata_all.write_h5ad(str(cfg.COMBINED_H5AD), compression="gzip")
    dprint("Done writing combined.")

    # ================================================================
    # HARD GATE (8): is_MN survives sanitize + write + READBACK as a real bool
    # ================================================================
    # Re-open the combined object (backed='r' -> obs/var in memory, X stays on disk) and
    # assert is_MN round-tripped as bool. Catches the is_numeric_dtype-stringifies-bool
    # regression (bool -> 'True'/'False' both truthy) that a live write would not surface.
    _rb = ad.read_h5ad(str(cfg.COMBINED_H5AD), backed="r")
    try:
        if "is_MN" not in _rb.obs.columns:
            raise RuntimeError("[readback] is_MN missing from the written combined obs")
        _dt = _rb.obs["is_MN"].dtype
        if _dt != bool:
            raise RuntimeError(f"[readback] is_MN readback dtype={_dt} != bool -- the bool-first "
                               f"sanitize regressed and stringified is_MN (both 'True'/'False' are "
                               f"truthy, silently breaking every is_MN mask).")
        _n_mn_rb = int(np.asarray(_rb.obs["is_MN"]).sum())
    finally:
        try:
            _rb.file.close()
        except Exception:
            pass
    dprint(f"[gate] is_MN readback OK: dtype==bool, True={_n_mn_rb}/{adata_all.n_obs}")

    # ================================================================
    # FINAL SUMMARY TABLE (per-sample, PRE-QC)
    # ================================================================
    print("\n" + "=" * 72)
    print("FINAL SUMMARY  |  per-sample (PRE-QC)")
    print("=" * 72)
    _summ = []
    for a in adatas:
        _orig = a.obsm.get("spatial_orig")
        if _orig is not None:
            (_x0, _x1), (_y0, _y1) = _minmax2(np.asarray(_orig, dtype=float))
            _ext = f"x[{_x0:.0f},{_x1:.0f}] y[{_y0:.0f},{_y1:.0f}]"
        else:
            _ext = "no spatial"
        lab = a.uns["sample_label"]
        pr = mn_prov_records.get(lab, {})
        _src = str(a.uns.get("is_MN_source", ""))
        _summ.append({
            "sample": lab,
            "sd_code": str(a.obs["sd_code"].iloc[0]),
            "status": str(a.obs["status"].iloc[0]),
            "n_obs": a.n_obs,
            "n_vars": a.n_vars,
            "n_is_MN": int(a.obs["is_MN"].sum()),
            "MN_overlap": round(float(pr.get("is_MN_overlap_frac", 0.0)), 4),
            "MN_prov": str(pr.get("is_MN_provenance", "")),
            "join_overlap": round(float(a.uns.get("cell_join_overlap", float("nan"))), 4),
            "spatial_orig_um": _ext,
            "is_MN_source": (_src[:44] + "...") if len(_src) > 47 else _src,
        })
    _summ_df = pd.DataFrame(_summ).sort_values(["status", "sample"]).reset_index(drop=True)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(_summ_df.to_string(index=True))
    print("-" * 72)
    print(f"TOTAL cells : {adata_all.n_obs:,}   |   samples : {len(adatas)}   |   "
          f"panel : {adata_all.n_vars}")
    print("Donors per status :", dict(_summ_df.groupby("status")["sample"].nunique()))
    print("Per-sample h5ads written :", len(persample_written))
    print("=" * 72, flush=True)

    # ================================================================
    # MN VALIDATION REPORT + LAUNCH CHECKLIST
    # ================================================================
    print("\n" + "=" * 72)
    print("MN VALIDATION REPORT  |  is_MN join provenance per sample")
    print("=" * 72)
    _mn_rows = [{"sample": lab, **{k: v for k, v in rec.items()
                                   if k in ("status", "n_is_MN", "is_MN_overlap_frac",
                                            "is_MN_join_key", "is_MN_provenance",
                                            "is_MN_source_nrows", "is_MN_final_ncells")}}
                for lab, rec in mn_prov_records.items()]
    _mn_df = pd.DataFrame(_mn_rows).sort_values("sample").reset_index(drop=True)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(_mn_df.to_string(index=False))
    _any_confirmed = any(r["is_MN_provenance"] == "row-count-ok" for r in mn_prov_records.values())
    _total_mn = int(sum(r["n_is_MN"] for r in mn_prov_records.values()))
    print("-" * 72)
    print(f"is_MN provenance states across cohort : {_prov_states}")
    print(f"any sample with a confirmed _final join (row-count-ok) : {_any_confirmed}")
    print(f"total is_MN cells across cohort : {_total_mn}")
    if not _any_confirmed:
        print("\n>>> MN ANNOTATION PENDING (expected at staging).")
        print("    No _final MN annotation exists yet -> is_MN=False for all samples; the")
        print("    combined + per-sample objects are otherwise complete and correct.")
        print("    LAUNCH CHECKLIST when Marcel drops the _final annotation:")
        print("      1. Re-run this script (no code change needed).")
        print("      2. Confirm each sample logs is_MN join overlap ~1.0 and "
              "MN_prov='row-count-ok'.")
        print("      3. Confirm is_MN counts land ~150-320 per sample (prototype range).")
        print("      4. Get Marcel's WRITTEN confirmation the annotation was built on _final")
        print("         (not _gauss) -- the row-count guard is necessary-not-sufficient.")
    print("=" * 72, flush=True)

    return adata_all


if __name__ == "__main__":
    main()
