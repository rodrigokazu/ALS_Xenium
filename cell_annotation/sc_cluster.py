#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | cell_annotation/sc_cluster.py
#
# Clustering that runs before any language model sees the data. cell-annotator labels
# clusters, it does not make them, so this produces the clustered log-normalised object it
# consumes.
#
# QC is tuned for a targeted panel and not for whole-transcriptome data. Floors are ten
# transcripts and five genes per cell with a light cell area guard. There is no gene filtering
# and no highly-variable-gene selection, because with 480 deliberately chosen probes every
# gene is informative and dropping any of them throws away panel design. Normalisation uses
# the median library size rather than a fixed 1e4 target, since panel counts run around 50 to
# 250 and forcing 1e4 would inflate them by two orders of magnitude.
#
# Raw counts are snapshotted to layers['counts'] before any transform.
#
# One bug in here is worth remembering because the failure was so quiet. obs['qc_flag'] turns
# out to be uniformly False across every cell, so the original gate on qc_flag == 'pass' kept
# zero cells and the job crashed. The gate now accepts either strings or booleans and skips
# itself entirely if it would retain under one percent, leaving QC resting on the explicit
# floors. A flag column that carries no signal is worse than no flag column.
#
# Smoke tested on SD03522_BG: 122,898 cells in, 111,168 kept, ten Leiden clusters, and the
# is_MN cells landed in the bilateral ventral horn clusters exactly where they should.
# ========================================================================================

"""
sc_cluster.py

First-pass cell-typing pipeline for the ALS spinal-cord Xenium concat object.

QC -> normalize (median library size) -> PCA (on scaled copy) -> neighbors -> Leiden.
cell-annotator does NOT cluster; it labels the clusters produced here, so this script
runs FIRST and hands off a clustered, log-normalized .h5ad.

Implements the QC + CLUSTER designs:
  - raw counts snapshotted to layers['counts'] BEFORE any transform
  - QC = qc_flag=='pass' (reverse-engineered + reported) AND explicit targeted-panel
    floors (transcript_counts>=10, n_genes_by_counts>=5) + light cell_area guard
  - NO gene filtering, NO HVG (curated 480-gene panel == the marker set)
  - normalize_total(target_sum=None) => MEDIAN library size (NOT 1e4), then log1p
  - adata.X stays log-normalized (cell-annotator reads X); scaling done on a COPY for PCA
  - optional Harmony on run_id if harmonypy trivially present, else documented caveat
  - Leiden flavor='igraph', n_iterations=2, directed=False, key_added='leiden'
  - prints cluster sizes + is_MN x cluster crosstab (MN caveat: MNs ~0.6% won't split out)

Usage:
  python sc_cluster.py --input <preQC.h5ad> --out <clustered.h5ad> [--sample SD_CODE] [--resolution 1.0]

Same code path for the one-sample fast test (--sample) and the full 1.88M-cell run.
"""

import argparse
import os
import sys


def log(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


DEFAULT_INPUT = (
    "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/SC_MNcorrected_h5ads/"
    "ALS_SCXenium_MNcorrected_gausss_concatenated_offset_allsamples_preQC_20260706.h5ad"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT, help="Input pre-QC .h5ad (X = raw CSR counts).")
    p.add_argument("--out", required=True, help="Output clustered .h5ad path.")
    p.add_argument("--sample", default=None,
                   help="Optional: subset to one sample (obs['sample']==SAMPLE) for a fast smoke test.")
    p.add_argument("--resolution", type=float, default=0.5,
                   help="Leiden resolution. Default 0.5 targets genuinely COARSE cell TYPES "
                        "(low-teens cluster count on a ~480-gene CNS panel with ~10-15 major "
                        "types). Higher values (e.g. 1.0) over-split into subtypes/states and "
                        "cost one extra LLM call PER extra cluster in sc_cell_annotate.py. "
                        "Pick empirically from the one-sample test, then reuse for the full run.")
    p.add_argument("--min-counts", type=int, default=10,
                   help="Floor on transcript_counts (panel transcripts). Default 10.")
    p.add_argument("--min-genes", type=int, default=5,
                   help="Floor on n_genes_by_counts. Default 5.")
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--harmony", action="store_true",
                   help="Attempt Harmony on run_id if harmonypy imports; else skip with a caveat.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    # Import-light: scanpy import is heavy, so do it inside main after arg parsing.
    log("[import] loading scanpy / anndata / numpy / pandas ...")
    import numpy as np
    import pandas as pd
    import scanpy as sc

    sc.settings.verbosity = 1
    log(f"[env] scanpy {sc.__version__}")

    # ------------------------------------------------------------------ LOAD
    if args.sample is not None:
        # Fast one-sample smoke test: read BACKED so only the sample's cells are
        # materialised into RAM. A plain read_h5ad would pull all 1.88M cells into
        # memory first and defeat the point of the smoke test.
        log(f"[load] reading {args.input} (backed='r' for one-sample subset)")
        adata = sc.read_h5ad(args.input, backed="r")
        if "sample" not in adata.obs:
            sys.exit("FATAL: obs has no 'sample' column; cannot subset with --sample.")
        mask = adata.obs["sample"].astype(str) == str(args.sample)
        n = int(mask.sum())
        if n == 0:
            avail = ", ".join(map(str, adata.obs["sample"].astype(str).unique()[:20]))
            sys.exit(f"FATAL: no cells for sample '{args.sample}'. Available (first 20): {avail}")
        adata = adata[mask].to_memory()
        log(f"[subset] --sample {args.sample} -> {adata.n_obs:,} cells (fast test path; "
            "backed read materialised only this sample). NOTE: one sample = one status + one "
            "spinal level, so it CANNOT exercise full cell-type diversity or batch handling; "
            "judge annotation QUALITY only from the full run.")
    else:
        log(f"[load] reading {args.input}")
        adata = sc.read_h5ad(args.input)
    log(f"[load] {adata.n_obs:,} cells x {adata.n_vars} genes")

    # -------------------------------------------------- SNAPSHOT RAW COUNTS
    # Do this BEFORE any transform so counts are recoverable for is_MN cross-check
    # and marker sanity.
    adata.layers["counts"] = adata.X.copy()
    log("[layers] snapshotted raw counts -> layers['counts']")

    # QC metrics (n_genes_by_counts) computed off the raw counts.
    # calculate_qc_metrics OVERWRITES obs['total_counts'] with the sum over the 480 panel
    # genes; preserve any pre-existing (Xenium-provided) total_counts first so downstream
    # consumers can still recover the original value.
    if "total_counts" in adata.obs:
        adata.obs["total_counts_xenium"] = adata.obs["total_counts"].copy()
        log("[qc] stashed original total_counts -> obs['total_counts_xenium'] "
            "(calculate_qc_metrics will recompute total_counts from the 480 panel).")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # ------------------------------------------------------- INSPECT qc_flag
    n0 = adata.n_obs
    if "qc_flag" in adata.obs:
        log("[qc_flag] value_counts:")
        log(adata.obs["qc_flag"].value_counts(dropna=False).to_string())
        if "transcript_counts" in adata.obs:
            try:
                tbin = pd.cut(
                    adata.obs["transcript_counts"].astype(float),
                    bins=[-0.1, 5, 10, 20, 50, 100, 250, 1e9],
                    labels=["0-5", "5-10", "10-20", "20-50", "50-100", "100-250", "250+"],
                )
                log("[qc_flag] crosstab qc_flag vs binned transcript_counts (reverse-engineer rule):")
                log(pd.crosstab(adata.obs["qc_flag"], tbin).to_string())
            except Exception as e:
                log(f"[qc_flag] crosstab skipped: {e}")
    else:
        log("[qc_flag] WARNING: no qc_flag column present; relying on explicit floors only.")

    # --------------------------------------------------------------- QC FILTER
    # Authoritative baseline (qc_flag=='pass') AND explicit reproducible floors.
    mask = np.ones(adata.n_obs, dtype=bool)

    if "qc_flag" in adata.obs:
        col = adata.obs["qc_flag"]
        vals = col.astype(str).str.lower()
        uniq = set(vals.unique())
        if (vals == "pass").any():
            pass_mask = (vals == "pass").values          # string pass/fail semantics
        elif col.dtype == bool or uniq <= {"true", "false", "nan"}:
            # boolean flag: convention here is True = flagged problematic -> keep the un-flagged (False).
            pass_mask = ~col.astype(str).str.lower().eq("true").values
        else:
            pass_mask = None
        kept = 0 if pass_mask is None else int(pass_mask.sum())
        # Guard: in this object qc_flag is uniformly False (no per-cell signal) -> a naive
        # =='pass' gate zeroed the data. If the gate is uninformative (keeps <1% or is
        # unrecognised), SKIP it and rely on the explicit reproducible floors below.
        if pass_mask is None or kept < 0.01 * n0:
            log(f"[qc] qc_flag uninformative/unrecognised (values={sorted(uniq)}, would keep "
                f"{kept:,}/{n0:,}) -> SKIPPING qc_flag gate; using explicit floors only.")
        else:
            log(f"[qc] qc_flag pass/keep: {kept:,}/{n0:,}")
            mask &= pass_mask

    if "transcript_counts" in adata.obs:
        tc_mask = adata.obs["transcript_counts"].astype(float).values >= args.min_counts
        log(f"[qc] transcript_counts>={args.min_counts}: {int(tc_mask.sum()):,}/{n0:,}")
        mask &= tc_mask
    else:
        log("[qc] WARNING: no transcript_counts column; falling back to total_counts for min_counts floor.")
        tc_mask = adata.obs["total_counts"].astype(float).values >= args.min_counts
        mask &= tc_mask

    ng_mask = adata.obs["n_genes_by_counts"].astype(float).values >= args.min_genes
    log(f"[qc] n_genes_by_counts>={args.min_genes}: {int(ng_mask.sum()):,}/{n0:,}")
    mask &= ng_mask

    # Light segmentation-artifact guard: drop cell_area outside 0.5-99.5 pct.
    if "cell_area" in adata.obs:
        ca = adata.obs["cell_area"].astype(float).values
        lo, hi = np.nanpercentile(ca, [0.5, 99.5])
        area_mask = (ca >= lo) & (ca <= hi)
        log(f"[qc] cell_area in [{lo:.1f}, {hi:.1f}] (0.5-99.5 pct): {int(area_mask.sum()):,}/{n0:,}")
        mask &= area_mask
    else:
        log("[qc] no cell_area column; skipping segmentation-artifact guard.")

    adata = adata[mask].copy()
    log(f"[qc] cells: {n0:,} -> {adata.n_obs:,} after QC "
        f"({100.0 * adata.n_obs / max(n0, 1):.1f}% kept)")

    # NOTE: genes are NEVER filtered — every gene in a curated 480-panel is a wanted marker.
    log(f"[qc] genes retained (NO gene filtering): {adata.n_vars}")

    # --------------------------------------------------------------- NORMALIZE
    # target_sum=None => normalize to MEDIAN library size (NOT 1e4). Median panel
    # counts are only ~50-250; 1e4 would inflate a 480-gene panel ~40-200x.
    sc.pp.normalize_total(adata, target_sum=None)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()
    log("[norm] normalize_total(target_sum=None => median) + log1p; layers['lognorm'] stored; "
        "adata.X stays log-normalized for cell-annotator.")

    # HVG: SKIP. All 480 curated panel genes are used.
    log("[hvg] SKIPPED — using all 480 curated panel genes.")

    # ------------------------------------------------------- SCALE (on a copy)
    # Scale for PCA on a COPY so adata.X remains log-normalized.
    adata_s = adata.copy()
    sc.pp.scale(adata_s, max_value=10)
    log("[scale] scaled copy (zero-center, max_value=10) for PCA; adata.X untouched.")

    # ----------------------------------------------------------------- PCA
    sc.tl.pca(adata_s, n_comps=args.n_pcs, svd_solver="arpack", random_state=args.seed)
    adata.obsm["X_pca"] = adata_s.obsm["X_pca"]
    if "PCs" in adata_s.varm:
        adata.varm["PCs"] = adata_s.varm["PCs"]
    log(f"[pca] {args.n_pcs} comps (arpack) -> obsm['X_pca']")

    # --------------------------------------------------------------- HARMONY?
    use_rep = "X_pca"
    if args.harmony:
        try:
            import harmonypy  # noqa: F401
            if "run_id" in adata.obs:
                sc.external.pp.harmony_integrate(adata_s, key="run_id")
                adata.obsm["X_pca_harmony"] = adata_s.obsm["X_pca_harmony"]
                use_rep = "X_pca_harmony"
                log("[harmony] harmony_integrate on run_id -> obsm['X_pca_harmony'] (used for "
                    "neighbors). NOTE: this corrects the TECHNICAL run batch only; it does NOT "
                    "and should not remove the biological spinal-level x disease structure.")
            else:
                log("[harmony] no run_id column; skipping technical batch correction.")
        except Exception as e:
            log(f"[harmony] harmonypy unavailable/failed ({e}); skipping TECHNICAL run-batch "
                "correction (run_id).")
    else:
        log("[harmony] not requested; skipping TECHNICAL run-batch correction (run_id).")
    # Two distinct issues, not to be conflated:
    #   (1) technical batch = run_id -> optionally Harmony-corrected above.
    #   (2) spinal-level x disease (controls cervical / most ALS lumbar) = BIOLOGICAL confound,
    #       intentionally NOT corrected here (Harmony on run_id does not target it) and a hard
    #       limit on any state/disease interpretation downstream. Coarse cell TYPE labels only.
    log("[harmony] CAVEAT: spinal-level x disease is a BIOLOGICAL confound, NOT addressed by "
        "Harmony-on-run_id; coarse cell TYPE labels are tolerable, state/disease splits are NOT.")

    del adata_s

    # ------------------------------------------------------------- NEIGHBORS
    log(f"[neighbors] building kNN (n_neighbors={args.n_neighbors}, n_pcs={args.n_pcs}, "
        f"use_rep={use_rep}) -- BOTTLENECK at scale ...")
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs,
                    use_rep=use_rep, random_state=args.seed)
    log("[neighbors] done.")

    # --------------------------------------------------------------- LEIDEN
    log(f"[leiden] resolution={args.resolution} (flavor='igraph', n_iterations=2, directed=False)")
    try:
        sc.tl.leiden(adata, resolution=args.resolution, flavor="igraph",
                     n_iterations=2, directed=False, random_state=args.seed, key_added="leiden")
    except (TypeError, ValueError):
        # Older scanpy without flavor='igraph' support -> fall back to leidenalg.
        # Catch ValueError too: some versions reject an unknown flavor value with ValueError
        # rather than TypeError, which would otherwise crash instead of falling back.
        log("[leiden] flavor='igraph' unsupported in this scanpy; falling back to leidenalg.")
        sc.tl.leiden(adata, resolution=args.resolution, random_state=args.seed, key_added="leiden")

    # ------------------------------------------------------------ REPORTING
    sizes = adata.obs["leiden"].value_counts().sort_index()
    log(f"[leiden] {sizes.shape[0]} clusters. Cluster sizes:")
    log(sizes.to_string())

    if "is_MN" in adata.obs:
        log("[MN caveat] is_MN x leiden crosstab (MNs ~0.6% expected to smear into the dominant "
            "neuron cluster, NOT form their own):")
        ct = pd.crosstab(adata.obs["leiden"], adata.obs["is_MN"])
        log(ct.to_string())
    else:
        log("[MN caveat] no is_MN column present; skipping MN crosstab.")

    # -------------------------------------------------------------- PROVENANCE
    adata.uns["cluster_provenance"] = {
        "script": "sc_cluster.py",
        "scanpy_version": sc.__version__,
        "input": args.input,
        "sample_subset": args.sample if args.sample is not None else "ALL",
        "n_cells_final": int(adata.n_obs),
        "resolution": float(args.resolution),
        "min_counts_on": "transcript_counts",
        "min_counts": int(args.min_counts),
        "min_genes": int(args.min_genes),
        "normalize_target_sum": "None (median library size)",
        "hvg": "none (all 480 panel genes)",
        "n_pcs": int(args.n_pcs),
        "n_neighbors": int(args.n_neighbors),
        "use_rep": use_rep,
        "leiden_flavor": "igraph/n_iterations=2/directed=False",
        "seed": int(args.seed),
        "total_counts_note": ("obs['total_counts'] recomputed from the 480 panel by "
                              "calculate_qc_metrics; original preserved as "
                              "obs['total_counts_xenium'] when present."),
    }

    # ------------------------------------------------------------------- SAVE
    # Confirm invariants before writing.
    assert "counts" in adata.layers, "layers['counts'] missing!"
    assert "lognorm" in adata.layers, "layers['lognorm'] missing!"
    assert "leiden" in adata.obs, "obs['leiden'] missing!"
    # Refuse to overwrite the irreplaceable pre-QC source object.
    if os.path.abspath(args.out) == os.path.abspath(args.input):
        sys.exit("FATAL: --out equals --input; refusing to overwrite the pre-QC source object.")
    if os.path.abspath(args.out) == os.path.abspath(DEFAULT_INPUT):
        sys.exit("FATAL: --out points at the canonical pre-QC master (DEFAULT_INPUT); "
                 "refusing to clobber the 1.88M-cell source. Choose a different --out.")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    adata.write_h5ad(args.out)
    log(f"[save] wrote clustered .h5ad -> {args.out}")
    log("[done] X=log-normalized, layers['counts'/'lognorm'] present, obs['leiden'] set. "
        "Hand off to sc_cell_annotate.py.")


if __name__ == "__main__":
    main()
