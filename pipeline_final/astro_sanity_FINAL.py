#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/astro_sanity_FINAL.py
#
# Asks one question: can we pull an astrocyte population out of this cohort by unsupervised
# clustering plus canonical marker scoring, with no label transfer and no reliance on the
# motor-neuron annotation? If yes, that population seeds a WDR49+ trajectory analysis in
# STREAM.
#
# The route is deliberately ordinary. Light per-cell QC, normalise and log, PCA, kNN, Leiden,
# then score seven lineage marker modules per cluster and call the astrocyte cluster by
# argmax. Then report the WDR49 signal inside it, checkpoint a lean clustered object, and
# write the astrocyte subset out.
#
# No batch integration, on purpose. Calling a cluster from its marker profile is robust to
# batch in a way that per-cell integration is not, and we would rather keep the raw structure
# visible than smooth it and wonder what we removed.
#
# --resume_from takes the clustered checkpoint and jumps straight to plotting. That flag
# exists because the plotting stage ran out of memory after the expensive clustering had
# already succeeded, and re-running the whole thing to redraw figures is a waste of a GPU
# night.
#
# Sanity check, not a result. It answers whether the population is separable, and nothing
# about disease.
# ========================================================================================

"""
astro_sanity_FINAL.py -- astrocyte-isolation SANITY CHECK on the MN-corrected
ALS spinal-cord Xenium _FINAL concat (Marcel's Ranger_procd_mw_final segmentation).

Question: can we isolate an astrocytic population by unsupervised clustering +
canonical marker scoring (NO label transfer, NOT is_MN-gated), so it can seed a
WDR49+ trajectory (STREAM)?

Pipeline: light per-cell QC -> normalise/log -> PCA -> kNN -> Leiden -> score
7 lineage marker modules per cluster -> argmax-call astrocyte cluster(s) -> report
WDR49 signal within them -> checkpoint the (lean) clustered object -> UMAP + t-SNE
plots (general, per-sample, batch-check) -> write the astrocyte subset for STREAM.

--resume_from <clustered_full.h5ad> skips the expensive clustering and jumps
straight to plotting (used after an OOM in the plotting stage).

No batch integration: cluster-level marker calling is integration-robust and we
want to see the raw structure first (per-sample UMAPs reveal any batch effect).

============================================================================
ADAPTED FROM the proven _gausss reference
  /home/rodrigok/SLURM_jobs/Spatial/VH_isolation/astro_sanity.py
All analysis logic is PRESERVED VERBATIM. The only _FINAL deltas (each flagged
inline with [FINAL]):
  F1  IMPORT the shared foundation (final_config) rather than re-deriving paths;
      --in_h5ad defaults to cfg.COMBINED_H5AD (the _FINAL combined preQC h5ad,
      X = RAW counts, 480 Gene-Expression panel) and --outdir to cfg.ASTRO_DIR
      (astro_isolation_FINAL). Both stay overridable; the runner passes them
      explicitly.
  F2  UMAP now uses init_pos="random" (documented gotcha: spectral init segfaults
      with NO traceback on large cell counts; the _gausss reference omitted this).
  F3  belt-and-braces ~/.local sys.path strip so a stale user-site anndata never
      shadows the env's anndata (the runner also exports PYTHONNOUSERSITE=1).

is_MN is NEVER used here (marker-based call only); if the concat builder joined
is_MN + MN_*/MNTDP_* into obs it simply FLOWS THROUGH unchanged into
clustered_full.h5ad + astro_subset.h5ad. WDR49 is on the 480 panel (single-gene
readout); GFAP is NOT on the panel (correctly absent from the astro marker set --
the `used` filter drops any marker not in var_names, so the panel is self-checking).

Kernel/env: run under pertpy_env. Runner MUST export PYTHONNOUSERSITE=1.
"""
import sys
# [FINAL F3] drop user-site paths so the env's own anndata/scanpy win.
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]
# [FINAL F1] make the shared foundation importable (final_config lives here).
sys.path.insert(0, "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code")

import json, argparse, gc
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math, traceback

import final_config as cfg  # [FINAL F1] single source of truth for _FINAL paths

sc.settings.verbosity = 2
sc.settings.n_jobs = 8

ap = argparse.ArgumentParser()
# [FINAL F1] defaults from the foundation; runner still passes them explicitly.
ap.add_argument("--in_h5ad", default=str(cfg.COMBINED_H5AD))
ap.add_argument("--outdir", default=str(cfg.ASTRO_DIR))
ap.add_argument("--resume_from", default=None, help="clustered_full.h5ad to skip clustering")
ap.add_argument("--min_counts", type=int, default=10)
ap.add_argument("--min_genes", type=int, default=5)
ap.add_argument("--resolution", type=float, default=1.0)
ap.add_argument("--umap_n", type=int, default=300000)
ap.add_argument("--tsne_n", type=int, default=150000)
args = ap.parse_args()

OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = str(OUT)
sc.settings.set_figure_params(dpi=150, dpi_save=200, frameon=False)
def log(m): print(f"[astro_sanity_FINAL] {m}", flush=True)

MARKERS = {
    "astro":  ["AQP4","AQP4-AS1","SLC1A2","GJA1","FGFR3","SOX9","GLIS3","PTPRZ1","TTYH1","GPC5"],
    "oligo":  ["MBP","MOBP","MOG","MAG","CLDN11","ERMN","OPALIN","ST18","CNDP1","UGT8","KLK6","PLP1"],
    "opc":    ["PDGFRA","CSPG4","OLIG1","OLIG2","SOX10"],
    "neuron": ["RBFOX3","SLC17A6","SLC17A7","GAD1","GAD2","STMN2","SNCG","SYNPR","NXPH1"],
    "micro":  ["P2RY12","P2RY13","C1QC","TYROBP","ITGAM","CD68","TREM2","FCER1G"],
    "endo":   ["PECAM1","VWF","CLDN5"],
    "vlmc":   ["DCN","COL5A2","FBLN1","POSTN"],
}
GOI = "WDR49"
SCORE_COLS = [f"score_{k}" for k in MARKERS]
CKPT = OUT / "clustered_full.h5ad"


# ======================================================================
# STAGE 1 — cluster (unless resuming)
# ======================================================================
if args.resume_from:
    log(f"RESUME: loading clustered checkpoint {args.resume_from}")
    adata = sc.read_h5ad(args.resume_from)
    present = set(adata.var_names)
    used = {k: [g for g in v if g in present] for k, v in MARKERS.items()}
    n0 = int(adata.uns.get("n_cells_preQC", adata.n_obs))
else:
    log(f"reading {args.in_h5ad}")
    adata = sc.read_h5ad(args.in_h5ad)
    log(f"loaded {adata.n_obs:,} cells x {adata.n_vars} genes")
    adata.layers["counts"] = adata.X.copy()

    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None)
    n0 = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=args.min_counts)
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    log(f"QC: {n0:,} -> {adata.n_obs:,} cells (min_counts={args.min_counts}, min_genes={args.min_genes})")

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # PCA on a scaled copy so we don't keep the dense scaled matrix around
    scaled = adata.copy()
    sc.pp.scale(scaled, max_value=10)
    sc.pp.pca(scaled, n_comps=50)
    adata.obsm["X_pca"] = scaled.obsm["X_pca"]
    adata.uns["pca"] = scaled.uns["pca"]
    del scaled; gc.collect()

    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    log("running leiden")
    sc.tl.leiden(adata, resolution=args.resolution, flavor="igraph", n_iterations=2, directed=False)
    log(f"leiden: {adata.obs['leiden'].nunique()} clusters")

    present = set(adata.var_names)
    used = {k: [g for g in v if g in present] for k, v in MARKERS.items()}
    for name, genes in used.items():
        sc.tl.score_genes(adata, genes, score_name=f"score_{name}")
    log("marker genes used: " + json.dumps(used))
    adata.uns["n_cells_preQC"] = n0


# ======================================================================
# STAGE 2 — per-cluster summary + astrocyte call + WDR49 signal
# ======================================================================
df = adata.obs.groupby("leiden")[SCORE_COLS].mean()
df["n_cells"] = adata.obs["leiden"].value_counts().sort_index()
key = [g for g in ["AQP4","SLC1A2","GJA1","FGFR3",GOI] if g in present]
X_key = adata[:, key].X
X_key = X_key.toarray() if hasattr(X_key, "toarray") else np.asarray(X_key)
expr = pd.DataFrame(X_key, columns=[f"mean_{g}" for g in key], index=adata.obs_names)
expr["leiden"] = adata.obs["leiden"].values
df = df.join(expr.groupby("leiden").mean())
df["call"] = df[SCORE_COLS].idxmax(axis=1).str.replace("score_", "")
df.to_csv(OUT / "cluster_marker_summary.csv")
log("wrote cluster_marker_summary.csv")

astro_clusters = df.index[df["call"] == "astro"].tolist()
adata.obs["is_astro"] = adata.obs["leiden"].isin(astro_clusters)
n_astro = int(adata.obs["is_astro"].sum())

summary = {
    "n_cells_preQC": int(n0),
    "n_cells_postQC": int(adata.n_obs),
    "n_leiden_clusters": int(adata.obs["leiden"].nunique()),
    "astro_clusters": astro_clusters,
    "n_astro_cells": n_astro,
    "astro_frac": round(n_astro / adata.n_obs, 4),
}
if GOI in present:
    w = adata[:, GOI].X
    w = w.toarray().ravel() if hasattr(w, "toarray") else np.asarray(w).ravel()
    m = adata.obs["is_astro"].values
    summary.update({
        "WDR49_pos_frac_in_astro": round(float((w[m] > 0).mean()), 4) if n_astro else None,
        "WDR49_pos_frac_nonastro": round(float((w[~m] > 0).mean()), 4),
        "WDR49_mean_in_astro": round(float(w[m].mean()), 4) if n_astro else None,
        "WDR49_mean_nonastro": round(float(w[~m].mean()), 4),
    })
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
log("SUMMARY: " + json.dumps(summary, indent=2))

# Free the big kNN graph IN PLACE first (embeddings recompute on subsamples).
# Do this BEFORE any copy/write: copying the full-data graph is what OOM'd before.
for attr in ["distances", "connectivities"]:
    if attr in adata.obsp: del adata.obsp[attr]
adata.uns.pop("neighbors", None)
gc.collect()

# checkpoint the (now lean) clustered object directly — NO .copy()
if not args.resume_from:
    adata.write_h5ad(CKPT, compression="gzip")
    log(f"wrote checkpoint {CKPT}")

# astrocyte subset for STREAM (lean adata still carries counts layer)
if n_astro and "counts" in adata.layers:
    adata[adata.obs["is_astro"]].copy().write_h5ad(OUT / "astro_subset.h5ad", compression="gzip")
    log(f"wrote astro_subset.h5ad ({n_astro:,} cells)")


# ======================================================================
# STAGE 3 — plots
# ======================================================================
all_markers = [g for k in MARKERS for g in used[k]] + ([GOI] if GOI in present else [])
km = [g for g in ["AQP4","SLC1A2","GJA1","MBP","PDGFRA","RBFOX3","P2RY12","PECAM1",GOI] if g in present]
SAMPLE_KEY = "sample" if "sample" in adata.obs else ("sd_code" if "sd_code" in adata.obs else None)

# general dotplot (clusters x markers)
try:
    sc.pl.dotplot(adata, all_markers, groupby="leiden", standard_scale="var",
                  show=False, save="_lineage_markers.png")
except Exception as e:
    log(f"general dotplot failed: {e}")

# per-sample dotplots
try:
    if SAMPLE_KEY:
        dpdir = OUT / "persample_dotplots"; dpdir.mkdir(exist_ok=True)
        for s in sorted(adata.obs[SAMPLE_KEY].astype(str).unique()):
            sel = (adata.obs[SAMPLE_KEY].astype(str) == s).values
            if sel.sum() < 200:
                continue
            ad_s = adata[sel]
            if ad_s.obs["leiden"].nunique() < 2:
                continue
            try:
                dp = sc.pl.dotplot(ad_s, all_markers, groupby="leiden",
                                   standard_scale="var", return_fig=True, title=str(s))
                dp.savefig(dpdir / f"dotplot_{s}.png"); plt.close("all")
            except Exception as ee:
                log(f"per-sample dotplot {s} failed: {ee}")
            del ad_s
        gc.collect()
        log("per-sample dotplots done")
except Exception as e:
    log(f"per-sample dotplots failed: {e}")


def embed_plots(emb, basis):
    """Full coloured-plot set for one embedding. basis in {'umap','tsne'}. Files
    are saved with the basis as prefix (umap_01_..., tsne_01_...)."""
    emb.obs["astrocyte"] = emb.obs["is_astro"].map({True:"astrocyte", False:"other"}).astype("category")
    def P(**kw):
        sc.pl.embedding(emb, basis=basis, show=False, **kw)
    P(color="leiden", legend_loc="on data", legend_fontsize=6, title=f"Leiden ({basis})", save="_01_leiden.png")
    P(color="astrocyte", palette={"astrocyte":"#d1495b","other":"#c9c9c9"}, title=f"Astrocyte call ({basis})", save="_02_astrocyte_call.png")
    P(color=SCORE_COLS, ncols=3, cmap="viridis", title=[c.replace("score_","") for c in SCORE_COLS], save="_03_lineage_scores.png")
    P(color=["score_astro"] + ([GOI] if GOI in present else []), ncols=2, cmap="magma", save="_04_astro_score_WDR49.png")
    P(color=km, ncols=3, cmap="viridis", save="_05_key_markers.png")
    if SAMPLE_KEY:
        emb.obs[SAMPLE_KEY] = emb.obs[SAMPLE_KEY].astype(str)
        P(color=SAMPLE_KEY, title=f"Sample batch-check ({basis})", save="_06_by_sample.png")
        xy = emb.obsm[f"X_{basis}"]; samples = sorted(emb.obs[SAMPLE_KEY].unique())
        ncol = 5; nrow = math.ceil(len(samples) / ncol)
        for colorby, tag in [("leiden", "07_persample_leiden"), ("astrocyte", "08_persample_astrocyte")]:
            cats = list(emb.obs[colorby].astype("category").cat.categories)
            base = plt.get_cmap("tab20")
            cmap_d = {c: base(i % 20) for i, c in enumerate(cats)}
            if colorby == "astrocyte":
                cmap_d = {"astrocyte":"#d1495b", "other":"#c9c9c9"}
            fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3, nrow*3)); axes = np.array(axes).ravel()
            for ax, s in zip(axes, samples):
                mk = (emb.obs[SAMPLE_KEY].values == s)
                ax.scatter(xy[~mk, 0], xy[~mk, 1], s=0.4, c="#ededed", linewidths=0, rasterized=True)
                cv = emb.obs[colorby].astype(str).values[mk]
                ax.scatter(xy[mk, 0], xy[mk, 1], s=0.9, c=[cmap_d.get(v, "#000000") for v in cv], linewidths=0, rasterized=True)
                ax.set_title(f"{s}  (n={int(mk.sum())})", fontsize=7); ax.set_xticks([]); ax.set_yticks([])
            for ax in axes[len(samples):]:
                ax.axis("off")
            fig.suptitle(f"Per-sample {basis.upper()} — coloured by {colorby}", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            fig.savefig(OUT / f"{basis}_{tag}.png", dpi=200); plt.close(fig)

# UMAP (recompute neighbours on the subsample — required by sc.tl.umap)
try:
    sub = adata if adata.n_obs <= args.umap_n else sc.pp.subsample(adata, n_obs=args.umap_n, copy=True)
    log(f"UMAP on {sub.n_obs:,} cells")
    sc.pp.neighbors(sub, n_neighbors=15, n_pcs=50)
    # [FINAL F2] init_pos="random": spectral init segfaults (no traceback) at scale.
    sc.tl.umap(sub, init_pos="random")
    embed_plots(sub, "umap")
    del sub; gc.collect()
except Exception as e:
    log(f"umap failed: {e}\n{traceback.format_exc()}")

# t-SNE (works from X_pca directly)
try:
    subt = adata if adata.n_obs <= args.tsne_n else sc.pp.subsample(adata, n_obs=args.tsne_n, copy=True)
    log(f"tSNE on {subt.n_obs:,} cells")
    sc.tl.tsne(subt, n_pcs=50, n_jobs=8)
    embed_plots(subt, "tsne")
    del subt; gc.collect()
except Exception as e:
    log(f"tsne failed: {e}\n{traceback.format_exc()}")

log("DONE")
