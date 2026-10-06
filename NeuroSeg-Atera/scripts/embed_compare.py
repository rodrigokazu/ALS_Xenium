#!/usr/bin/env python3
"""UMAP vs t-SNE for the annotated Atera cells (final pipeline), next to the earlier UMAP (our segmentation without the Ranger
glia step). Same preprocessing as atera_annot.py (median-count normalisation, log1p, 3,000 HVG, scale, PCA 50); t-SNE with
openTSNE (FFT, perplexity 30, PCA initialisation) on all QC-passing cells. Colours = lineage, as in poster panel a.
Run in the `resolvi` env (has openTSNE):  python embed_compare.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, matplotlib
from openTSNE import TSNE
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 11})
SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); H = SSD / "segmentation_hybridC"; F = SSD / "segmentation_full"
COL = {"Purkinje": "#2c6a9a", "Vascular, immune": "#5a5a5a", "Interneurons": "#a37cc4", "Astroglia": "#e0927a", "Oligodendroglia": "#77b56e", "Granule layer": "#cbbfa6"}
GROUP = {"Granule / UBC": "Granule layer", "Inhibitory interneuron": "Interneurons", "Astroglia": "Astroglia", "Oligodendroglia": "Oligodendroglia",
         "Immune": "Vascular, immune", "Vascular": "Vascular, immune", "Fibroblast / meningeal": "Vascular, immune", "Purkinje": "Purkinje"}
out = H / "annotation/tsne.parquet"
if not out.exists():
    a = ad.read_h5ad(H / "annotation/atera_cerebellum_annotated.h5ad"); a = a[a.obs.qc_pass.astype(bool)].copy()
    sc.pp.normalize_total(a, target_sum=float(np.median(np.asarray(a.X.sum(1)).ravel()))); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat"); a = a[:, a.var.highly_variable].copy(); sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50, svd_solver="arpack")
    Y = TSNE(perplexity=30, n_jobs=8, random_state=0, initialization="pca", verbose=True).fit(a.obsm["X_pca"])
    pd.DataFrame(np.asarray(Y), index=a.obs_names, columns=["tsne1", "tsne2"]).to_parquet(out)
ts = pd.read_parquet(out)
def load(d):
    x = pd.read_parquet(d / "annotation/cells_annotated.parquet"); x["grp"] = x.lineage.map(GROUP); return x[x.grp.notna()]
hy, fu = load(H), load(F); hy = hy.join(ts, how="inner") if ts.index.dtype == hy.index.dtype else hy.join(ts.set_index(ts.index.astype(hy.index.dtype)), how="inner")
def draw(ax, d, xk, yk, title):
    g = d[d.grp == "Granule layer"].sample(frac=0.1, random_state=0); o = d[d.grp != "Granule layer"]
    ax.scatter(g[xk], g[yk], s=0.15, color=COL["Granule layer"], lw=0, rasterized=True)
    for k in ["Vascular, immune", "Interneurons", "Astroglia", "Oligodendroglia"]:
        q = o[o.grp == k]; ax.scatter(q[xk], q[yk], s=0.25, color=COL[k], lw=0, rasterized=True)
    q = o[o.grp == "Purkinje"]; ax.scatter(q[xk], q[yk], s=1.5, color=COL["Purkinje"], lw=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
fig, axs = plt.subplots(1, 3, figsize=(18, 6.4), dpi=200)
draw(axs[0], fu, "umap1", "umap2", f"UMAP, our segmentation only\n({len(fu):,} cells)")
draw(axs[1], hy, "umap1", "umap2", f"UMAP, final pipeline\n({len(hy):,} cells)")
draw(axs[2], hy, "tsne1", "tsne2", "t-SNE, final pipeline")
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COL[k], ms=8, label=k) for k in COL], loc="lower center", ncol=6, frameon=False, fontsize=11)
fig.text(0.01, 0.005, "Granule cells subsampled to 10% for display; all other cells shown. Raw counts (no ResolVI).", fontsize=10, color="#52514e")
fig.subplots_adjust(bottom=0.1, top=0.9, wspace=0.05); fig.savefig(H / "annotation/embedding_compare.png", facecolor="white"); print("saved")
