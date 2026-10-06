#!/usr/bin/env python3
"""QC, clustering, UMAP and first-pass annotation of the full-section Atera segmentation (segmentation_full/).

QC (whole-sample, fixed): qc_pass = n_transcripts >= 200 and n_genes >= 100 and area >= 10 um2; failing cells are kept in the
object with qc_pass = False (not removed). Clustering on qc_pass cells: normalise to the median count, log1p, 3,000 highly
variable genes, scale (max 10), PCA 50, 15-NN graph, Leiden (resolution 2.0), UMAP.
Annotation: every Leiden cluster gets the cell type whose marker set has the highest mean scaled expression in that cluster;
a cluster whose best and second-best scores differ by < 0.25 is called "mixed: A / B". Types are grouped into eight lineages
for colour. Markers are a fixed literature panel for human cerebellum (below), chosen before looking at the clusters.
python atera_annot.py
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, matplotlib
from scipy import sparse
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
sc.settings.n_jobs = 8
import os
O = Path(os.environ.get("SEG_DIR", "/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/segmentation_full")); A = O / "annotation"; A.mkdir(exist_ok=True)
MARKERS = {
    "Granule": ["GABRA6", "CBLN3", "FAT2", "NEUROD1", "CADPS2"],
    "UBC": ["EOMES", "CALB2", "TRPC3"],
    "Purkinje": ["CALB1", "PCP2", "CA8", "ITPR1", "PCP4"],
    "Golgi": ["GRM2", "LGI2", "SST"],
    "MLI (basket / stellate)": ["PVALB", "SORCS3", "KIT", "LYPD6", "GAD2"],
    "Bergmann glia": ["GDF10", "TNC"],
    "Astrocyte": ["AQP4", "GFAP", "ALDH1L1", "SLC1A2"],
    "Oligodendrocyte": ["MBP", "MOBP", "PLP1", "MOG", "ST18"],
    "OPC": ["PDGFRA", "CSPG4", "OLIG2", "VCAN"],
    "Microglia": ["C1QC", "P2RY12", "CX3CR1", "CSF1R"],
    "Endothelial": ["CLDN5", "FLT1", "VWF"],
    "Pericyte / SMC": ["PDGFRB", "RGS5", "ACTA2", "MYH11"],
    "Fibroblast / meningeal": ["DCN", "COL1A2", "LUM", "SLC6A13"],
    "Border macrophage": ["MRC1", "F13A1", "LYVE1"],
    "Lymphocyte": ["PTPRC", "CD3E", "CD2"],
}
LINEAGE = {"Granule": "Granule / UBC", "UBC": "Granule / UBC", "Purkinje": "Purkinje", "Golgi": "Inhibitory interneuron",
           "MLI (basket / stellate)": "Inhibitory interneuron", "Bergmann glia": "Astroglia", "Astrocyte": "Astroglia",
           "Oligodendrocyte": "Oligodendroglia", "OPC": "Oligodendroglia", "Microglia": "Immune", "Lymphocyte": "Immune", "Border macrophage": "Immune",
           "Endothelial": "Vascular", "Pericyte / SMC": "Vascular", "Fibroblast / meningeal": "Fibroblast / meningeal"}
LIN_ORDER = ["Granule / UBC", "Purkinje", "Inhibitory interneuron", "Astroglia", "Oligodendroglia", "Immune", "Vascular", "Fibroblast / meningeal"]
LIN_COL = dict(zip(LIN_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]))  # validated slots 1-8
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)

X = sparse.load_npz(O / "cell_by_gene.npz").tocsr().astype(np.float32); cells = pd.read_parquet(O / "cells.parquet"); genes = pd.read_csv(O / "genes.csv").gene.astype(str)
adata = ad.AnnData(X=X, obs=cells.set_index(cells.cell.astype(str).rename("cell_key")), var=pd.DataFrame(index=genes.to_numpy()))
adata.obs["n_transcripts"] = np.asarray(X.sum(1)).ravel(); adata.obs["n_genes"] = np.asarray((X > 0).sum(1)).ravel()
adata.obs["qc_pass"] = (adata.obs.n_transcripts >= 200) & (adata.obs.n_genes >= 100) & (adata.obs.area_um2 >= 10)
adata.obs["nucleus_free_purkinje_repair"] = adata.obs.cell.astype(np.int64) >= 3_500_000_000
adata.obsm["spatial"] = adata.obs[["x_centroid_um", "y_centroid_um"]].to_numpy()
qc = {"cells": int(adata.n_obs), "qc_pass": int(adata.obs.qc_pass.sum()), "median_transcripts": float(adata.obs.n_transcripts.median()),
      "median_genes": float(adata.obs.n_genes.median())}; json.dump(qc, open(A / "qc.json", "w"), indent=1); log("QC", qc)
q = adata[adata.obs.qc_pass].copy(); q.layers["counts"] = q.X.copy()
sc.pp.normalize_total(q, target_sum=float(np.median(q.obs.n_transcripts))); sc.pp.log1p(q)
sc.pp.highly_variable_genes(q, n_top_genes=3000, flavor="seurat"); log("HVG")
qh = q[:, q.var.highly_variable].copy(); sc.pp.scale(qh, max_value=10); sc.tl.pca(qh, n_comps=50, svd_solver="arpack"); log("PCA")
q.obsm["X_pca"] = qh.obsm["X_pca"]; del qh
sc.pp.neighbors(q, n_neighbors=15, use_rep="X_pca"); log("neighbors")
sc.tl.leiden(q, resolution=2.0, flavor="igraph", n_iterations=2, key_added="leiden"); log("leiden", q.obs.leiden.nunique())
sc.tl.umap(q, min_dist=0.3); log("UMAP")
# ---- annotation: cluster means of scaled log expression over the marker sets
mk = {t: [g for g in gs if g in q.var_names] for t, gs in MARKERS.items()}
allm = sorted({g for gs in mk.values() for g in gs})
E = pd.DataFrame(q[:, allm].X.toarray() if sparse.issparse(q.X) else q[:, allm].X, columns=allm)
E["leiden"] = q.obs.leiden.to_numpy(); cm = E.groupby("leiden").mean(); cmz = (cm - cm.mean()) / cm.std(ddof=0)
score = pd.DataFrame({t: cmz[gs].mean(1) for t, gs in mk.items() if gs})
# absolute check: a type is only eligible in a cluster where its markers are clearly expressed (mean log expression of the set
# >= 2x its median over clusters); z-scores alone call weakly expressed sets (immune genes) in clusters that barely express them
absm = pd.DataFrame({t: cm[gs].mean(1) for t, gs in mk.items() if gs}); elig = absm >= 2 * absm.median()
score = score.where(elig, -np.inf)
score.loc[~np.isfinite(score).any(axis=1), "Granule"] = 0.0      # nothing eligible: fall back to the majority type, flagged by margin 0
best = score.idxmax(1); srt = np.sort(np.where(np.isfinite(score.to_numpy()), score.to_numpy(), -9), 1); margin = pd.Series(srt[:, -1] - srt[:, -2], index=score.index)
second = score.apply(lambda r: r.drop(r.idxmax()).idxmax(), axis=1)
lab = {c: (best[c] if margin[c] >= 0.25 else f"mixed: {best[c]} / {second[c]}") for c in score.index}
q.obs["cell_type"] = q.obs.leiden.map(lab).astype("category")
q.obs["lineage"] = q.obs.leiden.map(lambda c: LINEAGE[best[c]]).astype("category")
q.obs["annotation_margin"] = q.obs.leiden.map(margin).astype(float)
q.obs[["leiden"]].assign(umap1=q.obsm["X_umap"][:, 0], umap2=q.obsm["X_umap"][:, 1]).to_parquet(A / "leiden_umap.parquet")
score.assign(best=best, second=second, margin=margin, n_cells=q.obs.leiden.value_counts()).to_csv(A / "cluster_marker_scores.csv")
log("annotation", q.obs.cell_type.value_counts().to_dict())
# write: full object with annotation for qc_pass cells
for c in ("leiden", "cell_type", "lineage"):
    adata.obs[c] = q.obs[c].reindex(adata.obs_names).astype(str).replace("nan", "QC fail")
adata.obsm["X_umap"] = np.full((adata.n_obs, 2), np.nan); adata.obsm["X_umap"][adata.obs.qc_pass.to_numpy()] = q.obsm["X_umap"]
adata.write_h5ad(A / "atera_cerebellum_annotated.h5ad", compression="gzip"); log("written h5ad")
q.obs[["leiden", "cell_type", "lineage", "annotation_margin"]].assign(umap1=q.obsm["X_umap"][:, 0], umap2=q.obsm["X_umap"][:, 1],
    x_um=q.obs.x_centroid_um, y_um=q.obs.y_centroid_um, n_transcripts=q.obs.n_transcripts).to_parquet(A / "cells_annotated.parquet")
# ---- figures: UMAP by lineage with fine labels | spatial map | marker dot plot
u = q.obsm["X_umap"]; rng = np.random.default_rng(0); idx = rng.permutation(q.n_obs)
fig, axs = plt.subplots(1, 2, figsize=(19, 8.2), dpi=150, gridspec_kw=dict(width_ratios=[1, 1.55], wspace=0.04))
col = q.obs.lineage.map(LIN_COL).to_numpy()
axs[0].scatter(u[idx, 0], u[idx, 1], s=0.15, c=col[idx], lw=0, rasterized=True)
for t, g in q.obs.groupby("cell_type", observed=True):
    if len(g) < 500: continue
    m = np.median(u[q.obs_names.get_indexer(g.index)], 0)
    axs[0].text(m[0], m[1], f"{t}\n{len(g):,}", fontsize=7.5, ha="center", va="center", color="#0b0b0b",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
axs[0].set_title(f"UMAP, {q.n_obs:,} QC-passing cells, {q.obs.leiden.nunique()} Leiden clusters", loc="left", fontsize=11)
gr = (q.obs.lineage == "Granule / UBC").to_numpy()
xs_, ys_ = q.obs.x_centroid_um.to_numpy(), q.obs.y_centroid_um.to_numpy()
axs[1].scatter(xs_[gr], ys_[gr], s=0.04, c="#b7d3f6", lw=0, rasterized=True)                    # granule layer faint underneath
oi = idx[~gr[idx]]; axs[1].scatter(xs_[oi], ys_[oi], s=0.35, c=col[oi], lw=0, rasterized=True)
axs[1].set_aspect("equal"); axs[1].invert_yaxis(); axs[1].set_title("The same cells in the section", loc="left", fontsize=11)
for a in axs: a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
hand = [plt.Line2D([], [], marker="o", ls="", color=LIN_COL[l], ms=7, label=f"{l} ({(q.obs.lineage == l).sum():,})") for l in LIN_ORDER if (q.obs.lineage == l).any()]
fig.legend(handles=hand, loc="lower center", ncol=4, frameon=False, fontsize=10)
fig.suptitle("Atera human cerebellum: annotation of the new segmentation (first pass, marker-based)", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
fig.subplots_adjust(bottom=0.11, top=0.91, left=0.01, right=0.995); fig.savefig(A / "umap_spatial.png", facecolor="white"); log("figure umap")
order = [t for t in MARKERS if (q.obs.cell_type == t).any()] + sorted([t for t in q.obs.cell_type.cat.categories if t.startswith("mixed")])
q.obs["cell_type_ord"] = pd.Categorical(q.obs.cell_type.astype(str), categories=order)
dp = sc.pl.dotplot(q, {t: mk[t] for t in MARKERS if mk[t]}, groupby="cell_type_ord", standard_scale="var", show=False, return_fig=True, cmap="Blues")
dp.savefig(A / "marker_dotplot.png", dpi=150, bbox_inches="tight"); log("dotplot done")
