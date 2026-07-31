#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/celltype_plots_FINAL.py
#
# The combined and per-sample figures for coarse cell typing: UMAP and tSNE coloured by cell
# type and by Leiden cluster, a general ranked-marker dotplot, and per-sample composition
# bars. Reads the processed object plus X_tsne.npy.
#
# Two fixes are folded in and both should stay. Per-sample composition groupby uses
# observed=True, without which unused categorical levels produce empty groups and a misleading
# bar chart. And the fragile per-sample dotplot loop that part-failed job 52137644 has been
# removed entirely instead of being patched; per-sample dotplots now come only from
# celltype_dotplots_persample_FINAL.py.
#
# So this file draws the combined figures and the composition, and nothing per-sample except
# the scatter plots.
# ========================================================================================

"""celltype_plots_FINAL.py -- combined + per-sample UMAP & t-SNE (by coarse cell type and by
leiden cluster), a general ranked-marker dotplot, and per-sample composition bars, for the
MN-corrected _FINAL cohort. Reads the processed object + X_tsne.npy. pertpy_env.

Adapted from CellTyping_coarse/code/celltype_plots.py. Crew F-review deltas:
  F5: import final_config; OUT = cfg.COARSE_TYPING_DIR (no hardcoded _gausss path). OUT is
      identical to the pipeline/tsne so X_tsne.npy stays row-aligned with the h5ad.
  F6: (a) fold the plots_fix robustness -- per-sample composition groupby uses observed=True;
      (b) DROP the fragile per-sample dotplot loop that part-failed job 52137644. Per-sample
      dotplots now come SOLELY from celltype_dotplots_persample_FINAL.py (dense-submatrix,
      scipy-int32-safe). This script keeps only the combined/general figures + composition."""
import os, sys, numpy as np, pandas as pd, scanpy as sc, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_config as cfg
sc.settings.verbosity = 1
OUT = str(cfg.COARSE_TYPING_DIR)                       # F5: CellTyping_coarse_FINAL
FIG = f"{OUT}/figures"; os.makedirs(FIG, exist_ok=True)

CT_COLOR = {
    "Oligodendrocyte":"#3B6FB6","OPC":"#55A868","Astrocyte":"#C44E52","Microglia":"#8172B3",
    "Macrophage_perivasc":"#8C6D5C","Neuron_excit":"#DA8BC3","Neuron_inhib":"#B79F3B",
    "MotorNeuron":"#E8000B","Endothelial":"#5FBFD6","Mural":"#7F7F7F","Fibroblast_VLMC":"#FF9F1C",
    "Lymphoid":"#2CA02C","LowQuality":"#D9D9D9",
}
# compact ordered marker panel for the general dotplot (canonical, in-panel)
DOT = {
    "Oligodendrocyte":["MOBP","MOG","MAG"],"OPC":["PDGFRA","CSPG4","OLIG1"],
    "Astrocyte":["AQP4","GJA1","SLC1A2"],"Microglia":["P2RY12","C1QC","TYROBP"],
    "Macrophage_perivasc":["CD163","LYVE1","MSR1"],"Neuron_excit":["SLC17A6","RBFOX3","SNCG"],
    "Neuron_inhib":["GAD1","GAD2","PVALB"],"MotorNeuron":["MNX1","STMN2"],
    "Endothelial":["PECAM1","VWF","CAV1"],"Mural":["ABCC9"],
    "Fibroblast_VLMC":["DCN","FBLN1","COL5A2"],"Lymphoid":["PTPRC","CD4","THEMIS"],
}

def order_types(present):
    return [t for t in CT_COLOR if t in present]

def scatter(ax, xy, labels, order, s=0.6, title="", legend=False):
    ax.set_facecolor("white")
    for t in order:
        m = labels.values == t
        if m.sum()==0: continue
        ax.scatter(xy[m,0], xy[m,1], s=s, c=CT_COLOR.get(t,"#888"), linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
    for sp in ax.spines.values(): sp.set_visible(False)
    if legend:
        h=[Line2D([0],[0],marker='o',linestyle='none',markersize=6,markerfacecolor=CT_COLOR[t],markeredgecolor='none',label=t) for t in order]
        ax.legend(handles=h, loc='center left', bbox_to_anchor=(1.01,0.5), frameon=False, fontsize=8)

def main():
    adata = sc.read_h5ad(f"{OUT}/ALS_SCXenium_coarse_celltyped.h5ad")
    tsne = np.load(f"{OUT}/X_tsne.npy"); adata.obsm["X_tsne"] = tsne
    ct = adata.obs["cell_type_coarse"].astype(str)
    order = order_types(set(ct.unique()))
    ump = adata.obsm["X_umap"]; tsn = adata.obsm["X_tsne"]
    print("cells", adata.n_obs, "types", order, flush=True)

    # --- combined UMAP & tSNE by cell type ---
    for emb, name in [(ump,"umap"),(tsn,"tsne")]:
        fig, ax = plt.subplots(figsize=(9,8))
        scatter(ax, emb, ct, order, s=0.5, title=f"Combined {name.upper()} - coarse cell type (n={adata.n_obs:,})", legend=True)
        for e in ("png","pdf"): fig.savefig(f"{FIG}/combined_{name}_celltype.{e}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    # --- combined UMAP & tSNE by leiden cluster ---
    for key in ["leiden_r10"]:
        for emb, name in [(ump,"umap"),(tsn,"tsne")]:
            fig, ax = plt.subplots(figsize=(9,8))
            sc.pl.embedding(adata, basis=name, color=key, ax=ax, show=False, size=3, legend_loc="on data", legend_fontsize=6, title=f"Combined {name.upper()} - {key}")
            for e in ("png","pdf"): fig.savefig(f"{FIG}/combined_{name}_{key}.{e}", dpi=200, bbox_inches="tight")
            plt.close(fig)

    samples = sorted(adata.obs["sample"].astype(str).unique())
    # --- per-sample UMAP & tSNE grids (global embedding, sample cells coloured, rest grey) ---
    for emb, name in [(ump,"umap"),(tsn,"tsne")]:
        n=len(samples); ncol=5; nrow=int(np.ceil(n/ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1*ncol, 3.0*nrow)); axes=np.array(axes).ravel()
        for ax in axes: ax.axis("off")
        for i,s in enumerate(samples):
            ax=axes[i]; ax.axis("on")
            ax.scatter(emb[:,0], emb[:,1], s=0.15, c="#ECECEC", linewidths=0, rasterized=True)
            m=(adata.obs["sample"].astype(str)==s).values
            scatter(ax, emb[m], ct[m], order, s=0.5, title=f"{s} (n={m.sum():,})")
        for e in ("png","pdf"): fig.savefig(f"{FIG}/persample_{name}_celltype.{e}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    # --- dotplot general ---
    var_names = {t:[g for g in DOT[t] if g in adata.raw.var_names] for t in order if t in DOT}
    var_names = {t:v for t,v in var_names.items() if v}
    adata.obs["cell_type_coarse"] = adata.obs["cell_type_coarse"].cat.reorder_categories([t for t in order])
    dp = sc.pl.dotplot(adata, var_names, groupby="cell_type_coarse", use_raw=True, standard_scale="var", show=False, return_fig=True)
    dp.savefig(f"{FIG}/dotplot_general_celltype.png", dpi=200, bbox_inches="tight")
    dp.savefig(f"{FIG}/dotplot_general_celltype.pdf", bbox_inches="tight")

    # NOTE (F6): the fragile per-sample dotplot loop was REMOVED here (it part-failed job
    # 52137644 on an anndata view-copy / scipy int32 sparse crash). Per-sample dotplots are
    # produced SOLELY by celltype_dotplots_persample_FINAL.py (dense marker submatrix).

    # --- per-sample composition stacked bars (F6: observed=True, obs-only, robust) ---
    comp = (adata.obs.groupby("sample", observed=True)["cell_type_coarse"].value_counts(normalize=True).unstack().fillna(0))
    comp = comp[[t for t in order if t in comp.columns]]
    fig, ax = plt.subplots(figsize=(12,6)); bottom=np.zeros(len(comp))
    for t in comp.columns:
        ax.bar(comp.index, comp[t].values, bottom=bottom, color=CT_COLOR.get(t,"#888"), label=t); bottom+=comp[t].values
    ax.set_ylabel("fraction of cells"); ax.set_title("Coarse cell-type composition per sample")
    ax.legend(loc='center left', bbox_to_anchor=(1.01,0.5), frameon=False, fontsize=8); plt.xticks(rotation=90)
    for e in ("png","pdf"): fig.savefig(f"{FIG}/persample_composition.{e}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    comp.to_csv(f"{OUT}/persample_composition.csv")
    print("PLOTS DONE ->", FIG, flush=True)

if __name__ == "__main__":
    main()
