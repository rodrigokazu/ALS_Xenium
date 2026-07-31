#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | cell_annotation/sc_figs.py
#
# Figures from the clustered object: UMAP and spatial map by Leiden, an is_MN highlight, a
# per-cluster marker dotplot, and a grid of canonical CNS markers on the UMAP.
#
# Every figure is wrapped so one failure does not sink the rest. With 22 canonical markers
# against a 480-gene panel, some will be absent, and the run should produce the figures it can
# instead of stopping at the first missing gene. Thirteen of 22 were present on the smoke
# test.
#
# 300 dpi with editable text at fonttype 42, so panels can be adjusted in Illustrator later.
# ========================================================================================

"""sc_figs.py -- publication-quality PNGs from a clustered spinal-cord Xenium h5ad (sc_cluster.py output).
UMAP + spatial map by Leiden, is_MN highlight, per-cluster marker dotplot, canonical CNS marker UMAP grid.
Each figure guarded so one failure doesn't sink the rest. 300 dpi, editable text (fonttype 42)."""
import argparse, os, sys
def log(*a,**k): k.setdefault("flush",True); print(*a,**k)

CANON = [  # (gene, cell-type label) — filtered to panel-present
 ("RBFOX3","neuron"),("SNAP25","neuron"),("SLC17A7","excitatory"),("GAD1","inhibitory"),("GAD2","inhibitory"),
 ("AQP4","astrocyte"),("GFAP","astrocyte"),("SLC1A2","astrocyte"),
 ("MOBP","oligo"),("PLP1","oligo"),("MBP","oligo"),("PDGFRA","OPC"),("OLIG1","OPC"),
 ("P2RY12","microglia"),("CSF1R","microglia"),("C1QB","microglia"),
 ("CLDN5","endothelial"),("FLT1","endothelial"),
 ("MNX1","motor neuron"),("CHAT","motor neuron"),("ISL1","motor neuron"),("STMN2","MN/neuronal"),
]

def style():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","font.family":"sans-serif",
        "figure.dpi":120,"axes.titlesize":11,"axes.labelsize":10,"legend.fontsize":8,
        "axes.spines.top":False,"axes.spines.right":False})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--outdir",required=True)
    ap.add_argument("--tag",default="sample"); a=ap.parse_args()
    os.makedirs(a.outdir,exist_ok=True); style()
    import numpy as np, pandas as pd, scanpy as sc
    import matplotlib.pyplot as plt
    log(f"[env] scanpy {sc.__version__}")
    ad=sc.read_h5ad(a.input); log(f"[load] {ad.n_obs:,} x {ad.n_vars}; clusters={ad.obs['leiden'].nunique()}")
    sc.settings.figdir=a.outdir; DPI=300
    def save(fig,name):
        p=os.path.join(a.outdir,f"{a.tag}__{name}.png"); fig.savefig(p,dpi=DPI,bbox_inches="tight",facecolor="white"); plt.close(fig); log(f"[fig] {p}")

    # UMAP (neighbors already in the object)
    try:
        if "X_umap" not in ad.obsm: sc.tl.umap(ad,random_state=0); log("[umap] computed")
    except Exception as e: log(f"[umap] FAILED {e}")

    ncl=ad.obs["leiden"].nunique()
    pal=(sc.pl.palettes.default_20 if ncl<=20 else sc.pl.palettes.default_102)

    # 1) UMAP by leiden
    try:
        fig=sc.pl.umap(ad,color="leiden",legend_loc="on data",legend_fontsize=7,palette=pal,
                       frameon=False,title=f"{a.tag}: Leiden clusters (n={ncl})",show=False,return_fig=True)
        save(fig,"umap_leiden")
    except Exception as e: log(f"[fig umap_leiden] FAILED {e}")

    # 2) spatial map by leiden (microns; imaging y-down; equal aspect + scalebar)
    try:
        x=ad.obs["x_centroid"].astype(float).values; y=ad.obs["y_centroid"].astype(float).values
        cats=ad.obs["leiden"].astype("category"); codes=cats.cat.codes.values
        import matplotlib
        cmap=matplotlib.colors.ListedColormap(pal[:len(cats.cat.categories)])
        fig,ax=plt.subplots(figsize=(8,8))
        ax.scatter(x,y,c=codes,cmap=cmap,s=2,linewidths=0,rasterized=True)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
        x0=x.min()+ (x.max()-x.min())*0.03; y0=y.max()-(y.max()-y.min())*0.03
        ax.plot([x0,x0+500],[y0,y0],color="k",lw=3); ax.text(x0+250,y0-(y.max()-y.min())*0.02,"500 um",ha="center",va="bottom",fontsize=9)
        ax.set_title(f"{a.tag}: Leiden clusters in situ")
        # legend
        from matplotlib.lines import Line2D
        h=[Line2D([0],[0],marker='o',ls='none',mfc=pal[i%len(pal)],mec='none',ms=6,label=c) for i,c in enumerate(cats.cat.categories)]
        ax.legend(handles=h,loc='center left',bbox_to_anchor=(1.01,0.5),frameon=False,ncol=1 if ncl<=16 else 2,title="Leiden")
        save(fig,"spatial_leiden")
    except Exception as e: log(f"[fig spatial_leiden] FAILED {e}")

    # 3) is_MN highlight (UMAP + spatial)
    try:
        if "is_MN" in ad.obs:
            mn=ad.obs["is_MN"].astype(str).str.lower().isin(["true","1","1.0"]).values
            fig,axs=plt.subplots(1,2,figsize=(15,7.2))
            U=ad.obsm.get("X_umap")
            if U is not None:
                axs[0].scatter(U[~mn,0],U[~mn,1],s=2,c="0.8",linewidths=0,rasterized=True)
                axs[0].scatter(U[mn,0],U[mn,1],s=6,c="#D62728",linewidths=0,rasterized=True)
                axs[0].set_title(f"is_MN on UMAP (n={int(mn.sum())})"); axs[0].axis("off")
            x=ad.obs["x_centroid"].astype(float).values; y=ad.obs["y_centroid"].astype(float).values
            axs[1].scatter(x[~mn],y[~mn],s=1.5,c="0.85",linewidths=0,rasterized=True)
            axs[1].scatter(x[mn],y[mn],s=10,c="#D62728",linewidths=0,rasterized=True)
            axs[1].set_aspect("equal"); axs[1].invert_yaxis(); axs[1].axis("off"); axs[1].set_title("is_MN in situ")
            save(fig,"is_MN_highlight")
        else: log("[fig is_MN] no is_MN column")
    except Exception as e: log(f"[fig is_MN] FAILED {e}")

    # 4) marker dotplot: top-5 rank_genes_groups per cluster (on lognorm X)
    try:
        sc.tl.rank_genes_groups(ad,"leiden",method="wilcoxon",n_genes=5)
        fig=sc.pl.rank_genes_groups_dotplot(ad,n_genes=5,standard_scale="var",show=False,return_fig=True)
        # DotPlot.fig may be None until materialised -> use savefig
        p=os.path.join(a.outdir,f"{a.tag}__marker_dotplot_top5.png")
        try: fig.savefig(p,dpi=DPI); log(f"[fig] {p}")
        except Exception:
            import matplotlib.pyplot as plt; plt.gcf().savefig(p,dpi=DPI,bbox_inches="tight"); plt.close("all"); log(f"[fig] {p} (via gcf)")
    except Exception as e: log(f"[fig dotplot] FAILED {e}")

    # 5) canonical CNS marker UMAP grid
    try:
        genes=[g for g,_ in CANON if g in ad.var_names]
        log(f"[markers] present {len(genes)}/{len(CANON)}: {genes}")
        if genes and "X_umap" in ad.obsm:
            ncol=4; nrow=int(np.ceil(len(genes)/ncol))
            fig=sc.pl.umap(ad,color=genes,ncols=ncol,frameon=False,cmap="magma",show=False,return_fig=True,
                           title=[f"{g} ({lab})" for g,lab in CANON if g in ad.var_names])
            save(fig,"umap_canonical_markers")
        # canonical marker dotplot grouped by cluster too
        if genes:
            fig=sc.pl.dotplot(ad,var_names=genes,groupby="leiden",standard_scale="var",show=False,return_fig=True)
            p=os.path.join(a.outdir,f"{a.tag}__canonical_dotplot.png")
            try: fig.savefig(p,dpi=DPI); log(f"[fig] {p}")
            except Exception: import matplotlib.pyplot as plt; plt.gcf().savefig(p,dpi=DPI,bbox_inches="tight"); plt.close("all"); log(f"[fig] {p} (via gcf)")
    except Exception as e: log(f"[fig markers] FAILED {e}")

    log("FIGS_DONE")

if __name__=="__main__": main()
