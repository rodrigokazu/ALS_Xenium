#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/celltype_dotplots_persample_FINAL.py
#
# The only place per-sample marker dotplots are produced, split out after the in-pipeline
# version kept failing.
#
# It computes the dotplot by hand from a dense marker-gene submatrix rather than going through
# an AnnData copy or a raw slice. That is what the file is for. The AnnData route triggers a
# scipy int32 sparse indexing crash on objects this size. That is what killed the per-sample
# loop in the main plots job. Do not tidy this back into the idiomatic scanpy call.
#
# Dot size is the fraction of cells expressing; colour is the per-gene min-max scaled mean of
# the log-normalised values.
#
# If you touch it, preserve the np.asarray(...todense()) recipe on the submatrix.
# ========================================================================================

"""celltype_dotplots_persample_FINAL.py -- per-sample marker dotplots for the MN-corrected
_FINAL cohort, computed MANUALLY from a DENSE marker-gene submatrix (no AnnData copy / raw
slice -> avoids the scipy int32 sparse crash that killed the per-sample loop in the main
plots job). Dot size = fraction of cells expressing; colour = per-gene min-max scaled mean
lognorm. This is the SOLE source of per-sample dotplots (F6).

Adapted from CellTyping_coarse/code/celltype_dotplots_persample.py. Only F5 change:
OUT = cfg.COARSE_TYPING_DIR (no hardcoded _gausss path). The dense-submatrix / np.asarray(
todense()) recipe is PRESERVED verbatim -- do NOT regress to an AnnData raw-slice. pertpy_env."""
import os, sys, numpy as np, pandas as pd, scanpy as sc, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_config as cfg
OUT = str(cfg.COARSE_TYPING_DIR)                       # F5: CellTyping_coarse_FINAL
FIG = f"{OUT}/figures/dotplots_persample"; os.makedirs(FIG, exist_ok=True)
CT_ORDER = ["Oligodendrocyte","OPC","Astrocyte","Microglia","Neuron_excit","MotorNeuron","Endothelial","Fibroblast_VLMC","Lymphoid"]
DOT = {"Oligodendrocyte":["MOBP","MOG","MAG"],"OPC":["PDGFRA","CSPG4","OLIG1"],"Astrocyte":["AQP4","GJA1","SLC1A2"],
    "Microglia":["P2RY12","C1QC","TYROBP"],"Neuron_excit":["SLC17A6","RBFOX3","SNCG"],"MotorNeuron":["MNX1","STMN2"],
    "Endothelial":["PECAM1","VWF","CAV1"],"Fibroblast_VLMC":["DCN","FBLN1","COL5A2"],"Lymphoid":["PTPRC","CD4","THEMIS"]}

def main():
    adata = sc.read_h5ad(f"{OUT}/ALS_SCXenium_coarse_celltyped.h5ad")   # X = lognorm
    genes = [g for t in CT_ORDER for g in DOT.get(t,[]) if g in adata.var_names]
    # dense marker submatrix (no AnnData copy of raw) -- scipy-int32-safe (KEEP)
    Xg = adata[:, genes].X
    Xg = np.asarray(Xg.todense()) if hasattr(Xg, "todense") else np.asarray(Xg)
    ct = adata.obs["cell_type_coarse"].astype(str).values
    samp = adata.obs["sample"].astype(str).values
    df = pd.DataFrame(Xg, columns=genes); df["ct"]=ct; df["samp"]=samp
    samples = sorted(pd.unique(samp))
    for s in samples:
        d = df[df["samp"]==s]
        present = [t for t in CT_ORDER if t in set(d["ct"])]
        if not present: continue
        # per (celltype,gene): mean lognorm + fraction expressing
        mean = np.zeros((len(present),len(genes))); frac = np.zeros_like(mean)
        for i,t in enumerate(present):
            sub = d[d["ct"]==t][genes].values
            mean[i]=sub.mean(0); frac[i]=(sub>0).mean(0)
        # per-gene min-max scale across present celltypes (like standard_scale='var')
        mn=mean.min(0,keepdims=True); mx=mean.max(0,keepdims=True); scaled=(mean-mn)/np.clip(mx-mn,1e-9,None)
        fig,ax=plt.subplots(figsize=(0.42*len(genes)+2, 0.5*len(present)+1.5))
        for i in range(len(present)):
            for j in range(len(genes)):
                ax.scatter(j,i,s=8+frac[i,j]*230,c=[[0.55*scaled[i,j]+0.05, 0.05, 0.15+0.0*scaled[i,j]]] if False else plt.cm.Reds(scaled[i,j]),edgecolors="0.4",linewidths=0.3)
        ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes,rotation=90,fontsize=7)
        ax.set_yticks(range(len(present))); ax.set_yticklabels(present,fontsize=8)
        ax.set_ylim(-0.5,len(present)-0.5); ax.set_xlim(-0.5,len(genes)-0.5); ax.invert_yaxis()
        ax.set_title(f"{s}  (per-sample marker dotplot)",fontsize=9)
        for sp in ax.spines.values(): sp.set_visible(False)
        sizes=[0.2,0.5,1.0]; h=[Line2D([0],[0],marker='o',linestyle='none',markersize=np.sqrt(8+f*230)/2,markerfacecolor='0.5',markeredgecolor='0.4',label=f"{int(f*100)}%") for f in sizes]
        ax.legend(handles=h,title="% expressing",loc='center left',bbox_to_anchor=(1.01,0.5),frameon=False,fontsize=7,title_fontsize=7)
        fig.savefig(f"{FIG}/{s}_dotplot.png",dpi=150,bbox_inches="tight"); plt.close(fig)
        print("dotplot",s,flush=True)
    print("PER-SAMPLE DOTPLOTS DONE ->",FIG,flush=True)

if __name__ == "__main__":
    main()
