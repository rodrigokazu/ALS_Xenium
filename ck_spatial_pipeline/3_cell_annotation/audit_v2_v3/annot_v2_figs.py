#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/3_cell_annotation/audit_v2_v3/annot_v2_figs.py
#
# Figures for annotation v2: class totals, per-sample composition and per-niche composition.
# ========================================================================================

"""Annotation v2 figures: class totals, per-sample composition, per-niche composition."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B   = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
OUT = f"{B}/NicheCompass_SC/cell_audit/annotation_v2"
R = json.load(open(f"{OUT}/annotation_v2.json"))

# stacking order validated with the dataviz palette checker: this order is the one that
# keeps every ADJACENT pair above the CVD and normal-vision floors (worst adjacent
# dE 11.0 deutan / 19.2 normal). The artifact's own class colours are unchanged.
ORDER = ["Oligodendrocytes","Microglia","Astrocytes","Endothelial","Excitatory neurons",
         "OPCs","Fibroblast","Inhibitory neurons","Macrophages","Other neurons","Motor neurons"]
COL = {"Oligodendrocytes":"#0072B2","Microglia":"#a8701f","Astrocytes":"#7B52AB",
       "Endothelial":"#009E73","Excitatory neurons":"#D55E00","OPCs":"#56B4E9",
       "Fibroblast":"#882255","Inhibitory neurons":"#E69F00","Macrophages":"#CC79A7",
       "Other neurons":"#a63d3d","Motor neurons":"#000000"}
MOVED = {"Oligodendrocytes","Microglia","Astrocytes","Endothelial","Fibroblast"}
plt.rcParams.update({"font.size":7,"axes.titlesize":8,"legend.fontsize":7,
                     "figure.dpi":150,"savefig.dpi":150,"svg.fonttype":"none"})

def save(fig,name):
    p=f"{OUT}/{name}.svg"; fig.savefig(p,bbox_inches="tight",facecolor="white"); plt.close(fig)
    print(f"  wrote {name}.svg ({os.path.getsize(p)/1024:.0f} KB)",flush=True)

tot = pd.DataFrame(R["totals"])
bs  = pd.DataFrame(R["by_sample_pc"]).reindex(columns=ORDER)
bsc = pd.DataFrame(R["by_sample_counts"]).reindex(columns=ORDER)
bn  = pd.DataFrame(R["by_niche_pc"]).reindex(columns=ORDER)
status = R["sample_status"]
SORD = {"control":0,"sporadic":1,"c9":2}
samples = sorted(bs.index, key=lambda s:(SORD.get(status.get(s,"z"),9), s))
niches  = sorted(bn.index, key=lambda x:int(x))

# ---- FIG 1: what moved, v1 -> v2
fig,ax = plt.subplots(figsize=(7.4,3.2),constrained_layout=True)
o=[c for c in ORDER if tot.loc[c,"v1"]>0 or tot.loc[c,"v2"]>0]
y=np.arange(len(o))
ax.barh(y+.2,[tot.loc[c,"v1"] for c in o],height=.38,color=[COL[c] for c in o],alpha=.42,label="v1")
ax.barh(y-.2,[tot.loc[c,"v2"] for c in o],height=.38,color=[COL[c] for c in o],label="v2")
for i,c in enumerate(o):
    d=int(tot.loc[c,"delta"])
    if d: ax.text(max(tot.loc[c,"v1"],tot.loc[c,"v2"])*1.02,i,f"{d:+,}",va="center",fontsize=6.5,
                  color="#17201d",fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(o); ax.invert_yaxis()
ax.set_xlabel("cells"); ax.set_xscale("log")
ax.legend(handles=[Line2D([],[],marker="s",ls="",ms=6,mfc="#777",alpha=.42,mec="none",label="v1 (cell_type_mn)"),
                   Line2D([],[],marker="s",ls="",ms=6,mfc="#777",mec="none",label="v2 (cell_type_v2)")],
          frameon=False,loc="lower right")
for s in ("top","right"): ax.spines[s].set_visible(False)
save(fig,"v2_totals")

def stacked(ax, idx, frame, labels, gap=0.6):
    bot=np.zeros(len(idx))
    for c in ORDER:
        v=frame.loc[idx,c].values.astype(float)
        ax.bar(np.arange(len(idx)),v,bottom=bot,width=.8,color=COL[c],lw=gap,
               edgecolor="white",label=c)
        bot+=v
    ax.set_xticks(np.arange(len(idx))); ax.set_xticklabels(labels,rotation=90,fontsize=6.4)
    ax.set_ylim(0,100); ax.set_ylabel("% of cells")
    for s in ("top","right"): ax.spines[s].set_visible(False)

# ---- FIG 2: per sample
fig,ax = plt.subplots(figsize=(9.6,4.0),constrained_layout=True)
stacked(ax,samples,bs,[f"{s}\n{status.get(s,'?')}" for s in samples])
ax.legend(handles=[Line2D([],[],marker="s",ls="",ms=6,mfc=COL[c],mec="none",
                          label=c+(" ←new" if c=="Fibroblast" else ""))for c in ORDER],
          frameon=False,ncol=6,fontsize=6.6,loc="upper center",bbox_to_anchor=(.5,-.30))
save(fig,"v2_by_sample")

# ---- FIG 3: per niche
fig,ax = plt.subplots(figsize=(7.0,3.8),constrained_layout=True)
nl=[f"niche {n}\n{int(pd.DataFrame(R['by_niche_counts']).reindex(columns=ORDER).loc[n].sum()):,}" for n in niches]
stacked(ax,niches,bn,nl)
ax.legend(handles=[Line2D([],[],marker="s",ls="",ms=6,mfc=COL[c],mec="none",label=c) for c in ORDER],
          frameon=False,ncol=4,fontsize=6.6,loc="upper center",bbox_to_anchor=(.5,-.24))
save(fig,"v2_by_niche")
print("done")
