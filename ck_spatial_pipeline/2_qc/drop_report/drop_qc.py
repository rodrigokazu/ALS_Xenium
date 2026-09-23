#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/drop_report/drop_qc.py
#
# Reports what the v3 drop does to depth: counts and genes per cell before and after removing
# niches v1_3, v1_7, v1_8 and the junk sub-clusters, with a content gate on the counts. The
# drop itself happens in 4_nichecompass/nc3_step1.py. No wrapper and no log exist, so it most
# likely ran interactively.
# ========================================================================================

"""QC for the v3 exclusion: does removing niche v1_3/7/8 and the junk sub-clusters
actually raise transcripts per cell, and by how much?

Computes counts and genes per cell from the counts layer, splits by exclusion set,
and reports before/after for the cohort and per section. Writes a 4-panel figure.
"""
import os, json, numpy as np, pandas as pd, h5py, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B   = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
NC  = f"{B}/NicheCompass_SC/runs_v2/integrated_whole/20260902_ncv2/model/NC_v2_integrated_whole_Leiden.h5ad"
AUD = f"{B}/NicheCompass_SC/cell_audit"
OUT = f"{B}/NicheCompass_SC/runs_v3_noN3"
os.makedirs(OUT, exist_ok=True)
DROP_NICHES=["3","7","8"]; JUNK_SUBS=[2,9,10]
log=lambda *a: print(*a,flush=True)
def cat(f,c):
    g=f["obs"][c]; cc=[x.decode() if isinstance(x,bytes) else str(x) for x in g["categories"][:]]
    return np.array(cc,dtype=object)[g["codes"][:]].astype(str)

log("loading + counting")
f=h5py.File(NC,"r")
niche=cat(f,"latent_leiden_0.3"); ns=cat(f,"sample"); ctm=cat(f,"cell_type_mn"); stat=cat(f,"status")
var=[x.decode() for x in f["var"][f["var"].attrs["_index"]][:]]
n=len(niche); G=len(var)
g=f["layers"]["counts"]; indptr=g["indptr"][:]
tot=np.zeros(n,dtype=np.int64); ngene=np.zeros(n,dtype=np.int32)
CH=200_000
for s0 in range(0,n,CH):
    e=min(s0+CH,n); a,b=int(indptr[s0]),int(indptr[e])
    M=sp.csr_matrix((g["data"][a:b],g["indices"][a:b],indptr[s0:e+1]-indptr[s0]),shape=(e-s0,G))
    tot[s0:e]=np.asarray(M.sum(1)).ravel(); ngene[s0:e]=np.asarray((M>0).sum(1)).ravel()
f.close()

on=pd.read_csv(f"{AUD}/otherneurons_cells.csv.gz"); ON=ctm=="Other neurons"
assert (on["counts"].values.astype(np.int64)==tot[ON]).all(), "counts content gate FAILED"
sub=np.full(n,-1,dtype=int); sub[ON]=on["sub"].values.astype(int)
m_niche=np.isin(niche,DROP_NICHES); m_junk=np.isin(sub,JUNK_SUBS); drop=m_niche|m_junk
keep=~drop
log(f"  {n:,} cells | dropped {int(drop.sum()):,} | retained {int(keep.sum()):,}")

def row(lab,m):
    return dict(set=lab, cells=int(m.sum()), pct=round(100*m.mean(),2),
                median_counts=float(np.median(tot[m])), mean_counts=round(float(tot[m].mean()),1),
                median_genes=float(np.median(ngene[m])),
                pct_lt15=round(100*float((tot[m]<15).mean()),1),
                pct_lt25=round(100*float((tot[m]<25).mean()),1))
rows=[row("COHORT (before)",np.ones(n,bool)),
      row("niche v1_3",niche=="3"), row("niche v1_7",niche=="7"), row("niche v1_8",niche=="8"),
      row("junk sub 2",sub==2), row("junk sub 9",sub==9), row("junk sub 10",sub==10),
      row("ALL DROPPED",drop), row("RETAINED (after)",keep)]
t=pd.DataFrame(rows); t.to_csv(f"{OUT}/drop_qc_table.csv",index=False)
log("\n"+t.to_string(index=False))

mb,ma=float(np.median(tot)),float(np.median(tot[keep]))
gb,ga=float(np.median(ngene)),float(np.median(ngene[keep]))
log(f"\nMEDIAN COUNTS  before {mb:.0f} -> after {ma:.0f}  ({ma-mb:+.0f}, {100*(ma-mb)/mb:+.1f}%)")
log(f"MEDIAN GENES   before {gb:.0f} -> after {ga:.0f}  ({ga-gb:+.0f}, {100*(ga-gb)/gb:+.1f}%)")
log(f"MEAN COUNTS    before {tot.mean():.1f} -> after {tot[keep].mean():.1f} "
    f"({100*(tot[keep].mean()-tot.mean())/tot.mean():+.1f}%)")
log(f"%<15 counts    before {100*(tot<15).mean():.1f}% -> after {100*(tot[keep]<15).mean():.1f}%")

per=[]
for s_ in sorted(set(ns)):
    m=ns==s_; k=m&keep
    per.append(dict(sample=s_,status=stat[m][0],n_before=int(m.sum()),n_after=int(k.sum()),
                    dropped=int((m&drop).sum()),pct_dropped=round(100*(m&drop).sum()/m.sum(),1),
                    med_before=float(np.median(tot[m])),med_after=float(np.median(tot[k])),
                    delta=float(np.median(tot[k])-np.median(tot[m]))))
pf=pd.DataFrame(per); pf.to_csv(f"{OUT}/drop_qc_per_sample.csv",index=False)
log("\n"+pf.to_string(index=False))

# ------------------------------------------------------------------ figure
fig,ax=plt.subplots(2,2,figsize=(14.5,9.4),facecolor="white")
C_B,C_A,C_D="#b9c2bf","#2f7d5c","#a63d3d"

a1=ax[0,0]
bins=np.linspace(0,200,81)
a1.hist(tot,bins=bins,color=C_B,label=f"before (n={n:,})",alpha=.95)
a1.hist(tot[keep],bins=bins,color=C_A,label=f"after (n={int(keep.sum()):,})",
        histtype="step",lw=2.1)
a1.hist(tot[drop],bins=bins,color=C_D,label=f"dropped (n={int(drop.sum()):,})",
        histtype="step",lw=1.8,ls="--")
a1.axvline(mb,color="#555",ls=":",lw=1.4); a1.axvline(ma,color=C_A,ls=":",lw=1.4)
a1.set_xlabel("transcripts per cell"); a1.set_ylabel("cells")
a1.set_title(f"Transcripts per cell — median {mb:.0f} → {ma:.0f} ({100*(ma-mb)/mb:+.1f}%)",
             fontsize=12.5,loc="left")
a1.legend(frameon=False,fontsize=10)
for s in ("top","right"): a1.spines[s].set_visible(False)

a2=ax[0,1]
labs=["niche\nv1_3","junk\nsub 2","junk\nsub 9","junk\nsub 10","ALL\nDROPPED","RETAINED","COHORT\nbefore"]
ms=[niche=="3",sub==2,sub==9,sub==10,drop,keep,np.ones(n,bool)]
vals=[float(np.median(tot[m])) for m in ms]
cols=[C_D]*5+[C_A,C_B]
bars=a2.bar(range(len(labs)),vals,0.65,color=cols)
for i,(v,m) in enumerate(zip(vals,ms)):
    a2.text(i,v+0.9,f"{v:.0f}\nn={int(m.sum()):,}",ha="center",fontsize=8.6)
a2.axhline(mb,color="#555",ls=":",lw=1.3)
a2.set_xticks(range(len(labs))); a2.set_xticklabels(labs,fontsize=9)
a2.set_ylabel("median transcripts per cell")
a2.set_ylim(0,max(vals)*1.25)
a2.set_title("The dropped sets are the shallow ones",fontsize=12.5,loc="left")
for s in ("top","right"): a2.spines[s].set_visible(False)

a3=ax[1,0]
x=np.arange(len(pf))
a3.bar(x-0.2,pf.med_before,0.4,color=C_B,label="before")
a3.bar(x+0.2,pf.med_after,0.4,color=C_A,label="after")
a3.set_xticks(x); a3.set_xticklabels(pf["sample"],rotation=90,fontsize=7.5)
a3.set_ylabel("median transcripts per cell")
_up=int((pf.delta>0).sum()); _fl=int((pf.delta==0).sum()); _dn=int((pf.delta<0).sum())
a3.set_title(f"Per section — {_up} improve, {_fl} unchanged, {_dn} falls",
             fontsize=12.5,loc="left")
a3.legend(frameon=False,fontsize=10)
for s in ("top","right"): a3.spines[s].set_visible(False)

a4=ax[1,1]
a4.bar(x,pf.pct_dropped,0.62,color=[("#D55E00" if s!="control" else "#0072B2") for s in pf.status])
a4.set_xticks(x); a4.set_xticklabels(pf["sample"],rotation=90,fontsize=7.5)
a4.set_ylabel("% of section dropped")
a4.axhline(100*drop.mean(),color="#111",ls="--",lw=1.2)
a4.text(len(pf)-0.5,100*drop.mean()+0.8,f"cohort {100*drop.mean():.1f}%",ha="right",fontsize=9)
a4.set_title("How much each section loses (blue = control, orange = ALS)",fontsize=12.5,loc="left")
for s in ("top","right"): a4.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(f"{OUT}/drop_qc.png",dpi=145,facecolor="white")
log("\nwrote drop_qc.png + drop_qc_table.csv + drop_qc_per_sample.csv")

json.dump({"median_counts_before":mb,"median_counts_after":ma,
           "median_genes_before":gb,"median_genes_after":ga,
           "mean_counts_before":round(float(tot.mean()),2),
           "mean_counts_after":round(float(tot[keep].mean()),2),
           "pct_lt15_before":round(100*float((tot<15).mean()),2),
           "pct_lt15_after":round(100*float((tot[keep]<15).mean()),2),
           "n_before":int(n),"n_after":int(keep.sum())},
          open(f"{OUT}/drop_qc.json","w"),indent=2)
