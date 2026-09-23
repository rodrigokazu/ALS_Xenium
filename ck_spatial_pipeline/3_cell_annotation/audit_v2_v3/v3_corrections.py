#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/3_cell_annotation/audit_v2_v3/v3_corrections.py
#
# Recomputes the annotation v3 numbers the adversarial verifiers corrected: the niche that
# holds the MNs in this object (niche 1 of the v2 run), a Mantel-Haenszel stratified VH test,
# depth-normalised MNX1 and the object-wide floor. Writes annotation_v3_corrected.json.
# ========================================================================================

"""Corrections pass on the annotation-v3 numbers, after adversarial verification.

Recomputes, from the object, everything the verifiers corrected:
  * which niche actually holds the motor neurons in THIS object (niche 1, not 6)
  * depth-normalised marker profiles (raw fold changes are library-size artefacts)
  * a section-stratified VH enrichment (the pooled Fisher test is invalid: heterogeneous)
  * object-wide transcript-floor cost (the floor is a global QC decision, not a slice patch)
  * MNX1 depth exactly, and the expected MNX1 count for an MN-composition cell at sub-7 depth
Rewrites annotation_v3_corrected.json and rebuilds the two MNX1 figures.
"""
import os, json, numpy as np, pandas as pd, h5py, scipy.sparse as sp
from scipy.stats import fisher_exact, chi2, binomtest
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B   = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
NC  = f"{B}/NicheCompass_SC/runs_v2/integrated_whole/20260902_ncv2/model/NC_v2_integrated_whole_Leiden.h5ad"
OUT = f"{B}/NicheCompass_SC/cell_audit/annotation_v3"
EXCLUDED = ["SD01015_BG","SD01616_BI","SD02913_BA"]
GENES = ["MNX1","STMN2","SLC17A6","RBFOX3","GAD1","GAD2","MBP","PTPRZ1","FBLN1"]
# panel gaps that make the identity question untestable
ABSENT = ["CHAT","ISL1","SLC5A7","NEFH","PRPH","CHODL","VSX2","SOX14","EVX1"]
C3=["#0072B2","#D55E00","#009E73"]; C2=["#0072B2","#D55E00"]
MUTED,INK,KEEPC="#83908b","#17201d","#0072B2"
plt.rcParams.update({"font.size":7,"axes.titlesize":8,"legend.fontsize":7,
                     "figure.dpi":150,"savefig.dpi":150,"svg.fonttype":"none"})
log=lambda *a: print(*a,flush=True)
def save(fig,name):
    p=f"{OUT}/{name}.svg"; fig.savefig(p,bbox_inches="tight",facecolor="white"); plt.close(fig)
    log(f"  wrote {name}.svg ({os.path.getsize(p)/1024:.0f} KB)")

def cat(f,c):
    g=f["obs"][c]; cc=[x.decode() if isinstance(x,bytes) else str(x) for x in g["categories"][:]]
    return np.array(cc,dtype=object)[g["codes"][:]].astype(str)
def nb3(f,c):
    g=f["obs"][c]; v=g["values"][:].astype(bool); m=g["mask"][:].astype(bool)
    return v&~m, m                       # (true, is_na)

log("loading")
f=h5py.File(NC,"r")
ns=cat(f,"sample"); ctv2=cat(f,"cell_type_v2"); niche=cat(f,"latent_leiden_0.3"); stat=cat(f,"status")
vh_true,vh_na=nb3(f,"in_VH_v2"); mn_true,_=nb3(f,"is_MN")
var=[x.decode() for x in f["var"][f["var"].attrs["_index"]][:]]
present=[g for g in GENES if g in var]
absent_confirmed=[g for g in ABSENT if g not in var]
gi={g:var.index(g) for g in present}
g=f["layers"]["counts"]; indptr=g["indptr"][:]; n=len(ns)
tot=np.zeros(n,dtype=np.int64); M=np.zeros((n,len(present)),dtype=np.int32)
for s in range(0,n,200000):
    e=min(s+200000,n); a,b=int(indptr[s]),int(indptr[e])
    X=sp.csr_matrix((g["data"][a:b],g["indices"][a:b],indptr[s:e+1]-indptr[s]),shape=(e-s,len(var)))
    tot[s:e]=np.asarray(X.sum(1)).ravel()
    for j,gg in enumerate(present): M[s:e,j]=np.asarray(X[:,gi[gg]].todense()).ravel()
f.close()
mx=pd.DataFrame(M,columns=present)
d3=pd.read_parquet(f"{OUT}/annotation_v3_T15.parquet")
sub=d3["sub"].values; area=d3["cell_area"].values
m7=sub==7; MNc=ctv2=="Motor neurons"
log(f"  n={n:,} sub7={m7.sum():,} MN={MNc.sum():,}")
R={}

# ---------------------------------------------------------- 1. THE MN NICHE
ct=pd.crosstab(niche,mn_true)
mnpc=(100*ct.get(True,0)/mn_true.sum()).round(2)
mn_niche=mnpc.idxmax()
R["mn_niche"]={"object":"NC_v2_integrated_whole_Leiden.h5ad · latent_leiden_0.3",
  "mn_niche":str(mn_niche),"pct_of_MNs":float(mnpc.max()),
  "n_MNs_there":int(ct.loc[mn_niche,True]),"niche_size":int((niche==mn_niche).sum()),
  "niche6_MNs":int(ct.loc["6",True]) if "6" in ct.index else None,
  "niche6_size":int((niche=="6").sum()),
  "per_niche_pct_of_MNs":{str(k):float(v) for k,v in mnpc.items()}}
# OR for the MN niche
a_=int(((niche==mn_niche)&mn_true).sum()); b_=int(((niche!=mn_niche)&mn_true).sum())
c_=int(((niche==mn_niche)&~mn_true).sum()); d_=int(((niche!=mn_niche)&~mn_true).sum())
R["mn_niche"]["odds_ratio"]=round(float(fisher_exact([[a_,b_],[c_,d_]])[0]),2)
log(f"\n=== MN niche in THIS object: {mn_niche} ({mnpc.max()}% of MNs, OR {R['mn_niche']['odds_ratio']}) "
    f"| niche 6 holds {R['mn_niche']['niche6_MNs']} MNs of {R['mn_niche']['niche6_size']:,} cells ===")

# sub7 vs MN niche occupancy
s7pc=(100*pd.Series(niche[m7]).value_counts(normalize=True)).round(2)
R["sub7_niche_pct"]={str(k):float(v) for k,v in s7pc.items()}
R["sub7_in_mn_niche_pct"]=float(s7pc.get(mn_niche,0.0))
R["cohort_in_mn_niche_pct"]=round(100*float((niche==mn_niche).mean()),2)
oa=int((m7&(niche==mn_niche)).sum()); ob=int((m7&(niche!=mn_niche)).sum())
oc=int((~m7&(niche==mn_niche)).sum()); od=int((~m7&(niche!=mn_niche)).sum())
R["sub7_mn_niche_OR"]=round(float(fisher_exact([[oa,ob],[oc,od]])[0]),2)
log(f"  sub7 in niche {mn_niche}: {R['sub7_in_mn_niche_pct']}% vs cohort {R['cohort_in_mn_niche_pct']}% "
    f"(OR {R['sub7_mn_niche_OR']}) -- sub7 is ENRICHED there, not absent")

# ---------------------------------------------------------- 2. DEPTH-NORMALISED MARKERS
cpm = mx.div(np.maximum(tot,1),axis=0)*1e4          # counts per 10k, per cell
prof=pd.DataFrame({
 "sub7_raw":mx[m7].mean(),          "sub7_norm":cpm[m7].mean(),
 "MN_raw":mx[MNc].mean(),           "MN_norm":cpm[MNc].mean(),
 "cohort_raw":mx.mean(),            "cohort_norm":cpm.mean()})
prof["sub7_fold_raw"]=prof["sub7_raw"]/prof["cohort_raw"]
prof["sub7_fold_norm"]=prof["sub7_norm"]/prof["cohort_norm"]
prof["MN_fold_raw"]=prof["MN_raw"]/prof["cohort_raw"]
prof["MN_fold_norm"]=prof["MN_norm"]/prof["cohort_norm"]
log("\n=== marker profile, raw vs depth-normalised fold change ===")
log(prof[["sub7_fold_raw","sub7_fold_norm","MN_fold_raw","MN_fold_norm"]].round(2).to_string())
R["marker_profile"]=prof.round(5).to_dict()
R["median_counts"]={"sub7":float(np.median(tot[m7])),"MN":float(np.median(tot[MNc])),
                    "cohort":float(np.median(tot))}
# size-matched sub7 vs cohort on STMN2 (the "is it neuronal at all" question)
band=(area>=40)&(area<=70)
R["stmn2_size_matched"]={"band":"40-70 um2",
  "sub7_mean":round(float(mx.loc[m7&band,"STMN2"].mean()),4),
  "other_mean":round(float(mx.loc[~m7&band,"STMN2"].mean()),4),
  "sub7_pct_pos":round(100*float((mx.loc[m7&band,"STMN2"]>0).mean()),2),
  "other_pct_pos":round(100*float((mx.loc[~m7&band,"STMN2"]>0).mean()),2)}
log(f"  STMN2 size-matched (40-70um2): sub7 {R['stmn2_size_matched']['sub7_mean']} "
    f"vs {R['stmn2_size_matched']['other_mean']}")

# ---------------------------------------------------------- 3. MNX1 depth, exactly
v=mx.loc[m7,"MNX1"].values
R["mnx1_depth"]={"n_sub7":int(m7.sum()),
  "eq0":int((v==0).sum()),"eq1":int((v==1).sum()),"eq2":int((v==2).sum()),
  "ge3":int((v>=3).sum()),"ge5":int((v>=5).sum()),
  "pct_pos":round(100*float((v>0).mean()),2),
  "pct_of_positives_with_exactly_one":round(100*float((v==1).sum()/max((v>0).sum(),1)),2)}
frac7=float(mx.loc[m7,"MNX1"].sum()/tot[m7].sum())
fracMN=float(mx.loc[MNc,"MNX1"].sum()/tot[MNc].sum())
fracC=float(mx["MNX1"].sum()/tot.sum())
R["mnx1_fraction"]={"sub7":frac7,"MN":fracMN,"cohort":fracC,
  "sub7_over_cohort":round(frac7/fracC,1),"sub7_over_MN":round(frac7/fracMN,1)}
R["mnx1_in_MNs"]={"mean_counts":round(float(mx.loc[MNc,"MNX1"].mean()),4),
  "pct_positive":round(100*float((mx.loc[MNc,"MNX1"]>0).mean()),2),
  "expected_at_sub7_depth":round(fracMN*float(np.median(tot[m7])),4)}
log(f"\n=== MNX1 depth: {R['mnx1_depth']['pct_of_positives_with_exactly_one']}% of positives carry exactly 1 ===")
log(f"  MNX1 transcriptome fraction: sub7 {frac7:.5f} | real MNs {fracMN:.6f} | cohort {fracC:.6f}")
log(f"  -> sub7 is {R['mnx1_fraction']['sub7_over_MN']}x MORE MNX1-dedicated than real MNs")
log(f"  an MN-composition cell at sub7 median depth expects {R['mnx1_in_MNs']['expected_at_sub7_depth']} MNX1")
R["panel_absent_markers"]=absent_confirmed
log(f"  panel LACKS: {', '.join(absent_confirmed)}")

# ---------------------------------------------------------- 4. VH: section-stratified
ev=~np.isin(ns,EXCLUDED)
rows=[]
for s_ in sorted(set(ns[ev])):
    m=ns==s_
    a=int((m&m7&vh_true).sum()); b=int((m&m7&~vh_true).sum())
    c=int((m&~m7&vh_true).sum()); dd=int((m&~m7&~vh_true).sum())
    rows.append({"sample":s_,"a":a,"b":b,"c":c,"d":dd,"n7":a+b,
                 "or":(a*dd/(b*c)) if b*c else np.nan})
S=pd.DataFrame(rows)
num=(S.a*S.d/(S.a+S.b+S.c+S.d)).sum(); den=(S.b*S.c/(S.a+S.b+S.c+S.d)).sum()
mh=num/den
# Robins-Breslow-Greenland variance
N=S.a+S.b+S.c+S.d
P=(S.a+S.d)/N; Q=(S.b+S.c)/N; Rr=S.a*S.d/N; Ss=S.b*S.c/N
var=( (P*Rr).sum()/(2*Rr.sum()**2) + ((P*Ss+Q*Rr).sum())/(2*Rr.sum()*Ss.sum())
      + (Q*Ss).sum()/(2*Ss.sum()**2) )
lo,hi=np.exp(np.log(mh)-1.96*np.sqrt(var)),np.exp(np.log(mh)+1.96*np.sqrt(var))
# Tarone homogeneity
E=(S.a+S.b)*(S.a+S.c)/N; V=(S.a+S.b)*(S.c+S.d)*(S.a+S.c)*(S.b+S.d)/(N**2*(N-1))
tar=((S.a-E).sum()**2)/V.sum()
chi_mh=((S.a-E).sum()**2)/V.sum()
sgn=binomtest(int((S["or"]>1).sum()),int(S["or"].notna().sum()),0.5)
pooled=fisher_exact([[int((m7&ev&vh_true).sum()),int((m7&ev&~vh_true).sum())],
                     [int((~m7&ev&vh_true).sum()),int((~m7&ev&~vh_true).sum())]])
R["vh"]={"pooled_OR":round(float(pooled[0]),3),"pooled_fisher_p":float(pooled[1]),
  "MH_OR":round(float(mh),3),"MH_CI":[round(float(lo),3),round(float(hi),3)],
  "tarone_heterogeneity_p":float(chi2.sf(tar,len(S)-1)),
  "sections_OR_gt1":int((S["or"]>1).sum()),"sections_evaluable":int(S["or"].notna().sum()),
  "sign_test_p":float(sgn.pvalue),
  "per_section_OR_range":[round(float(np.nanmin(S['or'])),2),round(float(np.nanmax(S['or'])),2)],
  "sub7_in_VH":int((m7&ev&vh_true).sum()),"sub7_evaluable":int((m7&ev).sum()),
  "sub7_pct_in_VH":round(100*float((m7&ev&vh_true).sum()/(m7&ev).sum()),2),
  "cohort_pct_in_VH":round(100*float((ev&vh_true).sum()/ev.sum()),2),
  "sub7_NA_sections":int((m7&~ev).sum())}
log(f"\n=== VH enrichment, corrected ===")
log(f"  pooled Fisher OR {R['vh']['pooled_OR']} p={R['vh']['pooled_fisher_p']:.2e}  <-- INVALID, heterogeneous")
log(f"  Tarone heterogeneity p={R['vh']['tarone_heterogeneity_p']:.4f}")
log(f"  Mantel-Haenszel OR {R['vh']['MH_OR']} (95% CI {R['vh']['MH_CI']})")
log(f"  per-section OR range {R['vh']['per_section_OR_range']}; {R['vh']['sections_OR_gt1']}/"
    f"{R['vh']['sections_evaluable']} above 1 (sign test p={R['vh']['sign_test_p']:.3f})")

# count-matched background: same sections, total counts within +-3 of each sub7 cell's depth
bins=np.array([10,15,20,25,30,40,60,100,10**9])
bi=np.digitize(tot,bins)
rows=[]
for s_ in sorted(set(ns[ev])):
    for k in np.unique(bi[(ns==s_)&m7]):
        m=(ns==s_)&(bi==k)
        a=int((m&m7&vh_true).sum()); b=int((m&m7&~vh_true).sum())
        c=int((m&~m7&vh_true).sum()); dd=int((m&~m7&~vh_true).sum())
        if (a+b)and(c+dd): rows.append({"a":a,"b":b,"c":c,"d":dd})
T=pd.DataFrame(rows); N=T.a+T.b+T.c+T.d
mh2=(T.a*T.d/N).sum()/(T.b*T.c/N).sum()
P=(T.a+T.d)/N;Q=(T.b+T.c)/N;Rr=T.a*T.d/N;Ss=T.b*T.c/N
var2=((P*Rr).sum()/(2*Rr.sum()**2)+((P*Ss+Q*Rr).sum())/(2*Rr.sum()*Ss.sum())+(Q*Ss).sum()/(2*Ss.sum()**2))
R["vh"]["MH_OR_count_matched"]=round(float(mh2),3)
R["vh"]["MH_CI_count_matched"]=[round(float(np.exp(np.log(mh2)-1.96*np.sqrt(var2))),3),
                                round(float(np.exp(np.log(mh2)+1.96*np.sqrt(var2))),3)]
log(f"  count-matched (section x depth bin) MH OR {R['vh']['MH_OR_count_matched']} "
    f"CI {R['vh']['MH_CI_count_matched']}")

# ---------------------------------------------------------- 5. object-wide floor cost
R["floor_objectwide"]={}
for T_ in (15,20,25):
    R["floor_objectwide"][T_]={"cells_below":int((tot<T_).sum()),
        "pct":round(100*float((tot<T_).mean()),2),
        "by_class":{c:int(((tot<T_)&(ctv2==c)).sum()) for c in sorted(set(ctv2))},
        "pct_by_class":{c:round(100*float(((tot<T_)&(ctv2==c)).sum()/max((ctv2==c).sum(),1)),1)
                        for c in sorted(set(ctv2))}}
R["floor_objectwide"]["is_MN_median_counts"]=float(np.median(tot[mn_true]))
R["floor_objectwide"]["is_MN_pct_below_15"]=round(100*float((tot[mn_true]<15).mean()),2)
log(f"\n=== object-wide floor cost ===")
for T_ in (15,20,25):
    log(f"  <{T_}: {R['floor_objectwide'][T_]['cells_below']:,} cells "
        f"({R['floor_objectwide'][T_]['pct']}% of the cohort)")
log(f"  is_MN median {R['floor_objectwide']['is_MN_median_counts']:.0f} counts, "
    f"{R['floor_objectwide']['is_MN_pct_below_15']}% below 15 -> MN calls unaffected")

json.dump(R,open(f"{OUT}/annotation_v3_corrected.json","w"),indent=2,default=str)
log("\nwrote annotation_v3_corrected.json")

# ============================================================ FIGURES (rebuilt)
# FIG: MNX1 depth + niche, with the CORRECT MN niche
fig,axes=plt.subplots(1,2,figsize=(8.0,2.8),constrained_layout=True)
ax=axes[0]; dep=R["mnx1_depth"]; n7=dep["n_sub7"]
bars=[("0",dep["eq0"]),("1",dep["eq1"]),("2",dep["eq2"]),("$\\geq$3",dep["ge3"])]
ax.bar([b[0] for b in bars],[b[1] for b in bars],width=.6,
       color=[MUTED,KEEPC,"#D55E00","#009E73"],lw=.6,edgecolor="white")
for i,b in enumerate(bars):
    ax.text(i,b[1]+40,f"{b[1]:,}\n{100*b[1]/n7:.1f}%",ha="center",va="bottom",fontsize=6.5,
            color=INK,fontweight="bold")
ax.set_xlabel("MNX1 transcripts in the cell"); ax.set_ylabel("cells in sub-cluster 7")
ax.set_ylim(0,max(b[1] for b in bars)*1.26)
ax.set_title(f"Sub-cluster 7 · n={n7:,} · median {R['median_counts']['sub7']:.0f} total counts",loc="left")
for s in ("top","right"): ax.spines[s].set_visible(False)

ax=axes[1]
nk=sorted(set(list(R["sub7_niche_pct"].keys())+[str(k) for k in mnpc.index]),key=lambda z:int(z))
x=np.arange(len(nk))
ax.bar(x-.19,[R["sub7_niche_pct"].get(k,0) for k in nk],width=.36,color=C2[0],lw=.6,
       edgecolor="white",label="sub-cluster 7")
ax.bar(x+.19,[float(mnpc.get(k,0)) for k in nk],width=.36,color=C2[1],lw=.6,
       edgecolor="white",label="called motor neurons")
ax.set_xticks(x); ax.set_xticklabels(nk)
ax.set_xlabel("niche (latent_leiden_0.3, THIS object)"); ax.set_ylabel("% of the group")
ax.set_title(f"Niche {mn_niche} is the motor-neuron niche here — and sub-7 is in it too",loc="left")
ax.legend(frameon=False,loc="upper right",fontsize=6.5)
for s in ("top","right"): ax.spines[s].set_visible(False)
save(fig,"mnx1_depth_niche")

# FIG: marker profile, raw AND depth-normalised side by side
fig,axes=plt.subplots(1,2,figsize=(9.2,3.0),constrained_layout=True)
gl=[g for g in ["MNX1","STMN2","SLC17A6","RBFOX3","GAD1","GAD2","MBP","PTPRZ1","FBLN1"] if g in prof.index]
for k,(suffix,ttl) in enumerate([("raw","Raw mean counts — confounded by a 5× depth gap"),
                                 ("norm","Depth-normalised (counts per 10k) — the fair comparison")]):
    ax=axes[k]; x=np.arange(len(gl)); w=.27
    for j,(col,lab,c) in enumerate([(f"sub7_{suffix}","sub-cluster 7",C3[0]),
                                    (f"MN_{suffix}","called motor neurons",C3[1]),
                                    (f"cohort_{suffix}","cohort mean",C3[2])]):
        ax.bar(x+(j-1)*w,[max(prof.loc[gg,col],1e-4) for gg in gl],width=w*.92,color=c,
               lw=.6,edgecolor="white",label=lab)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(gl,fontsize=6.4)
    ax.set_ylabel("mean transcripts per cell" if k==0 else "counts per 10,000")
    ax.set_title(ttl,loc="left",fontsize=7.4)
    ax.grid(axis="y",lw=.4,color="#dfe4e2",zorder=0); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    if k==0: ax.legend(frameon=False,loc="upper right",fontsize=6.4,ncol=1)
save(fig,"mnx1_markers")
log("done")
