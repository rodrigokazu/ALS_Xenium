#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/nc3_split_inventory.py
#
# Information only. Lists the splits available on each v2 niche and what a re-cluster would
# give. It changes no labels and retrains nothing.
# ========================================================================================

"""What splits are available on the v2 niches, and what would a re-cluster give?

INFORMATION ONLY -- writes tables, changes no labels and retrains no model.

Two distinct levers, which are often confused:

  A. SUB-SPLIT one niche. Leiden on that niche's own latent sub-graph. Local, cheap,
     preserves every other label, and writes v2_<k>.<j>. Already done for v2_1/2/4;
     this run completes the inventory over all seven.

  B. RE-CLUSTER the whole object at a different Leiden resolution. This is the lever
     that changes how many niches exist. It does NOT require retraining NicheCompass:
     the model, the latent and the neighbour graph are unchanged, only the clustering
     of that latent. It DOES renumber everything.

A third option, a full NicheCompass retrain, is NOT run here. It is only needed when
the set of CELLS changes, because that is what rebuilds the neighbourhood graph.

Gates (same as the v2_1 split): size floor 2% of the parent, silhouette >= 0.10,
composition JS >= 0.25 between the most separated pair, >= 1.5x on at least one panel
measured as an absolute % of transcriptome (never z-scored within the partition).
"""
import os, json, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score

B   = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
OBJ = f"{B}/NicheCompass_SC/runs_v3_noN3/NC_v3_Leiden_nichev2sub.h5ad"
OUT = f"{B}/NicheCompass_SC/runs_v3_noN3/split_inventory"
os.makedirs(OUT, exist_ok=True)
RES_SUB  = [0.05, 0.10, 0.20, 0.30, 0.50]
RES_WHOLE= [0.20, 0.30, 0.40, 0.50, 0.70, 1.00]
MIN_PCT, SIL_MIN, JS_MIN, FC_MIN, SEED = 2.0, 0.10, 0.25, 1.50, 0
log = lambda *a: print(*a, flush=True)

PANELS = {
 "vascular":["VWF","CLDN5","PECAM1","FLT1","EPAS1","CAV1","RGS5","PDGFRB","ACTA2","TAGLN"],
 "pia / fibroblast / VLMC":["FBLN1","CEMIP","SFRP2","COL1A1","COL1A2","DCN","LUM","SLC6A13","PTGDS"],
 "ependymal":["ZBBX","CCDC146","WDR49","EYA4","FOXJ1","PIFO","TMEM212"],
 "immune":["C1QC","CD163","PTPRC","MRC1","LYVE1"],
 "white matter / oligo":["MOBP","PLP1","MBP","MAG","MOG","CNP","ST18"],
 "grey matter / neuron":["SNAP25","RBFOX3","SYT1","GAD1","GAD2","SLC17A6","CHAT","MNX1"],
 "astrocyte":["AQP4","SLC1A2","GFAP","GJA1","SLC1A3"],
 "OPC":["PDGFRA","OLIG1","OLIG2","PTPRZ1","MEGF11","CSPG4"],
}

log("loading")
A = sc.read_h5ad(OBJ)
A.obs["niche_v2"] = A.obs["niche_v2"].astype(str)
var = list(A.var_names)
gidx = {k: np.array([var.index(g) for g in v if g in var]) for k, v in PANELS.items()}
gidx = {k: v for k, v in gidx.items() if len(v)}
NICHES = sorted(A.obs["niche_v2"].unique())
log(f"{A.n_obs:,} cells, niches {NICHES}")

def panel_pct(ad, groups):
    X = ad.layers["counts"].tocsc()
    tot = np.maximum(np.asarray(ad.layers["counts"].sum(1)).ravel(), 1)
    return pd.DataFrame({k: pd.Series(100.0*np.asarray(X[:, i].sum(1)).ravel()/tot).groupby(groups).mean()
                         for k, i in gidx.items()})

def score(ad, lab, L, samp, n):
    vc = pd.Series(lab).value_counts()
    if len(vc) < 2: return None
    small = 100*vc.min()/n
    sil = float(silhouette_score(L[samp], lab[samp]))
    ct = pd.crosstab(lab, ad.obs["cell_type_v2"].astype(str))
    ctn = ct.div(ct.sum(1), axis=0)
    js = max(float(jensenshannon(ctn.loc[i], ctn.loc[j], base=2))
             for ii, i in enumerate(ctn.index) for j in ctn.index[ii+1:])
    PP = panel_pct(ad, lab)
    fc = float((PP.max(0)/np.maximum(PP.min(0), 1e-6)).max())
    ok = (small >= MIN_PCT) and (sil >= SIL_MIN) and (js >= JS_MIN) and (fc >= FC_MIN)
    return dict(n_sub=len(vc), smallest_pct=round(small,2), silhouette=round(sil,4),
                max_js=round(js,4), max_panel_fc=round(fc,3),
                size_floor_ok=bool(small>=MIN_PCT), pass_all=bool(ok))

# ---------------------------------------------------------------- A. per niche
log("\n" + "="*74 + "\nA. SUB-SPLIT INVENTORY — every v2 niche, resolutions 0.05-0.5\n" + "="*74)
rowsA = []
for k in NICHES:
    a = A[A.obs["niche_v2"] == k].copy()
    n = a.n_obs
    log(f"\n{k}: {n:,} cells (size floor {MIN_PCT}% = {int(n*MIN_PCT/100):,})")
    sc.pp.neighbors(a, n_neighbors=15, use_rep="nichecompass_latent", random_state=SEED)
    L = a.obsm["nichecompass_latent"]
    rng = np.random.default_rng(SEED)
    samp = rng.choice(n, size=min(20000, n), replace=False)
    for r in RES_SUB:
        key = f"s{r}"
        sc.tl.leiden(a, resolution=r, key_added=key, random_state=SEED,
                     flavor="igraph", n_iterations=2, directed=False)
        lab = a.obs[key].astype(str).values
        s = score(a, lab, L, samp, n)
        if s is None:
            rowsA.append(dict(niche=k, cells=n, res=r, n_sub=1, pass_all=False))
            log(f"  res {r}: 1 cluster"); continue
        rowsA.append(dict(niche=k, cells=n, res=r, **s))
        log(f"  res {r}: {s['n_sub']} sub | smallest {s['smallest_pct']:5.2f}% "
            f"{'OK ' if s['size_floor_ok'] else 'FLOOR'} | sil {s['silhouette']:.3f} | "
            f"maxJS {s['max_js']:.3f} | FC {s['max_panel_fc']:.2f}x | "
            f"{'PASS' if s['pass_all'] else 'fail'}")
    del a
RA = pd.DataFrame(rowsA); RA.to_csv(f"{OUT}/subsplit_inventory.csv", index=False)

log("\n--- per-niche verdict ---")
sumA = []
for k in NICHES:
    g = RA[(RA.niche == k) & RA.pass_all.fillna(False)]
    best = g.sort_values("silhouette", ascending=False).iloc[0] if len(g) else None
    allk = RA[RA.niche == k]
    sumA.append(dict(niche=k, cells=int(allk.cells.iloc[0]),
                     splittable=bool(len(g)),
                     best_res=float(best.res) if best is not None else None,
                     n_sub=int(best.n_sub) if best is not None else None,
                     silhouette=float(best.silhouette) if best is not None else None,
                     best_sil_any=float(allk.silhouette.max(skipna=True)) if allk.silhouette.notna().any() else None,
                     why_not=("" if best is not None else
                              ("all partitions below the size floor"
                               if not allk.size_floor_ok.fillna(False).any() else
                               "gates not cleared jointly"))))
SA = pd.DataFrame(sumA); SA.to_csv(f"{OUT}/subsplit_verdict.csv", index=False)
log(SA.to_string(index=False))

# ------------------------------------------------- B. whole-object resolution
log("\n" + "="*74 + "\nB. WHOLE-OBJECT RE-CLUSTER — same latent, same graph, no model retrain\n" + "="*74)
assert "connectivities" in A.obsp, "no neighbour graph on the object"
log("reusing the object's existing neighbour graph (obsp/connectivities)")
rng = np.random.default_rng(SEED)
sampW = rng.choice(A.n_obs, size=30000, replace=False)
LW = A.obsm["nichecompass_latent"]
rowsB = []
for r in RES_WHOLE:
    key = f"w{r}"
    sc.tl.leiden(A, resolution=r, key_added=key, random_state=SEED,
                 flavor="igraph", n_iterations=2, directed=False)
    lab = A.obs[key].astype(str).values
    vc = pd.Series(lab).value_counts()
    sil = float(silhouette_score(LW[sampW], lab[sampW]))
    ct = pd.crosstab(lab, A.obs["cell_type_v2"].astype(str))
    ctn = ct.div(ct.sum(1), axis=0)
    purity = float(ctn.max(1).mean())
    # how many of the new clusters are specks
    specks = int((100*vc/A.n_obs < 1.0).sum())
    # agreement with the current v2 labels
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = float(adjusted_rand_score(A.obs["niche_v2"].values, lab))
    nmi = float(normalized_mutual_info_score(A.obs["niche_v2"].values, lab))
    mn = A.obs["is_MN_v2"].map({True:1, False:0}).fillna(0).astype(int).values
    top_mn = pd.Series(lab[mn==1]).value_counts()
    mn_conc = float(top_mn.iloc[0]/max(mn.sum(),1))
    rowsB.append(dict(resolution=r, n_niches=len(vc),
                      smallest_pct=round(100*vc.min()/A.n_obs, 3),
                      largest_pct=round(100*vc.max()/A.n_obs, 2),
                      n_below_1pct=specks, silhouette=round(sil, 4),
                      mean_top_celltype=round(purity, 4),
                      ARI_vs_v2=round(ari, 4), NMI_vs_v2=round(nmi, 4),
                      MN_in_top_niche=round(100*mn_conc, 1)))
    log(f"  res {r}: {len(vc):>2} niches | largest {100*vc.max()/A.n_obs:5.2f}% | "
        f"smallest {100*vc.min()/A.n_obs:5.3f}% | {specks} under 1% | sil {sil:.3f} | "
        f"purity {purity:.3f} | ARI vs v2 {ari:.3f} | MN concentration {100*mn_conc:.1f}%")
    pd.crosstab(A.obs["niche_v2"], lab).to_csv(f"{OUT}/whole_res{r}_vs_v2.csv")
RB = pd.DataFrame(rowsB); RB.to_csv(f"{OUT}/whole_resolution_sweep.csv", index=False)
log("\n" + RB.to_string(index=False))

json.dump({"note": "information only; no labels written, no model retrained",
           "gates": dict(min_pct=MIN_PCT, silhouette=SIL_MIN, max_js=JS_MIN, panel_fc=FC_MIN),
           "current": dict(resolution=0.3, n_niches=len(NICHES)),
           "subsplit": SA.to_dict("records"),
           "whole_resolution": RB.to_dict("records")},
          open(f"{OUT}/split_inventory.json", "w"), indent=2)
log(f"\nwrote {OUT}/")
