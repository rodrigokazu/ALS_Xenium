#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/nc3_step3b_subcluster.py
#
# Tests for real sub-structure inside v2_1, v2_2 and v2_4 with Leiden at 0.05 to 0.5 on the
# latent. A split must clear four gates: every part at least 2% of the parent, silhouette
# 0.10, Jensen-Shannon divergence 0.25 on cell_type_v2 and a panel fold change of 1.5. Only
# v2_1 splits, at resolution 0.2, into three.
#
# This rewrite replaced a first pass with two broken gates. It writes the canonical object
# NC_v3_Leiden_nichev2sub.h5ad (965,285 cells).
# ========================================================================================

"""Step 3 (rewrite): does real, non-degenerate sub-structure exist inside v2_1/2/4?

The first pass (job 52579272) used two broken gates and is superseded:
  * the marker gate z-scored panel means ACROSS THE SUB-CLUSTERS THEMSELVES. With
    k=2 groups a z-score is always +-1, so "panel spread" was exactly 2.00 for every
    two-way partition -- a gate that cannot fail is not a gate.
  * maxJS and panel spread are MAXIMA over pairs/groups, so both rise with cluster
    count while silhouette falls. Taking the smallest resolution that cleared all
    gates therefore rewarded fragmentation: v2_4 "split" 158,033 vs 1,062 (0.7%)
    and v2_2 produced a 78-cell cluster.

Fixes:
  * panel score = mean % of a cell's transcriptome contributed by the panel's genes.
    Absolute, interpretable, identical scale for any grouping, no self-normalisation.
  * SIZE FLOOR: every sub-cluster must hold >= MIN_PCT of the niche. Partitions that
    produce specks are rejected outright, not rescued by the specks' own extremity.
  * composition and marker evidence are computed only over surviving clusters, and
    the marker criterion is a fold-change on an absolute scale, not a z spread.
  * the chosen partition is the one with the BEST SILHOUETTE among those that pass,
    not the one at the lowest resolution.

Labels: v2_<k>.<j>, renumbered by size. A bare integer niche label is never written.
"""
import os, json, numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score

B    = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
OBJ  = f"{B}/NicheCompass_SC/runs_v3_noN3/NC_v3_Leiden_nichev2tagged.h5ad"
OUT  = f"{B}/NicheCompass_SC/runs_v3_noN3/v2_characterisation"
CAND = ["v2_1", "v2_2", "v2_4"]
RES  = [0.05, 0.10, 0.20, 0.30, 0.50]
MIN_PCT   = 2.0      # every sub-cluster must be >= 2% of the niche
SIL_MIN   = 0.10     # geometric separation in the NicheCompass latent
JS_MIN    = 0.25     # cell_type_v2 composition divergence, most separated pair
FC_MIN    = 1.50     # >=1.5x on at least one panel, on the absolute CP100 scale
SEED = 0
log = lambda *a: print(*a, flush=True)

PANELS = {
 "vascular": ["VWF","CLDN5","PECAM1","FLT1","EPAS1","CAV1","RGS5","PDGFRB","ACTA2","TAGLN"],
 "pia / fibroblast / VLMC": ["FBLN1","CEMIP","SFRP2","COL1A1","COL1A2","DCN","LUM","SLC6A13","PTGDS"],
 "ependymal": ["ZBBX","CCDC146","WDR49","EYA4","FOXJ1","PIFO","TMEM212"],
 "immune": ["C1QC","CD163","PTPRC","MRC1","LYVE1"],
 "white matter / oligo": ["MOBP","PLP1","MBP","MAG","MOG","CNP","ST18"],
 "grey matter / neuron": ["SNAP25","RBFOX3","SYT1","GAD1","GAD2","SLC17A6","CHAT","MNX1"],
 "astrocyte": ["AQP4","SLC1A2","GFAP","GJA1","SLC1A3"],
 "OPC": ["PDGFRA","OLIG1","OLIG2","PTPRZ1","MEGF11","CSPG4"],
}
# no 'nerve root / Schwann': MPZ/PMP22/NGFR/L1CAM/S100B are absent from this 480-plex
# and the survivors (SOX10/MAL/ERBB3) are oligodendrocyte-lineage.

log("loading")
A = sc.read_h5ad(OBJ)
A.obs["niche_v2"] = A.obs["niche_v2"].astype(str)
var = list(A.var_names)
PAN = {k: [g for g in v if g in var] for k, v in PANELS.items()}
PAN = {k: v for k, v in PAN.items() if v}
log("panel sizes: " + str({k: len(v) for k, v in PAN.items()}))
gidx = {k: np.array([var.index(g) for g in v]) for k, v in PAN.items()}
# NB: pd.Series(col.values) WRAPS the column buffer without copying, so writing into
# this Series mutates A.obs["niche_v2"] IN PLACE and silently replaces the 7 parent
# labels with the sub-labels. Copy explicitly; the assert below enforces it survived.
NICHES_PARENT = sorted(A.obs["niche_v2"].unique())
sub_label = pd.Series(np.asarray(A.obs["niche_v2"].values, dtype=object).copy(),
                      index=A.obs_names, dtype=object)

def panel_pct(ad, groups):
    """mean % of the cell's transcriptome contributed by each panel. Absolute scale."""
    X = ad.layers["counts"].tocsc()
    tot = np.maximum(np.asarray(ad.layers["counts"].sum(1)).ravel(), 1)
    out = {}
    for k, idx in gidx.items():
        s = np.asarray(X[:, idx].sum(1)).ravel()
        out[k] = pd.Series(100.0 * s / tot).groupby(groups).mean()
    return pd.DataFrame(out)

report = {}
for k in CAND:
    log(f"\n================ {k} ================")
    a = A[A.obs["niche_v2"] == k].copy()
    n = a.n_obs
    log(f"  {n:,} cells | size floor {MIN_PCT}% = {int(n*MIN_PCT/100):,} cells")
    sc.pp.neighbors(a, n_neighbors=15, use_rep="nichecompass_latent", random_state=SEED)
    L = a.obsm["nichecompass_latent"]
    rng = np.random.default_rng(SEED)
    samp = rng.choice(n, size=min(20000, n), replace=False)
    recs, parts = [], {}
    for r in RES:
        key = f"sub_{r}"
        sc.tl.leiden(a, resolution=r, key_added=key, random_state=SEED,
                     flavor="igraph", n_iterations=2, directed=False)
        lab = a.obs[key].astype(str).values
        vc = pd.Series(lab).value_counts()
        nsub = len(vc)
        small = 100 * vc.min() / n
        if nsub < 2:
            recs.append(dict(res=r, n_sub=1, smallest_pct=100.0, size_floor_ok=False,
                             silhouette=np.nan, max_js=np.nan, max_panel_fc=np.nan, pass_all=False))
            log(f"  res {r}: 1 cluster"); continue
        floor_ok = small >= MIN_PCT
        sil = float(silhouette_score(L[samp], lab[samp]))
        ct = pd.crosstab(lab, a.obs["cell_type_v2"].astype(str))
        ctn = ct.div(ct.sum(1), axis=0)
        js = max(float(jensenshannon(ctn.loc[i], ctn.loc[j], base=2))
                 for ii, i in enumerate(ctn.index) for j in ctn.index[ii+1:])
        PP = panel_pct(a, lab)
        fc = float((PP.max(0) / np.maximum(PP.min(0), 1e-6)).max())
        ok = floor_ok and (sil >= SIL_MIN) and (js >= JS_MIN) and (fc >= FC_MIN)
        recs.append(dict(res=r, n_sub=nsub, smallest_pct=round(small, 2),
                         size_floor_ok=bool(floor_ok), silhouette=round(sil, 4),
                         max_js=round(js, 4), max_panel_fc=round(fc, 3), pass_all=bool(ok)))
        parts[r] = (lab, PP, ctn)
        log(f"  res {r}: {nsub} sub | smallest {small:5.2f}% {'OK ' if floor_ok else 'FLOOR'} | "
            f"sil {sil:.3f} | maxJS {js:.3f} | panel FC {fc:.2f}x | "
            f"{'PASS' if ok else 'fail'}")
    S = pd.DataFrame(recs); S.to_csv(f"{OUT}/subcluster_sweep_{k}.csv", index=False)
    good = S[S.pass_all]
    if len(good):
        r = float(good.sort_values("silhouette", ascending=False).iloc[0]["res"])
        lab, PP, ctn = parts[r]
        order = pd.Series(lab).value_counts().index.tolist()
        rmap = {o: f"{k}.{i+1}" for i, o in enumerate(order)}
        new = np.array([rmap[x] for x in lab], dtype=object)
        sub_label.loc[a.obs_names] = new
        PP.index = [rmap[i] for i in PP.index]; ctn.index = [rmap[i] for i in ctn.index]
        det = pd.Series((a.obs["detached"].astype(str).values == "detached").astype(float))
        vh  = pd.Series(a.obs["in_VH_v2"].map({True: 1.0, False: 0.0}).astype(float).values)
        mnv = pd.Series(a.obs["is_MN_v2"].map({True: 1, False: 0}).fillna(0).astype(int).values)
        dep = pd.Series(np.asarray(a.obs["cord_depth_um"].values, dtype=float))
        an = pd.DataFrame({"cells": pd.Series(new).value_counts(),
                           "pct_of_niche": (100*pd.Series(new).value_counts()/n).round(2),
                           "depth_median_um": dep.groupby(new).median().round(1),
                           "pct_detached": det.groupby(new).mean().mul(100).round(2),
                           "pct_VH": vh.groupby(new).mean().mul(100).round(2),
                           "n_MN_v2": mnv.groupby(new).sum()}).sort_index()
        log(f"\n  -> SPLIT at res {r} (best silhouette among passing) into {len(order)}")
        log(an.to_string())
        log("\n  top cell types:")
        for i in sorted(ctn.index):
            t = ctn.loc[i].sort_values(ascending=False).head(4)
            log(f"    {i}: " + " | ".join(f"{c} {100*v:.0f}%" for c, v in t.items()))
        log("\n  panel % of transcriptome (absolute):\n" + PP.round(3).sort_index().to_string())
        an.to_csv(f"{OUT}/subcluster_anatomy_{k}.csv")
        PP.round(4).sort_index().to_csv(f"{OUT}/subcluster_panelpct_{k}.csv")
        (100*ctn).round(1).sort_index().to_csv(f"{OUT}/subcluster_composition_{k}.csv")
        report[k] = dict(split=True, resolution=r, n_sub=len(order),
                         labels=sorted(rmap.values()),
                         silhouette=float(good.sort_values('silhouette',ascending=False).iloc[0]['silhouette']))
    else:
        why = []
        if not S.size_floor_ok.any(): why.append("every partition produced a cluster below the size floor")
        if S.silhouette.max(skipna=True) < SIL_MIN: why.append(f"best silhouette {S.silhouette.max():.3f} < {SIL_MIN}")
        if S.max_js.max(skipna=True) < JS_MIN: why.append(f"best composition JS {S.max_js.max():.3f} < {JS_MIN}")
        log(f"  -> NOT SPLIT. {'; '.join(why) if why else 'no partition cleared all gates jointly'}")
        report[k] = dict(split=False, reason=why or ["no partition cleared all gates jointly"],
                         best_silhouette=float(np.nanmax(S.silhouette)) if S.silhouette.notna().any() else None)
    del a

assert sorted(A.obs["niche_v2"].unique()) == NICHES_PARENT, \
    "niche_v2 was mutated: the parent labels must survive sub-clustering"
A.obs["niche_v2_sub"] = pd.Categorical(sub_label.values)
log("\n=== final niche_v2_sub ===")
log(A.obs["niche_v2_sub"].value_counts().sort_index().to_string())
A.obs[["sample","status","niche_v2","niche_v2_sub","niche_v1","niche_v1_call","cell_type_v2",
       "cell_type_mn","is_MN_v2","in_VH_v2","cord_depth_um","detached"]].to_parquet(
       f"{OUT}/v2_sub_obs.parquet")
A.write_h5ad(f"{B}/NicheCompass_SC/runs_v3_noN3/NC_v3_Leiden_nichev2sub.h5ad", compression="gzip")
json.dump({"supersedes": "job 52579272 (broken gates: k=2 z-spread is always 2.00; "
                         "maxima grow with cluster count; smallest-resolution rule rewarded specks)",
           "gates": dict(min_pct=MIN_PCT, silhouette=SIL_MIN, max_js=JS_MIN, panel_fc=FC_MIN),
           "selection": "best silhouette among passing partitions",
           "candidates": CAND, "result": report,
           "contract": "sub-clusters are v2_<k>.<j>; a bare integer niche label is never written"},
          open(f"{OUT}/subcluster_decision.json", "w"), indent=2)
log("\nDONE")
