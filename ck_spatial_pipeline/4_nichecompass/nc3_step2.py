#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/nc3_step2.py
#
# Labels the retrain's 7 Leiden niches as niche_v2 v2_0 to v2_6 and carries the v1 tags and
# cell_type_v2 over by obs_name. The gate requires centroids within 0.01 um, the same sample
# and the same is_MN_v2, and a permuted control must fail it. Job 52579254.
# ========================================================================================

"""Step 2 of the niche-3 removal + retrain: label, bridge, and re-characterise.

The 20260914_ncv3 retrain (job 52575218) landed 7 Leiden niches on the 965,285
retained cells. This script:

  1. writes niche_v2 = 'v2_0'..'v2_6' onto the retrained object, and carries the
     v1 tags across from runs_v3_noN3/NC_v2_Leiden_nichev1tagged.h5ad;
  2. crosstabs niche_v1 <-> niche_v2 on the retained cells, joined on obs_name --
     v2 is NOT nested in v1, the crosstab is the only bridge to the Bench;
  3. asks whether old v1_0 (pial surface / vasculature) and v1_2 (the 45% mixed
     block) actually SPLIT in the retrain;
  4. re-characterises every v2 niche the way the Bench does: size, cell_type_v2
     composition, marker z (CP100), targeted panels, transcript depth, cord depth,
     detachment, MN and VH content, peripherality, sample-spread gate.

JOIN GATE (cf. join_verification_content_gate): row counts and means PASS a broken
join. Every joined row is gated on centroid agreement < 0.01 um in BOTH axes plus
sample and is_MN_v2 agreement. Any failure aborts.

NUMBERING CONTRACT: a bare integer niche label is never written again.
"""
import os, json
import numpy as np, pandas as pd, h5py, scipy.sparse as sp
import scanpy as sc
from scipy import ndimage

B    = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
RUN  = f"{B}/NicheCompass_SC/runs_v2/integrated_whole/20260914_ncv3"
NEW  = f"{RUN}/model/NC_v2_integrated_whole_Leiden.h5ad"
OLD  = f"{B}/NicheCompass_SC/runs_v3_noN3/NC_v2_Leiden_nichev1tagged.h5ad"
OUT  = f"{B}/NicheCompass_SC/runs_v3_noN3/v2_characterisation"
os.makedirs(OUT, exist_ok=True)
BIN, MIN_BINS = 60.0, 8          # same geometry as depth.py / detached.py
log = lambda *a: print(*a, flush=True)

def cat(f, c):
    g = f["obs"][c]
    cc = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
    return np.array(cc, dtype=object)[g["codes"][:]].astype(str)

def nbool(f, c):
    g = f["obs"][c]
    v = g["values"][:].astype(bool); m = g["mask"][:].astype(bool)
    o = np.empty(len(v), dtype=object); o[:] = v; o[m] = None
    return o

def idx(f):
    return np.array([x.decode() for x in f["obs"][f["obs"].attrs["_index"]][:]], dtype=object)

# ---------------------------------------------------------------- 1. read ----
log("=== reading the retrained object ===")
with h5py.File(NEW, "r") as f:
    nm_new  = idx(f)
    leiden  = cat(f, "latent_leiden_0.3")
    ns      = cat(f, "sample")
    stat    = cat(f, "status")
    ctm     = cat(f, "cell_type_mn")
    is_mn2  = nbool(f, "is_MN_v2")
    in_vh   = nbool(f, "in_VH_v2")
    xy      = f["obsm"]["spatial"][:]
    var     = [x.decode() for x in f["var"][f["var"].attrs["_index"]][:]]
n, G = len(nm_new), len(var)
log(f"  {n:,} cells, {G} genes, leiden {sorted(set(leiden), key=int)}")

log("=== reading the v1-tagged object's obs ===")
with h5py.File(OLD, "r") as f:
    nm_old  = idx(f)
    v1      = cat(f, "niche_v1")
    v1call  = cat(f, "niche_v1_call")
    ctv2_o  = cat(f, "cell_type_v2")
    drop_o  = cat(f, "dropped_v3")
    ns_o    = cat(f, "sample")
    mn_o    = nbool(f, "is_MN_v2")
    xy_o    = f["obsm"]["spatial"][:]
    intr_o  = cat(f, "in_tract") if "in_tract" in f["obs"] and "categories" in f["obs"]["in_tract"] else None
log(f"  {len(nm_old):,} cells")

# ------------------------------------------------------- 2. join + GATE ------
log("\n=== join on obs_name, with a CONTENT gate ===")
assert len(set(nm_old)) == len(nm_old), "obs_name not unique in the v1-tagged object"
assert len(set(nm_new)) == len(nm_new), "obs_name not unique in the retrained object"
pos = pd.Series(np.arange(len(nm_old)), index=nm_old)
miss = ~pd.Index(nm_new).isin(pos.index)
assert not miss.any(), f"{int(miss.sum()):,} retrained obs_names absent from the v1 object"
j = pos.reindex(nm_new).to_numpy()

dxy = np.abs(xy_o[j] - xy)
gate = {
  "n_joined":            int(n),
  "centroid_x_max_dev":  float(dxy[:, 0].max()),
  "centroid_y_max_dev":  float(dxy[:, 1].max()),
  "n_centroid_fail":     int(((dxy > 0.01).any(1)).sum()),
  "n_sample_mismatch":   int((ns_o[j] != ns).sum()),
  "n_isMN_mismatch":     int(sum(a is not b for a, b in zip(mn_o[j], is_mn2))),
  "n_not_marked_retained": int((drop_o[j] != "retained").sum()),
}
for k, v in gate.items(): log(f"  {k:<24} {v}")
assert gate["n_centroid_fail"] == 0,       "CONTENT GATE FAILED: centroid disagreement"
assert gate["n_sample_mismatch"] == 0,     "CONTENT GATE FAILED: sample disagreement"
assert gate["n_isMN_mismatch"] == 0,       "CONTENT GATE FAILED: is_MN_v2 disagreement"
assert gate["n_not_marked_retained"] == 0, "CONTENT GATE FAILED: a dropped cell is in the retrain"
# a permuted control, so the gate is shown to be able to fail
rng = np.random.default_rng(0); perm = rng.permutation(n)
ctrl = float(np.hypot(*(xy_o[j][perm] - xy).T).mean())
gate["permuted_control_mean_dxy_um"] = ctrl
log(f"  permuted control mean |dxy| = {ctrl:,.0f} um (real join = 0.00) -> gate discriminates")
log("  JOIN GATE PASSED")

v1     = v1[j]; v1call = v1call[j]; ctv2 = ctv2_o[j]
intr   = intr_o[j] if intr_o is not None else None

# ------------------------------------------------- 3. the v2 labels ----------
niche_v2 = np.array(["v2_" + k for k in leiden], dtype=object)
KS  = sorted(set(leiden), key=int)
V2K = [f"v2_{k}" for k in KS]
V1K = sorted(set(v1), key=lambda s: int(s.split("_")[1]))
log("\n=== niche_v2 ===")
for k in V2K:
    m = niche_v2 == k
    log(f"  {k}: {int(m.sum()):>9,}  ({100*m.mean():.2f}%)")

# ------------------------------------------------- 4. the crosstab -----------
log("\n=== crosstab niche_v1 <-> niche_v2 (965,285 retained cells) ===")
X = pd.crosstab(pd.Series(v1, name="niche_v1"), pd.Series(niche_v2, name="niche_v2"))
X = X.reindex(index=V1K, columns=V2K).fillna(0).astype(int)
log("\ncounts\n" + X.to_string())
rowp = (100 * X.div(X.sum(1), axis=0)).round(2)   # where each v1 niche WENT
colp = (100 * X.div(X.sum(0), axis=1)).round(2)   # what each v2 niche is MADE OF
log("\nrow%% - where each v1 niche went\n" + rowp.to_string())
log("\ncol%% - what each v2 niche is made of\n" + colp.to_string())
X.to_csv(f"{OUT}/crosstab_v1_v2_counts.csv")
rowp.to_csv(f"{OUT}/crosstab_v1_v2_rowpct.csv")
colp.to_csv(f"{OUT}/crosstab_v1_v2_colpct.csv")

def ent(p):
    p = np.asarray(p, float); p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())

log("\n=== did v1_0 and v1_2 SPLIT? ===")
split_rows = []
for k in V1K:
    r = X.loc[k]
    if r.sum() == 0: continue
    share = r / r.sum()
    nz = int((r >= 0.05 * r.sum()).sum())      # v2 niches taking >=5% of it
    split_rows.append(dict(niche_v1=k, call=sorted(set(v1call[v1 == k]))[0],
                           n=int(r.sum()), top_v2=r.idxmax(),
                           top_share_pct=round(100 * share.max(), 2),
                           n_v2_over_5pct=nz, entropy_bits=round(ent(r), 3)))
S = pd.DataFrame(split_rows)
log(S.to_string(index=False))
S.to_csv(f"{OUT}/v1_split_summary.csv", index=False)
for k in ("v1_0", "v1_2"):
    if k not in X.index: continue
    r = X.loc[k]; sh = (100 * r / r.sum()).round(2)
    verdict = "SPLIT" if sh.max() < 70 else ("mostly INTACT" if sh.max() >= 85 else "PARTIAL split")
    log(f"\n  {k} ({int(r.sum()):,} cells) -> {verdict}: "
        + ", ".join(f"{c} {sh[c]}%" for c in sh[sh >= 1].sort_values(ascending=False).index))

# ------------------------------------------- 5. geometry: depth + detached ---
log("\n=== geometry per section (60 um grid, components >= 8 bins) ===")
depth = np.full(n, np.nan); detached = np.zeros(n, bool); inside = np.zeros(n, bool)
periph = np.zeros(n)
sec_rows = []
for s_ in sorted(set(ns)):
    m = ns == s_; p = xy[m]
    gx = ((p[:, 0] - p[:, 0].min()) / BIN).astype(int)
    gy = ((p[:, 1] - p[:, 1].min()) / BIN).astype(int)
    grid = np.zeros((gy.max() + 1, gx.max() + 1), dtype=bool); grid[gy, gx] = True
    lab, nlab = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
    sizes = ndimage.sum(grid, lab, range(1, nlab + 1))
    keep = np.where(sizes >= MIN_BINS)[0] + 1
    if not len(keep): continue
    main = keep[np.argmax(sizes[keep - 1])]
    cl = lab[gy, gx]
    det = cl != main
    cord = ndimage.binary_fill_holes(lab == main)
    edt = ndimage.distance_transform_edt(cord) * BIN
    d = edt[gy, gx]
    ins = cl == main
    ii = np.where(m)[0]
    detached[ii] = det; inside[ii] = ins
    depth[ii[ins]] = d[ins]
    c = p.mean(0); dd = np.hypot(*(p - c).T); periph[ii] = dd / np.percentile(dd, 99)
    sec_rows.append(dict(sample=s_, status=stat[m][0], n=int(m.sum()),
                         n_components=int(len(keep)), frac_detached=float(det.mean())))
pd.DataFrame(sec_rows).to_csv(f"{OUT}/section_geometry.csv", index=False)
log(f"  mean section detached fraction {100*np.mean([r['frac_detached'] for r in sec_rows]):.1f}%")

# ------------------------------------------- 6. counts, CP100 marker z -------
log("\n=== streaming counts for depth-comparable CP100 marker means ===")
kidx = {k: i for i, k in enumerate(V2K)}
code = np.array([kidx[x] for x in niche_v2])
tot = np.zeros(n, dtype=np.int64); ngenes = np.zeros(n, dtype=np.int32)
sums = np.zeros((len(V2K), G)); cnts = np.zeros(len(V2K))
with h5py.File(NEW, "r") as f:
    g = f["layers"]["counts"]; indptr = g["indptr"][:]
    CH = 200_000
    for s0 in range(0, n, CH):
        e = min(s0 + CH, n); a, b = int(indptr[s0]), int(indptr[e])
        M = sp.csr_matrix((g["data"][a:b], g["indices"][a:b], indptr[s0:e+1] - indptr[s0]),
                          shape=(e - s0, G))
        t = np.asarray(M.sum(1)).ravel(); tot[s0:e] = t
        ngenes[s0:e] = np.diff(M.indptr)
        Mn = np.asarray(M.todense()) / np.maximum(t, 1)[:, None] * 100
        for i in range(len(V2K)):
            mm = code[s0:e] == i
            if mm.any(): sums[i] += Mn[mm].sum(0); cnts[i] += mm.sum()
        log(f"  {e:,}/{n:,}")
mean = sums / np.maximum(cnts, 1)[:, None]
z = (mean - mean.mean(0)) / np.maximum(mean.std(0), 1e-9)
Z = pd.DataFrame(z, index=V2K, columns=var)
Z.to_csv(f"{OUT}/v2_marker_z.csv")

# ------------------------------------------- 7. characterisation table -------
mn = np.array([x is True for x in is_mn2])
PANELS = {
 "nerve root / Schwann": ["SOX10","MAL","ERBB3","PLP1","MPZ","PMP22","S100B","NGFR","L1CAM"],
 "vascular":             ["VWF","CLDN5","PECAM1","FLT1","EPAS1","CAV1","RGS5","PDGFRB","ACTA2","TAGLN"],
 "pia / fibroblast / VLMC":["FBLN1","CEMIP","SFRP2","COL1A1","COL1A2","DCN","LUM","SLC6A13","PTGDS"],
 "ependymal":            ["ZBBX","CCDC146","WDR49","EYA4","FOXJ1","PIFO","TMEM212"],
 "immune":               ["C1QC","CD163","PTPRC","MRC1","LYVE1"],
 "white matter / oligo": ["MOBP","PLP1","MBP","MAG","MOG","CNP","ST18"],
 "grey matter / neuron": ["SNAP25","RBFOX3","SYT1","GAD1","GAD2","SLC17A6","CHAT","MNX1"],
 "astrocyte":            ["AQP4","SLC1A2","GFAP","GJA1","SLC1A3"],
}
PAN = {k: [x for x in v if x in var] for k, v in PANELS.items()}
pan_score = pd.DataFrame({k: Z[v].mean(1) for k, v in PAN.items() if v})
pan_score.round(3).to_csv(f"{OUT}/v2_panel_scores.csv")

NSEC = len(set(ns))
rows = []
for k in V2K:
    m = niche_v2 == k
    ct = pd.Series(ctv2[m]).value_counts(normalize=True)
    vh = [x for x in in_vh[m] if x is not None]
    persamp = pd.Series(ns[m]).value_counts()
    dk = depth[m][~np.isnan(depth[m])]
    ps = pan_score.loc[k].sort_values(ascending=False)
    rows.append(dict(
        niche_v2=k, cells=int(m.sum()), pct=round(100 * m.mean(), 2),
        med_counts=int(np.median(tot[m])), med_genes=int(np.median(ngenes[m])),
        n_MN_v2=int((mn & m).sum()),
        pct_VH=round(100 * float(np.mean(vh)), 2) if vh else np.nan,
        pct_detached=round(100 * float(detached[m].mean()), 2),
        depth_median_um=round(float(np.median(dk)), 1) if dk.size else np.nan,
        depth_p90_um=round(float(np.percentile(dk, 90)), 1) if dk.size else np.nan,
        periph_median=round(float(np.median(periph[m])), 3),
        n_sections=int((persamp > 0).sum()),
        n_sections_ge1pct=int((persamp / m.sum() >= 0.01).sum()),
        max_section_share_pct=round(100 * float(persamp.max() / m.sum()), 2),
        top_cell_type=ct.index[0], top_cell_type_pct=round(100 * ct.iloc[0], 2),
        top4_cell_types=" | ".join(f"{a} {100*b:.0f}%" for a, b in ct.head(4).items()),
        top_panel=ps.index[0], top_panel_z=round(float(ps.iloc[0]), 2),
        next_panel=ps.index[1], next_panel_z=round(float(ps.iloc[1]), 2),
        panel_margin=round(float(ps.iloc[0] - ps.iloc[1]), 2),
        top12_markers=", ".join(f"{var[i]}({z[V2K.index(k), i]:.1f})"
                                for i in np.argsort(-z[V2K.index(k)])[:12]),
    ))
C = pd.DataFrame(rows)
C.to_csv(f"{OUT}/v2_characterisation.csv", index=False)

log("\n=== v2 niche characterisation ===")
log(C[["niche_v2","cells","pct","med_counts","n_MN_v2","pct_VH","pct_detached",
       "depth_median_um","periph_median","n_sections","max_section_share_pct",
       "top_cell_type","top_cell_type_pct","top_panel","panel_margin"]].to_string(index=False))
log("\n--- top cell types ---")
for r in rows: log(f"  {r['niche_v2']}: {r['top4_cell_types']}")
log("\n--- top 12 markers (z of CP100 mean) ---")
for r in rows: log(f"  {r['niche_v2']}: {r['top12_markers']}")
log("\n--- panel scores (mean z over panel) ---")
log(pan_score.round(2).to_string())

# --------------------------------------- 8. pre-registered mixedness rule ----
# The Bench called niche v1_2 LOW confidence on exactly two grounds: no dominant
# cell type, and no panel separating it from the next-best call. Same rule here.
log("\n=== mixedness rule: top cell type < 50% AND panel margin < 0.50 => sub-cluster ===")
mixed = []
for r in rows:
    flag_ct = r["top_cell_type_pct"] < 50
    flag_pn = r["panel_margin"] < 0.50
    verdict = "MIXED -> sub-cluster" if (flag_ct and flag_pn) else (
              "watch" if (flag_ct or flag_pn) else "clean")
    if flag_ct and flag_pn: mixed.append(r["niche_v2"])
    log(f"  {r['niche_v2']:<6} top {r['top_cell_type']} {r['top_cell_type_pct']:.1f}% "
        f"| margin {r['panel_margin']:.2f} ({r['top_panel']} vs {r['next_panel']}) -> {verdict}")
log(f"\n  NICHES TO SUB-CLUSTER: {mixed if mixed else 'none'}")

# --------------------------------------- 9. MN enrichment on v2 numbering ----
log("\n=== MN (is_MN_v2) by v2 niche ===")
tab = pd.DataFrame({"n_MN": pd.Series(niche_v2[mn]).value_counts().reindex(V2K).fillna(0).astype(int),
                    "n_cells": pd.Series(niche_v2).value_counts().reindex(V2K)})
tab["pct_of_MN"] = (100 * tab.n_MN / tab.n_MN.sum()).round(2)
tab["cohort_pct"] = (100 * tab.n_cells / tab.n_cells.sum()).round(2)
tab["OR"] = [( (a/(tab.n_MN.sum()-a)) / max((b-a)/(tab.n_cells.sum()-tab.n_MN.sum()-(b-a)),1e-12) )
             if a else 0.0 for a, b in zip(tab.n_MN, tab.n_cells)]
tab["OR"] = tab["OR"].round(2)
log(tab.to_string())
tab.to_csv(f"{OUT}/v2_mn_enrichment.csv")

# --------------------------------------- 10. write the labelled object -------
log("\n=== writing the v2-labelled object ===")
a = sc.read_h5ad(NEW)
assert a.n_obs == n and (a.obs_names.to_numpy().astype(object) == nm_new).all()
a.obs["niche_v2"]      = pd.Categorical(niche_v2, categories=V2K)
a.obs["niche_v1"]      = pd.Categorical(v1, categories=V1K)
a.obs["niche_v1_call"] = pd.Categorical(v1call)
a.obs["cell_type_v2"]  = pd.Categorical(ctv2)
if intr is not None: a.obs["in_tract"] = pd.Categorical(intr)
a.obs["cord_depth_um"] = depth
a.obs["detached"]      = pd.Categorical(np.where(detached, "detached", "main_mass"))
p = f"{B}/NicheCompass_SC/runs_v3_noN3/NC_v3_Leiden_nichev2tagged.h5ad"
a.write_h5ad(p, compression="gzip")
log(f"  wrote {p} ({os.path.getsize(p)/1e9:.2f} GB)")

obs = pd.DataFrame({"obs_name": nm_new, "sample": ns, "status": stat,
                    "niche_v2": niche_v2, "niche_v1": v1, "niche_v1_call": v1call,
                    "cell_type_v2": ctv2, "cell_type_mn": ctm,
                    "is_MN_v2": [None if x is None else bool(x) for x in is_mn2],
                    "in_VH_v2": [None if x is None else bool(x) for x in in_vh],
                    "cord_depth_um": depth, "detached": detached,
                    "periph": periph, "counts": tot, "n_genes": ngenes,
                    "x": xy[:, 0], "y": xy[:, 1]})
obs.to_parquet(f"{OUT}/v2_obs.parquet", index=False)
log(f"  wrote {OUT}/v2_obs.parquet")

json.dump({"runstamp": "20260914_ncv3", "slurm_job": 52575218,
           "n_cells": int(n), "niches_v2": V2K,
           "join_gate": gate,
           "mixedness_rule": "top cell_type_v2 share < 50% AND panel margin < 0.50",
           "to_subcluster": mixed,
           "numbering_contract": {
             "niche_v1": "v1_0..v1_8 — the 20260902_ncv2 run; the Bench numbering",
             "niche_v2": "v2_0..v2_6 — the 20260914_ncv3 retrain",
             "sub_clusters": "v2_<k>.<j>",
             "rule": "a bare integer niche label is never written again"}},
          open(f"{OUT}/v2_step2_manifest.json", "w"), indent=2)
log("\nDONE")
