#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/3_cell_annotation/audit_v2_v3/cellaudit.py
#
# Audits the Other neurons class and everything effectively unassigned. It joins DOT,
# CellTyping_coarse_FINAL and the v2 NicheCompass object on exact (sample, x, y), then
# re-clusters the Other neurons cells: CP10K, log1p, scale at 10, PCA 40, 15 neighbours,
# Leiden 0.5. That gives 12 sub-clusters. otherneurons_cells.csv.gz carries the sub-cluster
# id per cell and is the file that annot_v2, annot_v3 and nc3_step1 all read.
# ========================================================================================

"""Audit the ALS spinal cord Xenium cell annotation.

Focus: the "Other neurons" class and everything that is effectively unassigned.

Joins three annotation layers onto the object the science actually uses (the
NicheCompass object) by exact (sample, x, y):
  1. DOT deconvolution  -> dot_label, dot_label_score, dot_MN_weight, 9-way weights, X_tsne
  2. Reference scoring   -> cell_type_coarse, qc_flag, tissue_region, 12 sc_* scores, QC
  3. NicheCompass object -> cell_type, cell_type_mn, is_MN_v2, in_VH_v2, niche
Then characterises "Other neurons" transcriptionally and sub-clusters it.
"""
import json, os, sys, collections
import numpy as np, h5py, pandas as pd
import scipy.sparse as sp

B = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
NC = f"{B}/NicheCompass_SC/runs_v2/integrated_whole/20260902_ncv2/model/NC_v2_integrated_whole_Leiden.h5ad"
CT = f"{B}/CellTyping_coarse_FINAL/ALS_SCXenium_coarse_celltyped.h5ad"
TS = f"{B}/(RK)ALS_SC_concatenated_openTSNE.h5ad"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
R = {}


def cat(f, col):
    g = f["obs"][col]
    if g.attrs.get("encoding-type") == "categorical":
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.array(c, dtype=object)[g["codes"][:]].astype(str)
    v = g[:]
    if v.dtype.kind in "SO":
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])
    return v.astype(str)


def nbool(f, col):
    g = f["obs"][col]
    return g["values"][:] & ~g["mask"][:], g["mask"][:]


def num(f, col):
    return np.asarray(f["obs"][col][:], dtype=np.float64)


# ---------------------------------------------------------------- NC object
print("[1] NC object", flush=True)
fn = h5py.File(NC, "r")
ns = cat(fn, "sample"); nstat = cat(fn, "status")
nct = cat(fn, "cell_type"); nctm = cat(fn, "cell_type_mn")
nniche = cat(fn, "latent_leiden_0.3")
is_mn, mn_na = nbool(fn, "is_MN_v2")
in_vh, vh_na = nbool(fn, "in_VH_v2")
nxy = fn["obsm"]["spatial"][:]
genes = [x.decode() if isinstance(x, bytes) else str(x) for x in fn["var"]["_index"][:]]
N = len(ns)
print(f"    {N:,} cells, {len(genes)} genes", flush=True)

# counts layer -> CP10K + log1p (X itself is already log1p; use raw counts)
grp = fn["layers"]["counts"]
Xc = sp.csr_matrix((grp["data"][:], grp["indices"][:], grp["indptr"][:]),
                   shape=(N, len(genes)))
tot = np.asarray(Xc.sum(1)).ravel()
ngenes = np.diff(Xc.indptr)
fn.close()
print(f"    counts: median {np.median(tot):.0f}, median genes {np.median(ngenes):.0f}", flush=True)

L = sp.csr_matrix(Xc, dtype=np.float32, copy=True)
sf = np.maximum(tot, 1) / 1e4
L = sp.diags(1.0 / sf).dot(L)
L.data = np.log1p(L.data)
L = sp.csr_matrix(L)

# ---------------------------------------------------------------- join keys
def keyframe(s, x, y):
    return pd.MultiIndex.from_arrays([s, np.round(x, 3), np.round(y, 3)])


nkey = keyframe(ns, nxy[:, 0], nxy[:, 1])

print("[2] reference-scoring object", flush=True)
fc = h5py.File(CT, "r")
cs = cat(fc, "sample")
ckey = keyframe(cs, num(fc, "x_centroid"), num(fc, "y_centroid"))
SC = sorted([c for c in fc["obs"].keys() if c.startswith("sc_")])
cdf = pd.DataFrame({c: num(fc, c) for c in SC}, index=ckey)
for c in ("cell_type_coarse", "qc_flag", "tissue_region"):
    cdf[c] = cat(fc, c)
for c in ("total_counts", "tx_assigned_pct", "nucleus_area", "cell_area",
          "control_probe_counts", "unassigned_codeword_counts", "neg_ctrl_rate"):
    if c in fc["obs"]:
        cdf[c] = num(fc, c)
NCT = len(cs); fc.close()
cdf = cdf[~cdf.index.duplicated()]
cj = cdf.reindex(nkey)
print(f"    {NCT:,} cells -> matched {int(cj['cell_type_coarse'].notna().sum()):,} "
      f"of {N:,} NC cells", flush=True)

print("[3] DOT object", flush=True)
ft = h5py.File(TS, "r")
ts_s = cat(ft, "sample")
tkey = keyframe(ts_s, num(ft, "x_centroid"), num(ft, "y_centroid"))
W = ft["obsm"]["dot_v3_weights"][:]
tdf = pd.DataFrame(W, columns=[f"dotw_{i}" for i in range(W.shape[1])], index=tkey)
for c in ("dot_label", "dot_label_score", "dot_MN_weight", "dot_is_MN", "leiden",
          "n_genes", "pct_counts_in_top_10_genes"):
    if c in ft["obs"]:
        g = ft["obs"][c]
        tdf[c] = cat(ft, c) if g.attrs.get("encoding-type") == "categorical" else num(ft, c)
TN = len(ts_s)
tsne = pd.DataFrame(ft["obsm"]["X_tsne"][:], columns=["tsne1", "tsne2"], index=tkey)
ft.close()
tdf = tdf[~tdf.index.duplicated()]; tsne = tsne[~tsne.index.duplicated()]
tj = tdf.reindex(nkey); tsj = tsne.reindex(nkey)
print(f"    {TN:,} cells -> matched {int(tj['dot_label'].notna().sum()):,} of {N:,}", flush=True)

R["attrition"] = dict(
    dot_object=TN, reference_object=NCT, nichecompass_object=N,
    nc_matched_to_reference=int(cj["cell_type_coarse"].notna().sum()),
    nc_matched_to_dot=int(tj["dot_label"].notna().sum()),
    nc_unmatched_to_dot=int(tj["dot_label"].isna().sum()))

# ---------------------------------------------------------------- label maps
R["counts"] = {
    "cell_type_mn": {k: int(v) for k, v in collections.Counter(nctm).items()},
    "cell_type_nc": {k: int(v) for k, v in collections.Counter(nct).items()},
    "dot_label": {k: int(v) for k, v in
                  collections.Counter(tj["dot_label"].dropna()).items()},
    "cell_type_coarse": {k: int(v) for k, v in
                         collections.Counter(cj["cell_type_coarse"].dropna()).items()},
    "qc_flag": {k: int(v) for k, v in collections.Counter(cj["qc_flag"].dropna()).items()},
    "tissue_region": {k: int(v) for k, v in
                      collections.Counter(cj["tissue_region"].dropna()).items()},
}

# is NC cell_type identical to dot_label?
ok = tj["dot_label"].notna().values
if ok.sum():
    xt = pd.crosstab(pd.Series(nct[ok], name="nc"),
                     pd.Series(tj["dot_label"].values[ok], name="dot"))
    xt.to_csv(f"{OUT}/xtab_nc_vs_dot.csv")
    diag = sum(xt.loc[i, i] for i in xt.index if i in xt.columns)
    R["nc_equals_dot"] = dict(matched=int(ok.sum()), identical=int(diag),
                              pct=float(100 * diag / ok.sum()))
    print(f"    NC cell_type == dot_label for {100*diag/ok.sum():.1f}% of matched cells",
          flush=True)

# ---------------------------------------------------------------- per-class profile
print("[4] per-class profiles", flush=True)
CLASSES = sorted(set(nctm))
gidx = {g: i for i, g in enumerate(genes)}
mean_by_class = {}
for c in CLASSES:
    m = nctm == c
    mean_by_class[c] = np.asarray(L[m].mean(0)).ravel()
ME = pd.DataFrame(mean_by_class, index=genes)
ME.to_csv(f"{OUT}/class_mean_expression.csv")
Z = ME.sub(ME.mean(1), axis=0).div(ME.std(1).replace(0, np.nan), axis=0).fillna(0)
Z.to_csv(f"{OUT}/class_marker_z.csv")

prof = {}
for c in CLASSES:
    m = nctm == c
    z = Z[c].sort_values(ascending=False)
    d = dict(n=int(m.sum()), frac=float(m.mean()),
             median_counts=float(np.median(tot[m])), median_genes=float(np.median(ngenes[m])),
             pct_counts_lt20=float(100 * (tot[m] < 20).mean()),
             markers_up=[[g, round(float(v), 3)] for g, v in z.head(15).items()],
             markers_dn=[[g, round(float(v), 3)] for g, v in z.tail(6).items()],
             in_vh=float(in_vh[m].mean()), na_vh=float(vh_na[m].mean()),
             niche=[[k, int(v)] for k, v in
                    collections.Counter(nniche[m]).most_common(4)])
    q = cj["qc_flag"].values[m]
    d["qc_flag_true"] = float(np.mean(q == "True")) if len(q) else None
    d["qc_flag_na"] = float(np.mean(pd.isna(q) | (q == "NA")))
    tr = cj["tissue_region"].values[m]
    d["tissue_region"] = {k: int(v) for k, v in collections.Counter(
        [x for x in tr if not pd.isna(x)]).most_common(5)}
    d["coarse_label"] = {k: int(v) for k, v in collections.Counter(
        [x for x in cj["cell_type_coarse"].values[m] if not pd.isna(x)]).most_common(6)}
    d["dot_label_score"] = (float(np.nanmedian(tj["dot_label_score"].values[m]))
                            if "dot_label_score" in tj else None)
    d["sc_top"] = {}
    sub = cj.loc[:, SC].values[m]
    if np.isfinite(sub).any():
        mu = np.nanmean(sub, 0)
        d["sc_mean"] = {s.replace("sc_", ""): round(float(v), 4) for s, v in zip(SC, mu)}
        top = np.array([s.replace("sc_", "") for s in SC])[np.nanargmax(sub, 1)]
        d["sc_top"] = {k: int(v) for k, v in collections.Counter(top).most_common(6)}
    prof[c] = d
R["classes"] = prof

# ---------------------------------------------------------------- assignment confidence
print("[5] assignment confidence from the reference scores", flush=True)
Sv = cj.loc[:, SC].values
good = np.isfinite(Sv).all(1)
srt = np.sort(Sv[good], 1)[:, ::-1]
best, second = srt[:, 0], srt[:, 1]
topname = np.array([s.replace("sc_", "") for s in SC])[np.argmax(Sv[good], 1)]
ALIAS = {"Oligodendrocytes": "Oligodendrocyte", "Astrocytes": "Astrocyte", "OPCs": "OPC",
         "Microglia": "Microglia", "Macrophages": "Macrophage_perivasc",
         "Excitatory neurons": "Neuron_excit", "Inhibitory neurons": "Neuron_inhib",
         "Endothelial": "Endothelial", "Motor neurons": "MN", "Other neurons": None}
assigned = nctm[good]
agree = np.array([ALIAS.get(a) == t if ALIAS.get(a) else False
                  for a, t in zip(assigned, topname)])
R["confidence"] = dict(
    n_scored=int(good.sum()),
    median_best=float(np.median(best)), median_margin=float(np.median(best - second)),
    pct_best_le0=float(100 * (best <= 0).mean()),
    pct_margin_lt001=float(100 * ((best - second) < 0.01).mean()),
    overall_agreement=float(100 * agree.mean()),
    by_class={})
for c in CLASSES:
    m = assigned == c
    if m.sum() < 10:
        continue
    R["confidence"]["by_class"][c] = dict(
        n=int(m.sum()), agree=float(100 * agree[m].mean()),
        median_best=float(np.median(best[m])),
        median_margin=float(np.median((best - second)[m])),
        pct_best_le0=float(100 * (best[m] <= 0).mean()),
        top_ref={k: int(v) for k, v in collections.Counter(topname[m]).most_common(5)})

# ---------------------------------------------------------------- MN audit
print("[6] motor neuron audit", flush=True)
imn = SC.index("sc_MN")
mnsc = cj.loc[:, SC].values[:, imn]
dmw = tj["dot_MN_weight"].values if "dot_MN_weight" in tj else np.full(N, np.nan)
R["mn"] = dict(
    n_canonical=int(is_mn.sum()),
    coarse_MN_class=int((nct == "Motor neurons").sum()),
    other_neurons=int((nctm == "Other neurons").sum()),
    canonical_in_coarse_MN=int(((nct == "Motor neurons") & is_mn).sum()),
    canonical_outside_coarse_MN=int(((nct != "Motor neurons") & is_mn).sum()),
    canonical_by_coarse={k: int(v) for k, v in collections.Counter(nct[is_mn]).items()},
    sc_MN_median_all=float(np.nanmedian(mnsc)),
    sc_MN_p99_all=float(np.nanpercentile(mnsc, 99)),
    sc_MN_median_canonical=float(np.nanmedian(mnsc[is_mn])),
    sc_MN_median_other_neurons=float(np.nanmedian(mnsc[nctm == "Other neurons"])),
    dot_MN_weight_median_canonical=float(np.nanmedian(dmw[is_mn])),
    dot_MN_weight_median_other=float(np.nanmedian(dmw[nctm == "Other neurons"])),
    n_sc_MN_positive=int(np.nansum(mnsc > 0)),
)
fin = np.isfinite(mnsc)
if fin.sum():
    thr = np.nanpercentile(mnsc[fin], 99.9)
    top = fin & (mnsc >= thr)
    R["mn"]["labels_of_top_0p1pct_scMN"] = {k: int(v) for k, v in
                                            collections.Counter(nctm[top]).most_common()}

# ---------------------------------------------------------------- Other neurons deep dive
print("[7] sub-clustering Other neurons", flush=True)
ON = nctm == "Other neurons"
import scanpy as sc_
import anndata as ad
sub = ad.AnnData(X=L[ON].copy(), obs=pd.DataFrame(index=np.arange(int(ON.sum()))),
                 var=pd.DataFrame(index=genes))
sub.obs["sample"] = ns[ON]; sub.obs["status"] = nstat[ON]
sub.obs["niche"] = nniche[ON]
sub.obs["counts"] = tot[ON]; sub.obs["ngenes"] = ngenes[ON]
sub.obs["in_vh"] = in_vh[ON]
sc_.pp.scale(sub, max_value=10)
sc_.tl.pca(sub, n_comps=40, svd_solver="arpack")
sc_.pp.neighbors(sub, n_neighbors=15, n_pcs=40)
sc_.tl.leiden(sub, resolution=0.5, key_added="sub", flavor="igraph", n_iterations=2,
              directed=False)
subl = sub.obs["sub"].astype(str).values
print(f"    {len(set(subl))} sub-clusters", flush=True)

Lon = L[ON]
onm = {}
for cl in sorted(set(subl), key=lambda s: int(s)):
    m = subl == cl
    onm[cl] = np.asarray(Lon[m].mean(0)).ravel()
OME = pd.DataFrame(onm, index=genes)
OME.to_csv(f"{OUT}/otherneurons_subcluster_mean.csv")
OZ = OME.sub(OME.mean(1), axis=0).div(OME.std(1).replace(0, np.nan), axis=0).fillna(0)
# also score each sub-cluster against the whole-cohort class profile
GZ = Z.reindex(index=genes)
subs = {}
cjs = cj.loc[:, SC].values[ON]
tjd = tj["dot_label_score"].values[ON] if "dot_label_score" in tj else np.full(int(ON.sum()), np.nan)
qcs = cj["qc_flag"].values[ON]; trs = cj["tissue_region"].values[ON]
ccs = cj["cell_type_coarse"].values[ON]
for cl in sorted(set(subl), key=lambda s: int(s)):
    m = subl == cl
    z = OZ[cl].sort_values(ascending=False)
    # correlate this sub-cluster's profile with each cohort class mean
    corr = {c: float(np.corrcoef(OME[cl].values, ME[c].values)[0, 1]) for c in CLASSES}
    ssub = cjs[m]
    d = dict(n=int(m.sum()), frac_of_other=float(m.mean()),
             median_counts=float(np.median(tot[ON][m])),
             median_genes=float(np.median(ngenes[ON][m])),
             pct_counts_lt20=float(100 * (tot[ON][m] < 20).mean()),
             markers_up=[[g, round(float(v), 3)] for g, v in z.head(12).items()],
             best_class_corr=sorted(corr.items(), key=lambda kv: -kv[1])[:4],
             in_vh=float(np.mean(in_vh[ON][m])),
             status={k: int(v) for k, v in collections.Counter(nstat[ON][m]).most_common()},
             niche={k: int(v) for k, v in collections.Counter(nniche[ON][m]).most_common(3)},
             qc_flag_true=float(np.mean(qcs[m] == "True")),
             coarse={k: int(v) for k, v in collections.Counter(
                 [x for x in ccs[m] if not pd.isna(x)]).most_common(4)},
             dot_label_score=float(np.nanmedian(tjd[m])) if np.isfinite(tjd[m]).any() else None)
    if np.isfinite(ssub).any():
        top = np.array([s.replace("sc_", "") for s in SC])[np.nanargmax(ssub, 1)]
        d["sc_top"] = {k: int(v) for k, v in collections.Counter(top).most_common(4)}
        d["sc_MN_median"] = float(np.nanmedian(ssub[:, imn]))
    subs[cl] = d
R["other_neurons_subclusters"] = subs

# save per-cell sub-cluster + the joined frame the figures need
pd.DataFrame({"sample": ns[ON], "status": nstat[ON], "sub": subl,
              "x": nxy[ON, 0], "y": nxy[ON, 1], "counts": tot[ON],
              "ngenes": ngenes[ON], "niche": nniche[ON]}
             ).to_csv(f"{OUT}/otherneurons_cells.csv.gz", index=False, compression="gzip")

# ---------------------------------------------------------------- unassigned inventory
print("[8] unassigned inventory", flush=True)
qc = cj["qc_flag"].values
tr = cj["tissue_region"].values
R["unassigned"] = dict(
    total_nc=N,
    is_MN_v2_NA=int(mn_na.sum()), in_VH_v2_NA=int(vh_na.sum()),
    qc_flag_true=int(np.sum(qc == "True")),
    qc_flag_na=int(np.sum(pd.isna(qc) | (qc == "NA"))),
    tissue_not_spinal=int(np.sum(tr == "Frontal Precentral/Motor BA4")),
    tissue_na=int(np.sum(pd.isna(tr) | (tr == "NA"))),
    counts_lt_20=int((tot < 20).sum()), counts_lt_10=int((tot < 10).sum()),
    genes_lt_10=int((ngenes < 10).sum()),
    no_dot_match=int(tj["dot_label"].isna().sum()),
    no_reference_match=int(cj["cell_type_coarse"].isna().sum()),
)
# union of "should not be trusted"
bad = ((qc == "True") | pd.isna(qc) | (qc == "NA") |
       (tr == "Frontal Precentral/Motor BA4") | (tot < 20))
R["unassigned"]["union_untrustworthy"] = int(np.sum(bad))
R["unassigned"]["union_pct"] = float(100 * np.mean(bad))
R["unassigned"]["by_class"] = {
    c: dict(n=int((nctm == c).sum()),
            untrustworthy=int(np.sum(bad & (nctm == c))),
            pct=float(100 * np.mean(bad[nctm == c])))
    for c in CLASSES}

# per-sample QC for the figures
pd.DataFrame({"sample": ns, "status": nstat, "cell_type_mn": nctm,
              "counts": tot, "ngenes": ngenes,
              "qc_flag": qc, "tissue_region": tr,
              "niche": nniche, "is_MN_v2": is_mn, "mn_na": mn_na}
             ).to_csv(f"{OUT}/cells_joined.csv.gz", index=False, compression="gzip")
# tSNE coords for figures (only matched cells)
tsj2 = tsj.copy(); tsj2["cell_type_mn"] = nctm; tsj2["qc_flag"] = qc
tsj2["is_MN_v2"] = is_mn; tsj2["counts"] = tot
tsj2.dropna(subset=["tsne1"]).to_csv(f"{OUT}/tsne_joined.csv.gz", index=False,
                                     compression="gzip")

json.dump(R, open(f"{OUT}/cell_audit.json", "w"), indent=1, default=str)
print("\n=== HEADLINES ===")
print(f"  Other neurons        : {R['mn']['other_neurons']:,}")
print(f"  coarse MN class      : {R['mn']['coarse_MN_class']:,}")
print(f"  canonical is_MN_v2   : {R['mn']['n_canonical']:,}")
print(f"  sc_MN > 0 anywhere   : {R['mn']['n_sc_MN_positive']:,}")
print(f"  label agrees w/ own top reference: {R['confidence']['overall_agreement']:.1f}%")
print(f"  untrustworthy union  : {R['unassigned']['union_untrustworthy']:,} "
      f"({R['unassigned']['union_pct']:.1f}%)")
print(f"  Other-neuron subclusters: {len(subs)}")
print("DONE ->", OUT)
