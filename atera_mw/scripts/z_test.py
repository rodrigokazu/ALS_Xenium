#!/usr/bin/env python3
"""Does transcript depth (z) carry cell identity that 2D segmentation cannot use?

A. Pairs (as stage 0): nuclear transcripts in one 10x nucleus vs two neighbouring nuclei, matched by 2D distance and by
   distance to the nucleus edge; feature = -|dz|. AUC > 0.5 means neighbouring nuclei sit at different depths.
B. Foreign transcripts: inside each 10x cell (>= 200 informative transcripts), "native" = informative transcripts whose gene
   favours the cell's own type (log ratio <= -1 against every other type), "foreign" = gene favours another type by >= 2.
   Feature = |z - median z of the cell's native transcripts|, compared within radial bins (distance to the cell's centroid,
   deciles) so that edge position does not stand in for depth. AUC > 0.5 means mixed-in transcripts come from another depth.
python z_test.py <out.csv>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.draw import polygon as draw_poly
from skimage.segmentation import find_boundaries
from sklearn.metrics import roc_auc_score

UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); CMP = SSD / "compare"
CROPS = [("spot1_purkinje_left", 200, 200), ("spot2_gl_wm_centre", 200, 200), ("spot3_purkinje_right", 100, 170)]
SIZE, PAD = 100.0, 6.0; L = SIZE + 2 * PAD
GROUPS = {"granule": [1, 2, 3, 4, 5, 8, 9, 10, 15, 6, 7, 11], "PC/interneuron": [14, 23, 33], "astro": [13, 16, 19],
          "oligo": [12, 18, 27, 29, 30, 32], "vascular": [17, 20, 21, 22, 26, 28], "immune": [24, 25, 31, 34, 35, 36]}
T = len(GROUPS); rng = np.random.default_rng(0)
de = pd.read_csv(SSD / "Cerebellum_sample/analysis/diffexp/gene_expression_graphclust/differential_expression.csv")
clu = pd.read_csv(SSD / "Cerebellum_sample/analysis/clustering/gene_expression_graphclust/clusters.csv")
ncl = clu.Cluster.value_counts().sort_index().reindex(range(1, 37)).to_numpy().astype(float)
M = np.column_stack([de[f"Cluster {k} Mean Counts"] for k in range(1, 37)]); Lf = np.column_stack([de[f"Cluster {k} Log2 fold change"] for k in range(1, 37)])
Pv = np.column_stack([de[f"Cluster {k} Adjusted p value"] for k in range(1, 37)])
inf = ((Lf >= 2.0) & (M >= 0.3) & (Pv < 0.01)).any(1); gidx = pd.Index(de["Feature Name"].to_numpy()[inf]); Mi = M[inf]
prof = np.zeros((T, inf.sum()))
for t, ks in enumerate(GROUPS.values()):
    w = ncl[np.array(ks) - 1]; prof[t] = (Mi[:, np.array(ks) - 1] * w).sum(1) / w.sum()
prof += 1e-4; prof /= prof.sum(1, keepdims=True); logp = np.log(prof)
c2t = {c: t for t, cs in enumerate(GROUPS.values()) for c in cs}; cell_type = dict(zip(clu.Barcode, clu.Cluster.map(c2t)))

rows = []
for spot, ox, oy in CROPS:
    win = np.load(CMP / "option_C500" / spot / "crop.npz")["window"]
    bx0, by0 = win[0] * UM + ox - PAD, win[1] * UM + oy - PAD
    tx = ds.dataset(SSD / "work/local_bundle/transcripts.parquet").to_table(columns=["x_location", "y_location", "z_location", "feature_name", "cell_id", "overlaps_nucleus"],
        filter=(pc.field("x_location") >= bx0) & (pc.field("x_location") < bx0 + L) & (pc.field("y_location") >= by0) & (pc.field("y_location") < by0 + L)
        & (pc.field("qv") >= 20) & (pc.field("is_gene"))).to_pandas()
    x, y, zz = tx.x_location.to_numpy() - bx0, tx.y_location.to_numpy() - by0, tx.z_location.to_numpy()
    cid = tx.cell_id.astype(str).to_numpy(); inner = (x >= PAD) & (x < PAD + SIZE) & (y >= PAD) & (y < PAD + SIZE)
    # ---- A: nucleus pairs, distance- and edge-matched
    n = int(round(L / UM))
    nb = ds.dataset(SSD / "work/local_bundle/nucleus_boundaries.parquet").to_table(filter=(pc.field("vertex_x") >= bx0 - 10) & (pc.field("vertex_x") < bx0 + L + 10)
                                                                              & (pc.field("vertex_y") >= by0 - 10) & (pc.field("vertex_y") < by0 + L + 10)).to_pandas()
    nlab = np.zeros((n, n), np.int32)
    for k, (_, g) in enumerate(nb.groupby("cell_id", sort=False), start=1):
        rr, cc = draw_poly((g.vertex_y.to_numpy() - by0) / UM, (g.vertex_x.to_numpy() - bx0) / UM, shape=(n, n)); nlab[rr, cc] = k
    edge = ndi.distance_transform_edt((nlab > 0) & ~find_boundaries(nlab, mode="inner")) * UM
    nuc = (tx.overlaps_nucleus.to_numpy() == 1) & ~pd.Series(cid).isin(["", "UNASSIGNED"]).to_numpy() & inner
    ni = np.flatnonzero(nuc); c_n = cid[ni]; t_n = np.array([cell_type.get(c, -1) for c in c_n])
    e_n = edge[np.clip((y[ni] / UM).astype(int), 0, n - 1), np.clip((x[ni] / UM).astype(int), 0, n - 1)]
    Pn = np.c_[x[ni], y[ni]]; tree = cKDTree(Pn); anc = rng.choice(len(ni), min(20000, len(ni)), replace=False)
    pa, pb, dd, lab = [], [], [], []
    for a, nbr in zip(anc, tree.query_ball_point(Pn[anc], 4.0)):
        nbr = np.asarray(nbr); d = np.hypot(*(Pn[nbr] - Pn[a]).T)
        for lo in (1, 2, 3):
            m = (d >= lo) & (d < lo + 1)
            for cls, mm in (("same", m & (c_n[nbr] == c_n[a])), ("diff_same_type", m & (c_n[nbr] != c_n[a]) & (t_n[nbr] == t_n[a]) & (t_n[a] >= 0)),
                            ("diff_other_type", m & (c_n[nbr] != c_n[a]) & (t_n[nbr] != t_n[a]) & (t_n[nbr] >= 0) & (t_n[a] >= 0))):
                c = nbr[mm]
                for j in (rng.choice(c, min(2, len(c)), replace=False) if len(c) else []):
                    pa.append(a); pb.append(j); dd.append(lo); lab.append(cls)
    pa, pb, dd, lab = map(np.array, (pa, pb, dd, lab))
    fz = -np.abs(zz[ni[pa]] - zz[ni[pb]]); eb = np.minimum((np.maximum(e_n[pa], e_n[pb]) / 0.25).astype(int), 8) * 10 + np.minimum((np.minimum(e_n[pa], e_n[pb]) / 0.25).astype(int), 8)
    for lo in (1, 2, 3):
        for neg in ("diff_same_type", "diff_other_type"):
            idx = np.flatnonzero((dd == lo) & np.isin(lab, ["same", neg])); pos, ng = idx[lab[idx] == "same"], idx[lab[idx] != "same"]
            keep = [ng] + [rng.choice(pos[eb[pos] == b], min(int((eb[ng] == b).sum()), int((eb[pos] == b).sum())), replace=False) for b in np.unique(eb[ng]) if (eb[pos] == b).any()]
            mm = np.concatenate(keep); yv = (lab[mm] == "same").astype(int)
            if 30 <= yv.sum() < len(yv) - 30:
                rows.append(dict(crop=spot[:5], test="A_pairs", stratum=f"{lo}-{lo + 1} um, {neg}", auc=roc_auc_score(yv, fz[mm]), n_pos=int(yv.sum()), n_neg=int(len(yv) - yv.sum())))
    # ---- B: foreign vs native transcripts inside 10x cells
    gi = gidx.get_indexer(tx.feature_name.astype(str)); incell = inner & ~pd.Series(cid).isin(["", "UNASSIGNED"]).to_numpy() & (gi >= 0)
    df = pd.DataFrame({"c": cid[incell], "g": gi[incell], "x": x[incell], "y": y[incell], "z": zz[incell]})
    ll = pd.DataFrame(logp[:, df.g].T).groupby(df.c.to_numpy()).sum(); cnt = df.c.value_counts()
    ct = pd.Series(ll.to_numpy().argmax(1), index=ll.index)[cnt.reindex(ll.index) >= 200]
    df = df[df.c.isin(ct.index)].copy(); df["t"] = ct.reindex(df.c).to_numpy()
    lp = logp[:, df.g.to_numpy()]; own = lp[df.t.to_numpy(), np.arange(len(df))]; oth = lp.copy(); oth[df.t.to_numpy(), np.arange(len(df))] = -np.inf
    lr = oth.max(0) - own
    df["kind"] = np.where(lr >= 2, "foreign", np.where(lr <= -1, "native", "other"))
    zmed = df[df.kind == "native"].groupby("c").z.median(); df["dz"] = (df.z - zmed.reindex(df.c).to_numpy()).abs()
    cx, cy = df.groupby("c").x.transform("mean"), df.groupby("c").y.transform("mean"); df["r"] = np.hypot(df.x - cx, df.y - cy)
    df = df[df.kind.isin(["foreign", "native"]) & df.dz.notna()]; df["rbin"] = pd.qcut(df.r, 10, labels=False, duplicates="drop")
    aucs, ws = [], []
    for _, g in df.groupby("rbin"):
        yv = (g.kind == "foreign").astype(int)
        if 30 <= yv.sum() < len(yv) - 30: aucs.append(roc_auc_score(yv, g.dz)); ws.append(yv.sum())
    rows.append(dict(crop=spot[:5], test="B_foreign_vs_native", stratum="radial-matched", auc=float(np.average(aucs, weights=ws)), n_pos=int((df.kind == "foreign").sum()), n_neg=int((df.kind == "native").sum())))
    yv = (df.kind == "foreign").astype(int)
    rows.append(dict(crop=spot[:5], test="B_foreign_vs_native", stratum="unmatched", auc=roc_auc_score(yv, df.dz), n_pos=int(yv.sum()), n_neg=int(len(yv) - yv.sum())))
    rows.append(dict(crop=spot[:5], test="info", stratum=f"z sd crop {zz[inner].std():.2f} um, median per-cell native z sd {df[df.kind == 'native'].groupby('c').z.std().median():.2f} um", auc=np.nan, n_pos=0, n_neg=0))
    print(spot, "done", flush=True)
out = pd.DataFrame(rows); out.to_csv(sys.argv[1], index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
print(out.pivot_table(index=["test", "stratum"], columns="crop", values="auc").round(3).to_string())
print(out[out.test != "A_pairs"].to_string(index=False))
