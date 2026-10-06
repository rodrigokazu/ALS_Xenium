#!/usr/bin/env python3
"""Stage 0 of PLAN.md: does local transcript composition tell "same cell" from "different cell"?

Ground truth from DAPI nuclei (10x nucleus masks, overlaps_nucleus = 1): a pair of nuclear transcripts is "same" when both
sit in one nucleus and "different" when they sit in two nuclei. Pairs are matched by distance (1-2, 2-3, 3-4 um), and the
"different" pairs are split by whether the two nuclei are the same coarse type (granule next to granule) or not.
Features per pair (AUC = how well the feature ranks "same" above "different"):
  comp_clu_s   similarity of the local composition (mean p(cluster | gene) of the transcripts around each point, smoothed s um)
  comp_svd_s   the same with an unsupervised gene embedding (SVD of 2 um bin x gene counts in the crop)
  density      -|log density difference|
  membrane     -max of the membrane stain on the straight line between the two transcripts
python feasibility.py <out.csv>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc
from scipy import ndimage as ndi, sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from skimage.draw import polygon as draw_poly
from skimage.segmentation import find_boundaries

UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); CMP = SSD / "compare"
CROPS = [("spot1_purkinje_left", 200, 200), ("spot2_gl_wm_centre", 200, 200), ("spot3_purkinje_right", 100, 170)]  # 100 um crop, offset in window (um)
SIZE, PAD, BIN = 100.0, 6.0, 0.5
SIGMAS = (0.5, 1.0, 2.0); DBINS = ((1, 2), (2, 3), (3, 4))
GROUPS = {"granule": [1, 2, 3, 4, 5, 8, 9, 10, 15, 6, 7, 11], "PC/interneuron": [14, 23, 33], "astro": [13, 16, 19],
          "oligo": [12, 18, 27, 29, 30, 32], "vascular": [17, 20, 21, 22, 26, 28], "immune": [24, 25, 31, 34, 35, 36]}
rng = np.random.default_rng(0)

# p(cluster | gene) for informative genes (same filter as phase2.py)
de = pd.read_csv(SSD / "Cerebellum_sample/analysis/diffexp/gene_expression_graphclust/differential_expression.csv")
clu = pd.read_csv(SSD / "Cerebellum_sample/analysis/clustering/gene_expression_graphclust/clusters.csv")
ncl = clu.Cluster.value_counts().sort_index().reindex(range(1, 37)).to_numpy().astype(float)
M = np.column_stack([de[f"Cluster {k} Mean Counts"] for k in range(1, 37)]); Lf = np.column_stack([de[f"Cluster {k} Log2 fold change"] for k in range(1, 37)])
Pv = np.column_stack([de[f"Cluster {k} Adjusted p value"] for k in range(1, 37)])
inf = ((Lf >= 2.0) & (M >= 0.3) & (Pv < 0.01)).any(1)
pcg = M[inf] * ncl; pcg /= pcg.sum(1, keepdims=True); gidx = pd.Index(de["Feature Name"].to_numpy()[inf])
c2t = {c: t for t, cs in enumerate(GROUPS.values()) for c in cs}
cell_type = dict(zip(clu.Barcode, clu.Cluster.map(c2t)))

def field(xb, yb, vec, nx, ny, s):
    """smoothed mean vector per 0.5 um bin"""
    k = vec.shape[1]; flat = yb * nx + xb
    S = np.stack([np.bincount(flat, weights=vec[:, d], minlength=nx * ny) for d in range(k)]).reshape(k, ny, nx)
    n = np.bincount(flat, minlength=nx * ny).reshape(ny, nx).astype(float)
    sg = s / BIN; S = np.stack([ndi.gaussian_filter(S[d], sg) for d in range(k)]); n = ndi.gaussian_filter(n, sg)
    return S / np.maximum(n, 1e-6)

def cos(F, ia, ib):
    a = F[:, ia[1], ia[0]].T; b = F[:, ib[1], ib[0]].T
    mu = F.reshape(F.shape[0], -1).mean(1); a = a - mu; b = b - mu
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)

rows = []
for spot, ox, oy in CROPS:
    z = np.load(CMP / "option_C500" / spot / "crop.npz"); img, win = z["img"], z["window"]
    X0, Y0 = win[0] * UM + ox, win[1] * UM + oy
    bx0, by0 = X0 - PAD, Y0 - PAD; L = SIZE + 2 * PAD
    tx = ds.dataset(SSD / "work/local_bundle/transcripts.parquet").to_table(
        columns=["x_location", "y_location", "feature_name", "cell_id", "overlaps_nucleus"],
        filter=(pc.field("x_location") >= bx0) & (pc.field("x_location") < bx0 + L) & (pc.field("y_location") >= by0) & (pc.field("y_location") < by0 + L)
        & (pc.field("qv") >= 20) & (pc.field("is_gene"))).to_pandas()
    x, y = tx.x_location.to_numpy() - bx0, tx.y_location.to_numpy() - by0
    nx = ny = int(np.ceil(L / BIN)); xb = np.clip((x / BIN).astype(int), 0, nx - 1); yb = np.clip((y / BIN).astype(int), 0, ny - 1)
    gname = tx.feature_name.astype(str).to_numpy()
    # unsupervised gene embedding: 2 um bins x genes in this crop, log1p, SVD 30
    genes, gcode = np.unique(gname, return_inverse=True); keep = np.bincount(gcode) >= 30
    b2 = (yb // 4) * (nx // 4 + 1) + (xb // 4)
    X = sparse.csr_matrix((np.ones(len(gcode)), (b2, gcode)), shape=(b2.max() + 1, len(genes)))[:, keep]
    X = X.multiply(1 / np.maximum(X.sum(1), 1)).tocsr(); X.data = np.log1p(1e3 * X.data)
    emb = np.zeros((len(genes), 30)); svd = TruncatedSVD(30, random_state=0).fit(X); E = svd.components_.T
    emb[keep] = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    gi = gidx.get_indexer(gname); okc = gi >= 0
    F = {}
    for s in SIGMAS:
        F[f"comp_clu_{s}"] = field(xb[okc], yb[okc], pcg[gi[okc]], nx, ny, s)
        F[f"comp_svd_{s}"] = field(xb, yb, emb[gcode], nx, ny, s)
    dens = ndi.gaussian_filter(np.bincount(yb * nx + xb, minlength=nx * ny).reshape(ny, nx).astype(float), 1.0 / BIN) / BIN ** 2
    mem = img[1].astype(float)
    # distance of each point to its own nucleus edge (10x nucleus polygons on the image grid), for the edge-matched control
    nb_ = ds.dataset(SSD / "work/local_bundle/nucleus_boundaries.parquet").to_table(
        filter=(pc.field("vertex_x") >= bx0 - 10) & (pc.field("vertex_x") < bx0 + L + 10) & (pc.field("vertex_y") >= by0 - 10) & (pc.field("vertex_y") < by0 + L + 10)).to_pandas()
    nlab = np.zeros(mem.shape, np.int32)
    for k, (_, g) in enumerate(nb_.groupby("cell_id", sort=False), start=1):
        rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - win[1], g.vertex_x.to_numpy() / UM - win[0], shape=mem.shape); nlab[rr, cc] = k
    edge = ndi.distance_transform_edt((nlab > 0) & ~find_boundaries(nlab, mode="inner")) * UM
    # nuclear transcripts in the inner crop
    nuc = (tx.overlaps_nucleus.to_numpy() == 1) & ~tx.cell_id.isin(["", "UNASSIGNED"]).to_numpy() & (x >= PAD) & (x < PAD + SIZE) & (y >= PAD) & (y < PAD + SIZE)
    ni = np.where(nuc)[0]; cid = tx.cell_id.to_numpy()[ni]; ctp = np.array([cell_type.get(c, -1) for c in cid])
    P = np.c_[x[ni], y[ni]]; tree = cKDTree(P)
    anc = rng.choice(len(ni), min(20000, len(ni)), replace=False)
    pa, pb, dd, lab = [], [], [], []
    for a, nb in zip(anc, tree.query_ball_point(P[anc], 4.0)):
        nb = np.asarray(nb); d = np.hypot(*(P[nb] - P[a]).T)
        for lo, hi in DBINS:
            m = (d >= lo) & (d < hi)
            for cls in ("same", "diff_same_type", "diff_other_type"):
                if cls == "same": mm = m & (cid[nb] == cid[a])
                elif cls == "diff_same_type": mm = m & (cid[nb] != cid[a]) & (ctp[nb] == ctp[a]) & (ctp[a] >= 0)
                else: mm = m & (cid[nb] != cid[a]) & (ctp[nb] != ctp[a]) & (ctp[nb] >= 0) & (ctp[a] >= 0)
                c = nb[mm]
                if len(c):
                    for j in rng.choice(c, min(2, len(c)), replace=False):
                        pa.append(a); pb.append(j); dd.append(lo); lab.append(cls)
    pa, pb, dd, lab = map(np.array, (pa, pb, dd, lab))
    ia = (xb[ni[pa]], yb[ni[pa]]); ib = (xb[ni[pb]], yb[ni[pb]])
    feat = {k: cos(v, ia, ib) for k, v in F.items()}
    feat["density"] = -np.abs(np.log(dens[ia[1], ia[0]] + 1) - np.log(dens[ib[1], ib[0]] + 1))
    # membrane stain along the line, on the image pixel grid
    ax_, ay_ = (x[ni[pa]] + bx0) / UM - win[0], (y[ni[pa]] + by0) / UM - win[1]; bx_, by_ = (x[ni[pb]] + bx0) / UM - win[0], (y[ni[pb]] + by0) / UM - win[1]
    t = np.linspace(0, 1, 12)[:, None]
    lx = np.clip((ax_ + t * (bx_ - ax_)).astype(int), 0, mem.shape[1] - 1); ly = np.clip((ay_ + t * (by_ - ay_)).astype(int), 0, mem.shape[0] - 1)
    feat["membrane"] = -mem[ly, lx].max(0)
    ea = edge[np.clip(ay_.astype(int), 0, mem.shape[0] - 1), np.clip(ax_.astype(int), 0, mem.shape[1] - 1)]
    eb = edge[np.clip(by_.astype(int), 0, mem.shape[0] - 1), np.clip(bx_.astype(int), 0, mem.shape[1] - 1)]
    ebin = np.minimum((np.maximum(ea, eb) / 0.25).astype(int), 8) * 10 + np.minimum((np.minimum(ea, eb) / 0.25).astype(int), 8)
    for lo, hi in DBINS:
        for neg in ("diff_same_type", "diff_other_type"):
            m = (dd == lo) & np.isin(lab, ["same", neg]); y01 = (lab[m] == "same").astype(int)
            if y01.sum() < 30 or (1 - y01).sum() < 30: continue
            # edge-matched: keep same-nucleus pairs so that their (max, min) edge distance histogram equals the different-nucleus pairs'
            idx = np.where(m)[0]; pos, negs = idx[lab[idx] == "same"], idx[lab[idx] != "same"]; keep = [negs]
            for b in np.unique(ebin[negs]):
                cand = pos[ebin[pos] == b]; need = int((ebin[negs] == b).sum())
                if len(cand): keep.append(rng.choice(cand, min(need, len(cand)), replace=False))
            mm = np.concatenate(keep); ym = (lab[mm] == "same").astype(int)
            for k, v in feat.items():
                rows.append(dict(crop=spot[:5], dist=f"{lo}-{hi}", negatives=neg, feature=k, auc=roc_auc_score(y01, v[m]),
                                 auc_edge_matched=roc_auc_score(ym, v[mm]) if 30 <= ym.sum() < len(ym) - 30 else np.nan,
                                 n_same=int(y01.sum()), n_diff=int((1 - y01).sum()), n_same_matched=int(ym.sum())))
    print(spot, "transcripts", len(tx), "nuclear", len(ni), "pairs", len(lab), flush=True)

out = pd.DataFrame(rows); out.to_csv(sys.argv[1], index=False)
pd.set_option("display.width", 220)
print(out.pivot_table(index=["negatives", "dist", "feature"], columns="crop", values=["auc", "auc_edge_matched"]).round(3).to_string())
print(out.groupby(["negatives", "dist", "crop"])[["n_same", "n_diff", "n_same_matched"]].first().to_string())
