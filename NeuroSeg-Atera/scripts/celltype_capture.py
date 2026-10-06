#!/usr/bin/env python3
"""Does the new segmentation keep cells whole beyond Purkinje cells? Same test for every cell type, 10x vs ours.

Cell type of each 10x nucleus = its 10x graph-cluster, grouped (Purkinje 33, MLI 14, UBC 23, granule, astroglia, oligodendroglia,
vascular, immune). Specific genes per group: group mean >= 0.5 and >= 3x the mean of every other group (10x cluster table).
A nucleus "owns" the transcripts of its group's specific genes that lie closer to its centroid than to any other 10x nucleus
(Voronoi) and within 10 um. For nuclei owning >= 20 such transcripts, per method: capture = share in the single best cell,
split = >= 2 cells hold >= 10%, missed = no cell holds >= 10%. Paired across methods (same nuclei): Wilcoxon (capture),
McNemar (split). Also by 10x cell area (quartiles over all nuclei). Nine 500 um windows.
python celltype_capture.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, zarr
from scipy import stats
from scipy.spatial import cKDTree
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); OUT = SSD / "compare/celltype_capture"; OUT.mkdir(exist_ok=True)
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
WINS = ["spot1_purkinje_left", "spot2_gl_wm_centre", "spot3_purkinje_right"] + [f"held{i}" for i in range(1, 7)]
GRP = {"Purkinje": [33], "MLI (basket/stellate)": [14], "UBC": [23], "Granule": [1, 2, 3, 4, 5, 8, 9, 10, 15, 6, 7, 11], "Astroglia": [13, 16, 19],
       "Oligodendroglia": [12, 18, 27, 29, 30, 32], "Vascular": [17, 20, 21, 22, 26, 28], "Immune": [24, 25, 31, 34, 35, 36]}
de = pd.read_csv(SSD / "Cerebellum_sample/analysis/diffexp/gene_expression_graphclust/differential_expression.csv").set_index("Feature Name")
cl = pd.read_csv(SSD / "Cerebellum_sample/analysis/clustering/gene_expression_graphclust/clusters.csv"); ncl = cl.Cluster.value_counts()
M = de[[f"Cluster {k} Mean Counts" for k in range(1, 37)]]; M.columns = range(1, 37)
GM = pd.DataFrame({g: (M[ks] * ncl.reindex(ks).to_numpy()).sum(1) / ncl.reindex(ks).sum() for g, ks in GRP.items()})
SPEC = {g: set(GM.index[(GM[g] >= 0.5) & (GM[g] >= 3 * GM.drop(columns=g).max(1))]) for g in GRP}
print({g: len(v) for g, v in SPEC.items()})
c2g = {k: g for g, ks in GRP.items() for k in ks}; ctype = dict(zip(cl.Barcode, cl.Cluster.map(c2g)))
area = pd.read_parquet(SSD / "Cerebellum_sample/cells.parquet", columns=["cell_id", "cell_area"]).set_index("cell_id").cell_area
nb = pd.read_parquet(SSD / "Cerebellum_sample/nucleus_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"]).groupby("cell_id")[["vertex_x", "vertex_y"]].mean()
lab_full = zarr.open_array(str(SSD / "segmentation_full/final_labels.zarr"), mode="r"); rows = []
for w in WINS:
    x0, y0, x1, y1 = json.load(open(RUNS / "crops_PF_500" / w / "stats.json"))["window"]
    b = SSD / "work" / ("local_bundle_heldout" if w.startswith("held") else "local_bundle")
    tx = ds.dataset(b / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name", "cell_id"], filter=(pc.field("x_location") >= x0 * UM)
         & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM) & (pc.field("qv") >= 20) & pc.field("is_gene")).to_pandas()
    EXTRA = {v: np.asarray(np.load(RUNS / f"crops_{v}_500" / w / "final_labels.npy", mmap_mode="r")[y0:y1, x0:x1]) for v in ("HYBA", "HYBB", "HYBC")}
    L = np.asarray(lab_full[y0:y1, x0:x1]); r = np.clip((tx.y_location / UM).astype(int) - y0, 0, y1 - y0 - 1); c = np.clip((tx.x_location / UM).astype(int) - x0, 0, x1 - x0 - 1)
    tx["ours"] = L[r, c].astype(np.int64).astype(str); tx["tenx"] = tx.cell_id.astype(str).where(~tx.cell_id.astype(str).isin(["", "UNASSIGNED"]), "0"); tx.loc[tx.ours == "0", "ours"] = "0"
    for v, LL in EXTRA.items(): tx[v] = LL[r, c].astype(np.int64).astype(str)
    nn = nb[(nb.vertex_x >= x0 * UM - 15) & (nb.vertex_x < x1 * UM + 15) & (nb.vertex_y >= y0 * UM - 15) & (nb.vertex_y < y1 * UM + 15)]
    d, i = cKDTree(nn.to_numpy()).query(tx[["x_location", "y_location"]].to_numpy()); tx["nuc"] = np.where(d <= 10, nn.index.to_numpy()[i], "")
    inner = nn[(nn.vertex_x >= x0 * UM) & (nn.vertex_x < x1 * UM) & (nn.vertex_y >= y0 * UM) & (nn.vertex_y < y1 * UM)].index
    tx["g"] = tx.nuc.map(ctype); fn = tx.feature_name.astype(str)
    own = tx[(tx.nuc != "") & tx.g.notna() & np.array([f in SPEC.get(g, ()) for f, g in zip(fn, tx.g)]) & tx.nuc.isin(inner)]
    for nuc, t in own.groupby("nuc"):
        if len(t) < 20: continue
        row = dict(window=w, nucleus=nuc, type=ctype[nuc], area_10x=area.get(nuc, np.nan), n_own=len(t))
        for m in ("tenx", "ours", "HYBA", "HYBB", "HYBC"):
            vc = t[m][t[m] != "0"].value_counts() / len(t)
            row[f"capture_{m}"] = vc.iloc[0] if len(vc) else 0.0; row[f"ncells_{m}"] = int((vc >= 0.10).sum())
        rows.append(row)
    print(w, len(rows), flush=True)
d = pd.DataFrame(rows); d.to_csv(OUT / "celltype_capture.csv", index=False)
d["size_q"] = pd.qcut(d.area_10x, 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"])
def summ(g, m="ours"):
    a, b = g.capture_tenx, g[f"capture_{m}"]; s1, s2 = g.ncells_tenx >= 2, g[f"ncells_{m}"] >= 2
    bb, cc = int((s1 & ~s2).sum()), int((~s1 & s2).sum())
    return pd.Series({"n": len(g), "capture_10x": a.median(), "capture_ours": b.median(), "wilcoxon_p": stats.wilcoxon(b, a).pvalue if (a != b).any() else 1.0,
                      "split_10x_%": 100 * s1.mean(), "split_ours_%": 100 * s2.mean(), "mcnemar_p": stats.binomtest(bb, bb + cc).pvalue if bb + cc else 1.0,
                      "missed_10x_%": 100 * (g.ncells_tenx == 0).mean(), "missed_ours_%": 100 * (g[f"ncells_{m}"] == 0).mean()})
pd.set_option("display.width", 220)
for m in ("ours", "HYBA", "HYBB", "HYBC"):
    bt = d.groupby("type").apply(lambda g: summ(g, m)).sort_values("n"); print("====", m, "vs 10x"); print(bt.round(3).to_string())
    bt.to_csv(OUT / f"by_type_{m}.csv")
