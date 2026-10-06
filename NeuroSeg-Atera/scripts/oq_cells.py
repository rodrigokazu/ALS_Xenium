#!/usr/bin/env python3
"""Open cerebellum questions on the annotated full section (cell level). Showcase analyses: one donor, one section.

0 Layer map: 25 um bins classified from the annotated cells around them (3x3 bins): white matter (oligodendrocytes >= 40%),
  granule layer (granule >= 60%, >= 3 cells), else molecular side; molecular-side bins within 25 um of the granule layer =
  Purkinje layer band, beyond = molecular layer. Every cell gets the layer of its bin.
1 Connexins: share of cells with >= 1 transcript and mean per 10k, by cell type.
2 Zebrin: Purkinje cells = >= 200 tx and >= 4.3% Purkinje genes. ALDOC per 10k (log) is corrected for contamination by
  regressing out the cell's glia-gene and granule-gene fractions; the residual is the zebrin score. Tests: spatial coherence
  (neighbour correlation over the 5 nearest Purkinje cells vs 1,000 permutations); genes differing between top and bottom
  tercile (Mann-Whitney on per-10k, genes in >= 10% of Purkinje cells, glia / granule genes left out, BH FDR).
5 Purkinje-layer interneurons: interneurons (MLI, Golgi types) in the Purkinje layer band vs the molecular layer, same test.
python oq_cells.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
from scipy import sparse, stats, ndimage as ndi
from scipy.spatial import cKDTree
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); O = SSD / "segmentation_full"; Q = O / "open_questions"; Q.mkdir(exist_ok=True)
MUTED, GRID, BLUE, ORANGE = "#52514e", "#e1e0d9", "#2a78d6", "#eb6834"
cells = pd.read_parquet(O / "cells.parquet"); X = sparse.load_npz(O / "cell_by_gene.npz").tocsr(); genes = pd.read_csv(O / "genes.csv").gene.astype(str); G = pd.Index(genes)
ann = pd.read_parquet(O / "annotation/cells_annotated.parquet"); ann.index = ann.index.astype(str)
cells["key"] = cells.cell.astype(str); cells = cells.join(ann[["cell_type", "lineage"]], on="key")
n = np.asarray(X.sum(1)).ravel(); cells["n"] = n
gs = json.load(open(SSD / "compare/purkinje/gene_sets.json")); frac = lambda names: np.asarray(X[:, np.isin(genes, names)].sum(1)).ravel() / np.maximum(n, 1)
cells["f_pc"], cells["f_glia"], cells["f_gran"] = frac(gs["purkinje_A"] + gs["purkinje_B"]), frac(gs["glia"]), frac(gs["granule"])
def cp10k(name): return 1e4 * np.asarray(X[:, G.get_loc(name)].todense()).ravel() / np.maximum(n, 1)
rep = {}
# ---------------------------------------------------------------- 0 layer map
B = 25.0; c = cells[cells.cell_type.notna()]
bx, by = (c.x_centroid_um // B).astype(int), (c.y_centroid_um // B).astype(int); W, H = bx.max() + 2, by.max() + 2
def grid(mask):
    g = np.zeros((H, W)); np.add.at(g, (by[mask], bx[mask]), 1); return ndi.uniform_filter(g, 3) * 9
tot = grid(np.ones(len(c), bool)); gran = grid((c.cell_type == "Granule").to_numpy()); olig = grid(c.cell_type.isin(["Oligodendrocyte", "OPC"]).to_numpy())
layer = np.full((H, W), "", dtype=object); tis = tot >= 1
wm = tis & (olig / np.maximum(tot, 1) >= 0.4); gl = tis & ~wm & (gran / np.maximum(tot, 1) >= 0.6) & (tot >= 3)
mol = tis & ~wm & ~gl; dgl = ndi.distance_transform_edt(~gl) * B
layer[wm] = "white matter"; layer[gl] = "granule layer"; layer[mol & (dgl <= 25)] = "Purkinje layer"; layer[mol & (dgl > 25)] = "molecular layer"
cells["layer"] = ""; cells.loc[c.index, "layer"] = layer[by, bx]
cells["layer_dist_gl_um"] = np.nan; cells.loc[c.index, "layer_dist_gl_um"] = dgl[by, bx]
rep["layer_cells"] = cells.layer.value_counts().to_dict()
# ---------------------------------------------------------------- 1 connexins
cx = ["GJD2", "GJA1", "GJC2", "GJB1", "GJB6", "GJC1", "GJA5", "GJB2"]; cx = [g for g in cx if g in G]
tt = cells[cells.cell_type.notna() & ~cells.cell_type.astype(str).str.startswith("mixed")]
rows = []
for g in cx:
    v = cp10k(g)[tt.index]; s = pd.DataFrame({"t": tt.cell_type.to_numpy(), "v": v}).groupby("t").v
    for t, x in s: rows.append(dict(gene=g, cell_type=t, pct_cells=100 * (x > 0).mean(), mean_cp10k=x.mean()))
con = pd.DataFrame(rows); con.to_csv(Q / "connexins_by_type.csv", index=False)
# ---------------------------------------------------------------- 2 zebrin
pk = cells[(cells.n >= 200) & (cells.f_pc >= 0.043)].copy(); rep["purkinje_cells"] = len(pk)
pk["aldoc"] = np.log1p(cp10k("ALDOC")[pk.index]); pk["plcb4"] = np.log1p(cp10k("PLCB4")[pk.index])
A = np.c_[np.ones(len(pk)), pk.f_glia, pk.f_gran]; beta = np.linalg.lstsq(A, pk.aldoc, rcond=None)[0]; pk["zebrin"] = pk.aldoc - A @ beta
rep["zebrin_regression"] = {"r_aldoc_glia_frac": float(np.corrcoef(pk.aldoc, pk.f_glia)[0, 1]), "beta_glia": float(beta[1]), "beta_granule": float(beta[2])}
rep["zebrin_vs_PLCB4_spearman"] = float(stats.spearmanr(pk.zebrin, pk.plcb4).statistic)
xy = pk[["x_centroid_um", "y_centroid_um"]].to_numpy(); _, nn = cKDTree(xy).query(xy, k=6); zb = pk.zebrin.to_numpy()
obs = stats.spearmanr(zb, zb[nn[:, 1:]].mean(1)).statistic; rng = np.random.default_rng(0)
null = np.array([stats.spearmanr(zb, (p := rng.permutation(zb))[nn[:, 1:]].mean(1)).statistic for _ in range(1000)])
rep["zebrin_neighbour_rho"] = float(obs); rep["zebrin_neighbour_null95"] = float(np.percentile(null, 95)); rep["zebrin_neighbour_p"] = float((1 + (null >= obs).sum()) / 1001)
d_nn = np.linalg.norm(xy[nn[:, 1]] - xy, axis=1); rep["purkinje_nn_distance_median_um"] = float(np.median(d_nn))
excl = set(gs["glia"]) | set(gs["granule"])
def de(idx_a, idx_b, min_frac=0.10):
    Xa, Xb = X[idx_a], X[idx_b]; na, nb = n[idx_a], n[idx_b]
    det = (np.asarray((Xa > 0).mean(0)).ravel() >= min_frac) | (np.asarray((Xb > 0).mean(0)).ravel() >= min_frac)
    out = []
    for j in np.flatnonzero(det):
        g = genes[j]
        if g in excl: continue
        a = 1e4 * np.asarray(Xa[:, j].todense()).ravel() / na; b = 1e4 * np.asarray(Xb[:, j].todense()).ravel() / nb
        out.append(dict(gene=g, mean_a=a.mean(), mean_b=b.mean(), log2fc=np.log2((a.mean() + 0.1) / (b.mean() + 0.1)), p=stats.mannwhitneyu(a, b).pvalue))
    r = pd.DataFrame(out); r["fdr"] = stats.false_discovery_control(r.p); return r.sort_values("p")
q3 = pk.zebrin.quantile([1 / 3, 2 / 3]).to_numpy(); hi, lo = pk.index[pk.zebrin >= q3[1]], pk.index[pk.zebrin <= q3[0]]
zde = de(hi, lo); zde.to_csv(Q / "zebrin_high_vs_low_DE.csv", index=False)
KNOWN = {"PLCB3": "+", "SLC1A6": "+", "HSPB1": "+", "KCTD12": "+", "PLCB4": "−", "TRPC3": "−", "CA8": "?"}
rep["zebrin_known_markers"] = {g: (KNOWN[g], round(float(zde.set_index("gene").log2fc.get(g, np.nan)), 3), float(zde.set_index("gene").fdr.get(g, np.nan))) for g in KNOWN}
rep["zebrin_DE_fdr05"] = int((zde.fdr < 0.05).sum())
# ---------------------------------------------------------------- 5 Purkinje-layer interneurons
inter = cells[cells.cell_type.isin(["MLI (basket / stellate)", "Golgi"])]
pli, mli = inter.index[inter.layer == "Purkinje layer"], inter.index[inter.layer == "molecular layer"]
rep["interneurons_by_layer"] = inter.groupby(["cell_type", "layer"]).size().unstack(fill_value=0).to_dict()
pde = de(pli, mli); pde.to_csv(Q / "PCL_vs_ML_interneurons_DE.csv", index=False); rep["pli_DE_fdr05"] = int((pde.fdr < 0.05).sum())
json.dump(rep, open(Q / "oq_cells_report.json", "w"), indent=1, default=str); print(json.dumps(rep, indent=1, default=str))
print("zebrin top:\n", zde.head(25).round(4).to_string()); print("PCL vs ML interneurons top:\n", pde.head(25).round(4).to_string())
cells[["cell", "layer", "layer_dist_gl_um"]].to_parquet(Q / "cell_layers.parquet", index=False); pk[["cell", "x_centroid_um", "y_centroid_um", "aldoc", "plcb4", "zebrin", "f_glia", "f_gran"]].to_parquet(Q / "purkinje_zebrin.parquet", index=False)
np.save(Q / "layer_map.npy", layer.astype(str))
