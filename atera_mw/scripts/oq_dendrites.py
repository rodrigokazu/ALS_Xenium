#!/usr/bin/env python3
"""Which mRNAs leave the Purkinje soma for the dendrites? Showcase from one section.

For each gene: N_ML = its unassigned (neuropil) transcripts in molecular-layer bins (layer map from oq_cells.py), S_PC = its
transcripts inside the 945 Purkinje cells. Dendritic index = log2((N_ML + 1) / (S_PC + 1)), centred on the median of the
Purkinje-specific genes. Only for Purkinje-specific genes (10x cluster 33 mean >= 3x every other cluster except 13 and 23, which carry Purkinje contamination) can the molecular-layer
neuropil be read as Purkinje dendrite; the same index for Bergmann-glia, interneuron (cluster 14) and granule genes is shown as a
reference for what neuropil looks like for each source. Genes with >= 2,000 soma transcripts; reference classes use the somata of their own type.
python oq_dendrites.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib, zarr
from scipy import sparse, ndimage as ndi
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); O = SSD / "segmentation_full"; Q = O / "open_questions"
MUTED, GRID, BLUE, ORANGE = "#52514e", "#e1e0d9", "#2a78d6", "#eb6834"
layer = np.load(Q / "layer_map.npy", allow_pickle=False); Hh, Ww = layer.shape
nb = pd.read_parquet(Q / "neuropil_bin_gene.parquet"); ok = (nb.by < Hh) & (nb.bx < Ww); nb = nb[ok]
nb["layer"] = layer[nb.by.to_numpy(), nb.bx.to_numpy()]
N_ML = nb[nb.layer == "molecular layer"].groupby("gene").n.sum()
cells = pd.read_parquet(O / "cells.parquet"); X = sparse.load_npz(O / "cell_by_gene.npz").tocsr(); genes = pd.read_csv(O / "genes.csv").gene.astype(str)
pz = pd.read_parquet(Q / "purkinje_zebrin_v2.parquet"); idx = pd.Index(cells.cell).get_indexer(pz.cell)
S_PC = pd.Series(np.asarray(X[idx].sum(0)).ravel(), index=genes)
ann = pd.read_parquet(O / "annotation/cells_annotated.parquet"); ann.index = ann.index.astype(str); ct = cells.cell.astype(str).map(ann.cell_type).astype(str).to_numpy()
own = {"Bergmann glia / astrocyte": np.flatnonzero(np.isin(ct, ["Astrocyte", "mixed: Bergmann glia / Astrocyte"])),
       "Interneuron (MLI)": np.flatnonzero(ct == "MLI (basket / stellate)"), "Granule": np.flatnonzero(ct == "Granule")}
S_OWN = {k: pd.Series(np.asarray(X[v].sum(0)).ravel(), index=genes) for k, v in own.items()}
de = pd.read_csv(SSD / "Cerebellum_sample/analysis/diffexp/gene_expression_graphclust/differential_expression.csv").set_index("Feature Name")
M = de[[f"Cluster {k} Mean Counts" for k in range(1, 37)]]; M.columns = range(1, 37)
def specific(k, fold=5, min_mean=1.0, skip=()):
    other = M.drop(columns=[k, *skip]).max(1); return set(M.index[(M[k] >= min_mean) & (M[k] >= fold * other)])
# 10x clusters 13 (astrocyte) and 23 carry Purkinje transcripts through 10x's own segmentation, so they are left out of the
# reference when deciding Purkinje specificity
GRAN = [1, 2, 3, 4, 5, 8, 9, 10, 15, 6, 7, 11]
classes = {"Purkinje": specific(33, 3, 1.0, skip=(13, 23)), "Bergmann glia / astrocyte": specific(13, 3, 0.5) | specific(16, 3, 0.5) | specific(19, 3, 0.5),
           "Interneuron (MLI)": specific(14, 3, 0.5), "Granule": set(M.index[(M[GRAN].mean(1) >= 0.5) & (M[GRAN].mean(1) >= 3 * M.drop(columns=GRAN + [13, 23]).max(1))])}
rows = []
for cl, gs in classes.items():
    for g in gs:
        soma = S_PC if cl == "Purkinje" else S_OWN[cl]
        if g in N_ML.index and soma.get(g, 0) >= 2000:            # robust soma counts only
            rows.append(dict(gene=g, cls=cl, N_ML=int(N_ML[g]), S_PC=int(soma.get(g, 0))))
d = pd.DataFrame(rows); d["di"] = np.log2((d.N_ML + 1) / (d.S_PC + 1))
med = d[d.cls == "Purkinje"].di.median(); d["di_rel"] = d.di - med
pcd = d[d.cls == "Purkinje"].sort_values("di_rel", ascending=False).reset_index(drop=True); pcd["rank"] = np.arange(1, len(pcd) + 1)
pcd.to_csv(Q / "purkinje_dendritic_index.csv", index=False); d.to_csv(Q / "dendritic_index_all_classes.csv", index=False)
print(len(pcd), "Purkinje-specific genes; top 20 dendritic:\n", pcd.head(20)[["gene", "N_ML", "S_PC", "di_rel"]].to_string(index=False))
print("bottom 10 (soma-restricted):\n", pcd.tail(10)[["gene", "N_ML", "S_PC", "di_rel"]].to_string(index=False))
for g in ["PCP2", "ITPR1", "CALB1", "CA8", "PCP4", "GRID2", "CAMK2A"]:
    r = pcd[pcd.gene == g]; print(g, "rank", int(r["rank"].iloc[0]) if len(r) else None, "of", len(pcd), "di_rel", round(float(r.di_rel.iloc[0]), 2) if len(r) else None)
# ---- figure: A ranking of Purkinje genes | B reference classes | C neuropil maps of the top and bottom gene in a molecular-layer crop
fig = plt.figure(figsize=(18, 6.6), dpi=150); gs_ = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.8, 1.0], wspace=0.4)
ax = fig.add_subplot(gs_[0]); ax.scatter(pcd["rank"], pcd.di_rel, s=14, color=BLUE, lw=0); ax.axhline(0, color="#898781", lw=0.8)
lab = pd.concat([pcd.head(10), pcd.tail(6), pcd[pcd.gene.isin(["PCP2", "ITPR1", "CALB1"])]]).drop_duplicates("gene")
for _, r in lab.iterrows(): ax.annotate(r.gene, (r["rank"], r.di_rel), xytext=(4, 0), textcoords="offset points", fontsize=8, color="#0b0b0b", va="center")
ax.set_xlabel("rank among Purkinje-specific genes"); ax.set_ylabel("dendritic index (log2 molecular-layer neuropil / soma, centred)")
ax.set_title(f"A  Which Purkinje mRNAs reach the dendrites ({len(pcd)} Purkinje-specific genes)", loc="left", fontsize=11, fontweight="bold"); ax.grid(axis="y", color=GRID, lw=0.6)
ax = fig.add_subplot(gs_[1]); order = [c for c in classes if (d.cls == c).any()]
data = [d[d.cls == c].di_rel.dropna().to_numpy() for c in order]
bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="#0b0b0b"))
for b in bp["boxes"]: b.set_facecolor("#cde2fb"); b.set_edgecolor(BLUE)
ax.set_yticks(range(1, len(order) + 1), [f"{c}\n({len(x)} genes)" for c, x in zip(order, data)]); ax.invert_yaxis(); ax.axvline(0, color="#898781", lw=0.8)
ax.set_xlabel("log2 molecular-layer neuropil / own-type somata,\nrelative to the Purkinje median"); ax.set_title("B  Reference: other cell types", loc="left", fontsize=11, fontweight="bold")
# C: transcripts of the most dendritic and the most soma-restricted abundant gene in a 150 um crop across the Purkinje layer
import pyarrow.dataset as ds, pyarrow.compute as pc, tifffile
top, bot = "ITPR1", "CA8"                                                   # abundant pair: dendritic (known in mouse) vs soma
UM = 0.2125; X0, Y0, S = 46824 + 350, 17365 + 1050, int(150 / UM)                       # spot3 window: Purkinje layer into the molecular layer
z = zarr.open(tifffile.TiffFile(SSD / "Cerebellum_sample/morphology_focus/ch0000_dapi.ome.tif").aszarr(level=0, series=0), mode="r")
im = np.asarray(z[0, Y0:Y0 + S, X0:X0 + S]).astype(float)
t = ds.dataset(SSD / "work/local_bundle/transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name"], filter=(pc.field("x_location") >= X0 * UM) & (pc.field("x_location") < (X0 + S) * UM)
    & (pc.field("y_location") >= Y0 * UM) & (pc.field("y_location") < (Y0 + S) * UM) & (pc.field("qv") >= 20) & pc.field("feature_name").isin([top, bot])).to_pandas()
ax = fig.add_subplot(gs_[2]); ax.imshow(np.clip((im - 3) / 318, 0, 1), cmap="Greys_r", extent=(0, S * UM, S * UM, 0), alpha=0.55)
for g, colr in ((bot, ORANGE), (top, BLUE)):
    q = t[t.feature_name == g]; ax.scatter(q.x_location - X0 * UM, q.y_location - Y0 * UM, s=2.2, color=colr, lw=0, label=f"{g} ({len(q):,})")
ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False, loc="upper right", markerscale=6, fontsize=9)
ax.set_title(f"C  {top} (index +0.76) vs {bot} (−0.45), Purkinje → molecular layer", loc="left", fontsize=10.5, fontweight="bold")
ax.plot([S * UM - 35, S * UM - 10], [S * UM - 6] * 2, color="#0b0b0b", lw=2.5); ax.text(S * UM - 22, S * UM - 9, "25 µm", ha="center", fontsize=8.5)
fig.suptitle("Dendritic mRNA localisation in human Purkinje cells (one section, not validated)", x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=1.0)
fig.savefig(Q / "purkinje_dendritic_atlas.png", facecolor="white", bbox_inches="tight"); print("saved")
