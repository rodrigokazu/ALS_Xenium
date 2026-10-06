#!/usr/bin/env python3
"""Overview of the full-sample segmentation: cell density map | Purkinje cells | 500 um zoom of the final masks (spot3 window).
Purkinje cell = >= 200 transcripts and >= 4.3% of them from the Purkinje genes (midpoint of 10x clusters 33 and 14) (sets A + B of pc_eval.py).
python full_overview.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd, zarr, tifffile, matplotlib
from scipy import sparse
from skimage.segmentation import find_boundaries
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); O = SSD / "segmentation_full"
cells = pd.read_parquet(O / "cells.parquet"); X = sparse.load_npz(O / "cell_by_gene.npz").tocsr(); genes = pd.read_csv(O / "genes.csv").gene
gs = json.load(open(SSD / "compare/purkinje/gene_sets.json")); pcg = np.isin(genes, gs["purkinje_A"] + gs["purkinje_B"])
n = np.asarray(X.sum(1)).ravel(); fpc = np.asarray(X[:, pcg].sum(1)).ravel() / np.maximum(n, 1)
cells["n"] = n; cells["purkinje"] = (n >= 200) & (fpc >= 0.043)   # midpoint between 10x Purkinje cluster 33 (6.1%) and interneuron cluster 14 (2.5%)
summ = json.load(open(O / "summary.json"))
fig = plt.figure(figsize=(18, 4.6), dpi=150); gsp = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.6, 1], wspace=0.06)
B = 50.0; ext = (0, 75409 * UM, 42511 * UM, 0)
h, _, _ = np.histogram2d(cells.y_centroid_um, cells.x_centroid_um, bins=[int(42511 * UM / B), int(75409 * UM / B)], range=[[0, 42511 * UM], [0, 75409 * UM]])
ax = fig.add_subplot(gsp[0]); A0 = ax; im = ax.imshow(h, extent=ext, cmap="Greys", vmin=0, vmax=np.percentile(h[h > 0], 99))
ax.set_title(f"Cells per 50 µm bin   ({len(cells):,} cells, {summ['pct_assigned']:.1f}% of QC transcripts in cells)", loc="left", fontsize=10.5)
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
ax = fig.add_subplot(gsp[1]); A1 = ax; ax.imshow(h > 0, extent=ext, cmap="Greys", vmin=0, vmax=4)
p = cells[cells.purkinje]; ax.scatter(p.x_centroid_um, p.y_centroid_um, s=1.2, color="#d95926", lw=0)
ax.set_title(f"Purkinje cells ({len(p):,}; ≥ 4.3% Purkinje genes)", loc="left", fontsize=10.5)
for a in (A0, A1): a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
# zoom: the spot3 window at full resolution
x0, y0, x1, y1 = 46824, 17365, 49176, 19718
lab = np.asarray(zarr.open_array(str(O / "final_labels.zarr"), mode="r")[y0:y1, x0:x1])
z = zarr.open(tifffile.TiffFile(SSD / "Cerebellum_sample/morphology_focus/ch0000_dapi.ome.tif").aszarr(level=0, series=0), mode="r")
img = np.asarray(z[:, y0:y1, x0:x1]).astype(float)
mg = np.clip(np.clip((img[0] - 3) / 318, 0, 1)[..., None] * np.array([.22, .42, 1]) + np.clip((img[2] - 2) / 606, 0, 1)[..., None] * np.array([1, .82, .12]), 0, 1) * 0.8
mg[find_boundaries(lab, mode="inner") & (lab > 0)] = 1.0
pcl = set(cells.cell[cells.purkinje].astype(np.int64)); pm = np.isin(lab, list(pcl)); mg[pm] = 0.5 * mg[pm] + 0.5 * np.array(matplotlib.colors.to_rgb("#d95926"))
ax = fig.add_subplot(gsp[2]); ax.imshow(mg, extent=(0, (x1 - x0) * UM, (y1 - y0) * UM, 0)); ax.set_xticks([]); ax.set_yticks([])
ax.plot([400, 500], [480, 480], color="white", lw=3); ax.text(450, 470, "100 µm", color="white", ha="center", fontsize=9)
ax.set_title("500 µm zoom: final masks (white), Purkinje cells orange", loc="left", fontsize=10.5)
fig.suptitle("Atera human cerebellum, full section: I2 identity-gated segmentation + Purkinje repair", x=0.01, y=1.02, ha="left", fontsize=12, fontweight="bold")
fig.savefig(O / "overview.png", facecolor="white", bbox_inches="tight"); print("saved", O / "overview.png", "purkinje cells", len(p))
