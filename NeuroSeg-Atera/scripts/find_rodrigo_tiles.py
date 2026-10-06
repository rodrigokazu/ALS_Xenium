#!/usr/bin/env python3
"""Candidate tiles for Rodrigo's panel b boxes: the gene-defined Purkinje cells (>= 4.3% Purkinje genes) nearest to the box
centres estimated from his panel a, rendered as DAPI + 18S with Ranger (magenta) and our (gold) outlines, numbered for matching."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr, matplotlib
from scipy import sparse
from skimage.draw import polygon as draw_poly
from skimage.segmentation import find_boundaries
matplotlib.use("Agg"); import matplotlib.pyplot as plt
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); O = SSD / "segmentation_full"; B = SSD / "Cerebellum_sample"
cells = pd.read_parquet(O / "cells.parquet"); X = sparse.load_npz(O / "cell_by_gene.npz").tocsr(); genes = pd.read_csv(O / "genes.csv").gene.astype(str)
gs = json.load(open(SSD / "compare/purkinje/gene_sets.json")); n = np.asarray(X.sum(1)).ravel()
f = np.asarray(X[:, np.isin(genes, gs["purkinje_A"] + gs["purkinje_B"])].sum(1)).ravel() / np.maximum(n, 1)
pk = cells[(n >= 200) & (f >= 0.043)]
z = zarr.open(tifffile.TiffFile(B / "morphology_focus" / "ch0000_dapi.ome.tif").aszarr(level=0, series=0), mode="r"); lab = zarr.open_array(str(O / "final_labels.zarr"), mode="r")
HALF = int(35 / UM); N = 2 * HALF
for name, (ex, ey) in {"tile1": (3211, 4175), "tile2": (9279, 5115)}.items():
    d = np.hypot(pk.x_centroid_um - ex, pk.y_centroid_um - ey); cand = pk.assign(d=d).sort_values("d").head(12)
    fig, axs = plt.subplots(3, 4, figsize=(13, 10), dpi=110)
    for ax, (_, c) in zip(axs.ravel(), cand.iterrows()):
        X0, Y0 = int(c.x_centroid_um / UM) - HALF, int(c.y_centroid_um / UM) - HALF
        im = np.asarray(z[:, Y0:Y0 + N, X0:X0 + N]).astype(float)
        mg = np.clip(np.clip((im[0] - 3) / 318, 0, 1)[..., None] * np.array([.22, .42, 1]) + np.clip((im[2] - 2) / 606, 0, 1)[..., None] * np.array([1, .82, .12]) * 0.8, 0, 1)
        cb = ds.dataset(B / "cell_boundaries.parquet").to_table(filter=(pc.field("vertex_x") >= X0 * UM - 30) & (pc.field("vertex_x") < (X0 + N) * UM + 30)
            & (pc.field("vertex_y") >= Y0 * UM - 30) & (pc.field("vertex_y") < (Y0 + N) * UM + 30)).to_pandas()
        rl = np.zeros((N, N), np.int64)
        for k, (_, g) in enumerate(cb.groupby("cell_id", sort=False), start=1):
            rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - Y0, g.vertex_x.to_numpy() / UM - X0, shape=(N, N)); rl[rr, cc] = k
        ol = np.asarray(lab[Y0:Y0 + N, X0:X0 + N]).astype(np.int64)
        mg[find_boundaries(rl, mode="inner") & (rl > 0)] = (0.8, 0.2, 0.6); mg[find_boundaries(ol, mode="inner") & (ol > 0)] = (0.9, 0.62, 0)
        ax.imshow(mg); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"cell {int(c.cell)} · {c.d:.0f} µm off\nx {c.x_centroid_um:.0f}, y {c.y_centroid_um:.0f} µm", fontsize=8)
    fig.suptitle(f"{name}: nearest Purkinje cells to the estimated box centre ({ex}, {ey} µm)", x=0.01, ha="left")
    out = SSD / f"compare/purkinje/rodrigo_{name}_candidates.png"; fig.savefig(out, bbox_inches="tight", facecolor="white"); print("saved", out)
