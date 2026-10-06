#!/usr/bin/env python3
"""Presentation figure for one window: merge | 10x segmentation | ours, full window (top) and a 120 um zoom (bottom).
python nice_spot.py <compare root> <spot> <ours variant dir> "<ours label>" <zoom_x_um> <zoom_y_um> [output name]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
from matplotlib.patches import Rectangle
from skimage.draw import polygon as draw_poly
from skimage.segmentation import find_boundaries
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 10})
UM = 0.2125; C = Path(sys.argv[1]); spot = sys.argv[2]; VAR = sys.argv[3]; OURS = sys.argv[4]; zx, zy = float(sys.argv[5]), float(sys.argv[6])
name = sys.argv[7] if len(sys.argv) > 7 else f"nice_{spot[:5]}.png"
PCT = [(3.0, 321.0), (2.0, 112.0), (2.0, 608.0), (6.0, 363.0)]; RGB = np.array([[.22, .42, 1], [.92, .25, .78], [1, .82, .12], [.2, .9, .35]])
z = np.load(C / VAR / spot / "crop.npz"); img, ours, (x0, y0, x1, y1) = z["img"], z["labels"], z["window"]
H, W = ours.shape
merge = np.zeros((H, W, 3))
for i in range(4):
    lo, hi = PCT[i]; merge += np.clip((img[i].astype(float) - lo) / (hi - lo), 0, 1)[..., None] * RGB[i]
merge = np.clip(merge, 0, 1)
# 10x polygons -> label image on the window grid
tb = pd.read_parquet(C / VAR / spot / "tenx_cells.parquet"); tenx = np.zeros((H, W), np.int32)
for k, (cid, g) in enumerate(tb.groupby("cell_id", sort=False), start=1):
    if len(g) < 3: continue
    rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - y0, g.vertex_x.to_numpy() / UM - x0, shape=(H, W)); tenx[rr, cc] = k
met = json.load(open(C / VAR / spot / "metrics.json")); b = met["benchmark"]

def paint(lab, seed):
    n = int(lab.max()) + 1; rng = np.random.default_rng(seed)
    hue = (np.arange(n) * 0.618033988 + rng.uniform(0, 1)) % 1.0
    lut = matplotlib.colors.hsv_to_rgb(np.c_[hue, rng.uniform(0.38, 0.62, n), rng.uniform(0.88, 1.0, n)])
    out = np.where((lab > 0)[..., None], 0.28 * merge * 0.9 + 0.72 * lut[lab], merge * 0.32)       # muted fill over dimmed morphology
    edge = find_boundaries(lab, mode="inner") & (lab > 0)
    out[edge] = 0.97
    return out

panels = [merge, paint(tenx, 3), paint(ours, 5)]
titles = ["Morphology merge", f"10x Xenium segmentation   {b['tenx']['n_cells']:,} cells",
          f"{OURS}   {b['new']['n_cells']:,} cells"]
subs = ["DAPI blue · membrane stain magenta · 18S yellow · αSMA/vimentin green",
        f"{b['tenx']['pct_assigned']:.0f}% of transcripts in cells · MECR at 1,000 tx/cell {b['tenx']['mecr_n1000']:.4f}",
        f"{b['new']['pct_assigned']:.0f}% of transcripts in cells · MECR at 1,000 tx/cell {b['new']['mecr_n1000']:.4f}"]
ext = (0, W * UM, H * UM, 0); zoom = (zx, zy, zx + 120, zy + 120)
fig, axs = plt.subplots(2, 3, figsize=(21, 14.6), dpi=150, gridspec_kw=dict(hspace=0.12, wspace=0.03))
for j in range(3):
    for i in range(2):
        ax = axs[i, j]; ax.imshow(panels[j], extent=ext, interpolation="none" if i else "antialiased"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if i == 0:
            ax.set_xlim(0, W * UM); ax.set_ylim(H * UM, 0)
            ax.add_patch(Rectangle((zoom[0], zoom[1]), 120, 120, fill=False, ec="#ffffff", lw=1.4))
            ax.plot([W * UM - 125, W * UM - 25], [H * UM - 22] * 2, color="white", lw=3, solid_capstyle="butt"); ax.text(W * UM - 75, H * UM - 34, "100 µm", color="white", ha="center", va="bottom", fontsize=10)
            ax.set_title(titles[j], loc="left", fontsize=14, fontweight="bold", pad=24)
            ax.text(0, 1.012, subs[j], transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color="#52514e")
        else:
            ax.set_xlim(zoom[0], zoom[2]); ax.set_ylim(zoom[3], zoom[1])
            ax.plot([zoom[2] - 30, zoom[2] - 10], [zoom[3] - 6] * 2, color="white", lw=3, solid_capstyle="butt"); ax.text(zoom[2] - 20, zoom[3] - 8, "20 µm", color="white", ha="center", va="bottom", fontsize=10)
fig.text(0.008, 0.965, f"Atera human cerebellum, 500 × 500 µm window ({spot.split('_', 1)[1].replace('_', ' ')}), x {x0 * UM:.0f}–{x1 * UM:.0f} µm, y {y0 * UM:.0f}–{y1 * UM:.0f} µm. Bottom row: 120 µm zoom at the boundary of the granule layer and the Purkinje layer.",
         fontsize=11, color="#52514e", ha="left")
fig.subplots_adjust(top=0.90, bottom=0.01, left=0.005, right=0.995)
fig.savefig(C / name, facecolor="white"); print("saved", C / name)
