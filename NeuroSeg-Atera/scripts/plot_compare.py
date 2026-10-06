#!/usr/bin/env python3
"""10x vs option A (18S) vs option B (transcript density) on the same crop: outlines over the morphology merge.

python plot_compare.py <compare root> <spot> [zoom_x0_um zoom_y0_um]   (compare root holds option_A_18s/ and
option_B_density/, each with <spot>/{crop.npz, metrics.json, tenx_cells.parquet} from compare_crop.py)
"""
import sys, json, os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
from skimage.segmentation import find_boundaries
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 9})
UM = 0.2125
root, spot = Path(sys.argv[1]), sys.argv[2]
zx, zy = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (140.0, 140.0)
PCT = [(3.0, 321.0), (2.0, 112.0), (2.0, 608.0), (6.0, 363.0)]  # whole-sample tissue p1 / p99.7 per stain
RGB = np.array([[0.25, 0.45, 1.0], [0.9, 0.2, 0.8], [1.0, 0.8, 0.1], [0.2, 0.9, 0.3]])

BASE = os.environ.get("BASE", "option_A_18s")
A = np.load(root / BASE / spot / "crop.npz")
img, x0, y0, x1, y1 = A["img"], *A["window"]
merge = np.zeros(img.shape[1:] + (3,))
for i in range(4):
    lo, hi = PCT[i]
    merge += np.clip((img[i].astype(float) - lo) / (hi - lo), 0, 1)[..., None] * RGB[i]
merge = np.clip(merge, 0, 1) * 0.75
ext = (0, (x1 - x0) * UM, (y1 - y0) * UM, 0)
tenx = pd.read_parquet(root / BASE / spot / "tenx_cells.parquet")
panels = [("10x (Xenium Ranger)", None, json.load(open(root / BASE / spot / "metrics.json"))["tenx"])]
VAR = [tuple(v.split(":", 1)) for v in os.environ["VARIANTS"].split(",")] if os.environ.get("VARIANTS") else [("option_A_18s", "Option A: our pipeline on 18S"), ("option_B_density", "Option B: our pipeline on transcript density")]
for key, name in VAR:
    f = root / key / spot
    if (f / "crop.npz").exists():
        panels.append((name, np.load(f / "crop.npz")["labels"], json.load(open(f / "metrics.json"))["new"]))

fig, axs = plt.subplots(2, len(panels), figsize=(6.2 * len(panels), 12.6), dpi=170, squeeze=False)
for j, (name, lab, m) in enumerate(panels):
    for i, (xlim, ylim) in enumerate((((0, ext[1]), (ext[2], 0)), ((zx, zx + 120), (zy + 120, zy)))):
        ax = axs[i, j]
        ax.imshow(merge, extent=ext, interpolation="none")
        if lab is None:
            for _, g in tenx.groupby("cell_id", sort=False):
                ax.plot(g.vertex_x - x0 * UM, g.vertex_y - y0 * UM, color="white", lw=0.35 if i == 0 else 0.8)
        else:
            b = find_boundaries(lab, mode="inner")
            ov = np.zeros(lab.shape + (4,)); ov[b] = (1, 1, 1, 1)
            ax.imshow(ov, extent=ext, interpolation="none")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7, colors="#898781", length=2)
        if i == 0:
            ax.add_patch(plt.Rectangle((zx, zy), 120, 120, fill=False, ec="#f5c400", lw=1.2))
            src = m.get("source_counts")
            extra = ("\n" + ", ".join(f"{k} {v:,}" for k, v in src.items())) if src else ""
            ax.set_title(f"{name}\n{m['n_cells']:,} cells, {100 * m['frac_tx_assigned']:.0f}% transcripts in cells, "
                         f"median {m['median_tx_per_cell']:,.0f} tx / cell{extra}", loc="left", fontsize=9)
        else:
            ax.set_title("120 µm zoom (yellow box)", loc="left", fontsize=8, color="#52514e")
fig.suptitle(f"{spot}: x {x0 * UM:.0f}–{x1 * UM:.0f} µm, y {y0 * UM:.0f}–{y1 * UM:.0f} µm  |  merge: DAPI blue, "
             f"boundary magenta, 18S yellow, αSMA/Vim green (whole-sample contrast)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.965))
out = root / (os.environ.get("OUTPREFIX", "compare") + f"_{spot}.png")
fig.savefig(out, dpi=170, facecolor="white")
print("saved", out)
