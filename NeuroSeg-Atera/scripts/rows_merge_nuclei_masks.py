#!/usr/bin/env python3
"""One row per window: merge | nuclei | masks in random colours.
python rows_merge_nuclei_masks.py <compare root> [variant dir = option_C500] [label]
Merge: DAPI blue, boundary magenta, 18S yellow, aSMA/Vim green (whole-sample scaling).
Nuclei: DAPI alone (whole-sample scaling, gamma 0.6) with 10x's detected nuclei outlined (yellow = neuron-sized >= 55 um2,
cyan = smaller) and nuclei added by the pipeline's DAPI rescue (magenta).
Masks: the variant's final cell labels, one colour per cell, black = no cell."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
from matplotlib.collections import LineCollection
from shapely.geometry import Polygon
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 9})
UM = 0.2125; C = Path(sys.argv[1]); VAR = sys.argv[2] if len(sys.argv) > 2 else "option_C500"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "combined 18S + density"
PCT = [(3.0, 321.0), (2.0, 112.0), (2.0, 608.0), (6.0, 363.0)]; RGB = np.array([[.25, .45, 1], [.9, .2, .8], [1, .8, .1], [.2, .9, .3]])
W = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
SPOTS = [("spot1_purkinje_left", "spot 1"), ("spot2_gl_wm_centre", "spot 2"), ("spot3_purkinje_right", "spot 3")]
nbd = pd.read_parquet("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/Cerebellum_sample/nucleus_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
fig, axs = plt.subplots(len(SPOTS), 3, figsize=(21, 7.15 * len(SPOTS)), dpi=130)
for r, (spot, name) in enumerate(SPOTS):
    z = np.load(C / VAR / spot / "crop.npz"); img, lab, (x0, y0, x1, y1) = z["img"], z["labels"], z["window"]
    ext = (0, (x1 - x0) * UM, (y1 - y0) * UM, 0)
    merge = np.zeros(img.shape[1:] + (3,))
    for i in range(4):
        lo, hi = PCT[i]; merge += np.clip((img[i].astype(float) - lo) / (hi - lo), 0, 1)[..., None] * RGB[i]
    merge = np.clip(merge, 0, 1) * 0.85
    dapi = np.clip((img[0].astype(float) - PCT[0][0]) / (PCT[0][1] - PCT[0][0]), 0, 1) ** 0.6
    # nuclei in the window
    sub = nbd[nbd.vertex_x.between(x0 * UM - 20, x1 * UM + 20) & nbd.vertex_y.between(y0 * UM - 20, y1 * UM + 20)]
    big, small = [], []
    for cid, g in sub.groupby("cell_id", sort=False):
        if len(g) < 3: continue
        xy = np.c_[g.vertex_x - x0 * UM, g.vertex_y - y0 * UM]
        (big if Polygon(xy).area >= 55 else small).append(np.vstack([xy, xy[:1]]))
    rp = W / f"crops_{VAR[7:]}" / spot / "rescued_nuclei.parquet"
    resc = []
    if rp.exists():
        rs = pd.read_parquet(rp)
        for cid, g in rs.groupby("cell_id", sort=False):
            xy = np.c_[g.vertex_x - x0 * UM, g.vertex_y - y0 * UM]; resc.append(np.vstack([xy, xy[:1]]))
    # masks in random colours
    n = int(lab.max()); rng = np.random.default_rng(7)
    h = (np.arange(n + 1) * 0.618033988) % 1.0; s = rng.uniform(0.55, 0.95, n + 1); v = rng.uniform(0.80, 1.0, n + 1)
    lut = matplotlib.colors.hsv_to_rgb(np.c_[h, s, v]); lut[0] = 0
    colour = lut[lab]
    ncell = len(np.unique(lab[lab > 0]))
    axs[r, 0].imshow(merge, extent=ext, interpolation="none")
    axs[r, 0].set_title(f"{name}: merge (DAPI blue, boundary magenta, 18S yellow, aSMA/Vim green)", loc="left", fontsize=9)
    axs[r, 1].imshow(dapi, extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="none")
    for segs, col, lw in ((small, "#5fd0ff", 0.5), (big, "#f5c400", 0.8), (resc, "#ff4fd8", 0.9)):
        if segs: axs[r, 1].add_collection(LineCollection(segs, colors=col, linewidths=lw))
    axs[r, 1].set_title(f"{name}: DAPI + detected nuclei ({len(small) + len(big):,}; yellow = neuron-sized {len(big)}, magenta = rescued {len(resc)})", loc="left", fontsize=9)
    axs[r, 2].imshow(colour, extent=ext, interpolation="none")
    axs[r, 2].set_title(f"{name}: masks, {LABEL} ({ncell:,} cells, one colour each)", loc="left", fontsize=9)
    for ax in axs[r]:
        ax.set_xlim(0, ext[1]); ax.set_ylim(ext[2], 0); ax.set_xticks([0, 100, 200, 300, 400, 500]); ax.set_yticks([0, 100, 200, 300, 400, 500])
        ax.tick_params(labelsize=7, colors="#898781", length=2)
        for sp in ax.spines.values(): sp.set_color("#c3c2b7")
        ax.set_xlabel("µm", fontsize=7, color="#898781")
fig.tight_layout(); out = C / f"rows_{VAR[7:]}.png"; fig.savefig(out, facecolor="white"); print("saved", out)
