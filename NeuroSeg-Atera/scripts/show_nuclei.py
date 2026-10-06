#!/usr/bin/env python3
"""Detected nuclei on the 12 tiles (six largest cells of option_C500 per window, 80 um tiles).
Row 1: DAPI alone (whole-sample scaling, gamma 0.6 so dim nuclei show). Row 2: DAPI + 10x's detected nuclei
(yellow = neuron-sized, >= 55 um2, the threshold the rules use; cyan = smaller) + nuclei added by the pipeline's DAPI rescue (magenta).
python show_nuclei.py <compare root> [N=6]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
from shapely.geometry import Polygon
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 8})
UM = 0.2125; C = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LO, HI = 3.0, 321.0       # whole-sample tissue DAPI p1 / p99.7
W = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
nb = pd.read_parquet("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/Cerebellum_sample/nucleus_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
rows = []
for spot in ("spot1_purkinje_left", "spot2_gl_wm_centre"):
    z = np.load(C / "option_C500" / spot / "crop.npz"); img, lab, (x0, y0, x1, y1) = z["img"], z["labels"], z["window"]
    resc = pd.read_parquet(W / "crops_C500" / spot / "rescued_nuclei.parquet")
    areas = np.bincount(lab.ravel()) * UM ** 2; areas[0] = 0; top = np.argsort(areas)[::-1][:N]
    sub = nb[nb.vertex_x.between(x0 * UM - 30, x1 * UM + 30) & nb.vertex_y.between(y0 * UM - 30, y1 * UM + 30)]
    polys = [(cid, Polygon(np.c_[g.vertex_x, g.vertex_y])) for cid, g in sub.groupby("cell_id", sort=False) if len(g) >= 3]
    rpolys = [(cid, np.c_[g.vertex_x, g.vertex_y]) for cid, g in resc.groupby("cell_id", sort=False)] if len(resc) else []
    dapi = np.clip((img[0].astype(float) - LO) / (HI - LO), 0, 1) ** 0.6
    fig, axs = plt.subplots(2, N, figsize=(3.3 * N, 7.4), dpi=110)
    for k, L in enumerate(top):
        rr, cc = np.nonzero(lab == L); cy, cx = int(rr.mean()), int(cc.mean()); h = int(40 / UM)
        sl = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h)); ox, oy = sl[1].start, sl[0].start
        e = (0, (sl[1].stop - ox) * UM, (sl[0].stop - oy) * UM, 0)
        xa, ya = (x0 + ox) * UM, (y0 + oy) * UM
        n_all = n_big = 0
        for j in (0, 1):
            ax = axs[j, k]; ax.imshow(dapi[sl], extent=e, cmap="gray", vmin=0, vmax=1, interpolation="none")
            ax.set_xlim(0, e[1]); ax.set_ylim(e[2], 0); ax.set_xticks([]); ax.set_yticks([])
        for cid, p in polys:
            bx = p.bounds
            if bx[2] < xa or bx[0] > xa + e[1] or bx[3] < ya or bx[1] > ya + e[2]: continue
            c = p.centroid
            if not (xa <= c.x < xa + e[1] and ya <= c.y < ya + e[2]): continue
            big = p.area >= 55; n_all += 1; n_big += big
            xx, yy = np.array(p.exterior.xy[0]) - xa, np.array(p.exterior.xy[1]) - ya
            axs[1, k].plot(xx, yy, color="#f5c400" if big else "#5fd0ff", lw=1.0)
        n_resc = 0
        for cid, xy in rpolys:
            c = xy.mean(0)
            if xa <= c[0] < xa + e[1] and ya <= c[1] < ya + e[2]:
                n_resc += 1; axs[1, k].plot(np.r_[xy[:, 0], xy[0, 0]] - xa, np.r_[xy[:, 1], xy[0, 1]] - ya, color="#ff4fd8", lw=1.2)
        axs[0, k].set_title(f"{spot[:5]} #{k + 1}: cell {areas[L]:.0f} µm²", loc="left", fontsize=8)
        axs[1, k].set_title(f"{n_all} nuclei ({n_big} neuron-sized" + (f", {n_resc} rescued" if n_resc else "") + ")", loc="left", fontsize=8)
        rows.append({"window": spot[:5], "tile": k + 1, "cell_um2": round(areas[L]), "nuclei": n_all, "neuron_sized": int(n_big), "rescued": n_resc})
    fig.suptitle(f"{spot}: top = DAPI alone (whole-sample scale, gamma 0.6); bottom = detected nuclei: yellow neuron-sized (>= 55 µm²), cyan smaller, magenta added by DAPI rescue. 80 µm tiles", x=0.005, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(C / f"nuclei_tiles_{spot[:5]}.png", facecolor="white"); plt.close(fig)
print(pd.DataFrame(rows).to_string(index=False))
