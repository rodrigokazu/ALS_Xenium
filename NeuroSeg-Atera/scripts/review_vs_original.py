#!/usr/bin/env python3
"""The N largest cells of option_C500 (tile centres) on spot1 / spot2: the ORIGINAL pipeline (option_A500, standard 18S,
default Params) vs 10x on the same tiles. Per tile: what each method has at the tile centre (area, source, nuclei).
python review_vs_original.py <compare root> [N=6]"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
from shapely.geometry import Polygon, Point
from skimage.segmentation import find_boundaries
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 8})
UM = 0.2125; C = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
PCT = [(3.0, 321.0), (2.0, 112.0), (2.0, 608.0), (6.0, 363.0)]; RGB = np.array([[.25, .45, 1], [.9, .2, .8], [1, .8, .1], [.2, .9, .3]])
W = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
rows = []
for spot in ("spot1_purkinje_left", "spot2_gl_wm_centre"):
    VAR = os.environ.get("VAR"); LBL = os.environ.get("LABEL", "original pipeline")
    zc = np.load(C / "option_C500" / spot / "crop.npz"); za = np.load(C / "option_A500" / spot / "crop.npz")
    zv = np.load(C / VAR / spot / "crop.npz") if VAR else za
    labv = zv["labels"]
    img, labc, laba, (x0, y0, x1, y1) = zc["img"], zc["labels"], za["labels"], zc["window"]
    areas = np.bincount(labc.ravel()) * UM ** 2; areas[0] = 0; top = np.argsort(areas)[::-1][:N]
    ca = pd.read_parquet(W / "crops_A500" / spot / "cells.parquet").set_index("label")
    cv = pd.read_parquet(W / (f"crops_{VAR[7:]}" if VAR else "crops_A500") / spot / "cells.parquet").set_index("label")
    tenx = pd.read_parquet(C / "option_C500" / spot / "tenx_cells.parquet")
    polys = {cid: Polygon(np.c_[g.vertex_x, g.vertex_y]) for cid, g in tenx.groupby("cell_id", sort=False) if len(g) >= 3}
    nuc = pd.read_parquet("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/Cerebellum_sample/nucleus_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
    merge = np.zeros(img.shape[1:] + (3,))
    for i in range(4):
        lo, hi = PCT[i]; merge += np.clip((img[i].astype(float) - lo) / (hi - lo), 0, 1)[..., None] * RGB[i]
    merge = np.clip(merge, 0, 1) * 0.8
    fig, axs = plt.subplots(2, N, figsize=(3.3 * N, 7.4), dpi=110)
    for k, L in enumerate(top):
        rr, cc = np.nonzero(labc == L); cy, cx = int(rr.mean()), int(cc.mean()); h = int(40 / UM)
        sl = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h)); ox, oy = sl[1].start, sl[0].start
        e = (0, (sl[1].stop - ox) * UM, (sl[0].stop - oy) * UM, 0)
        # what each method has at the centre of our cell (its pixel centre; fall back to the nearest label within 4 um)
        def at_centre(lab_img, tab):
            l = lab_img[cy, cx]
            if l == 0:
                win = lab_img[max(cy - 19, 0):cy + 19, max(cx - 19, 0):cx + 19]; v = win[win > 0]
                l = int(np.bincount(v).argmax()) if v.size else 0
            return l, (round((lab_img == l).sum() * UM ** 2) if l else 0), (tab.source.get(l, "none") if l else "none")
        _, o_area, o_src = at_centre(laba, ca)
        la, a_area, a_src = at_centre(labv, cv)
        pt = Point((cx + x0) * UM, (cy + y0) * UM); hit = [(p.area, cid) for cid, p in polys.items() if p.contains(pt)]
        t_area = round(hit[0][0]) if hit else 0
        nn = int(sum(Polygon(np.c_[g.vertex_x, g.vertex_y]).contains(pt) for _, g in nuc[nuc.vertex_x.between(pt.x - 25, pt.x + 25) & nuc.vertex_y.between(pt.y - 25, pt.y + 25)].groupby("cell_id") if len(g) >= 3))
        rows.append({"window": spot[:5], "tile": k + 1, "our_C500_um2": round(areas[L]), "original_um2": o_area, "variant_um2": a_area, "variant_source": a_src, "10x_um2": t_area, "10x_cell_at_centre": bool(hit), "nucleus_at_centre": nn > 0})
        for j in (0, 1):
            ax = axs[j, k]; ax.imshow(merge[sl], extent=e, interpolation="none")
            if j == 0:
                b = find_boundaries(labv[sl], mode="inner"); o = np.zeros(b.shape + (4,)); o[b] = (1, 1, 1, .9)
                if la:
                    bb = find_boundaries(np.where(labv[sl] == la, 1, 0), mode="inner"); o[bb] = (1, .8, 0, 1)
                ax.imshow(o, extent=e, interpolation="none")
                ax.set_title(f"{spot[:5]} #{k + 1}: {LBL}\ncell at centre {a_area} µm² ({a_src})  | original {o_area} | C500 {areas[L]:.0f}", loc="left", fontsize=7)
            else:
                for cid, p in polys.items():
                    xx, yy = np.array(p.exterior.xy[0]) - (x0 + ox) * UM, np.array(p.exterior.xy[1]) - (y0 + oy) * UM
                    if xx.max() > 0 and xx.min() < e[1] and yy.max() > 0 and yy.min() < e[2]:
                        ax.plot(xx, yy, color=(1, .8, 0) if hit and cid == hit[0][1] else "white", lw=1.2 if hit and cid == hit[0][1] else 0.7)
                ax.set_title(f"10x: " + (f"cell at centre {t_area} µm²" if hit else "NO cell at centre") + ("" if nn else "  (no nucleus)"), loc="left", fontsize=7, color="#52514e")
            ax.set_xlim(0, e[1]); ax.set_ylim(e[2], 0); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{spot}: the {N} largest cells of our combined run. Top: {LBL}; bottom: 10x. Yellow = the cell at the tile centre. 80 µm tiles", x=0.005, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(C / (f"{VAR[7:]}_vs_10x_{spot[:5]}.png" if VAR else f"original_vs_10x_{spot[:5]}.png"), facecolor="white"); plt.close(fig)
df = pd.DataFrame(rows); pd.set_option("display.width", 220); print(df.to_string(index=False)); df.to_csv(C / (f"{VAR[7:]}_vs_10x_tiles.csv" if VAR else "original_vs_10x_tiles.csv"), index=False)
