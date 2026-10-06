#!/usr/bin/env python3
"""Spot 3 (never tuned on): where do the transcripts go? Left morphology, middle 10x, right ours (frozen settings).
Orphan transcripts (not in any cell) are red; bottom row zooms on the large Purkinje-marker cell with the weakest 10x match.
python convince_fig.py <compare root> <PF crops dir> <out.png>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_poly
from skimage.segmentation import find_boundaries
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 10})
UM = 0.2125; C = Path(sys.argv[1]); PF = Path(sys.argv[2]); OUT = sys.argv[3]; spot = "spot3_purkinje_right"
z = np.load(C / "option_C500" / spot / "crop.npz"); img, (x0, y0, x1, y1) = z["img"], z["window"]
ours = np.asarray(np.load(PF / spot / "final_labels.npy", mmap_mode="r")[y0:y1, x0:x1]).astype(np.int64)
H, W = ours.shape
PCT = [(3.0, 321.0), (2.0, 112.0), (2.0, 608.0), (6.0, 363.0)]; RGB = np.array([[.22, .42, 1], [.92, .25, .78], [1, .82, .12], [.2, .9, .35]])
merge = np.zeros((H, W, 3))
for i in range(4):
    lo, hi = PCT[i]; merge += np.clip((img[i].astype(float) - lo) / (hi - lo), 0, 1)[..., None] * RGB[i]
merge = np.clip(merge, 0, 1)
tb = pd.read_parquet(C / "option_C500" / spot / "tenx_cells.parquet"); tenx = np.zeros((H, W), np.int64)
for k, (cid, g) in enumerate(tb.groupby("cell_id", sort=False), start=1):
    if len(g) >= 3:
        rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - y0, g.vertex_x.to_numpy() / UM - x0, shape=(H, W)); tenx[rr, cc] = k
t = pd.read_parquet(C / "option_C500" / spot / "tx_sub.parquet")
t["px"] = t.x_location - x0 * UM; t["py"] = t.y_location - y0 * UM
u = (t.x_location / UM - x0).astype(int).clip(0, W - 1); v = (t.y_location / UM - y0).astype(int).clip(0, H - 1)
t["in_ten"] = tenx[v, u] > 0; t["in_new"] = ours[v, u] > 0; t["lab"] = ours[v, u]
n = len(t); r10, rn = t.in_ten.mean() * 100, t.in_new.mean() * 100
# candidate cell: large in ours, mostly uncovered by 10x, rich in Purkinje markers
PK = {"PCP2", "CALB1", "ITPR1", "CA8", "PVALB"}
t["pk"] = t.feature_name.isin(PK)
area = ndi.sum(np.ones_like(ours), ours, index=np.arange(ours.max() + 1)) * UM ** 2
best = None
for lab in np.where(area > 90)[0]:
    if lab == 0: continue
    m = ours == lab; cov = (tenx[m] > 0).mean(); npk = int((t.lab == lab).mul(t.pk).sum())
    ntx = int((t.lab == lab).sum())
    if cov < 0.6:
        sc = area[lab] * (1 - cov)
        if best is None or sc > best[0]: best = (sc, lab, area[lab], npk, ntx)
print("zoom cell", best, "n>200:", (area > 90).sum(), "max area", area.max())
lab = best[1]; cy, cx = ndi.center_of_mass(ours == lab); zc = (cx * UM, cy * UM); half = 30
def paint(L, seed, dim=0.34):
    nn = int(L.max()) + 1; rng = np.random.default_rng(seed)
    hue = (np.arange(nn) * 0.618033988 + rng.uniform(0, 1)) % 1.0
    lut = matplotlib.colors.hsv_to_rgb(np.c_[hue, rng.uniform(.38, .62, nn), rng.uniform(.88, 1, nn)])
    out = np.where((L > 0)[..., None], 0.28 * merge * 0.9 + 0.72 * lut[L], merge * dim)
    out[find_boundaries(L, mode="inner") & (L > 0)] = 0.97
    return out
RED = "#e03a2f"; ext = (0, W * UM, H * UM, 0)
fig, axs = plt.subplots(2, 3, figsize=(21, 14.8), dpi=150, gridspec_kw=dict(hspace=0.10, wspace=0.03))
pan = [merge * 0.9, paint(tenx, 3), paint(ours, 5)]
sub = ["DAPI blue · membrane magenta · 18S yellow · αSMA/vimentin green",
       f"{r10:.0f}% of transcripts in a cell · {100 - r10:.0f}% orphan (red)", f"{rn:.0f}% of transcripts in a cell · {100 - rn:.0f}% orphan (red)"]
tit = ["Morphology", "10x Xenium segmentation", "This work, frozen settings"]
for j in range(3):
    for i in range(2):
        ax = axs[i, j]; ax.imshow(pan[j], extent=ext, interpolation="none" if i else "antialiased"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if j > 0:
            orph = t[~(t.in_ten if j == 1 else t.in_new)]
            sel = orph if i == 0 else orph[(orph.px.sub(zc[0]).abs() < half) & (orph.py.sub(zc[1]).abs() < half)]
            if i == 0: sel = sel.sample(min(len(sel), 9000), random_state=0)
            ax.scatter(sel.px, sel.py, s=3 if i == 0 else 14, color=RED, lw=0, alpha=0.85, zorder=5)
        if j == 0 and i == 1:
            m = t[t.pk & (t.px.sub(zc[0]).abs() < half) & (t.py.sub(zc[1]).abs() < half)]
            ax.scatter(m.px, m.py, s=16, color="white", lw=0, zorder=5)
        if i == 0:
            ax.set_xlim(0, W * UM); ax.set_ylim(H * UM, 0)
            ax.add_patch(plt.Rectangle((zc[0] - half, zc[1] - half), 2 * half, 2 * half, fill=False, ec="white", lw=1.4))
            ax.set_title(tit[j], loc="left", fontsize=14, fontweight="bold", pad=24); ax.text(0, 1.012, sub[j], transform=ax.transAxes, fontsize=10.5, color="#52514e", va="bottom")
            ax.plot([W * UM - 125, W * UM - 25], [H * UM - 22] * 2, color="white", lw=3, solid_capstyle="butt"); ax.text(W * UM - 75, H * UM - 34, "100 µm", color="white", ha="center", fontsize=10)
        else:
            ax.set_xlim(zc[0] - half, zc[0] + half); ax.set_ylim(zc[1] + half, zc[1] - half)
            if j == 0: ax.set_title(f"Zoom: Purkinje markers (PCP2, CALB1, ITPR1, CA8, PVALB) white", loc="left", fontsize=10.5, color="#52514e")
            ax.plot([zc[0] + half - 15, zc[0] + half - 5], [zc[1] + half - 3] * 2, color="white", lw=3, solid_capstyle="butt"); ax.text(zc[0] + half - 10, zc[1] + half - 4, "10 µm", color="white", ha="center", va="bottom", fontsize=9)
fig.text(0.008, 0.965, f"Atera human cerebellum, spot 3 (500 µm window, not used for tuning). Red dots: transcripts (random 300k subsample of the window) that fall in no cell. Bottom: the large cell of ours with the weakest 10x coverage, {best[3]} Purkinje-marker transcripts.",
         fontsize=11, color="#52514e")
fig.subplots_adjust(top=0.90, bottom=0.01, left=0.005, right=0.995)
fig.savefig(OUT, facecolor="white"); print("saved", OUT)
