#!/usr/bin/env python3
"""Showcase: two Purkinje cells from held-out windows where the new segmentation wins, and the quantification over all.

Rows A, B: morphology (DAPI blue, 18S yellow) | 10x | ours (I2 + Purkinje repair). Cells overlapping the soma are filled, each
in its own colour; the held-out set-B Purkinje transcripts are drawn as dots in the colour of the cell holding them, red when
no cell holds them. The two cells were chosen (largest held-out somata 10x misses / splits), not drawn at random.
Row C: (1) per soma, share of set-B Purkinje transcripts in the single best cell, 10x vs ours, Wilcoxon signed-rank;
(2) share of somata split across >= 2 cells and with no cell holding >= 10%, exact McNemar; (3) held-out windows: transcripts
in cells and MECR at 1,000 tx per window, exact sign test.
python showcase.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr, matplotlib
from scipy import stats
from skimage.draw import polygon as draw_poly
from skimage.segmentation import find_boundaries
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); D = SSD / "compare/purkinje"
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
TENX, OURS, INK, MUTED, GRID = "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#e1e0d9"
FILL = [matplotlib.colors.to_rgb(c) for c in ["#3987e5", "#1baf7a", "#e87ba4", "#eda100", "#9085e9", "#86b6ef"]]
EX = [("held2", 14, "A  Purkinje cell 10x leaves unassigned"), ("held5", 25, "B  Purkinje cell 10x splits into four")]
setB = set(json.load(open(D / "gene_sets.json"))["purkinje_B"]); so = pd.read_csv(D / "pc_somata.csv"); sc = pd.read_csv(D / "pc_scores.csv")
HALF = int(35 / UM)
fig = plt.figure(figsize=(15.5, 15.2), dpi=150)
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.82], hspace=0.22, wspace=0.05)

def tile(win, soma, row, title):
    s = so[(so.window == win) & (so.soma == soma)].iloc[0]; cx, cy = int(s.cx_px), int(s.cy_px)
    X0, Y0, X1, Y1 = cx - HALF, cy - HALF, cx + HALF, cy + HALF; n = 2 * HALF
    bundle = SSD / "work" / ("local_bundle_heldout" if win.startswith("held") else "local_bundle")
    z = zarr.open(tifffile.TiffFile(bundle / "morphology_focus" / "ch0000_dapi.ome.tif").aszarr(level=0, series=0), mode="r")
    img = np.asarray(z[:, Y0:Y1, X0:X1]).astype(float)
    mg = np.clip(np.clip((img[0] - 3) / 318, 0, 1)[..., None] * np.array([.22, .42, 1]) + np.clip((img[2] - 2) / 606, 0, 1)[..., None] * np.array([1, .82, .12]), 0, 1)
    tx = ds.dataset(bundle / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name", "cell_id"], filter=(pc.field("x_location") >= X0 * UM)
        & (pc.field("x_location") < X1 * UM) & (pc.field("y_location") >= Y0 * UM) & (pc.field("y_location") < Y1 * UM) & (pc.field("qv") >= 20)).to_pandas()
    tb = tx[tx.feature_name.isin(setB)].copy(); tb["px"] = tb.x_location / UM - X0; tb["py"] = tb.y_location / UM - Y0
    cb = ds.dataset(bundle / "cell_boundaries.parquet").to_table(filter=(pc.field("vertex_x") >= X0 * UM - 30) & (pc.field("vertex_x") < X1 * UM + 30)
        & (pc.field("vertex_y") >= Y0 * UM - 30) & (pc.field("vertex_y") < Y1 * UM + 30)).to_pandas()
    tl = np.zeros((n, n), np.int64); ids = {}
    for k, (cid, g) in enumerate(cb.groupby("cell_id", sort=False), start=1):
        rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - Y0, g.vertex_x.to_numpy() / UM - X0, shape=(n, n)); tl[rr, cc] = k; ids[cid] = k
    ours = np.asarray(np.load(RUNS / "crops_PCR_I2_500" / win / "final_labels.npy", mmap_mode="r")[Y0:Y1, X0:X1]).astype(np.int64)
    r = np.clip(tb.py.astype(int), 0, n - 1); c = np.clip(tb.px.astype(int), 0, n - 1)
    tx_lab = {"10x": tb.cell_id.map(ids).fillna(0).astype(np.int64).to_numpy(), "ours": ours[r, c]}
    ext = (0, n * UM, n * UM, 0)
    ax = fig.add_subplot(gs[row, 0]); ax.imshow(mg, extent=ext)
    ax.text(0, 1.035, title, transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0, 1.005, f"{win}, held out · soma {s.area_um2:.0f} µm² · {len(tb):,} held-out Purkinje tx", transform=ax.transAxes, fontsize=9, color=MUTED)
    for j, (name, lb) in enumerate((("10x", tl), ("ours", ours)), start=1):
        ax = fig.add_subplot(gs[row, j]); out = mg * 0.35
        tl_ = tx_lab[name]; counts = pd.Series(tl_[tl_ > 0]).value_counts(); top = list(counts.index[:len(FILL)])
        cmap = {l: FILL[q] for q, l in enumerate(top)}
        for l, colr in cmap.items(): m = lb == l; out[m] = 0.45 * mg[m] + 0.55 * np.array(colr)
        out[find_boundaries(lb, mode="inner") & (lb > 0)] = 0.85
        ax.imshow(out, extent=ext, interpolation="none")
        dc = np.array([cmap.get(l, (0.9, 0.9, 0.9)) if l > 0 else matplotlib.colors.to_rgb("#e34948") for l in tl_])
        ax.scatter(tb.px * UM, tb.py * UM, s=3.5, c=dc, lw=0.25, edgecolors="black")
        k = sc[(sc.window == win) & (sc.soma == soma) & (sc.method == ("10x" if name == "10x" else "I2 + Purkinje repair"))].iloc[0]
        ax.set_title(f"{'10x Xenium' if name == '10x' else 'This work'}:  {100 * k.capture:.0f}% in one cell · {int(k.n_cells)} cell{'s' if k.n_cells != 1 else ''} hold ≥ 10% · {100 * k.orphan:.0f}% in no cell",
                     loc="left", fontsize=9.5, color=INK)
    for a in fig.axes[-3:]:
        a.set_xticks([]); a.set_yticks([]); [x.set_visible(True) for x in a.spines.values()]; [x.set_color("#c3c2b7") for x in a.spines.values()]
        a.plot([n * UM - 14, n * UM - 4], [n * UM - 4] * 2, color="white", lw=3, solid_capstyle="butt"); a.text(n * UM - 9, n * UM - 5.5, "10 µm", color="white", ha="center", fontsize=8.5)

for i, (w, s, t) in enumerate(EX): tile(w, s, i, t)
# ---- C: quantification
p = sc.pivot_table(index=["window", "soma"], columns="method", values=["capture", "n_cells"])
a, b = p["capture"]["10x"], p["capture"]["I2 + Purkinje repair"]
w = stats.wilcoxon(b, a, alternative="two-sided")
ax = fig.add_subplot(gs[2, 0])
for x0, y0 in zip(a, b): ax.plot([0, 1], [x0, y0], color="#c3c2b7", lw=0.6, alpha=0.6, zorder=1)
ax.scatter(np.zeros(len(a)) + np.random.default_rng(0).uniform(-.04, .04, len(a)), a, s=10, color=TENX, zorder=2, lw=0)
ax.scatter(np.ones(len(b)) + np.random.default_rng(1).uniform(-.04, .04, len(b)), b, s=10, color=OURS, zorder=2, lw=0)
for x, v, colr in ((0, a, TENX), (1, b, OURS)): ax.plot([x - .18, x + .18], [v.median()] * 2, color=colr, lw=2.5, solid_capstyle="butt", zorder=3)
ax.set_xticks([0, 1], ["10x", "This work"]); ax.set_xlim(-.4, 1.4); ax.set_ylim(-0.02, 1.02)
ax.set_ylabel("held-out Purkinje transcripts in the single best cell")
ax.set_title(f"C  Every Purkinje soma ({len(a)}, 9 windows)\nmedian {100 * a.median():.0f}% → {100 * b.median():.0f}%, Wilcoxon p = {w.pvalue:.1e}", loc="left", fontsize=10.5)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax = fig.add_subplot(gs[2, 1]); nc = p["n_cells"]
def mcn(x, y):
    b_, c_ = int((x & ~y).sum()), int((~x & y).sum()); return stats.binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
rows = [("split across ≥ 2 cells", nc["10x"] >= 2, nc["I2 + Purkinje repair"] >= 2), ("no cell holds ≥ 10%", nc["10x"] == 0, nc["I2 + Purkinje repair"] == 0)]
for k, (lab, x, y) in enumerate(rows):
    for dx, v, colr in ((-0.19, x.mean(), TENX), (0.19, y.mean(), OURS)):
        ax.bar(k + dx, 100 * v, width=0.34, color=colr); ax.text(k + dx, 100 * v + 1, f"{100 * v:.0f}%", ha="center", fontsize=9, color=INK)
    ax.text(k, max(x.mean(), y.mean()) * 100 + 7, f"McNemar p = {mcn(x, y):.1e}", ha="center", fontsize=8.5, color=MUTED)
ax.set_xticks(range(len(rows)), [r[0] for r in rows]); ax.set_ylabel("share of Purkinje somata (%)"); ax.set_ylim(0, 45)
ax.set_title("Fewer split and fewer missed Purkinje cells", loc="left", fontsize=10.5); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
hd = [plt.Rectangle((0, 0), 1, 1, color=TENX, label="10x"), plt.Rectangle((0, 0), 1, 1, color=OURS, label="This work")]; ax.legend(handles=hd, frameon=False, loc="upper right")
ws = pd.read_csv(D / "win_scores_pcr_v2.csv"); ws = ws[~ws.window.isin(["spot1_purkinje_left", "spot2_gl_wm_centre"])]
pv = ws.pivot_table(index="window", columns="method", values=["pct_assigned", "mecr_n1000"])
ax = fig.add_subplot(gs[2, 2]); ax2 = None
d1 = pv["pct_assigned"]["PCR_I2"] - pv["pct_assigned"]["10x"]; d2 = 100 * (pv["mecr_n1000"]["PCR_I2"] - pv["mecr_n1000"]["10x"]) / pv["mecr_n1000"]["10x"]
st1 = stats.binomtest(int((d1 > 0).sum()), len(d1), 0.5).pvalue; st2 = stats.binomtest(int((d2 < 0).sum()), len(d2), 0.5).pvalue
for k, (d, lab, good) in enumerate(((d1, "transcripts in cells\n(points, higher = better)", d1 > 0), (d2, "MECR at 1,000 tx\n(% change, lower = cleaner)", d2 < 0))):
    ax.scatter(np.full(len(d), k) + np.random.default_rng(k).uniform(-.08, .08, len(d)), d, s=26, color=OURS, lw=0, zorder=3)
    ax.text(k + 0.14, float(d.median()), f"{int(good.sum())}/{len(d)} windows better\nsign test p = {(st1 if k == 0 else st2):.3f}", ha="left", va="center", fontsize=8.5, color=MUTED)
ax.axhline(0, color=INK, lw=0.8); ax.set_ylim(-28, 8); ax.set_xticks([0, 1], ["transcripts in cells\n(+ points vs 10x)", "MECR at 1,000 tx\n(% vs 10x, < 0 cleaner)"]); ax.set_xlim(-.6, 1.6)
ax.set_ylabel("difference to 10x per held-out window"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_title("Whole windows never used for tuning (7)", loc="left", fontsize=10.5)
fig.text(0.01, 0.995, "Atera human cerebellum: new segmentation (identity-gated growth + Purkinje repair) against 10x Xenium", fontsize=13.5, fontweight="bold", va="top")
fig.text(0.01, 0.978, "Dots in A and B: held-out Purkinje genes (never used by the repair), coloured by the cell holding them; red = in no cell. A and B are chosen examples; C shows all somata and windows.",
         fontsize=9.5, color=MUTED, va="top")
fig.subplots_adjust(top=0.94, bottom=0.05, left=0.05, right=0.99)
fig.savefig(D / "showcase_purkinje.png", facecolor="white"); print("saved", D / "showcase_purkinje.png")
print("wilcoxon", w, "split", [(r[0], r[1].mean(), r[2].mean(), mcn(r[1], r[2])) for r in rows], "windows", d1.round(2).to_dict(), d2.round(1).to_dict(), st1, st2)
