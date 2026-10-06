#!/usr/bin/env python3
"""Gallery of Purkinje somata: 12 drawn at random (seed 0) from all somata pc_eval.py scored, one row each.
Columns: morphology (DAPI blue, 18S yellow) | Purkinje set-A density | 10x | I2 | I2 + Purkinje repair.
Cells overlapping the soma are filled (one colour each), all other cells outlined; the title gives the held-out (set-B)
capture of the best cell. python pc_gallery.py <compare/purkinje dir>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr, matplotlib
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_poly
from skimage.segmentation import find_boundaries
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 9})
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); D = Path(sys.argv[1])
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
FILL = [matplotlib.colors.to_rgb(c) for c in ["#d95926", "#3987e5", "#1baf7a", "#eda100", "#e87ba4", "#9085e9"]]
HALF = int(30 / UM)
sc = pd.read_csv(D / "pc_scores.csv"); so = pd.read_csv(D / "pc_somata.csv"); setA = set(json.load(open(D / "gene_sets.json"))["purkinje_A"])
pick = so.merge(sc[sc.method == "10x"][["window", "soma"]], on=["window", "soma"]).sample(12, random_state=0).reset_index(drop=True)
COLS = [("10x", None), ("I2", "crops_I2_500"), ("I2 + Purkinje repair", "crops_PCR_I2_500")]
fig, axs = plt.subplots(len(pick), 5, figsize=(14.5, 3.0 * len(pick)), dpi=130, gridspec_kw=dict(wspace=0.03, hspace=0.2))
for i, s in pick.iterrows():
    w = s.window; bundle = SSD / "work" / ("local_bundle_heldout" if w.startswith("held") else "local_bundle")
    cx, cy = int(s.cx_px), int(s.cy_px); X0, Y0, X1, Y1 = cx - HALF, cy - HALF, cx + HALF, cy + HALF; n = 2 * HALF
    tf = tifffile.TiffFile(bundle / "morphology_focus" / "ch0000_dapi.ome.tif"); z = zarr.open(tf.aszarr(level=0, series=0), mode="r")
    img = np.asarray(z[:, Y0:Y1, X0:X1]).astype(float)
    mg = np.clip((img[0] - 3) / 318, 0, 1)[..., None] * np.array([.22, .42, 1]) + np.clip((img[2] - 2) / 606, 0, 1)[..., None] * np.array([1, .82, .12])
    mg = np.clip(mg, 0, 1)
    tx = ds.dataset(bundle / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name"], filter=(pc.field("x_location") >= X0 * UM) & (pc.field("x_location") < X1 * UM)
        & (pc.field("y_location") >= Y0 * UM) & (pc.field("y_location") < Y1 * UM) & (pc.field("qv") >= 20)).to_pandas()
    a = tx[tx.feature_name.isin(setA)]; r = np.clip((a.y_location / UM).astype(int) - Y0, 0, n - 1); c = np.clip((a.x_location / UM).astype(int) - X0, 0, n - 1)
    dA = ndi.gaussian_filter(np.bincount(r * n + c, minlength=n * n).reshape(n, n).astype(float), 1.5 / UM) / UM ** 2
    core = ndi.binary_opening(dA >= 0.534, iterations=int(1 / UM)); cl, _ = ndi.label(core); core = cl == cl[HALF, HALF] if cl[HALF, HALF] else core
    cb = ds.dataset(bundle / "cell_boundaries.parquet").to_table(filter=(pc.field("vertex_x") >= X0 * UM - 30) & (pc.field("vertex_x") < X1 * UM + 30)
        & (pc.field("vertex_y") >= Y0 * UM - 30) & (pc.field("vertex_y") < Y1 * UM + 30)).to_pandas()
    tl = np.zeros((n, n), np.int64)
    for k, (_, g) in enumerate(cb.groupby("cell_id", sort=False), start=1):
        rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - Y0, g.vertex_x.to_numpy() / UM - X0, shape=(n, n)); tl[rr, cc] = k
    labs = [tl] + [np.asarray(np.load(RUNS / d / w / "final_labels.npy", mmap_mode="r")[Y0:Y1, X0:X1]).astype(np.int64) for _, d in COLS[1:]]
    ext = (0, n * UM, n * UM, 0)
    axs[i, 0].imshow(mg, extent=ext); axs[i, 1].imshow(dA, extent=ext, cmap="magma", vmin=0, vmax=3)
    for j, ((name, _), lb) in enumerate(zip(COLS, labs), start=2):
        out = mg * 0.45; ov = [l for l in pd.Series(lb[core]).value_counts().index if l > 0][:len(FILL)]
        for q, l in enumerate(ov): m = lb == l; out[m] = 0.35 * mg[m] + 0.65 * np.array(FILL[q])
        out[find_boundaries(lb, mode="inner") & (lb > 0)] = 0.92
        axs[i, j].imshow(out, extent=ext, interpolation="none")
        cap = sc[(sc.window == w) & (sc.soma == s.soma) & (sc.method == name)]
        t = f"{100 * cap.capture.iloc[0]:.0f}% in one cell · {int(cap.n_cells.iloc[0])} cells" if len(cap) else ""
        axs[i, j].set_title(t, loc="left", fontsize=8.5, color="#52514e")
    b = find_boundaries(core.astype(int), mode="inner")
    for j in range(5):
        ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([]); [x.set_visible(False) for x in ax.spines.values()]
        if j: ax.contour(np.linspace(0, n * UM, n), np.linspace(0, n * UM, n), core.astype(float), levels=[0.5], colors="white", linewidths=0.6, linestyles="dashed")
        ax.plot([n * UM - 13, n * UM - 3], [n * UM - 3] * 2, color="white", lw=2.5)
    axs[i, 0].set_ylabel(f"{w}\nsoma {int(s.soma)} · {s.area_um2:.0f} µm²", fontsize=8.5)
for j, t in enumerate(["Morphology (DAPI, 18S)", "Purkinje genes, set A", "10x", "I2", "I2 + Purkinje repair"]):
    axs[0, j].text(0, 1.18, t, transform=axs[0, j].transAxes, fontsize=11, fontweight="bold")
fig.text(0.01, 0.995, "Twelve Purkinje somata drawn at random from the nine windows (60 µm tiles, scale bar 10 µm). Dashed = soma from set-A genes; filled = cells that overlap it. "
         "Titles: share of the soma's held-out set-B Purkinje transcripts in the single best cell.", fontsize=9.5, color="#52514e", va="top")
fig.subplots_adjust(top=0.965, bottom=0.005, left=0.06, right=0.995)
fig.savefig(D / "purkinje_gallery.png", facecolor="white"); print("saved", D / "purkinje_gallery.png")
