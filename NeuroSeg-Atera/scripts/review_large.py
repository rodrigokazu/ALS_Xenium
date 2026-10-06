#!/usr/bin/env python3
"""Large-cell review: the N largest cells of a segmentation on each 500 um window, against 10x's outlines.
Per cell: area, QC transcripts, nuclei inside (and neuron-sized, >= 55 um2), marker-class composition. A cell with
two or more neuron-sized nuclei, or with strong markers of two classes, is flagged as a possible merge.
python review_large.py <compare root> <variant dir, e.g. option_C500> [N=6]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, matplotlib
from shapely.geometry import Polygon
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 8})
UM = 0.2125; C = Path(sys.argv[1]); VAR = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 6
B = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/work/local_bundle")
CLASSES = {"Purkinje": ["CALB1", "PCP2", "PCP4", "ITPR1", "CA8", "GRID2"], "granule": ["GABRA6", "NEUROD1", "CBLN3"],
           "myelin": ["MBP", "MOBP", "PLP1", "MOG"], "astro": ["SLC1A3", "GFAP", "AQP4", "GJA1"], "immune": ["CX3CR1", "P2RY12", "PTPRC"],
           "GABA": ["GAD1", "GAD2"]}
ALLM = sorted({g for v in CLASSES.values() for g in v})
PCT = [(3.0, 321.0), (2.0, 112.0), (2.0, 608.0), (6.0, 363.0)]; RGB = np.array([[.25, .45, 1], [.9, .2, .8], [1, .8, .1], [.2, .9, .3]])
nbd = pd.read_parquet(B / "nucleus_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
rows = []
SPOTS = ["spot1_purkinje_left", "spot2_gl_wm_centre", "spot3_purkinje_right"]
for si, spot in enumerate(SPOTS):
    fig, axs = plt.subplots(2, N, figsize=(3.3 * N, 7.4), dpi=110)
    z = np.load(C / VAR / spot / "crop.npz"); img, lab, (x0, y0, x1, y1) = z["img"], z["labels"], z["window"]
    areas = np.bincount(lab.ravel()) * UM ** 2; areas[0] = 0
    top = np.argsort(areas)[::-1][:N]
    # transcripts of marker genes in the window, label under each
    t = ds.dataset(B / "transcripts.parquet").to_table(
        columns=["x_location", "y_location", "feature_name", "qv", "is_gene"],
        filter=(pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM)
        & pc.field("feature_name").isin(ALLM) & (pc.field("qv") >= 20)).to_pandas()
    t["feature_name"] = t.feature_name.astype(str)
    col = np.clip((t.x_location / UM).astype(int) - x0, 0, lab.shape[1] - 1); row = np.clip((t.y_location / UM).astype(int) - y0, 0, lab.shape[0] - 1)
    t["lab"] = lab[row, col]
    # n total transcripts per label need all genes: use the bincount of the stored run? approximate with tx_sub scale is poor, so count via the dataset once
    tot = ds.dataset(B / "transcripts.parquet").to_table(columns=["x_location", "y_location", "qv", "is_gene"],
        filter=(pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM) & (pc.field("qv") >= 20)).to_pandas()
    tcol = np.clip((tot.x_location / UM).astype(int) - x0, 0, lab.shape[1] - 1); trow = np.clip((tot.y_location / UM).astype(int) - y0, 0, lab.shape[0] - 1)
    ntx = np.bincount(lab[trow, tcol], minlength=lab.max() + 1)
    nb = nbd[nbd.vertex_x.between(x0 * UM - 30, x1 * UM + 30) & nbd.vertex_y.between(y0 * UM - 30, y1 * UM + 30)]
    nuc = [(cid, Polygon(np.c_[g.vertex_x, g.vertex_y])) for cid, g in nb.groupby("cell_id", sort=False) if len(g) >= 3]
    cent = np.array([[p.centroid.x, p.centroid.y] for _, p in nuc]); narea = np.array([p.area for _, p in nuc])
    ncol = np.clip((cent[:, 0] / UM).astype(int) - x0, 0, lab.shape[1] - 1); nrow = np.clip((cent[:, 1] / UM).astype(int) - y0, 0, lab.shape[0] - 1)
    nlab = np.where((cent[:, 0] >= x0 * UM) & (cent[:, 0] < x1 * UM) & (cent[:, 1] >= y0 * UM) & (cent[:, 1] < y1 * UM), lab[nrow, ncol], 0)
    tenx = pd.read_parquet(C / VAR / spot / "tenx_cells.parquet")
    merge = np.zeros(img.shape[1:] + (3,))
    for i in range(4):
        lo, hi = PCT[i]; merge += np.clip((img[i].astype(float) - lo) / (hi - lo), 0, 1)[..., None] * RGB[i]
    merge = np.clip(merge, 0, 1) * 0.8
    for k, L in enumerate(top):
        m = t[t.lab == L].feature_name.value_counts()
        cls = {c: int(sum(m.get(g, 0) for g in gs)) for c, gs in CLASSES.items()}
        s = sorted(cls.items(), key=lambda kv: -kv[1]); tot_m = max(sum(cls.values()), 1)
        inside = (nlab == L); nn = int((narea[inside] >= 55).sum())
        flag = []
        if nn >= 2: flag.append(f"{nn} neuron-sized nuclei")
        if s[1][1] >= 0.25 * max(s[0][1], 1) and s[1][1] >= 15: flag.append(f"{s[0][0]}+{s[1][0]}")
        rr, cc = np.nonzero(lab == L); cy, cx = int(rr.mean()), int(cc.mean()); h = int(40 / UM)
        sl = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h)); ox, oy = sl[1].start, sl[0].start
        e = (0, (sl[1].stop - ox) * UM, (sl[0].stop - oy) * UM, 0)
        rows.append({"window": spot[:5], "rank": k + 1, "label": int(L), "area_um2": round(areas[L]), "n_tx": int(ntx[L]), "nuclei": int(inside.sum()),
                     "neuron_nuclei": nn, "top_class": f"{s[0][0]} {s[0][1]}", "second": f"{s[1][0]} {s[1][1]}", "flag": "; ".join(flag)})
        for j in (0, 1):
            ax = axs[j, k]; ax.imshow(merge[sl], extent=e, interpolation="none")
            if j == 0:
                from skimage.segmentation import find_boundaries
                b = find_boundaries(lab[sl], mode="inner"); o = np.zeros(b.shape + (4,)); o[b] = (1, 1, 1, .9)
                bb = find_boundaries(np.where(lab[sl] == L, 1, 0), mode="inner"); o[bb] = (1, .8, 0, 1)
                ax.imshow(o, extent=e, interpolation="none")
                ax.set_title(f"{spot[:5]} #{k + 1}: {areas[L]:.0f} µm², {ntx[L]:,} tx, {int(inside.sum())} nuc ({nn} neuron-sized)\n{s[0][0]} {s[0][1]} / {s[1][0]} {s[1][1]}" + (f"  FLAG: {'; '.join(flag)}" if flag else ""), loc="left", fontsize=7)
            else:
                for _, g in tenx.groupby("cell_id", sort=False):
                    xx, yy = g.vertex_x.to_numpy() - (x0 + ox) * UM, g.vertex_y.to_numpy() - (y0 + oy) * UM
                    if xx.max() > 0 and xx.min() < e[1] and yy.max() > 0 and yy.min() < e[2]: ax.plot(xx, yy, color="white", lw=0.7)
                ax.set_title("10x outlines", loc="left", fontsize=6.8, color="#52514e")
            ax.set_xlim(0, e[1]); ax.set_ylim(e[2], 0); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{spot}, {VAR}: the {N} largest cells (top: yellow outline = the cell, white = neighbours; bottom: 10x outlines). Merge: DAPI blue, boundary magenta, 18S yellow, aSMA/Vim green. 80 µm tiles", x=0.005, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(C / f"large_cells_{VAR}_{spot[:5]}.png", facecolor="white"); plt.close(fig)
df = pd.DataFrame(rows); pd.set_option("display.width", 220); print(df.to_string(index=False)); df.to_csv(C / f"large_cells_{VAR}.csv", index=False); print("saved large_cells_*.png")
