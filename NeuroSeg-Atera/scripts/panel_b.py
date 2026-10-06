#!/usr/bin/env python3
"""Panel b for the Atera poster figure: per example, left = morphology (DAPI blue, 18S yellow) with the held-out Purkinje
transcripts (set B) as dots; right = the cells that overlap the soma, Xenium Ranger outlined magenta, this pipeline gold.
Colours follow panel d. Examples are chosen from windows not used for tuning.
python panel_b.py   ->  compare/purkinje/panel_b_2examples.png, panel_b_4examples.png"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr, matplotlib
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_poly
from skimage.segmentation import find_boundaries
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Arial", "font.size": 12})
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); D = SSD / "compare/purkinje"
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
RANGER, OURS = np.array(matplotlib.colors.to_rgb("#cc3399")), np.array(matplotlib.colors.to_rgb("#e69f00"))
so = pd.read_csv(D / "pc_somata.csv"); sc = pd.read_csv(D / "pc_scores.csv"); setA = set(json.load(open(D / "gene_sets.json"))["purkinje_A"])
setB = set(json.load(open(D / "gene_sets.json"))["purkinje_B"])
import os
# best cases (86th-96th percentile of the improvement over Ranger) vs typical cases (held-out somata closest to the median improvement)
SETS = {"best": [("held2", 14, ""), ("held5", 25, ""), ("held5", 19, ""), ("spot3_purkinje_right", 16, "")],
        "typical": [("held1", 37, ""), ("held2", 44, ""), ("spot3_purkinje_right", 63, ""), ("spot3_purkinje_right", 29, "")]}
# held3 soma 2 (also near the median) was dropped: in the full-section labels our cell over-merges into the neighbouring Purkinje region
SET = os.environ.get("SET", "best"); EX = SETS[SET]
HALF = int(30 / UM)

def tile(win, soma):
    s = so[(so.window == win) & (so.soma == soma)].iloc[0]; cx, cy = int(s.cx_px), int(s.cy_px); X0, Y0 = cx - HALF, cy - HALF; n = 2 * HALF
    b = SSD / "Cerebellum_sample" if win == "full" else SSD / "work" / ("local_bundle_heldout" if win.startswith("held") else "local_bundle")
    z = zarr.open(tifffile.TiffFile(b / "morphology_focus" / "ch0000_dapi.ome.tif").aszarr(level=0, series=0), mode="r")
    im = np.asarray(z[:, Y0:Y0 + n, X0:X0 + n]).astype(float)
    mg = np.clip(np.clip((im[0] - 3) / 318, 0, 1)[..., None] * np.array([.22, .42, 1]) + np.clip((im[2] - 2) / 606, 0, 1)[..., None] * np.array([1, .82, .12]) * 0.85, 0, 1)
    tx = ds.dataset(b / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name"], filter=(pc.field("x_location") >= X0 * UM) & (pc.field("x_location") < (X0 + n) * UM)
        & (pc.field("y_location") >= Y0 * UM) & (pc.field("y_location") < (Y0 + n) * UM) & (pc.field("qv") >= 20)).to_pandas()
    a = tx[tx.feature_name.isin(setA)]; r, c = np.clip((a.y_location / UM).astype(int) - Y0, 0, n - 1), np.clip((a.x_location / UM).astype(int) - X0, 0, n - 1)
    dA = ndi.gaussian_filter(np.bincount(r * n + c, minlength=n * n).reshape(n, n).astype(float), 1.5 / UM) / UM ** 2
    cl, nc = ndi.label(ndi.binary_opening(dA >= 0.534, iterations=int(1 / UM)))
    if cl[HALF, HALF]: core = cl == cl[HALF, HALF]
    else:                                                                   # the soma core nearest the tile centre
        cms = np.array(ndi.center_of_mass(cl > 0, cl, range(1, nc + 1))); k_ = 1 + int(np.argmin(np.hypot(cms[:, 0] - HALF, cms[:, 1] - HALF))); core = cl == k_
    yy, xx = np.ogrid[:n, :n]; core &= (np.hypot(yy - HALF, xx - HALF) * UM <= 12)          # only the centre soma, not a touching neighbour
    cb = ds.dataset(b / "cell_boundaries.parquet").to_table(filter=(pc.field("vertex_x") >= X0 * UM - 30) & (pc.field("vertex_x") < (X0 + n) * UM + 30)
        & (pc.field("vertex_y") >= Y0 * UM - 30) & (pc.field("vertex_y") < (Y0 + n) * UM + 30)).to_pandas()
    rl = np.zeros((n, n), np.int64)
    for k, (_, g) in enumerate(cb.groupby("cell_id", sort=False), start=1):
        rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - Y0, g.vertex_x.to_numpy() / UM - X0, shape=(n, n)); rl[rr, cc] = k
    ol = np.asarray(zarr.open_array(str(SSD / os.environ.get("SEG", "segmentation_hybridC") / "final_labels.zarr"), mode="r")[Y0:Y0 + n, X0:X0 + n]).astype(np.int64)   # final pipeline, full section
    def soma_cells(L):
        ids = pd.Series(L[core]).value_counts(); ids = ids[(ids.index > 0) & (ids >= 0.05 * core.sum())]; return np.isin(L, ids.index) & (L > 0), L
    tb = tx[tx.feature_name.isin(setB)]
    k = {m: sc[(sc.window == win) & (sc.soma == soma) & (sc.method == m)].iloc[0] for m in ("10x", "I2 + Purkinje repair")}
    return dict(mg=mg, tb=(tb.x_location.to_numpy() / UM - X0, tb.y_location.to_numpy() / UM - Y0), r=soma_cells(rl), o=soma_cells(ol), n=n, k=k)

def edges(mask_lab, width):
    m, L = mask_lab; e = find_boundaries(np.where(m, L, 0), mode="inner") & m
    return ndi.binary_dilation(e, iterations=width)

def draw(axl, axr, t, label, first):
    """left: merge only; right: black, held-out Purkinje transcripts, then the two masks on top. No text."""
    n = t["n"]; L = n * UM
    axl.imshow(t["mg"], extent=(0, L, L, 0))
    axr.imshow(np.zeros((n, n, 3)), extent=(0, L, L, 0))                     # black background, no morphology under the masks
    axr.scatter(t["tb"][0] * UM, t["tb"][1] * UM, s=2.0, color="white", lw=0, alpha=0.9, zorder=2)
    ov = np.zeros((n, n, 4))
    # every other cell: thin lines (1 px), Ranger under ours
    for (m_, L_), colr in ((t["r"], RANGER), (t["o"], OURS)):
        thin = find_boundaries(L_, mode="inner") & (L_ > 0); ov[thin] = (*colr, 0.75)
    # the cells that hold the soma: thick lines on top
    er, eo = edges(t["r"], 2), edges(t["o"], 2)
    ov[er] = (*RANGER, 1); ov[eo] = (*OURS, 1); ov[er & eo] = (*(0.5 * (RANGER + OURS)), 1)
    axr.imshow(ov, extent=(0, L, L, 0), interpolation="none", zorder=3)
    for a in (axl, axr):
        a.set_xlim(0, L); a.set_ylim(L, 0); a.set_xticks([]); a.set_yticks([])
        for s_ in a.spines.values(): s_.set_visible(False)

def legend(fig, y):
    h = [plt.Line2D([], [], color="#cc3399", lw=3, label="Xenium Ranger"), plt.Line2D([], [], color="#e69f00", lw=3, label="This pipeline"),
         plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#0b0b0b", ms=6, label="held-out Purkinje transcripts")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, y))

tiles = [tile(w, s) for w, s, _ in EX]
for nex, name, shape in ((2, f"panel_b_2examples{'' if SET == 'best' else '_typical'}.png", (2, 2)), (4, f"panel_b_4examples{'' if SET == 'best' else '_typical'}.png", (2, 4))):
    fig, axs = plt.subplots(*shape, figsize=(3.1 * shape[1], 3.1 * shape[0]), dpi=200, gridspec_kw=dict(wspace=0.03, hspace=0.05))
    for i in range(nex):
        r_, c_ = (i, 0) if nex == 2 else (i // 2, (i % 2) * 2)
        draw(axs[r_, c_], axs[r_, c_ + 1], tiles[i], f"{i + 1}  {EX[i][2]}", i == 0)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(D / name, facecolor="white"); print("saved", D / name)

# ---- contact sheet: all eight candidates (1-4 best, 5-8 typical), numbered for picking. PICK="a,b" makes a clean 2-example figure
if os.environ.get("SHEET") == "1" or os.environ.get("PICK"):
    allex = SETS["best"] + SETS["typical"]
    if os.environ.get("PICK"):
        sel = [int(i) - 1 for i in os.environ["PICK"].split(",")]; tl = [tile(allex[i][0], allex[i][1]) for i in sel]
        fig, axs = plt.subplots(len(sel), 2, figsize=(6.2, 3.1 * len(sel)), dpi=200, gridspec_kw=dict(wspace=0.03, hspace=0.05))
        for i, t in enumerate(tl): draw(axs[i, 0], axs[i, 1], t, "", False)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005); out = D / f"panel_b_pick_{os.environ['PICK'].replace(',', '_')}.png"
    else:
        tl = [tile(w, s) for w, s, _ in allex]
        fig, axs = plt.subplots(4, 4, figsize=(12.4, 12.4), dpi=150, gridspec_kw=dict(wspace=0.03, hspace=0.05))
        for i, t in enumerate(tl):
            r_, c_ = i // 2, (i % 2) * 2; draw(axs[r_, c_], axs[r_, c_ + 1], t, "", False)
            axs[r_, c_].text(0.03, 0.96, str(i + 1), transform=axs[r_, c_].transAxes, va="top", fontsize=16, color="white", fontweight="bold")
        fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005); out = D / "panel_b_contact_sheet.png"
    fig.savefig(out, facecolor="white"); print("saved", out)

# ---- any positions in the full section: CENTER="x_um,y_um[;x_um,y_um...]" SIZE_UM=90 [NUMBERS=1] -> one row per position
if os.environ.get("CENTER"):
    pts = [tuple(map(float, q.split(","))) for q in os.environ["CENTER"].split(";")]; HALF = int(float(os.environ.get("SIZE_UM", 90)) / 2 / UM)
    for k, (cxu, cyu) in enumerate(pts):
        so = pd.concat([so, pd.DataFrame([{"window": "full", "soma": -1 - k, "cx_px": cxu / UM, "cy_px": cyu / UM, "area_um2": np.nan}])], ignore_index=True)
        sc = pd.concat([sc, pd.DataFrame([{"window": "full", "soma": -1 - k, "method": m, "capture": np.nan, "n_cells": np.nan, "orphan": np.nan} for m in ("10x", "I2 + Purkinje repair")])], ignore_index=True)
    tl = [tile("full", -1 - k) for k in range(len(pts))]
    fig, axs = plt.subplots(len(pts), 2, figsize=(6.2, 3.1 * len(pts)), dpi=220, gridspec_kw=dict(wspace=0.03, hspace=0.05), squeeze=False)
    for k, t in enumerate(tl):
        draw(axs[k, 0], axs[k, 1], t, "", False)
        if os.environ.get("NUMBERS") == "1": axs[k, 0].text(0.03, 0.96, os.environ.get("NAMES", ",".join(str(i + 1) for i in range(len(pts)))).split(",")[k], transform=axs[k, 0].transAxes, va="top", fontsize=15, color="white", fontweight="bold")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    out = D / (os.environ.get("OUTNAME") or f"panel_b_at_{int(pts[0][0])}_{int(pts[0][1])}.png"); fig.savefig(out, facecolor="white"); print("saved", out)
