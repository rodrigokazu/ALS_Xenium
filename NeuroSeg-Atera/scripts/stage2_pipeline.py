#!/usr/bin/env python3
"""Stages 2-4 of PLAN.md on the small crops, then a direct comparison with 10x and the earlier runs.

Grid: image pixels (0.2125 um) over the 100 um crop plus a 6 um margin. Stage 1 features (compare/stage1/*_features.npz,
0.5 um grid) are resampled onto it.
  Stage 2 seeds: 10x nuclei (DAPI), each typed by the mean stage 1 type probability over its pixels; plus nucleus-free
    seeds = connected regions of confident (p >= 0.99) non-granule type in tissue, >= 20 um2, not touching a nucleus of the
    same type (cell bodies cut without their nucleus, e.g. Purkinje profiles).
  Stage 3+4 assignment, one flood per type t: watershed on (membrane stain + (1 - p_t)) from the type-t seeds, inside tissue
    (density > 20 tx/um2), over free pixels where type t is not ruled out (p_t >= 0.05) and within a type reach R[t] of a
    type-t seed. A pixel claimed by several types goes to the type with the highest p_t; within a type the watershed decides.
    Only the part connected to the seed is kept; holes are filled. Pixels nobody claims stay free (neuropil).
  Stage 5 split/merge is not applied yet.
Comparison on the inner 100 um (QC transcripts): 10x, frozen phase 2 (PF), I2, and this pipeline, scored with
splitmerge.metrics.score unchanged plus MECR at 1,000 tx.
python stage2_pipeline.py <out dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, matplotlib
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_poly
from skimage.segmentation import watershed, find_boundaries
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_reseg_pipeline"))
from splitmerge.metrics import score, mecr_downsampled
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 9})
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); CMP = SSD / "compare"; F1 = CMP / "stage1"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
CROPS = [("spot1_purkinje_left", 200, 200), ("spot2_gl_wm_centre", 200, 200), ("spot3_purkinje_right", 100, 170)]
SIZE, PAD = 100.0, 6.0; L = SIZE + 2 * PAD
TYPES = ["granule", "PC/interneuron", "astro", "oligo", "vascular", "immune"]; T = len(TYPES)
REACH = {"granule": 2.0, "PC/interneuron": 15.0, "astro": 8.0, "oligo": 4.0, "vascular": 4.0, "immune": 3.0}   # um from the seed
import os
TISSUE, PSEED, SEED_MIN_UM2 = 20.0, 0.99, 20.0
PMIN = float(os.environ.get("PMIN", 0.05)); REACH["astro"] = float(os.environ.get("REACH_ASTRO", 8.0)); TRIM = os.environ.get("TRIM", "0") == "1"
GRID = os.environ.get("GRID")          # "pmin:reach_astro:trim,..." -> tuning mode: score every setting, no figure
ONLY = os.environ.get("ONLY")          # comma list of crop prefixes
COL = np.array([matplotlib.colors.to_rgb(c) for c in ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]])
PCT = [(3.0, 321.0), (2.0, 112.0), (2.0, 608.0), (6.0, 363.0)]; RGB = np.array([[.22, .42, 1], [.92, .25, .78], [1, .82, .12], [.2, .9, .35]])

def raster(path, bx0, by0, n):
    b = ds.dataset(path).to_table(filter=(pc.field("vertex_x") >= bx0 - 15) & (pc.field("vertex_x") < bx0 + L + 15)
                                  & (pc.field("vertex_y") >= by0 - 15) & (pc.field("vertex_y") < by0 + L + 15)).to_pandas()
    lab = np.zeros((n, n), np.int32); ids = []
    for k, (cid, g) in enumerate(b.groupby("cell_id", sort=False), start=1):
        rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - by0 / UM, g.vertex_x.to_numpy() / UM - bx0 / UM, shape=(n, n)); lab[rr, cc] = k; ids.append(cid)
    return lab, ids

def cell_types(lab, P):
    idx = np.arange(1, lab.max() + 1)
    if not len(idx): return np.zeros(1, int)
    m = np.stack([ndi.mean(P[t], lab, idx) for t in range(T)]); return np.r_[-1, np.nan_to_num(m, nan=-1).argmax(0)]

def paint(lab, P, merge):
    ct = cell_types(lab, P); out = merge * 0.35
    fill = COL[np.clip(ct, 0, None)][lab]; m = lab > 0
    out[m] = 0.35 * merge[m] + 0.65 * fill[m]; out[find_boundaries(lab, mode="inner") & m] = 0.95
    return out

def segment(P, tissue, mem, nuc, pmin, reach, trim):
    n = nuc.shape[0]
    ntype = cell_types(nuc, P)
    seeds = nuc.copy()
    if trim:   # stage 5 (first rule): seed pixels that are confidently another type leave the seed and go back to the pool
        conf, pty = P.max(0), P.argmax(0)
        bad = (seeds > 0) & (conf >= PSEED) & (pty != ntype[seeds]); seeds[bad] = 0
    stype = list(ntype); nxt = nuc.max() + 1; n_blob = 0
    for t in range(1, T):
        cc, k = ndi.label((P[t] >= PSEED) & tissue & (seeds == 0))
        for i, sl in enumerate(ndi.find_objects(cc), start=1):
            m = cc[sl] == i
            if m.sum() * UM ** 2 < SEED_MIN_UM2: continue
            ring = ndi.binary_dilation(cc == i, iterations=2) & (seeds > 0)
            if np.any(ntype[np.unique(seeds[ring])] == t): continue      # a same-type nucleus will grow into it
            seeds[sl][m] = nxt; stype.append(t); nxt += 1; n_blob += 1
    stype = np.array(stype); stype[0] = -1
    free = seeds == 0; best = np.full((n, n), -1.0, np.float32); owner = np.zeros((n, n), np.int32)
    for t in range(T):
        st = np.isin(seeds, np.where(stype == t)[0]) & (seeds > 0)
        if not st.any(): continue
        dt = ndi.distance_transform_edt(~st) * UM
        allow = free & tissue & (P[t] >= pmin) & (dt <= reach[TYPES[t]])
        wt = watershed(mem + (1 - P[t]), markers=np.where(st, seeds, 0), mask=st | allow)
        take = allow & (wt > 0) & (P[t] > best)
        owner[take] = wt[take]; best[take] = P[t][take]
    lab1 = np.where(free, owner, seeds)
    lab = np.zeros_like(lab1)
    for i, sl in enumerate(ndi.find_objects(lab1), start=1):
        if sl is None: continue
        m = lab1[sl] == i; cc, _ = ndi.label(m); keep = np.unique(cc[(seeds[sl] == i) & m]); keep = keep[keep > 0]
        mm = ndi.binary_fill_holes(np.isin(cc, keep)) & ((lab1[sl] == i) | (lab1[sl] == 0))
        lab[sl][mm] = i
    return lab, seeds, n_blob

rows, figrows = [], []
for spot, ox, oy in [c for c in CROPS if not ONLY or c[0][:5] in ONLY.split(",")]:
    z = np.load(CMP / "option_C500" / spot / "crop.npz"); img, win = z["img"], z["window"]
    bx0, by0 = win[0] * UM + ox - PAD, win[1] * UM + oy - PAD
    gx0, gy0 = int(round(bx0 / UM)), int(round(by0 / UM)); n = int(round(L / UM))
    c0, r0 = gx0 - win[0], gy0 - win[1]; im = img[:, r0:r0 + n, c0:c0 + n].astype(float)
    f = np.load(F1 / f"{spot[:5]}_features.npz")
    zf = n / f["P"].shape[1]
    P = np.stack([ndi.zoom(f["P"][t].astype(np.float32), zf, order=1)[:n, :n] for t in range(T)]); P /= P.sum(0, keepdims=True)
    dens = ndi.zoom(f["dens"], zf, order=1)[:n, :n]
    mem = np.clip((im[1] - PCT[1][0]) / (PCT[1][1] - PCT[1][0]), 0, 1)
    tissue = dens > TISSUE
    nuc, _ = raster(SSD / "work/local_bundle/nucleus_boundaries.parquet", bx0, by0, n)
    # ---- comparison on the inner 100 um
    tx = ds.dataset(SSD / "work/local_bundle/transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name", "cell_id", "overlaps_nucleus"],
        filter=(pc.field("x_location") >= bx0 + PAD) & (pc.field("x_location") < bx0 + PAD + SIZE) & (pc.field("y_location") >= by0 + PAD) & (pc.field("y_location") < by0 + PAD + SIZE)
        & (pc.field("qv") >= 20) & (pc.field("is_gene"))).to_pandas()
    gc = (tx.x_location.to_numpy() / UM).astype(int); gr = (tx.y_location.to_numpy() / UM).astype(int)
    cl, rl = np.clip(gc - gx0, 0, n - 1), np.clip(gr - gy0, 0, n - 1)
    genes = tx.feature_name.to_numpy(); on_nuc = tx.overlaps_nucleus.astype(bool).to_numpy(); xy = tx[["x_location", "y_location"]].to_numpy()
    labs = {"10x": (pd.factorize(tx.cell_id.where(~tx.cell_id.isin(["", "UNASSIGNED"]), None))[0] + 1).astype(np.int64)}
    win_labs = {}
    for name, d in (("phase 2 frozen (PF)", "crops_PF_500"), ("I2", "crops_I2_500")):
        fl = np.load(RUNS / d / spot / "final_labels.npy", mmap_mode="r")
        labs[name] = np.asarray(fl[gr, gc]).astype(np.int64); win_labs[name] = np.asarray(fl[gy0:gy0 + n, gx0:gx0 + n]).astype(np.int64)
    if GRID:
        sc = score("10x", labs["10x"], genes, on_nuc, xy); sc["mecr_n1000"] = round(mecr_downsampled(labs["10x"], genes, n=1000), 4)
        rows.append({"crop": spot[:5], "method": "10x", **{k: sc[k] for k in ("pct_assigned", "n_cells", "median_tx_per_cell", "mecr", "mecr_n1000")}})
        for g in GRID.split(","):
            pm, ra, tr = g.split(":"); rch = {**REACH, "astro": float(ra)}
            lab_g, _, nb = segment(P, tissue, mem, nuc, float(pm), rch, tr == "1")
            lb = lab_g[rl, cl].astype(np.int64)
            sc = score(g, lb, genes, on_nuc, xy); sc["mecr_n1000"] = round(mecr_downsampled(lb, genes, n=1000), 4)
            rows.append({"crop": spot[:5], "method": g, **{k: sc[k] for k in ("pct_assigned", "n_cells", "median_tx_per_cell", "mecr", "mecr_n1000")}})
            print(spot[:5], g, sc["pct_assigned"], sc["mecr"], sc["mecr_n1000"], flush=True)
        continue
    lab, seeds, n_blob = segment(P, tissue, mem, nuc, PMIN, REACH, TRIM)
    labs["stage 2-4"] = lab[rl, cl].astype(np.int64)
    for name, lb in labs.items():
        s = score(name, lb, genes, on_nuc, xy); s["mecr_n1000"] = round(mecr_downsampled(lb, genes, n=1000), 4)
        rows.append({"crop": spot[:5], "method": name, **{k: s[k] for k in ("pct_assigned", "n_cells", "median_tx_per_cell", "mecr", "mecr_n1000", "shell_coherence_median")}})
    tenx_lab, _ = raster(SSD / "work/local_bundle/cell_boundaries.parquet", bx0, by0, n)
    merge = np.zeros((n, n, 3))
    for k in range(4):
        lo, hi = PCT[k]; merge += np.clip((im[k] - lo) / (hi - lo), 0, 1)[..., None] * RGB[k]
    merge = np.clip(merge, 0, 1)
    orph = {"10x": labs["10x"] == 0, "I2": labs["I2"] == 0, "stage 2-4": labs["stage 2-4"] == 0}
    figrows.append(dict(spot=spot, merge=merge, panels=[("10x", paint(tenx_lab, P, merge)), ("I2", paint(win_labs["I2"] if win_labs["I2"].max() < 2**31 else win_labs["I2"], P, merge)),
                        ("stage 2-4", paint(lab, P, merge))], tx=(tx.x_location.to_numpy() - bx0, tx.y_location.to_numpy() - by0), orph=orph,
                        seeds=dict(nuclei=int(nuc.max()), blob=n_blob)))
    np.savez_compressed(OUT / f"{spot[:5]}_labels.npz", labels=lab, seeds=seeds, origin=np.array([gx0, gy0]))
    print(spot, "nuclei", nuc.max(), "nucleus-free seeds", n_blob, flush=True)

df = pd.DataFrame(rows); df.to_csv(OUT / ("stage2_grid.csv" if GRID else os.environ.get("FIGNAME", "stage2_compare.png").replace(".png", ".csv")), index=False)
if GRID: sys.exit(0)
pd.set_option("display.width", 200); print(df.to_string(index=False))

# ---- figure: one row per crop; morphology | 10x | I2 | stage 2-4, cells filled by their type, red = transcripts in no cell
fig, axs = plt.subplots(3, 4, figsize=(18, 13.6), dpi=150, gridspec_kw=dict(wspace=0.03, hspace=0.17))
rng = np.random.default_rng(0)
for i, r in enumerate(figrows):
    ext = (0, L, L, 0)
    for j, (name, a) in enumerate([("Morphology", r["merge"])] + r["panels"]):
        ax = axs[i, j]; ax.imshow(a, extent=ext, interpolation="none"); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_xlim(PAD, PAD + SIZE); ax.set_ylim(PAD + SIZE, PAD)
        ax.plot([PAD + 85, PAD + 95], [PAD + 96] * 2, color="white", lw=2.5, solid_capstyle="butt")
        if j:
            o = r["orph"][name]; idx = np.where(o)[0]; idx = rng.choice(idx, min(len(idx), 6000), replace=False)
            ax.scatter(r["tx"][0][idx], r["tx"][1][idx], s=0.6, color="#e34948", lw=0, alpha=0.8)
            s = df[(df.crop == r["spot"][:5]) & (df.method == name)].iloc[0]
            ax.set_title(f"{s.pct_assigned:.0f}% in cells · MECR {s.mecr:.3f} · at 1,000 tx {s.mecr_n1000:.4f}", loc="left", fontsize=8.5, color="#52514e")
        if i == 0: ax.text(0, 1.08, name if name != "stage 2-4" else "New pipeline (stages 2-4)", transform=ax.transAxes, fontsize=11.5, fontweight="bold")
    axs[i, 0].set_ylabel(f"{r['spot'].replace('_', ' ')}\n{r['seeds']['nuclei']} nuclei + {r['seeds']['blob']} nucleus-free seeds", fontsize=9.5)
hand = [plt.Line2D([], [], marker="s", ls="", color=COL[t], ms=8, label=TYPES[t]) for t in range(T)] + [plt.Line2D([], [], marker="o", ls="", color="#e34948", ms=4, label="transcript in no cell")]
fig.legend(handles=hand, loc="lower center", ncol=7, frameon=False, fontsize=10)
fig.text(0.01, 0.975, "Three 100 µm crops (scale bars 10 µm). Cells filled by the type read from their transcripts; the same colours for every method.", fontsize=11, color="#52514e")
fig.subplots_adjust(top=0.92, bottom=0.05, left=0.04, right=0.995)
fig.savefig(OUT / os.environ.get("FIGNAME", "stage2_compare.png"), facecolor="white"); print("saved", OUT / "stage2_compare.png")
