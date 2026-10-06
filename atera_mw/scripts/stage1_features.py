#!/usr/bin/env python3
"""Stage 1 of PLAN.md: transcript feature layer on small crops (0.5 um grid).

Per crop: density (all QC transcripts, sigma 1 um), per-type evidence S_t = Gaussian-smoothed sum of log p(gene | type)
over informative transcripts (sigma 2 um, the best scale in stage 0), posterior p_t = softmax(tau * S_t), confidence
(max p_t), and a type-edge map |grad p| (where the local identity changes). tau is calibrated on spot1 + spot2 nuclear
transcripts against the 10x type of their nucleus (maximum mean log posterior) and then frozen for spot3.
Checks: per-pixel type accuracy and calibration at nuclear transcripts; per-nucleus agreement between the 10x label and
the type read from the nucleus's own transcripts (diagnoses spot1).
python stage1_features.py <out dir>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, matplotlib
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_poly
from skimage.segmentation import find_boundaries
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 9})
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); CMP = SSD / "compare"; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
CROPS = [("spot1_purkinje_left", 200, 200), ("spot2_gl_wm_centre", 200, 200), ("spot3_purkinje_right", 100, 170)]
SIZE, PAD, BIN, SIG = 100.0, 6.0, 0.5, 2.0
GROUPS = {"granule": [1, 2, 3, 4, 5, 8, 9, 10, 15, 6, 7, 11], "PC/interneuron": [14, 23, 33], "astro": [13, 16, 19],
          "oligo": [12, 18, 27, 29, 30, 32], "vascular": [17, 20, 21, 22, 26, 28], "immune": [24, 25, 31, 34, 35, 36]}
TYPES = list(GROUPS); T = len(TYPES)
COL = np.array([matplotlib.colors.to_rgb(c) for c in ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]])  # validated, dark
PCT = [(3.0, 321.0), (2.0, 112.0), (2.0, 608.0), (6.0, 363.0)]; RGB = np.array([[.22, .42, 1], [.92, .25, .78], [1, .82, .12], [.2, .9, .35]])

de = pd.read_csv(SSD / "Cerebellum_sample/analysis/diffexp/gene_expression_graphclust/differential_expression.csv")
clu = pd.read_csv(SSD / "Cerebellum_sample/analysis/clustering/gene_expression_graphclust/clusters.csv")
ncl = clu.Cluster.value_counts().sort_index().reindex(range(1, 37)).to_numpy().astype(float)
M = np.column_stack([de[f"Cluster {k} Mean Counts"] for k in range(1, 37)]); Lf = np.column_stack([de[f"Cluster {k} Log2 fold change"] for k in range(1, 37)])
Pv = np.column_stack([de[f"Cluster {k} Adjusted p value"] for k in range(1, 37)])
inf = ((Lf >= 2.0) & (M >= 0.3) & (Pv < 0.01)).any(1); gidx = pd.Index(de["Feature Name"].to_numpy()[inf]); Mi = M[inf]
prof = np.zeros((T, inf.sum()))
for t, ks in enumerate(GROUPS.values()):
    w = ncl[np.array(ks) - 1]; prof[t] = (Mi[:, np.array(ks) - 1] * w).sum(1) / w.sum()
prof += 1e-4; prof /= prof.sum(1, keepdims=True); logp = np.log(prof)
c2t = {c: t for t, cs in enumerate(GROUPS.values()) for c in cs}; cell_type = dict(zip(clu.Barcode, clu.Cluster.map(c2t)))

def softmax(S, tau):
    z = tau * (S - S.max(0, keepdims=True)); e = np.exp(z); return e / e.sum(0, keepdims=True)

crops = {}
for spot, ox, oy in CROPS:
    z = np.load(CMP / "option_C500" / spot / "crop.npz"); img, win = z["img"], z["window"]
    X0, Y0 = win[0] * UM + ox, win[1] * UM + oy; bx0, by0 = X0 - PAD, Y0 - PAD; L = SIZE + 2 * PAD
    tx = ds.dataset(SSD / "work/local_bundle/transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name", "cell_id", "overlaps_nucleus"],
        filter=(pc.field("x_location") >= bx0) & (pc.field("x_location") < bx0 + L) & (pc.field("y_location") >= by0) & (pc.field("y_location") < by0 + L)
        & (pc.field("qv") >= 20) & (pc.field("is_gene"))).to_pandas()
    x, y = tx.x_location.to_numpy() - bx0, tx.y_location.to_numpy() - by0
    n = int(np.ceil(L / BIN)); xb = np.clip((x / BIN).astype(int), 0, n - 1); yb = np.clip((y / BIN).astype(int), 0, n - 1); flat = yb * n + xb
    gi = gidx.get_indexer(tx.feature_name.astype(str)); ok = gi >= 0
    S = np.stack([ndi.gaussian_filter(np.bincount(flat[ok], weights=logp[t][gi[ok]], minlength=n * n).reshape(n, n), SIG / BIN) for t in range(T)])
    ninf = ndi.gaussian_filter(np.bincount(flat[ok], minlength=n * n).reshape(n, n).astype(float), SIG / BIN) * 2 * np.pi * (SIG / BIN) ** 2  # ~ informative tx in the kernel
    dens = ndi.gaussian_filter(np.bincount(flat, minlength=n * n).reshape(n, n).astype(float), 1.0 / BIN) / BIN ** 2
    # nuclei: 10x type label vs the type read from the nucleus's own informative transcripts
    nuc = (tx.overlaps_nucleus.to_numpy() == 1) & ~tx.cell_id.isin(["", "UNASSIGNED"]).to_numpy()
    inner = (x >= PAD) & (x < PAD + SIZE) & (y >= PAD) & (y < PAD + SIZE)
    sel = nuc & inner & ok; cid = tx.cell_id.to_numpy()[sel]
    own = pd.DataFrame(logp[:, gi[sel]].T, columns=TYPES).assign(cid=cid).groupby("cid").agg(["sum"]).droplevel(1, axis=1)
    nn = pd.Series(cid).value_counts()
    nucdf = pd.DataFrame({"own": own.to_numpy().argmax(1), "n_inf": nn.reindex(own.index).to_numpy()}, index=own.index)
    nucdf["tenx"] = [cell_type.get(c, -1) for c in nucdf.index]
    # morphology crop on the image grid
    c0, r0 = int((ox - PAD) / UM), int((oy - PAD) / UM); npx = int(L / UM)
    im = img[:, max(r0, 0):r0 + npx, max(c0, 0):c0 + npx]
    nb_ = ds.dataset(SSD / "work/local_bundle/nucleus_boundaries.parquet").to_table(
        filter=(pc.field("vertex_x") >= bx0 - 10) & (pc.field("vertex_x") < bx0 + L + 10) & (pc.field("vertex_y") >= by0 - 10) & (pc.field("vertex_y") < by0 + L + 10)).to_pandas()
    nlab = np.zeros((n, n), np.int32)
    for k, (_, g) in enumerate(nb_.groupby("cell_id", sort=False), start=1):
        rr, cc = draw_poly((g.vertex_y.to_numpy() - by0) / BIN, (g.vertex_x.to_numpy() - bx0) / BIN, shape=(n, n)); nlab[rr, cc] = k
    crops[spot] = dict(S=S, ninf=ninf, dens=dens, nucdf=nucdf, im=im, nlab=nlab,
                       px=(xb[sel], yb[sel]), ptype=nucdf.tenx.reindex(cid).to_numpy(), pown=nucdf.own.reindex(cid).to_numpy())
    print(spot, "transcripts", len(tx), "informative", int(ok.sum()), "nuclei", len(nucdf), flush=True)

# ---- calibrate tau on spot1 + spot2 nuclear transcripts (10x nucleus type as label), frozen for spot3
taus = np.geomspace(0.003, 100, 60)
def mean_logpost(c, tau):
    p = softmax(c["S"][:, c["px"][1], c["px"][0]], tau); m = c["ptype"] >= 0
    return np.log(np.clip(p[c["ptype"][m], np.where(m)[0]], 1e-9, 1)).mean()
cal = [np.mean([mean_logpost(crops[s], tau) for s in ("spot1_purkinje_left", "spot2_gl_wm_centre")]) for tau in taus]
tau = float(taus[int(np.argmax(cal))]); print("tau", round(tau, 4))

rep = {"tau": tau, "crops": {}}
for spot, c in crops.items():
    P = softmax(c["S"], tau); conf = P.max(0); ptyp = P.argmax(0)
    gy = np.stack([np.hypot(*np.gradient(ndi.gaussian_filter(P[t], 1.0))) for t in range(T)]).sum(0) / BIN    # per um
    c.update(P=P, conf=conf, edge=gy)
    xs, ys = c["px"]; m = c["ptype"] >= 0; pt = ptyp[ys, xs]; cf = conf[ys, xs]
    d = c["nucdf"]; dm = d[(d.tenx >= 0) & (d.n_inf >= 30)]
    tissue = c["dens"] > 10
    r = dict(pixel_acc_vs_10x=float((pt[m] == c["ptype"][m]).mean()), pixel_acc_vs_own=float((pt == c["pown"]).mean()),
             mean_conf_at_nuclei=float(cf[m].mean()),
             calib={f"{lo:.1f}-{hi:.1f}": [float((pt[m][(cf[m] >= lo) & (cf[m] < hi)] == c["ptype"][m][(cf[m] >= lo) & (cf[m] < hi)]).mean()) if ((cf[m] >= lo) & (cf[m] < hi)).sum() else None,
                                           int(((cf[m] >= lo) & (cf[m] < hi)).sum())] for lo, hi in ((0, .5), (.5, .7), (.7, .9), (.9, .99), (.99, 1.01))},
             nuclei=len(dm), nucleus_agree_own_vs_10x=float((dm.own == dm.tenx).mean()),
             confusion_10x_rows_own_cols=pd.crosstab(dm.tenx.map(dict(enumerate(TYPES))), dm.own.map(dict(enumerate(TYPES)))).to_dict(),
             tissue_frac_conf90=float((conf[tissue] >= 0.9).mean()), tissue_frac=float(tissue.mean()),
             median_inf_per_kernel=float(np.median(c["ninf"][tissue])),
             type_area_frac={TYPES[t]: float((ptyp[tissue] == t).mean()) for t in range(T)})
    rep["crops"][spot] = r
    np.savez_compressed(OUT / f"{spot[:5]}_features.npz", P=P.astype(np.float16), conf=conf.astype(np.float16), edge=gy.astype(np.float16), dens=c["dens"].astype(np.float32), nlab=nlab if False else c["nlab"])
json.dump(rep, open(OUT / "stage1_report.json", "w"), indent=1)

# ---- figure: one row per crop; morphology | density | type map | type-edge map with 10x nuclei
fig, axs = plt.subplots(3, 4, figsize=(17, 13.2), dpi=150, gridspec_kw=dict(wspace=0.04, hspace=0.16))
for i, (spot, c) in enumerate(crops.items()):
    im = c["im"]; mg = np.zeros(im.shape[1:] + (3,))
    for k in range(4):
        lo, hi = PCT[k]; mg += np.clip((im[k].astype(float) - lo) / (hi - lo), 0, 1)[..., None] * RGB[k]
    ext = (0, 112, 112, 0)
    tm = np.einsum("tyx,tc->yxc", c["P"], COL) * np.clip((c["dens"] / 40), 0, 1)[..., None]
    nb = find_boundaries(c["nlab"], mode="inner")
    ed = np.clip(c["edge"] / np.percentile(c["edge"][c["dens"] > 10], 99), 0, 1)
    panels = [(np.clip(mg, 0, 1), None), (c["dens"], "magma"), (np.clip(tm, 0, 1), None), (ed, "Greys_r")]
    for j, (a, cm) in enumerate(panels):
        ax = axs[i, j]; ax.imshow(a, extent=ext, cmap=cm, interpolation="none", vmin=0 if cm else None, vmax=(np.percentile(a, 99.5) if cm == "magma" else None))
        if j == 3:
            ov = np.zeros(nb.shape + (4,)); ov[nb] = (*matplotlib.colors.to_rgb("#3987e5"), 0.9); ax.imshow(ov, extent=ext, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.plot([92, 102], [106, 106], color="white", lw=2.5, solid_capstyle="butt")
    r = rep["crops"][spot]
    axs[i, 0].set_ylabel(spot.replace("_", " "), fontsize=10)
    axs[i, 2].set_title(f"pixel type = 10x nucleus type {100 * r['pixel_acc_vs_10x']:.0f}% · nucleus own vs 10x {100 * r['nucleus_agree_own_vs_10x']:.0f}%",
                        loc="left", fontsize=8.5, color="#52514e")
for j, t in enumerate(["Morphology (DAPI, membrane, 18S, αSMA/vim)", "Transcript density (σ 1 µm)", "Type probability (brightness = density)", "Type edge (gradient of p) with 10x nuclei (blue)"]):
    axs[0, j].text(0, 1.09, t, transform=axs[0, j].transAxes, fontsize=11, fontweight="bold")
hand = [plt.Line2D([], [], marker="s", ls="", color=COL[t], ms=8, label=TYPES[t]) for t in range(T)]
fig.legend(handles=hand, loc="lower center", ncol=6, frameon=False, fontsize=10)
fig.text(0.01, 0.975, f"Stage 1 feature layer on three 100 µm crops (scale bars 10 µm). Temperature τ = {tau:.3f}, fitted on spot 1 and 2 only.", fontsize=11, color="#52514e")
fig.subplots_adjust(top=0.92, bottom=0.05, left=0.03, right=0.995)
fig.savefig(OUT / "stage1_features.png", facecolor="white"); print("saved", OUT / "stage1_features.png")
print(json.dumps({s: {k: v for k, v in r.items() if k != "confusion_10x_rows_own_cols"} for s, r in rep["crops"].items()}, indent=1))
for s, r in rep["crops"].items():
    print(s); print(pd.DataFrame(r["confusion_10x_rows_own_cols"]).fillna(0).astype(int).to_string())
