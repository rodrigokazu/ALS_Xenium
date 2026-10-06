#!/usr/bin/env python3
"""PLAN.md stages 1-4 on a 500 um window (spot1-3, held1-6), scored against 10x, PF and I2 on the same QC transcripts.

Same method as stage1_features.py + stage2_pipeline.py, at window scale: type evidence on a 0.5 um grid (sigma 2 um,
temperature tau frozen from stage 1), resampled to the image grid; seeds = 10x nuclei typed by their mean type probability
plus nucleus-free confident non-granule blobs; one flood per type on (membrane + 1 - p_t), claim only where p_t >= pmin and
within the type reach; between types the highest p_t wins. The window is processed with a 6 um margin and scored inside.
python stage_win.py <window> <out csv dir>   env: GRID="pmin:reach_astro,..." (score each) or PMIN / REACH_ASTRO + SAVE=1
(writes final_labels.npy like a phase2 run to crops_S2_500/<window>, for build_bench_data.py)
"""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_poly
from skimage.segmentation import watershed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_reseg_pipeline"))
from splitmerge.metrics import score, mecr_downsampled

UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB")
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
WIN = sys.argv[1]; OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
BUNDLE = SSD / "work" / ("local_bundle_heldout" if WIN.startswith("held") else "local_bundle")
TYPES = ["granule", "PC/interneuron", "astro", "oligo", "vascular", "immune"]; T = len(TYPES)
GROUPS = {"granule": [1, 2, 3, 4, 5, 8, 9, 10, 15, 6, 7, 11], "PC/interneuron": [14, 23, 33], "astro": [13, 16, 19],
          "oligo": [12, 18, 27, 29, 30, 32], "vascular": [17, 20, 21, 22, 26, 28], "immune": [24, 25, 31, 34, 35, 36]}
REACH = {"granule": 2.0, "PC/interneuron": 15.0, "astro": 8.0, "oligo": 4.0, "vascular": 4.0, "immune": 3.0}
TISSUE, PSEED, SEED_MIN_UM2, BIN, SIG, PADUM = 20.0, 0.99, 20.0, 0.5, 2.0, 6.0
TAU = json.load(open(SSD / "compare/stage1/stage1_report.json"))["tau"]          # frozen in stage 1 (spot1 + spot2 crops)
PCT_MEM = (2.0, 112.0)                                                           # whole-sample membrane p1 / p99.7

x0, y0, x1, y1 = json.load(open(RUNS / "crops_PF_500" / WIN / "stats.json"))["window"]
pad = int(round(PADUM / UM)); X0, Y0, X1, Y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad; H, W = Y1 - Y0, X1 - X0
bx0, by0 = X0 * UM, Y0 * UM

# ---- type profiles (as stage 1)
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

# ---- transcripts in the padded window
tx = ds.dataset(BUNDLE / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name", "cell_id", "overlaps_nucleus"],
    filter=(pc.field("x_location") >= bx0) & (pc.field("x_location") < X1 * UM) & (pc.field("y_location") >= by0) & (pc.field("y_location") < Y1 * UM)
    & (pc.field("qv") >= 20) & (pc.field("is_gene"))).to_pandas(strings_to_categorical=True)
xs, ys = tx.x_location.to_numpy() - bx0, tx.y_location.to_numpy() - by0
nb = int(np.ceil(W * UM / BIN)) + 1; xb = np.clip((xs / BIN).astype(int), 0, nb - 1); yb = np.clip((ys / BIN).astype(int), 0, nb - 1); flat = yb * nb + xb
cats = tx.feature_name.cat.categories; gi_cat = gidx.get_indexer(cats.astype(str)); gi = gi_cat[tx.feature_name.cat.codes.to_numpy()]; ok = gi >= 0
S = np.stack([ndi.gaussian_filter(np.bincount(flat[ok], weights=logp[t][gi[ok]], minlength=nb * nb).reshape(nb, nb), SIG / BIN) for t in range(T)])
dens_b = ndi.gaussian_filter(np.bincount(flat, minlength=nb * nb).reshape(nb, nb).astype(float), 1.0 / BIN) / BIN ** 2
z = TAU * (S - S.max(0, keepdims=True)); Pb = np.exp(z); Pb /= Pb.sum(0, keepdims=True); del S, z
zf = (BIN / UM)
def up(a): return ndi.zoom(a.astype(np.float32), zf, order=1)[:H, :W]
P = np.stack([up(Pb[t]) for t in range(T)]); P /= P.sum(0, keepdims=True); dens = up(dens_b)
if P.shape[1:] != (H, W): raise SystemExit(f"grid mismatch {P.shape} vs {(H, W)}")
tissue = dens > TISSUE
tf = tifffile.TiffFile(BUNDLE / "morphology_focus" / "ch0000_dapi.ome.tif"); zimg = zarr.open(tf.aszarr(level=0, series=0), mode="r")
mem = np.clip((np.asarray(zimg[1, Y0:Y1, X0:X1]).astype(np.float32) - PCT_MEM[0]) / (PCT_MEM[1] - PCT_MEM[0]), 0, 1)
nbd = ds.dataset(BUNDLE / "nucleus_boundaries.parquet").to_table(filter=(pc.field("vertex_x") >= bx0 - 15) & (pc.field("vertex_x") < X1 * UM + 15)
                                                                  & (pc.field("vertex_y") >= by0 - 15) & (pc.field("vertex_y") < Y1 * UM + 15)).to_pandas()
nuc = np.zeros((H, W), np.int32)
for k, (_, g) in enumerate(nbd.groupby("cell_id", sort=False), start=1):
    rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - Y0, g.vertex_x.to_numpy() / UM - X0, shape=(H, W)); nuc[rr, cc] = k
if os.environ.get("EXTRA_NUC") == "1":   # add our own nuclei that have no 10x overlap (nuclei_detect.py, same padded grid)
    r = np.load(SSD / "compare/nuclei" / f"{WIN}_nuclei.npz"); assert tuple(r["origin"]) == (X0, Y0)
    own, new = r["own"], r["new"]; m = np.isin(own, new) & (nuc == 0)
    _, inv = np.unique(own[m], return_inverse=True); nuc[m] = nuc.max() + 1 + inv
print(WIN, "transcripts", len(tx), "nuclei", nuc.max(), "tissue", round(float(tissue.mean()), 3), flush=True)

def cell_types(lab):
    idx = np.arange(1, lab.max() + 1)
    m = np.stack([ndi.mean(P[t], lab, idx) for t in range(T)]); return np.r_[-1, np.nan_to_num(m, nan=-1).argmax(0)]
ntype = cell_types(nuc)

def segment(pmin, reach):
    seeds = nuc.copy(); stype = list(ntype); nxt = nuc.max() + 1; n_blob = 0
    for t in range(1, T):
        cc, _ = ndi.label((P[t] >= PSEED) & tissue & (seeds == 0))
        for i, sl in enumerate(ndi.find_objects(cc), start=1):
            m = cc[sl] == i
            if m.sum() * UM ** 2 < SEED_MIN_UM2: continue
            sl2 = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in sl)
            ring = ndi.binary_dilation(cc[sl2] == i, iterations=2) & (seeds[sl2] > 0)
            if np.any(ntype[np.unique(seeds[sl2][ring])] == t): continue
            seeds[sl][m] = nxt; stype.append(t); nxt += 1; n_blob += 1
    stype = np.array(stype); stype[0] = -1
    free = seeds == 0; best = np.full((H, W), -1.0, np.float32); owner = np.zeros((H, W), np.int32)
    for t in range(T):
        st = (seeds > 0) & (stype[seeds] == t)
        if not st.any(): continue
        dt = ndi.distance_transform_edt(~st) * UM
        allow = free & tissue & (P[t] >= pmin) & (dt <= reach[TYPES[t]])
        wt = watershed(mem + (1 - P[t]), markers=np.where(st, seeds, 0), mask=st | allow)
        take = allow & (wt > 0) & (P[t] > best); owner[take] = wt[take]; best[take] = P[t][take]
    lab1 = np.where(free, owner, seeds); lab = np.zeros_like(lab1)
    for i, sl in enumerate(ndi.find_objects(lab1), start=1):
        if sl is None: continue
        m = lab1[sl] == i; cc, _ = ndi.label(m); keep = np.unique(cc[(seeds[sl] == i) & m]); keep = keep[keep > 0]
        mm = ndi.binary_fill_holes(np.isin(cc, keep)) & ((lab1[sl] == i) | (lab1[sl] == 0)); lab[sl][mm] = i
    return lab, n_blob

# ---- scoring inside the window
inner = (xs >= pad * UM) & (xs < (pad + x1 - x0) * UM) & (ys >= pad * UM) & (ys < (pad + y1 - y0) * UM)
ti = tx[inner]; gc = (ti.x_location.to_numpy() / UM).astype(int); gr = (ti.y_location.to_numpy() / UM).astype(int)
genes = ti.feature_name.astype(str).to_numpy(); on_nuc = ti.overlaps_nucleus.astype(bool).to_numpy(); xy = ti[["x_location", "y_location"]].to_numpy()
rows = []
def add(name, lb):
    s = score(name, lb, genes, on_nuc, xy); s["mecr_n1000"] = round(mecr_downsampled(lb, genes, n=1000), 4)
    rows.append({"window": WIN, "method": name, **{k: s[k] for k in ("pct_assigned", "n_cells", "median_tx_per_cell", "mecr", "mecr_n1000", "shell_coherence_median")}})
    print(WIN, name, s["pct_assigned"], s["mecr"], s["mecr_n1000"], flush=True)
cid = ti.cell_id.astype(str)
add("10x", (pd.factorize(cid.where(~cid.isin(["", "UNASSIGNED"]), None))[0] + 1).astype(np.int64))
for name, d in (("PF", "crops_PF_500"), ("I2", "crops_I2_500")):
    f = RUNS / d / WIN / "final_labels.npy"
    if f.exists(): add(name, np.asarray(np.load(f, mmap_mode="r")[gr, gc]).astype(np.int64))
# ---- per-transcript cleaning (post-step for any segmentation): a transcript of an informative gene leaves its cell when
# its gene favours another type t' by >= lr (log units) over the cell's own type AND type t' has local support (p_t' >= q
# at the transcript). The cell type is read from the cell's own transcripts. Border-swap test: transcripts of a type-b cell
# within 1.5 um of a type-a cell are treated as if assigned to the a cell (known errors); recall = share rejected; the a cell's
# own transcripts within 1.5 um of a b cell give the false-rejection rate.
rr_in, cc_in = gr - Y0, gc - X0; gi_in = gi[inner]; ok_in = gi_in >= 0
Ppix = P[:, rr_in, cc_in]
def clean(lb, lr, q, test=False):
    nl = int(lb.max()) + 1; okl = ok_in & (lb > 0)
    ll = np.stack([np.bincount(lb[okl], weights=logp[t][gi_in[okl]], minlength=nl) for t in range(T)])
    tcell = ll.argmax(0); tcell[np.bincount(lb[okl], minlength=nl) == 0] = -1
    tc = np.where(lb > 0, tcell[lb], -1)
    def rule(tc_, idx):
        g = gi_in[idx]; lp = logp[:, g]; own = lp[np.clip(tc_, 0, None), np.arange(len(idx))]
        return (((lp - own) >= lr) & (Ppix[:, idx] >= q)).any(0)
    rej = np.zeros(len(lb), bool); idx = np.flatnonzero(okl & (tc >= 0)); rej[idx] = rule(tc[idx], idx)
    out = np.where(rej, 0, lb)
    res = {"removed_pct_of_assigned": round(100 * rej.sum() / max((lb > 0).sum(), 1), 2)}
    if test:
        tcp = np.full(int(lab_for_test.max()) + 1, -1); k = min(len(tcp), len(tcell_px)); tcp[:k] = tcell_px[:k]
        tmap = np.where(lab_for_test > 0, tcp[lab_for_test], -1)
        D = np.stack([ndi.distance_transform_edt(tmap != t) * UM for t in range(T)])[:, rr_in, cc_in]
        Dm = D.copy(); Dm[np.clip(tc, 0, None), np.arange(len(tc))] = np.inf
        near_t, near_d = Dm.argmin(0), Dm.min(0)
        band = okl & (tc >= 0) & (near_d <= 1.5)
        sw = np.flatnonzero(band); rec = rule(near_t[sw], sw)                    # pretend they belong to the neighbour
        nat = rej[np.flatnonzero(band)]
        res.update(swap_recall_informative=round(float(rec.mean()), 3), native_false_reject_informative=round(float(nat.mean()), 3),
                   swap_recall_all_tx=round(float(rec.sum() / max(((lb > 0) & (tc >= 0) & (near_d <= 1.5)).sum(), 1)), 3))
    return out, res, tcell
CLEAN = [tuple(map(float, c.split(":"))) for c in os.environ["CLEAN"].split(",")] if os.environ.get("CLEAN") else []
def add_clean(name, lb, lab_px=None):
    global lab_for_test, tcell_px
    for lr, q in CLEAN:
        test = lab_px is not None
        if test:
            _, _, tcell_px = clean(lb, lr, q); lab_for_test = lab_px
        out, res, _ = clean(lb, lr, q, test)
        add(f"{name} +clean {lr:g}:{q:g}", out); rows[-1].update(res); print("   ", res, flush=True)
if CLEAN:
    add_clean("10x", (pd.factorize(cid.where(~cid.isin(["", "UNASSIGNED"]), None))[0] + 1).astype(np.int64))
    f = RUNS / "crops_I2_500" / WIN / "final_labels.npy"
    if f.exists(): add_clean("I2", np.asarray(np.load(f, mmap_mode="r")[gr, gc]).astype(np.int64))
settings = [g.split(":") for g in os.environ["GRID"].split(",")] if os.environ.get("GRID") else [(os.environ.get("PMIN", "0.5"), os.environ.get("REACH_ASTRO", "3"))]
for pm, ra in settings:
    lab, n_blob = segment(float(pm), {**REACH, "astro": float(ra)})
    add(f"S2 {pm}:{ra}", lab[gr - Y0, gc - X0].astype(np.int64))
    if CLEAN: add_clean(f"S2 {pm}:{ra}", lab[gr - Y0, gc - X0].astype(np.int64), lab)
    if os.environ.get("SAVE") == "1":
        od = RUNS / "crops_S2_500" / WIN; od.mkdir(parents=True, exist_ok=True)
        shape = np.load(RUNS / "crops_PF_500" / WIN / "final_labels.npy", mmap_mode="r").shape
        fl = np.lib.format.open_memmap(od / "final_labels.npy", mode="w+", dtype=np.uint32, shape=shape)
        fl[y0:y1, x0:x1] = lab[pad:pad + y1 - y0, pad:pad + x1 - x0].astype(np.uint32); fl.flush()
        json.dump({"window": [x0, y0, x1, y1], "stage2": {"pmin": pm, "reach_astro": ra, "tau": TAU}, "nucleus_free_seeds": n_blob}, open(od / "stats.json", "w"))
pd.DataFrame(rows).to_csv(OUT / f"{WIN}_{'grid' if os.environ.get('GRID') else 'final'}.csv", index=False)
