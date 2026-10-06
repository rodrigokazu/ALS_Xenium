#!/usr/bin/env python3
"""Full-sample assembly and post-steps for the Atera cerebellum, tile by tile (4096 px tiles = the zarr chunks).

  stitch : chunk runs of run_fullsample.py --window (C500 settings, sparse memmaps on scratch) -> base labels zarr.
           Chunk i's labels get offset i * 1,000,000; a pixel claimed by two chunks keeps the first (each chunk only keeps
           the cells whose centroid lies in its window, so overlaps are rare seam cases). Also writes base_cells.parquet.
  i2     : phase3.py with the I2 settings (identity-driven territory: conf 2, low-confidence pixels free, reach 4 um, Purkinje /
           interneuron reach 14 um, density floor 40), unchanged logic, on each tile plus a 160 px margin; writes the core.
  pcr    : pc_repair.py logic (Purkinje soma = set-A Purkinje genes + 18S, one whole cell per soma) on each tile plus a 320 px
           margin; new cells get an id from their soma centroid so neighbouring tiles agree.
  count  : cell x gene counts of QC transcripts (qv >= 20) under the final labels, plus a cells table.
python full_post.py <stage> [--workers N]
"""
import sys, os, json, argparse, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, zarr, tifffile
from scipy import ndimage as ndi, sparse
from skimage.segmentation import watershed
from skimage.draw import polygon as draw_poly
sys.path.insert(0, str(Path(__file__).parent)); from pc_soma import soma_cores

UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); BUNDLE = SSD / "Cerebellum_sample"
FULL = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/c63ac4c6-74e3-40d6-bf81-dcc9df19b8fd/scratchpad/full")
OUTD = Path(os.environ.get("FULL_OUT", SSD / "segmentation_full")); H, W, T = 42511, 75409, 4096
HYBD = SSD / "segmentation_hybridC"
STORES = {"base": OUTD / "base_labels.zarr", "i2": OUTD / "i2_labels.zarr", "pcr": OUTD / "final_labels.zarr", "hybc": HYBD / "final_labels.zarr"}
GROUPS = {"granule": [1, 2, 3, 4, 5, 8, 9, 10, 15, 6, 7, 11], "PC/interneuron": [14, 23, 33], "astro": [13, 16, 19],
          "oligo": [12, 18, 27, 29, 30, 32], "vascular": [17, 20, 21, 22, 26, 28], "immune": [24, 25, 31, 34, 35, 36]}
TYPES = list(GROUPS)
I2 = dict(lfc=2.0, mean_min=0.3, tlow=40.0, r_um=4.0, r_pc_um=14.0, sigma_um=2.0, bin_px=4, min_tx=30, conf=2.0)
PCR_THR, PCR_GROW, PCT18 = 0.534, 4.0, (2.0, 608.0)

def tiles():
    out = []
    for l in open(os.environ.get("CHUNKS", FULL / "chunks.txt")):
        n, x0, y0, x1, y1 = l.split(); out.append((int(x0), int(y0), int(x1), int(y1)))
    return out

def store(name, mode="r"):
    if mode == "w":
        return zarr.open_array(str(STORES[name]), mode="w", shape=(H, W), chunks=(T, T), dtype=np.uint32, fill_value=0)
    return zarr.open_array(str(STORES[name]), mode=mode)

def read_tx(x0, y0, x1, y1, cols, gene_only=False):
    f = (pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM) & (pc.field("qv") >= 20)
    if gene_only: f = f & pc.field("is_gene")
    return ds.dataset(BUNDLE / "transcripts.parquet").to_table(columns=cols, filter=f).to_pandas(strings_to_categorical=True)

def profiles():
    de = pd.read_csv(BUNDLE / "analysis/diffexp/gene_expression_graphclust/differential_expression.csv")
    cl = pd.read_csv(BUNDLE / "analysis/clustering/gene_expression_graphclust/clusters.csv")
    ncl = cl.Cluster.value_counts().sort_index().reindex(range(1, 37)).to_numpy().astype(float)
    M = np.column_stack([de[f"Cluster {k} Mean Counts"] for k in range(1, 37)]); Lf = np.column_stack([de[f"Cluster {k} Log2 fold change"] for k in range(1, 37)])
    Pv = np.column_stack([de[f"Cluster {k} Adjusted p value"] for k in range(1, 37)])
    inf = ((Lf >= I2["lfc"]) & (M >= I2["mean_min"]) & (Pv < 0.01)).any(1)
    Mi = M[inf]; prof = np.zeros((len(TYPES), inf.sum()))
    for t, ks in enumerate(GROUPS.values()):
        w = ncl[np.array(ks) - 1]; prof[t] = (Mi[:, np.array(ks) - 1] * w).sum(1) / w.sum()
    prof = prof + 1e-4; prof /= prof.sum(1, keepdims=True)
    return pd.Index(de["Feature Name"].to_numpy()[inf]), np.log(prof)

def padded(core, m):
    x0, y0, x1, y1 = core
    return max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, W), min(y1 + m, H)

# ------------------------------------------------------------------ stitch
def stitch():
    OUTD.mkdir(parents=True, exist_ok=True); z = store("base", "w"); cells = []
    for i, l in enumerate(open(FULL / "chunks.txt")):
        if os.environ.get("CHUNKS") and l.split()[0] not in open(os.environ["CHUNKS"]).read(): continue
        name, x0, y0, x1, y1 = l.split(); x0, y0, x1, y1 = map(int, (x0, y0, x1, y1)); d = FULL / "base" / name
        if not (d / "cells.parquet").exists(): print("MISSING", name, flush=True); continue
        fl = np.load(d / "final_labels.npy", mmap_mode="r"); X0, Y0, X1, Y1 = padded((x0, y0, x1, y1), 400)
        a = np.asarray(fl[Y0:Y1, X0:X1]).astype(np.uint32); off = np.uint32((i + 1) * 1_000_000)
        assert a.max() < 1_000_000
        cur = np.asarray(z[Y0:Y1, X0:X1]); put = (a > 0) & (cur == 0); cur[put] = a[put] + off; z[Y0:Y1, X0:X1] = cur
        c = pd.read_parquet(d / "cells.parquet"); c["label"] = c["label"].astype(np.int64) + int(off); c["chunk"] = name; cells.append(c)
        print(f"stitched {i + 1} {name} cells {len(c)} seam-lost px {int(((a > 0) & ~put).sum())}", flush=True)
    pd.concat(cells).to_parquet(OUTD / "base_cells.parquet", index=False)

# ------------------------------------------------------------------ i2 (phase3.py logic, I2 settings)
def i2_tile(core):
    t0 = time.time(); gidx, logp = profiles(); X0, Y0, X1, Y1 = padded(core, 160); Hh, Wd = Y1 - Y0, X1 - X0
    raw = np.asarray(store("base")[Y0:Y1, X0:X1]).astype(np.int64)
    uq, inv = np.unique(raw, return_inverse=True); lab0 = inv.reshape(raw.shape).astype(np.int64)      # compact; 0 stays 0 (uq[0] == 0)
    if uq[0] != 0: lab0 += 1; uq = np.r_[0, uq]
    out = lab0
    if lab0.max() > 0:
        tx = read_tx(X0, Y0, X1, Y1, ["x_location", "y_location", "feature_name"])
        gi = gidx.get_indexer(tx.feature_name.astype(str)); ok = gi >= 0
        col = np.clip((tx.x_location.to_numpy() / UM).astype(int) - X0, 0, Wd - 1)[ok]; row = np.clip((tx.y_location.to_numpy() / UM).astype(int) - Y0, 0, Hh - 1)[ok]; gi = gi[ok]
        g = I2["bin_px"]; nby, nbx = Hh // g + 1, Wd // g + 1; flat = (row // g) * nbx + (col // g); nT = len(TYPES)
        S = np.stack([np.bincount(flat, weights=logp[t][gi], minlength=nby * nbx).reshape(nby, nbx) for t in range(nT)])
        cnt = np.bincount(flat, minlength=nby * nbx).reshape(nby, nbx).astype(float); sg = I2["sigma_um"] / (g * UM)
        Ss = np.stack([ndi.gaussian_filter(S[t], sg) for t in range(nT)]); cs = ndi.gaussian_filter(cnt, sg)
        lab_at = lab0[row, col]; nlab = int(lab0.max()) + 1
        cell_ll = np.stack([np.bincount(lab_at, weights=logp[t][gi], minlength=nlab) for t in range(nT)]); cell_n = np.bincount(lab_at, minlength=nlab)
        ctype = np.where(cell_n >= I2["min_tx"], cell_ll.argmax(0), -1); ctype[0] = -1
        dens = ndi.gaussian_filter(np.asarray(zarr.open_array(str(SSD / "work/density/tx_density.zarr"), mode="r")[Y0:Y1, X0:X1]).astype(np.float32), 1.0 / UM)
        marg = (Ss - Ss.max(0, keepdims=True)).astype(np.float32)
        up = np.stack([ndi.zoom(marg[t], g, order=1)[:Hh, :Wd] for t in range(nT)]); csu = ndi.zoom(cs.astype(np.float32), g, order=1)[:Hh, :Wd]
        if up.shape[1:] != (Hh, Wd):
            pd_ = ((0, 0), (0, Hh - up.shape[1]), (0, Wd - up.shape[2])); up = np.pad(up, pd_, mode="edge"); csu = np.pad(csu, pd_[1:], mode="edge")
        ptype = up.argmax(0); srt = np.sort(up, axis=0); pconf = -srt[-2]; del up, srt
        confident = (csu >= 0.5) & (pconf >= I2["conf"]); free = lab0 == 0; tissue = free & (dens > I2["tlow"])
        PCI = TYPES.index("PC/interneuron"); owner = np.zeros_like(lab0); reach = np.zeros(lab0.shape, bool)
        for t in range(nT):
            ct = (ctype[lab0] == t) & (lab0 > 0)
            if not ct.any(): continue
            dt = ndi.distance_transform_edt(~ct) * UM; lim = I2["r_pc_um"] if t == PCI else I2["r_um"]
            allow = tissue & confident & (ptype == t) & (dt <= lim)
            wt = watershed(-dens, markers=np.where(ct, lab0, 0).astype(np.int32), mask=ct | allow)
            take = allow & (wt > 0); owner[take] = wt[take]; reach |= take
        new = reach & free; owner = np.where(new, owner, 0); lab1 = np.where(new, owner, lab0); out = lab1.copy()
        for L, sl in enumerate(ndi.find_objects(lab1.astype(np.int32)), start=1):
            if sl is None or L >= nlab: continue
            m = lab1[sl] == L; cc, _ = ndi.label(m); keep = np.unique(cc[(lab0[sl] == L) & m]); keep = keep[keep > 0]
            mm = ndi.binary_fill_holes(np.isin(cc, keep)) & ((lab0[sl] == 0) | (lab0[sl] == L))
            out[sl][(lab1[sl] == L) & ~mm] = 0; out[sl][mm & (out[sl] == 0)] = L
    res = uq[out].astype(np.uint32)
    x0, y0, x1, y1 = core
    store("i2", "r+")[y0:y1, x0:x1] = res[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    return core, round(time.time() - t0)

# ------------------------------------------------------------------ pcr (pc_repair.py logic)
def pcr_tile(core):
    t0 = time.time(); setA = set(json.load(open(SSD / "compare/purkinje/gene_sets.json"))["purkinje_A"])
    X0, Y0, X1, Y1 = padded(core, 320); Hh, Wd = Y1 - Y0, X1 - X0; x0, y0, x1, y1 = core
    lab = np.asarray(store("i2")[Y0:Y1, X0:X1]).astype(np.int64); stats = dict(somata=0, new_cells=0, absorbed=0)
    tx = read_tx(X0, Y0, X1, Y1, ["x_location", "y_location", "feature_name"], gene_only=True)
    if len(tx):
        r = np.clip((tx.y_location.to_numpy() / UM).astype(int) - Y0, 0, Hh - 1); c = np.clip((tx.x_location.to_numpy() / UM).astype(int) - X0, 0, Wd - 1)
        isA = tx.feature_name.astype(str).isin(setA).to_numpy()
        cntA = np.bincount(r[isA] * Wd + c[isA], minlength=Hh * Wd).reshape(Hh, Wd).astype(np.float32)
        dA = ndi.gaussian_filter(cntA, 1.5 / UM) / UM ** 2
        dall = ndi.gaussian_filter(np.bincount(r * Wd + c, minlength=Hh * Wd).reshape(Hh, Wd).astype(np.float32), 1.0 / UM) / UM ** 2
        tf = tifffile.TiffFile(BUNDLE / "morphology_focus" / "ch0000_dapi.ome.tif"); zi = zarr.open(tf.aszarr(level=0, series=0), mode="r")
        s18 = (ndi.gaussian_filter(np.asarray(zi[2, Y0:Y1, X0:X1]).astype(np.float32), 0.5 / UM) - PCT18[0]) / (PCT18[1] - PCT18[0])
        cl, sst = soma_cores(dA, dall, s18, PCR_THR); stats.update(sst); n = int(cl.max())
        if n:
            big = np.arange(1, n + 1)
            if len(big):
                nb = ds.dataset(BUNDLE / "nucleus_boundaries.parquet").to_table(filter=(pc.field("vertex_x") >= X0 * UM - 15) & (pc.field("vertex_x") < X1 * UM + 15)
                                                                                 & (pc.field("vertex_y") >= Y0 * UM - 15) & (pc.field("vertex_y") < Y1 * UM + 15)).to_pandas()
                nuc = np.zeros((Hh, Wd), np.int32)
                for k, (_, gg) in enumerate(nb.groupby("cell_id", sort=False), start=1):
                    rr, cc_ = draw_poly(gg.vertex_y.to_numpy() / UM - Y0, gg.vertex_x.to_numpy() / UM - X0, shape=(Hh, Wd)); nuc[rr, cc_] = k
                nuc_area = np.bincount(nuc.ravel()); objs = ndi.find_objects(cl); used = set()
                for k in big:
                    sl0 = objs[k - 1]; cyx = ndi.center_of_mass(cl[sl0] == k)
                    cy, cx = Y0 + sl0[0].start + cyx[0], X0 + sl0[1].start + cyx[1]
                    gm = int(PCR_GROW / UM) + 2
                    # every soma whose (grown) box touches this tile's core is repaired here, the same way in each tile it touches;
                    # the 320 px margin holds the rest of its body, and only the core is written
                    by0, by1 = Y0 + sl0[0].start - gm, Y0 + sl0[0].stop + gm; bx0, bx1 = X0 + sl0[1].start - gm, X0 + sl0[1].stop + gm
                    if by1 <= y0 or by0 >= y1 or bx1 <= x0 or bx0 >= x1: continue
                    sl = tuple(slice(max(s.start - gm, 0), s.stop + gm) for s in sl0)
                    co = cl[sl] == k; med18 = float(np.median(s18[sl][co]))
                    allow = co | ((s18[sl] >= 0.5 * med18) & (dA[sl] >= PCR_THR / 3) & (dall[sl] > 20))
                    ext = co.copy()
                    for _ in range(int(PCR_GROW / UM)): ext = ndi.binary_dilation(ext) & allow
                    ext = ndi.binary_fill_holes(ext) & ~((cl[sl] > 0) & (cl[sl] != k))
                    nl = nuc[sl]; inside = np.bincount(nl[co].ravel(), minlength=len(nuc_area)) / np.maximum(nuc_area, 1)
                    prot = (nl > 0) & (inside[nl] < 0.5); ext &= ~prot
                    ls = lab[sl]; aA = cntA[sl]
                    cand = pd.Series(aA[ext], index=ls[ext]).groupby(level=0).sum().drop(0, errors="ignore")
                    cand = cand.drop([u for u in used if u in cand.index]).sort_values(ascending=False)   # one soma per owner
                    if len(cand) and cand.iloc[0] >= 0.2 * aA[ext].sum(): owner = int(cand.index[0])
                    else: owner = 3_500_000_000 + int(cy // 8) * 10_000 + int(cx // 8); stats["new_cells"] += 1
                    for cid in np.unique(ls[ext]):
                        if cid in (0, owner): continue
                        m = ls == cid
                        if (m & ext).sum() >= 0.6 * m.sum() and not (m & prot).any(): ls[m] = owner; stats["absorbed"] += 1
                    ls[ext] = owner; lab[sl] = ls; stats["somata"] += 1; used.add(owner)
    store("pcr", "r+")[y0:y1, x0:x1] = lab[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0].astype(np.uint32)
    return core, round(time.time() - t0), stats

# ------------------------------------------------------------------ count
# ------------------------------------------------------------------ hybc (hybrid_seg.py rule C, whole section)
_HC = {}
def _hc():
    if not _HC:
        ann = pd.read_parquet(OUTD / "annotation/cells_annotated.parquet"); _HC["lin"] = dict(zip(ann.index.astype(np.int64), ann.lineage))
        cf = pd.read_parquet(OUTD / "cells.parquet"); Xf = sparse.load_npz(OUTD / "cell_by_gene.npz").tocsr(); gf = pd.read_csv(OUTD / "genes.csv").gene.astype(str)
        setA = json.load(open(SSD / "compare/purkinje/gene_sets.json"))["purkinje_A"]
        fa = np.asarray(Xf[:, np.isin(gf, setA)].sum(1)).ravel() / np.maximum(np.asarray(Xf.sum(1)).ravel(), 1); _HC["fA"] = dict(zip(cf.cell.astype(np.int64), fa))
        b = pd.read_parquet(OUTD / "base_cells.parquet")[["label", "nucleus_id"]]; _HC["nuc"] = dict(zip(b.label.astype(np.int64), b.nucleus_id))
        tc = pd.read_parquet(BUNDLE / "cells.parquet", columns=["cell_id"]); _HC["tid"] = dict(zip(tc.cell_id, np.arange(len(tc)) + 2_000_000_000))
    return _HC
NONNEURO = {"Astroglia", "Oligodendroglia", "Vascular", "Immune", "Fibroblast / meningeal"}
def hybc_tile(core):
    from skimage.draw import polygon as dpoly
    t0 = time.time(); h = _hc(); X0, Y0, X1, Y1 = padded(core, 160); Hh, Wd = Y1 - Y0, X1 - X0; x0, y0, x1, y1 = core
    O = np.asarray(store("pcr")[Y0:Y1, X0:X1]).astype(np.int64)
    cb = ds.dataset(BUNDLE / "cell_boundaries.parquet").to_table(filter=(pc.field("vertex_x") >= X0 * UM - 30) & (pc.field("vertex_x") < X1 * UM + 30)
         & (pc.field("vertex_y") >= Y0 * UM - 30) & (pc.field("vertex_y") < Y1 * UM + 30)).to_pandas()
    T = np.zeros((Hh, Wd), np.int64)
    for cid, g in cb.groupby("cell_id", sort=False):
        rr, cc = dpoly(g.vertex_y.to_numpy() / UM - Y0, g.vertex_x.to_numpy() / UM - X0, shape=(Hh, Wd)); T[rr, cc] = h["tid"].get(cid, 0)
    ids = np.unique(O[O > 0])
    keep = [i for i in ids if not (h["lin"].get(i) in NONNEURO and h["fA"].get(i, 0.0) < 0.0164)]
    drop = [h["tid"][h["nuc"][i]] for i in keep if h["nuc"].get(i) in h["tid"]]
    Hc = np.where(np.isin(T, drop), 0, T); km = np.isin(O, keep); Hc[km] = O[km]
    u, cnt = np.unique(Hc[Hc >= 2_000_000_000], return_counts=True); Hc[np.isin(Hc, u[cnt * UM ** 2 < 10])] = 0
    store("hybc", "r+")[y0:y1, x0:x1] = Hc[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0].astype(np.uint32)
    return core, round(time.time() - t0), {"ours_kept": len(keep), "ours": len(ids)}

def count_tile(core):
    x0, y0, x1, y1 = core; lab = np.asarray(store(os.environ.get("COUNT_STORE", "pcr"))[y0:y1, x0:x1])
    tx = read_tx(x0, y0, x1, y1, ["x_location", "y_location", "feature_name", "overlaps_nucleus"], gene_only=True)
    if not len(tx): return core, None
    r = np.clip((tx.y_location.to_numpy() / UM).astype(int) - y0, 0, y1 - y0 - 1); c = np.clip((tx.x_location.to_numpy() / UM).astype(int) - x0, 0, x1 - x0 - 1)
    lb = lab[r, c].astype(np.int64)
    genes_all = pd.Index(pd.read_csv(BUNDLE / "analysis/diffexp/gene_expression_graphclust/differential_expression.csv", usecols=["Feature Name"])["Feature Name"])
    gmap = genes_all.get_indexer(tx.feature_name.cat.categories.astype(str)); g = gmap[tx.feature_name.cat.codes.to_numpy()]
    tot = dict(n_tx=len(lb), n_assigned=int((lb > 0).sum()))
    k = (lb > 0) & (g >= 0); key, n = np.unique(lb[k] * 32768 + g[k], return_counts=True)
    m = pd.DataFrame({"cell": key // 32768, "gene": genes_all.to_numpy()[key % 32768], "n": n})
    ys, xs = np.nonzero(lab); cl_ = lab[ys, xs].astype(np.int64); u, inv, npx = np.unique(cl_, return_inverse=True, return_counts=True)
    cell_px = pd.DataFrame({"cell": u, "px": npx, "sx": np.bincount(inv, weights=xs + x0), "sy": np.bincount(inv, weights=ys + y0)})
    return core, (m, cell_px, tot)

def run(stage, workers):
    fn = {"i2": i2_tile, "pcr": pcr_tile, "hybc": hybc_tile, "count": count_tile}[stage]; ts = tiles(); t0 = time.time()
    if stage in ("i2", "pcr", "hybc"): (HYBD.mkdir(exist_ok=True) if stage == "hybc" else None); store(stage, "w")
    acc, px, tot = [], [], dict(n_tx=0, n_assigned=0)
    with ProcessPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(fn, ts), start=1):
            if stage == "count":
                if res[1] is not None:
                    acc.append(res[1][0]); px.append(res[1][1]); tot = {k: tot[k] + res[1][2][k] for k in tot}
            else:
                print(f"{stage} tile {i}/{len(ts)} {res[0]} {res[1]}s {res[2] if len(res) > 2 else ''} ({time.time() - t0:.0f}s)", flush=True)
    if stage == "count":
        m = pd.concat(acc).groupby(["cell", "gene"]).n.sum().reset_index()
        p = pd.concat(px).groupby("cell").sum().reset_index()
        cells = p.assign(area_um2=p.px * UM ** 2, x_centroid_um=p.sx / p.px * UM, y_centroid_um=p.sy / p.px * UM)[["cell", "area_um2", "x_centroid_um", "y_centroid_um"]]
        cells = cells.merge(m.groupby("cell").n.sum().rename("n_transcripts").reset_index(), on="cell", how="left").fillna({"n_transcripts": 0})
        genes = sorted(m.gene.unique()); gi = pd.Index(genes); ci = pd.Index(cells.cell)
        X = sparse.csr_matrix((m.n.to_numpy(), (ci.get_indexer(m.cell), gi.get_indexer(m.gene))), shape=(len(ci), len(gi)))
        CO = Path(os.environ.get("COUNT_OUT", OUTD)); CO.mkdir(exist_ok=True)
        sparse.save_npz(CO / "cell_by_gene.npz", X); pd.Series(genes).to_csv(CO / "genes.csv", index=False, header=["gene"])
        cells.to_parquet(CO / "cells.parquet", index=False)
        json.dump({**tot, "pct_assigned": round(100 * tot["n_assigned"] / max(tot["n_tx"], 1), 2), "n_cells": len(cells)}, open(CO / "summary.json", "w"), indent=1)
        print("count done", tot, "cells", len(cells), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if a.stage == "stitch": stitch()
    else: run(a.stage, a.workers)
