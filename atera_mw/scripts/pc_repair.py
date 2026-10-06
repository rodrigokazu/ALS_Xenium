#!/usr/bin/env python3
"""Purkinje repair: a post-step on any segmentation that makes every Purkinje soma one whole cell.

Uses only Purkinje set-A genes (compare/purkinje/gene_sets.json) and the 18S stain, so pc_eval.py can score the result with
the set-B genes. Per window:
  core   = pc_soma.soma_cores: set-A density >= the pc_eval.py threshold, with an 18S body, compact, split at 2+ density peaks
  extent = the core grown, at most GROW um, into pixels that are 18S-bright (>= half the core's median 18S) and still carry
           Purkinje signal (set-A density >= thr / 3): 18S draws the cell body edge, the transcripts say it is a Purkinje cell
  protected = 10x nuclei that do not belong to the soma (< 50% inside the core): they keep their own cells
  owner  = the base cell holding most set-A transcripts in the extent, or a new cell when no base cell holds >= 20%
  The owner gets the whole extent minus protected nuclei; base cells lying >= 60% inside the extent and holding no protected
  nucleus are absorbed into it. Written as final_labels.npy (full-size memmap) to crops_<out>_500/<window>.
python pc_repair.py <base run dir name, e.g. crops_I2_500> <out name, e.g. PCR_I2> <soma threshold> <window> [...]
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_poly
sys.path.insert(0, str(Path(__file__).parent)); from pc_soma import soma_cores

UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB")
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
BASE, NAME, THR = sys.argv[1], sys.argv[2], float(sys.argv[3]); GROW = 4.0; PCT18 = (2.0, 608.0)
setA = set(json.load(open(SSD / "compare/purkinje/gene_sets.json"))["purkinje_A"])

for win in sys.argv[4:]:
    x0, y0, x1, y1 = json.load(open(RUNS / "crops_PF_500" / win / "stats.json"))["window"]; H, W = y1 - y0, x1 - x0
    bundle = SSD / "work" / ("local_bundle_heldout" if win.startswith("held") else "local_bundle")
    tx = ds.dataset(bundle / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name"],
        filter=(pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM)
        & (pc.field("qv") >= 20) & (pc.field("is_gene"))).to_pandas(strings_to_categorical=True)
    r = np.clip((tx.y_location.to_numpy() / UM).astype(int) - y0, 0, H - 1); c = np.clip((tx.x_location.to_numpy() / UM).astype(int) - x0, 0, W - 1)
    isA = tx.feature_name.astype(str).isin(setA).to_numpy()
    cntA = np.bincount(r[isA] * W + c[isA], minlength=H * W).reshape(H, W).astype(np.float32)
    dA = ndi.gaussian_filter(cntA, 1.5 / UM) / UM ** 2
    dall = ndi.gaussian_filter(np.bincount(r * W + c, minlength=H * W).reshape(H, W).astype(np.float32), 1.0 / UM) / UM ** 2
    tf = tifffile.TiffFile(bundle / "morphology_focus" / "ch0000_dapi.ome.tif"); z = zarr.open(tf.aszarr(level=0, series=0), mode="r")
    s18 = (ndi.gaussian_filter(np.asarray(z[2, y0:y1, x0:x1]).astype(np.float32), 0.5 / UM) - PCT18[0]) / (PCT18[1] - PCT18[0])
    nb = ds.dataset(bundle / "nucleus_boundaries.parquet").to_table(filter=(pc.field("vertex_x") >= x0 * UM - 15) & (pc.field("vertex_x") < x1 * UM + 15)
                                                                     & (pc.field("vertex_y") >= y0 * UM - 15) & (pc.field("vertex_y") < y1 * UM + 15)).to_pandas()
    nuc = np.zeros((H, W), np.int32)
    for k, (_, g) in enumerate(nb.groupby("cell_id", sort=False), start=1):
        rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - y0, g.vertex_x.to_numpy() / UM - x0, shape=(H, W)); nuc[rr, cc] = k
    fl0 = np.load(RUNS / BASE / win / "final_labels.npy", mmap_mode="r")
    lab = np.asarray(fl0[y0:y1, x0:x1]).astype(np.int64); nxt = int(fl0.max()) + 1 if False else int(lab.max()) + 10_000_000
    cl, sst = soma_cores(dA, dall, s18, THR); objs = ndi.find_objects(cl)
    stats = dict(somata=0, new_cells=0, absorbed=0, **sst)
    nuc_area = np.bincount(nuc.ravel()); used = set()
    for k in range(1, int(cl.max()) + 1):
        if objs[k - 1] is None: continue
        core = cl == k; sl = tuple(slice(max(s.start - int(GROW / UM) - 2, 0), s.stop + int(GROW / UM) + 2) for s in objs[k - 1])
        co = core[sl]; med18 = float(np.median(s18[sl][co]))
        allow = co | ((s18[sl] >= 0.5 * med18) & (dA[sl] >= THR / 3) & (dall[sl] > 20))
        ext = co.copy()
        for _ in range(int(GROW / UM)):
            ext = ndi.binary_dilation(ext) & allow
        ext = ndi.binary_fill_holes(ext) & ~((cl[sl] > 0) & (cl[sl] != k))
        nl = nuc[sl]; inside = np.bincount(nl[co].ravel(), minlength=len(nuc_area)) / np.maximum(nuc_area, 1)
        prot = (nl > 0) & (inside[nl] < 0.5)
        ext &= ~prot
        ls = lab[sl]; aA = cntA[sl]
        cand = pd.Series(aA[ext], index=ls[ext]).groupby(level=0).sum().drop(0, errors="ignore")
        cand = cand.drop([u for u in used if u in cand.index]).sort_values(ascending=False)   # one soma per owner
        if len(cand) and cand.iloc[0] >= 0.2 * aA[ext].sum():
            owner = int(cand.index[0])
        else:
            owner = nxt; nxt += 1; stats["new_cells"] += 1
        # absorb base cells mostly inside the extent that hold no protected nucleus
        for cid in np.unique(ls[ext]):
            if cid in (0, owner): continue
            m = ls == cid
            if (m & ext).sum() >= 0.6 * m.sum() and not (m & prot).any():
                ls[m] = owner; stats["absorbed"] += 1
        ls[ext] = owner; lab[sl] = ls; stats["somata"] += 1; used.add(owner)
    od = RUNS / f"crops_{NAME}_500" / win; od.mkdir(parents=True, exist_ok=True)
    fl = np.lib.format.open_memmap(od / "final_labels.npy", mode="w+", dtype=np.uint32, shape=fl0.shape)
    fl[y0:y1, x0:x1] = lab.astype(np.uint32); fl.flush()
    json.dump({"window": [x0, y0, x1, y1], "pc_repair": {"base": BASE, "thr": THR, "grow_um": GROW}, **stats}, open(od / "stats.json", "w"))
    print(win, stats, flush=True)
