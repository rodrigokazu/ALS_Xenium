#!/usr/bin/env python3
"""Hybrid segmentation on the nine 500 um windows: our cells where we are better (neurons), 10x cells elsewhere.

Type of each of our cells = 10x graph-cluster of the 10x nucleus it was seeded from (base_cells.nucleus_id); Purkinje-repair
cells without a nucleus (id >= 3.5e9) are Purkinje. Groups as in celltype_capture.py.
  A: ours for Purkinje, MLI, UBC; 10x for everything else
  B: ours for Purkinje, MLI, UBC and granule cells; 10x for astroglia, oligodendroglia, vascular, immune
Construction: start from the 10x label raster (cell_boundaries polygons); 10x cells whose nucleus belongs to a kept "ours" cell
are removed; the kept ours cells are painted on top; 10x remnants smaller than 10 um2 after that are dropped.
Writes crops_HYB<A|B>_500/<window>/final_labels.npy (full-size memmap) for win_score.py / pc_eval.py / celltype_capture.py.
python hybrid_seg.py"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, zarr
from skimage.draw import polygon as draw_poly
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB")
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
WINS = ["spot1_purkinje_left", "spot2_gl_wm_centre", "spot3_purkinje_right"] + [f"held{i}" for i in range(1, 7)]
GRP = {"Purkinje": [33], "MLI": [14], "UBC": [23], "Granule": [1, 2, 3, 4, 5, 8, 9, 10, 15, 6, 7, 11], "Astroglia": [13, 16, 19],
       "Oligodendroglia": [12, 18, 27, 29, 30, 32], "Vascular": [17, 20, 21, 22, 26, 28], "Immune": [24, 25, 31, 34, 35, 36]}
KEEP = {"A": {"Purkinje", "MLI", "UBC"}, "B": {"Purkinje", "MLI", "UBC", "Granule"}, "C": None}
# C (rule fixed before scoring): a cell of ours is replaced by 10x cells only if OUR annotation calls it glia / vascular / immune
# AND its share of Purkinje set-A transcripts is below 0.0164 (midpoint between 10x Purkinje cluster 33, 0.0270, and the
# highest non-neuronal cluster 13, 0.0058, from the 10x cluster table). Scoring uses set-B genes / 10x cluster genes only.
from scipy import sparse
ann = pd.read_parquet(SSD / "segmentation_full/annotation/cells_annotated.parquet"); lin = dict(zip(ann.index.astype(np.int64), ann.lineage))
cells_f = pd.read_parquet(SSD / "segmentation_full/cells.parquet"); Xf = sparse.load_npz(SSD / "segmentation_full/cell_by_gene.npz").tocsr()
genes_f = pd.read_csv(SSD / "segmentation_full/genes.csv").gene.astype(str)
setA = json.load(open(SSD / "compare/purkinje/gene_sets.json"))["purkinje_A"]
fA = np.asarray(Xf[:, np.isin(genes_f, setA)].sum(1)).ravel() / np.maximum(np.asarray(Xf.sum(1)).ravel(), 1)
fA = dict(zip(cells_f.cell.astype(np.int64), fA)); NONNEURO = {"Astroglia", "Oligodendroglia", "Vascular", "Immune", "Fibroblast / meningeal"}
def keep_C(i): return not (lin.get(i) in NONNEURO and fA.get(i, 0) < 0.0164)
cl = pd.read_csv(SSD / "Cerebellum_sample/analysis/clustering/gene_expression_graphclust/clusters.csv")
c2g = {k: g for g, ks in GRP.items() for k in ks}; ntype = dict(zip(cl.Barcode, cl.Cluster.map(c2g)))
base = pd.read_parquet(SSD / "segmentation_full/base_cells.parquet")[["label", "nucleus_id"]]
lab_type = dict(zip(base.label, base.nucleus_id.map(ntype)))
lab_nuc = dict(zip(base.label, base.nucleus_id))
lab_full = zarr.open_array(str(SSD / "segmentation_full/final_labels.zarr"), mode="r"); SHAPE = lab_full.shape
for w in WINS:
    x0, y0, x1, y1 = json.load(open(RUNS / "crops_PF_500" / w / "stats.json"))["window"]; H, W = y1 - y0, x1 - x0
    b = SSD / "work" / ("local_bundle_heldout" if w.startswith("held") else "local_bundle")
    cb = ds.dataset(b / "cell_boundaries.parquet").to_table(filter=(pc.field("vertex_x") >= x0 * UM - 30) & (pc.field("vertex_x") < x1 * UM + 30)
         & (pc.field("vertex_y") >= y0 * UM - 30) & (pc.field("vertex_y") < y1 * UM + 30)).to_pandas()
    T = np.zeros((H, W), np.int64); tid = {}
    for k, (cid, g) in enumerate(cb.groupby("cell_id", sort=False), start=1):
        rr, cc = draw_poly(g.vertex_y.to_numpy() / UM - y0, g.vertex_x.to_numpy() / UM - x0, shape=(H, W)); T[rr, cc] = k; tid[cid] = k
    O = np.asarray(lab_full[y0:y1, x0:x1]).astype(np.int64)
    ids = np.unique(O[O > 0]); typ = {i: ("Purkinje" if i >= 3_500_000_000 else lab_type.get(i)) for i in ids}
    for v, keep in [(v, k) for v, k in KEEP.items() if v in os.environ.get('VARIANTS', 'ABC')]:
        sel = [i for i in ids if (keep_C(i) if keep is None else typ[i] in keep)]; ours_m = np.isin(O, sel)
        drop10x = [tid[lab_nuc[i]] for i in sel if lab_nuc.get(i) in tid]                 # 10x cells of the same nuclei
        Hh = np.where(np.isin(T, drop10x), 0, T) + 2_000_000_000                             # 10x ids offset
        Hh[np.isin(T, drop10x) | (T == 0)] = 0
        Hh[ours_m] = O[ours_m]
        u, cnt = np.unique(Hh[(Hh >= 2_000_000_000) & (Hh < 3_000_000_000)], return_counts=True)
        Hh[np.isin(Hh, u[cnt * UM ** 2 < 10])] = 0                                            # small 10x remnants
        od = RUNS / f"crops_HYB{v}_500" / w; od.mkdir(parents=True, exist_ok=True)
        fl = np.lib.format.open_memmap(od / "final_labels.npy", mode="w+", dtype=np.uint32, shape=SHAPE); fl[y0:y1, x0:x1] = Hh.astype(np.uint32); fl.flush()
        json.dump({"window": [x0, y0, x1, y1], "hybrid": v}, open(od / "stats.json", "w"))
    print(w, {v: (sum(keep_C(i) for i in ids) if k is None else sum(typ[i] in k for i in ids)) for v, k in KEEP.items()}, len(ids), flush=True)
