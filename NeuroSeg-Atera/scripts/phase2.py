#!/usr/bin/env python3
"""Phase 2 (first version): transcript-identity + density assignment of the pixels around existing cells.

Starts from a segmentation (default option_C500 labels) and lets each cell take free pixels around it when
  - density: the pixel is in tissue (smoothed transcript density > tlow) and reached from the cell by a flood that follows
    high density first (watershed on -density), within r_um of the cell (r_pc_um for Purkinje / interneuron cells);
  - identity: the local transcript composition fits the cell's type at least as well as any other type, within delta.
Identity: genes are filtered to informative ones (log2FC >= lfc, mean >= mean_min, adj p < 0.01 in some 10x graph cluster); the 36
clusters are merged into 6 types; a type's gene profile = mean of its clusters' mean counts (cell-count weighted). Every informative
transcript adds log p_type(gene) to its pixel bin; bins are smoothed (sigma_um); a cell's type is the best-scoring type over its
own transcripts. Output: new labels, written like a run_fullsample run so compare_crop.py scores it unchanged.
python phase2.py <spot> [overrides as key=value ...]   (base run: _work/.../crops_C500/<spot>)"""
import sys, json, shutil
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, zarr
from scipy import ndimage as ndi
from skimage.segmentation import watershed

UM = 0.2125
SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB")
W = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
P = dict(lfc=2.0, mean_min=0.3, tlow=28.0, r_um=6.0, r_pc_um=14.0, sigma_um=2.0, delta=2.0, bin_px=4, min_tx=30, base="C500", out="P2", exclude="", bundle="local_bundle")
spot = sys.argv[1]
for kv in sys.argv[2:]:
    k, v = kv.split("="); P[k] = type(P[k])(v) if not isinstance(P[k], str) else v
GROUPS = {"granule": [1, 2, 3, 4, 5, 8, 9, 10, 15, 6, 7, 11], "PC/interneuron": [14, 23, 33], "astro": [13, 16, 19],
          "oligo": [12, 18, 27, 29, 30, 32], "vascular": [17, 20, 21, 22, 26, 28], "immune": [24, 25, 31, 34, 35, 36]}
TYPES = list(GROUPS)

B = SSD / "work" / P["bundle"]
# ---- informative genes and type profiles from 10x's cluster table
de = pd.read_csv(SSD / "Cerebellum_sample/analysis/diffexp/gene_expression_graphclust/differential_expression.csv")
cl = pd.read_csv(SSD / "Cerebellum_sample/analysis/clustering/gene_expression_graphclust/clusters.csv")
ncl = cl.Cluster.value_counts().sort_index().reindex(range(1, 37)).to_numpy().astype(float)
M = np.column_stack([de[f"Cluster {k} Mean Counts"] for k in range(1, 37)]); Lf = np.column_stack([de[f"Cluster {k} Log2 fold change"] for k in range(1, 37)])
Pv = np.column_stack([de[f"Cluster {k} Adjusted p value"] for k in range(1, 37)])
inf = ((Lf >= P["lfc"]) & (M >= P["mean_min"]) & (Pv < 0.01)).any(1)
# held-out control: genes used by the scoring (MECR markers, Purkinje blob markers) can be removed from the identity step
inf &= ~np.isin(de["Feature Name"].to_numpy(), [g for g in P["exclude"].split(",") if g])
names = de["Feature Name"].to_numpy()[inf]
Mi = M[inf]
prof = np.zeros((len(TYPES), inf.sum()))
for t, ks in enumerate(GROUPS.values()):
    w = ncl[np.array(ks) - 1]; prof[t] = (Mi[:, np.array(ks) - 1] * w).sum(1) / w.sum()
prof = (prof + 1e-4); prof /= prof.sum(1, keepdims=True); logp = np.log(prof)          # (T, G_inf)
gidx = pd.Index(names)

# ---- window data
base = Path(W / f"crops_{P['base']}" / spot); st = json.load(open(base / "stats.json")); x0, y0, x1, y1 = st["window"]
H, Wd = y1 - y0, x1 - x0
lab0 = np.load(base / "final_labels.npy", mmap_mode="r")[y0:y1, x0:x1].astype(np.int64)
tx = ds.dataset(B / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name", "qv"],
    filter=(pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM) & (pc.field("qv") >= 20)).to_pandas()
gi = gidx.get_indexer(tx.feature_name.astype(str)); ok = gi >= 0
col = np.clip((tx.x_location.to_numpy() / UM).astype(int) - x0, 0, Wd - 1); row = np.clip((tx.y_location.to_numpy() / UM).astype(int) - y0, 0, H - 1)
col, row, gi = col[ok], row[ok], gi[ok]
g = P["bin_px"]; nby, nbx = H // g + 1, Wd // g + 1; flat = (row // g) * nbx + (col // g)
S = np.stack([np.bincount(flat, weights=logp[t][gi], minlength=nby * nbx).reshape(nby, nbx) for t in range(len(TYPES))])    # summed log p per bin
cnt = np.bincount(flat, minlength=nby * nbx).reshape(nby, nbx).astype(float)
sg = P["sigma_um"] / (g * UM)
Ss = np.stack([ndi.gaussian_filter(S[t], sg) for t in range(len(TYPES))]); cs = ndi.gaussian_filter(cnt, sg)

# ---- cell types from their own informative transcripts
lab_at = lab0[row, col]; nlab = int(lab0.max()) + 1
cell_ll = np.stack([np.bincount(lab_at, weights=logp[t][gi], minlength=nlab) for t in range(len(TYPES))])
cell_n = np.bincount(lab_at, minlength=nlab)
ctype = np.where(cell_n >= P["min_tx"], cell_ll.argmax(0), -1); ctype[0] = -1

# ---- territory: flood from the cells along high density, inside tissue, within r of a cell
z = zarr.open_array(str(SSD / "work/density/tx_density.zarr"), mode="r")
dens = ndi.gaussian_filter(np.asarray(z[y0:y1, x0:x1]).astype(np.float32), 1.0 / UM)
dist = ndi.distance_transform_edt(lab0 == 0) * UM
rmax = max(P["r_um"], P["r_pc_um"])
mask = (lab0 > 0) | ((dist <= rmax) & (dens > P["tlow"]))
ws = watershed(-dens, markers=lab0.astype(np.int32), mask=mask)
new = (lab0 == 0) & (ws > 0)
owner = np.where(new, ws, 0)
tp = ctype[owner]
limit = np.where(tp == TYPES.index("PC/interneuron"), P["r_pc_um"], P["r_um"])
new &= (tp >= 0) & (dist <= limit)
# ---- identity: the owner's type must fit the local composition within delta of the best type
# identity test on a smooth field: per-type margin (type score minus the best type's) is computed on the coarse bins and
# interpolated bilinearly to the pixel grid, so cell edges do not inherit the 0.85 um bin staircase
marg = (Ss - Ss.max(0, keepdims=True)).astype(np.float32)                                   # (T, nby, nbx), <= 0
up = np.stack([ndi.zoom(marg[t], g, order=1)[:H, :Wd] for t in range(len(TYPES))])
csu = ndi.zoom(cs.astype(np.float32), g, order=1)[:H, :Wd]
if up.shape[1:] != (H, Wd):                                                                  # pad if the zoom came out short
    pad = ((0, 0), (0, H - up.shape[1]), (0, Wd - up.shape[2])); up = np.pad(up, pad, mode="edge"); csu = np.pad(csu, pad[1:], mode="edge")
rr_, cc_ = np.indices((H, Wd))
m_owner = up[np.clip(tp, 0, None), rr_, cc_]
fits = (m_owner >= -P["delta"]) | (csu < 0.5)       # no informative transcripts nearby: density alone decides
new &= fits
lab1 = np.where(new, owner, lab0)
# keep what is connected to the cell it was added to; fill holes
out = lab1.copy()
for L, sl in enumerate(ndi.find_objects(lab1.astype(np.int32)), start=1):
    if sl is None or L >= nlab: continue
    m = lab1[sl] == L; cc, _ = ndi.label(m); keep = np.unique(cc[(lab0[sl] == L) & m]); keep = keep[keep > 0]
    mm = ndi.binary_fill_holes(np.isin(cc, keep)) & ((lab0[sl] == 0) | (lab0[sl] == L))
    out[sl][(lab1[sl] == L) & ~mm] = 0
    out[sl][mm & (out[sl] == 0)] = L
# ---- write like a run_fullsample run
od = W / f"crops_{P['out']}_500" / spot; od.mkdir(parents=True, exist_ok=True)
fl = np.lib.format.open_memmap(od / "final_labels.npy", mode="w+", dtype=np.uint32, shape=np.load(base / "final_labels.npy", mmap_mode="r").shape)
fl[y0:y1, x0:x1] = out.astype(np.uint32); fl.flush()
cells = pd.read_parquet(base / "cells.parquet"); a = np.bincount(out.ravel(), minlength=nlab) * UM ** 2
cells["area_um2"] = a[cells.label.to_numpy()].round(2); cells.to_parquet(od / "cells.parquet", index=False)
shutil.copy(base / "rescued_nuclei.parquet", od / "rescued_nuclei.parquet"); json.dump({**st, "phase2": P}, open(od / "stats.json", "w"))
print(f"{spot}: {len(names):,} informative genes | {int((cell_n >= P['min_tx']).sum()):,} typed cells {dict(zip(TYPES, np.bincount(ctype[ctype >= 0], minlength=len(TYPES)).tolist()))}")
print(f"  candidate pixels {(lab0 == 0).sum() - 0:,} free; territory {int(((lab0 == 0) & (ws > 0)).sum()):,}; added {int(new.sum()):,} px = {new.sum() * UM ** 2:,.0f} um2 "
      f"({100 * new.sum() / max((lab0 > 0).sum(), 1):.0f}% more cell area); rejected by identity {int((((lab0==0)&(ws>0)&(tp>=0)&(dist<=limit))&~fits).sum()):,} px")
