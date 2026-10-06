#!/usr/bin/env python3
"""New segmentation vs 10x on one Atera crop window: metrics on the same transcripts, plus arrays for panels.

Reads the crop run (run_fullsample.py --window output), the 10x bundle, and the QC transcripts inside the window
(10x assignment = transcripts.parquet cell_id; new assignment = label under each transcript). Cells are counted by
centroid inside the window so edge cells are not double counted between methods.

Output (--out): metrics.json, crop.npz (4 stains, new labels, window), tenx_cells.parquet (10x boundaries near
the window), tx_sub.parquet (random 300k QC transcripts with both assignments, for plotting).
"""
import argparse, json
from pathlib import Path
import sys
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, tifffile, zarr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_reseg_pipeline"))
from splitmerge.metrics import score, mecr_downsampled  # the paper benchmark's scoring, unchanged (MECR pairs, MECR_N, shell coherence)

UM = 0.2125
ap = argparse.ArgumentParser()
ap.add_argument("--bundle", type=Path, required=True)
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
st = json.load(open(a.run / "stats.json"))
x0, y0, x1, y1 = st["window"]
labels = np.load(a.run / "final_labels.npy", mmap_mode="r")
lab = np.asarray(labels[y0:y1, x0:x1])
new_cells = pd.read_parquet(a.run / "cells.parquet")

f = ((pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM)
     & (pc.field("y_location") < y1 * UM) & (pc.field("qv") >= 20) & (pc.field("is_gene") == True))
tx = ds.dataset(a.bundle / "transcripts.parquet").to_table(
    columns=["x_location", "y_location", "feature_name", "cell_id", "overlaps_nucleus"], filter=f).to_pandas()
tx["feature_name"] = tx.feature_name.astype(str)
tx["tenx"] = tx.cell_id.astype(str)
tx.loc[tx.tenx.isin(["UNASSIGNED", "-1", "", "None"]), "tenx"] = ""
col = np.clip(np.floor(tx.x_location.to_numpy() / UM).astype(int) - x0, 0, lab.shape[1] - 1)
row = np.clip(np.floor(tx.y_location.to_numpy() / UM).astype(int) - y0, 0, lab.shape[0] - 1)
tx["new"] = lab[row, col]

tenx = pd.read_parquet(a.bundle / "cells.parquet", columns=["cell_id", "x_centroid", "y_centroid", "cell_area", "nucleus_area"])
inwin = lambda d, xc, yc: d[d[xc].between(x0 * UM, x1 * UM) & d[yc].between(y0 * UM, y1 * UM)]
tenx_in = inwin(tenx, "x_centroid", "y_centroid")
new_in = inwin(new_cells, "x_centroid", "y_centroid")

def per_cell(key, ids):
    n = tx[tx[key].isin(ids)].groupby(key).size().reindex(ids, fill_value=0)
    return n
n_tenx = per_cell("tenx", list(tenx_in.cell_id.astype(str)))
n_new = per_cell("new", list(new_in.label))
m = {"window_px": [x0, y0, x1, y1], "n_qc_transcripts": int(len(tx)),
     "tenx": {"n_cells": int(len(tenx_in)), "frac_tx_assigned": float((tx.tenx != "").mean()),
              "median_tx_per_cell": float(n_tenx.median()), "median_area_um2": float(tenx_in.cell_area.median()),
              "frac_cells_with_nucleus": float((tenx_in.nucleus_area.fillna(0) > 0).mean())},
     "new": {"n_cells": int(len(new_in)), "frac_tx_assigned": float((tx.new > 0).mean()),
             "median_tx_per_cell": float(n_new.median()), "median_area_um2": float(new_in.area_um2.median()),
             "frac_cells_with_nucleus": float(new_in.has_nucleus.mean()),
             "source_counts": new_in.source.value_counts().to_dict()},
     "agreement": {"frac_tx_both_assigned": float(((tx.tenx != "") & (tx.new > 0)).mean()),
                   "frac_tx_only_tenx": float(((tx.tenx != "") & (tx.new == 0)).mean()),
                   "frac_tx_only_new": float(((tx.tenx == "") & (tx.new > 0)).mean())}}
# benchmark metrics on the same QC transcripts (10x labels as integer codes, 0 = unassigned)
genes = tx.feature_name.to_numpy()
on_nuc = tx.overlaps_nucleus.astype(bool).to_numpy()
xy = tx[["x_location", "y_location"]].to_numpy()
tenx_codes = pd.factorize(tx.tenx.where(tx.tenx != "", None))[0] + 1  # NaN -> -1 -> 0
m["benchmark"] = {"tenx": score("10x", tenx_codes.astype(np.int64), genes, on_nuc, xy),
                  "new": score("new", tx.new.to_numpy().astype(np.int64), genes, on_nuc, xy)}
# WTA depth: at 20 transcripts per cell two marker genes almost never co-occur (mecr_n20 ~ 0 for every method), so
# the same coverage-controlled MECR is also reported with cells cut to 1000 transcripts (cells with fewer left out)
for k, lab_k in (("tenx", tenx_codes), ("new", tx.new.to_numpy())):
    lab_k = np.asarray(lab_k, dtype=np.int64)
    m["benchmark"][k]["mecr_n1000"] = round(mecr_downsampled(lab_k, genes, n=1000), 4)
    m["benchmark"][k]["n_cells_ge1000_tx"] = int((pd.Series(lab_k[lab_k > 0]).value_counts() >= 1000).sum())
json.dump(m, open(a.out / "metrics.json", "w"), indent=1)
print(json.dumps(m, indent=1))

tf = tifffile.TiffFile(a.bundle / "morphology_focus" / "ch0000_dapi.ome.tif")
z = zarr.open(tf.aszarr(level=0, series=0), mode="r")
np.savez_compressed(a.out / "crop.npz", img=np.asarray(z[:, y0:y1, x0:x1]), labels=lab, window=np.array([x0, y0, x1, y1]))
cb = pd.read_parquet(a.bundle / "cell_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
near = cb.vertex_x.between(x0 * UM - 20, x1 * UM + 20) & cb.vertex_y.between(y0 * UM - 20, y1 * UM + 20)
cb[cb.cell_id.isin(cb.cell_id[near].unique())].to_parquet(a.out / "tenx_cells.parquet", index=False)
tx.sample(min(len(tx), 300_000), random_state=0).drop(columns=["cell_id"]).to_parquet(a.out / "tx_sub.parquet", index=False)
