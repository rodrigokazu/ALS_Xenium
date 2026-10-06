#!/usr/bin/env python3
"""Is the molecular-layer neuropil of KCNC3 / CACNA1G Purkinje-derived or from interneurons / granule cells?
Nine 500 um windows. Unassigned QC transcripts in molecular-layer bins; per gene, share lying within 4 um of an interneuron (MLI)
centroid and within 4 um of a granule centroid, divided by the same share for all unassigned molecular-layer transcripts
(enrichment 1 = no preference). Interneuron genes (KIT, SORCS3, LYPD6) and granule genes (GABRA6, CBLN3) are the positive
references, known Purkinje dendritic mRNAs (PCP2, RGS8) the Purkinje reference.  python oq_mli_check.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, zarr
from scipy.spatial import cKDTree
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); O = SSD / "segmentation_full"; Q = O / "open_questions"
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
G = ["KCNC3", "CACNA1G", "GRID2IP", "DAGLA", "ARHGEF33", "PCP2", "RGS8", "ITPR1", "KIT", "SORCS3", "LYPD6", "GABRA6", "CBLN3"]
layer = np.load(Q / "layer_map.npy"); lab = zarr.open_array(str(O / "final_labels.zarr"), mode="r")
ann = pd.read_parquet(O / "annotation/cells_annotated.parquet")
tm = cKDTree(ann.loc[ann.cell_type == "MLI (basket / stellate)", ["x_um", "y_um"]].to_numpy()); tg = cKDTree(ann.loc[ann.cell_type == "Granule", ["x_um", "y_um"]].to_numpy())
rows = []
for w in ["spot1_purkinje_left", "spot2_gl_wm_centre", "spot3_purkinje_right"] + [f"held{i}" for i in range(1, 7)]:
    x0, y0, x1, y1 = json.load(open(RUNS / "crops_PF_500" / w / "stats.json"))["window"]
    b = SSD / "work" / ("local_bundle_heldout" if w.startswith("held") else "local_bundle")
    t = ds.dataset(b / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name"], filter=(pc.field("x_location") >= x0 * UM) & (pc.field("x_location") < x1 * UM)
        & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM) & (pc.field("qv") >= 20) & pc.field("is_gene")).to_pandas()
    L = np.asarray(lab[y0:y1, x0:x1]); r = np.clip((t.y_location / UM).astype(int) - y0, 0, y1 - y0 - 1); c = np.clip((t.x_location / UM).astype(int) - x0, 0, x1 - x0 - 1)
    by, bx = (t.y_location // 25).astype(int).clip(0, layer.shape[0] - 1), (t.x_location // 25).astype(int).clip(0, layer.shape[1] - 1)
    m = (L[r, c] == 0) & (layer[by, bx] == "molecular layer"); t = t[m]
    if not len(t): continue
    xy = t[["x_location", "y_location"]].to_numpy(); t["near_mli"] = tm.query(xy)[0] <= 4; t["near_gran"] = tg.query(xy)[0] <= 4
    rows.append(t[["feature_name", "near_mli", "near_gran"]])
d = pd.concat(rows); base_m, base_g = d.near_mli.mean(), d.near_gran.mean()
out = d[d.feature_name.isin(G)].groupby("feature_name").agg(n=("near_mli", "size"), mli=("near_mli", "mean"), gran=("near_gran", "mean"))
out["enrich_near_MLI"], out["enrich_near_granule"] = out.mli / base_m, out.gran / base_g
out = out.reindex([g for g in G if g in out.index]); out.to_csv(Q / "neuropil_MLI_granule_proximity.csv")
print(f"all molecular-layer neuropil: {len(d):,} transcripts, {100 * base_m:.1f}% within 4 um of an MLI centroid, {100 * base_g:.1f}% of a granule centroid")
print(out.round(3).to_string())
