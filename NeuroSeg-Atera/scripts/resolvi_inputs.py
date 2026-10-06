#!/usr/bin/env python3
"""Inputs for the resolVI comparison: one AnnData per segmentation (10x, ours = final full-section labels) over the nine 500 um
benchmark windows. Cells with >= 20 QC transcripts; genes = informative genes (10x cluster DE, as phase2) + MECR markers +
Purkinje / granule / glia gene sets. obsm["X_spatial"] = transcript centroid (um); obs["window"] = batch.
python resolvi_inputs.py <out dir>"""
import sys, json, os
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc, zarr, anndata as ad
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_reseg_pipeline"))
from splitmerge.metrics import MARKERS
UM = 0.2125; SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RUNS = Path("/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local")
WINS = ["spot1_purkinje_left", "spot2_gl_wm_centre", "spot3_purkinje_right"] + [f"held{i}" for i in range(1, 7)]
WINFILE = sys.argv[2] if len(sys.argv) > 2 else None          # optional "name x0 y0 x1 y1" list; read from the full-section bundle
WIN_XY = {l.split()[0]: tuple(map(int, l.split()[1:])) for l in open(WINFILE)} if WINFILE else {}
if WINFILE: WINS = list(WIN_XY)
de = pd.read_csv(SSD / "Cerebellum_sample/analysis/diffexp/gene_expression_graphclust/differential_expression.csv")
M = np.column_stack([de[f"Cluster {k} Mean Counts"] for k in range(1, 37)]); Lf = np.column_stack([de[f"Cluster {k} Log2 fold change"] for k in range(1, 37)])
Pv = np.column_stack([de[f"Cluster {k} Adjusted p value"] for k in range(1, 37)])
gs = json.load(open(SSD / "compare/purkinje/gene_sets.json"))
G = sorted(set(de["Feature Name"][((Lf >= 2) & (M >= 0.3) & (Pv < 0.01)).any(1)]) | {g for v in MARKERS.values() for g in v} | set(sum(gs.values(), [])))
G = [g for g in G if g in set(de["Feature Name"])]; Gi = pd.Index(G); print(len(G), "genes")
lab_full = zarr.open_array(os.environ.get("LABELS", str(SSD / "segmentation_full/final_labels.zarr")), mode="r")
acc = {"tenx": [], "ours": []}
for w in WINS:
    if w in WIN_XY: x0, y0, x1, y1 = WIN_XY[w]; b = SSD / "Cerebellum_sample"
    else:
        x0, y0, x1, y1 = json.load(open(RUNS / "crops_PF_500" / w / "stats.json"))["window"]
        b = SSD / "work" / ("local_bundle_heldout" if w.startswith("held") else "local_bundle")
    tx = ds.dataset(b / "transcripts.parquet").to_table(columns=["x_location", "y_location", "feature_name", "cell_id"], filter=(pc.field("x_location") >= x0 * UM)
         & (pc.field("x_location") < x1 * UM) & (pc.field("y_location") >= y0 * UM) & (pc.field("y_location") < y1 * UM) & (pc.field("qv") >= 20) & pc.field("is_gene")).to_pandas()
    L = np.asarray(lab_full[y0:y1, x0:x1])
    r = np.clip((tx.y_location / UM).astype(int) - y0, 0, y1 - y0 - 1); c = np.clip((tx.x_location / UM).astype(int) - x0, 0, x1 - x0 - 1)
    tx["ours"] = L[r, c].astype(np.int64); tx["tenx"] = tx.cell_id.astype(str).where(~tx.cell_id.astype(str).isin(["", "UNASSIGNED"]), "0")
    gi = Gi.get_indexer(tx.feature_name.astype(str))
    for k in ("tenx", "ours"):
        t = tx[tx[k].astype(str) != "0"]; cid, ci = np.unique(t[k].astype(str), return_inverse=True)
        n = np.bincount(ci); keep = n >= 20
        m = gi[t.index.get_indexer(t.index)] if False else Gi.get_indexer(t.feature_name.astype(str))
        ok = m >= 0
        X = sparse.csr_matrix((np.ones(ok.sum(), np.float32), (ci[ok], m[ok])), shape=(len(cid), len(G)))
        xy = np.c_[np.bincount(ci, t.x_location) / n, np.bincount(ci, t.y_location) / n]
        obs = pd.DataFrame({"cell": cid, "window": w, "n_tx_all": n}, index=[f"{w}|{x}" for x in cid])
        acc[k].append(ad.AnnData(X=X[keep], obs=obs[keep], var=pd.DataFrame(index=G), obsm={"X_spatial": xy[keep]}))
    print(w, {k: acc[k][-1].n_obs for k in acc}, flush=True)
for k, v in acc.items():
    a = ad.concat(v); a.layers["counts"] = a.X.copy(); a.write_h5ad(OUT / f"{k}{os.environ.get('SUFFIX', '')}.h5ad"); print(k, a.shape)
