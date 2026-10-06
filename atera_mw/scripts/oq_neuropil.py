#!/usr/bin/env python3
"""Unassigned (neuropil) transcripts per 25 um bin and gene, over the whole section, for the dendritic mRNA atlas.

Per 4096 px tile: QC transcripts (qv >= 20, genes only) with no cell under them in final_labels.zarr, counted per
(25 um bin, gene) for the informative genes plus the Purkinje genes (about 3,400 genes). Output: neuropil_bin_gene.parquet.
python oq_neuropil.py [--workers 6]
"""
import argparse, json, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, zarr
sys.path.insert(0, str(Path(__file__).parent))
import full_post as FP

B_UM = 25.0; SSD = FP.SSD; O = SSD / "segmentation_full/open_questions"

def genes():
    de = pd.read_csv(FP.BUNDLE / "analysis/diffexp/gene_expression_graphclust/differential_expression.csv")
    M = np.column_stack([de[f"Cluster {k} Mean Counts"] for k in range(1, 37)]); Lf = np.column_stack([de[f"Cluster {k} Log2 fold change"] for k in range(1, 37)])
    Pv = np.column_stack([de[f"Cluster {k} Adjusted p value"] for k in range(1, 37)])
    inf = set(de["Feature Name"][((Lf >= 2.0) & (M >= 0.3) & (Pv < 0.01)).any(1)])
    gs = json.load(open(SSD / "compare/purkinje/gene_sets.json"))
    return sorted(inf | set(gs["purkinje_A"]) | set(gs["purkinje_B"]) | {"GJD2", "GJA1", "GJC2", "GJB1", "GJB6", "ALDOC", "PLCB4"})

def job(core):
    x0, y0, x1, y1 = core; G = pd.Index(genes())
    lab = np.asarray(zarr.open_array(str(FP.STORES["pcr"]), mode="r")[y0:y1, x0:x1])
    tx = FP.read_tx(x0, y0, x1, y1, ["x_location", "y_location", "feature_name"], gene_only=True)
    if not len(tx): return None
    r = np.clip((tx.y_location.to_numpy() / FP.UM).astype(int) - y0, 0, y1 - y0 - 1); c = np.clip((tx.x_location.to_numpy() / FP.UM).astype(int) - x0, 0, x1 - x0 - 1)
    gi = G.get_indexer(tx.feature_name.astype(str)); m = (lab[r, c] == 0) & (gi >= 0)
    bx = (tx.x_location.to_numpy()[m] // B_UM).astype(np.int64); by = (tx.y_location.to_numpy()[m] // B_UM).astype(np.int64)
    key, n = np.unique((by * 4000 + bx) * 4096 + gi[m], return_counts=True)
    return pd.DataFrame({"by": key // 4096 // 4000, "bx": key // 4096 % 4000, "gene": G.to_numpy()[key % 4096], "n": n})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args(); O.mkdir(parents=True, exist_ok=True)
    out = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(job, FP.tiles()), start=1):
            if r is not None: out.append(r)
            if i % 20 == 0: print("tile", i, flush=True)
    df = pd.concat(out).groupby(["by", "bx", "gene"]).n.sum().reset_index(); df.to_parquet(O / "neuropil_bin_gene.parquet", index=False)
    print("done", len(df), "entries", int(df.n.sum()), "unassigned transcripts")
