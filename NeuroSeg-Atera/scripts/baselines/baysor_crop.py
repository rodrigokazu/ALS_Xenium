#!/usr/bin/env python3
"""Baysor baseline on one Atera crop: paper_reseg_pipeline/scg/baselines/baysor/run_baysor_fullsample.py's Baysor call
(run_baysor, outer_ring, fallback_scale imported unchanged) on one window.

Settings unchanged: Baysor C++ cpp-0.8.3 (d7077a7), `baysor run -c xenium.toml --output-style legacy -o <out>
<transcripts.parquet> <nucleus_boundaries.parquet>` with the unmodified configs/xenium.toml, prior = the bundle's
nucleus_boundaries.parquet vertex rows within the window +- 5 um, molecules = is_gene & qv >= 20 with columns
transcript_id, feature_name, x/y/z_location, qv, cell_id, overlaps_nucleus; fallback --scale only on "Scale could
not be determined"; OMP_NUM_THREADS=1.
Crop change: one Baysor window = the crop +- 25 um (the context pad every method gets here) instead of the
whole-sample 1500 um tiles + 100 um margins (a 1700 um Atera tile would hold ~180 M molecules); every Baysor cell is
kept, as in the 108-crop benchmark runs (single window, no tile ownership needed). Polygons are rasterized on the
level-0 grid (pixel centre inside, later polygon wins) and cut to the crop window; transcripts take the label of
their pixel (= the whole-sample driver's pixel-centre point-in-polygon lookup).

Usage: python baysor_crop.py --crop spot1_purkinje_left --baysor <bin> --config <xenium.toml>
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pds
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atera_common as A  # noqa: E402

A.paper_path()
sys.path.insert(0, str(A.PAPER_BASELINES / "baysor"))
import common as C  # noqa: E402
import run_baysor_fullsample as RB  # noqa: E402  (unchanged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--baysor", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    win = A.crop_window(a.crop)
    x0, y0, x1, y1 = win
    out = a.out or A.WORK / "baysor" / a.crop
    tdir = out / "_tile"
    tdir.mkdir(parents=True, exist_ok=True)
    H, W = A.image_shape()
    wx0, wy0 = x0 * A.UM - A.PAD_UM, y0 * A.UM - A.PAD_UM
    wx1, wy1 = x1 * A.UM + A.PAD_UM, y1 * A.UM + A.PAD_UM
    if not (tdir / "nucleus_boundaries.parquet").exists():
        tx = A.read_tx(wx0, wy0, wx1, wy1, RB.TX_COLS, right_closed=True)
        tx["feature_name"] = tx["feature_name"].astype(str)
        tx.to_parquet(tdir / "transcripts.parquet")
        f = ((pc.field("vertex_x") >= wx0 - 5) & (pc.field("vertex_x") <= wx1 + 5)
             & (pc.field("vertex_y") >= wy0 - 5) & (pc.field("vertex_y") <= wy1 + 5))
        nb = pds.dataset(A.BUNDLE / "nucleus_boundaries.parquet").to_table(filter=f).to_pandas()
        nb.to_parquet(tdir / "nucleus_boundaries.parquet")
        n_tx, n_nuc = len(tx), int(nb.cell_id.nunique())
        del tx, nb
    else:
        n_tx = pds.dataset(tdir / "transcripts.parquet").count_rows()
        n_nuc = int(pd.read_parquet(tdir / "nucleus_boundaries.parquet", columns=["cell_id"]).cell_id.nunique())
    print(f"{a.crop}: Baysor window [{wx0:.2f},{wy0:.2f},{wx1:.2f},{wy1:.2f}] um, {n_tx:,} molecules, {n_nuc:,} "
          f"nuclei ({time.time() - t0:.0f}s)", flush=True)
    run = RB.run_baysor(tdir, a.baysor, a.config, a.threads)
    print(f"baysor: {run}", flush=True)
    if not run.get("ok"):
        json.dump({"tool": "baysor", "crop": a.crop, "ok": False, "run": run, "n_molecules": n_tx,
                   "runtime_s": round(time.time() - t0, 1)}, open(out / "run_info.json", "w"), indent=1)
        raise SystemExit("FAILED: Baysor did not produce segmentation_polygons_2d.json")

    gj = json.loads((tdir / "baysor_out" / "segmentation_polygons_2d.json").read_text())
    ids, polys, n_multi = [], [], 0
    for feat in gj["features"]:
        v, kind = RB.outer_ring(feat["geometry"])
        if v is None or len(v) < 3:
            continue
        n_multi += kind == "multipolygon"
        ids.append(str(feat["id"]))
        polys.append(v)
    labels = np.zeros((y1 - y0, x1 - x0), dtype=np.int64)
    areas = {}
    for k, v in enumerate(polys, start=1):
        areas[ids[k - 1]] = abs(float(Polygon(v).area))
        rz = C.rasterize_polygon(v / A.UM, H, W)
        if rz is None:
            continue
        r0, c0, inside = rz
        ra, ca = max(r0, y0), max(c0, x0)
        rb, cb = min(r0 + inside.shape[0], y1), min(c0 + inside.shape[1], x1)
        if rb <= ra or cb <= ca:
            continue
        m = inside[ra - r0:rb - r0, ca - c0:cb - c0]
        sub = labels[ra - y0:rb - y0, ca - x0:cb - x0]
        sub[m] = k                              # later polygon wins
    print(f"{len(ids):,} Baysor polygons ({n_multi} multipolygons) ({time.time() - t0:.0f}s)", flush=True)
    info = {"tool": "baysor", "crop": a.crop, "bundle": str(A.BUNDLE), "baysor_version": RB.BAYSOR_VERSION,
            "reference_driver": "paper_reseg_pipeline/scg/baselines/baysor/run_baysor_fullsample.py "
                                "(run_baysor / outer_ring imported)",
            "baysor_binary": a.baysor, "config_file": a.config, "config_toml": Path(a.config).read_text(),
            "baysor_window_um": [wx0, wy0, wx1, wy1], "n_molecules_input": int(n_tx), "n_nuclei_prior": n_nuc,
            "molecule_filter": f"is_gene & qv >= {A.QV_MIN}", "omp_threads": a.threads, "run": run,
            "n_polygons": len(ids), "n_multipolygons": int(n_multi),
            "max_rss_children_gb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)}
    A.write_crop_outputs(out, win, labels.astype(np.uint32), np.asarray(ids), info, t0)
    cells = pd.DataFrame({"cell_id": list(areas.keys()), "cell_area": list(areas.values())})
    cells.to_parquet(out / "cells_all_polygons.parquet", index=False)


if __name__ == "__main__":
    main()
