#!/usr/bin/env python3
"""Turn a run_fullsample.py result into a Xenium-Explorer-readable bundle (in --work).

Every final cell becomes a polygon traced from the label image; a cell's nucleus is its Xenium nucleus polygon
(same cell_id) or its rescued-nucleus polygon, and cells without a nucleus get none, so Explorer shows exactly
the masks and the nucleus assignment of the method. Cluster labels are inherited from the Xenium cell whose
nucleus a cell owns. --work must already hold copies of experiment.xenium, transcripts.parquet and
cell_feature_matrix/features.tsv.gz (the sbatch stages them; for a local test --stage copies them).

Usage:
  python scg/export_xenium.py --run-dir <run_fullsample out> --bundle <xenium dir> --work <dir> [--stage]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage import measure

# splitmerge/rules.py is used unchanged from the paper pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_reseg_pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from splitmerge import rules as R  # noqa: E402
import xenium_writer as XW  # noqa: E402

BAND = 2048  # label rows scanned at a time when locating cells


def locate(labels: np.ndarray, n: int, window) -> list:
    """Bounding boxes of labels 1..n, scanning the memory-mapped image in row bands."""
    x0, y0, x1, y1 = window
    boxes = [None] * (n + 1)
    for r in range(y0, y1, BAND):
        band = np.asarray(labels[r:min(y1, r + BAND), x0:x1])
        for lab, sl in enumerate(ndi.find_objects(band), start=1):
            if sl is None:
                continue
            b = (sl[0].start + r, sl[0].stop + r, sl[1].start + x0, sl[1].stop + x0)
            o = boxes[lab]
            boxes[lab] = b if o is None else (min(o[0], b[0]), max(o[1], b[1]), min(o[2], b[2]), max(o[3], b[3]))
    return boxes


def trace(mask: np.ndarray, r0: int, c0: int, um: float):
    cs = measure.find_contours(np.pad(mask, 1).astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    poly = Polygon(np.column_stack([(c[:, 1] - 1 + c0 + 0.5) * um, (c[:, 0] - 1 + r0 + 0.5) * um]))
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly if poly.area > 0 else None


def polys_by_id(df: pd.DataFrame, ids) -> dict:
    sub = df[df.cell_id.isin(set(ids))]
    out = {}
    for cid, g in sub.groupby("cell_id", sort=False):
        if len(g) >= 3:
            p = Polygon(np.column_stack([g.vertex_x.to_numpy(), g.vertex_y.to_numpy()]))
            out[cid] = p if p.is_valid else p.buffer(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--stage", action="store_true", help="copy the files the writer rewrites into --work first")
    a = ap.parse_args()
    t0 = time.time()
    um = R.UM_PER_PX
    run = json.load(open(a.run_dir / "stats.json"))
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / "cell_feature_matrix").mkdir(exist_ok=True)
    if a.stage:
        for f in ("experiment.xenium", "transcripts.parquet"):
            shutil.copy(a.bundle / f, a.work / f)
        shutil.copy(a.bundle / "cell_feature_matrix" / "features.tsv.gz", a.work / "cell_feature_matrix" / "features.tsv.gz")

    labels = np.load(a.run_dir / "final_labels.npy", mmap_mode="r")
    cells = pd.read_parquet(a.run_dir / "cells.parquet").set_index("label").sort_index()
    N = int(cells.index.max())
    assert list(cells.index) == list(range(1, N + 1)), "labels must be 1..N"
    boxes = locate(labels, N, run["window"])
    print(f"{N} cells located ({time.time() - t0:.0f}s)", flush=True)

    cell_polys = {}
    for lab in range(1, N + 1):
        b = boxes[lab]
        if b is None:
            continue
        r0, r1, c0, c1 = b
        p = trace(np.asarray(labels[r0:r1, c0:c1]) == lab, r0, c0, um)
        if p is not None:
            cell_polys[lab] = p
    print(f"{len(cell_polys)} cell polygons ({time.time() - t0:.0f}s)", flush=True)

    nb = pd.read_parquet(a.bundle / "nucleus_boundaries.parquet", columns=["cell_id", "vertex_x", "vertex_y"])
    nb["cell_id"] = nb["cell_id"].astype(str)
    rescued = pd.read_parquet(a.run_dir / "rescued_nuclei.parquet")
    with_nuc = cells[cells.has_nucleus]
    xen = polys_by_id(nb, with_nuc.nucleus_id)
    res = polys_by_id(rescued, with_nuc.nucleus_id)
    nuc_polys = {lab: (xen.get(nid) or res.get(nid)) for lab, nid in with_nuc.nucleus_id.items()}
    nuc_polys = {k: v for k, v in nuc_polys.items() if v is not None and k in cell_polys}

    bk_cells = pd.read_parquet(a.bundle / "cells.parquet", columns=["cell_id"])
    row_of = pd.Series(np.arange(len(bk_cells)), index=bk_cells.cell_id.astype(str))
    backup_index = cells.nucleus_id.map(row_of).fillna(-1).astype(np.int64).to_numpy()

    XW.write_bundle(a.work, a.bundle, labels, um, cell_polys, nuc_polys, backup_index)
    summary = {"n_cells": N, "n_cell_polygons": len(cell_polys), "n_with_nucleus": len(nuc_polys),
               "n_rescued_nuclei_used": int(sum(1 for nid in with_nuc.nucleus_id if nid.startswith("dapi_rescue_"))),
               "runtime_s": round(time.time() - t0)}
    json.dump(summary, open(a.work / "export_summary.json", "w"), indent=1)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
