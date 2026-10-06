#!/usr/bin/env python3
"""Segger crop run -> the crop outputs: the unchanged driver's transcripts.parquet (segger loader rules already
applied: no -nx ids, >= 5 transcripts, non-degenerate hull) restricted to the scoring transcripts
(assignment.parquet), and its loader polygon (convex hull of each cell's transcripts) rasterized at pixel centres,
later cell wins, cut to the crop window (labels_window.npy).

Usage: python segger_post.py --crop spot1_purkinje_left
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atera_common as A  # noqa: E402

A.paper_path()
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True)
    a = ap.parse_args()
    t0 = time.time()
    win = A.crop_window(a.crop)
    x0, y0, x1, y1 = win
    d = A.WORK / "segger" / a.crop
    H, W = A.image_shape()
    t = pd.read_parquet(d / "transcripts.parquet")
    mb = pd.read_parquet(A.WORK / "_minibundle" / a.crop / "transcripts.parquet",
                         columns=["transcript_id", "x_location", "y_location"])
    t = t.merge(mb, on="transcript_id", how="left")
    tx = A.scoring_tx(win)
    assign = t.set_index("transcript_id")["cell_id"].reindex(tx.transcript_id.to_numpy()).fillna("UNASSIGNED")
    ids = np.array(sorted(set(t.cell_id) - {"UNASSIGNED"}))
    code = {c: i + 1 for i, c in enumerate(ids)}
    tx_codes = np.array([code.get(c, 0) for c in assign.to_numpy()], dtype=np.int64)
    labels = np.zeros((y1 - y0, x1 - x0), dtype=np.int64)
    for cid, g in t[t.cell_id != "UNASSIGNED"].groupby("cell_id", sort=True):
        pts = g[["x_location", "y_location"]].to_numpy(np.float64) / A.UM
        try:
            v = pts[ConvexHull(pts).vertices]
        except Exception:
            continue
        rz = C.rasterize_polygon(v, H, W)
        if rz is None:
            continue
        r0, c0, inside = rz
        ra, ca = max(r0, y0), max(c0, x0)
        rb, cb = min(r0 + inside.shape[0], y1), min(c0 + inside.shape[1], x1)
        if rb <= ra or cb <= ca:
            continue
        m = inside[ra - r0:rb - r0, ca - c0:cb - c0]
        labels[ra - y0:rb - y0, ca - x0:cb - x0][m] = code[cid]
    info = json.load(open(d / "run_info.json"))
    import shutil
    shutil.copy(d / "run_info.json", d / "run_info_segger_driver.json")
    shutil.copy(d / "cells.parquet", d / "cells_segger_driver.parquet")
    info["segger_driver_runtime_s"] = info.pop("runtime_s", None)
    info["crop_post"] = {"assignment": "segger's own per-transcript assignment (driver loader rules)",
                         "labels_window": "convex hull of each cell's transcripts, pixel-centre rasterized, later wins "
                                          "(display only; scoring uses the per-transcript assignment)"}
    A.write_crop_outputs(d, win, labels.astype(np.uint32), ids, info, t0, tx_codes=tx_codes, tx=tx)


if __name__ == "__main__":
    main()
