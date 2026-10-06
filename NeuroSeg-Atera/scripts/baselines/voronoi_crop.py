#!/usr/bin/env python3
"""Voronoi baseline on one Atera crop: paper_reseg_pipeline/scg/baselines/voronoi_fullsample.py restricted to a window.

Same algorithm and parameters as the whole-sample driver: seeds = vertex-mean centroids of every 10x nucleus polygon
of the slide (>= 3 vertex rows, common.load_nuclei), reflected across the 4 image edges, Voronoi (qhull), each region
clipped to the image box, regions < 1 px^2 dropped, no distance cap; a pixel / transcript goes to the seed nearest
to its pixel centre. Since the seeds are the whole slide's nuclei, the result inside the window is exactly the
whole-sample result there. Only change: the label image is built for the window only (nearest seed of each window
pixel centre = rasterizing the clipped polygons at pixel centres), and transcripts are read for the window only.

Usage: python voronoi_crop.py --crop spot1_purkinje_left
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atera_common as A  # noqa: E402

A.paper_path()
import common as C  # noqa: E402  (paper_reseg_pipeline/scg/baselines/common.py, unchanged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    t0 = time.time()
    win = A.crop_window(a.crop)
    x0, y0, x1, y1 = win
    out = a.out or A.WORK / "voronoi" / a.crop
    H, W = A.image_shape()
    ids, _, _, _, cents = C.load_nuclei(A.BUNDLE)
    print(f"{a.crop}: window {win}, image {W}x{H}, {len(ids)} nuclei ({time.time() - t0:.0f}s)", flush=True)

    # window pixel centres -> nearest seed (the whole-sample transcript rule applied to every window pixel)
    tree = cKDTree(cents)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    _, nn = tree.query(np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5]), workers=A.WORKERS)
    nn = nn.reshape(yy.shape)
    used = np.unique(nn)

    # the whole-sample "kept" rule (clipped region area >= 1 px^2), evaluated for the seeds that own window pixels
    rw, rh = W, H
    left = cents * [-1, 1]
    right = cents * [-1, 1] + [2 * rw, 0]
    top = cents * [1, -1]
    bottom = cents * [1, -1] + [0, 2 * rh]
    vor = Voronoi(np.vstack([cents, left, right, top, bottom]))
    bounds = shapely_box(0, 0, rw, rh)
    kept = np.zeros(len(ids), dtype=bool)
    area_px = np.zeros(len(ids))
    for i in used:
        region = vor.regions[vor.point_region[i]]
        if -1 in region or len(region) == 0:
            continue
        clipped = ShapelyPolygon(vor.vertices[region]).intersection(bounds)
        if clipped.is_empty or clipped.area < 1 or clipped.geom_type != "Polygon":
            continue
        kept[i] = True
        area_px[i] = clipped.area
    labels = np.where(kept[nn], nn + 1, 0).astype(np.uint32)
    n_dup = int(len(cents) - len(np.unique(np.round(cents, 6), axis=0)))
    print(f"voronoi: {len(used)} seeds own window pixels, {int(kept[used].sum())} kept ({time.time() - t0:.0f}s)",
          flush=True)
    info = {"tool": "voronoi", "crop": a.crop, "bundle": str(A.BUNDLE), "version": "scipy.spatial.Voronoi (qhull)",
            "reference_driver": "paper_reseg_pipeline/scg/baselines/voronoi_fullsample.py",
            "parameters": {"seeds": "vertex-mean centroid of each 10x nucleus polygon (>= 3 vertices), whole slide",
                           "reflection": "4 image edges", "clip": [0, 0, int(W), int(H)], "distance_cap": None,
                           "min_clipped_area_px2": 1, "assignment": "nearest seed of transcript pixel centre"},
            "n_nuclei": int(len(ids)), "n_duplicate_seeds": n_dup,
            "median_region_area_um2_window_cells": float(np.median(area_px[used][kept[used]]) * C.UM ** 2)}
    A.write_crop_outputs(out, win, labels, ids, info, t0)


if __name__ == "__main__":
    main()
