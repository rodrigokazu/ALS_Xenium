#!/usr/bin/env python3
"""Watershed baseline on one Atera crop: the tile function of paper_reseg_pipeline/scg/baselines/watershed_fullsample.py
(run_tile, imported unchanged) on the crop's tile(s), with the whole-sample mask threshold.

As in the whole-sample run: markers = every vertex pixel of each 10x nucleus polygon (whole slide, later nucleus
wins) dilated with disk(3); density = 2D histogram of the QC transcripts (is_gene, qv >= 20) on the level-0 grid,
gaussian_filter(sigma=6 px); skimage watershed(-density, markers, mask=density > thr) with thr = the WHOLE-SAMPLE
5th percentile (stats_watershed_threshold.py); tile = core + 800 px margin; a tile keeps the cells whose pixel
centroid lies in its core; loader = labels_to_polygons (largest contour, <= 40 vertices, < 15 px dropped) ->
pixel-centre rasterization, later (higher) label wins. cell_id = the seeding nucleus' 10x cell_id.
Crop changes only: the cores are the crop window +- 25 um (4096-px tiles over it: one tile), transcripts are read
for the tiles' windows only, and the label image is kept for the tile windows only, then cut to the crop window.

Usage: python watershed_crop.py --crop spot1_purkinje_left [--thr-json <_stats/watershed_threshold.json>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atera_common as A  # noqa: E402

A.paper_path()
import common as C  # noqa: E402
import watershed_fullsample as WS  # noqa: E402  (unchanged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--thr-json", type=Path, default=A.WORK / "_stats" / "watershed_threshold.json")
    a = ap.parse_args()
    t0 = time.time()
    win = A.crop_window(a.crop)
    x0, y0, x1, y1 = win
    out = a.out or A.WORK / "watershed" / a.crop
    thr_rec = json.load(open(a.thr_json))
    thr = float(thr_rec["thr"])
    H, W = A.image_shape()
    ids, vid, vx, vy, _ = C.load_nuclei(A.BUNDLE)
    vr = np.clip(vy, 0, H - 1).astype(np.int64)
    vc = np.clip(vx, 0, W - 1).astype(np.int64)

    core_all = A.padded_core(win, H, W)
    cores = C.tiles(*core_all)
    wins = [C.window(c, (H, W)) for c in cores]
    fx0, fy0 = min(w[0] for w in wins), min(w[1] for w in wins)
    fx1, fy1 = max(w[2] for w in wins), max(w[3] for w in wins)
    # every QC transcript the tiles' density sees (px in [fx0, fx1], py in [fy0, fy1]) plus 1 um slack
    tx = A.read_tx(fx0 * A.UM - 1, fy0 * A.UM - 1, fx1 * A.UM + 1, fy1 * A.UM + 1, ["x_location", "y_location"])
    px = tx.x_location.to_numpy() / C.UM          # float32, as the driver
    py = tx.y_location.to_numpy() / C.UM
    del tx
    print(f"{a.crop}: window {win}, cores {cores}, tile windows [{fx0},{fy0},{fx1},{fy1}], {len(px):,} QC "
          f"transcripts, thr {thr!r} ({time.time() - t0:.0f}s)", flush=True)
    WS._init((px, py), H, W, vid, vr, vc)

    local = np.zeros((fy1 - fy0, fx1 - fx0), dtype=np.int64)
    n_native = {}
    for core in cores:
        _, cells = WS.run_tile(core, thr)
        for c in cells:
            A.paste_local(local, fy0, fx0, c, c["label"])
            n_native[c["label"]] = c["native_px"]
    labels_win = local[y0 - fy0:y1 - fy0, x0 - fx0:x1 - fx0].astype(np.uint32)
    print(f"watershed: {len(n_native)} cells kept in the cores ({time.time() - t0:.0f}s)", flush=True)
    info = {"tool": "watershed", "crop": a.crop, "bundle": str(A.BUNDLE),
            "reference_driver": "paper_reseg_pipeline/scg/baselines/watershed_fullsample.py (run_tile imported)",
            "version": f"scikit-image {__import__('skimage').__version__}, scipy {__import__('scipy').__version__}",
            "parameters": {"markers": "nucleus polygon vertex pixels, dilation disk(3)", "sigma_px": WS.SIGMA,
                           "density_transcripts": "is_gene & qv >= 20", "mask": f"density > p{WS.MASK_PCT}",
                           "mask_threshold": thr, "mask_threshold_info": thr_rec, "tile_px": C.TILE_PX,
                           "margin_px": C.MARGIN_PX, "cores": cores,
                           "loader": "labels_to_polygons(min_px=15, n_verts=40) + pixel-centre rasterization, "
                                     "later label wins"},
            "n_nuclei": int(len(ids)), "n_watershed_cells_kept": len(n_native)}
    A.write_crop_outputs(out, win, labels_win, ids, info, t0)


if __name__ == "__main__":
    main()
