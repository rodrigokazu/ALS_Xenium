#!/usr/bin/env python3
"""Whole-sample Cellpose normalization of the Atera slide: exact p1 / p99 of the boundary stain (ch0001) and DAPI
(ch0000) over the whole level-0 image, with cellpose_fullsample.py's own count_tile / pct_from_counts (imported
unchanged), i.e. the statistic the whole-sample Cellpose driver computes before any tile is segmented.

Output: <out>/cellpose_norm.json  {"b_lo", "b_hi", "d_lo", "d_hi"}
Run with the spatial env (needs tifffile / zarr only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atera_common as A  # noqa: E402

A.paper_path()
import common as C  # noqa: E402
import cellpose_fullsample as CP  # noqa: E402  (unchanged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=A.WORK / "_stats")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    z = C.open_morphology(A.BUNDLE)
    H, W = z.shape[-2:]
    tl = C.tiles(0, 0, W, H)
    cb = np.zeros(65536, dtype=np.int64)
    cd = np.zeros(65536, dtype=np.int64)
    with ProcessPoolExecutor(a.workers, initializer=CP._init, initargs=(str(A.BUNDLE),)) as ex:
        for i, (b, d) in enumerate(ex.map(CP.count_tile, tl), start=1):
            cb += b
            cd += d
            if i % 20 == 0 or i == len(tl):
                print(f"  tile {i}/{len(tl)} ({time.time() - t0:.0f}s)", flush=True)
    rec = {"b_lo": CP.pct_from_counts(cb, 1), "b_hi": CP.pct_from_counts(cb, 99),
           "d_lo": CP.pct_from_counts(cd, 1), "d_hi": CP.pct_from_counts(cd, 99),
           "source": "whole-sample p1 / p99 per channel (exact), Atera slide", "image_px": [int(W), int(H)],
           "channels": {"boundary": CP.CH_BOUNDARY, "dapi": CP.CH_DAPI}, "runtime_s": round(time.time() - t0)}
    json.dump(rec, open(a.out / "cellpose_norm.json", "w"), indent=1)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
