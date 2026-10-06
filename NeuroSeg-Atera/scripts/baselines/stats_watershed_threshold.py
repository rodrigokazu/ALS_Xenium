#!/usr/bin/env python3
"""Whole-sample mask threshold of the Watershed baseline for the Atera slide.

paper_reseg_pipeline/scg/baselines/watershed_fullsample.py masks the watershed with density > the 5th percentile of
the WHOLE sample's density image (2D histogram of the QC transcripts on the level-0 grid, gaussian_filter sigma 6 px),
computed exactly (zero count + log histogram of the positive values, then the bracketing bin's values sorted:
global_percentile). Its implementation holds every QC transcript's px / py in memory (3.0 B here, > 24 GB) and
rebuilds the density per tile from them; this script computes the same statistic from a streamed per-pixel count
image instead (uint16, 6.4 GB; floor(um / 0.2125) in float32 as the driver's px = x / UM), smoothed in 4096-row bands
with a 32-row pad (> the kernel's 24 px radius, as the driver's KPAD), so every pixel's density is the one the
driver computes. Same percentile arithmetic (np.percentile linear interpolation, same histogram edges).

Output: <out>/watershed_threshold.json  {"thr": ..., "n_pixels", "n_zero", "zero_fraction", ...}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atera_common as A  # noqa: E402

A.paper_path()
import watershed_fullsample as WS  # noqa: E402  (unchanged; SIGMA, MASK_PCT, KPAD)

BAND = 4096


def bands(H):
    for y0 in range(0, H, BAND):
        y1 = min(H, y0 + BAND)
        p0, p1 = max(0, y0 - WS.KPAD), min(H, y1 + WS.KPAD)
        yield y0, y1, p0, p1


def density_band(cnt, y0, y1, p0, p1):
    return gaussian_filter(cnt[p0:p1].astype(np.float64), sigma=WS.SIGMA)[y0 - p0:y1 - p0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=A.WORK / "_stats")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    H, W = A.image_shape()
    cnt = np.zeros(H * W, dtype=np.uint16)
    n = 0
    for b in pq.ParquetFile(A.BUNDLE / "transcripts.parquet").iter_batches(
            batch_size=50_000_000, columns=["x_location", "y_location", "qv", "is_gene"]):
        b = b.filter(pc.and_(pc.cast(b.column("is_gene"), pa.bool_()), pc.greater_equal(b.column("qv"), A.QV_MIN)))
        px = b.column("x_location").to_numpy() / A.UM          # float32, as watershed_fullsample
        py = b.column("y_location").to_numpy() / A.UM
        c = np.floor(px).astype(np.int64)
        r = np.floor(py).astype(np.int64)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        u, k = np.unique(r[ok] * W + c[ok], return_counts=True)
        cnt[u] += k.astype(np.uint16)
        n += int(ok.sum())
        print(f"  {n:,} QC transcripts counted ({time.time() - t0:.0f}s)", flush=True)
    cnt = cnt.reshape(H, W)
    print(f"max count per px {int(cnt.max())}", flush=True)

    edges = np.concatenate([[0.0], np.logspace(-30, 4, 3401)])
    n_zero, hist = 0, np.zeros(len(edges) - 1, dtype=np.int64)
    for y0, y1, p0, p1 in bands(H):
        d = density_band(cnt, y0, y1, p0, p1)
        pos = d[d > 0]
        n_zero += int(d.size - pos.size)
        hist += np.histogram(pos, bins=edges)[0]
        print(f"  pass 1 rows {y0}-{y1} ({time.time() - t0:.0f}s)", flush=True)
    N = n_zero + int(hist.sum())
    q = WS.MASK_PCT
    k = q / 100 * (N - 1)
    lo_i, hi_i = int(np.floor(k)), int(np.ceil(k))
    frac = k - lo_i

    def value_at(i):
        if i < n_zero:
            return 0.0
        j = i - n_zero
        cum = np.cumsum(hist)
        b = int(np.searchsorted(cum, j, side="right"))
        before = int(cum[b - 1]) if b > 0 else 0
        vals = []
        for y0, y1, p0, p1 in bands(H):
            d = density_band(cnt, y0, y1, p0, p1)
            vals.append(d[(d >= edges[b]) & (d < edges[b + 1])])
        vals = np.sort(np.concatenate(vals))
        return float(vals[j - before])

    v_lo = value_at(lo_i)
    v_hi = v_lo if hi_i == lo_i else value_at(hi_i)
    thr = v_lo + (v_hi - v_lo) * frac
    rec = {"thr": thr, "source": "whole-sample 5th percentile of the density image (Atera slide, exact)",
           "n_pixels": N, "n_zero": n_zero, "zero_fraction": n_zero / N, "n_qc_transcripts": n,
           "sigma_px": WS.SIGMA, "mask_pct": q, "image_px": [int(W), int(H)], "runtime_s": round(time.time() - t0)}
    json.dump(rec, open(a.out / "watershed_threshold.json", "w"), indent=1)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
