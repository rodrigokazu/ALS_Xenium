#!/usr/bin/env python3
"""Whole-sample transcript-density image on the morphology pixel grid (level 0, 0.2125 um/px).

QC transcripts (is_gene, qv >= 20) are counted per pixel (floor(x / um_per_px), as rules.transcript_pixel_labels
maps them), smoothed with a Gaussian of --sigma-um and scaled to transcripts / um^2. Written as a float32 zarr
(Y, X) with 4096 x 4096 chunks, to stand in for the 18S channel (option B: segment on transcript density).

Usage: make_density_image.py --bundle <Atera bundle> --out <dir>/tx_density.zarr [--sigma-um 0.5]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq, tifffile, zarr
from scipy import ndimage as ndi

UM = 0.2125
ap = argparse.ArgumentParser()
ap.add_argument("--bundle", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--sigma-um", type=float, default=0.5)
a = ap.parse_args()
t0 = time.time()
H, W = tifffile.TiffFile(a.bundle / "morphology_focus" / "ch0000_dapi.ome.tif").series[0].shape[-2:]
cnt = np.zeros(H * W, dtype=np.uint16)
n = 0
for b in pq.ParquetFile(a.bundle / "transcripts.parquet").iter_batches(
        batch_size=50_000_000, columns=["x_location", "y_location", "qv", "is_gene"]):
    b = b.filter(pc.and_(pc.cast(b.column("is_gene"), pa.bool_()), pc.greater_equal(b.column("qv"), 20)))
    c = np.floor(b.column("x_location").to_numpy() / UM).astype(np.int64)
    r = np.floor(b.column("y_location").to_numpy() / UM).astype(np.int64)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    idx = r[ok] * W + c[ok]
    u, k = np.unique(idx, return_counts=True)  # indices within a batch are unique after this
    cnt[u] += k.astype(np.uint16)
    n += int(ok.sum())
    print(f"  {n:,} QC transcripts counted ({time.time() - t0:.0f}s)", flush=True)
cnt = cnt.reshape(H, W)
s_px = a.sigma_um / UM
pad = int(np.ceil(4 * s_px)) + 2
z = zarr.open_array(str(a.out), mode="w", shape=(H, W), chunks=(4096, 4096), dtype=np.float32, zarr_format=2)
for y0 in range(0, H, 4096):
    y1 = min(H, y0 + 4096)
    p0, p1 = max(0, y0 - pad), min(H, y1 + pad)
    band = ndi.gaussian_filter(cnt[p0:p1].astype(np.float32), s_px) / (UM * UM)
    z[y0:y1] = band[y0 - p0:y0 - p0 + (y1 - y0)]
    print(f"  smoothed rows {y0}-{y1} ({time.time() - t0:.0f}s)", flush=True)
json.dump({"sigma_um": a.sigma_um, "um_per_px": UM, "units": "QC transcripts per um^2", "n_qc_transcripts": n,
           "shape": [H, W], "max_count_per_px": int(cnt.max())}, open(a.out.parent / "tx_density.json", "w"), indent=1)
print(f"done {time.time() - t0:.0f}s", flush=True)
