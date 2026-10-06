#!/usr/bin/env python3
"""Chunk list for the full-section run: 4096 px tiles of the density image with > 0.2% tissue (density > 10 tx/um2 on a 1/16 grid).
python make_chunks.py <tx_density.zarr> <out chunks.txt>   (2026-10-01: 147 of 209 tiles)"""
import sys, zarr, numpy as np
z = zarr.open_array(sys.argv[1], mode="r"); H, W = z.shape; T = 4096; out = []
for r in range(0, H, T):
    for c in range(0, W, T):
        a = np.asarray(z[r:min(r + T, H):16, c:min(c + T, W):16])
        if (a > 10).mean() > 0.002: out.append(f"ch_{r}_{c} {c} {r} {min(c + T, W)} {min(r + T, H)}")
open(sys.argv[2], "w").write("\n".join(out) + "\n"); print(len(out), "chunks")
