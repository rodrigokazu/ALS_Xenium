#!/usr/bin/env python3
"""Crop-sized copy of the Atera bundle's inputs, so the benchmark's own Segger / BIDCell drivers (which read the whole
transcripts.parquet into memory, 47 GB here) can run unchanged on a crop.

Writes <work>/_minibundle/<crop>/transcripts.parquet (every column, every transcript in the crop +- 25 um, same
schema) and nucleus_boundaries.parquet (every vertex row of each nucleus with a vertex in that region); the drivers'
own --window (crop +- 25 um, um) then applies their standard window rules.

Usage: python make_minibundle.py --crop spot1_purkinje_left
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atera_common as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True)
    a = ap.parse_args()
    x0, y0, x1, y1 = A.crop_window(a.crop)
    wx0, wy0, wx1, wy1 = x0 * A.UM - A.PAD_UM, y0 * A.UM - A.PAD_UM, x1 * A.UM + A.PAD_UM, y1 * A.UM + A.PAD_UM
    out = A.WORK / "_minibundle" / a.crop
    out.mkdir(parents=True, exist_ok=True)
    f = ((pc.field("x_location") >= wx0) & (pc.field("x_location") <= wx1)
         & (pc.field("y_location") >= wy0) & (pc.field("y_location") <= wy1))
    t = pds.dataset(A.BUNDLE / "transcripts.parquet").to_table(filter=f)
    pq.write_table(t, out / "transcripts.parquet")
    nb = pds.dataset(A.BUNDLE / "nucleus_boundaries.parquet").to_table().to_pandas()
    near = nb.vertex_x.between(wx0, wx1) & nb.vertex_y.between(wy0, wy1)
    nb[nb.cell_id.isin(set(nb.cell_id[near]))].to_parquet(out / "nucleus_boundaries.parquet", index=False)
    print(a.crop, "window um", [wx0, wy0, wx1, wy1], t.num_rows, "transcripts", int(near.sum()), "nucleus vertex rows",
          flush=True)
    (out / "WINDOW_UM").write_text(" ".join(f"{v:.4f}" for v in (wx0, wy0, wx1, wy1)) + "\n")


if __name__ == "__main__":
    main()
