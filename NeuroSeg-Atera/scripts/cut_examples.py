#!/usr/bin/env python3
"""Cut example windows (transcripts + 10x cells/boundaries) from the Atera bundle, for local viewing.

--windows TSV: name x0_um y0_um x1_um y1_um. Writes <out>/<name>/{transcripts,cells,cell_boundaries,
nucleus_boundaries}.parquet; stains are cut locally from the SSD copy of morphology_focus.
"""
import argparse
from pathlib import Path
import pandas as pd, pyarrow.dataset as ds, pyarrow.compute as pc

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", type=Path, required=True)
ap.add_argument("--windows", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
win = pd.read_csv(a.windows, sep="\t", header=None, names=["name", "x0", "y0", "x1", "y1"])
tds = ds.dataset(a.bundle / "transcripts.parquet")
cells = pd.read_parquet(a.bundle / "cells.parquet")
cb = pd.read_parquet(a.bundle / "cell_boundaries.parquet")
nb = pd.read_parquet(a.bundle / "nucleus_boundaries.parquet")
for w in win.itertuples():
    o = a.out / w.name
    o.mkdir(parents=True, exist_ok=True)
    f = (pc.field("x_location") >= w.x0) & (pc.field("x_location") < w.x1) & (pc.field("y_location") >= w.y0) & (pc.field("y_location") < w.y1)
    t = tds.to_table(filter=f).to_pandas()
    t.to_parquet(o / "transcripts.parquet", index=False)
    c = cells[cells.x_centroid.between(w.x0, w.x1) & cells.y_centroid.between(w.y0, w.y1)]
    c.to_parquet(o / "cells.parquet", index=False)
    for df, nm in ((cb, "cell_boundaries"), (nb, "nucleus_boundaries")):
        df[df.cell_id.isin(set(c.cell_id))].to_parquet(o / f"{nm}.parquet", index=False)
    print(w.name, len(t), "transcripts", len(c), "cells", flush=True)
