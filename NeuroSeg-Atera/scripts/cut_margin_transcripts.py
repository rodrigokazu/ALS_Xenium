#!/usr/bin/env python3
"""One streaming pass over transcripts.parquet keeping transcripts inside the crop windows plus the tile margin
(what run_fullsample.py --window reads), all columns. Low memory: filtered row batches appended to one file."""
import sys
import pyarrow.parquet as pq, pyarrow.compute as pc

UM, MARGIN_PX = 0.2125, 800
src, crops_tsv, out = sys.argv[1:4]
boxes = []
for line in open(crops_tsv):
    _, x0, y0, x1, y1 = line.split()
    boxes.append(tuple(int(v) for v in (x0, y0, x1, y1)))
pf = pq.ParquetFile(src)
w = None
n = 0
for b in pf.iter_batches(batch_size=2_000_000):
    x, y = b.column("x_location"), b.column("y_location")
    keep = None
    for x0, y0, x1, y1 in boxes:
        m = pc.and_(pc.and_(pc.greater_equal(x, max(0, x0 - MARGIN_PX) * UM), pc.less_equal(x, (x1 + MARGIN_PX) * UM)),
                    pc.and_(pc.greater_equal(y, max(0, y0 - MARGIN_PX) * UM), pc.less_equal(y, (y1 + MARGIN_PX) * UM)))
        keep = m if keep is None else pc.or_(keep, m)
    s = b.filter(keep)
    if w is None:
        w = pq.ParquetWriter(out, s.schema)
    w.write_batch(s)
    n += len(s)
w.close()
print("kept", n, flush=True)
