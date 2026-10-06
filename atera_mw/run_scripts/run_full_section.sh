#!/bin/bash
# Whole section, in order. Base chunks (7 parallel, ~4 h on a 12-core Mac) -> stitch -> I2 -> Purkinje repair -> counts
# [-> hybrid with Ranger glia -> counts] -> annotation [-> ResolVI]. Paths: see ../README.md section 6.
set -e; cd "$(dirname "$0")/../scripts"; PY=~/miniforge3/envs/spatial/bin/python
$PY make_chunks.py "$ATERA_ROOT/work/density/tx_density.zarr" chunks.txt
xargs -P 7 -L 1 ../run_scripts/run_chunk.sh < chunks.txt
for st in stitch i2 pcr count; do $PY full_post.py $st --workers 6; done                       # -> segmentation_full/
$PY atera_annot.py                                                                             # annotation of segmentation_full
$PY full_post.py hybc --workers 5                                                              # optional: Ranger glia (rule C)
COUNT_STORE=hybc COUNT_OUT="$ATERA_ROOT/segmentation_hybridC" $PY full_post.py count --workers 5
SEG_DIR="$ATERA_ROOT/segmentation_hybridC" $PY atera_annot.py
~/miniforge3/envs/resolvi/bin/python resolvi_full.py                                           # optional, ~9 h on CPU
