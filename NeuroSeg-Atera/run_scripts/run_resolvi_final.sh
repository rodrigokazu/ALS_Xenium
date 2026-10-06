#!/bin/bash
cd /Users/marcel/PycharmProjects/spatial/atera_reseg; O="/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/compare/resolvi"; export LABELS="/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/segmentation_hybridC/final_labels.zarr"
SUFFIX=_final9 ~/miniforge3/envs/spatial/bin/python resolvi_inputs.py "$O" > "$O/inputs_final9.log" 2>&1 || { echo FAIL in9; exit 1; }
SUFFIX=_finalnew20 ~/miniforge3/envs/spatial/bin/python resolvi_inputs.py "$O" "$O/windows_new20.txt" > "$O/inputs_finalnew20.log" 2>&1 || { echo FAIL in20; exit 1; }
for k in final9 finalnew20; do KMP_DUPLICATE_LIB_OK=TRUE ~/miniforge3/envs/resolvi/bin/python resolvi_run.py "$O/ours_${k}.h5ad" "$O/ours_${k}" > "$O/ours_${k}.log" 2>&1; echo "$k exit $?"; done; echo ALLDONE
