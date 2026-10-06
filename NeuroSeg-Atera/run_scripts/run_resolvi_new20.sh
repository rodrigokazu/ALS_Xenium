#!/bin/bash
cd /Users/marcel/PycharmProjects/spatial/atera_reseg; O="/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/compare/resolvi"
for k in tenx ours; do KMP_DUPLICATE_LIB_OK=TRUE ~/miniforge3/envs/resolvi/bin/python resolvi_run.py "$O/${k}_new20.h5ad" "$O/${k}_new20" > "$O/${k}_new20.log" 2>&1; echo "$k exit $?"; done; echo ALLDONE
