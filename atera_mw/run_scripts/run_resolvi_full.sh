#!/bin/bash
S=/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/c63ac4c6-74e3-40d6-bf81-dcc9df19b8fd/scratchpad
until grep -q ALLDONE $S/run_resolvi_final.log 2>/dev/null || ! pgrep -f run_resolvi_final.sh >/dev/null; do sleep 60; done
cd /Users/marcel/PycharmProjects/spatial/atera_reseg; KMP_DUPLICATE_LIB_OK=TRUE ~/miniforge3/envs/resolvi/bin/python resolvi_full.py && echo FULLDONE || echo FULLFAILED
