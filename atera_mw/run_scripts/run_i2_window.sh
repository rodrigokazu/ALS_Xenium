#!/bin/bash
# I2 (identity-gated growth) on one 500 um window from its C500 base run. Final settings, chosen on spot1 + spot2.
# usage: run_i2_window.sh <window> <bundle: local_bundle | local_bundle_heldout>
cd "$(dirname "$0")/../scripts"; OMP_NUM_THREADS=2 ~/miniforge3/envs/spatial/bin/python phase3.py $1 r_um=4 tlow=40 out=I2 conf=2.0 lowconf=none bundle=$2
