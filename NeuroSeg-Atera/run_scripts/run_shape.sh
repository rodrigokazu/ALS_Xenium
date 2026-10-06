#!/bin/bash
# I2 shape variants on the 9 windows, each followed by the Purkinje repair: m = membrane landscape, o = 1 um opening of the grown rim
cd /Users/marcel/PycharmProjects/spatial/atera_reseg; PY=~/miniforge3/envs/spatial/bin/python; L=/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/c63ac4c6-74e3-40d6-bf81-dcc9df19b8fd/scratchpad
export OMP_NUM_THREADS=2
V=$1; shift; ARGS="$@"
for w in spot1_purkinje_left spot2_gl_wm_centre spot3_purkinje_right held1 held2 held3 held4 held5 held6; do
  B=local_bundle; case $w in held*) B=local_bundle_heldout;; esac
  $PY phase3.py $w r_um=4 tlow=40 conf=2.0 lowconf=none bundle=$B out=$V $ARGS > $L/shape_${V}_$w.log 2>&1 || { echo "FAIL $V $w"; continue; }
  $PY pc_repair.py crops_${V}_500 PCR_$V 0.534 $w >> $L/shape_${V}_$w.log 2>&1 || echo "FAIL pcr $V $w"
  echo "done $V $w $(date +%H:%M)"
done
