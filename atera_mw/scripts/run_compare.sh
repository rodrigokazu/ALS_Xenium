#!/bin/bash
# usage: run_compare.sh <crops subdir under _work_reseg_atera> <output subdir>   e.g. crops compare_18s
R=/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial
for n in spot1_purkinje_left spot2_gl_wm_centre spot3_purkinje_right; do
  /home/mw28/miniforge3/envs/spatial/bin/python3 /home/mw28/spatial_dapi_pipeline/atera_reseg/compare_crop.py \
    --bundle $R/Atera_CB/Cerebellum_sample --run $R/_work_reseg_atera/$1/$n --out $R/_work_reseg_atera/$2/$n || echo "FAILED $n"
done
