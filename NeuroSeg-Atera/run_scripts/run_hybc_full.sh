#!/bin/bash
cd /Users/marcel/PycharmProjects/spatial/atera_reseg; export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1; PY=~/miniforge3/envs/spatial/bin/python
$PY full_post.py hybc --workers 5 || { echo FAILED hybc; exit 1; }; echo "hybc done $(date)"
COUNT_STORE=hybc COUNT_OUT="/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/segmentation_hybridC" $PY full_post.py count --workers 5 || { echo FAILED count; exit 1; }; echo "count done $(date)"
echo ALLDONE
