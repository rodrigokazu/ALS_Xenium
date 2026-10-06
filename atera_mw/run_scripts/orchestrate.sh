#!/bin/bash
# waits for all base chunks, then stitch -> i2 -> pcr -> count. Reads $F/decision.txt before pcr: "pcr" (default) or "i2" (skip the repair;
# then count reads the i2 labels by copying the store name). Logs to $F/orchestrate.log
F=/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/c63ac4c6-74e3-40d6-bf81-dcc9df19b8fd/scratchpad/full
PY=~/miniforge3/envs/spatial/bin/python; cd /Users/marcel/PycharmProjects/spatial/atera_reseg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
N=$(wc -l < $F/chunks.txt)
while true; do
  done_n=$(ls $F/base/*/cells.parquet 2>/dev/null | wc -l); running=$(pgrep -f "run_fullsample.py" | wc -l)
  [ "$done_n" -ge "$N" ] && break
  [ "$running" -eq 0 ] && ! pgrep -f "xargs -P 7" > /dev/null && { echo "base stopped with $done_n/$N chunks $(date)"; break; }
  sleep 60
done
echo "base finished $(ls $F/base/*/cells.parquet | wc -l)/$N $(date)"
for st in stitch i2; do $PY full_post.py $st --workers 6 > $F/post_$st.log 2>&1 || { echo "FAILED $st $(date)"; exit 1; }; echo "$st done $(date)"; done
if [ "$(cat $F/decision.txt 2>/dev/null || echo pcr)" = "pcr" ]; then
  $PY full_post.py pcr --workers 6 > $F/post_pcr.log 2>&1 || { echo "FAILED pcr $(date)"; exit 1; }; echo "pcr done $(date)"
else
  echo "repair skipped by decision.txt; final = i2"; rm -rf "/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/segmentation_full/final_labels.zarr"
  cp -R "/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/segmentation_full/i2_labels.zarr" "/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/segmentation_full/final_labels.zarr"
fi
$PY full_post.py count --workers 6 > $F/post_count.log 2>&1 || { echo "FAILED count $(date)"; exit 1; }
echo "ALL DONE $(date)"; cat "/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/segmentation_full/summary.json"
