#!/bin/bash
# Waits for the three local Baysor crop runs to end, then scores every method present and copies the table, per-method
# JSONs and label images to the SSD. Started detached (nohup) so it outlives the agent session.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local/baselines
D="/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/compare"
CROPS="spot1_purkinje_left spot2_gl_wm_centre spot3_purkinje_right"
until [ "$(cat $W/_logs/baysor_spot*.log 2>/dev/null | grep -c '=== baysor .* end')" -ge 3 ]; do sleep 300; done
echo "baysor done $(date)"
bash "$HERE/run_local.sh" score > "$W/_logs/score_final.log" 2>&1
for m in voronoi watershed cellpose baysor segger bidcell; do
  for c in $CROPS; do
    [ -f "$W/$m/$c/labels_window.npy" ] || continue
    mkdir -p "$D/baselines/$m/$c"
    cp "$W/$m/$c/labels_window.npy" "$W/$m/$c/run_info.json" "$D/baselines/$m/$c/"
  done
done
mkdir -p "$D/baselines/scores"
cp "$W"/scores/*.json "$D/baselines/scores/"
cp "$W/scores/all.csv" "$D/baselines_scores.csv"
echo "finished $(date)"
