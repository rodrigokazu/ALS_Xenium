#!/bin/bash
# 500 um windows, local. Same drivers as run_local.sh; ATERA_CROPS / ATERA_WORK point to the 500 um window list and
# work dir, ATERA_MARGIN_PX=282 (+118 px core pad = 400 px of context = what the local transcripts cover).
#   bash run_local500.sh <voronoi|watershed|cellpose>
set -uo pipefail
S=/private/tmp/claude-501/-Users-marcel-PycharmProjects-spatial/49c3767a-5532-42a2-ba97-674a52d58694/scratchpad/atera_local
export ATERA_WORK=$S/baselines500 ATERA_CROPS=$S/baselines500/crops_500.tsv ATERA_MARGIN_PX=282
exec bash "$(dirname "$0")/run_local.sh" "$@"
