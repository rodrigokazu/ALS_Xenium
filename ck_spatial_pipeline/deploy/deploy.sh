#!/bin/bash
# deploy.sh: copy ck_spatial_pipeline/ back onto its flat SCG folders.
#
# The repo groups scripts by tool. SCG runs them from flat folders, and many of them import
# siblings by bare name (final_config, nature_style, dpqc_style_FINAL, ps_io_FINAL_fix,
# mn_annotation_join, jck_palette, smear_gate). manifest.tsv maps every repo file to the
# SCG folder and file name it runs from, so the tool folders here never break an import there.
#
# Default is a dry run. It prints one line per file with the state of the SCG copy:
#   same      SCG already holds this repo file
#   shipped   SCG holds the version recorded at ship time (safe to overwrite)
#   DRIFTED   SCG changed after ship time. Pull it into the repo first, or pass --force.
#   absent    no file at the SCG path yet
#
# Usage: deploy/deploy.sh [--apply] [--force] [--notebooks] [path-filter]
#   --apply      copy files (needs a live `ssh scg` master)
#   --force      overwrite DRIFTED files too
#   --notebooks  include the 20 dual-pass notebooks. They ship with outputs stripped, so
#                deploying them replaces the executed copies on SCG. Off by default.
#   path-filter  only rows whose repo path contains this string, e.g. 4_nichecompass
set -euo pipefail
cd "$(dirname "$0")/../.."
APPLY=0; FORCE=0; NB=0; FILTER=""
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;; --force) FORCE=1 ;; --notebooks) NB=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) FILTER="$a" ;;
  esac
done
MAN=ck_spatial_pipeline/deploy/manifest.tsv
n=0; nd=0
while IFS=$'\t' read -r repo dir name shipmd5 dep; do
  [[ "$repo" == \#* || -z "$repo" ]] && continue
  [[ -n "$FILTER" && "$repo" != *"$FILTER"* ]] && continue
  [[ "$dep" == "no" && $NB -eq 0 ]] && continue
  local_md5=$(md5sum < "$repo" | cut -c1-32)
  remote_md5=$(ssh -n -o BatchMode=yes scg "md5sum < '$dir/$name' 2>/dev/null | cut -c1-32" || true)
  if   [[ -z "$remote_md5" ]];              then state=absent
  elif [[ "$remote_md5" == "$local_md5" ]]; then state=same
  elif [[ "$remote_md5" == "$shipmd5" ]];   then state=shipped
  else state=DRIFTED; nd=$((nd+1)); fi
  printf '%-8s %s -> %s/%s\n' "$state" "$repo" "$dir" "$name"
  if [[ $APPLY -eq 1 && "$state" != same ]]; then
    if [[ "$state" == DRIFTED && $FORCE -eq 0 ]]; then echo "         skipped (drifted)"; continue; fi
    ssh -n -o BatchMode=yes scg "mkdir -p '$dir' && cp -p '$dir/$name' '$dir/$name.bak_predeploy_$(date +%Y%m%d)' 2>/dev/null || true"
    scp -q -p "$repo" "scg:$dir/$name"
    n=$((n+1))
  fi
done < "$MAN"
echo "copied $n file(s); $nd drifted"
