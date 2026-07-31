#!/bin/bash
# ========================================================================================
# ALS_Xenium repo commentary | qc_clustering/run_xenium_workflow.sh
#
# A thin launcher for the standalone workflow, six lines, carrying no SLURM directives of its
# own and invoking the python through a heredoc.
#
# Written for an interactive or already-allocated session rather than for submission. If you
# want it on the queue, either wrap it or copy the directive block from run_clustering.sh.
# Kept because it records how the workflow was actually invoked.
# ========================================================================================

set -euo pipefail
python -u - <<'PY'
print("PYTHON START OK", flush=True)
PY
python -u xenium_workflow.py
