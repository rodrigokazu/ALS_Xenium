#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/NC_v2_downstream.py
#
# Runs inside each training job: neighbours on nichecompass_latent (k 15), UMAP, Leiden 0.3
# into latent_leiden_0.3, then differential gene-programme tests (log Bayes factor 2.3) and
# the niche by cell type and niche by MN tables. It imports jck_palette and nature_style
# flat and falls back to an inline palette when they are missing.
# ========================================================================================

"""
NC_v2_downstream.py
===================

Generalised downstream analysis for ANY NicheCompass run directory produced by
NC_v2_train.py -- the four 20260902_ncv2 configurations:

  C1 persample_whole   one model per section, whole section        (20 sections)
  C2 persample_vh      one model per section, in_VH_v2 subset      (17 sections)
  C3 integrated_whole  one model, 20 whole sections, cat cov 'sample'
  C4 integrated_vh     one model, 17 VH subsets,   cat cov 'sample'

Provenance
----------
Generalisation of three scripts in ../NicheCompasso/ (READ, not modified):
  * NC_SC_downstream.py        -- the mechanics: load -> neighbors on the latent
                                  -> UMAP -> Leiden 0.3 -> GP summary ->
                                  differential GP tests -> MinMax GP heatmaps ->
                                  composition crosstabs -> per-sample spatial
                                  maps -> run_info.json. Palettes are built per
                                  category (create_new_color_dict, tab20 modulo
                                  fallback), NEVER a fixed short colour list.
  * NC_SC_downstream_v2_ismn.py -- the LABEL PARTITION (is_MN wins over every
                                  base class; the coarse transcriptional
                                  "Motor neurons" cluster is demoted to "Other
                                  neurons"), the gate ledger, and the dense UMAP
                                  recipe (explicit point size, alpha 0.9, seed-0
                                  shuffled draw order, motor neurons overplotted
                                  last) that scanpy defaults ruin at 1.1M cells.
  * niche_ismn_cross.py        -- the niche x is_MN Fisher-exact / BH-FDR logic
                                  with NA-preserving three-valued handling
                                  (True / False / Unknown), pooled and per
                                  status, each stratum its own BH family.
The gate MACHINERY is ported; none of the frozen is_MN_v1 / 8-niche /
1,097,669-cell expectations are: every expectation here is derived at runtime
from this run's own object and its own niche_sizes.csv.

THE LABEL CONTRACT (the point of the exercise)
----------------------------------------------
obs['cell_type_mn'] = obs['cell_type'] with the coarse transcriptional class
"Motor neurons" renamed "Other neurons"; then every cell with
is_MN_v2 == True becomes "Motor neurons", winning over every base class. The
result is a mutually exclusive, exhaustive 10-class partition:

  Astrocytes, Endothelial, Excitatory neurons, Inhibitory neurons, Macrophages,
  Microglia, OPCs, Oligodendrocytes, Other neurons, Motor neurons

The 10 levels are DECLARED even when a level is empty, because pandas' crosstab
drops unobserved categorical levels (verified on pandas 2.3.2) and because the
three vh_v2_source == 'excluded' sections (SD01015BG, SD01616BI, SD02913BA)
carry is_MN_v2 entirely <NA> and therefore contribute an EMPTY "Motor neurons"
class in a whole-section run. Those runs do not crash: the class is present with
0 cells and every MN table carries an explicit 'n/a' note instead of a fake test.

Cells with is_MN_v2 == <NA> are simply NOT "Motor neurons" -- they keep their
demoted base class -- so every motor-neuron count in a run containing <NA> cells
is a FLOOR, and the count of <NA> cells is recorded in the manifest. The name
"Motor neurons" NEVER refers to the transcriptional cluster anywhere in this
script's CSV headers, figure legends, titles or manifest.

is_MN_v2 was canonicalised on 2026-09-02 so that obs['is_MN'] == obs['is_MN_v2']
bit-for-bit; this script reads is_MN_v2 explicitly and GATES that equality
(NA-aware). obs['is_MN_v1'] (the superseded 774-cell call) is reported side by
side but never gated for equality: SD02913BA is 'excluded' (v2 all <NA>) yet
carries 12 non-NA v1 True cells, so a v1-vs-v2 equality assert would misfire.

Nullable booleans are read ONLY through
    pd.Series(obs[col]).astype("boolean").fillna(False).to_numpy(dtype=bool)
Every naive path (np.asarray(...).astype(bool), s.to_numpy(dtype=bool),
(s == True).to_numpy(dtype=bool), adata[obs['in_VH_v2']]) RAISES on an all-<NA>
column in this env (anndata 0.12.19 / pandas 2.3.2 / numpy 2.2.6).

Robustness for tiny runs
------------------------
A 931-cell VH subset may yield only 2-4 Leiden clusters, and can yield ONE.
Nothing here assumes >= 8 niches, a niche count, or a 10-colour palette:
  * n_neighbors for the latent graph is clamped to n_obs - 1;
  * niche categories are re-ordered NUMERICALLY (so 10 sorts after 9, not after 1);
  * run_differential_gp_tests is SKIPPED when fewer than 2 categories are
    observed -- its comparison group would be empty and numpy raises
    "a must be greater than 0 unless no samples are taken";
  * the Fisher enrichment is skipped (with an explicit note in the CSV) when the
    run has no is_MN_v2 call at all, or fewer than --min-mn-for-test flagged
    cells;
  * a DEGENERATE 2x2 table (a niche with no non-<NA> cell inside it, or none
    outside it -- a single-niche run, a niche made only of is_MN_v2-<NA> cells,
    a status stratum with no cell in some niche) is never handed to
    fisher_exact, which would return (nan, 1.0) and pose as a real negative.
    Such rows carry odds_ratio = p = q = NaN plus an explicit note, and are
    EXCLUDED from the BH family; the family size is written into every row
    (column n_tests_in_bh_family);
  * per-status enrichment runs only when more than one status is present, i.e.
    on the integrated configurations;
  * point sizes for the UMAP and the spatial maps scale with n_cells.

Colours
-------
Every class- or status-coloured surface uses the canonical JCK (multiome-paper)
palette of 2026-09-02: cell_type_mn classes (Motor neurons = teal #00868B,
drawn last and larger in every scatter), status (control / sporadic / c9), and
WDR49+ astrocytes red if ever shown. If a sibling jck_palette.py is importable
at runtime it is the source of truth, otherwise the identical inline dict is
used; the retired Wong / Okabe-Ito hexes are asserted absent at import. Niche
(Leiden) and sample colours are categorical and stay tab20 / nichecompass
create_new_color_dict based. Legends always show class counts, or an explicit
'n/a' where a section carries no is_MN_v2 call.

Outputs (all under <run-dir>/, nothing outside it)
--------------------------------------------------
  model/<adata>_Leiden.h5ad          the object with cell_type_mn + the niches
  results/*.csv, results/run_info.json, results/per_sample/*.csv
  figures/*.png  (dpi 150) and two HERO figures also as .pdf (dpi 300):
      niche_celltype_mn_composition_stacked.{pdf,png}
      umap_celltype_mn.{pdf,png}
  results/run_manifest.json          the trainer's manifest, with a NEW
                                     "downstream" block appended in place
                                     (idempotent: a re-run replaces that block
                                     and leaves every trainer key intact)
Nothing NC_v2_train.py wrote is overwritten: its own
results/cohort_composition_cell_type_mn.csv, results/section_stats.json,
results/active_gp_names.txt and figures/training_loss_curves.png all survive,
and the recomputed cohort table is written alongside as
cohort_composition_cell_type_mn_downstream.csv, gated against the trainer's
numbers (gate label_contract_matches_trainer).
This script NEVER touches artifacts/sample_integration/ or Ranger_procd/.

Exit code
---------
0 iff every content gate passed. A failing gate does NOT abort early: all
outputs are still written and the gate ledger is printed and recorded, then the
process exits 1 so an unattended SLURM job fails loudly without losing the
expensive GPU training that preceded it in the same job.

Environment
-----------
  module load python/3.11.1 gcc/13.3.0
  export PYTHONNOUSERSITE=1
  export LD_LIBRARY_PATH=/scg/apps/software/openssl/1.1.1g/lib:$LD_LIBRARY_PATH
  /oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/envs/nichecompass_env/bin/python \
      NC_v2_downstream.py --run-dir <...> --config <...>
(The openssl export is mandatory for the whole job: the H200 node image lacks OS
libssl1.1, and this CPU half now runs on that same node.)

The GPU is optional here: only the differential-GP forward pass uses it, and it
cost 110 s on 1.1M cells on CPU in the 2026-08-26 run. If moving the model onto
a visible GPU fails -- a small or already-busy device, which is exactly what a
login node offers -- the load is retried on CPU rather than throwing away a
training run that has already succeeded in the same job. --no-cuda forces CPU.
Run it on a compute node, not a login node: measured on the same 931-cell input,
sc.tl.umap took 4.2 s on a batch node and had not finished after 30 min on a
contended login node (numba JIT, single-threaded, against a busy machine).

Usage
-----
  NC_v2_downstream.py --run-dir .../runs_v2/integrated_whole/20260902_ncv2
  NC_v2_downstream.py --run-dir .../runs_v2/persample_vh/20260902_ncv2/SD01923BI
  NC_v2_downstream.py --run-dir <...> --config persample_vh --min-mn-for-test 3

NUMBA_CACHE_DIR: honoured, never set here. scanpy imports numba at import time,
so the variable must already be in the environment when python starts (the
sbatch exports it to a writable path so the JIT is not re-paid per section);
this script only reads and records it.
"""

from __future__ import annotations

# Headless backend MUST be set before any pyplot import (scanpy imports pyplot).
import matplotlib

matplotlib.use("Agg")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

# nichecompass is imported LAZILY by load_nichecompass() below: a cold import
# off /oak costs about 2.5 minutes, and a typo'd flag must fail in a second
# rather than after it. Both names are rebound on the module by that call.
NicheCompass = None
create_new_color_dict = None


def load_nichecompass():
    """Import nichecompass AFTER argparse has validated the command line."""
    global NicheCompass, create_new_color_dict
    from nichecompass.models import NicheCompass as _NicheCompass
    NicheCompass = _NicheCompass
    try:  # 97-colour palette helper; tab20-modulo fallback if ever absent
        from nichecompass.utils import create_new_color_dict as _ccd
        create_new_color_dict = _ccd
    except Exception:  # pragma: no cover
        create_new_color_dict = None
    return NicheCompass


try:  # BH-FDR; a local implementation stands in if statsmodels ever goes missing
    from statsmodels.stats.multitest import multipletests
except Exception:  # pragma: no cover
    multipletests = None

# Sibling-project publication style (pins pdf.fonttype=42 -- MUST be applied
# before the first figure or PDF text pages come out blank). Soft: this script
# runs without it.
sys.path.insert(0, "/home/rodrigok/SLURM_jobs/Spatial/VH_isolation")
try:
    import nature_style  # noqa: E402
except Exception:  # pragma: no cover
    nature_style = None

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CONFIGS = ("persample_whole", "persample_vh", "integrated_whole", "integrated_vh")
INTEGRATED_CONFIGS = ("integrated_whole", "integrated_vh")
VH_CONFIGS = ("persample_vh", "integrated_vh")

# AnnData keys -- must match NC_v2_train.py / nichecompass 0.3.3 defaults
LATENT_KEY = "nichecompass_latent"
GP_NAMES_KEY = "nichecompass_gp_names"
ACTIVE_GP_NAMES_KEY = "nichecompass_active_gp_names"
DGP_KEY_NICHE = "nichecompass_differential_gp_test_results"
DGP_KEY_CLASS = "nichecompass_differential_gp_test_results_cell_type_mn"
SAMPLE_KEY = "sample"
STATUS_KEY = "status"
CELL_TYPE_KEY = "cell_type"
CELL_TYPE_MN_KEY = "cell_type_mn"
SPATIAL_KEY = "spatial"          # NEVER 'spatial_grid_offset' / 'spatial_orig'
UMAP_KEY = "X_umap"

ISMN_V2_COL = "is_MN_v2"
ISMN_COL = "is_MN"
ISMN_V1_COL = "is_MN_v1"
IN_VH_COL = "in_VH_v2"
HAS_VH_COL = "has_vh_v2"
VH_SOURCE_COL = "vh_v2_source"

# The label partition
TRANSCRIPTIONAL_MN_CAT = "Motor neurons"   # the coarse cell_type cluster
MN_CLASS = "Motor neurons"                 # is_MN_v2 == True, and nothing else
OTHER_NEURONS = "Other neurons"            # the demoted transcriptional cluster

Q_LOG_PSEUDOCOUNT = 1e-300  # guards -log10(q) if a q underflows to exactly 0.0

STATUS_ORDER = ["control", "sporadic", "c9"]
STATUS_TITLES = {"control": "Control", "sporadic": "Sporadic", "c9": "C9"}


# --------------------------------------------------------------------------- #
# Colours: the canonical JCK (multiome-paper) palette, 2026-09-02
# --------------------------------------------------------------------------- #
# Every class- or status-coloured figure takes its hexes from CLASS_COLORS /
# STATUS_COLORS below. A sibling jck_palette.py (the project's canonical module)
# wins when importable at runtime; the inline dict carries the same values so a
# run without it is identical. Niche (Leiden) and sample colours are categorical
# and are NOT covered by the palette (tab20 / create_new_color_dict).

JCK_PALETTE_INLINE = {
    "cell_type_mn": {
        "Astrocytes": "#9E78BA",
        "Oligodendrocytes": "#6C4BA6",
        "OPCs": "#8B8DC0",
        "Microglia": "#EEEEA8",
        "Macrophages": "#C9752A",
        "Endothelial": "#F2D07A",
        "Excitatory neurons": "#2E83B0",
        "Inhibitory neurons": "#E8627A",
        OTHER_NEURONS: "#F9B446",   # the demoted coarse transcriptional cluster
        MN_CLASS: "#00868B",        # is_MN_v2 == True only: teal, drawn LAST and larger
    },
    "status": {"control": "#8FA7C7", "sporadic": "#A5B7A1", "c9": "#C9A08F"},
    "wdr49_astrocytes": "#FF0000",  # only if a WDR49+ astrocyte surface is ever drawn
    "neutral": "#6d7773",           # the one sanctioned neutral grey
}
# The Wong / Okabe-Ito set retired on 2026-09-02. Listed ONLY so the palette can
# be asserted clean at import; nothing in this script draws with them.
LEGACY_HEX = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7")


def _load_palette():
    """(class colours, status colours, WDR49 red, neutral grey, source label).

    Tries a sibling jck_palette.py first (same directory as this script, or
    anywhere on sys.path); falls back to JCK_PALETTE_INLINE. Any hex on which
    the two disagree is logged so palette drift is visible in the SLURM log.
    """
    inline = JCK_PALETTE_INLINE
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import jck_palette as _jck  # noqa: E402
        classes = {c: str(_jck.CELLTYPE[c]) for c in inline["cell_type_mn"]
                   if c in _jck.CELLTYPE}
        classes[MN_CLASS] = str(_jck.MN["all"])
        status = {s: str(_jck.STATUS[s]) for s in inline["status"]}
        wdr49 = str(_jck.WDR49_COLOR)
        neutral = str(getattr(_jck, "NEUTRAL", inline["neutral"]))
        missing = [c for c in inline["cell_type_mn"] if c not in classes]
        if missing:
            raise KeyError(f"jck_palette lacks {missing}")
        src = (f"jck_palette.py {getattr(_jck, 'PALETTE_VERSION', '?')} "
               f"({getattr(_jck, '__file__', '?')})")
        drift = {k: (v, inline["cell_type_mn"][k])
                 for k, v in classes.items()
                 if v.lower() != inline["cell_type_mn"][k].lower()}
        drift.update({k: (v, inline["status"][k]) for k, v in status.items()
                      if v.lower() != inline["status"][k].lower()})
        if drift:
            print(f"WARNING: jck_palette.py differs from the inline palette "
                  f"(module, inline): {drift} -- the module wins.", flush=True)
        return classes, status, wdr49, neutral, src
    except Exception as exc:  # jck_palette absent or incomplete: inline dict
        return (dict(inline["cell_type_mn"]), dict(inline["status"]),
                inline["wdr49_astrocytes"], inline["neutral"],
                f"inline JCK dict (jck_palette.py not importable: "
                f"{type(exc).__name__}: {exc})")


CLASS_COLORS, STATUS_COLORS, WDR49_COLOR, NEUTRAL, PALETTE_SOURCE = _load_palette()
MN_COLOR = CLASS_COLORS[MN_CLASS]                 # #00868B teal
OTHER_NEURONS_COLOR = CLASS_COLORS[OTHER_NEURONS]  # #F9B446
NONSIG_COLOR = NEUTRAL          # non-significant enrichment bars
UNKNOWN_COLOR = "#E0E0E0"       # the is_MN_v2 == <NA> band (structural, not a class)

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
for _name, _hex in list(CLASS_COLORS.items()) + list(STATUS_COLORS.items()):
    if not _HEX_RE.match(str(_hex)):
        raise SystemExit(f"Palette entry {_name!r} is not a 6-digit hex: {_hex!r}")
    if str(_hex).upper() in LEGACY_HEX:
        raise SystemExit(f"Palette entry {_name!r} = {_hex} is a retired Wong "
                         f"colour; the 2026-09-02 JCK palette forbids it.")

N_NEIGHBORS_LATENT = 15
LOG_BAYES_FACTOR_THRESH = 2.3
DEFAULT_Q_THRESH = 0.05

NA_MN_NOTE = "n/a - no is_MN_v2 call in this run (column is entirely <NA>)"
# Per-sample surfaces: an explicit call flag instead of a bare 0
NO_CALL_LABEL = "n/a (all <NA>)"
CALLED_LABEL = "called"
NO_CALL_FIG_NOTE = f"no {ISMN_V2_COL} call"
# A 2x2 table with an empty margin is never tested (scipy would return (nan, 1.0))
DEGENERATE_NOTE = ("n/a - degenerate 2x2 table ({zero}): no comparison group in "
                   "this stratum; not tested and excluded from the BH family")

# GP summary columns exported for enriched GPs (intersected with what
# get_gp_summary actually returns)
GP_TABLE_COLS = [
    "gp_name", "gp_active", "n_source_genes", "n_non_zero_source_genes",
    "n_target_genes", "n_non_zero_target_genes",
    "gp_source_genes", "gp_target_genes",
    "gp_source_genes_importances", "gp_target_genes_importances",
]


# --------------------------------------------------------------------------- #
# Print helpers (flush=True everywhere: SLURM buffers otherwise)
# --------------------------------------------------------------------------- #

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def section(msg):
    print(f"\n=== {msg} ===", flush=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Gate ledger (machinery ported from NC_SC_downstream_v2_ismn.py; every
# expectation is computed at runtime, none is frozen)
# --------------------------------------------------------------------------- #

class Gates:
    """Collect hard content gates, print each verdict, report at the end."""

    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail):
        ok = bool(ok)
        self.rows.append({"gate": name, "passed": ok, "detail": detail})
        print(f"GATE {name:<28} {'PASS' if ok else '*** FAIL ***'} | {detail}",
              flush=True)
        return ok

    def skip(self, name, detail):
        """A gate that does not apply to this configuration; never fails."""
        self.rows.append({"gate": name, "passed": True, "detail": f"n/a - {detail}"})
        print(f"GATE {name:<28} n/a  | {detail}", flush=True)
        return True

    @property
    def all_passed(self):
        return all(r["passed"] for r in self.rows)

    def summary(self):
        n_pass = sum(r["passed"] for r in self.rows)
        head = f"{n_pass}/{len(self.rows)} gates passed"
        if self.all_passed:
            return head + " (all)"
        bad = ", ".join(r["gate"] for r in self.rows if not r["passed"])
        return head + f"; FAILED: {bad}"

    def as_dict(self):
        return {r["gate"]: {"passed": r["passed"], "detail": r["detail"]}
                for r in self.rows}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generalised NicheCompass downstream for any runs_v2 run "
                    "directory: niches on the latent space, differential gene "
                    "programs, and every composition surface driven by the "
                    "cell_type_mn label contract (motor neurons = is_MN_v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True,
                   help="Run directory written by NC_v2_train.py: "
                        ".../runs_v2/<config>/<runstamp> for the integrated "
                        "configurations, .../runs_v2/<config>/<runstamp>/<TAG> "
                        "for the per-sample ones. Must contain model/.")
    p.add_argument("--config", default=None, choices=list(CONFIGS),
                   help="Configuration. Default: read from the run's "
                        "run_manifest.json, else inferred from the run-dir path.")
    p.add_argument("--leiden-res", type=float, default=0.3,
                   help="Leiden resolution on the NicheCompass latent space. "
                        "The niche column is obs['latent_leiden_<res>'].")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the shuffled scatter draw order and the "
                        "differential GP tests.")
    p.add_argument("--min-mn-for-test", type=int, default=5,
                   help="Minimum number of is_MN_v2 == True cells required "
                        "before the niche x motor-neuron Fisher enrichment is "
                        "run at all. Below it the CSV is written with explicit "
                        "'skipped' notes instead of a meaningless test.")
    # Secondary knobs: defaults are the proven 2026-08-26 settings.
    p.add_argument("--n-neighbors-latent", type=int, default=N_NEIGHBORS_LATENT,
                   help="k for sc.pp.neighbors on the latent space "
                        "(clamped to n_obs - 1 on tiny runs).")
    p.add_argument("--log-bayes-factor-thresh", type=float,
                   default=LOG_BAYES_FACTOR_THRESH,
                   help="Threshold for run_differential_gp_tests.")
    p.add_argument("--q-thresh", type=float, default=DEFAULT_Q_THRESH,
                   help="BH-FDR q cutoff for the enrichment tests and the "
                        "dashed line in the enrichment figure.")
    p.add_argument("--dpi", type=int, default=150,
                   help="dpi for the bulk PNG figures.")
    p.add_argument("--hero-dpi", type=int, default=300,
                   help="dpi for the two hero figures (also written as PDF).")
    p.add_argument("--adata-file-name", default=None,
                   help="Override the model's adata file name. Default: read "
                        "from run_manifest.json, else the single non-_Leiden "
                        "h5ad in model/.")
    p.add_argument("--skip-mn-gp-tests", action="store_true",
                   help="Skip the second run_differential_gp_tests call with "
                        "cat_key='cell_type_mn' (one extra encoder forward "
                        "pass over every cell).")
    p.add_argument("--no-cuda", action="store_true",
                   help="Force CPU even when a GPU is visible. Only the "
                        "differential-GP forward pass is GPU-accelerated "
                        "(110 s on 1.1M cells on CPU in the 2026-08-26 run), "
                        "so CPU is a perfectly good fallback -- useful on a "
                        "login node, or beside a small or busy GPU. A GPU "
                        "load failure falls back to CPU automatically anyway.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Run-directory / manifest plumbing
# --------------------------------------------------------------------------- #

def read_trainer_manifest(result_dir):
    """The trainer's results/run_manifest.json, or {} if it is not there yet."""
    fp = os.path.join(result_dir, "run_manifest.json")
    if not os.path.isfile(fp):
        log(f"WARNING: no trainer manifest at {fp}; the downstream block will "
            f"be written into a fresh file.")
        return {}, fp
    try:
        with open(fp) as fh:
            man = json.load(fh)
        if not isinstance(man, dict):
            log(f"WARNING: {fp} is not a JSON object; ignoring its content.")
            return {}, fp
        log(f"Read trainer manifest ({len(man)} top-level keys): {fp}")
        return man, fp
    except Exception as exc:
        log(f"WARNING: could not parse {fp} ({exc}); the downstream block will "
            f"be written into a fresh file.")
        return {}, fp


def infer_config(run_dir, manifest, override):
    """Configuration, from the CLI, then the manifest, then the run-dir path."""
    if override:
        return override, "--config"
    for holder, label in ((manifest, "run_manifest.json"),
                          (manifest.get("inputs") or {}, "run_manifest.json['inputs']"),
                          (manifest.get("run") or {}, "run_manifest.json['run']")):
        if isinstance(holder, dict):
            v = holder.get("config")
            if isinstance(v, str) and v in CONFIGS:
                return v, label
    for part in reversed(os.path.abspath(run_dir).split(os.sep)):
        if part in CONFIGS:
            return part, "run-dir path"
    raise SystemExit(
        f"Could not determine the configuration for {run_dir}. It is not in "
        f"the run manifest and no path component matches {CONFIGS}. "
        f"Pass --config explicitly."
    )


def resolve_adata_file_name(model_dir, manifest, override):
    """Name of the trained h5ad inside model/, from the manifest or by glob."""
    if override:
        if not os.path.isfile(os.path.join(model_dir, override)):
            raise FileNotFoundError(
                f"--adata-file-name {override!r} not found in {model_dir}")
        return override, "--adata-file-name"

    outputs = manifest.get("outputs") or {}
    candidates = []
    for holder in (outputs, manifest):
        if not isinstance(holder, dict):
            continue
        for key in ("adata", "adata_path", "adata_file", "h5ad",
                    "adata_file_name", "adata_filename"):
            v = holder.get(key)
            if isinstance(v, str) and v.endswith(".h5ad"):
                candidates.append(os.path.basename(v))
    for cand in candidates:
        if os.path.isfile(os.path.join(model_dir, cand)):
            return cand, "run_manifest.json"

    found = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(model_dir, "*.h5ad"))
        if not os.path.basename(p).endswith("_Leiden.h5ad")
    )
    if len(found) == 1:
        return found[0], "glob of model/*.h5ad"
    if not found:
        raise FileNotFoundError(
            f"No trained h5ad in {model_dir}. Did NC_v2_train.py finish with "
            f"model.save(..., save_adata=True)?"
        )
    raise RuntimeError(
        f"Ambiguous: {len(found)} candidate h5ad files in {model_dir} "
        f"({found}) and the manifest names none of them. "
        f"Pass --adata-file-name."
    )


def atomic_write_json(obj, path):
    """Write JSON via a temp file in the same directory, then os.replace."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# Nullable-boolean idioms (the ONLY safe paths in this env)
# --------------------------------------------------------------------------- #

def nullable_bool(obs, col):
    """obs[col] as a pandas nullable BooleanDtype Series (NA preserved)."""
    return pd.Series(obs[col]).astype("boolean")


def bool_mask(obs, col):
    """obs[col] as a plain numpy bool array with <NA> -> False.

    Every other route raises on an all-<NA> column in this env: verified
    np.asarray(s).astype(bool), s.astype(bool), s.to_numpy(dtype=bool),
    (s == True).to_numpy(dtype=bool) and adata[s] all raise.
    """
    return nullable_bool(obs, col).fillna(False).to_numpy(dtype=bool)


def nullable_bool_equal(a, b):
    """NA-aware bit-for-bit equality of two nullable booleans.

    Compares the NA masks and the filled values separately, so it is immune to
    the Series name / index differences that Series.equals also compares.
    """
    a = pd.Series(a).astype("boolean")
    b = pd.Series(b).astype("boolean")
    if len(a) != len(b):
        return False
    return bool(
        np.array_equal(a.isna().to_numpy(), b.isna().to_numpy())
        and np.array_equal(a.fillna(False).to_numpy(dtype=bool),
                           b.fillna(False).to_numpy(dtype=bool))
    )


def three_valued(obs, col, true_label="True", false_label="False",
                 na_label="Unknown"):
    """3-valued display label from a nullable boolean; NA is NEVER collapsed."""
    s = nullable_bool(obs, col)
    is_na = s.isna().to_numpy()
    is_true = s.fillna(False).to_numpy(dtype=bool)
    out = np.where(is_na, na_label, np.where(is_true, true_label, false_label))
    return pd.Series(out, index=obs.index, name=f"{col}_status")


# --------------------------------------------------------------------------- #
# The label partition
# --------------------------------------------------------------------------- #

def build_partition(obs):
    """The 10-class cell_type_mn partition. is_MN_v2 wins over every base class.

    Returns (cat, base, mn_mask, class_order). class_order keeps the surviving
    cell_type categories in their native order, then Other neurons, then Motor
    neurons LAST so the motor-neuron sliver sits on top of every stacked bar.

    The full class list is DECLARED on the Categorical even when a class is
    empty: pandas' crosstab drops unobserved categorical levels, so a section
    with no motor-neuron call would otherwise silently lose the column.
    """
    ct = obs[CELL_TYPE_KEY]
    if isinstance(ct.dtype, pd.CategoricalDtype):
        native = list(ct.cat.categories)
    else:
        native = sorted(pd.unique(ct.astype(str)))
    if TRANSCRIPTIONAL_MN_CAT not in native:
        raise SystemExit(
            f"obs['{CELL_TYPE_KEY}'] has no '{TRANSCRIPTIONAL_MN_CAT}' "
            f"category, so the demotion step cannot run. Categories: {native}"
        )
    survivors = [c for c in native if c != TRANSCRIPTIONAL_MN_CAT]
    class_order = survivors + [OTHER_NEURONS, MN_CLASS]

    base = ct.astype(str).to_numpy()
    mn = bool_mask(obs, ISMN_V2_COL)
    labels = np.where(base == TRANSCRIPTIONAL_MN_CAT, OTHER_NEURONS, base)
    labels = np.where(mn, MN_CLASS, labels)     # the flag wins, always
    cat = pd.Categorical(labels, categories=class_order, ordered=False)
    return cat, base, mn, class_order


def partition_colors(adata, class_order, survivors):
    """Colours for the 10 cell_type_mn classes: the canonical JCK palette.

    Every declared class takes its CLASS_COLORS hex ("Motor neurons" = teal
    #00868B is is_MN_v2 only; the demoted transcriptional cluster is "Other
    neurons" #F9B446). A class the palette does not name -- it should not
    happen, the 10 names are fixed -- falls back to the colour obs['cell_type']
    carries for it (build_color_dict) with a logged WARNING, so a renamed
    category never crashes the run and never silently borrows a semantic hue.
    `survivors` is kept for the call sites; the palette is keyed by name.
    """
    native = list(adata.obs[CELL_TYPE_KEY].cat.categories)
    uns_cols = list(adata.uns.get(f"{CELL_TYPE_KEY}_colors", []))
    native_map = (dict(zip(native, uns_cols[:len(native)]))
                  if len(uns_cols) >= len(native) else {})
    cmap = matplotlib.colormaps["tab20"]
    colors, unknown = {}, []
    for i, c in enumerate(class_order):
        if c in CLASS_COLORS:
            colors[c] = CLASS_COLORS[c]
        else:
            unknown.append(c)
            colors[c] = native_map.get(c) or mcolors.to_hex(cmap(i % 20))
    if unknown:
        log(f"WARNING: {len(unknown)} cell_type_mn class(es) are not in the JCK "
            f"palette and fall back to the cell_type / tab20 colour: {unknown}")
    log(f"cell_type_mn colours from {PALETTE_SOURCE}: '{MN_CLASS}' "
        f"{colors[MN_CLASS]} (teal; is_MN_v2 only; drawn last and larger), "
        f"'{OTHER_NEURONS}' {colors[OTHER_NEURONS]}; "
        f"{len(survivors)} survivor classes keep their palette hexes.")
    return colors


# --------------------------------------------------------------------------- #
# Palettes
# --------------------------------------------------------------------------- #

def jck_cell_type_colors(cats):
    """JCK palette for obs['cell_type'] (the 9 coarse classes).

    The coarse transcriptional "Motor neurons" cluster takes the "Other
    neurons" hex (#F9B446): that is what it becomes under the label contract,
    and the teal is reserved for is_MN_v2 == True. Returns None for any
    category the palette does not name, so the caller can fall back.
    """
    out = {}
    for c in cats:
        if c == TRANSCRIPTIONAL_MN_CAT:
            out[c] = CLASS_COLORS[OTHER_NEURONS]
        elif c in CLASS_COLORS:
            out[c] = CLASS_COLORS[c]
        else:
            out[c] = None
    return out


def build_color_dict(adata, cat_key):
    """Colour dict covering EVERY declared category of adata.obs[cat_key].

    obs['cell_type'] takes the JCK palette (jck_cell_type_colors), any category
    it does not name falling back to tab20 with a WARNING. Every other key
    (niches, samples) tries nichecompass' create_new_color_dict (97-colour
    default palette), validates that it covers all categories, otherwise falls
    back to tab20 with modulo indexing. Never indexes a fixed short colour list
    (Sanger palette landmine: >10 Leiden clusters caused a KeyError), and never
    keys off value_counts (a per-section object keeps all 9 cell_type levels
    even where one is empty).
    """
    if not isinstance(adata.obs[cat_key].dtype, pd.CategoricalDtype):
        adata.obs[cat_key] = adata.obs[cat_key].astype("category")
    cats = list(adata.obs[cat_key].cat.categories)

    if cat_key == CELL_TYPE_KEY:
        jck = jck_cell_type_colors(cats)
        cmap = matplotlib.colormaps["tab20"]
        unnamed = [c for c in cats if jck[c] is None]
        color_dict = {c: (jck[c] if jck[c] is not None
                          else mcolors.to_hex(cmap(i % 20)))
                      for i, c in enumerate(cats)}
        if unnamed:
            log(f"WARNING: {len(unnamed)} '{cat_key}' categor(ies) not in the "
                f"JCK palette, tab20 fallback: {unnamed}")
        adata.uns[f"{cat_key}_colors"] = [color_dict[c] for c in cats]
        return color_dict

    color_dict = None
    if create_new_color_dict is not None:
        try:
            cd = create_new_color_dict(adata=adata, cat_key=cat_key)
            if isinstance(cd, dict) and all(c in cd for c in cats):
                color_dict = {c: cd[c] for c in cats}
            else:
                log(f"WARNING: create_new_color_dict incomplete for "
                    f"'{cat_key}' ({len(cats)} categories); tab20 fallback.")
        except Exception as exc:
            log(f"WARNING: create_new_color_dict failed for '{cat_key}' "
                f"({exc}); tab20 fallback.")
    if color_dict is None:
        cmap = matplotlib.colormaps["tab20"]
        color_dict = {c: mcolors.to_hex(cmap(i % 20)) for i, c in enumerate(cats)}

    adata.uns[f"{cat_key}_colors"] = [color_dict[c] for c in cats]
    return color_dict


def point_size_for(n_cells):
    """Scatter point size that stays visible from 900 to 1.1M cells."""
    if n_cells <= 2_000:
        return 14.0
    if n_cells <= 10_000:
        return 8.0
    if n_cells <= 50_000:
        return 4.0
    if n_cells <= 150_000:
        return 2.5
    if n_cells <= 500_000:
        return 2.0
    return 1.5


# --------------------------------------------------------------------------- #
# h5ad hygiene
# --------------------------------------------------------------------------- #

def sanitize_uns_nones(mapping, path="uns"):
    """Replace None leaves in (nested) uns dicts with the string 'None'.

    None anywhere in .uns produces an unreadable h5ad in every anndata version.
    """
    n_fixed = 0
    for key in list(mapping.keys()):
        val = mapping[key]
        if val is None:
            mapping[key] = "None"
            log(f"WARNING: replaced None at {path}['{key}'] with 'None' "
                f"before the h5ad write.")
            n_fixed += 1
        elif isinstance(val, dict):
            n_fixed += sanitize_uns_nones(val, path=f"{path}['{key}']")
    return n_fixed


DGP_RAW_COLS = ["category", "gene_program", "p_h0", "p_h1", "log_bayes_factor"]


def dgp_raw_frame(adata, key):
    """The raw differential-GP results table, or an empty one with the right
    header when the test was skipped or nothing passed the threshold."""
    if key not in adata.uns:
        return pd.DataFrame(columns=DGP_RAW_COLS)
    try:
        df = pd.DataFrame(adata.uns[key])
    except Exception as exc:  # pragma: no cover
        log(f"WARNING: could not coerce adata.uns['{key}'] to a DataFrame "
            f"({exc}); writing an empty table.")
        return pd.DataFrame(columns=DGP_RAW_COLS)
    if df.shape[1] == 0:
        return pd.DataFrame(columns=DGP_RAW_COLS)
    return df


# --------------------------------------------------------------------------- #
# Figure saving
# --------------------------------------------------------------------------- #

def save_png(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {path}")
    return [path]


def save_hero(fig, base, hero_dpi):
    """A hero figure: vector PDF + PNG, both at hero_dpi.

    Delegates to nature_style.save_both when the sibling style module is
    importable (same convention as NC_SC_downstream_v2_ismn.py), otherwise
    writes the same two files directly. pdf.fonttype=42 is pinned at startup
    either way, before the first figure is drawn.
    """
    if nature_style is not None:
        pdf, png = nature_style.save_both(fig, base, dpi=hero_dpi)
    else:  # pragma: no cover
        pdf, png = f"{base}.pdf", f"{base}.png"
        fig.savefig(pdf, dpi=hero_dpi, bbox_inches="tight")
        fig.savefig(png, dpi=hero_dpi, bbox_inches="tight")
        plt.close(fig)
    log(f"  wrote {pdf}")
    log(f"  wrote {png}")
    return [pdf, png]


# --------------------------------------------------------------------------- #
# Figures: GP heatmap
# --------------------------------------------------------------------------- #

def plot_gp_heatmap(df_act, gps, outfn, title, dpi):
    """MinMax-scaled (per GP) heatmap of mean GP activity by niche.

    Safe with a single niche: MinMaxScaler maps a zero-range column to 0.0
    rather than raising (verified on this env), so a 1-row heatmap renders flat
    instead of crashing.
    """
    data = df_act[gps]
    norm = MinMaxScaler().fit_transform(data.values)
    norm_df = pd.DataFrame(norm, index=data.index.astype(str),
                           columns=data.columns)
    n_gps = len(gps)
    n_niches = norm_df.shape[0]
    fig_w = min(4.0 + 0.24 * n_gps, 110.0)
    fig_h = max(4.0, 0.35 * n_niches + 2.5)
    xtick_fs = 8 if n_gps <= 80 else (6 if n_gps <= 160 else 4)
    fig = plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(norm_df, cmap="viridis", linewidths=0,
                cbar_kws={"label": "MinMax-scaled mean GP activity"})
    plt.xticks(rotation=90, fontsize=xtick_fs)
    plt.yticks(fontsize=9)
    plt.ylabel("Niche")
    plt.title(title)
    plt.tight_layout()
    if n_niches == 1:
        log("  NOTE: a single niche makes every MinMax-scaled value 0 -- the "
            "heatmap is written for completeness but carries no contrast.")
    return save_png(fig, outfn, dpi)


# --------------------------------------------------------------------------- #
# Figures: stacked composition
# --------------------------------------------------------------------------- #

def stacked_composition(props, counts, colors, out_base, title, hero_dpi,
                        xlabel, highlight=MN_CLASS, highlight_note=None):
    """Stacked proportion bar per niche, with the highlighted class's COUNT
    printed above each bar.

    At 844 motor neurons in 1.1M cells (and 5 in some sections) the motor-neuron
    band is sub-pixel in a proportion bar, so the factual integer is printed
    above it. Column order comes from props, whose last column is the
    highlighted class -- drawn last, i.e. on top of the stack. Colours come
    from the JCK palette dict passed in (`colors`); the count annotations take
    the highlighted class's own hex.

    `highlight_note` (e.g. "no is_MN_v2 call") marks a run whose highlighted
    class is EMPTY BY CONSTRUCTION -- the section carries no call at all -- so
    the annotation above every bar and the legend read 'n/a', never a 0 that is
    indistinguishable from an observed zero.
    """
    cols = list(props.columns)
    x = np.arange(len(props))
    fig_w = max(7.0, 0.55 * len(props) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    bottoms = np.zeros(len(props), dtype=float)
    for c in cols:
        vals = np.nan_to_num(props[c].to_numpy(dtype=float), nan=0.0)
        ax.bar(x, vals, bottom=bottoms, width=0.85, color=colors[c],
               linewidth=0, label=c)
        bottoms += vals

    hi_color = colors.get(highlight, MN_COLOR)
    if highlight in counts.columns:
        for xi, n in zip(x, counts[highlight].to_numpy()):
            ax.plot([xi], [1.035], marker="v", markersize=4.0, color=hi_color,
                    markeredgecolor="none", clip_on=False, zorder=6)
            ax.text(xi, 1.065, "n/a" if highlight_note else f"{int(n):,}",
                    ha="center", va="bottom", fontsize=7, color=hi_color,
                    fontweight="bold", clip_on=False, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in props.index])
    ax.set_xlim(-0.7, len(props) - 0.3)
    ax.set_ylim(0.0, 1.16)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Proportion of niche")
    ax.set_title(title)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # legend top-of-stack first, so its order matches what the eye reads; every
    # class carries its count, the highlighted class 'n/a' when there is no call
    def _label(c):
        if c == highlight and highlight_note:
            return f"{c} (n/a - {highlight_note})"
        if c in counts.columns:
            return f"{c} (n = {int(counts[c].sum()):,})"
        return str(c)

    handles = [Patch(facecolor=colors[c], edgecolor="none", label=_label(c))
               for c in reversed(cols)]
    if highlight in counts.columns:
        handles.append(Line2D([0], [0], marker="v", linestyle="", markersize=4.0,
                              markerfacecolor=hi_color, markeredgecolor="none",
                              label=(f"{highlight} count above bar"
                                     if not highlight_note else
                                     f"{highlight}: n/a ({highlight_note})")))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, handletextpad=0.5, borderaxespad=0.0,
              ncol=2 if len(handles) > 26 else 1)
    return save_hero(fig, out_base, hero_dpi)


def stacked_bar_png(props, color_dict, outfn, title, xlabel, dpi,
                    ylabel="Proportion"):
    """Plain stacked proportion bar (bulk PNG); one bar per row of props."""
    cols = list(props.columns)
    x = np.arange(len(props))
    fig_w = max(8.0, 0.45 * len(props) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    bottoms = np.zeros(len(props), dtype=float)
    for c in cols:
        vals = np.nan_to_num(props[c].to_numpy(dtype=float), nan=0.0)
        ax.bar(x, vals, bottom=bottoms, width=0.85, color=color_dict[c],
               linewidth=0, label=str(c))
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in props.index], rotation=90)
    ax.set_xlim(-0.7, len(props) - 0.3)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(title=props.columns.name, bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=8,
              ncol=2 if len(cols) > 25 else 1, frameon=False)
    plt.tight_layout()
    return save_png(fig, outfn, dpi)


# --------------------------------------------------------------------------- #
# Figures: dense scatter (UMAP and spatial)
# --------------------------------------------------------------------------- #

def dense_scatter(xy, codes, cats, colors, out_base, title, dpi, seed,
                  s_bg, highlight=None, s_hi=None, hero_dpi=None,
                  xlabel="UMAP1", ylabel="UMAP2", equal_aspect=False,
                  invert_y=False, despine=True, legend_counts=True,
                  highlight_note=None):
    """The dense-render recipe: explicit point size, alpha 0.9, seed-shuffled
    draw order so no large class buries a small one, and the highlighted class
    drawn LAST, larger, white-edged, in its OWN palette colour (teal for motor
    neurons), so it reads on top of up to 1.1M points.

    scanpy defaults collapse to sub-pixel points at this cell count; that is why
    the point size is explicit and scaled to n_cells. The legend carries every
    class count; `highlight_note` replaces the highlighted class's count with
    'n/a (<note>)' where the class is empty by construction (no is_MN_v2 call).
    """
    cats = list(cats)
    codes = np.asarray(codes)
    # A -1 code (a cell with no category) would index the LAST palette entry
    # and mis-colour silently, so it is an error, not a warning.
    if codes.size and int(codes.min()) < 0:
        raise SystemExit(
            f"{int((codes < 0).sum()):,} cell(s) carry category code -1 in the "
            f"'{title}' panel. A -1 silently takes the last colour in the "
            f"palette; fix the categorical before plotting.")
    pal = np.array([mcolors.to_rgba(colors[c]) for c in cats])

    rng = np.random.default_rng(seed)
    order = rng.permutation(xy.shape[0])
    hi_code = cats.index(highlight) if (highlight in cats) else None
    if hi_code is None:
        bg, hi = order, np.empty(0, dtype=np.int64)
    else:
        bg = order[codes[order] != hi_code]
        hi = np.flatnonzero(codes == hi_code)

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    if bg.size:
        ax.scatter(xy[bg, 0], xy[bg, 1], s=s_bg, c=pal[codes[bg]], alpha=0.9,
                   linewidths=0, rasterized=True)
    if hi.size:
        ax.scatter(xy[hi, 0], xy[hi, 1], s=s_hi, c=colors[highlight], alpha=1.0,
                   linewidths=0.3, edgecolors="white", rasterized=True, zorder=5)

    if equal_aspect:
        ax.set_aspect("equal")
    if invert_y:
        ylim = ax.get_ylim()
        if ylim[0] < ylim[1]:
            ax.invert_yaxis()
    if despine:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    else:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    handles = []
    for i, c in enumerate(cats):
        n = int((codes == i).sum())
        is_hi = (hi_code is not None and i == hi_code)
        if is_hi and highlight_note:
            label = f"{c} (n/a - {highlight_note})"
        else:
            label = f"{c} (n = {n:,})" if legend_counts else str(c)
        handles.append(Line2D([0], [0], marker="o", linestyle="",
                              markersize=7 if is_hi else 6,
                              markerfacecolor=colors[c],
                              markeredgecolor="white" if is_hi else "none",
                              markeredgewidth=0.5 if is_hi else 0.0,
                              label=label))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, handletextpad=0.3, borderaxespad=0.0,
              ncol=1 if len(cats) <= 14 else 2)

    if hero_dpi is not None:
        return save_hero(fig, out_base, hero_dpi)
    return save_png(fig, f"{out_base}.png", dpi)


# --------------------------------------------------------------------------- #
# niche x is_MN_v2: crosstab, Fisher enrichment, figures
# --------------------------------------------------------------------------- #

def bh_fdr(pvals, alpha):
    """Benjamini-Hochberg q values. Uses statsmodels when available."""
    pvals = np.asarray(pvals, dtype=float)
    if pvals.size == 0:
        return pvals
    if multipletests is not None:
        return np.asarray(multipletests(pvals, alpha=alpha, method="fdr_bh")[1],
                          dtype=float)
    # Local fallback (same algorithm) so a missing statsmodels cannot lose a run
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


THREE_VALUED_LABELS = {"True", "False", "Unknown"}


def mn_crosstab(obs, niche_col, niches, three_valued_col):
    """niche x {True, False, Unknown} counts and proportions of a THREE-VALUED
    column (the output of three_valued(): is_MN_v2_status, in_VH_v2_status).

    `three_valued_col` is NOT the disease-status column: passing obs['status']
    would yield none of the three labels, every count would silently fill as 0
    and the table would render all-zero, so the values are checked first.

    True/False fractions are of the non-NA total per niche; Unknown's fraction
    is of the full per-niche total, because NA is not part of the non-NA
    denominator by construction.
    """
    obs_vals = set(pd.unique(obs[three_valued_col].astype(str)))
    if not obs_vals <= THREE_VALUED_LABELS:
        raise ValueError(
            f"mn_crosstab expects a three-valued True/False/Unknown column "
            f"(from three_valued()), got values {sorted(obs_vals)[:5]} from "
            f"'{three_valued_col}'")
    ct = pd.crosstab(obs[niche_col], obs[three_valued_col])
    for col in ("True", "False", "Unknown"):
        if col not in ct.columns:
            ct[col] = 0
    ct = ct.reindex(index=niches, fill_value=0)[["True", "False", "Unknown"]]
    ct.columns = ["n_true", "n_false", "n_unknown"]
    ct["n_total"] = ct["n_true"] + ct["n_false"] + ct["n_unknown"]
    ct.index.name = "niche"
    counts_df = ct.reset_index()

    n_non_na = counts_df["n_true"] + counts_df["n_false"]
    props_df = counts_df.copy()
    props_df["n_nonNA"] = n_non_na
    props_df["frac_true_of_nonNA"] = counts_df["n_true"] / n_non_na.replace(0, np.nan)
    props_df["frac_false_of_nonNA"] = counts_df["n_false"] / n_non_na.replace(0, np.nan)
    props_df["frac_unknown_of_total"] = (
        counts_df["n_unknown"] / counts_df["n_total"].replace(0, np.nan))
    return counts_df, props_df


ENRICH_COLS = ["niche", "n_mn_in_niche", "n_mn_out_niche", "n_total_in_niche",
               "n_total_out_niche", "odds_ratio", "p", "q", "significant",
               "n_tests_in_bh_family", "note"]


def degenerate_margins(table):
    """Names of the empty margins of a 2x2 table, or [] if it is testable.

    fisher_exact does NOT raise on an empty row or column: it returns
    (nan, 1.0), a fake negative. A zero row is a stratum with no non-<NA> cell
    inside (or outside) the niche; a zero column is a stratum with no flagged
    (or no unflagged) cell at all. None of the four is a test.
    """
    (a, b), (c, d) = table
    zero = []
    if a + b == 0:
        zero.append("n_total_in_niche=0")
    if c + d == 0:
        zero.append("n_total_out_niche=0")
    if a + c == 0:
        zero.append("n_mn_total=0")
    if b + d == 0:
        zero.append("n_non_mn_total=0")
    return zero


def niche_enrichment(obs, niche_col, mn_col, niches, q_thresh, min_mn,
                     context):
    """One 2x2 Fisher-exact test per niche (in-niche vs out, non-NA cells only),
    BH-FDR corrected across the TESTABLE niches passed in as ONE family.

    Three explicit non-test outcomes, each written into the CSV with a note
    instead of a number, because scipy returns (nan, 1.0) for a degenerate
    table rather than raising -- a silent fake result is worse than a gap:
      * no is_MN_v2 call at all in this run (the three 'excluded' sections);
      * fewer than `min_mn` flagged cells to test with (never fewer than 1);
      * a DEGENERATE 2x2 table for that niche: no non-<NA> cell inside it
        (a niche made only of is_MN_v2-<NA> cells, or a status stratum with no
        cell in the niche) or none outside it (a single-niche run). Such rows
        get odds_ratio = p = q = NaN, a note, and are EXCLUDED from the BH
        family, so they cannot inflate q for the niches that were tested.
    Every row records n_tests_in_bh_family, the size of the family its q was
    corrected within (0 where nothing was tested).
    """
    non_na = obs[nullable_bool(obs, mn_col).notna().to_numpy()]
    total_non_na = int(len(non_na))
    if total_non_na == 0:
        log(f"[{context}] {NA_MN_NOTE} -- writing explicit n/a rows, no test.")
        rows = [{"niche": n, "n_mn_in_niche": pd.NA, "n_mn_out_niche": pd.NA,
                 "n_total_in_niche": int((obs[niche_col] == n).sum()),
                 "n_total_out_niche": pd.NA, "odds_ratio": np.nan,
                 "p": np.nan, "q": np.nan, "significant": False,
                 "n_tests_in_bh_family": 0,
                 "note": NA_MN_NOTE} for n in niches]
        return pd.DataFrame(rows, columns=ENRICH_COLS), NA_MN_NOTE

    mn_bool = pd.Series(non_na[mn_col]).astype("boolean").fillna(False)
    mn_bool = mn_bool.to_numpy(dtype=bool)
    total_mn_pos = int(mn_bool.sum())
    n_na = int(len(obs) - total_non_na)
    log(f"[{context}] {total_non_na:,} cells enter the test "
        f"({total_mn_pos:,} {MN_CLASS.lower()}); {n_na:,} is_MN_v2-<NA> cells "
        f"dropped.")

    skip_note = ""
    # never test with zero flagged cells, whatever --min-mn-for-test says: the
    # motor-neuron column of every table would be empty (a degenerate family)
    if total_mn_pos < max(1, int(min_mn)):
        skip_note = (f"skipped - only {total_mn_pos} is_MN_v2 True cell(s) in "
                     f"this stratum (< --min-mn-for-test {min_mn})")
        log(f"[{context}] {skip_note}")

    niche_vals = non_na[niche_col].to_numpy()
    rows, pvals, testable_pos = [], [], []
    for niche in niches:
        in_mask = niche_vals == niche
        n_total_in = int(in_mask.sum())
        n_mn_in = int((mn_bool & in_mask).sum())
        n_total_out = total_non_na - n_total_in
        n_mn_out = total_mn_pos - n_mn_in
        row = {"niche": niche, "n_mn_in_niche": n_mn_in,
               "n_mn_out_niche": n_mn_out, "n_total_in_niche": n_total_in,
               "n_total_out_niche": n_total_out, "odds_ratio": np.nan,
               "p": np.nan, "note": skip_note}
        if not skip_note:
            table = [[n_mn_in, n_total_in - n_mn_in],
                     [n_mn_out, n_total_out - n_mn_out]]
            zero = degenerate_margins(table)
            if zero:
                # fisher_exact is NOT called: it would return (nan, 1.0)
                row["note"] = DEGENERATE_NOTE.format(zero=", ".join(zero))
            else:
                odds_ratio, p = fisher_exact(table, alternative="greater")
                row.update({"odds_ratio": float(odds_ratio), "p": float(p)})
                pvals.append(float(p))
                testable_pos.append(len(rows))
        rows.append(row)

    df = pd.DataFrame(rows)
    df["q"] = np.nan
    if testable_pos:
        df.loc[df.index[testable_pos], "q"] = bh_fdr(pvals, q_thresh)
    df["significant"] = (df["q"] < q_thresh).fillna(False).astype(bool)
    df["n_tests_in_bh_family"] = int(len(testable_pos))
    n_degenerate = int(df["note"].str.startswith("n/a - degenerate").sum())
    if n_degenerate:
        log(f"[{context}] {n_degenerate} of {len(niches)} niche(s) have a "
            f"degenerate 2x2 table (empty margin): written as NaN + note, NOT "
            f"tested, excluded from the BH family of {len(testable_pos)}.")
    return df.reindex(columns=ENRICH_COLS), skip_note


def json_safe_odds_ratio(value):
    """Odds ratio for the manifest, JSON-safe but never falsified.

    An INFINITE odds ratio is a real and important result -- it means the niche
    holds every motor neuron in the run and none sits outside it, which is a
    likely outcome for a section carrying only a handful of flagged cells. JSON
    has no infinity, so it is recorded as the string "inf" rather than dropped.
    A NaN odds ratio is different: it comes from a degenerate 2x2 table (an
    empty margin) and carries no information, so it becomes null.
    """
    v = float(value)
    if np.isnan(v):
        return None
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def mn_verdict(pooled_enrich, q_thresh):
    """The run's motor-neuron niche verdict for the manifest.

    Ranks the niches by (significant, then q ascending, then odds ratio
    descending) and returns the winner, or an explicit 'no test' record.
    """
    df = pooled_enrich.copy()
    tested = df[df["q"].notna()]
    n_untestable = int(len(df) - len(tested))
    if tested.empty:
        note = str(df["note"].iloc[0]) if len(df) else "no niches"
        return {"tested": False, "note": note, "top_niche": None,
                "odds_ratio": None, "q": None, "n_mn_in_niche": None,
                "significant": False, "n_significant_niches": 0,
                "n_tested_niches": 0, "n_untestable_niches": n_untestable}
    tested = tested.sort_values(
        by=["significant", "q", "odds_ratio"],
        ascending=[False, True, False], kind="mergesort")
    top = tested.iloc[0]
    orr = json_safe_odds_ratio(top["odds_ratio"])
    note = (f"pooled Fisher-exact (greater), BH-FDR across {len(tested)} "
            f"testable niches, q < {q_thresh}")
    if n_untestable:
        note += (f"; {n_untestable} niche(s) untestable (degenerate 2x2 table) "
                 f"and excluded from the family")
    if orr == "inf":
        note += ("; the odds ratio is infinite because every motor neuron in "
                 "this run lies inside this niche")
    elif orr is None:
        note += ("; the odds ratio is undefined (a degenerate 2x2 table, i.e. "
                 "an empty margin)")
    return {
        "tested": True,
        "note": note,
        "top_niche": str(top["niche"]),
        "odds_ratio": orr,
        "q": float(top["q"]),
        "n_mn_in_niche": int(top["n_mn_in_niche"]),
        "significant": bool(top["significant"]),
        "n_significant_niches": int(tested["significant"].sum()),
        "n_tested_niches": int(len(tested)),
        "n_untestable_niches": n_untestable,
    }


def sample_niche_mn_table(obs_small, samples, niches, niche_col, mn_status_col,
                          sample_key=SAMPLE_KEY, mn_col=ISMN_V2_COL):
    """Per-sample x niche counts of is_MN_v2 == True, with an EXPLICIT call flag.

    A section whose is_MN_v2 is entirely <NA> (the three 'excluded' sections in
    a whole-cohort run) has no call: its row reads 'n/a' in every niche column
    and NO_CALL_LABEL in `is_MN_v2_call`, so a zero by construction can never be
    read as a zero by observation. Returns (table, {sample: has_no_call}).
    """
    mn_true = obs_small[obs_small[mn_status_col] == "True"]
    sn = pd.crosstab(mn_true[sample_key], mn_true[niche_col])
    sn = sn.reindex(index=samples, columns=niches, fill_value=0).astype("int64")
    sn.columns = [f"niche_{n}" for n in niches]
    na_all = (nullable_bool(obs_small, mn_col).isna()
              .groupby(obs_small[sample_key].astype(str).to_numpy()).all())
    no_call = {s: bool(na_all.get(s, False)) for s in samples}
    sn = sn.astype(object)
    for s in samples:
        if no_call[s]:
            sn.loc[s, :] = "n/a"
    sn.insert(0, f"{mn_col}_call",
              [NO_CALL_LABEL if no_call[s] else CALLED_LABEL for s in samples])
    sn.insert(1, f"n_{mn_col}_true",
              ["n/a" if no_call[s] else int(mn_true[sample_key].astype(str).eq(s).sum())
               for s in samples])
    sn.index.name = sample_key
    return sn, no_call


def plot_mn_composition(counts_df, niches, out_png, dpi, niche_col):
    """Three-valued niche x is_MN_v2 composition (True / False / Unknown)."""
    df = counts_df.set_index("niche").reindex(niches)
    total = df["n_total"].replace(0, np.nan)
    frac_true = (df["n_true"] / total).fillna(0.0)
    frac_false = (df["n_false"] / total).fillna(0.0)
    frac_unknown = (df["n_unknown"] / total).fillna(0.0)

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(niches)), 5))
    x = np.arange(len(niches))
    n_true_total = int(df["n_true"].fillna(0).sum())
    n_false_total = int(df["n_false"].fillna(0).sum())
    n_unknown_total = int(df["n_unknown"].fillna(0).sum())
    ax.bar(x, frac_true,
           label=f"{ISMN_V2_COL} = True ({MN_CLASS}; n = {n_true_total:,})",
           color=MN_COLOR, width=0.85, linewidth=0)
    ax.bar(x, frac_false, bottom=frac_true,
           label=f"{ISMN_V2_COL} = False (n = {n_false_total:,})",
           color=NEUTRAL, width=0.85, linewidth=0)
    ax.bar(x, frac_unknown, bottom=frac_true + frac_false,
           label=f"{ISMN_V2_COL} = Unknown (<NA>; n = {n_unknown_total:,})",
           color=UNKNOWN_COLOR, width=0.85, linewidth=0)
    # count above each bar; 'n/a' where the niche holds no called cell at all
    n_called = (df["n_true"].fillna(0) + df["n_false"].fillna(0)).to_numpy()
    for xi, n, nc in zip(x, df["n_true"].fillna(0).to_numpy(), n_called):
        ax.text(xi, 1.01, "n/a" if nc == 0 else f"{int(n):,}", ha="center",
                va="bottom", fontsize=7, color=MN_COLOR, fontweight="bold",
                clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in niches])
    ax.set_xlabel(f"Niche ({niche_col})")
    ax.set_ylabel("Proportion of cells")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Niche x {ISMN_V2_COL} composition "
                 f"(motor neurons = {ISMN_V2_COL})")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              frameon=False)
    plt.tight_layout()
    return save_png(fig, out_png, dpi)


def _enrichment_panel(ax, df, title, q_thresh, accent):
    """One -log10(q) bar panel. Significant niches take `accent` (the motor-
    neuron teal for the pooled panel, the JCK status colour for a status
    stratum); non-significant niches the neutral grey; untestable niches
    (degenerate table, q = NaN) get no bar and an explicit 'n/a' label."""
    if df is None or len(df) == 0:
        ax.set_title(f"{title} (no data)")
        ax.axis("off")
        return
    if df["q"].notna().sum() == 0:
        note = str(df["note"].iloc[0]) if "note" in df.columns else "no test"
        ax.set_title(f"{title}\n({note})", fontsize=9)
        ax.axis("off")
        return
    df = df.sort_values("niche", key=_niche_sort_key, kind="mergesort")
    q = df["q"].to_numpy(dtype=float)
    tested = ~np.isnan(q)
    neglog_q = np.where(tested, -np.log10(np.nan_to_num(q, nan=1.0)
                                          + Q_LOG_PSEUDOCOUNT), 0.0)
    colors = [accent if (t and sig) else NONSIG_COLOR
              for t, sig in zip(tested, df["significant"].to_numpy())]
    x = np.arange(len(df))
    ax.bar(x, neglog_q, color=colors, width=0.85, linewidth=0)
    for xi, t in zip(x, tested):
        if not t:
            ax.text(xi, 0.02, "n/a", ha="center", va="bottom", fontsize=7,
                    color=NONSIG_COLOR, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(df["niche"].astype(str).tolist())
    ax.axhline(-np.log10(q_thresh), color="black", linestyle="--", linewidth=1,
               label=f"q = {q_thresh}")
    n_tested = int(tested.sum())
    n_sig = int((df["significant"].to_numpy() & tested).sum())
    ax.set_title(f"{title}: {MN_CLASS.lower()} enrichment per niche\n"
                 f"(BH family n = {n_tested}; {n_sig} significant"
                 + (f"; {int((~tested).sum())} untestable" if (~tested).any()
                    else "") + ")", fontsize=9)
    ax.set_xlabel("Niche")
    ax.set_ylabel("-log10(q)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor=accent, edgecolor="none",
                     label=f"significant (q < {q_thresh})"),
               Patch(facecolor=NONSIG_COLOR, edgecolor="none",
                     label="not significant"),
               Line2D([0], [0], color="black", linestyle="--", linewidth=1,
                      label=f"q = {q_thresh}")]
    ax.legend(handles=handles, fontsize=7, loc="upper right", frameon=False)


def plot_enrichment_grid(pooled_df, by_status_df, statuses, q_thresh, out_png,
                         dpi):
    """Pooled panel, plus one panel per status present (integrated runs only).

    A status absent from the data is never invented, and a single-status run
    gets a single panel rather than three empty ones. The pooled panel is
    accented in the motor-neuron teal, each status panel in its JCK status
    colour (control / sporadic / c9).
    """
    panels = [("Pooled", pooled_df, MN_COLOR)]
    for status in statuses:
        sub = (by_status_df[by_status_df["status"] == status]
               if by_status_df is not None and len(by_status_df) else None)
        panels.append((STATUS_TITLES.get(status, str(status)), sub,
                       STATUS_COLORS.get(status, NEUTRAL)))
    ncols = 1 if len(panels) == 1 else 2
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.0 * ncols, 4.5 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for ax, (title, df, accent) in zip(flat, panels):
        _enrichment_panel(ax, df, title, q_thresh, accent)
    for ax in flat[len(panels):]:
        ax.axis("off")
    plt.tight_layout()
    return save_png(fig, out_png, dpi)


def _niche_sort_key(series):
    """Numeric sort for Leiden labels ('10' after '9', not after '1')."""
    try:
        return series.astype(int)
    except Exception:
        return series.astype(str)


def niche_sort(values):
    vals = [str(v) for v in values]
    try:
        return sorted(vals, key=lambda x: int(x))
    except (TypeError, ValueError):
        return sorted(vals)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    started = datetime.now(timezone.utc)

    # Seeds. run_differential_gp_tests uses the GLOBAL numpy RNG internally
    # (np.random.choice, not a seeded Generator), so np.random.seed matters.
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Style BEFORE the first figure: apply_style pins pdf.fonttype=42, and
    # applying it later yields PDFs with blank text pages.
    if nature_style is not None:
        nature_style.apply_style()
    else:  # pragma: no cover
        matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                                    "svg.fonttype": "none"})
    matplotlib.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "savefig.dpi": args.dpi, "figure.dpi": args.dpi,
    })

    niche_col = f"latent_leiden_{args.leiden_res}"
    gates = Gates()
    written = []

    # ---------------------------------------------------------------- [0/12]
    section("[0/12] Startup")
    log(f"Host: {platform.node()}; python {platform.python_version()}; "
        f"SLURM job {os.environ.get('SLURM_JOB_ID', 'NA')}"
        + (f" task {os.environ['SLURM_ARRAY_TASK_ID']}"
           if "SLURM_ARRAY_TASK_ID" in os.environ else ""))
    from importlib.metadata import version as _pkg_version
    versions = {}
    for pkg in ("nichecompass", "scanpy", "anndata", "torch", "pandas",
                "numpy", "scipy", "scikit-learn", "statsmodels", "seaborn",
                "matplotlib", "igraph", "leidenalg"):
        try:
            versions[pkg] = _pkg_version(pkg)
            log(f"  {pkg} {versions[pkg]}")
        except Exception:
            versions[pkg] = None
            log(f"  {pkg} NOT INSTALLED")
    cuda_present = bool(torch.cuda.is_available())
    use_cuda = cuda_present and not args.no_cuda
    if cuda_present:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:  # pragma: no cover
            gpu_name, gpu_gb = "unknown", float("nan")
        log(f"CUDA available: {gpu_name} ({gpu_gb:.1f} GB)"
            + ("" if use_cuda else " -- overridden by --no-cuda, using CPU"))
    else:
        gpu_name, gpu_gb = None, None
        log("CUDA not available; running on CPU "
            "(neighbors/UMAP/Leiden are CPU-only regardless)")
    log(f"nature_style: {'imported' if nature_style is not None else 'ABSENT (local fallback)'}")
    # Honoured, never set: numba reads it at import (scanpy imports numba), so
    # it must come from the environment the sbatch exported before python began.
    numba_cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    log(f"NUMBA_CACHE_DIR = {numba_cache_dir if numba_cache_dir else '(unset: JIT is re-paid every run)'}")
    log(f"Palette: {PALETTE_SOURCE}; {MN_CLASS} {MN_COLOR}, {OTHER_NEURONS} "
        f"{OTHER_NEURONS_COLOR}, status {STATUS_COLORS}; retired Wong hexes "
        f"asserted absent")

    # Fail fast on the Leiden backend BEFORE the expensive steps
    have_igraph = have_leidenalg = False
    try:
        import igraph  # noqa: F401 - real import: find_spec passes on broken installs
        have_igraph = True
    except Exception:
        pass
    try:
        import leidenalg  # noqa: F401
        have_leidenalg = True
    except Exception:
        pass
    if not (have_igraph or have_leidenalg):
        raise RuntimeError(
            "Neither 'igraph' nor 'leidenalg' is importable: sc.tl.leiden "
            "cannot run. Fix with: PIP_ONLY_BINARY=:all: "
            "<nichecompass_env>/bin/pip install igraph leidenalg")
    log(f"Leiden backend: igraph={have_igraph}, leidenalg={have_leidenalg}")
    sc.settings.n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    log(f"sc.settings.n_jobs = {sc.settings.n_jobs}")

    # ---------------------------------------------------------------- [1/12]
    section("[1/12] Resolving the run directory and configuration")
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
    model_dir = os.path.join(run_dir, "model")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"No model/ inside {run_dir}. Point --run-dir at the directory "
            f"NC_v2_train.py wrote (it holds model/, figures/, results/).")
    figure_dir = ensure_dir(os.path.join(run_dir, "figures"))
    result_dir = ensure_dir(os.path.join(run_dir, "results"))

    manifest, manifest_fp = read_trainer_manifest(result_dir)
    config, config_src = infer_config(run_dir, manifest, args.config)
    is_integrated = config in INTEGRATED_CONFIGS
    is_vh_run = config in VH_CONFIGS
    run_tag = os.path.basename(run_dir) if not is_integrated else None
    adata_file_name, adata_src = resolve_adata_file_name(
        model_dir, manifest, args.adata_file_name)
    adata_path = os.path.join(model_dir, adata_file_name)

    per_sample_fig_dir = (ensure_dir(os.path.join(figure_dir, "per_sample"))
                          if is_integrated else figure_dir)
    per_sample_res_dir = (ensure_dir(os.path.join(result_dir, "per_sample"))
                          if is_integrated else result_dir)

    log(f"Run dir     : {run_dir}")
    log(f"Config      : {config} (from {config_src}) -> "
        f"{'INTEGRATED (multi-section)' if is_integrated else 'PER-SECTION'}, "
        f"{'VH subset' if is_vh_run else 'whole section'}")
    if run_tag:
        log(f"Section TAG : {run_tag} (run-dir basename)")
    log(f"Trained h5ad: {adata_path} (name from {adata_src})")
    log(f"Niche column: obs['{niche_col}']")
    log(f"Figures     : {figure_dir}")
    log(f"Results     : {result_dir}")

    # ---------------------------------------------------------------- [2/12]
    section("[2/12] Loading the trained NicheCompass model")
    # Imported only now: every cheap precondition above (a typo'd flag, a
    # missing run dir, an absent or ambiguous h5ad) has already been checked,
    # so a mistake costs a second instead of a 2.5-minute cold import off /oak.
    t0 = time.time()
    log("Importing nichecompass (cold off /oak: ~2.5 min) ...")
    load_nichecompass()
    log(f"nichecompass imported in {time.time() - t0:.1f}s; "
        f"create_new_color_dict "
        f"{'available' if create_new_color_dict is not None else 'ABSENT (tab20 fallback)'}")

    # NOTE the parameter name: load() takes use_cuda, not use_cuda_if_available.
    def _load(on_gpu):
        return NicheCompass.load(dir_path=model_dir,
                                 adata=None,
                                 adata_file_name=adata_file_name,
                                 use_cuda=on_gpu)

    t0 = time.time()
    cuda_fallback = None
    try:
        model = _load(use_cuda)
    except Exception as exc:
        if not use_cuda:
            raise
        # basemodelmixin.load() calls model.cuda() unconditionally when
        # use_cuda is set, so a small or already-busy GPU raises here -- AFTER
        # the training half of the job has succeeded. Falling back to CPU costs
        # minutes (the differential-GP forward pass took 110 s on 1.1M cells on
        # CPU on 2026-08-26); crashing costs the whole run.
        cuda_fallback = f"{type(exc).__name__}: {exc}"
        log(f"WARNING: loading the model onto {gpu_name} failed "
            f"({type(exc).__name__}: {str(exc).splitlines()[0]}). Falling "
            f"back to CPU -- only the differential-GP forward pass is "
            f"GPU-accelerated, so this costs minutes rather than losing a "
            f"finished training run. Pass --no-cuda to skip the attempt.")
        use_cuda = False
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass
        model = _load(False)
    adata = model.adata
    log(f"Loaded model + AnnData in {time.time() - t0:.1f}s on "
        f"{'GPU' if use_cuda else 'CPU'}: "
        f"{adata.n_obs:,} cells x {adata.n_vars:,} genes")
    cat_keys = getattr(model, "cat_covariates_keys_", "unknown")
    log(f"model.cat_covariates_keys_ = {cat_keys} "
        f"(None is how 0.3.3 spells 'no categorical covariate')")

    for key in (CELL_TYPE_KEY, ISMN_V2_COL, SAMPLE_KEY):
        if key not in adata.obs.columns:
            raise KeyError(
                f"Required column '{key}' missing from adata.obs. The v2 "
                f"trainer must carry cell_type, is_MN_v2, is_MN, is_MN_v1, "
                f"in_VH_v2, vh_v2_source, has_vh_v2, status and sample into "
                f"the saved h5ad.")
    if LATENT_KEY not in adata.obsm:
        raise KeyError(f"'{LATENT_KEY}' missing from adata.obsm -- was the "
                       f"model trained and saved with save_adata=True?")
    for key in (GP_NAMES_KEY, ACTIVE_GP_NAMES_KEY):
        if key not in adata.uns:
            raise KeyError(f"GP name key '{key}' missing from adata.uns.")
    have_spatial = SPATIAL_KEY in adata.obsm
    if not have_spatial:
        log(f"WARNING: obsm['{SPATIAL_KEY}'] absent -- spatial niche maps will "
            f"be skipped. (Never substitute 'spatial_grid_offset' or "
            f"'spatial_orig': they are different geometries.)")
    have_status = STATUS_KEY in adata.obs.columns
    have_in_vh = IN_VH_COL in adata.obs.columns

    for key in (SAMPLE_KEY, CELL_TYPE_KEY):
        if not isinstance(adata.obs[key].dtype, pd.CategoricalDtype):
            adata.obs[key] = adata.obs[key].astype("category")
    if have_status and not isinstance(adata.obs[STATUS_KEY].dtype,
                                      pd.CategoricalDtype):
        adata.obs[STATUS_KEY] = adata.obs[STATUS_KEY].astype("category")
    samples = [str(s) for s in adata.obs[SAMPLE_KEY].cat.categories
               if (adata.obs[SAMPLE_KEY] == s).any()]
    log(f"Samples ({len(samples)}): {samples}")
    if not is_integrated and len(samples) != 1:
        log(f"WARNING: config {config} is per-section but obs['{SAMPLE_KEY}'] "
            f"holds {len(samples)} labels. Continuing; the by-status "
            f"enrichment will switch on if more than one status is present.")
    log(f"Latent space: {adata.obsm[LATENT_KEY].shape[1]} dimensions "
        f"(= active GPs)")

    # ---- label contract --------------------------------------------------
    section("[2b/12] Label contract: cell_type_mn (motor neurons = is_MN_v2)")
    obs = adata.obs

    # is_MN_v2 == is_MN, NA-aware and bit-for-bit (the 2026-09-02
    # canonicalisation). A failure is recorded and the run finishes, then exits
    # non-zero: aborting here would throw away the GPU training that preceded
    # this step in the same SLURM job.
    if ISMN_COL in obs.columns:
        gates.check("ismn_v2_equals_ismn",
                    nullable_bool_equal(obs[ISMN_V2_COL], obs[ISMN_COL]),
                    f"obs['{ISMN_V2_COL}'] vs obs['{ISMN_COL}']: NA masks and "
                    f"values compared elementwise "
                    f"(v2 True={int(bool_mask(obs, ISMN_V2_COL).sum()):,}, "
                    f"<NA>={int(nullable_bool(obs, ISMN_V2_COL).isna().sum()):,})")
    else:
        gates.check("ismn_v2_equals_ismn", False,
                    f"obs['{ISMN_COL}'] is absent, so the canonicalisation "
                    f"contract cannot be verified; the trainer must carry it.")

    v1_summary = "absent"
    if ISMN_V1_COL in obs.columns:
        v1 = nullable_bool(obs, ISMN_V1_COL)
        n_v1_true = int(v1.fillna(False).sum())
        n_v1_na = int(v1.isna().sum())
        v1_summary = f"True={n_v1_true:,}, <NA>={n_v1_na:,}"
        # Deliberately NOT gated for equality with v2: SD02913BA is 'excluded'
        # (v2 all <NA>) yet carries 12 non-NA v1 True cells.
        log(f"obs['{ISMN_V1_COL}'] (superseded call): {v1_summary} -- reported "
            f"side by side, never asserted equal to v2.")

    cat, base, mn_mask, class_order = build_partition(obs)
    survivors = [c for c in class_order if c not in (OTHER_NEURONS, MN_CLASS)]
    n_mn = int(mn_mask.sum())
    n_mn_na = int(nullable_bool(obs, ISMN_V2_COL).isna().sum())
    # The run carries NO motor-neuron call at all (an 'excluded' section):
    # every MN surface below then says 'n/a', never a 0 posing as an observation.
    no_mn_call = (n_mn_na == adata.n_obs)
    if no_mn_call:
        log(f"NOTE: {NA_MN_NOTE} -- every motor-neuron count and annotation in "
            f"this run is written as 'n/a'.")
    log(f"cell_type_mn: {len(class_order)} declared classes "
        f"(the last two are '{OTHER_NEURONS}' and '{MN_CLASS}'): {class_order}")
    log(f"{MN_CLASS} = {n_mn:,} cells (is_MN_v2 True); "
        f"is_MN_v2 <NA> = {n_mn_na:,} cells "
        f"(NOT motor neurons: they keep their demoted base class, so every "
        f"motor-neuron count in this run is a FLOOR)")

    debits = pd.crosstab(pd.Series(base, name="base"),
                         pd.Series(mn_mask, name=ISMN_V2_COL))
    flagged = (debits[True] if True in debits.columns
               else pd.Series(dtype="int64"))
    debit_obs = {str(k): int(v) for k, v in flagged.items() if int(v) > 0}
    n_from_coarse = int(debit_obs.get(TRANSCRIPTIONAL_MN_CAT, 0))
    n_from_other = n_mn - n_from_coarse
    log(f"Promotions into '{MN_CLASS}': {n_from_coarse:,} from the coarse "
        f"transcriptional '{TRANSCRIPTIONAL_MN_CAT}' class, {n_from_other:,} "
        f"from other base classes {debit_obs}")
    n_demoted = int((base == TRANSCRIPTIONAL_MN_CAT).sum()) - n_from_coarse
    log(f"Demotions into '{OTHER_NEURONS}': {n_demoted:,} cells "
        f"(the coarse transcriptional class minus its promoted cells)")

    # Validate a trainer-written cell_type_mn against the recipe, then use the
    # locally recomputed one so the downstream is self-consistent either way.
    if CELL_TYPE_MN_KEY in obs.columns:
        stored = pd.Series(obs[CELL_TYPE_MN_KEY]).astype(str).to_numpy()
        recomputed = np.asarray(cat.astype(str))
        n_diff = int((stored != recomputed).sum())
        gates.check("cell_type_mn_matches_recipe", n_diff == 0,
                    f"the trainer's obs['{CELL_TYPE_MN_KEY}'] matches the "
                    f"recipe on all {len(stored):,} cells"
                    if n_diff == 0 else
                    f"{n_diff:,} cell(s) disagree with the recipe "
                    f"(is_MN_v2 wins over every base class; the coarse "
                    f"'{TRANSCRIPTIONAL_MN_CAT}' class becomes "
                    f"'{OTHER_NEURONS}')")
    else:
        gates.skip("cell_type_mn_matches_recipe",
                   f"obs['{CELL_TYPE_MN_KEY}'] was not in the saved h5ad; "
                   f"built here from cell_type + {ISMN_V2_COL}")
    adata.obs[CELL_TYPE_MN_KEY] = cat

    vc_class = pd.Series(cat).value_counts().reindex(class_order).fillna(0)
    vc_class = vc_class.astype("int64")
    gates.check("partition_tiles_cells",
                int(vc_class.sum()) == adata.n_obs
                and int(pd.isna(cat).sum()) == 0,
                f"the {len(class_order)} classes cover {int(vc_class.sum()):,} "
                f"of {adata.n_obs:,} cells with "
                f"{int(pd.isna(cat).sum())} unlabelled")
    gates.check("mn_class_equals_flag",
                int(vc_class[MN_CLASS]) == n_mn,
                f"'{MN_CLASS}' = {int(vc_class[MN_CLASS]):,} = "
                f"{ISMN_V2_COL}.fillna(False).sum() {n_mn:,}")
    gates.check("mn_debits_accounted",
                n_from_coarse + n_from_other == n_mn
                and sum(debit_obs.values()) == n_mn,
                f"per-base-class debits sum to {sum(debit_obs.values()):,} = "
                f"{MN_CLASS} total {n_mn:,} "
                f"({n_from_coarse:,} from the coarse class, "
                f"{n_from_other:,} from other classes)")
    gates.check("mn_name_not_transcriptional",
                int(vc_class[MN_CLASS]) == n_mn
                and int(vc_class[OTHER_NEURONS]) == n_demoted,
                f"'{MN_CLASS}' carries {int(vc_class[MN_CLASS]):,} "
                f"({ISMN_V2_COL}); the "
                f"{int((base == TRANSCRIPTIONAL_MN_CAT).sum()):,}-cell "
                f"transcriptional cluster appears only inside "
                f"'{OTHER_NEURONS}' ({int(vc_class[OTHER_NEURONS]):,} cells, "
                f"expected {n_demoted:,})")

    # Cross-check the partition against the TRAINER's own label_contract block.
    # The two halves compute it independently (the trainer per section before
    # concat, this script from the saved h5ad), so agreement is evidence the
    # model was saved with exactly these labels -- a content gate, not a
    # row-count one.
    tr_lc = manifest.get("label_contract") if isinstance(manifest, dict) else None
    if isinstance(tr_lc, dict) and isinstance(tr_lc.get("class_counts"), dict):
        observed_counts = {c: int(vc_class[c]) for c in class_order}
        trainer_counts = {str(k): int(v)
                          for k, v in tr_lc["class_counts"].items()}
        count_diffs = {k: (observed_counts.get(k), trainer_counts.get(k))
                       for k in set(observed_counts) | set(trainer_counts)
                       if observed_counts.get(k) != trainer_counts.get(k)}
        scalar_diffs = []
        for key, mine in (("n_promoted_from_coarse_MN", n_from_coarse),
                          ("n_promoted_from_other_classes", n_from_other),
                          ("n_demoted_to_other_neurons", n_demoted),
                          ("n_na_is_mn_v2", n_mn_na),
                          ("n_is_mn_v2_true", n_mn)):
            theirs = tr_lc.get(key)
            if isinstance(theirs, bool) or not isinstance(theirs, (int, float)):
                continue
            if int(theirs) != int(mine):
                scalar_diffs.append((key, int(mine), int(theirs)))
        gates.check("label_contract_matches_trainer",
                    not count_diffs and not scalar_diffs,
                    "the downstream partition reproduces the trainer's "
                    "label_contract exactly (10 class counts, promotions, "
                    "demotions and the <NA> count)"
                    if not (count_diffs or scalar_diffs)
                    else f"class-count diffs (downstream, trainer): "
                         f"{count_diffs}; scalar diffs (key, downstream, "
                         f"trainer): {scalar_diffs}")
    else:
        gates.skip("label_contract_matches_trainer",
                   "the trainer manifest carries no label_contract block")

    if have_in_vh:
        in_vh = nullable_bool(obs, IN_VH_COL)
        n_in_vh = int(in_vh.fillna(False).sum())
        n_in_vh_na = int(in_vh.isna().sum())
        log(f"obs['{IN_VH_COL}']: True={n_in_vh:,}, "
            f"False={adata.n_obs - n_in_vh - n_in_vh_na:,}, "
            f"<NA>={n_in_vh_na:,}")
        if is_vh_run:
            gates.check("vh_run_is_all_in_vh", n_in_vh == adata.n_obs,
                        f"config {config} subsets to the ventral horn: "
                        f"{n_in_vh:,} of {adata.n_obs:,} cells have "
                        f"{IN_VH_COL} == True")
        else:
            gates.skip("vh_run_is_all_in_vh",
                       f"config {config} is a whole-section run "
                       f"({n_in_vh:,} of {adata.n_obs:,} cells inside the "
                       f"ventral horn, {n_in_vh_na:,} <NA>)")
        # 0 is_MN_v2 True cells may legitimately lie outside in_VH_v2
        n_mn_outside = int((mn_mask & ~in_vh.fillna(False).to_numpy(dtype=bool)).sum())
        gates.check("mn_inside_vh", n_mn_outside == 0,
                    f"{n_mn_outside} motor neuron(s) lie outside "
                    f"{IN_VH_COL} == True (expected 0: the call is "
                    f"ventral-horn gated)")
    else:
        n_in_vh = n_in_vh_na = None
        log(f"WARNING: obs['{IN_VH_COL}'] absent -- the niche x ventral-horn "
            f"share table will be skipped.")
        gates.skip("vh_run_is_all_in_vh", f"obs['{IN_VH_COL}'] absent")
        gates.skip("mn_inside_vh", f"obs['{IN_VH_COL}'] absent")

    vh_sources = (sorted(str(s) for s in pd.unique(obs[VH_SOURCE_COL].astype(str)))
                  if VH_SOURCE_COL in obs.columns else [])
    if vh_sources:
        log(f"obs['{VH_SOURCE_COL}'] values present: {vh_sources}")

    # ---------------------------------------------------------------- [3/12]
    section("[3/12] Gene-program summary")
    n_all_gps = int(len(adata.uns[GP_NAMES_KEY]))
    active_gps = list(model.get_active_gps())
    log(f"Total GP names: {n_all_gps} (prior + add-on); active: {len(active_gps)}")
    gp_summary_df = model.get_gp_summary()
    fp = os.path.join(result_dir, "gp_summary_all_gps.csv")
    gp_summary_df.to_csv(fp, index=False)
    written.append(fp)
    log(f"Saved full GP summary ({gp_summary_df.shape[0]} rows): {fp}")

    # ---------------------------------------------------------------- [4/12]
    section("[4/12] Neighbors -> UMAP -> Leiden on the latent space")
    n_neighbors_eff = int(min(args.n_neighbors_latent,
                              max(2, adata.n_obs - 1)))
    if n_neighbors_eff != args.n_neighbors_latent:
        log(f"NOTE: clamped n_neighbors {args.n_neighbors_latent} -> "
            f"{n_neighbors_eff} for {adata.n_obs:,} cells.")
    t0 = time.time()
    log(f"sc.pp.neighbors(use_rep='{LATENT_KEY}', "
        f"n_neighbors={n_neighbors_eff}, n_jobs={sc.settings.n_jobs}) ...")
    sc.pp.neighbors(adata, use_rep=LATENT_KEY, n_neighbors=n_neighbors_eff,
                    random_state=args.seed)
    log(f"Neighbors done in {time.time() - t0:.1f}s. Computing UMAP ...")
    t0 = time.time()
    sc.tl.umap(adata, random_state=args.seed)
    log(f"UMAP done in {time.time() - t0:.1f}s. Computing Leiden "
        f"(resolution={args.leiden_res}) ...")
    t0 = time.time()
    if have_igraph:
        sc.tl.leiden(adata, resolution=args.leiden_res, key_added=niche_col,
                     flavor="igraph", n_iterations=2, directed=False,
                     random_state=args.seed)
    else:
        sc.tl.leiden(adata, resolution=args.leiden_res, key_added=niche_col,
                     flavor="leidenalg", random_state=args.seed)
    leiden_seconds = time.time() - t0

    # Numeric category order, so '10' sorts after '9' in every table and plot.
    niches = niche_sort(pd.unique(adata.obs[niche_col].astype(str)))
    if isinstance(adata.obs[niche_col].dtype, pd.CategoricalDtype):
        if set(adata.obs[niche_col].cat.categories.astype(str)) == set(niches):
            adata.obs[niche_col] = (adata.obs[niche_col]
                                    .cat.reorder_categories(niches))
        else:
            adata.obs[niche_col] = pd.Categorical(
                adata.obs[niche_col].astype(str), categories=niches)
    else:  # pragma: no cover
        adata.obs[niche_col] = pd.Categorical(
            adata.obs[niche_col].astype(str), categories=niches)
    n_niches = len(niches)
    log(f"Leiden done in {leiden_seconds:.1f}s: {n_niches} niche(s) in "
        f"obs['{niche_col}'] (numeric category order: {niches})")
    if n_niches < 2:
        log("NOTE: a single niche. Every one-vs-rest test has an empty "
            "comparison group and is skipped below; the composition surfaces "
            "still render as a single bar.")

    niche_sizes = (adata.obs[niche_col].value_counts()
                   .reindex(niches).fillna(0).astype("int64"))
    niche_sizes_df = pd.DataFrame({
        "n_cells": niche_sizes,
        "fraction": (niche_sizes / max(1, int(niche_sizes.sum()))).round(6),
    })
    niche_sizes_fp = os.path.join(result_dir, "niche_sizes.csv")
    niche_sizes_df.to_csv(niche_sizes_fp, index_label="niche")
    written.append(niche_sizes_fp)
    log(f"Saved niche sizes: {niche_sizes_fp}")

    # ---------------------------------------------------------------- [5/12]
    section("[5/12] Palettes (JCK palette for cell classes and status; tab20 "
            "modulo / create_new_color_dict for niches and samples)")
    niche_colors = build_color_dict(adata, niche_col)
    cell_type_colors = build_color_dict(adata, CELL_TYPE_KEY)
    sample_colors = build_color_dict(adata, SAMPLE_KEY)
    class_colors = partition_colors(adata, class_order, survivors)
    adata.uns[f"{CELL_TYPE_MN_KEY}_colors"] = [class_colors[c]
                                               for c in class_order]
    log(f"Palettes: {len(niche_colors)} niches, {len(cell_type_colors)} "
        f"cell_type, {len(sample_colors)} samples, {len(class_colors)} "
        f"cell_type_mn classes")

    # ---------------------------------------------------------------- [6/12]
    section("[6/12] Saving the Leiden-annotated AnnData beside the model")
    base_name, _ = os.path.splitext(adata_file_name)
    out_h5ad = os.path.join(model_dir, f"{base_name}_Leiden.h5ad")
    n_fixed = sanitize_uns_nones(adata.uns)
    if n_fixed:
        log(f"Sanitized {n_fixed} None value(s) in adata.uns before writing.")
    t0 = time.time()
    adata.write(out_h5ad, compression="gzip")
    written.append(out_h5ad)
    log(f"Saved {out_h5ad} in {time.time() - t0:.1f}s "
        f"({os.path.getsize(out_h5ad) / 1024**3:.2f} GB). It carries "
        f"{CELL_TYPE_MN_KEY}, {ISMN_V2_COL}, {niche_col} and X_umap. "
        f"GP activity scores are written to the CSVs below, not into this "
        f"file (the 2026-08-26 contract writes the h5ad before the tests).")

    # ---------------------------------------------------------------- [7/12]
    section("[7/12] Differential gene-program tests (one-vs-rest)")

    def run_dgp(cat_key, key_added, label):
        """run_differential_gp_tests with the mandatory single-category guard.

        With only one observed category the comparison group is empty and
        numpy raises "a must be greater than 0 unless no samples are taken"
        inside np.random.choice; an empty or tiny category is safe (unique()
        skips declared-but-empty levels, and the sampling is with replacement).
        """
        vc = adata.obs[cat_key].astype(str).value_counts()
        vc = vc[vc > 0]
        if len(vc) < 2:
            log(f"[skip] run_differential_gp_tests(cat_key='{cat_key}'): only "
                f"{len(vc)} observed category -- no comparison group.")
            return [], f"skipped: only {len(vc)} observed category"
        log(f"run_differential_gp_tests(cat_key='{cat_key}', "
            f"comparison_cats='rest', "
            f"log_bayes_factor_thresh={args.log_bayes_factor_thresh}) over "
            f"{adata.n_obs:,} cells on {'GPU' if use_cuda else 'CPU'} "
            f"({len(vc)} {label}) ...")
        t = time.time()
        enriched = model.run_differential_gp_tests(
            cat_key=cat_key,
            selected_cats=None,
            comparison_cats="rest",
            log_bayes_factor_thresh=args.log_bayes_factor_thresh,
            key_added=key_added,
            seed=args.seed,
        )
        log(f"  done in {time.time() - t:.1f}s: {len(enriched)} enriched GP(s) "
            f"across {len(vc)} {label}")
        return list(enriched), ""

    enriched_gps, dgp_niche_note = run_dgp(niche_col, DGP_KEY_NICHE, "niches")
    raw_fp = os.path.join(result_dir, "differential_gp_tests_raw.csv")
    raw_results = dgp_raw_frame(adata, DGP_KEY_NICHE)
    raw_results.to_csv(raw_fp, index=False)
    written.append(raw_fp)
    log(f"Saved raw niche differential-GP results ({raw_results.shape[0]} "
        f"rows): {raw_fp}")

    cols = [c for c in GP_TABLE_COLS if c in gp_summary_df.columns]
    missing_cols = sorted(set(GP_TABLE_COLS) - set(cols))
    if missing_cols:
        log(f"WARNING: gp_summary lacks {missing_cols}; exporting the "
            f"available columns only.")
    enriched_full_df = gp_summary_df[gp_summary_df["gp_name"].isin(enriched_gps)]
    enriched_filtered_df = enriched_full_df[
        ~enriched_full_df["gp_name"].str.startswith("Add-on")]
    for df, name in ((enriched_full_df, "gps_summary_enriched_full.csv"),
                     (enriched_filtered_df, "gps_summary_enriched_filtered.csv")):
        fp = os.path.join(result_dir, name)
        df.to_csv(fp, columns=cols, index=False)
        written.append(fp)
    log(f"Saved enriched GP summaries: {enriched_full_df.shape[0]} GPs "
        f"(full), {enriched_filtered_df.shape[0]} non-Add-on")

    dgp_class_note = "not requested (--skip-mn-gp-tests)"
    enriched_class_gps = []
    if not args.skip_mn_gp_tests:
        enriched_class_gps, dgp_class_note = run_dgp(
            CELL_TYPE_MN_KEY, DGP_KEY_CLASS, "cell classes")
        fp = os.path.join(result_dir,
                          "differential_gp_tests_cell_type_mn_raw.csv")
        raw_class = dgp_raw_frame(adata, DGP_KEY_CLASS)
        raw_class.to_csv(fp, index=False)
        written.append(fp)
        log(f"Saved raw cell_type_mn differential-GP results "
            f"({raw_class.shape[0]} rows): {fp}")
    else:
        log("Skipping the cell_type_mn differential-GP tests "
            "(--skip-mn-gp-tests).")

    # ---------------------------------------------------------------- [8/12]
    section("[8/12] Enriched GP activity by niche: table + MinMax heatmaps")
    if len(enriched_gps) == 0:
        log("WARNING: no enriched GPs at this threshold -- skipping the GP "
            "activity table and heatmaps (an expected outcome on a 480-gene "
            "panel and on tiny sections; consider a lower "
            "--log-bayes-factor-thresh).")
    else:
        enriched_present = [gp for gp in enriched_gps
                            if gp in adata.obs.columns]
        absent = sorted(set(enriched_gps) - set(enriched_present))
        if absent:
            log(f"WARNING: {len(absent)} enriched GP score column(s) missing "
                f"from adata.obs (name collision?): {absent[:5]} ...")
        if not enriched_present:
            log("WARNING: none of the enriched GP score columns reached "
                "adata.obs; skipping the activity table and heatmaps.")
        else:
            df_act = (adata.obs[[niche_col] + enriched_present]
                      .groupby(niche_col, observed=True)
                      .mean()
                      .reindex(niches))
            fp = os.path.join(result_dir,
                              "niche_gp_activity_mean_enriched.csv")
            df_act.to_csv(fp, index_label="niche")
            written.append(fp)
            log(f"Saved mean enriched GP activity by niche "
                f"({df_act.shape[0]} niches x {df_act.shape[1]} GPs): {fp}")
            written += plot_gp_heatmap(
                df_act, enriched_present,
                os.path.join(figure_dir, "heatmap_enriched_gps_full.png"),
                f"Enriched GPs by niche (full, n={len(enriched_present)})",
                args.dpi)
            non_addon = [gp for gp in enriched_present
                         if not gp.startswith("Add-on")]
            if non_addon:
                written += plot_gp_heatmap(
                    df_act, non_addon,
                    os.path.join(figure_dir,
                                 "heatmap_enriched_gps_filtered.png"),
                    f"Enriched GPs by niche (Add-on filtered, "
                    f"n={len(non_addon)})", args.dpi)
            else:
                log("WARNING: every enriched GP is an Add-on GP; the filtered "
                    "heatmap is skipped.")

    # ---------------------------------------------------------------- [9/12]
    section(f"[9/12] Niche x {CELL_TYPE_MN_KEY} composition "
            f"(motor neurons = {ISMN_V2_COL})")
    class_series = pd.Series(cat, index=adata.obs.index, name="cell_class")
    ct_counts = pd.crosstab(adata.obs[niche_col], class_series)
    # MANDATORY: pandas' crosstab drops unobserved categorical levels, so an
    # empty 'Motor neurons' class would vanish from the header without this.
    ct_counts = (ct_counts.reindex(index=niches, columns=class_order,
                                  fill_value=0).astype("int64"))
    ct_counts.index.name = niche_col
    ct_counts.columns.name = "cell_class"
    ct_props = ct_counts.div(ct_counts.sum(axis=1).replace(0, np.nan), axis=0)
    ct_props.index.name = niche_col
    ct_props.columns.name = "cell_class"

    for df, name in ((ct_counts, "niche_celltype_mn_counts.csv"),
                     (ct_props, "niche_celltype_mn_proportions.csv")):
        fp = os.path.join(result_dir, name)
        df.to_csv(fp)
        written.append(fp)
    log(f"Saved niche x {CELL_TYPE_MN_KEY} crosstabs "
        f"({ct_counts.shape[0]} niches x {ct_counts.shape[1]} classes; "
        f"all {len(class_order)} classes always in the header)")
    log("niche x cell class counts:\n" + ct_counts.to_string())

    row_sums = {str(i): int(v) for i, v in ct_counts.sum(axis=1).items()}
    size_map = {str(i): int(v) for i, v in niche_sizes.items()}
    mismatch = {k: (v, size_map.get(k)) for k, v in row_sums.items()
                if size_map.get(k) != v}
    gates.check("crosstab_row_sums", not mismatch,
                f"every row sum matches niche_sizes.csv for all "
                f"{len(row_sums)} niches"
                if not mismatch
                else f"row-sum mismatch (observed, niche_sizes): {mismatch}")
    gates.check("crosstab_total",
                int(ct_counts.to_numpy().sum()) == adata.n_obs,
                f"crosstab sums to {int(ct_counts.to_numpy().sum()):,} "
                f"(n_obs {adata.n_obs:,})")
    gates.check("proportion_row_sums",
                bool(np.allclose(np.nan_to_num(
                    ct_props.sum(axis=1).to_numpy(), nan=1.0), 1.0, atol=1e-9)),
                f"max |rowsum - 1| = "
                f"{float(np.nanmax(np.abs(ct_props.sum(axis=1).to_numpy() - 1.0))):.2e}")

    cohort = pd.DataFrame({
        "cell_class": class_order,
        "n_cells": [int(vc_class[c]) for c in class_order],
    })
    denom = max(1, adata.n_obs)
    cohort["pct"] = (100.0 * cohort["n_cells"] / denom).round(6)
    cohort["definition"] = [
        (f"cell_type == '{c}' minus its {ISMN_V2_COL}-flagged cells"
         if c in survivors else
         (f"cell_type == '{TRANSCRIPTIONAL_MN_CAT}' (coarse transcriptional "
          f"cluster) minus its {n_from_coarse} {ISMN_V2_COL}-flagged cells"
          if c == OTHER_NEURONS else
          f"{ISMN_V2_COL} == True (curated, ventral-horn-gated call); wins "
          f"over every base class"))
        for c in class_order
    ]
    # Explicit note on the Motor neurons row: 'n/a' where the section carries no
    # motor-neuron call at all, and a FLOOR warning wherever any <NA> remains.
    mn_notes = []
    if n_mn_na == adata.n_obs:
        mn_notes.append(NA_MN_NOTE)
    elif n_mn == 0:
        mn_notes.append(f"0 cells flagged {ISMN_V2_COL} == True in this run")
    if 0 < n_mn_na < adata.n_obs:
        mn_notes.append(f"{n_mn_na:,} cells have {ISMN_V2_COL} == <NA> and are "
                        f"NOT counted as motor neurons -- this count is a FLOOR")
    cohort["note"] = ["; ".join(mn_notes) if c == MN_CLASS else ""
                      for c in class_order]
    # Deliberately NOT the trainer's file name: NC_v2_train.py writes its own
    # results/cohort_composition_cell_type_mn.csv at training time. This is the
    # independently recomputed downstream copy, kept under a distinct name so
    # neither overwrites the other, and gated against the trainer's numbers
    # above (label_contract_matches_trainer).
    fp = os.path.join(result_dir,
                      "cohort_composition_cell_type_mn_downstream.csv")
    cohort.to_csv(fp, index=False)
    written.append(fp)
    log("cohort composition:\n"
        + cohort[["cell_class", "n_cells", "pct"]].to_string(index=False))

    written += stacked_composition(
        ct_props.fillna(0.0), ct_counts, class_colors,
        os.path.join(figure_dir, "niche_celltype_mn_composition_stacked"),
        f"Cell-class composition per niche "
        f"(motor neurons = {ISMN_V2_COL}) - {config}"
        + (f" {run_tag}" if run_tag else "")
        + (f" - {NO_CALL_FIG_NOTE}" if no_mn_call else ""),
        args.hero_dpi, xlabel=f"Niche ({niche_col})",
        highlight_note=NO_CALL_FIG_NOTE if no_mn_call else None)

    # ---------------------------------------------------------------- [10/12]
    section(f"[10/12] Niche x {ISMN_V2_COL} enrichment (Fisher exact, BH-FDR)")
    mn_status_col = f"{ISMN_V2_COL}_status"
    obs_small = pd.DataFrame({
        niche_col: adata.obs[niche_col].astype(str).to_numpy(),
        ISMN_V2_COL: nullable_bool(adata.obs, ISMN_V2_COL).to_numpy(),
        SAMPLE_KEY: adata.obs[SAMPLE_KEY].astype(str).to_numpy(),
    }, index=adata.obs.index)
    obs_small[ISMN_V2_COL] = obs_small[ISMN_V2_COL].astype("boolean")
    obs_small[STATUS_KEY] = (adata.obs[STATUS_KEY].astype(str).to_numpy()
                             if have_status else "unknown")
    obs_small[mn_status_col] = three_valued(obs_small, ISMN_V2_COL)

    mn_counts_df, mn_props_df = mn_crosstab(obs_small, niche_col, niches,
                                            three_valued_col=mn_status_col)
    for df, name in ((mn_counts_df, "niche_ismn_v2_counts.csv"),
                     (mn_props_df, "niche_ismn_v2_proportions.csv")):
        fp = os.path.join(result_dir, name)
        df.to_csv(fp, index=False)
        written.append(fp)
    log(f"Saved niche x {ISMN_V2_COL} counts and proportions "
        f"(True / False / Unknown kept separate; <NA> is never collapsed "
        f"into False)")

    pooled_enrich, pooled_note = niche_enrichment(
        obs_small, niche_col, ISMN_V2_COL, niches, args.q_thresh,
        args.min_mn_for_test, "pooled")
    fp = os.path.join(result_dir, "niche_ismn_v2_enrichment_pooled.csv")
    pooled_enrich.to_csv(fp, index=False)
    written.append(fp)
    n_sig = int(pooled_enrich["significant"].sum())
    n_tested_pooled = int(pooled_enrich["q"].notna().sum())
    log(f"Saved pooled enrichment: {fp} "
        f"({n_sig}/{n_tested_pooled} tested niches significant at "
        f"q<{args.q_thresh}; {n_niches - n_tested_pooled} of {n_niches} "
        f"untestable"
        + (f"; {pooled_note}" if pooled_note else "") + ")")

    statuses_present = [s for s in STATUS_ORDER
                        if have_status and (obs_small[STATUS_KEY] == s).any()]
    other_statuses = (sorted(set(pd.unique(obs_small[STATUS_KEY].dropna()))
                             - set(STATUS_ORDER)) if have_status else [])
    if other_statuses:
        log(f"WARNING: obs['{STATUS_KEY}'] carries values outside "
            f"{STATUS_ORDER}, skipped: {other_statuses}")
    by_status_frames = []
    per_stratum_n = {}
    if len(statuses_present) > 1:
        log(f"More than one status present {statuses_present}: running the "
            f"per-status enrichment, each stratum its own BH-FDR family.")
        for status in statuses_present:
            sub = obs_small[obs_small[STATUS_KEY] == status]
            per_stratum_n[status] = int(len(sub))
            enr, _ = niche_enrichment(sub, niche_col, ISMN_V2_COL, niches,
                                      args.q_thresh, args.min_mn_for_test,
                                      f"status={status}")
            enr.insert(0, "status", status)
            by_status_frames.append(enr)
            log(f"  [status={status}] {int(enr['significant'].sum())}/"
                f"{int(enr['q'].notna().sum())} tested niches significant at "
                f"q<{args.q_thresh} ({int(enr['q'].isna().sum())} untestable)")
    else:
        log(f"Only {len(statuses_present)} status present "
            f"({statuses_present or 'none'}): the per-status enrichment does "
            f"not apply to a {config} run and is written as an empty table.")
    by_status_df = (pd.concat(by_status_frames, ignore_index=True)
                    if by_status_frames
                    else pd.DataFrame(columns=["status"] + ENRICH_COLS))
    fp = os.path.join(result_dir, "niche_ismn_v2_enrichment_by_status.csv")
    by_status_df.to_csv(fp, index=False)
    written.append(fp)
    log(f"Saved per-status enrichment ({len(by_status_df)} rows): {fp}")

    # Per-sample x niche motor-neuron counts with an EXPLICIT call flag: a
    # section with no is_MN_v2 call reads 'n/a' in every cell, never 0.
    sn, no_call_by_sample = sample_niche_mn_table(
        obs_small, samples, niches, niche_col, mn_status_col)
    fp = os.path.join(result_dir, "niche_sample_ismn_v2_counts.csv")
    sn.reset_index().to_csv(fp, index=False)
    written.append(fp)
    n_no_call = sum(no_call_by_sample.values())
    log(f"Saved per-sample x niche {ISMN_V2_COL}==True counts: {fp} "
        f"({n_no_call} of {len(samples)} sections have no {ISMN_V2_COL} call "
        f"and are written as 'n/a')")

    written += plot_mn_composition(
        mn_counts_df, niches,
        os.path.join(figure_dir, "niche_ismn_v2_composition_stacked.png"),
        args.dpi, niche_col)
    written += plot_enrichment_grid(
        pooled_enrich, by_status_df,
        statuses_present if len(statuses_present) > 1 else [],
        args.q_thresh,
        os.path.join(figure_dir, "niche_ismn_v2_enrichment.png"), args.dpi)

    verdict = mn_verdict(pooled_enrich, args.q_thresh)
    log(f"MOTOR-NEURON NICHE VERDICT: {verdict}")

    # ---- niche x in_VH_v2 share ------------------------------------------
    vh_share_fp = None
    if have_in_vh:
        vh_status_col = f"{IN_VH_COL}_status"
        vh_small = pd.DataFrame({
            niche_col: adata.obs[niche_col].astype(str).to_numpy(),
            IN_VH_COL: nullable_bool(adata.obs, IN_VH_COL).to_numpy(),
        }, index=adata.obs.index)
        vh_small[IN_VH_COL] = vh_small[IN_VH_COL].astype("boolean")
        vh_small[vh_status_col] = three_valued(vh_small, IN_VH_COL)
        vh_counts, vh_props = mn_crosstab(vh_small, niche_col, niches,
                                          three_valued_col=vh_status_col)
        vh_props = vh_props.rename(columns={
            "n_true": "n_in_vh", "n_false": "n_outside_vh",
            "n_unknown": "n_vh_unknown",
            "frac_true_of_nonNA": "frac_in_vh_of_nonNA",
            "frac_false_of_nonNA": "frac_outside_vh_of_nonNA",
            "frac_unknown_of_total": "frac_vh_unknown_of_total"})
        vh_props["note"] = (
            f"config {config} subsets to {IN_VH_COL} == True, so the share is "
            f"1.0 by construction" if is_vh_run else
            (NA_MN_NOTE.replace(ISMN_V2_COL, IN_VH_COL)
             if int(vh_counts["n_unknown"].sum()) == adata.n_obs
             else "whole-section run: fraction of each niche inside the "
                  "ventral horn"))
        vh_share_fp = os.path.join(result_dir, "niche_in_vh_v2_share.csv")
        vh_props.to_csv(vh_share_fp, index=False)
        written.append(vh_share_fp)
        log(f"Saved niche x {IN_VH_COL} share: {vh_share_fp}")
        log("niche x ventral-horn share:\n"
            + vh_props[["niche", "n_in_vh", "n_outside_vh", "n_vh_unknown",
                        "n_total", "frac_in_vh_of_nonNA"]].to_string(index=False))
    else:
        log(f"Skipping the niche x {IN_VH_COL} share table (column absent).")

    # ---------------------------------------------------------------- [11/12]
    section("[11/12] Sample composition and spatial niche maps")
    if is_integrated and len(samples) > 1:
        ns_counts = pd.crosstab(adata.obs[niche_col], adata.obs[SAMPLE_KEY])
        ns_counts = ns_counts.reindex(index=niches, columns=samples,
                                      fill_value=0).astype("int64")
        fp = os.path.join(result_dir, "niche_sample_counts.csv")
        ns_counts.to_csv(fp)
        written.append(fp)
        sample_niche_props = pd.crosstab(adata.obs[SAMPLE_KEY],
                                         adata.obs[niche_col],
                                         normalize="index")
        sample_niche_props = sample_niche_props.reindex(
            index=samples, columns=niches, fill_value=0.0)
        sample_niche_props.index.name = SAMPLE_KEY
        sample_niche_props.columns.name = niche_col
        fp = os.path.join(result_dir, "sample_niche_proportions.csv")
        sample_niche_props.to_csv(fp)
        written.append(fp)
        log(f"Saved niche x sample tables ({len(samples)} samples)")
        written += stacked_bar_png(
            sample_niche_props, niche_colors,
            os.path.join(figure_dir, "sample_niche_composition_stacked.png"),
            "Niche composition per sample", xlabel="Sample", dpi=args.dpi)
    else:
        log("Single-section run: the niche x sample tables and the per-sample "
            "stacked bar do not apply.")

    # Sample metadata from a SMALL frame, never a groupby over the ~172-column
    # obs.
    meta_cols = [c for c in (STATUS_KEY, VH_SOURCE_COL, HAS_VH_COL,
                             "xmeta_tissue_region_runsheet",
                             "xmeta_disease_runsheet", "xmeta_run_batch",
                             "clin_sex") if c in adata.obs.columns]
    meta_frame = pd.DataFrame(
        {SAMPLE_KEY: adata.obs[SAMPLE_KEY].astype(str).to_numpy()})
    for c in meta_cols:
        meta_frame[c] = adata.obs[c].astype(str).to_numpy()
    meta_frame["_mn"] = mn_mask
    meta_frame["_mn_na"] = nullable_bool(adata.obs, ISMN_V2_COL).isna().to_numpy()
    if have_in_vh:
        meta_frame["_in_vh"] = bool_mask(adata.obs, IN_VH_COL)
    grouped = meta_frame.groupby(SAMPLE_KEY, observed=True)
    sample_meta = pd.DataFrame({"n_cells": grouped.size()})
    for c in meta_cols:
        sample_meta[c] = grouped[c].first()
    sample_meta[f"n_{ISMN_V2_COL}_true"] = grouped["_mn"].sum().astype("int64")
    sample_meta[f"n_{ISMN_V2_COL}_na"] = grouped["_mn_na"].sum().astype("int64")
    if have_in_vh:
        sample_meta[f"n_{IN_VH_COL}_true"] = grouped["_in_vh"].sum().astype("int64")
    # Explicit 'n/a' rather than a bare 0 where the section has no MN call:
    # the call flag AND the count column both say so.
    sample_no_call = (sample_meta[f"n_{ISMN_V2_COL}_na"]
                      == sample_meta["n_cells"]).to_numpy()
    sample_meta[f"{ISMN_V2_COL}_call"] = np.where(
        sample_no_call, NO_CALL_LABEL, CALLED_LABEL)
    sample_meta[f"n_{ISMN_V2_COL}_true"] = np.where(
        sample_no_call, "n/a",
        sample_meta[f"n_{ISMN_V2_COL}_true"].astype(str)).astype(object)
    sample_meta = sample_meta.reindex(samples)
    fp = os.path.join(result_dir, "sample_metadata.csv")
    sample_meta.to_csv(fp, index_label=SAMPLE_KEY)
    written.append(fp)
    log(f"Saved sample metadata ({['n_cells'] + meta_cols}): {fp}")
    log("sample metadata:\n" + sample_meta.to_string())

    if have_spatial:
        coords_all = np.asarray(adata.obsm[SPATIAL_KEY], dtype=np.float64)
        if coords_all.shape[0] != adata.n_obs or coords_all.shape[1] < 2:
            log(f"WARNING: obsm['{SPATIAL_KEY}'] has shape "
                f"{coords_all.shape}; skipping the spatial maps.")
            have_spatial = False
    if have_spatial:
        niche_codes_all = adata.obs[niche_col].cat.codes.to_numpy()
        class_codes_all = np.asarray(cat.codes)
        sample_vec = adata.obs[SAMPLE_KEY].astype(str).to_numpy()
        status_vec = (adata.obs[STATUS_KEY].astype(str).to_numpy()
                      if have_status else np.full(adata.n_obs, "NA"))
        for i, s in enumerate(samples, start=1):
            t0 = time.time()
            m = sample_vec == s
            n_sub = int(m.sum())
            if n_sub == 0:  # pragma: no cover
                continue
            status = str(status_vec[m][0])
            s_pt = point_size_for(n_sub)
            stem = s if is_integrated else f"{run_tag or s}"
            written += dense_scatter(
                coords_all[m][:, :2], niche_codes_all[m], niches, niche_colors,
                os.path.join(per_sample_fig_dir, f"{stem}_niches_spatial"),
                f"Niches ({niche_col}): {s} ({status}, n={n_sub:,})",
                args.dpi, args.seed, s_pt,
                xlabel="x (um)", ylabel="y (um)", equal_aspect=True,
                invert_y=True, despine=False)
            s_no_call = bool(no_call_by_sample.get(s, False))
            written += dense_scatter(
                coords_all[m][:, :2], class_codes_all[m], class_order,
                class_colors,
                os.path.join(per_sample_fig_dir, f"{stem}_celltype_mn_spatial"),
                f"Cell classes (motor neurons = {ISMN_V2_COL}): {s} "
                f"({status}, n={n_sub:,})"
                + (f" - {NO_CALL_FIG_NOTE}" if s_no_call else ""),
                args.dpi, args.seed, s_pt, highlight=MN_CLASS,
                s_hi=max(18.0, 3.5 * s_pt),
                xlabel="x (um)", ylabel="y (um)", equal_aspect=True,
                invert_y=True, despine=False,
                highlight_note=NO_CALL_FIG_NOTE if s_no_call else None)
            vc = (pd.Series(adata.obs[niche_col].astype(str).to_numpy()[m])
                  .value_counts().reindex(niches).fillna(0).astype("int64"))
            fp = os.path.join(per_sample_res_dir, f"{stem}_niche_counts.csv")
            vc.rename("n_cells").to_csv(fp, index_label="niche")
            written.append(fp)
            log(f"[{i}/{len(samples)}] {s}: 2 spatial maps + niche counts "
                f"({time.time() - t0:.1f}s)")
    else:
        log("Spatial niche maps skipped.")

    # ---------------------------------------------------------------- [12/12]
    section("[12/12] Dense latent UMAPs")
    if UMAP_KEY not in adata.obsm:  # pragma: no cover
        log(f"WARNING: obsm['{UMAP_KEY}'] absent; UMAP figures skipped.")
    else:
        um = np.asarray(adata.obsm[UMAP_KEY], dtype=np.float64)[:, :2]
        if um.shape[0] != adata.n_obs:  # pragma: no cover
            raise SystemExit(f"obsm['{UMAP_KEY}'] has {um.shape[0]} rows, obs "
                             f"has {adata.n_obs}")
        s_bg = point_size_for(adata.n_obs)
        s_mn = max(14.0, 3.5 * s_bg)
        log(f"Dense-render point sizes for {adata.n_obs:,} cells: "
            f"background s={s_bg}, {MN_CLASS.lower()} s={s_mn} "
            f"(seed-{args.seed} shuffled draw order, alpha 0.9)")
        written += dense_scatter(
            um, adata.obs[niche_col].cat.codes.to_numpy(), niches,
            niche_colors, os.path.join(figure_dir, "umap_niches"),
            f"NicheCompass latent UMAP: niches ({niche_col}) - {config}"
            + (f" {run_tag}" if run_tag else ""),
            args.dpi, args.seed, s_bg)
        written += dense_scatter(
            um, np.asarray(cat.codes), class_order, class_colors,
            os.path.join(figure_dir, "umap_celltype_mn"),
            f"NicheCompass latent UMAP: cell classes "
            f"(motor neurons = {ISMN_V2_COL}) - {config}"
            + (f" {run_tag}" if run_tag else "")
            + (f" - {NO_CALL_FIG_NOTE}" if no_mn_call else ""),
            args.dpi, args.seed, s_bg, highlight=MN_CLASS, s_hi=s_mn,
            hero_dpi=args.hero_dpi,
            highlight_note=NO_CALL_FIG_NOTE if no_mn_call else None)
        if is_integrated and len(samples) > 1:
            written += dense_scatter(
                um, adata.obs[SAMPLE_KEY].cat.codes.to_numpy(),
                list(adata.obs[SAMPLE_KEY].cat.categories), sample_colors,
                os.path.join(figure_dir, "umap_samples"),
                f"NicheCompass latent UMAP: sample - {config}",
                args.dpi, args.seed, s_bg)

    # ---- output gate ------------------------------------------------------
    section("Gates and manifest")
    run_root = os.path.realpath(run_dir)
    strays = [p for p in written
              if not os.path.realpath(p).startswith(run_root + os.sep)]
    gates.check("outputs_inside_run_dir", not strays,
                f"all {len(written)} outputs live under {run_dir}"
                if not strays else f"stray writes outside the run dir: {strays}")
    absent = [p for p in written if not os.path.isfile(p)]
    gates.check("outputs_present", not absent,
                f"all {len(written)} declared outputs exist on disk"
                if not absent else f"missing: {absent}")

    # ---- manifest ---------------------------------------------------------
    downstream = {
        "script": os.path.abspath(__file__),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t_start, 1),
        "host": platform.node(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "numba_cache_dir": numba_cache_dir,
        "palette": {
            "source": PALETTE_SOURCE,
            "cell_type_mn": class_colors,
            "status": STATUS_COLORS,
            "wdr49_astrocytes": WDR49_COLOR,
            "neutral": NEUTRAL,
            "legacy_hex_never_drawn": list(LEGACY_HEX),
            "niche_and_sample_colours": "categorical (tab20 / create_new_color_dict)",
        },
        "versions": versions,
        "args": vars(args),
        "config": config,
        "config_source": config_src,
        "is_integrated": is_integrated,
        "is_vh_run": is_vh_run,
        "run_tag": run_tag,
        "run_dir": run_dir,
        "input_adata": adata_path,
        "input_adata_name_source": adata_src,
        "annotated_h5ad": out_h5ad,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "samples": samples,
        "statuses_present": statuses_present,
        "statuses_skipped": other_statuses,
        "per_stratum_n_cells": per_stratum_n,
        "used_cuda": use_cuda,
        "cuda_present": cuda_present,
        "gpu": gpu_name,
        "gpu_total_gb": (round(gpu_gb, 1) if isinstance(gpu_gb, float)
                         and not math.isnan(gpu_gb) else None),
        "cuda_load_fallback": cuda_fallback,
        "leiden": {
            "resolution": args.leiden_res,
            "niche_col": niche_col,
            "flavor": "igraph" if have_igraph else "leidenalg",
            "n_neighbors_latent": n_neighbors_eff,
            "n_neighbors_requested": args.n_neighbors_latent,
            "seconds": round(leiden_seconds, 1),
            "n_niches": n_niches,
            "niches": niches,
            "niche_sizes": size_map,
        },
        "gene_programs": {
            "n_gp_names": n_all_gps,
            "n_active_gps": len(active_gps),
            "log_bayes_factor_thresh": args.log_bayes_factor_thresh,
            "n_enriched_gps_by_niche": len(enriched_gps),
            "enriched_gps_by_niche": list(enriched_gps),
            "niche_test_note": dgp_niche_note,
            "n_enriched_gps_by_cell_class": len(enriched_class_gps),
            "enriched_gps_by_cell_class": list(enriched_class_gps),
            "cell_class_test_note": dgp_class_note,
        },
        "label_contract": {
            "rule": (f"obs['{CELL_TYPE_MN_KEY}'] = obs['{CELL_TYPE_KEY}'] with "
                     f"'{TRANSCRIPTIONAL_MN_CAT}' renamed '{OTHER_NEURONS}', "
                     f"then {ISMN_V2_COL} == True -> '{MN_CLASS}', winning "
                     f"over every base class. Mutually exclusive and "
                     f"exhaustive."),
            "class_order": class_order,
            "class_totals": {c: int(vc_class[c]) for c in class_order},
            "grand_total": int(vc_class.sum()),
            "n_motor_neurons": n_mn,
            "n_promoted_from_coarse_MN": n_from_coarse,
            "n_promoted_from_other_classes": n_from_other,
            "promotion_debits_by_base_class": debit_obs,
            "n_demoted_to_other_neurons": n_demoted,
            "n_is_MN_v2_na": n_mn_na,
            "is_MN_v2_na_handling": (
                f"<NA> cells are NOT '{MN_CLASS}': they keep their demoted "
                f"base class, so every motor-neuron count in this run is a "
                f"FLOOR and a zero contribution is by construction, not by "
                f"observation."),
            "is_MN_v1_summary": v1_summary,
            "in_VH_v2_true": n_in_vh,
            "in_VH_v2_na": n_in_vh_na,
            "vh_v2_source_values": vh_sources,
            "colors": class_colors,
            "naming_rule": (f"'{MN_CLASS}' denotes {ISMN_V2_COL} == True only. "
                            f"The coarse transcriptional cluster is never "
                            f"presented under that name anywhere."),
        },
        "mn_niche_verdict": verdict,
        "mn_enrichment": {
            "test": "Fisher exact, alternative='greater', BH-FDR per family",
            "q_thresh": args.q_thresh,
            "min_mn_for_test": args.min_mn_for_test,
            "pooled_note": pooled_note,
            "n_significant_niches_pooled": n_sig,
            "n_tests_in_bh_family_pooled": n_tested_pooled,
            "n_untestable_niches_pooled": int(n_niches - n_tested_pooled),
            "by_status_ran": len(statuses_present) > 1,
            "na_handling": ("three-valued True/False/Unknown in the crosstabs; "
                            "<NA> cells are dropped from the test, never "
                            "counted as non-MN"),
            "degenerate_table_handling": (
                "a niche whose 2x2 table has an empty margin (no non-<NA> "
                "cell inside or outside it in the stratum) is never handed to "
                "fisher_exact: odds_ratio = p = q = NaN with a note, excluded "
                "from the BH family; n_tests_in_bh_family is written per row"),
            "no_call_sections": [s for s, v in no_call_by_sample.items() if v],
        },
        "gates": gates.as_dict(),
        "gates_summary": gates.summary(),
        "all_gates_passed": gates.all_passed,
        "outputs": [os.path.relpath(p, run_dir) for p in written],
    }

    fp = os.path.join(result_dir, "run_info.json")
    atomic_write_json(downstream, fp)
    log(f"Saved downstream run info: {fp}")

    manifest_out = dict(manifest) if manifest else {
        "note": ("The trainer manifest was absent or unreadable when the "
                 "downstream ran; this file holds the downstream block only."),
    }
    manifest_out["downstream"] = downstream
    atomic_write_json(manifest_out, manifest_fp)
    log(f"Appended the 'downstream' block to {manifest_fp}")

    # ---- final --------------------------------------------------------- #
    n_figs = sum(len(files) for _, _, files in os.walk(figure_dir))
    n_res = sum(len(files) for _, _, files in os.walk(result_dir))
    print("=" * 78, flush=True)
    log(f"GATE SUMMARY: {gates.summary()}")
    log(f"config        : {config}"
        + (f" | section {run_tag}" if run_tag else ""))
    log(f"niches        : {n_niches} ({niches})")
    log(f"enriched GPs  : {len(enriched_gps)} by niche, "
        f"{len(enriched_class_gps)} by cell class")
    log(f"{MN_CLASS:<14}: {n_mn:,} cells ({ISMN_V2_COL}); "
        f"{n_mn_na:,} <NA>")
    log(f"MN verdict    : top niche {verdict['top_niche']}, "
        f"OR {verdict['odds_ratio']}, q {verdict['q']}, "
        f"significant {verdict['significant']}"
        + (f" | {verdict['note']}" if not verdict["tested"] else ""))
    log(f"annotated h5ad: {out_h5ad}")
    log(f"figures       : {n_figs} files under {figure_dir}")
    log(f"results       : {n_res} files under {result_dir}")
    log(f"manifest      : {manifest_fp}")
    log(f"ALL DONE in {(time.time() - t_start) / 60:.1f} min "
        f"({'gates all passed' if gates.all_passed else 'GATES FAILED'}).")
    print("=" * 78, flush=True)

    if not gates.all_passed:
        log(f"EXIT 1: {gates.summary()} -- every output above was still "
            f"written; nothing from the training half is lost.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
