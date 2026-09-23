#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/NC_v2_train.py
#
# The trainer behind both the 20260902 base run and the 20260914 v3 retrain. --exclude-cells
# was added for the retrain. build_cell_type_mn() lives here: the coarse Motor neurons class
# becomes Other neurons, then any cell with is_MN_v2 becomes Motor neurons.
#
# Retrain settings: spatial kNN with n_neighbors 4, symmetrised and block-diagonal per sample.
# NicheNet plus MEBOCOST programmes, 77 surviving the gates (OmniPath extraction fails in
# 0.3.3). Sample as a categorical covariate. GATv2 with 4 heads and 100 add-on programmes.
# Up to 400 epochs with patience 8. The retrain stopped at 49 epochs, best epoch 41, H200.
# ========================================================================================

"""
NC_v2_train.py — generalised NicheCompass trainer for the four v2 configurations
of the QCed 20-sample ALS spinal-cord Xenium cohort (480-gene panel) on SCG.

Generalised from NC_SC_training_all.py (the 2026-08-26 integrated trainer, job
52418732) — every proven mechanic is carried over verbatim; only the things that
MUST vary per configuration were parameterised:

    C1 persample_whole   one model per section, whole section        (20 sections)
    C2 persample_vh      one model per section, in_VH_v2 subset only (17 sections)
    C3 integrated_whole  ONE model, 20 whole sections, cat cov ['sample'] (20 cats)
    C4 integrated_vh     ONE model, in_VH_v2 cells of 17 sections     (17 cats)

Carried over unchanged from the reference trainer:
  * per-sample sklearn kneighbors_graph(n_neighbors=4, mode='connectivity',
    include_self=False) then A.maximum(A.T) — squidpy is absent from the env;
  * scipy.sparse.block_diag assembly for the integrated configurations, guarded
    by the four alignment gates (sample labels, obs_names, spatial rows, and the
    adjacency's own shape/nnz/symmetry/min-degree);
  * counts CSC -> CSR plus the integer-valued provenance check (X in the pass2
    files is log1p — NicheCompass must NEVER fall back to it);
  * EXPECTED_N_GENES = 480 hard abort, per file and post-concat;
  * cache-aware GP extractors with os.chdir(gene_programs/) beforehand, the
    extract_or_warn tolerance wrapper, filter_and_combine_gp_dict_gps_v2, and
    add_gps_from_gp_dict_to_adata(min_genes_per_gp=2, min_source=1, min_target=1)
    followed by the NC_MIN_GPS survival gate;
  * the Sanger/H200-proven hyperparameters (gatv2conv, active_gp_thresh_ratio
    0.01, n_epochs 400 ceiling, n_epochs_all_gps 25, lr 0.001,
    lambda_edge_recon 500000., lambda_gene_expr_recon 300., lambda_l1_masked 0.,
    lambda_l1_addon 30., n_sampled_neighbors 4, species human, counts 'counts');
  * NC_EDGE_BATCH / NC_MIN_GPS / NC_ALLOW_CPU environment hooks and the
    "no GPU visible" hard failure.

Deliberate changes vs the reference (all recorded in the run manifest):
  * NO 'latest' symlink. The v2 tree uses a fixed runstamp and the script
    refuses to write anywhere under a path containing '/artifacts/', so
    artifacts/sample_integration/latest (which four existing scripts default to)
    can never be repointed by this trainer. The gate is enforced on RESOLVED
    paths (symlinks and '..' followed) and --tag / --runstamp are validated as
    a strict file TAG (SD + 5 digits + 2 capitals) and a single path-safe
    component, so neither can carry a traversing path into the output tree.
  * EXPECTED_N_SAMPLES is derived from the tag manifest (1 for C1/C2), never
    hardcoded; EXPECTED_N_CELLS is a warn-only cross-check against the manifest.
  * node_batch_size is ALWAYS passed explicitly, which skips the auto formula
    at nichecompass/train/trainer.py:194-196 (a ZeroDivisionError whenever the
    graph's mean degree drops below ~2.2) and makes the Trainer's "Node batch
    size:" banner truthful.
  * edge_batch_size is capped so that a tiny section still gets >= 8 gradient
    steps per epoch; NC_EDGE_BATCH remains the ceiling. The H200 sbatch files
    export 4096 (the Sanger-proven point, at which every input above ~37k
    undirected edges trains); unset, the default is 2048 so the same script is
    usable on a smaller card (e.g. an 8 GB qrtx4000) without edits. Nothing
    else in this script is specific to a GPU model.
  * early_stopping_kwargs are always passed explicitly and then verified back
    off model.trainer.early_stopping — Trainer.__init__ swallows unknown kwargs
    silently, so a typo would otherwise be invisible. The verification is
    NON-FATAL: a mismatch is a banner warning recorded in the manifest
    (training.early_stopping_mismatch), never a reason to discard a finished
    run before model.save(). Integrated runs use the 0.3.3 defaults (patience
    8 / lr_patience 4) so C3 reproduces the 08-26 training regime exactly;
    per-sample runs use patience 30 > n_epochs_all_gps so a tiny section can
    never early-stop before the active-GP-only phase.
  * obs is slimmed to the ten label-contract columns and obsp/uns/varm/varp are
    cleared per section: the pass2 files carry 171 obs columns, 8 obsp graphs
    and 54 uns entries, and on the per-sample path (no ad.concat to drop them)
    the stale uns['neighbors'] would otherwise be reused by a downstream
    sc.tl.umap and the invalid obsp graphs would ride into the saved h5ad. The
    obsp/uns/varm/varp clearing runs BEFORE the VH subset copy so the copy
    never re-slices eight ~100k-node graphs it is about to delete.
  * The GP cache is REQUIRED by default (NC_REQUIRE_GP_CACHE=0 to relax): with
    up to 6 concurrent array tasks a cache miss would mean concurrent live
    fetches all writing the same CSV.
  * A finished run (results/run_manifest.json present AND parseable, which is
    written last and ATOMICALLY via tmp + os.replace) is SKIPPED with exit 0
    rather than retrained. The fixed runstamp makes collisions real, and the
    C1/C2 arrays process several sections per task — so a task that dies on
    section 3 of 4 can simply be resubmitted. A manifest write failure is
    FATAL (non-zero exit), so the wrapper's path gate always reflects reality.
    Pass --overwrite to force a retrain.
  * Output directories are created only AFTER the tag manifest is loaded and
    --tag is accepted (a member of the right list, not an excluded section), so
    a refused run leaves no empty directory behind.

THE LABEL CONTRACT (the point of the exercise)
    obs['cell_type_mn'] = obs['cell_type'] with the coarse transcriptional class
    "Motor neurons" renamed "Other neurons", after which every cell with
    is_MN_v2 == True becomes "Motor neurons" (the flag wins over every base
    class). Ten mutually exclusive classes tile every cell, declared in a FIXED
    order with "Motor neurons" LAST, so an excluded section still emits a
    zero-count "Motor neurons" level instead of dropping the column from every
    downstream crosstab. is_MN_v2 <NA> cells simply keep their demoted base
    class; the count is recorded per run and per section in the manifest.
    NicheCompass never consumes labels (counts + graph + GP mask only), so
    cell_type_mn exists purely to drive the downstream surfaces.

TAG MANIFEST SCHEMA (written by the preflight; the ONLY source of truth for
array-index -> TAG mapping, so no tag list is hardcoded here):
    {
      "tags_all20": ["SD00614BG", ...],          # 20 file TAGs, whole-section runs
      "tags_vh17":  ["SD00614BG", ...],          # 17 TAGs with a VH annotation
      "excluded":   ["SD01015BG", "SD01616BI", "SD02913BA"],   # optional
      "per_tag": {"SD00614BG": {"sample": "SD00614_BG", "status": "control",
                                "n_obs": 50432, "n_in_vh_v2": 6493,
                                "n_is_mn_v2": 77, "vh_v2_source": "manual"}, ...}
    }
    A handful of key aliases are accepted (tags_all/tags_whole, tags_vh,
    tags_excluded, per_sample/tags) and the resolved key is logged.
    Consistency rule: 'excluded' and 'tags_vh17' must be DISJOINT and both
    SUBSETS of 'tags_all20' (hard failures). A tag of tags_all20 in neither
    list is WARNED about by name (it stays trainable as a whole section, and
    is simply never a VH section) — the two lists need not be exact complements.

Inputs : <pass2-dir>/ALS_SCXenium_<TAG>_pass2.h5ad
Outputs: <runs-base>/<config>/<runstamp>[/<TAG>]/{model,figures,results}
         model/NC_v2_<config>[_<TAG>].h5ad + results/run_manifest.json

Examples
    # smoke test the tiny end of C2 first (931 VH cells), CPU-tolerant
    NC_ALLOW_CPU=1 python NC_v2_train.py --config persample_vh --tag SD01923BI \
        --manifest .../tags_20260902_ncv2.json --n-epochs 6
    python NC_v2_train.py --config integrated_whole \
        --manifest .../tags_20260902_ncv2.json
"""

# --------------------------------------------------------------------------- #
# Environment guards — BEFORE any heavy import                                 #
# --------------------------------------------------------------------------- #
import os

# Must be set before torch initialises CUDA; the sbatch wrapper exports it too
# (setdefault keeps whatever the job environment already provides).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")  # headless: MUST precede any pyplot import

import argparse
import gc
import glob
import hashlib
import json
import random
import re
import shlex
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.neighbors import kneighbors_graph

from nichecompass.models import NicheCompass
from nichecompass.utils import (
    add_gps_from_gp_dict_to_adata,
    extract_gp_dict_from_mebocost_ms_interactions,  # 0.3.3 name (0.2.3: *_es_*)
    extract_gp_dict_from_nichenet_lrt_interactions,
    extract_gp_dict_from_omnipath_lr_interactions,
    filter_and_combine_gp_dict_gps_v2,
)

warnings.filterwarnings("ignore")
plt.rcParams["savefig.dpi"] = 150  # PNG @ 150 dpi everywhere

SCRIPT_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def section(title: str) -> None:
    print("", flush=True)
    log("#" * 78)
    log(f"### {title}")
    log("#" * 78)


def warn(msg: str) -> None:
    log(f"WARNING: {msg}")


def banner_warn(lines) -> None:
    log("!" * 78)
    for line in lines:
        log(f"!!! {line}")
    log("!" * 78)


# =========================================================================== #
# CONFIG BLOCK — everything that does NOT vary per configuration              #
# =========================================================================== #

### Dataset and Model Parameters ###
species = "human"
spatial_key = "spatial"          # obsm['spatial'] = per-section micron coords.
                                 # The pass2 files also carry 'spatial_orig' and
                                 # 'spatial_grid_offset' (cohort-plot grid) —
                                 # both are dropped, never used for the kNN.
n_neighbors = 4

# AnnData keys and GP keys
counts_key = "counts"               # layers['counts'] — integer-valued (X is log1p!)
adj_key = "spatial_connectivities"  # stored in obsp ONLY (0.3.3 contract)
gp_names_key = "nichecompass_gp_names"
active_gp_names_key = "nichecompass_active_gp_names"
gp_targets_mask_key = "nichecompass_gp_targets"
gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
gp_sources_mask_key = "nichecompass_gp_sources"
gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
latent_key = "nichecompass_latent"

# Architecture — every value below equals the nichecompass 0.3.3 default AND the
# Sanger-proven configuration. They are passed explicitly (the reference trainer
# relied on the defaults) so the manifest is complete and a future default drift
# in the library cannot silently change the model.
cat_covariates_embeds_injection = ["gene_expr_decoder"]
conv_layer_encoder = "gatv2conv"
active_gp_thresh_ratio = 0.01
active_gp_type = "separate"
node_label_method = "one-hop-norm"
gene_expr_recon_dist = "nb"
n_addon_gp = 100
n_fc_layers_encoder = 1
n_layers_encoder = 1
encoder_n_attention_heads = 4
log_variational = True
include_edge_kl_loss = True

# Trainer (Sanger/H200-proven)
n_epochs_all_gps = 25
lr = 0.001
lambda_edge_recon = 500000.
lambda_gene_expr_recon = 300.
lambda_l1_masked = 0.   # prior GP regularisation
lambda_l1_addon = 30.   # de novo GP regularisation
n_sampled_neighbors = 4
edge_val_ratio = 0.1    # keep >0: with BOTH val ratios >0 the early-stopping
node_val_ratio = 0.1    # metric auto-selects 'val_global_loss'
latent_dtype = np.float64   # float16 is only for >1M obs; the 08-26 1.1M-cell
                            # run used float64 and peaked at 10.6 GiB RSS
# NOTE: nichecompass 0.3.3 has NO dataloader num_workers plumbing — passing
# num_workers would TypeError inside Trainer. SLURM --cpus-per-task still feeds
# BLAS/OMP and the sklearn kNN below (n_jobs=-1).

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """A positive-integer environment hook; a malformed value is a hard abort
    (a silently ignored NC_EDGE_BATCH=abc would train at the wrong size)."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        val = int(raw.strip())
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not an integer") from None
    if val < minimum:
        raise SystemExit(f"{name}={val} must be >= {minimum}")
    return val


# Edge-batch ceiling — driven ONLY by NC_EDGE_BATCH. The H200 sbatch files
# export 4096 (the Sanger-proven point); unset, the default is 2048 so the
# trainer runs as-is on a smaller card (e.g. an 8 GB qrtx4000). No other value
# in this script is specific to a GPU model.
MIN_EDGE_BATCH = 256          # floor for tiny sections
MIN_STEPS_PER_EPOCH = 8       # a 1-batch epoch is 1 gradient step — statistically useless
EDGE_BATCH_DEFAULT = 2048
EDGE_BATCH_CEILING = _env_int("NC_EDGE_BATCH", EDGE_BATCH_DEFAULT,
                              minimum=MIN_EDGE_BATCH)
EDGE_BATCH_CEILING_SOURCE = ("NC_EDGE_BATCH" if os.environ.get("NC_EDGE_BATCH")
                             else "default (NC_EDGE_BATCH unset)")

# Contract gates
EXPECTED_N_GENES = 480
MIN_SURVIVING_GPS = _env_int("NC_MIN_GPS", 20)  # GP-survival gate
REQUIRE_GP_CACHE = os.environ.get("NC_REQUIRE_GP_CACHE", "1") == "1"

# Analysis keys
sample_key = "sample"
status_key = "status"
cell_type_key = "cell_type"
cell_type_mn_key = "cell_type_mn"
vh_key = "in_VH_v2"
ismn_key = "is_MN_v2"
ismn_alias_key = "is_MN"      # canonicalised 2026-09-02 to equal is_MN_v2
ismn_v1_key = "is_MN_v1"
has_vh_key = "has_vh_v2"
vh_source_key = "vh_v2_source"

# ---- THE LABEL CONTRACT ---------------------------------------------------- #
# The nine coarse transcriptional classes, in the byte-identical order carried
# by all 20 pass2 files. Any deviation is a hard abort: a 10th class would be
# silently coded -1 (i.e. NaN) by the fixed-category Categorical below, and the
# partition would stop tiling every cell.
NATIVE_CELL_TYPES = (
    "Astrocytes", "Endothelial", "Excitatory neurons", "Inhibitory neurons",
    "Macrophages", "Microglia", "Motor neurons", "OPCs", "Oligodendrocytes",
)
COARSE_MN_CAT = "Motor neurons"   # the ~101,660-cell transcriptional cluster
OTHER_NEURONS = "Other neurons"   # where that cluster is demoted to
MN_CLASS = "Motor neurons"        # the name, re-bound to is_MN_v2 == True
# Ten classes, FIXED order, MN LAST (a 5-to-109-cell sliver belongs on top of
# every stacked bar, and a fixed list keeps every run's crosstab identically
# shaped even where a class is empty).
CELL_TYPE_MN_CLASSES = [c for c in NATIVE_CELL_TYPES if c != COARSE_MN_CAT] + \
                       [OTHER_NEURONS, MN_CLASS]

# obs columns that survive into the saved h5ad (the pass2 files carry 171; the
# tenth, cell_type_mn, is synthesised here).
KEEP_OBS = [
    sample_key, status_key, cell_type_key, cell_type_mn_key,
    ismn_key, ismn_alias_key, ismn_v1_key,
    vh_key, has_vh_key, vh_source_key,
]
# Columns whose presence is a hard per-file gate
REQUIRED_OBS = [
    sample_key, status_key, cell_type_key,
    ismn_key, ismn_alias_key, ismn_v1_key, vh_key, has_vh_key, vh_source_key,
]

# Default folder paths
DEFAULT_PASS2_DIR = ("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/"
                     "Spatial/Ranger_procd")
DEFAULT_BASE_FOLDER = ("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/"
                       "Spatial/NicheCompass_SC")
DEFAULT_RUNS_BASE = f"{DEFAULT_BASE_FOLDER}/runs_v2"
DEFAULT_RUNSTAMP = "20260902_ncv2"

CONFIGS = {
    "persample_whole":  {"scope": "persample",  "subset": "whole"},
    "persample_vh":     {"scope": "persample",  "subset": "vh"},
    "integrated_whole": {"scope": "integrated", "subset": "whole"},
    "integrated_vh":    {"scope": "integrated", "subset": "vh"},
}

# File TAG: 'SD' + 5 digits + 2 capitals (SD00614BG ... SD05413BG). STRICT on
# purpose — --tag is a path component of the output tree, so anything that is
# not exactly a TAG (a '..', a '/', a lowercase typo) is refused at parse time.
# The two groups feed the tag-derived sample label 'SD00614_BG'.
TAG_RE = re.compile(r"^(SD\d{5})([A-Z]{2})$")
# --runstamp: ONE path-safe component. Must start with an alphanumeric (so '.',
# '..' and hidden names are out), then letters/digits/_/-/. only, max 64 chars.
RUNSTAMP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


# =========================================================================== #
# Small pure helpers                                                          #
# =========================================================================== #

def nullable_bool_mask(series, what: str) -> np.ndarray:
    """A plain numpy bool mask from a pandas nullable-boolean column, <NA> -> False.

    MANDATORY idiom. On this env (anndata 0.12.19 / pandas 2.3.2 / numpy 2.2.6)
    every naive path RAISES on an all-<NA> column (np.asarray(s).astype(bool),
    s.astype(bool), s.to_numpy(dtype=bool), (s == True).to_numpy(dtype=bool),
    and adata[s] / adata[s == True] indexing) — so a forgotten .fillna(False)
    would abort the three vh_v2_source=='excluded' sections rather than
    mislabel them, but it would abort them at a confusing place.
    """
    s = pd.Series(series).astype("boolean")
    return s.fillna(False).to_numpy(dtype=bool)


def nullable_bool_codes(series) -> np.ndarray:
    """int8 codes for a nullable boolean: 1 True / 0 False / -1 <NA>.

    NA-safe way to compare two nullable-boolean columns element-wise (object
    == object would evaluate pd.NA == pd.NA and raise on truthiness).
    """
    s = pd.Series(series).astype("boolean")
    vals = s.fillna(False).to_numpy(dtype=np.int8)
    return np.where(s.isna().to_numpy(dtype=bool), np.int8(-1), vals).astype(np.int8)


def build_cell_type_mn(obs: pd.DataFrame):
    """THE LABEL CONTRACT. Returns (Categorical, mn_mask, stats).

    obs['cell_type'] with the coarse transcriptional "Motor neurons" class
    renamed "Other neurons", then every is_MN_v2 == True cell promoted to
    "Motor neurons" — the flag wins over every base class. The categories are
    the FIXED ten in CELL_TYPE_MN_CLASSES (MN last), so:
      * an excluded section still declares a zero-count "Motor neurons" level
        instead of dropping the column from every downstream crosstab, and
      * a class the pass2 vocabulary grows would be coded -1 rather than
        silently absorbed — which the assertion below turns into a hard abort.

    is_MN_v2 <NA> cells are simply not "Motor neurons": they keep their demoted
    base class, so the partition tiles every cell. Of the 844 cohort-wide
    is_MN_v2 True cells only 472 sit inside the coarse transcriptional class,
    so the promotion genuinely takes 372 cells off Astrocytes / Oligodendrocytes
    / Endothelial etc. — both figures are recorded per section and per run.
    """
    base = obs[cell_type_key].astype(str).to_numpy()
    mn = nullable_bool_mask(obs[ismn_key], ismn_key)
    labels = np.where(base == COARSE_MN_CAT, OTHER_NEURONS, base)
    labels = np.where(mn, MN_CLASS, labels)  # the flag wins over every base class
    cat = pd.Categorical(labels, categories=CELL_TYPE_MN_CLASSES, ordered=False)
    if (np.asarray(cat.codes) < 0).any():
        bad = sorted(set(labels.tolist()) - set(CELL_TYPE_MN_CLASSES))
        raise AssertionError(
            f"{cell_type_mn_key} does not tile every cell — labels outside the "
            f"fixed class list: {bad}")

    codes_v2 = nullable_bool_codes(obs[ismn_key])
    n_obs = int(len(base))
    n_na = int((codes_v2 == -1).sum())
    n_coarse_mn = int((base == COARSE_MN_CAT).sum())
    n_from_coarse = int((mn & (base == COARSE_MN_CAT)).sum())
    n_from_other = int((mn & (base != COARSE_MN_CAT)).sum())
    class_counts = {c: int(n) for c, n in
                    pd.Series(cat).value_counts()
                    .reindex(CELL_TYPE_MN_CLASSES, fill_value=0).items()}
    if class_counts[MN_CLASS] != int(mn.sum()):
        raise AssertionError(
            f"label contract broken: {class_counts[MN_CLASS]} cells labelled "
            f"'{MN_CLASS}' but {int(mn.sum())} have {ismn_key} == True")
    if sum(class_counts.values()) != n_obs:
        raise AssertionError(f"{cell_type_mn_key} class counts sum to "
                             f"{sum(class_counts.values())}, not {n_obs}")
    stats = {
        "n_is_mn_v2_true": int(mn.sum()),
        "n_is_mn_v2_na": n_na,
        "motor_neuron_call_available": bool(n_na < n_obs),
        "mn_class_status": ("observed" if n_na < n_obs else
                            "n/a (section excluded from VH annotation)"),
        "n_coarse_transcriptional_mn": n_coarse_mn,
        "n_promoted_from_coarse_MN": n_from_coarse,
        "n_promoted_from_other_classes": n_from_other,
        "n_demoted_to_other_neurons": n_coarse_mn - n_from_coarse,
        "promoted_source_classes": (
            {str(k): int(v) for k, v in
             pd.Series(base[mn]).value_counts().items()} if mn.any() else {}),
        "class_counts": class_counts,
    }
    return cat, mn, stats


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "unavailable"


def size_batches(nnz: int, n_obs: int):
    """Size edge/node batches from the assembled adjacency.

    The undirected edge count is nnz//2 (A is symmetric); .train() splits off
    edge_val_ratio of it, so n_edges_train ~ 0.9 * nnz/2 (measured: 2296
    undirected -> 2022 train + 224 val on the 931-cell VH graph). Capping the
    edge batch at n_edges_train//8 buys >= 8 gradient steps per epoch on a tiny
    section; EDGE_BATCH_CEILING (NC_EDGE_BATCH, default 2048; the H200 sbatch
    files export 4096) stays the ceiling, so anything above ~37k undirected
    edges (i.e. every whole section and both integrated runs) trains at it.
    node_batch_size is always explicit: that skips the auto formula
    int(edge_batch_size / floor(n_edges_train / n_nodes_train)) at
    train/trainer.py:194-196, whose floor() is a ZeroDivisionError whenever the
    mean graph degree falls below ~2.2, and it makes the Trainer's
    "Node batch size:" print truthful (it echoes the argument, not the result).
    """
    n_und = int(nnz // 2)
    n_edges_train = int(round((1.0 - edge_val_ratio) * n_und))
    n_nodes_train = int(round((1.0 - node_val_ratio) * n_obs))
    target = max(MIN_EDGE_BATCH, n_edges_train // MIN_STEPS_PER_EPOCH)
    ebs = min(EDGE_BATCH_CEILING, target)
    if ebs < EDGE_BATCH_CEILING:
        # tidy multiple of 32 (reproduces the verified 256/128 tiny-run point)
        ebs = max(MIN_EDGE_BATCH, (ebs // 32) * 32)
    nbs = max(32, ebs // 2)
    return int(ebs), int(nbs), n_und, n_edges_train, n_nodes_train


def gpu_inventory() -> dict:
    """torch's view of the GPU plus the nvidia-smi-visible count.

    Under --gres=gpu:h200:1 CUDA_VISIBLE_DEVICES is the cgroup-relative index
    and is always '0'; SLURM_JOB_GPUS carries the physical id and differs
    between concurrent array tasks. Nothing here selects a device — bare 'cuda'
    == local index 0 is the task's private H200 (verified, array 52471822).
    """
    inv = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count())
        if torch.cuda.is_available() else 0,
        "torch_cuda_version": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"),
        "GPU_DEVICE_ORDINAL": os.environ.get("GPU_DEVICE_ORDINAL"),
        "nvidia_smi_visible_gpus": None,
        "nvidia_smi_names": [],
        "device_name": None,
        "device_total_gb": None,
    }
    if inv["torch_cuda_available"]:
        try:
            inv["device_name"] = torch.cuda.get_device_name(0)
            inv["device_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        except Exception:
            pass
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=30)
        if out.returncode == 0:
            names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
            inv["nvidia_smi_visible_gpus"] = len(names)
            inv["nvidia_smi_names"] = names
    except Exception:
        pass  # nvidia-smi absent on a CPU node — not an error here
    return inv


def _safe_int(val, default=None):
    """int(val) or `default` — never raises (used on introspected attributes)."""
    try:
        return int(val)
    except (TypeError, ValueError, OverflowError):
        return default


# =========================================================================== #
# Output-tree safety: argument validation, containment, atomic manifest       #
# =========================================================================== #

def validate_tag(tag: str) -> str:
    """A --tag must be exactly a file TAG (TAG_RE); returns it unchanged.

    The TAG is a path component of the output tree, so a free-form string here
    would be a path-traversal vector — refused before any path is built.
    """
    if not isinstance(tag, str) or not TAG_RE.fullmatch(tag):
        raise ValueError(
            f"--tag {tag!r} is not a valid file TAG (expected 'SD' + 5 digits + "
            f"2 capitals, e.g. SD01923BI)")
    return tag


def validate_runstamp(runstamp: str) -> str:
    """A --runstamp must be ONE path-safe component (RUNSTAMP_RE); returns it."""
    if not isinstance(runstamp, str) or not RUNSTAMP_RE.fullmatch(runstamp):
        raise ValueError(
            f"--runstamp {runstamp!r} must be a single path-safe component: "
            "start with a letter/digit, then letters, digits, '_', '-' or '.' "
            "only (max 64 chars; no '/', no '..', e.g. 20260902_ncv2)")
    return runstamp


def check_run_folder_containment(run_folder_path: str, runs_base: str) -> str:
    """ISOLATION GATE on RESOLVED paths. Returns the resolved run folder.

    pathlib's is_relative_to() is purely lexical (it never follows '..' or
    symlinks), so both sides are resolved first. Refuses when:
      * the run folder is not strictly inside --runs-base once resolved (a
        traversing component, or an intermediate directory that is a symlink
        pointing elsewhere), or
      * 'artifacts' is a path part of --runs-base (as given OR resolved) or of
        the resolved run folder — artifacts/sample_integration/latest is the
        default input of four existing scripts and must never be touched.
    """
    base_res = Path(runs_base).resolve()
    run_res = Path(run_folder_path).resolve()
    if run_res == base_res or not run_res.is_relative_to(base_res):
        raise SystemExit(
            f"computed run folder {run_folder_path} (resolves to {run_res}) is "
            f"not strictly inside --runs-base {runs_base} (resolves to "
            f"{base_res}). Refusing to run.")
    for label, path in (("--runs-base", Path(runs_base)),
                        ("--runs-base (resolved)", base_res),
                        ("run folder (resolved)", run_res)):
        if "artifacts" in path.parts:
            raise SystemExit(
                f"{label} {path} sits under an 'artifacts' directory. The v2 "
                "tree must be separate from artifacts/sample_integration (whose "
                "'latest' symlink four existing scripts default to). Refusing "
                "to run.")
    return str(run_res)


def write_json_atomic(path: str, obj) -> None:
    """Write JSON to <path> via <path>.tmp.<pid> + os.replace.

    A reader (the sbatch wrappers test for results/run_manifest.json) can never
    see a partial file: either the previous content or the complete new one.
    ANY failure removes the temp file and re-raises — the caller decides whether
    that is fatal (for the run manifest it is).
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def finished_manifest_state(path: str) -> str:
    """'absent' | 'complete' | 'corrupt' for a results/run_manifest.json.

    'complete' means a parseable JSON object carrying the 'config' key this
    trainer always writes. A 'corrupt' file (e.g. a pre-atomic-write truncated
    manifest) must NOT count as finished — the run is retrained instead.
    """
    if not os.path.isfile(path):
        return "absent"
    try:
        with open(path) as fh:
            man = json.load(fh)
    except (OSError, ValueError):
        return "corrupt"
    return "complete" if isinstance(man, dict) and "config" in man else "corrupt"


# =========================================================================== #
# Tag manifest                                                                #
# =========================================================================== #

_MANIFEST_ALIASES = {
    "tags_all20": ("tags_all20", "tags_all", "tags_whole", "tags_all_20"),
    "tags_vh17": ("tags_vh17", "tags_vh", "tags_vh_17", "tags_vh_annotated"),
    "excluded": ("excluded", "tags_excluded", "excluded_tags"),
    "per_tag": ("per_tag", "per_sample", "tags", "per_tag_counts"),
}


def _pick(man: dict, logical: str):
    for key in _MANIFEST_ALIASES[logical]:
        if key in man:
            return key, man[key]
    return None, None


def load_tag_manifest(path: str) -> dict:
    """Read + validate the preflight tag manifest (the only tag source of truth)."""
    if not os.path.isfile(path):
        raise SystemExit(
            f"tag manifest not found: {path}\nRun the v2 preflight first; this "
            "trainer never hardcodes a tag list (two lists in two places is how "
            "an array-index -> TAG mapping silently drifts).")
    with open(path) as fh:
        man = json.load(fh)
    if not isinstance(man, dict):
        raise SystemExit(f"tag manifest {path} is not a JSON object")

    key_all, tags_all = _pick(man, "tags_all20")
    key_vh, tags_vh = _pick(man, "tags_vh17")
    if tags_all is None or tags_vh is None:
        raise SystemExit(
            f"tag manifest {path} must provide 'tags_all20' and 'tags_vh17' "
            f"(aliases accepted: {_MANIFEST_ALIASES['tags_all20']} / "
            f"{_MANIFEST_ALIASES['tags_vh17']}). Found keys: {sorted(man)}")
    tags_all = [str(t) for t in tags_all]
    tags_vh = [str(t) for t in tags_vh]
    for name, tags in (("tags_all20", tags_all), ("tags_vh17", tags_vh)):
        if not tags:
            raise SystemExit(f"tag manifest {path}: '{name}' is empty")
        if len(set(tags)) != len(tags):
            dupes = sorted({t for t in tags if tags.count(t) > 1})
            raise SystemExit(f"tag manifest {path}: '{name}' has duplicates {dupes}")
        bad = [t for t in tags if not TAG_RE.fullmatch(t)]
        if bad:
            raise SystemExit(f"tag manifest {path}: '{name}' has malformed TAGs "
                             f"{bad} (expected e.g. 'SD00614BG')")
    stray = [t for t in tags_vh if t not in set(tags_all)]
    if stray:
        raise SystemExit(f"tag manifest {path}: tags_vh17 has TAGs absent from "
                         f"tags_all20: {stray}")

    key_exc, excluded = _pick(man, "excluded")
    derived_excluded = [t for t in tags_all if t not in set(tags_vh)]
    unclassified = []
    if excluded is None:
        excluded = derived_excluded
        key_exc = "(derived: tags_all20 - tags_vh17)"
    else:
        excluded = [str(t) for t in excluded]
        # The two conditions that matter for tag SELECTION: 'excluded' and
        # 'tags_vh17' must be disjoint, and both must be subsets of tags_all20.
        # They need NOT be exact complements — the preflight derives tags_vh17
        # and excluded from its ok_records but tags_all20 from all tags, so a
        # section that fails a preflight record can legitimately sit in neither
        # list. Such a tag is warned about by name below and stays trainable as
        # a whole section (it is simply never a VH section).
        in_both = sorted(set(excluded) & set(tags_vh))
        if in_both:
            raise SystemExit(
                f"tag manifest {path} is internally inconsistent: "
                f"{in_both} appear in BOTH '{key_exc}' and '{key_vh}'. A "
                "section cannot be excluded from the VH annotation and VH-"
                "annotated at once. Fix the preflight.")
        stray_exc = [t for t in excluded if t not in set(tags_all)]
        if stray_exc:
            raise SystemExit(
                f"tag manifest {path}: '{key_exc}' has TAGs absent from "
                f"tags_all20: {stray_exc}. Fix the preflight.")
        dupes_exc = sorted({t for t in excluded if excluded.count(t) > 1})
        if dupes_exc:
            raise SystemExit(f"tag manifest {path}: '{key_exc}' has duplicates "
                             f"{dupes_exc}")
        unclassified = [t for t in tags_all
                        if t not in set(tags_vh) and t not in set(excluded)]
        if unclassified:
            warn(f"tag manifest {path}: {len(unclassified)} TAG(s) of "
                 f"tags_all20 are in NEITHER '{key_vh}' nor '{key_exc}': "
                 f"{unclassified}. They remain whole-section runs (C1/C3) and "
                 "are never VH sections (C2/C4); check the preflight if that "
                 "is not intended.")

    key_per, per_tag = _pick(man, "per_tag")
    per_tag_norm = {}
    if isinstance(per_tag, dict):
        per_tag_norm = {str(k): v for k, v in per_tag.items()
                        if isinstance(v, dict)}
    elif isinstance(per_tag, list):
        for row in per_tag:
            if isinstance(row, dict):
                tag = row.get("tag") or row.get("TAG") or row.get("sample_tag")
                if tag is not None:
                    per_tag_norm[str(tag)] = row
    if not per_tag_norm:
        warn(f"tag manifest {path} has no usable per-tag counts "
             f"(looked for {_MANIFEST_ALIASES['per_tag']}) — the cell-count "
             "cross-check will be skipped")

    log(f"tag manifest: {path}")
    log(f"  tags_all20 <- '{key_all}': {len(tags_all)} TAGs")
    log(f"  tags_vh17  <- '{key_vh}': {len(tags_vh)} TAGs")
    log(f"  excluded   <- '{key_exc}': {sorted(excluded)}")
    if unclassified:
        log(f"  unclassified (neither list): {unclassified}")
    if per_tag_norm:
        log(f"  per-tag counts <- '{key_per}': {len(per_tag_norm)} entries")
    return {"path": path, "tags_all20": tags_all, "tags_vh17": tags_vh,
            "excluded": excluded, "unclassified": unclassified,
            "per_tag": per_tag_norm}


def manifest_expected_cells(tag_manifest: dict, tags, subset: str):
    """Warn-only expected cell count for this run, from the preflight counts."""
    per_tag = tag_manifest["per_tag"]
    if not per_tag:
        return None
    keys = (("n_obs", "n_cells", "n_obs_total")
            if subset == "whole" else
            ("n_in_vh_v2", "n_vh", "n_in_vh", "in_VH_v2_true"))
    total = 0
    for tag in tags:
        row = per_tag.get(tag)
        if not isinstance(row, dict):
            return None
        val = next((row[k] for k in keys if k in row), None)
        if val is None:
            return None
        total += int(val)
    return total


# =========================================================================== #
# Argument parsing / derived configuration                                    #
# =========================================================================== #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("Generalised NicheCompass trainer for the four v2 "
                     "configurations of the ALS spinal-cord Xenium cohort."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", required=True, choices=sorted(CONFIGS),
                   help="which of the four configurations to train")
    p.add_argument("--tag", default=None,
                   help="file TAG (e.g. SD01923BI). REQUIRED for persample_*, "
                        "FORBIDDEN for integrated_*")
    p.add_argument("--runstamp", default=DEFAULT_RUNSTAMP,
                   help="fixed run stamp; the output dir level below <config>")
    p.add_argument("--manifest", required=True,
                   help="tag manifest JSON written by the v2 preflight")
    p.add_argument("--pass2-dir", default=DEFAULT_PASS2_DIR,
                   help="directory holding ALS_SCXenium_<TAG>_pass2.h5ad")
    p.add_argument("--exclude-cells", default=None,
                   help="JSON written by nc3_step1.py holding per_tag_obs_names: "
                        "{TAG: [pass2 obs_name, ...]}. Those cells are dropped "
                        "from each section AFTER the file-level gates and BEFORE "
                        "the 4-NN graph, so neighbourhoods are rebuilt without "
                        "them. Used to retrain without niche v1_3 (non-cord).")
    p.add_argument("--runs-base", default=DEFAULT_RUNS_BASE,
                   help="root of the v2 output tree (must NOT be under artifacts/)")
    p.add_argument("--base-folder", default=DEFAULT_BASE_FOLDER,
                   help="NicheCompass_SC root; supplies gene_programs/ and "
                        "gene_annotations/ (the cached GP resources)")
    p.add_argument("--n-epochs", type=int, default=400,
                   help="epoch CEILING; 0.3.3 early stopping is on by default")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for numpy/random/torch, NicheCompass(seed=) and "
                        "Trainer(seed=) — 0 is the library default, i.e. the "
                        "2026-08-26 integrated run's split")
    p.add_argument("--early-stopping-patience", type=int, default=None,
                   help="default 8 for integrated_* (the 0.3.3 default, so C3 "
                        "reproduces the 08-26 regime) and 30 for persample_* "
                        "(> n_epochs_all_gps=25, so a tiny section always "
                        "reaches the active-GP-only phase before it can stop)")
    p.add_argument("--early-stopping-lr-patience", type=int, default=None,
                   help="default 4 for integrated_*, 10 for persample_*")
    p.add_argument("--keep-obs-extra", default="",
                   help="comma-separated extra obs columns to carry into the "
                        "saved h5ad on top of the ten label-contract columns")
    p.add_argument("--no-monitor", action="store_true",
                   help="suppress the Trainer per-epoch progress line")
    p.add_argument("--overwrite", action="store_true",
                   help="retrain a run that already holds a "
                        "results/run_manifest.json. Without it such a run is "
                        "SKIPPED with exit 0, which is what makes a chunked "
                        "C1/C2 array task safely resubmittable")
    args = p.parse_args(argv)

    scope = CONFIGS[args.config]["scope"]
    if scope == "persample" and not args.tag:
        p.error(f"--tag is required for --config {args.config}")
    if scope == "integrated" and args.tag:
        p.error(f"--tag is forbidden for --config {args.config} "
                "(the tag set comes from the manifest)")
    if args.n_epochs < 1:
        p.error("--n-epochs must be >= 1")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 2:
        p.error("--early-stopping-patience must be >= 2 (EarlyStopping.step "
                "cannot fire before epoch == patience, and 1 would let a run "
                "stop on its first epoch)")
    # Both values become path components of the output tree: validate them
    # BEFORE any path is built (a free-form --tag '../../x' would otherwise pass
    # a lexical containment check and create directories outside --runs-base).
    try:
        if args.tag is not None:
            validate_tag(args.tag)
        validate_runstamp(args.runstamp)
    except ValueError as exc:
        p.error(str(exc))
    return args


args = parse_args()
config = args.config
scope = CONFIGS[config]["scope"]
subset_mode = CONFIGS[config]["subset"]
is_persample = scope == "persample"
runstamp = args.runstamp
n_epochs = int(args.n_epochs)
SEED = int(args.seed)

es_patience = (args.early_stopping_patience
               if args.early_stopping_patience is not None
               else (30 if is_persample else 8))
es_lr_patience = (args.early_stopping_lr_patience
                  if args.early_stopping_lr_patience is not None
                  else (10 if is_persample else 4))
monitor = not args.no_monitor

# Reproducibility. NicheCompass 0.3.3 re-seeds np/torch with its own seed inside
# __init__ and Trainer(seed=) governs the train/val split, so these top-level
# seeds only pin the pre-model steps (GP extraction order, the example print).
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
sc.settings.n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

keep_obs_extra = [c.strip() for c in args.keep_obs_extra.split(",") if c.strip()]
keep_obs = list(dict.fromkeys(KEEP_OBS + keep_obs_extra))

# ---- Folder paths --------------------------------------------------------- #
# ---- cell exclusion list (optional) --------------------------------------- #
# Keyed by TAG, holding pass2 obs_names. Loaded once; applied per section below.
exclude_map, exclude_meta = {}, None
if args.exclude_cells:
    with open(args.exclude_cells) as _fh:
        _payload = json.load(_fh)
    exclude_map = {k: set(v) for k, v in _payload["per_tag_obs_names"].items()}
    exclude_meta = {k: v for k, v in _payload.items() if k != "per_tag_obs_names"}
    _tot = sum(len(v) for v in exclude_map.values())
    log(f"EXCLUSION LIST: {args.exclude_cells}")
    log(f"  dropping {_tot:,} cells across {len(exclude_map)} tags "
        f"(label {_payload.get('dropped_label')!r}: {_payload.get('bench_call')!r})")

sample_data_folder = os.path.abspath(args.pass2_dir)
base_folder = os.path.abspath(args.base_folder)
runs_base = os.path.abspath(args.runs_base)
ga_data_folder_path = f"{base_folder}/gene_annotations"
gp_data_folder_path = f"{base_folder}/gene_programs"
omnipath_lr_network_file_path = f"{gp_data_folder_path}/omnipath_lr_network.csv"
nichenet_lr_network_file_path = f"{gp_data_folder_path}/nichenet_lr_network.csv"
nichenet_ligand_target_matrix_file_path = (
    f"{gp_data_folder_path}/nichenet_ligand_target_matrix.csv")
mebocost_folder_path = f"{gp_data_folder_path}/metabolite_enzyme_sensor_gps"
mebocost_enzymes_tsv = f"{mebocost_folder_path}/human_metabolite_enzymes.tsv"
mebocost_sensors_tsv = f"{mebocost_folder_path}/human_metabolite_sensors.tsv"
# Only read for species='mouse' — kept for API parity with the Sanger call:
gene_orthologs_mapping_file_path = (f"{ga_data_folder_path}/"
                                    "human_mouse_gene_orthologs.csv")

# ISOLATION GATE: this trainer must never be able to write into the 08-26 tree.
# artifacts/sample_integration/latest is the default input of four existing
# scripts (NC_SC_downstream.py, NC_SC_downstream_v2_ismn.py, niche_ismn_cross.py,
# umap_replot_dense.py); repointing or overwriting it would silently retarget them.
# --tag and --runstamp were validated in parse_args() (strict TAG / single
# path-safe component), and the containment below compares RESOLVED paths, so
# neither a '..' component nor a symlinked intermediate directory can escape.
run_root_folder_path = f"{runs_base}/{config}"
run_folder_path = (f"{run_root_folder_path}/{runstamp}/{args.tag}"
                   if is_persample else f"{run_root_folder_path}/{runstamp}")
run_folder_resolved = check_run_folder_containment(run_folder_path, runs_base)
model_folder_path = f"{run_folder_path}/model"
figure_folder_path = f"{run_folder_path}/figures"
result_folder_path = f"{run_folder_path}/results"
manifest_path = f"{result_folder_path}/run_manifest.json"
adata_file_name = (f"NC_v2_{config}_{args.tag}.h5ad" if is_persample
                   else f"NC_v2_{config}.h5ad")

_manifest_state = finished_manifest_state(manifest_path)
if _manifest_state == "complete" and not args.overwrite:
    # SKIP, not fail: the runstamp is fixed and the C1/C2 arrays run several
    # sections per task, so a task that dies on section 3 of 4 must be
    # resubmittable without re-training (or clobbering) sections 1 and 2.
    # Exit 0 so SLURM records the retry as success.
    print("", flush=True)
    log("=" * 78)
    log(f"SKIP — this run already FINISHED: {manifest_path}")
    log("     (the manifest is written last and atomically, so its presence "
        "means the model, adata and results are all in place)")
    log("     Pass --overwrite to retrain it, or bump --runstamp for a new run.")
    log("=" * 78)
    sys.exit(0)
if _manifest_state == "complete":
    warn(f"--overwrite: retraining over the finished run at {run_folder_path}")
elif _manifest_state == "corrupt":
    warn(f"{manifest_path} exists but is NOT a parseable run manifest — "
         "treating this run as UNFINISHED and retraining it (the file will be "
         "replaced atomically on success)")
# NOTE: no directory is created here. The output dirs are made only after the
# tag manifest is loaded and --tag is accepted (see below), so a refused tag
# (excluded section, typo, wrong list) leaves nothing behind on disk.


# --------------------------------------------------------------------------- #
section(f"[1/8] Environment / GPU / configuration — {config}")
# --------------------------------------------------------------------------- #
log(f"python {sys.version.split()[0]} @ {sys.executable}")
log(f"script {os.path.abspath(__file__)}")
log(f"argv   {shlex.join(sys.argv)}")
log(f"SLURM job {os.environ.get('SLURM_JOB_ID', 'n/a')} "
    f"(array {os.environ.get('SLURM_ARRAY_JOB_ID', 'n/a')}"
    f"_{os.environ.get('SLURM_ARRAY_TASK_ID', 'n/a')}) on {os.uname().nodename}")

versions = {}
for _pkg in ("nichecompass", "torch", "anndata", "scanpy", "numpy", "scipy",
             "scikit-learn", "pandas"):
    try:
        versions[_pkg] = pkg_version(_pkg)
    except Exception:
        versions[_pkg] = None
    log(f"  {_pkg} {versions[_pkg]}")
log(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
log(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}")

gpu_info = gpu_inventory()
use_cuda_if_available = bool(gpu_info["torch_cuda_available"])
if use_cuda_if_available:
    device_name = gpu_info["device_name"]
    log(f"CUDA available: {device_name} ({gpu_info['device_total_gb']} GB), "
        f"toolkit {gpu_info['torch_cuda_version']}")
    log(f"torch.cuda.device_count()={gpu_info['torch_device_count']}, "
        f"nvidia-smi visible GPUs={gpu_info['nvidia_smi_visible_gpus']}")
    for line in gpu_info["nvidia_smi_names"]:
        log(f"  nvidia-smi: {line}")
    log(f"CUDA_VISIBLE_DEVICES={gpu_info['CUDA_VISIBLE_DEVICES']} "
        f"(cgroup-relative), SLURM_JOB_GPUS={gpu_info['SLURM_JOB_GPUS']} "
        "(physical id) — nothing in this script selects a device index")
else:
    device_name = "CPU"
    log("WARNING: CUDA NOT available!")
    if os.environ.get("NC_ALLOW_CPU", "0") != "1":
        raise RuntimeError(
            "No GPU visible, but this is a GPU training job (submit with a GPU "
            "allocation, e.g. --gres=gpu:h200:1 on gpu_long or any other "
            "--gres=gpu:<type>:1). Export NC_ALLOW_CPU=1 to force CPU training "
            "(fine for a tiny smoke test, hopeless on the integrated cohort).")
    log("NC_ALLOW_CPU=1 set — continuing on CPU (very slow).")
# The edge-batch ceiling is the ONLY size knob that depends on the card. It is
# read from NC_EDGE_BATCH (the H200 sbatch files export 4096); the default of
# 2048 applies when the variable is unset (e.g. an ad-hoc 8 GB qrtx4000 job).
log(f"edge-batch ceiling: {EDGE_BATCH_CEILING} <- {EDGE_BATCH_CEILING_SOURCE}; "
    f"device memory: {gpu_info['device_total_gb'] or 'n/a'} GB")

tag_manifest = load_tag_manifest(args.manifest)
if is_persample:
    tag = args.tag
    if tag not in set(tag_manifest["tags_all20"]):
        raise SystemExit(f"--tag {tag} is not in the manifest's tags_all20 "
                         f"({tag_manifest['path']})")
    if subset_mode == "vh":
        if tag in set(tag_manifest["excluded"]):
            raise SystemExit(
                f"--tag {tag} is an EXCLUDED section (vh_v2_source='excluded'): "
                f"in_VH_v2 and is_MN_v2 are entirely <NA> there, so a VH subset "
                f"is empty by construction. Config {config} covers the "
                f"{len(tag_manifest['tags_vh17'])} annotated sections only.")
        if tag not in set(tag_manifest["tags_vh17"]):
            extra = (" (it is in NEITHER tags_vh17 nor excluded — the preflight "
                     "did not classify it; only whole-section runs are possible)"
                     if tag in set(tag_manifest.get("unclassified", [])) else "")
            raise SystemExit(f"--tag {tag} is not in the manifest's tags_vh17"
                             f"{extra}")
    tags = [tag]
else:
    tags = (list(tag_manifest["tags_all20"]) if subset_mode == "whole"
            else list(tag_manifest["tags_vh17"]))
n_expected_samples = len(tags)

# The tag is accepted and the run is not a finished one: NOW create the output
# tree (plus the GP cache dirs). Everything above this line is read-only on
# disk, so a refused run leaves no empty <TAG>/ or <runstamp>/ directory behind
# and a "which of the 40 runs exist?" directory sweep stays truthful.
for _dir in (model_folder_path, figure_folder_path, result_folder_path,
             ga_data_folder_path, gp_data_folder_path, mebocost_folder_path):
    os.makedirs(_dir, exist_ok=True)

expected_n_cells = manifest_expected_cells(tag_manifest, tags, subset_mode)
log(f"config={config} scope={scope} subset={subset_mode}")
log(f"sections ({n_expected_samples}): {tags}")
log(f"expected cells from the manifest: "
    f"{'n/a' if expected_n_cells is None else format(expected_n_cells, ',')}")
log(f"label contract: obs['{cell_type_mn_key}'] = 10 classes, "
    f"'{MN_CLASS}' := {ismn_key} True (wins over every base class), coarse "
    f"'{COARSE_MN_CAT}' -> '{OTHER_NEURONS}'")
log(f"class order: {CELL_TYPE_MN_CLASSES}")
log(f"obs kept in the saved h5ad: {keep_obs}")
log(f"seed={SEED}, n_epochs={n_epochs} (ceiling), n_epochs_all_gps={n_epochs_all_gps}")
log(f"early stopping: patience={es_patience}, lr_patience={es_lr_patience} "
    f"(earliest possible stop is epoch {es_patience}; "
    f"train/utils.py keeps training while epochs < patience)")
if n_epochs <= n_epochs_all_gps:
    warn(f"--n-epochs {n_epochs} <= n_epochs_all_gps {n_epochs_all_gps}: the "
         "model can never enter active-GP-only mode, so every GP will report "
         "as active. Fine for a smoke test, not for a production run.")
log(f"run folder: {run_folder_path}")


# --------------------------------------------------------------------------- #
section("[2/8] Gene-program resources (OmniPath / NicheNet / MEBOCOST)")
# --------------------------------------------------------------------------- #
gp_cache_files = {
    "omnipath_lr_network.csv": omnipath_lr_network_file_path,
    "nichenet_lr_network.csv": nichenet_lr_network_file_path,
    "nichenet_ligand_target_matrix.csv": nichenet_ligand_target_matrix_file_path,
    "human_metabolite_enzymes.tsv": mebocost_enzymes_tsv,
    "human_metabolite_sensors.tsv": mebocost_sensors_tsv,
}
missing_cache = sorted(k for k, v in gp_cache_files.items()
                       if not os.path.isfile(v))
if missing_cache:
    msg = [f"GP resource cache MISS: {missing_cache}",
           f"expected under {gp_data_folder_path}"]
    if REQUIRE_GP_CACHE:
        banner_warn(msg)
        raise SystemExit(
            "GP cache gate FAILED. Every v2 run must take the load_from_disk "
            "branch with zero downloads: up to 6 array tasks run concurrently "
            "and a cache miss makes them all fetch and write the same CSV. "
            "Stage the missing files (stage_gp_resources.sh) or export "
            "NC_REQUIRE_GP_CACHE=0 to allow a live fetch from THIS run only.")
    banner_warn(msg + ["NC_REQUIRE_GP_CACHE=0 — allowing a live fetch"])
else:
    log(f"GP resource cache HIT for all {len(gp_cache_files)} files")

# The NicheNet extractor writes temp .rds into the CWD, but only inside its
# `if not load_from_disk:` branch — with the cache staged nothing is written, so
# a shared CWD is safe for concurrent array tasks. Run from the writable oak
# gene_programs dir anyway (never /tmp: a stray /tmp/struct.py shadows the
# stdlib on SCG). Every other path in this script is absolute.
os.chdir(gp_data_folder_path)
log(f"cwd -> {os.getcwd()}")


def extract_or_warn(source_name, extract_fn):
    """Run one GP extractor; on ANY failure warn loudly and return None so the
    remaining sources can still carry the run (gated later)."""
    try:
        t0 = time.time()
        gp_dict = extract_fn()
        log(f"{source_name}: {len(gp_dict)} GPs extracted "
            f"({time.time() - t0:.0f}s)")
        return gp_dict
    except Exception:
        lines = [f"WARNING: {source_name} GP extraction FAILED — "
                 "this source will be SKIPPED"]
        lines += traceback.format_exc().splitlines()
        if source_name == "OmniPath":
            lines += [
                "NOTE: OmniPath fails DETERMINISTICALLY in nichecompass 0.3.3 on "
                "the cached omnipath_lr_network.csv (a NaN in "
                "'genesymbol_intercell_target' reaches resolve_protein_complexes "
                "-> TypeError: argument of type 'float' is not iterable). The "
                "2026-08-26 reference run (job 52418732) failed here too and "
                "trained on NicheNet + MEBOCOST only. Expected, not new."]
        banner_warn(lines)
        return None
    finally:
        plt.close("all")  # the extractors leave their QC figures open


def _extract_omnipath():
    from_disk = os.path.isfile(omnipath_lr_network_file_path)
    log(f"OmniPath LR network cache "
        f"{'HIT (loading from disk)' if from_disk else 'MISS (live fetch)'}: "
        f"{omnipath_lr_network_file_path}")
    return extract_gp_dict_from_omnipath_lr_interactions(
        species=species,
        min_curation_effort=0,  # 0.3.3 default drifted to 2 — keep permissive
        load_from_disk=from_disk,
        save_to_disk=not from_disk,
        lr_network_file_path=omnipath_lr_network_file_path,
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        plot_gp_gene_count_distributions=True,
        gp_gene_count_distributions_save_path=(
            f"{figure_folder_path}/omnipath_gp_gene_count_distributions.png"))


def _extract_nichenet():
    from_disk = (os.path.isfile(nichenet_lr_network_file_path)
                 and os.path.isfile(nichenet_ligand_target_matrix_file_path))
    log(f"NicheNet CSV cache "
        f"{'HIT (loading from disk)' if from_disk else 'MISS (zenodo download)'}")
    return extract_gp_dict_from_nichenet_lrt_interactions(
        species=species,
        version="v2",
        keep_target_genes_ratio=1.,
        max_n_target_genes_per_gp=250,
        load_from_disk=from_disk,
        save_to_disk=not from_disk,
        lr_network_file_path=nichenet_lr_network_file_path,
        ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        plot_gp_gene_count_distributions=True,
        gp_gene_count_distributions_save_path=(
            f"{figure_folder_path}/nichenet_gp_gene_count_distributions.png"))


def _extract_mebocost():
    for tsv in (mebocost_enzymes_tsv, mebocost_sensors_tsv):
        if not os.path.isfile(tsv):
            raise FileNotFoundError(
                f"MEBOCOST TSV missing: {tsv}. These are repo files (NOT in "
                "the wheel) — stage them first (stage_gp_resources.sh, or curl "
                "the two TSVs from "
                "raw.githubusercontent.com/Lotfollahi-lab/nichecompass/0.3.3/"
                "data/gene_programs/metabolite_enzyme_sensor_gps/).")
    return extract_gp_dict_from_mebocost_ms_interactions(
        species=species,
        dir_path=mebocost_folder_path,
        plot_gp_gene_count_distributions=True,
        gp_gene_count_distributions_save_path=(
            f"{figure_folder_path}/mebocost_gp_gene_count_distributions.png"))


omnipath_gp_dict = extract_or_warn("OmniPath", _extract_omnipath)
nichenet_gp_dict = extract_or_warn("NicheNet", _extract_nichenet)
mebocost_gp_dict = extract_or_warn("MEBOCOST", _extract_mebocost)

if nichenet_gp_dict:
    # Display example NicheNet GP (deterministic under SEED)
    example_gp = random.choice(sorted(nichenet_gp_dict))
    log(f"Example NicheNet GP '{example_gp}': {nichenet_gp_dict[example_gp]}")

gp_source_dicts = {"omnipath": omnipath_gp_dict,
                   "nichenet": nichenet_gp_dict,
                   "mebocost": mebocost_gp_dict}
gp_source_counts = {k: (len(v) if v is not None else None)
                    for k, v in gp_source_dicts.items()}
skipped_gp_sources = [k for k, v in gp_source_dicts.items() if v is None]
gp_dicts = [v for v in gp_source_dicts.values() if v]

if skipped_gp_sources:
    banner_warn([f"proceeding WITHOUT GP source(s): {skipped_gp_sources} — "
                 "prior-knowledge coverage is reduced"])
if not gp_dicts:
    raise RuntimeError("ALL GP sources failed to extract — nothing to train "
                       "on. Fix the staging/network errors above and rerun.")

combined_gp_dict = filter_and_combine_gp_dict_gps_v2(gp_dicts, verbose=True)
log(f"GPs per source: {gp_source_counts}")
log(f"Number of gene programs after filtering and combining: "
    f"{len(combined_gp_dict)}.")


# --------------------------------------------------------------------------- #
section(f"[3/8] Load {n_expected_samples} section(s) "
        f"({subset_mode}) + label contract + per-sample 4-NN graphs")
# --------------------------------------------------------------------------- #
# Drift detection: warn about pass2 files on disk that are not in the manifest
_found = {os.path.basename(p) for p in
          glob.glob(f"{sample_data_folder}/ALS_SCXenium_*_pass2.h5ad")}
_expected = {f"ALS_SCXenium_{t}_pass2.h5ad" for t in tag_manifest["tags_all20"]}
_unexpected = sorted(_found - _expected)
if _unexpected:
    warn(f"{len(_unexpected)} pass2 file(s) on disk are NOT in the "
         f"{len(tag_manifest['tags_all20'])}-sample manifest and will be "
         f"IGNORED: {_unexpected}")

adata_batch_list = []
adjacencies = []
sample_labels = []
section_stats = []
slim_logged = False

for i, tag in enumerate(tags, start=1):
    t0 = time.time()
    fpath = f"{sample_data_folder}/ALS_SCXenium_{tag}_pass2.h5ad"
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Missing input file: {fpath}")
    adata_batch = sc.read_h5ad(fpath)
    n_obs_file = int(adata_batch.n_obs)

    # ---- per-file contract gates ------------------------------------------
    if adata_batch.n_vars != EXPECTED_N_GENES:
        raise ValueError(f"{tag}: {adata_batch.n_vars} genes, expected "
                         f"{EXPECTED_N_GENES} — wrong input generation?")
    if counts_key not in adata_batch.layers:
        raise KeyError(f"{tag}: layers['{counts_key}'] missing — NicheCompass "
                       "must NOT fall back to X (X is log1p-normalised)")
    if spatial_key not in adata_batch.obsm:
        raise KeyError(f"{tag}: obsm['{spatial_key}'] missing")
    missing_obs = [c for c in REQUIRED_OBS if c not in adata_batch.obs.columns]
    if missing_obs:
        raise KeyError(
            f"{tag}: obs columns missing: {missing_obs}. The v2 label contract "
            "needs the columns attached on 2026-09-02 (in_VH_v2, is_MN_v2, "
            "has_vh_v2, vh_v2_source) plus the canonicalised is_MN/is_MN_v1.")

    # cell_type category vocabulary is load-bearing: the fixed 10-class
    # Categorical below would code an unknown class as -1 (NaN) and the
    # partition would stop tiling every cell.
    ct_col = adata_batch.obs[cell_type_key]
    native = (tuple(ct_col.cat.categories) if hasattr(ct_col, "cat")
              else tuple(sorted(pd.unique(ct_col.astype(str)))))
    if native != NATIVE_CELL_TYPES:
        raise ValueError(
            f"{tag}: obs['{cell_type_key}'] categories {list(native)} != the "
            f"expected {list(NATIVE_CELL_TYPES)}. The label contract's fixed "
            "10-class order is derived from this vocabulary — refusing to guess.")

    # Sample label: obs is the authority ('SD00614_BG' style); the filename TAG
    # is only a fallback for label derivation if the column were ever absent.
    m = TAG_RE.fullmatch(tag)
    tag_derived_label = f"{m.group(1)}_{m.group(2)}" if m else tag
    uniq = adata_batch.obs[sample_key].astype(str).unique().tolist()
    if len(uniq) != 1:
        raise ValueError(f"{tag}: expected exactly one obs['{sample_key}'] "
                         f"value, got {uniq}")
    sample_label = uniq[0]
    if sample_label != tag_derived_label:
        warn(f"{tag}: obs sample label '{sample_label}' != tag-derived "
             f"'{tag_derived_label}' — trusting obs")
    status_uniq = adata_batch.obs[status_key].astype(str).unique().tolist()
    if len(status_uniq) != 1:
        raise ValueError(f"{tag}: expected exactly one obs['{status_key}'] "
                         f"value, got {status_uniq}")
    status_label = status_uniq[0]
    vh_source_uniq = adata_batch.obs[vh_source_key].astype(str).unique().tolist()
    vh_source_label = vh_source_uniq[0] if len(vh_source_uniq) == 1 else \
        "|".join(sorted(vh_source_uniq))

    # ---- canonicalisation contract: is_MN is an ALIAS of is_MN_v2 ----------
    codes_v2 = nullable_bool_codes(adata_batch.obs[ismn_key])
    codes_alias = nullable_bool_codes(adata_batch.obs[ismn_alias_key])
    if not np.array_equal(codes_v2, codes_alias):
        n_diff = int((codes_v2 != codes_alias).sum())
        raise AssertionError(
            f"{tag}: obs['{ismn_alias_key}'] != obs['{ismn_key}'] in {n_diff} "
            f"of {n_obs_file} cells. The 2026-09-02 canonicalisation made them "
            "bit-for-bit identical (844 True / 122,902 <NA> cohort-wide); a "
            "difference means the pass2 file was regenerated or re-annotated. "
            "Refusing to train on an ambiguous motor-neuron call.")

    # File-level counts (cross-check surface against the preflight table)
    vh_file_codes = nullable_bool_codes(adata_batch.obs[vh_key])
    file_counts = {
        "n_in_vh_v2_true": int((vh_file_codes == 1).sum()),
        "n_in_vh_v2_false": int((vh_file_codes == 0).sum()),
        "n_in_vh_v2_na": int((vh_file_codes == -1).sum()),
        "n_is_mn_v2_true": int((codes_v2 == 1).sum()),
        "n_is_mn_v2_na": int((codes_v2 == -1).sum()),
        "n_is_mn_v1_true": int((nullable_bool_codes(
            adata_batch.obs[ismn_v1_key]) == 1).sum()),
    }
    # Verified invariant (2026-09-02): no is_MN_v2 True cell lies outside
    # in_VH_v2 True, in any section.
    n_mn_outside_vh = int(((codes_v2 == 1) & (vh_file_codes != 1)).sum())
    if n_mn_outside_vh:
        raise AssertionError(
            f"{tag}: {n_mn_outside_vh} {ismn_key}==True cells lie outside "
            f"{vh_key}==True. That invariant held for all 20 sections on "
            "2026-09-02; a violation means the VH mask and the MN call have "
            "drifted apart.")

    # ---- provenance: obs column count AS READ from the pass2 file -----------
    # Measured before anything is attached (cell_type_mn) or dropped, so the
    # manifest's obs_columns_in_pass2_file is the number a reviewer can diff
    # against the preflight (171 on 2026-09-02).
    obs_cols_before = int(adata_batch.obs.shape[1])

    # ---- drop obsp / uns / varm / varp / raw BEFORE any subset copy ----------
    # ad.concat drops all of these for the integrated configurations, but the
    # per-sample path has no concat — and a subset copy would first RE-SLICE
    # all 8 pass2 obsp graphs (~100k-node sparse matrices, one literally called
    # 'connectivities') plus carry all 54 uns entries (stale 'neighbors'/'umap'/
    # 'pca' dicts pointing at those graphs), only for them to be deleted. Left
    # in place they would ride into the saved h5ad and a downstream sc.tl.umap
    # would happily reuse the stale uns['neighbors']. Clearing them here makes
    # the VH subset copy below cheap and keeps the bookkeeping for the log and
    # the manifest.
    dropped_obsp = list(adata_batch.obsp.keys())
    dropped_uns = list(adata_batch.uns.keys())
    dropped_varm = list(adata_batch.varm.keys())
    dropped_varp = list(adata_batch.varp.keys())
    for k in dropped_obsp:
        del adata_batch.obsp[k]
    for k in dropped_varm:
        del adata_batch.varm[k]
    for k in dropped_varp:
        del adata_batch.varp[k]
    for k in dropped_uns:
        del adata_batch.uns[k]
    if adata_batch.raw is not None:
        adata_batch.raw = None

    # ---- VH subset (C2 / C4) ----------------------------------------------
    if subset_mode == "vh":
        vh_mask = nullable_bool_mask(adata_batch.obs[vh_key], vh_key)
        n_keep = int(vh_mask.sum())
        if n_keep == 0:
            raise ValueError(
                f"{tag}: the {vh_key} subset is EMPTY "
                f"({file_counts['n_in_vh_v2_na']} of {n_obs_file} cells are "
                f"<NA>, vh_v2_source='{vh_source_label}'). Excluded sections "
                "have no VH run; fix the manifest's tags_vh17.")
        adata_batch = adata_batch[vh_mask].copy()
        log(f"[{i:2d}/{len(tags)}] {tag}: {vh_key} subset "
            f"{n_keep:,} / {n_obs_file:,} cells kept")
    # ---- cell exclusion (--exclude-cells) ---------------------------------
    # Applied AFTER the per-file contract gates and BEFORE the 4-NN graph, so
    # the retained cells' neighbourhoods are rebuilt as if the dropped cells had
    # never been on the slide. That is the whole point of retraining rather than
    # masking after the fact.
    if exclude_map:
        _want = exclude_map.get(tag, set())
        _names = adata_batch.obs_names.to_numpy()
        _mask = ~np.isin(_names, list(_want)) if _want else np.ones(len(_names), bool)
        _hit = int((~_mask).sum())
        if _want and _hit != len(_want):
            raise ValueError(
                f"{tag}: exclusion list names {len(_want):,} cells but only "
                f"{_hit:,} matched the pass2 obs_names. The list was built "
                "against a different pass2 generation — refusing to train on a "
                "silently partial exclusion.")
        adata_batch = adata_batch[_mask].copy()
        log(f"[{i:2d}/{len(tags)}] {tag}: excluded {_hit:,} cells "
            f"({n_obs_file - _hit:,} kept of {n_obs_file:,})")

    n_obs_used = int(adata_batch.n_obs)
    if n_obs_used <= n_neighbors:
        raise ValueError(f"{tag}: only {n_obs_used} cells after subsetting "
                         f"(<= n_neighbors={n_neighbors}) — cannot build a "
                         "kNN graph")

    # ---- THE LABEL CONTRACT: obs['cell_type_mn'] --------------------------
    try:
        cell_type_mn, mn, label_stats = build_cell_type_mn(adata_batch.obs)
    except AssertionError as exc:
        raise AssertionError(f"{tag}: {exc}") from exc
    adata_batch.obs[cell_type_mn_key] = cell_type_mn
    class_counts = label_stats["class_counts"]
    n_na_used = label_stats["n_is_mn_v2_na"]

    # ---- slim: obs to the label-contract columns, layers to counts, obsm to
    # ---- spatial (these depend on cell_type_mn / the subset, so they stay here)
    missing_keep = [c for c in keep_obs if c not in adata_batch.obs.columns]
    if missing_keep:
        raise KeyError(f"{tag}: --keep-obs-extra names absent obs columns: "
                       f"{missing_keep}")
    dropped_obs = [c for c in adata_batch.obs.columns if c not in keep_obs]
    adata_batch.obs = adata_batch.obs[keep_obs].copy()
    dropped_layers = [k for k in list(adata_batch.layers.keys())
                      if k != counts_key]
    for k in dropped_layers:
        del adata_batch.layers[k]
    dropped_obsm = [k for k in list(adata_batch.obsm.keys())
                    if k != spatial_key]
    for k in dropped_obsm:
        del adata_batch.obsm[k]
    if not slim_logged:
        log(f"slimming per-sample objects — obs {obs_cols_before} cols in the "
            f"file (+1 synthesised {cell_type_mn_key}) -> {len(keep_obs)} kept "
            f"(dropped {len(dropped_obs)}), "
            f"layers dropped={dropped_layers}, obsm dropped={dropped_obsm}, "
            f"obsp dropped={dropped_obsp}, uns dropped={len(dropped_uns)} keys, "
            f"varm dropped={dropped_varm}, varp dropped={dropped_varp} "
            "(obsp/uns/varm/varp cleared before the subset copy)")
        slim_logged = True

    # ---- counts provenance: CSC -> CSR + integer-valued check --------------
    counts_mat = adata_batch.layers[counts_key]
    if sp.issparse(counts_mat):
        if not sp.isspmatrix_csr(counts_mat):
            adata_batch.layers[counts_key] = counts_mat.tocsr()
        head = adata_batch.layers[counts_key].data[:100_000]
    else:
        head = np.asarray(counts_mat).ravel()[:100_000]
    if head.size and np.any(head != np.round(head)):
        raise ValueError(f"{tag}: layers['{counts_key}'] is NOT integer-valued "
                         "— provenance broken (normalised data in the counts "
                         "layer?). Refusing to train.")
    if head.size and np.any(head < 0):
        raise ValueError(f"{tag}: layers['{counts_key}'] has negative values")
    # X stays put (log1p, sparse): models/nichecompass.py:334 derives
    # features_scale_factors_ from adata.X.sum(0)[0], which needs X sparse.
    if sp.issparse(adata_batch.X) and not sp.isspmatrix_csr(adata_batch.X):
        adata_batch.X = adata_batch.X.tocsr()
    if not sp.issparse(adata_batch.X):
        raise ValueError(f"{tag}: adata.X is dense; 0.3.3 indexes "
                         "adata.X.sum(0)[0] and needs a sparse X")
    if i == 1:
        log(f"matrix types: X={type(adata_batch.X).__name__}, "
            f"counts={type(adata_batch.layers[counts_key]).__name__}")

    # ---- spatial sanity + 4-NN graph (sklearn replaces squidpy) ------------
    coords = np.asarray(adata_batch.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"{tag}: obsm['{spatial_key}'] has shape "
                         f"{coords.shape}, expected (n_cells, 2)")
    if not np.isfinite(coords).all():
        raise ValueError(f"{tag}: non-finite spatial coordinates")

    knn = kneighbors_graph(coords, n_neighbors=n_neighbors,
                           mode="connectivity", include_self=False, n_jobs=-1)
    knn = knn.maximum(knn.T).tocsr()  # symmetrise, as in the Sanger flow

    # obs_names -> '<sample>|<orig>': unique across samples, identical to the
    # cohort-concat convention, and identical across all four configurations so
    # a cell can be traced between C1/C2/C3/C4 outputs.
    adata_batch.obs_names = [f"{sample_label}|{cid}"
                             for cid in adata_batch.obs_names]

    adjacencies.append(knn)
    sample_labels.append(sample_label)
    adata_batch_list.append(adata_batch)
    section_stats.append({
        "tag": tag,
        "sample": sample_label,
        "status": status_label,
        "vh_v2_source": vh_source_label,
        "n_obs_file": n_obs_file,
        "n_obs_used": n_obs_used,
        "n_obs_cols_before": obs_cols_before,   # as read, before cell_type_mn
        "n_obs_cols_synthesised": 1,            # cell_type_mn
        "n_obs_cols_kept": len(keep_obs),
        "n_obs_cols_dropped": len(dropped_obs),
        "knn_nnz": int(knn.nnz),
        "knn_min_degree": int(knn.getnnz(axis=1).min()),
        "file_counts": file_counts,
        "label_contract": label_stats,
    })
    log(f"[{i:2d}/{len(tags)}] {tag} -> '{sample_label}' ({status_label}): "
        f"{n_obs_used:,} cells, knn nnz={knn.nnz:,}, "
        f"{MN_CLASS}={class_counts[MN_CLASS]:,} "
        f"({ismn_key} <NA>={n_na_used:,}), "
        f"{OTHER_NEURONS}={class_counts[OTHER_NEURONS]:,} "
        f"({time.time() - t0:.0f}s)")

if len(set(sample_labels)) != n_expected_samples:
    raise ValueError(f"Expected {n_expected_samples} unique sample labels, got "
                     f"{len(set(sample_labels))}: {sorted(set(sample_labels))}")


# --------------------------------------------------------------------------- #
section("[4/8] Assemble the training object + spatial graph (obsp only)")
# --------------------------------------------------------------------------- #
t0 = time.time()
block_sizes = [a.n_obs for a in adata_batch_list]
alignment_gates = {}

if len(adata_batch_list) == 1:
    # Per-sample configurations: no concat (nothing to align), so the object is
    # the loaded/subset section itself and the graph is its own kNN matrix.
    adata_combined = adata_batch_list[0]
    log(f"single-section object: {adata_combined.n_obs:,} cells x "
        f"{adata_combined.n_vars} genes (no concat)")
    if not adata_combined.obs_names.is_unique:
        raise AssertionError("obs_names are not unique after "
                             "'<sample>|<cell>' prefixing")
    if adata_combined.obs[sample_key].astype(str).nunique() != 1:
        raise AssertionError("single-section object carries more than one "
                             f"obs['{sample_key}'] value")
    alignment_gates["mode"] = "single_section (concat gates not applicable)"
    alignment_gates["obs_names_unique"] = True
else:
    # Module-level ad.concat (the instance .concatenate() is deprecated). obsp,
    # uns, varm and varp are dropped here — the graph is rebuilt block-
    # diagonally below; merge='same' keeps var columns (identical panels).
    adata_combined = ad.concat(adata_batch_list, join="inner", merge="same")
    log(f"ad.concat done: {adata_combined.n_obs:,} cells x "
        f"{adata_combined.n_vars} genes ({time.time() - t0:.0f}s)")
    if adata_combined.n_vars != EXPECTED_N_GENES:
        raise ValueError(f"Cohort has {adata_combined.n_vars} genes after inner "
                         f"join, expected {EXPECTED_N_GENES} — per-sample var "
                         "drift, aborting.")
    assert adata_combined.n_obs == sum(block_sizes)

    # ---- alignment gates: concat row order MUST equal block order, otherwise
    # ---- the block-diagonal graph would silently wire the wrong cells together
    expected_sample_vec = np.repeat(np.array(sample_labels, dtype=object),
                                    block_sizes)
    got_sample_vec = adata_combined.obs[sample_key].astype(str).to_numpy()
    if not np.array_equal(got_sample_vec, expected_sample_vec):
        raise AssertionError("obs['sample'] order after ad.concat does not "
                             "match the per-sample block order")

    expected_obs_names = np.concatenate(
        [a.obs_names.to_numpy() for a in adata_batch_list])
    if not np.array_equal(adata_combined.obs_names.to_numpy(),
                          expected_obs_names):
        raise AssertionError("obs_names order after ad.concat does not match "
                             "the per-sample block order")
    if not adata_combined.obs_names.is_unique:
        raise AssertionError("obs_names are not unique after "
                             "'<sample>|<cell>' prefixing")

    expected_spatial = np.vstack(
        [np.asarray(a.obsm[spatial_key]) for a in adata_batch_list])
    if not np.array_equal(np.asarray(adata_combined.obsm[spatial_key]),
                          expected_spatial):
        raise AssertionError("obsm['spatial'] rows after ad.concat do not "
                             "match the per-sample block order")
    log("alignment gates PASSED (sample labels, obs_names, spatial coords)")
    alignment_gates["mode"] = "block_diagonal"
    alignment_gates["sample_label_order"] = True
    alignment_gates["obs_names_order"] = True
    alignment_gates["obs_names_unique"] = True
    alignment_gates["spatial_row_order"] = True
    del expected_spatial, expected_obs_names, expected_sample_vec, got_sample_vec

if expected_n_cells is not None and adata_combined.n_obs != expected_n_cells:
    warn(f"{adata_combined.n_obs:,} cells assembled, manifest expected "
         f"{expected_n_cells:,} — the pass2 inputs or the preflight counts may "
         "have drifted (continuing).")

# Block-diagonal assembly (a single-block block_diag is a no-op copy, so the
# same gate runs for every configuration)
A = (adjacencies[0] if len(adjacencies) == 1
     else sp.block_diag(adjacencies, format="csr").tocsr())
if A.shape != (adata_combined.n_obs, adata_combined.n_obs):
    raise AssertionError(f"adjacency shape {A.shape} != "
                         f"({adata_combined.n_obs}, {adata_combined.n_obs})")
if A.nnz != sum(m.nnz for m in adjacencies):
    raise AssertionError("block-diagonal adjacency lost or gained edges")
if (A - A.T).count_nonzero() != 0:
    raise AssertionError("spatial adjacency is not symmetric")
min_degree = int(A.getnnz(axis=1).min())
if min_degree < n_neighbors:
    raise AssertionError(f"adjacency min degree {min_degree} < "
                         f"n_neighbors={n_neighbors}")
alignment_gates["adjacency_shape_nnz_symmetry_min_degree"] = True

# 0.3.3 reads the graph from adata.obsp[adj_key] ONLY (init hard-checks it at
# models/nichecompass.py:528) and never from obsm.
adata_combined.obsp[adj_key] = A
log(f"obsp['{adj_key}']: shape={A.shape}, nnz={A.nnz:,}, "
    f"min_degree={min_degree}, mean_degree={A.nnz / A.shape[0]:.2f}")

# ---- categorical covariates ----------------------------------------------- #
if scope == "integrated":
    # models/nichecompass.py:472-474 derives the category list from
    # adata.obs['sample'].unique(), so an unused level would desync cats_ from
    # cat_covariates_embeds_nums. ad.concat unions the levels (alphabetically),
    # never positionally — drop anything unused and count from the data.
    adata_combined.obs[sample_key] = (
        adata_combined.obs[sample_key].astype("category")
        .cat.remove_unused_categories())
    n_samples_found = int(adata_combined.obs[sample_key].nunique())
    if n_samples_found != n_expected_samples:
        raise ValueError(f"{n_samples_found} unique obs['{sample_key}'] values "
                         f"in the assembled object, expected "
                         f"{n_expected_samples}")
    if set(adata_combined.obs[sample_key].cat.categories) != set(sample_labels):
        raise AssertionError("obs['sample'] categories do not match the loaded "
                             "sample labels")
    run_cat_covariates_keys = [sample_key]
    run_cat_covariates_no_edges = [True]
    run_cat_covariates_embeds_nums = [n_samples_found]
else:
    # nichecompass 0.3.3 spells "no categorical covariate" as None, NOT [].
    # An empty list survives __init__ and then blows up minutes later inside
    # Trainer (data/datasets.py:109 torch.stack([]) -> RuntimeError: stack
    # expects a non-empty TensorList). With None, cat_covariates_cats_ becomes
    # [], embeds_nums becomes [], no_edges_ stays None and is never iterated
    # (its only consumer sits inside an `if len(cat_covariates_cats_) > 0`
    # branch), and both RNA decoders report n_cat_covariates_embed_input: 0.
    n_samples_found = 1
    run_cat_covariates_keys = None
    run_cat_covariates_no_edges = None
    run_cat_covariates_embeds_nums = None
log(f"cat_covariates_keys={run_cat_covariates_keys}, "
    f"cat_covariates_embeds_nums={run_cat_covariates_embeds_nums}, "
    f"cat_covariates_no_edges={run_cat_covariates_no_edges}")

# ---- run-level label-contract roll-up ------------------------------------- #
cell_type_mn_series = adata_combined.obs[cell_type_mn_key].astype("category")
if list(cell_type_mn_series.cat.categories) != CELL_TYPE_MN_CLASSES:
    # ad.concat unions categories; re-pin the fixed order so every run's
    # crosstabs and legends are identically shaped.
    cell_type_mn_series = pd.Categorical(
        cell_type_mn_series.astype(str).to_numpy(),
        categories=CELL_TYPE_MN_CLASSES, ordered=False)
    if (np.asarray(cell_type_mn_series.codes) < 0).any():
        raise AssertionError("cell_type_mn stopped tiling every cell after "
                             "assembly")
    adata_combined.obs[cell_type_mn_key] = cell_type_mn_series
run_class_counts = {c: int(n) for c, n in
                    pd.Series(adata_combined.obs[cell_type_mn_key])
                    .value_counts().reindex(CELL_TYPE_MN_CLASSES,
                                            fill_value=0).items()}
run_codes_v2 = nullable_bool_codes(adata_combined.obs[ismn_key])
n_na_is_mn_v2 = int((run_codes_v2 == -1).sum())
n_true_is_mn_v2 = int((run_codes_v2 == 1).sum())
if run_class_counts[MN_CLASS] != n_true_is_mn_v2:
    raise AssertionError(
        f"label contract broken: {run_class_counts[MN_CLASS]} cells labelled "
        f"'{MN_CLASS}' but {n_true_is_mn_v2} cells have {ismn_key} == True")
if sum(run_class_counts.values()) != adata_combined.n_obs:
    raise AssertionError("cell_type_mn class counts do not sum to n_obs")
na_samples = sorted(
    str(s) for s, frac in pd.DataFrame({
        "sample": adata_combined.obs[sample_key].astype(str).to_numpy(),
        "na": (run_codes_v2 == -1)}).groupby("sample", observed=True)["na"]
    .mean().items() if frac == 1.0)
log(f"{cell_type_mn_key} class counts: {run_class_counts}")
log(f"{ismn_key}: True={n_true_is_mn_v2:,}, <NA>={n_na_is_mn_v2:,} "
    f"(fully-<NA> sections: {na_samples if na_samples else 'none'})")
log(f"'{MN_CLASS}' now means {ismn_key} == True and nothing else; the coarse "
    f"transcriptional cluster is '{OTHER_NEURONS}' "
    f"({run_class_counts[OTHER_NEURONS]:,} cells)")
log(f"obs['{status_key}'] value counts: "
    f"{adata_combined.obs[status_key].astype(str).value_counts().to_dict()}")

# Free the per-sample copies (keep sample_labels/block_sizes for the manifest)
if len(adata_batch_list) > 1:
    del adata_batch_list
else:
    adata_batch_list = None
del adjacencies
gc.collect()


# --------------------------------------------------------------------------- #
section(f"[5/8] GP masks + survival gate ({EXPECTED_N_GENES}-gene panel)")
# --------------------------------------------------------------------------- #
n_gps_before = len(combined_gp_dict)
add_gps_from_gp_dict_to_adata(
    gp_dict=combined_gp_dict,
    adata=adata_combined,
    gp_targets_mask_key=gp_targets_mask_key,
    gp_targets_categories_mask_key=gp_targets_categories_mask_key,
    gp_sources_mask_key=gp_sources_mask_key,
    gp_sources_categories_mask_key=gp_sources_categories_mask_key,
    gp_names_key=gp_names_key,
    min_genes_per_gp=2,
    min_source_genes_per_gp=1,
    min_target_genes_per_gp=1,
    max_genes_per_gp=None,
    max_source_genes_per_gp=None,
    max_target_genes_per_gp=None,
)
plt.close("all")
n_gps_after = int(len(adata_combined.uns[gp_names_key]))
log(f"GPs before the min-genes filters (2/1/1): {n_gps_before}")
log(f"GPs surviving on the {EXPECTED_N_GENES}-gene panel: {n_gps_after} "
    f"({n_gps_after / max(n_gps_before, 1):.1%})")
log("GP survival is gene-based, so it is identical for every configuration and "
    "every cell subset (the 2026-08-26 reference run got 77 from 1333 with "
    "NicheNet + MEBOCOST only).")
if n_gps_after < MIN_SURVIVING_GPS:
    raise RuntimeError(
        f"GP-survival gate FAILED: only {n_gps_after} GPs survived the "
        f"min-genes filters (< MIN_SURVIVING_GPS={MIN_SURVIVING_GPS}). Most "
        "GPs failing is expected on a 480-gene panel, but this few cannot "
        "carry a meaningful model. Inspect the figures/ GP-distribution PNGs "
        "and the skipped-source warnings above. Override with NC_MIN_GPS if "
        "deliberate.")
log("GP-survival gate PASSED")

# The GP dicts are dead weight from here on (the masks live in adata.varm/uns).
# Freeing them matters more than it did for one integrated job: up to 6 array
# tasks hold a NicheNet dict apiece on the shared node.
del combined_gp_dict, gp_dicts, gp_source_dicts
del omnipath_gp_dict, nichenet_gp_dict, mebocost_gp_dict
gc.collect()


# --------------------------------------------------------------------------- #
section("[6/8] NicheCompass model init + training")
# --------------------------------------------------------------------------- #
edge_batch_size, node_batch_size, n_undirected_edges, n_edges_train_est, \
    n_nodes_train_est = size_batches(A.nnz, adata_combined.n_obs)
steps_per_epoch = max(1, int(np.ceil(n_edges_train_est / edge_batch_size)))
degree_ratio = (n_edges_train_est / n_nodes_train_est
                if n_nodes_train_est else float("nan"))
log(f"graph: {n_undirected_edges:,} undirected edges "
    f"(~{n_edges_train_est:,} train / ~{n_nodes_train_est:,} train nodes, "
    f"edges/nodes ~{degree_ratio:.2f})")
log(f"edge_batch_size={edge_batch_size} (ceiling {EDGE_BATCH_CEILING} <- "
    f"{EDGE_BATCH_CEILING_SOURCE}), node_batch_size={node_batch_size} "
    f"(explicit — skips the auto formula), ~{steps_per_epoch} edge batches/epoch")
if edge_batch_size < EDGE_BATCH_CEILING:
    log(f"NOTE: edge_batch_size capped below the ceiling because this input "
        f"only has ~{n_edges_train_est:,} training edges; at {EDGE_BATCH_CEILING} "
        "the whole epoch would be a single gradient step.")
if degree_ratio < 2.2:
    warn(f"edges/nodes ratio {degree_ratio:.2f} is below ~2.2. Harmless here "
         "because node_batch_size is explicit, but the auto formula at "
         "train/trainer.py:194-196 would have raised ZeroDivisionError.")

early_stopping_kwargs = {
    "early_stopping_metric": "val_global_loss",
    "metric_improvement_threshold": 0.,
    "patience": int(es_patience),
    "reduce_lr_on_plateau": True,
    "lr_patience": int(es_lr_patience),
    "lr_factor": 0.1,
}

t0 = time.time()
model = NicheCompass(
    adata_combined,
    counts_key=counts_key,
    adj_key=adj_key,
    cat_covariates_embeds_injection=cat_covariates_embeds_injection,
    cat_covariates_keys=run_cat_covariates_keys,
    cat_covariates_no_edges=run_cat_covariates_no_edges,
    cat_covariates_embeds_nums=run_cat_covariates_embeds_nums,
    gp_names_key=gp_names_key,
    active_gp_names_key=active_gp_names_key,
    gp_targets_mask_key=gp_targets_mask_key,
    gp_targets_categories_mask_key=gp_targets_categories_mask_key,
    gp_sources_mask_key=gp_sources_mask_key,
    gp_sources_categories_mask_key=gp_sources_categories_mask_key,
    latent_key=latent_key,
    conv_layer_encoder=conv_layer_encoder,
    active_gp_thresh_ratio=active_gp_thresh_ratio,
    active_gp_type=active_gp_type,
    node_label_method=node_label_method,
    gene_expr_recon_dist=gene_expr_recon_dist,
    n_addon_gp=n_addon_gp,
    n_fc_layers_encoder=n_fc_layers_encoder,
    n_layers_encoder=n_layers_encoder,
    encoder_n_attention_heads=encoder_n_attention_heads,
    log_variational=log_variational,
    include_edge_kl_loss=include_edge_kl_loss,
    use_cuda_if_available=use_cuda_if_available,
    seed=SEED,
)
log(f"model initialised ({time.time() - t0:.0f}s)")
# n_cat_covariates_ lives on the VGPGAE module (modules/vgpgae.py:228), not on
# the NicheCompass wrapper. Assert it is 0 for the per-sample configurations —
# that is the whole point of cat_covariates_keys=None.
n_cat_covariates_built = getattr(getattr(model, "model", None),
                                 "n_cat_covariates_", None)
log(f"n_prior_gp={getattr(model, 'n_prior_gp_', 'n/a')}, "
    f"n_addon_gp={getattr(model, 'n_addon_gp_', 'n/a')}, "
    f"n_hidden_encoder={getattr(model, 'n_hidden_encoder_', 'n/a')}, "
    f"n_cat_covariates={n_cat_covariates_built}, "
    f"cat_covariates_cats_={getattr(model, 'cat_covariates_cats_', 'n/a')}")
expected_n_cat_cov = 1 if scope == "integrated" else 0
if n_cat_covariates_built is not None and \
        int(n_cat_covariates_built) != expected_n_cat_cov:
    raise AssertionError(
        f"the built model has {n_cat_covariates_built} categorical covariate(s), "
        f"expected {expected_n_cat_cov} for config {config}")

log(f"training: n_epochs={n_epochs} (ceiling — 0.3.3 early stopping is ON by "
    f"default), edge_batch_size={edge_batch_size}, "
    f"node_batch_size={node_batch_size}, "
    f"n_sampled_neighbors={n_sampled_neighbors}, device={device_name}")
t_train0 = time.time()
model.train(
    n_epochs=n_epochs,
    n_epochs_all_gps=n_epochs_all_gps,
    lr=lr,
    lambda_edge_recon=lambda_edge_recon,
    lambda_gene_expr_recon=lambda_gene_expr_recon,
    lambda_l1_masked=lambda_l1_masked,
    lambda_l1_addon=lambda_l1_addon,
    edge_batch_size=edge_batch_size,
    node_batch_size=node_batch_size,
    edge_val_ratio=edge_val_ratio,
    node_val_ratio=node_val_ratio,
    n_sampled_neighbors=n_sampled_neighbors,
    latent_dtype=latent_dtype,
    use_cuda_if_available=use_cuda_if_available,
    # --- **trainer_kwargs -> Trainer(...) --------------------------------- #
    early_stopping_kwargs=early_stopping_kwargs,
    use_early_stopping=True,
    reload_best_model=True,
    monitor=monitor,
    verbose=False,
    seed=SEED,
)
train_seconds = time.time() - t_train0
log(f"training finished in {train_seconds / 3600:.2f} h "
    f"({train_seconds / 60:.1f} min)")

# Trainer.__init__ has a bare **kwargs, so a MISSPELLED trainer kwarg is
# silently swallowed. Verify that the early-stopping settings actually landed —
# NON-FATALLY. Training is finished at this point but model.save() has not run
# yet, so nothing here may raise: a mismatch is a banner warning that is
# recorded in the manifest (training.early_stopping_mismatch), never a reason
# to discard a 72-min run. (Against the installed 0.3.3 it cannot fire anyway:
# trainer.py:143 does EarlyStopping(**early_stopping_kwargs_), so a misspelled
# key is a TypeError at Trainer init, long before training.)
trainer = getattr(model, "trainer", None)
train_report = {}
if trainer is not None:
    try:
        es = getattr(trainer, "early_stopping", None)
        if es is None:
            warn("model.trainer.early_stopping is absent — cannot verify that "
                 "early_stopping_kwargs were applied")
            train_report["early_stopping_verified"] = None
        else:
            got = {"early_stopping_metric":
                       getattr(es, "early_stopping_metric", None),
                   "patience": _safe_int(getattr(es, "patience", None)),
                   "lr_patience": _safe_int(getattr(es, "lr_patience", None)),
                   "lr_factor": getattr(es, "lr_factor", None),
                   "metric_improvement_threshold":
                       getattr(es, "metric_improvement_threshold", None),
                   "reduce_lr_on_plateau":
                       getattr(es, "reduce_lr_on_plateau", None)}
            mismatch = {k: {"requested": early_stopping_kwargs[k],
                            "observed": got[k]}
                        for k in early_stopping_kwargs
                        if k in got and got[k] != early_stopping_kwargs[k]}
            train_report["early_stopping_observed"] = got
            train_report["early_stopping_verified"] = not mismatch
            if mismatch:
                train_report["early_stopping_mismatch"] = mismatch
                banner_warn([
                    "early_stopping_kwargs did NOT all reach EarlyStopping "
                    f"(requested vs observed: {mismatch}).",
                    "Trainer.__init__ absorbs unknown kwargs silently — check "
                    "the spelling. Recorded in the manifest as "
                    "training.early_stopping_mismatch; the trained model is "
                    "saved regardless."])
            else:
                log(f"early stopping verified: {got}")
    except Exception:
        warn("early-stopping verification itself failed (non-fatal): "
             f"{traceback.format_exc().splitlines()[-1]}")
        train_report["early_stopping_verified"] = None
    try:
        epochs_run = int(trainer.epoch) + 1           # trainer.epoch is 0-based
        best_epoch = int(trainer.best_epoch) + 1      # ditto
        train_report["epochs_run"] = epochs_run
        train_report["best_epoch"] = best_epoch
        train_report["early_stopped"] = bool(epochs_run < n_epochs)
        log(f"epochs run: {epochs_run}/{n_epochs} "
            f"({'early-stopped' if epochs_run < n_epochs else 'ran to ceiling'}), "
            f"best epoch {best_epoch}")
    except Exception:
        warn("could not read trainer.epoch / trainer.best_epoch")
    try:
        logs = dict(trainer.epoch_logs)
        train_report["final_losses"] = {
            k: float(v[-1]) for k, v in logs.items()
            if isinstance(v, (list, tuple)) and len(v)}
        train_report["best_val_global_loss"] = (
            float(min(logs["val_global_loss"]))
            if logs.get("val_global_loss") else None)
        log(f"final losses: {train_report['final_losses']}")
        # Loss curves — one PNG per run, dpi 150, colours from tab10 modulo
        keys = [k for k in ("train_global_loss", "val_global_loss",
                            "train_optim_loss", "val_optim_loss")
                if logs.get(k)]
        if keys:
            cmap = matplotlib.colormaps["tab10"]
            fig, ax = plt.subplots(figsize=(8.0, 5.0))
            for j, k in enumerate(keys):
                ys = np.asarray(logs[k], dtype=float)
                ax.plot(np.arange(1, len(ys) + 1), ys, label=k,
                        color=cmap(j % 10), linewidth=1.4)
            if train_report.get("best_epoch"):
                ax.axvline(train_report["best_epoch"], color="0.4",
                           linestyle="--", linewidth=1.0,
                           label=f"best epoch {train_report['best_epoch']}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{config}"
                         f"{' / ' + args.tag if is_persample else ''} "
                         f"({adata_combined.n_obs:,} cells)")
            ax.legend(fontsize=8, frameon=False)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            fig.savefig(f"{figure_folder_path}/training_loss_curves.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            log("loss curves -> figures/training_loss_curves.png")
    except Exception:
        warn("could not build the loss-curve figure: "
             f"{traceback.format_exc().splitlines()[-1]}")
    train_report["node_batch_size_resolved"] = int(
        getattr(trainer, "node_batch_size_", node_batch_size))


# --------------------------------------------------------------------------- #
section("[7/8] Save model + adata")
# --------------------------------------------------------------------------- #
t0 = time.time()
model.save(
    dir_path=model_folder_path,
    overwrite=True,
    save_adata=True,
    adata_file_name=adata_file_name,
)
h5ad_path = f"{model_folder_path}/{adata_file_name}"
log(f"model + adata saved -> {model_folder_path} ({time.time() - t0:.0f}s)")
h5ad_gb = None
try:
    h5ad_gb = round(os.path.getsize(h5ad_path) / 1024**3, 3)
    log(f"{adata_file_name}: {h5ad_gb:.3f} GB")
except OSError:
    pass

# Best-effort post-save extras — never jeopardise the saved artifact
n_active_gps = None
try:
    active_gps = model.get_active_gps()
    n_active_gps = int(len(active_gps))
    with open(f"{result_folder_path}/active_gp_names.txt", "w") as fh:
        fh.write("\n".join(map(str, active_gps)) + "\n")
    log(f"active GPs after training: {n_active_gps} of {n_gps_after} prior + "
        f"{n_addon_gp} add-on (list -> results/active_gp_names.txt)")
except Exception:
    warn("could not retrieve active GPs: "
         f"{traceback.format_exc().splitlines()[-1]}")

# The label-contract roll-up as a CSV too — the downstream stages read it, and
# it is the surface a reviewer checks first.
try:
    rows = []
    for cls in CELL_TYPE_MN_CLASSES:
        if cls == MN_CLASS:
            definition = f"{ismn_key} == True (wins over every base class)"
        elif cls == OTHER_NEURONS:
            definition = (f"coarse transcriptional '{COARSE_MN_CAT}' class, "
                          f"demoted; NOT motor neurons")
        else:
            definition = f"coarse transcriptional '{cls}' class, unchanged"
        n = run_class_counts[cls]
        rows.append({
            "cell_class": cls,
            "n_cells": n,
            "pct": (round(100.0 * n / adata_combined.n_obs, 4)
                    if adata_combined.n_obs else 0.0),
            "definition": definition,
            "status": ("n/a (no VH annotation in this run)"
                       if (cls == MN_CLASS
                           and n_na_is_mn_v2 == adata_combined.n_obs)
                       else "observed"),
        })
    pd.DataFrame(rows).to_csv(
        f"{result_folder_path}/cohort_composition_cell_type_mn.csv", index=False)
    pd.DataFrame(section_stats).to_json(
        f"{result_folder_path}/section_stats.json", orient="records", indent=2)
    log("label-contract tables -> results/cohort_composition_cell_type_mn.csv "
        "+ results/section_stats.json")
except Exception:
    warn("could not write the label-contract tables: "
         f"{traceback.format_exc().splitlines()[-1]}")


# --------------------------------------------------------------------------- #
section("[8/8] Run manifest")
# --------------------------------------------------------------------------- #
script_path = os.path.abspath(__file__)
manifest = {
    "config": config,
    "scope": scope,
    "subset": subset_mode,
    "runstamp": runstamp,
    "tag": args.tag,
    "tags": list(tags),
    "provenance": {
        "script": script_path,
        "script_sha256": sha256_of(script_path),
        "script_mtime_utc": datetime.fromtimestamp(
            os.path.getmtime(script_path), tz=timezone.utc).isoformat(),
        "argv": shlex.join(sys.argv),
        "cwd_at_exit": os.getcwd(),
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "started_utc": datetime.fromtimestamp(
            SCRIPT_T0, tz=timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "slurm": {k: os.environ.get(k) for k in (
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
            "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE", "SLURM_JOB_GPUS", "SLURM_SUBMIT_DIR")},
        "env": {k: os.environ.get(k) for k in (
            "NC_EDGE_BATCH", "NC_MIN_GPS", "NC_ALLOW_CPU",
            "NC_REQUIRE_GP_CACHE", "PYTORCH_CUDA_ALLOC_CONF",
            "OMP_NUM_THREADS", "LD_LIBRARY_PATH", "PYTHONNOUSERSITE",
            "CUDA_VISIBLE_DEVICES")},
        "versions": versions,
        "seed": SEED,
        "tag_manifest": tag_manifest["path"],
        "tag_manifest_sha256": sha256_of(tag_manifest["path"]),
    },
    "device": device_name,
    "gpu": gpu_info,
    "inputs": {
        "sample_data_folder": sample_data_folder,
        "file_pattern": "ALS_SCXenium_<TAG>_pass2.h5ad",
        "sample_tags": list(tags),
        "sample_labels": sample_labels,
        "block_sizes": [int(b) for b in block_sizes],
        "unexpected_pass2_files_ignored": _unexpected,
        "expected_n_cells_from_manifest": expected_n_cells,
        "exclude_cells_file": args.exclude_cells,
        "exclude_cells_meta": exclude_meta,
        "subset_rule": ("whole section (no row filter)" if subset_mode == "whole"
                        else f"obs['{vh_key}'].astype('boolean').fillna(False) "
                             "== True"),
    },
    "cohort": {
        "n_cells": int(adata_combined.n_obs),
        "n_genes": int(adata_combined.n_vars),
        "n_samples": int(n_samples_found),
        "status_counts": {str(k): int(v) for k, v in
                          adata_combined.obs[status_key].astype(str)
                          .value_counts().items()},
    },
    "label_contract": {
        "column": cell_type_mn_key,
        "rule": (f"obs['{cell_type_key}'] with '{COARSE_MN_CAT}' renamed "
                 f"'{OTHER_NEURONS}', then obs['{ismn_key}'] == True -> "
                 f"'{MN_CLASS}' (wins over every base class)"),
        "class_order": CELL_TYPE_MN_CLASSES,
        "class_counts": run_class_counts,
        "n_is_mn_v2_true": n_true_is_mn_v2,
        "n_na_is_mn_v2": n_na_is_mn_v2,
        "na_handling": (f"{ismn_key} <NA> cells are NOT '{MN_CLASS}'; they keep "
                        "their demoted base class, so the partition tiles "
                        "every cell"),
        "fully_na_samples": na_samples,
        "n_promoted_from_coarse_MN": sum(
            s["label_contract"]["n_promoted_from_coarse_MN"]
            for s in section_stats),
        "n_promoted_from_other_classes": sum(
            s["label_contract"]["n_promoted_from_other_classes"]
            for s in section_stats),
        "n_demoted_to_other_neurons": sum(
            s["label_contract"]["n_demoted_to_other_neurons"]
            for s in section_stats),
        "n_coarse_transcriptional_mn": sum(
            s["label_contract"]["n_coarse_transcriptional_mn"]
            for s in section_stats),
        "motor_neuron_call_available": bool(
            n_na_is_mn_v2 < adata_combined.n_obs),
        "ismn_alias_asserted_equal": f"{ismn_alias_key} == {ismn_key} "
                                     "(bit-for-bit, per section)",
        "obs_columns_kept": keep_obs,
        "obs_columns_kept_n": len(keep_obs),
        "obs_columns_synthesised": [cell_type_mn_key],
        # measured as read, BEFORE cell_type_mn is attached (171 on 2026-09-02)
        "obs_columns_in_pass2_file": int(section_stats[0]["n_obs_cols_before"]),
        "obs_columns_dropped_per_section": int(
            section_stats[0]["n_obs_cols_dropped"]),
    },
    "sections": section_stats,
    "spatial_graph": {
        "method": ("sklearn.kneighbors_graph connectivity, symmetrised "
                   "(A.maximum(A.T)), " +
                   ("single section" if len(tags) == 1
                    else "block-diagonal per sample")),
        "spatial_key": spatial_key,
        "n_neighbors": n_neighbors,
        "nnz": int(A.nnz),
        "n_undirected_edges": int(n_undirected_edges),
        "min_degree": min_degree,
        "mean_degree": round(float(A.nnz / A.shape[0]), 4),
        "adj_key": adj_key,
        "stored_in": "obsp",
        "alignment_gates": alignment_gates,
    },
    "gene_programs": {
        "per_source": gp_source_counts,
        "skipped_sources": skipped_gp_sources,
        "gp_cache_dir": gp_data_folder_path,
        "gp_cache_missing": missing_cache,
        "n_combined": n_gps_before,
        "n_after_min_genes_filter": n_gps_after,
        "min_genes_per_gp": 2,
        "min_source_genes_per_gp": 1,
        "min_target_genes_per_gp": 1,
        "min_surviving_gps_gate": MIN_SURVIVING_GPS,
        "n_active_after_training": n_active_gps,
    },
    "hyperparams": {
        "conv_layer_encoder": conv_layer_encoder,
        "active_gp_thresh_ratio": active_gp_thresh_ratio,
        "active_gp_type": active_gp_type,
        "node_label_method": node_label_method,
        "gene_expr_recon_dist": gene_expr_recon_dist,
        "n_addon_gp": n_addon_gp,
        "n_fc_layers_encoder": n_fc_layers_encoder,
        "n_layers_encoder": n_layers_encoder,
        "encoder_n_attention_heads": encoder_n_attention_heads,
        "log_variational": log_variational,
        "include_edge_kl_loss": include_edge_kl_loss,
        "n_epochs": n_epochs,
        "n_epochs_all_gps": n_epochs_all_gps,
        "lr": lr,
        "lambda_edge_recon": lambda_edge_recon,
        "lambda_gene_expr_recon": lambda_gene_expr_recon,
        "lambda_l1_masked": lambda_l1_masked,
        "lambda_l1_addon": lambda_l1_addon,
        "edge_batch_size": edge_batch_size,
        "edge_batch_ceiling": EDGE_BATCH_CEILING,
        "edge_batch_ceiling_source": EDGE_BATCH_CEILING_SOURCE,
        "edge_batch_default_when_unset": EDGE_BATCH_DEFAULT,
        "node_batch_size": node_batch_size,
        "edge_batches_per_epoch_est": steps_per_epoch,
        "edge_val_ratio": edge_val_ratio,
        "node_val_ratio": node_val_ratio,
        "n_sampled_neighbors": n_sampled_neighbors,
        "latent_dtype": str(np.dtype(latent_dtype)),
        "counts_key": counts_key,
        "cat_covariates_keys": run_cat_covariates_keys,
        "cat_covariates_embeds_nums": run_cat_covariates_embeds_nums,
        "cat_covariates_no_edges": run_cat_covariates_no_edges,
        "cat_covariates_embeds_injection": cat_covariates_embeds_injection,
        "early_stopping_kwargs": early_stopping_kwargs,
        "use_early_stopping": True,
        "reload_best_model": True,
        "monitor": monitor,
        "seed": SEED,
    },
    "training": train_report,
    "durations_s": {
        "train": round(train_seconds),
        "total": round(time.time() - SCRIPT_T0),
    },
    "outputs": {
        "run_dir": run_folder_path,
        "run_dir_resolved": run_folder_resolved,
        "model_dir": model_folder_path,
        "adata": h5ad_path,
        "adata_gb": h5ad_gb,
        "figures": figure_folder_path,
        "results": result_folder_path,
        "latest_symlink": None,  # deliberately none: see the module docstring
    },
}
# ATOMIC and FATAL-on-failure. This file is the wrapper's sole "train finished"
# / "skip train on resume" signal, so (a) it must never exist half-written and
# (b) a failed write must surface as a non-zero exit — otherwise the trainer
# would exit 0 with a saved model but no manifest and the wrapper would mark
# the section FAILED and retrain it from scratch on the next submission.
try:
    write_json_atomic(manifest_path, manifest)
except Exception:
    banner_warn([f"FATAL: run manifest write FAILED: {manifest_path}",
                 traceback.format_exc().splitlines()[-1],
                 "The model/adata/results ARE on disk, but without a manifest "
                 "the run does not count as finished. Fix the cause (quota? "
                 "permissions?) and resubmit; the retry retrains this run."])
    raise
log(f"run manifest -> {manifest_path} (atomic write)")

print("", flush=True)
log("=" * 78)
log(f"ALL DONE — NicheCompass v2 training | config={config}"
    f"{' | tag=' + args.tag if is_persample else ''}")
log(f"  run dir : {run_folder_path}")
log(f"  model   : {model_folder_path}")
log(f"  adata   : {h5ad_path}")
log(f"  figures : {figure_folder_path}")
log(f"  results : {result_folder_path}")
log(f"  manifest: {manifest_path}")
log(f"  cells   : {adata_combined.n_obs:,} x {adata_combined.n_vars} genes, "
    f"{n_samples_found} sample(s)")
log(f"  {MN_CLASS} ({ismn_key} True): {run_class_counts[MN_CLASS]:,} | "
    f"{OTHER_NEURONS}: {run_class_counts[OTHER_NEURONS]:,} | "
    f"{ismn_key} <NA>: {n_na_is_mn_v2:,}")
log(f"  active GPs: {n_active_gps} | epochs: "
    f"{train_report.get('epochs_run', 'n/a')}/{n_epochs} "
    f"(best {train_report.get('best_epoch', 'n/a')})")
log(f"  total   : {(time.time() - SCRIPT_T0) / 3600:.2f} h")
log("=" * 78)
