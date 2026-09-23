#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/4_nichecompass/NC_SC_gp_precheck.py
#
# The go/no-go gate on the 480-gene panel question: how many gene programmes survive? It
# answered yes for the v1 port and it also builds the cache the v3 retrain still reads.
# The rest of the v1 lineage is superseded.
# ========================================================================================

"""
NC_SC_gp_precheck.py — gene-program (GP) survival go/no-go gate for the
NicheCompass ALS spinal-cord Xenium run (SCG SLURM port of the Sanger LSF
pipeline; base script: NC_training_all.py, ported 0.2.3 -> 0.3.3).

Why: the SC Xenium panel has only 480 genes, so most prior-knowledge GPs fail
the Sanger min-genes filters (min_genes_per_gp=2, min_source_genes_per_gp=1,
min_target_genes_per_gp=1). This script builds the exact GP masks training
will build, on a 1-cell dummy AnnData that carries only the panel var_names
(read via h5py — X is never touched), and counts survivors.

CPU-only. Exit contract (drives the sbatch afterok chain):
    exit 0 — >= MIN_SURVIVING_GPS GPs survive (even if thin: human decides)
    exit 1 — CATASTROPHIC: fewer survive, or the masks could not be built

Idempotent side effects:
    * first run fetches OmniPath live + downloads NicheNet v2 (zenodo) and
      saves CSV caches under <base>/gene_programs/ — reruns (and the training
      job) auto-switch to load_from_disk=True on the same paths
    * writes <base>/results/gp_precheck_surviving_gps.csv

0.3.3 port notes (vs the Sanger 0.2.3 calls):
    * extract_gp_dict_from_mebocost_MS_interactions (renamed from *_es_* in
      0.2.3); its TSVs must be staged first — run stage_gp_resources.sh
    * min_curation_effort=0 passed explicitly to the OmniPath extractor
      (0.3.3 default drifted to 2; keeps the 0.2.3-era permissive network)
    * all plot flags off; matplotlib forced to Agg regardless (headless)

Run from the deployment dir (never /tmp — a stray /tmp/struct.py shadows the
stdlib); the script chdir's to the oak gene_programs dir because the NicheNet
extractor writes temporary .rds files into the CWD.
"""

import os
import sys
import time
import traceback
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless backend BEFORE any pyplot import (landmine)

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# SLURM logs are block-buffered files: force line buffering so progress is
# visible in logs/*.out while the job runs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] importing nichecompass "
      "(cold import off /oak can take ~2.5 min)...")
_t_import = time.time()
from nichecompass.utils import (
    add_gps_from_gp_dict_to_adata,
    extract_gp_dict_from_mebocost_ms_interactions,
    extract_gp_dict_from_nichenet_lrt_interactions,
    extract_gp_dict_from_omnipath_lr_interactions,
    filter_and_combine_gp_dict_gps_v2,
)
print(f"nichecompass imported in {time.time() - _t_import:.1f}s")

### Dataset and GP Parameters ###
species = "human"

# Sanger-proven min/max-genes filters — MUST stay identical to the training
# script so this gate predicts training's GP survival exactly
min_genes_per_gp = 2
min_source_genes_per_gp = 1
min_target_genes_per_gp = 1
max_genes_per_gp = None
max_source_genes_per_gp = None
max_target_genes_per_gp = None

# Go/no-go threshold: below this a 480-gene-panel run is pointless
min_surviving_gps = 10
# Advisory (exit 0 either way): flag a thin GP space to the human
thin_surviving_gps = 50

# GP mask keys (same as training)
gp_names_key = "nichecompass_gp_names"
gp_targets_mask_key = "nichecompass_gp_targets"
gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
gp_sources_mask_key = "nichecompass_gp_sources"
gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"

# Folder paths (oak scaffold — created by stage_gp_resources.sh; makedirs
# below are belt-and-braces)
base_folder = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/NicheCompass_SC"
ga_data_folder_path = f"{base_folder}/gene_annotations"
gp_data_folder_path = f"{base_folder}/gene_programs"
result_folder_path = f"{base_folder}/results"
omnipath_lr_network_file_path = f"{gp_data_folder_path}/omnipath_lr_network.csv"
nichenet_lr_network_file_path = f"{gp_data_folder_path}/nichenet_lr_network.csv"
nichenet_ligand_target_matrix_file_path = (
    f"{gp_data_folder_path}/nichenet_ligand_target_matrix.csv")
mebocost_enzyme_sensor_interactions_folder_path = (
    f"{gp_data_folder_path}/metabolite_enzyme_sensor_gps")
# Only read by the extractors for species='mouse'; kept for signature parity
gene_orthologs_mapping_file_path = (
    f"{ga_data_folder_path}/human_mouse_gene_orthologs.csv")
surviving_gps_csv_path = f"{result_folder_path}/gp_precheck_surviving_gps.csv"

# Input: the 20 QCed per-sample pass2 h5ads (all carry the same 480-gene panel)
ranger_folder_path = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Ranger_procd"
sample_tags = [
    "SD00614BG", "SD01015BG", "SD01115BG", "SD01320BG", "SD01413BA",
    "SD01616BI", "SD01620BI", "SD01623BI", "SD01915BG", "SD01922BI",
    "SD01923BI", "SD02022BI", "SD02622BI", "SD02818BG", "SD02913BA",
    "SD03522BG", "SD03614BG", "SD03914BG", "SD04219BI", "SD05413BG",
]
per_sample_h5ad_paths = [
    f"{ranger_folder_path}/ALS_SCXenium_{tag}_pass2.h5ad" for tag in sample_tags
]
expected_n_genes = 480

os.makedirs(ga_data_folder_path, exist_ok=True)
os.makedirs(gp_data_folder_path, exist_ok=True)
os.makedirs(result_folder_path, exist_ok=True)

# NicheNet extraction writes temp .rds files into the CWD before removing
# them — park the process in the writable oak gene_programs dir (never /tmp)
os.chdir(gp_data_folder_path)
print(f"cwd set to {os.getcwd()} (NicheNet temp .rds files land here)")


def read_var_names(h5ad_path):
    """Read var_names straight from the h5ad var group — X is never touched."""
    with h5py.File(h5ad_path, "r") as f:
        var = f["var"]
        idx_key = var.attrs.get("_index", "_index")
        if isinstance(idx_key, bytes):
            idx_key = idx_key.decode("utf-8")
        raw = var[idx_key][:]
    return [g.decode("utf-8") if isinstance(g, bytes) else str(g) for g in raw]


def classify_gp_source(gp_name):
    """Infer the prior-knowledge source from the 0.3.3 GP naming scheme."""
    if gp_name.endswith("_ligand_receptor_target_gene_GP"):
        return "nichenet"
    if gp_name.endswith("_ligand_receptor_GP"):
        return "omnipath"
    if gp_name.endswith("_metabolite_enzyme_sensor_GP"):
        return "mebocost"
    if gp_name.endswith("_combined_GP"):
        return "combined"  # merged by filter_and_combine_gp_dict_gps_v2
    return "other"


print("\n### 1. Panel var_names from the 20 per-sample pass2 h5ads "
      "(h5py; X never read) ###")
t0 = time.time()
missing_h5ads = [p for p in per_sample_h5ad_paths if not os.path.isfile(p)]
if missing_h5ads:
    raise FileNotFoundError(
        "Missing per-sample pass2 h5ads:\n  " + "\n  ".join(missing_h5ads))

var_names_per_sample = {
    tag: read_var_names(path)
    for tag, path in zip(sample_tags, per_sample_h5ad_paths)
}
var_names = var_names_per_sample[sample_tags[0]]
ref_gene_set = set(var_names)

if len(ref_gene_set) != len(var_names):
    raise ValueError(f"Duplicated var_names in {per_sample_h5ad_paths[0]}")
mismatched_tags = [tag for tag, vn in var_names_per_sample.items()
                   if set(vn) != ref_gene_set]
if mismatched_tags:
    raise ValueError(
        "Per-sample gene panels differ from "
        f"{sample_tags[0]}: {mismatched_tags} — data contract broken.")

print(f"panel: {len(var_names)} genes, identical across all "
      f"{len(sample_tags)} samples ({time.time() - t0:.1f}s)")
print(f"  first/last genes: {var_names[:3]} ... {var_names[-3:]}")
if len(var_names) != expected_n_genes:
    print(f"WARNING: expected {expected_n_genes} genes, found "
          f"{len(var_names)} — proceeding, but check upstream QC outputs.")


print("\n### 2. GP dict extraction (OmniPath / NicheNet / MEBOCOST) ###")
gp_dict_by_source = {}
gp_source_errors = {}

# --- OmniPath L-R interactions (live web fetch; CSV-cached after first run) ---
omnipath_from_disk = (os.path.isfile(omnipath_lr_network_file_path)
                      and os.path.getsize(omnipath_lr_network_file_path) > 0)
try:
    t0 = time.time()
    omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
        species=species,
        min_curation_effort=0,  # 0.3.3 default drifted to 2 — keep permissive
        load_from_disk=omnipath_from_disk,
        save_to_disk=not omnipath_from_disk,
        lr_network_file_path=omnipath_lr_network_file_path,
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        plot_gp_gene_count_distributions=False,
    )
    gp_dict_by_source["omnipath"] = omnipath_gp_dict
    print(f"OmniPath: {len(omnipath_gp_dict)} GPs "
          f"(load_from_disk={omnipath_from_disk}, {time.time() - t0:.1f}s)")
except Exception:
    gp_source_errors["omnipath"] = traceback.format_exc()
    print("WARNING: OmniPath GP extraction FAILED — continuing without it.\n"
          + gp_source_errors["omnipath"])

# --- NicheNet v2 L-R-target GPs (zenodo download; CSV-cached after first run) ---
nichenet_from_disk = all(
    os.path.isfile(p) and os.path.getsize(p) > 0
    for p in (nichenet_lr_network_file_path,
              nichenet_ligand_target_matrix_file_path))
try:
    t0 = time.time()
    if not nichenet_from_disk:
        print("NicheNet CSV caches absent -> downloading from zenodo "
              "(ligand-target matrix is large; this can take a while)...")
    nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
        species=species,
        version="v2",
        keep_target_genes_ratio=1.,
        max_n_target_genes_per_gp=250,
        load_from_disk=nichenet_from_disk,
        save_to_disk=not nichenet_from_disk,
        lr_network_file_path=nichenet_lr_network_file_path,
        ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        plot_gp_gene_count_distributions=False,
    )
    gp_dict_by_source["nichenet"] = nichenet_gp_dict
    print(f"NicheNet: {len(nichenet_gp_dict)} GPs "
          f"(load_from_disk={nichenet_from_disk}, {time.time() - t0:.1f}s)")
except Exception:
    gp_source_errors["nichenet"] = traceback.format_exc()
    print("WARNING: NicheNet GP extraction FAILED — continuing without it.\n"
          + gp_source_errors["nichenet"])

# --- MEBOCOST metabolite-sensor GPs (local TSVs staged from the GitHub repo) ---
mebocost_tsv_paths = [
    f"{mebocost_enzyme_sensor_interactions_folder_path}/human_metabolite_enzymes.tsv",
    f"{mebocost_enzyme_sensor_interactions_folder_path}/human_metabolite_sensors.tsv",
]
try:
    t0 = time.time()
    missing_tsvs = [p for p in mebocost_tsv_paths
                    if not (os.path.isfile(p) and os.path.getsize(p) > 0)]
    if missing_tsvs:
        raise FileNotFoundError(
            "MEBOCOST enzyme/sensor TSVs not staged: "
            + ", ".join(missing_tsvs)
            + " — run stage_gp_resources.sh first (repo files, not in the wheel).")
    mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
        species=species,
        dir_path=mebocost_enzyme_sensor_interactions_folder_path,
        plot_gp_gene_count_distributions=False,
    )
    gp_dict_by_source["mebocost"] = mebocost_gp_dict
    print(f"MEBOCOST: {len(mebocost_gp_dict)} GPs ({time.time() - t0:.1f}s)")
except Exception:
    gp_source_errors["mebocost"] = traceback.format_exc()
    print("WARNING: MEBOCOST GP extraction FAILED — continuing without it.\n"
          + gp_source_errors["mebocost"])

for source_name, gp_dict in gp_dict_by_source.items():
    if not gp_dict:
        print(f"  {source_name}: 0 GPs extracted")
        continue
    example_gp_name, example_gp = next(iter(gp_dict.items()))
    print(f"  {source_name} example GP: {example_gp_name} "
          f"(n_sources={len(example_gp['sources'])}, "
          f"n_targets={len(example_gp['targets'])})")


print("\n### 3. Combine + filter GP dicts (filter_and_combine_gp_dict_gps_v2) ###")
# Sanger source order preserved: omnipath, nichenet, mebocost
gp_dicts = [gp_dict_by_source[s] for s in ("omnipath", "nichenet", "mebocost")
            if s in gp_dict_by_source]
combined_gp_dict = {}
if not gp_dicts:
    print("WARNING: no GP dict could be built from ANY source — "
          "0 gene programs go into the survival filter.")
else:
    try:
        t0 = time.time()
        combined_gp_dict = filter_and_combine_gp_dict_gps_v2(
            gp_dicts, verbose=True)
        print(f"Number of gene programs after filtering and combining: "
              f"{len(combined_gp_dict)} ({time.time() - t0:.1f}s).")
    except Exception:
        combined_gp_dict = {}
        print("WARNING: filter_and_combine_gp_dict_gps_v2 FAILED — "
              "treating as 0 combined GPs.\n" + traceback.format_exc())


print("\n### 4. Panel-survival filter (add_gps_from_gp_dict_to_adata; "
      "min-genes 2/1/1 as in training) ###")
# Minimal dummy AnnData: 1 cell x panel genes, X never used by the mask builder
adata_dummy = ad.AnnData(
    X=sp.csr_matrix((1, len(var_names)), dtype=np.float32),
    obs=pd.DataFrame(index=["dummy_cell"]),
    var=pd.DataFrame(index=pd.Index(var_names)),
)
masks_built = False
try:
    t0 = time.time()
    add_gps_from_gp_dict_to_adata(
        gp_dict=combined_gp_dict,
        adata=adata_dummy,
        gp_targets_mask_key=gp_targets_mask_key,
        gp_targets_categories_mask_key=gp_targets_categories_mask_key,
        gp_sources_mask_key=gp_sources_mask_key,
        gp_sources_categories_mask_key=gp_sources_categories_mask_key,
        gp_names_key=gp_names_key,
        min_genes_per_gp=min_genes_per_gp,
        min_source_genes_per_gp=min_source_genes_per_gp,
        min_target_genes_per_gp=min_target_genes_per_gp,
        max_genes_per_gp=max_genes_per_gp,
        max_source_genes_per_gp=max_source_genes_per_gp,
        max_target_genes_per_gp=max_target_genes_per_gp,
    )
    masks_built = True
    print(f"GP masks built on the dummy panel AnnData ({time.time() - t0:.1f}s)")
except Exception:
    print("WARNING: add_gps_from_gp_dict_to_adata FAILED — "
          "treating as 0 surviving GPs.\n" + traceback.format_exc())


print("\n### 5. Report + CSV + verdict ###")
n_surviving = 0
surviving_df = None
if masks_built:
    surviving_gp_names = [str(n) for n in adata_dummy.uns[gp_names_key]]
    sources_mask = np.asarray(adata_dummy.varm[gp_sources_mask_key])
    targets_mask = np.asarray(adata_dummy.varm[gp_targets_mask_key])
    if not (sources_mask.shape[1] == targets_mask.shape[1]
            == len(surviving_gp_names)):
        raise AssertionError(
            "GP mask/name misalignment: "
            f"sources {sources_mask.shape}, targets {targets_mask.shape}, "
            f"names {len(surviving_gp_names)}")
    n_source_genes_present = sources_mask.sum(axis=0).astype(int)
    n_target_genes_present = targets_mask.sum(axis=0).astype(int)
    surviving_df = pd.DataFrame({
        "gp_name": surviving_gp_names,
        "gp_source": [classify_gp_source(n) for n in surviving_gp_names],
        "n_source_genes_present": n_source_genes_present,
        "n_target_genes_present": n_target_genes_present,
        # source+target sum = the exact quantity min_genes_per_gp filters on
        "n_genes_present": n_source_genes_present + n_target_genes_present,
    }).sort_values(["n_genes_present", "gp_name"],
                   ascending=[False, True]).reset_index(drop=True)
    n_surviving = len(surviving_df)

    surviving_df.to_csv(surviving_gps_csv_path, index=False)
    print(f"surviving-GP table written: {surviving_gps_csv_path}")

    if n_surviving > 0:
        panel_genes_covered = int(
            ((sources_mask.sum(axis=1) + targets_mask.sum(axis=1)) > 0).sum())
        print(f"panel genes in >=1 surviving GP mask: "
              f"{panel_genes_covered}/{len(var_names)}")
        print("\n-- surviving GPs (stdout copy of the CSV) --")
        with pd.option_context("display.max_rows", None,
                               "display.max_columns", None,
                               "display.width", 200):
            print(surviving_df.to_string(index=False))

print(f"\n=== GP PRECHECK SUMMARY | {datetime.now():%Y-%m-%d %H:%M:%S} ===")
for source_name in ("omnipath", "nichenet", "mebocost"):
    if source_name in gp_dict_by_source:
        print(f"  {source_name:<9} GPs extracted (pre-filter): "
              f"{len(gp_dict_by_source[source_name])}")
    else:
        print(f"  {source_name:<9} EXTRACTION FAILED (see WARNING above)")
print(f"  combined after filter_and_combine_gp_dict_gps_v2: "
      f"{len(combined_gp_dict)}")
print(f"  surviving on the {len(var_names)}-gene panel "
      f"(min genes {min_genes_per_gp}/{min_source_genes_per_gp}/"
      f"{min_target_genes_per_gp}): {n_surviving}")
if n_surviving > 0:
    print("  surviving by source:")
    for source_name, count in surviving_df["gp_source"].value_counts().items():
        print(f"    {source_name}: {count}")

if n_surviving < min_surviving_gps:
    print(f"\nCATASTROPHIC: only {n_surviving} gene programs survive the "
          f"panel filters (< {min_surviving_gps}). The 480-gene panel cannot "
          "support this GP configuration — do NOT launch training. Exit 1.")
    sys.exit(1)

if gp_source_errors:
    print(f"\nNOTE: {len(gp_source_errors)} GP source(s) failed "
          f"({', '.join(sorted(gp_source_errors))}) — gate passed on the rest; "
          "fix and rerun if all three sources are wanted.")
if n_surviving < thin_surviving_gps:
    print(f"\nTHIN: only {n_surviving} surviving GPs "
          f"(< {thin_surviving_gps}) — training will run, but the GP space is "
          "narrow on this panel. Proceeding is a human call. Exit 0.")

print(f"\nGO: {n_surviving} gene programs survive the panel filters "
      f"(>= {min_surviving_gps}). Exit 0.")
sys.exit(0)
