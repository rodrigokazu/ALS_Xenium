# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/export_niche_data_joblib.py
#
# Exports the data behind the niche plots rather than a picture of them, for Marcel.
#
# Per sample it writes a dict of arrays: true-micron coordinates, the domain labels at three
# levels, the 64-dimensional Novae latent embedding, and a validity mask marking cells Novae
# could not place. Compressed joblib.
#
# The distinction matters. Handing over a PNG makes a collaborator re-derive positions from
# pixels; handing over the coordinates and labels lets them plot it their own way or run their
# own analysis. The neighborhood_valid mask is included so unplaceable cells are visibly
# excluded rather than silently absorbed into a domain.
# ========================================================================================

"""
export_niche_data_joblib.py
============================================================
Export the DATA behind the Novae niche plots (NOT a raster of the PNG) as
compressed-joblib numpy arrays, for Marcel.

Per sample -> converted/niche_data/<sample>__niche_data.joblib : a dict of arrays
  coords_um          float32 (N,2)  true-micron x,y (= what the niche map plots)
  niche_level5       str     (N,)   primary Novae domain label (the map colours)
  niche_level4       str     (N,)   coarser domains
  niche_level7       str     (N,)   finer domains
  novae_latent       float32 (N,64) Novae cell embedding
  neighborhood_valid bool    (N,)   False = Novae could not place the cell ('unassigned')
  cell_id            str     (N,)   Xenium cell id
  sample/status/run_id/seg_version  str scalars; n_cells int

Combined -> converted/niche_data_all_samples.joblib : same keys but per-cell arrays
  across all 20 samples (sample/status/run_id/seg_version are (N,) arrays).

Also copies the composition count matrices (CSV) into converted/composition/.

Run with PYTHONNOUSERSITE=1 (a broken ~/.local anndata otherwise shadows pertpy_env).
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd
import joblib
import anndata as ad

SRC = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_niches/per_sample"
OUT = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_niches/converted"
SUM = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_niches/summary"

ND = os.path.join(OUT, "niche_data")
COMP = os.path.join(OUT, "composition")
os.makedirs(ND, exist_ok=True)
os.makedirs(COMP, exist_ok=True)


def as_bool(series):
    arr = series.to_numpy()
    if arr.dtype == bool:
        return arr
    return pd.Series(arr).astype(str).str.lower().isin(["true", "1"]).to_numpy()


files = sorted(glob.glob(os.path.join(SRC, "*.h5ad")))
print(f"[export] {len(files)} per-sample h5ads", flush=True)

ACC = {k: [] for k in ["coords_um", "niche_level4", "niche_level5", "niche_level7",
                        "novae_latent", "sample", "status", "run_id", "seg_version",
                        "neighborhood_valid", "cell_id"]}

for f in files:
    a = ad.read_h5ad(f, backed="r")          # backed: obs/obsm in memory, X stays on disk
    n = a.n_obs
    sample = str(a.obs["sample"].iloc[0])
    status = str(a.obs["status"].iloc[0])
    run_id = str(a.obs["run_id"].iloc[0])
    segv = str(a.obs["seg_version"].iloc[0])

    coords = np.asarray(a.obsm["spatial_orig"], dtype="float32")
    l4 = a.obs["novae_domains_4"].astype(str).to_numpy()
    l5 = a.obs["novae_domains_5"].astype(str).to_numpy()
    l7 = a.obs["novae_domains_7"].astype(str).to_numpy()
    latent = np.asarray(a.obsm["novae_latent"], dtype="float32")
    nvalid = as_bool(a.obs["neighborhood_valid"])
    cid = a.obs["xenium_cell_id"].astype(str).to_numpy()

    d = {
        "sample": sample, "status": status, "run_id": run_id, "seg_version": segv,
        "n_cells": int(n),
        "coords_um": coords,
        "niche_level5": l5, "niche_level4": l4, "niche_level7": l7,
        "novae_latent": latent,
        "neighborhood_valid": nvalid,
        "cell_id": cid,
    }
    joblib.dump(d, os.path.join(ND, f"{sample.replace('/', '_')}__niche_data.joblib"), compress=3)

    ACC["coords_um"].append(coords)
    ACC["niche_level4"].append(l4); ACC["niche_level5"].append(l5); ACC["niche_level7"].append(l7)
    ACC["novae_latent"].append(latent)
    ACC["neighborhood_valid"].append(nvalid)
    ACC["cell_id"].append(cid)
    ACC["sample"].append(np.full(n, sample))
    ACC["status"].append(np.full(n, status))
    ACC["run_id"].append(np.full(n, run_id))
    ACC["seg_version"].append(np.full(n, segv))
    try:
        a.file.close()
    except Exception:
        pass
    print(f"  {sample}: n={n}", flush=True)

combined = {k: np.concatenate(v) for k, v in ACC.items()}
joblib.dump(combined, os.path.join(OUT, "niche_data_all_samples.joblib"), compress=3)
print(f"[export] combined: {combined['coords_um'].shape[0]} cells", flush=True)

for c in ["domain_by_sample_counts.csv", "domain_by_status_counts.csv"]:
    s = os.path.join(SUM, c)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(COMP, c))

with open(os.path.join(OUT, "README.txt"), "w") as fh:
    fh.write(
        "Novae niche RESULTS as data arrays (not plot rasters).\n\n"
        "niche_data/<sample>__niche_data.joblib  -> dict of numpy arrays for one sample:\n"
        "    coords_um          float32 (N,2)   true-micron x,y (what the niche map plots)\n"
        "    niche_level5       str     (N,)    primary Novae domain (the map colours)\n"
        "    niche_level4/7     str     (N,)    coarser / finer domains\n"
        "    novae_latent       float32 (N,64)  Novae cell embedding\n"
        "    neighborhood_valid bool    (N,)    False = 'unassigned'\n"
        "    cell_id            str     (N,)    Xenium cell id\n"
        "    sample/status/run_id/seg_version (str scalars), n_cells (int)\n\n"
        "niche_data_all_samples.joblib  -> same keys, per-cell arrays across all 20 samples\n"
        "    (sample/status/run_id/seg_version are (N,) arrays here).\n\n"
        "composition/*.csv  -> domain x sample and domain x status cell-count matrices.\n\n"
        "Load:  import joblib; d = joblib.load('niche_data_all_samples.joblib'); d['coords_um'], d['niche_level5']\n"
    )

print(f"[export] DONE -> {OUT}", flush=True)
