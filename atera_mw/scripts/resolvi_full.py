#!/usr/bin/env python3
"""ResolVI on the whole section (final pipeline = segmentation_hybridC), standard settings, unsupervised, one section (no batch).
Genes: the same 3,361-gene set as the window benchmark (resolvi_inputs.py). Cells: QC-passing cells of the annotation with
>= 5 counts on that gene set. Outputs (segmentation_hybridC/resolvi/): latent.parquet (resolVI latent, for UMAP / t-SNE /
clustering), corrected_counts.npy (float16, cells x genes, posterior median of the corrected expression), cells.csv, genes.csv.
Run in the `resolvi` env:  python resolvi_full.py"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scvi, torch
from scipy import sparse
from scvi.external import RESOLVI
SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); H = SSD / "segmentation_hybridC"; OUT = H / "resolvi"; OUT.mkdir(exist_ok=True)
scvi.settings.seed = 0; torch.set_num_threads(10); t0 = time.time()
G = list(ad.read_h5ad(SSD / "compare/resolvi/tenx.h5ad", backed="r").var_names)            # the benchmark gene set
cells = pd.read_parquet(H / "cells.parquet"); genes = pd.read_csv(H / "genes.csv").gene.astype(str)
ann = pd.read_parquet(H / "annotation/cells_annotated.parquet"); keep = cells.cell.isin(ann.index.astype(np.int64)).to_numpy()
X = sparse.load_npz(H / "cell_by_gene.npz").tocsr()[keep][:, pd.Index(genes).get_indexer(G)].astype(np.float32)
c = cells[keep].reset_index(drop=True); ok = np.asarray(X.sum(1)).ravel() >= 5; X, c = X[ok], c[ok].reset_index(drop=True)
a = ad.AnnData(X=X, obs=pd.DataFrame({"cell": c.cell.to_numpy(), "section": "AT1183"}, index=c.cell.astype(str).to_numpy()), var=pd.DataFrame(index=G))
a.layers["counts"] = a.X.copy(); a.obsm["X_spatial"] = c[["x_centroid_um", "y_centroid_um"]].to_numpy()
print(a.shape, f"{time.time() - t0:.0f}s", flush=True)
RESOLVI.setup_anndata(a, layer="counts", batch_key="section")
m = RESOLVI(a); m.train(max_epochs=50, accelerator="cpu", batch_size=1024); m.save(OUT / "model", overwrite=True)
print("trained", f"{time.time() - t0:.0f}s", flush=True)
pd.DataFrame(m.get_latent_representation(), index=a.obs_names).add_prefix("z").to_parquet(OUT / "latent.parquet")
s = m.sample_posterior(model=m.module.model_corrected, return_sites=["px_rate"], summary_fun={"q50": np.median}, num_samples=3, summary_frequency=100, accelerator="cpu")
corr = np.asarray(pd.DataFrame(s).T.loc["q50", "px_rate"]).astype(np.float16)
np.save(OUT / "corrected_counts.npy", corr); a.obs.to_csv(OUT / "cells.csv"); pd.Series(G).to_csv(OUT / "genes.csv", index=False, header=["gene"])
print("done", corr.shape, f"{time.time() - t0:.0f}s", flush=True)
