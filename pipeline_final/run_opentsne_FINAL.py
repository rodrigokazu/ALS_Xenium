#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_opentsne_FINAL.py
#
# FFT-accelerated tSNE over the whole cohort from the precomputed PCA, run under the
# opentsne_gpu environment.
#
# Reads X_pca.npy from the typing pipeline and writes X_tsne.npy in the same row order, which
# keeps it aligned with the typed h5ad and obs_celltype.csv. Row alignment across those files
# is an unenforced contract held together by every script writing to the same output
# directory, so do not redirect this one.
#
# Deterministic PCA initialisation with random_state=0, so the embedding reproduces.
#
# It imports final_config. That module pulls only numpy, pandas and the standard library at
# import time, with scanpy deferred. That is what lets it import cleanly inside opentsne_gpu,
# where scanpy is not installed.
# ========================================================================================

"""run_opentsne_FINAL.py -- FFT-accelerated t-SNE of the whole _FINAL cohort from precomputed
PCA. Runs in the opentsne_gpu env (openTSNE). Reads X_pca.npy written by
celltype_pipeline_FINAL.py, writes X_tsne.npy in the SAME row order (row-aligned with the
celltyped h5ad + obs_celltype.csv). Deterministic PCA init + random_state=0 -> reproducible.

Adapted from CellTyping_coarse/code/run_opentsne.py. Only F5 change: OUT = cfg.COARSE_TYPING_DIR
(no hardcoded _gausss path). final_config imports only numpy/pandas/stdlib (scanpy is deferred),
so it imports cleanly in the opentsne_gpu env. Runner exports PYTHONNOUSERSITE=1 + LD_LIBRARY_PATH."""
import os, sys, numpy as np
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_config as cfg
from openTSNE import TSNE

OUT = str(cfg.COARSE_TYPING_DIR)                       # F5: CellTyping_coarse_FINAL
X = np.load(f"{OUT}/X_pca.npy")
print("X_pca", X.shape, flush=True)
tsne = TSNE(n_components=2, perplexity=30, metric="euclidean",
            initialization="pca", n_jobs=-1, random_state=0, verbose=True)
emb = tsne.fit(X)
np.save(f"{OUT}/X_tsne.npy", np.asarray(emb))
print("WROTE X_tsne.npy", np.asarray(emb).shape, flush=True)
