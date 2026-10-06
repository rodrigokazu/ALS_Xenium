#!/usr/bin/env python3
"""Zebrin DE with covariates: per gene, OLS of log1p(per-10k) on zebrin score + granule-gene fraction + glia-gene fraction +
log total transcripts, across the 945 Purkinje cells; t-test on the zebrin coefficient, BH FDR. Genes in >= 10% of Purkinje
cells; ALDOC itself and the granule / glia contamination genes left out.  python oq_zebrin_de.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse, stats
SSD = Path("/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB"); O = SSD / "segmentation_full"; Q = O / "open_questions"
cells = pd.read_parquet(O / "cells.parquet"); X = sparse.load_npz(O / "cell_by_gene.npz").tocsr(); genes = pd.read_csv(O / "genes.csv").gene.astype(str).to_numpy()
pz = pd.read_parquet(Q / "purkinje_zebrin.parquet"); idx = pd.Index(cells.cell).get_indexer(pz.cell); Xp = X[idx].astype(np.float64); n = np.asarray(Xp.sum(1)).ravel()
# contamination / purity covariates: Purkinje-gene fraction and interneuron-gene fraction (10x cluster 14 genes >= 10x cluster 33)
de = pd.read_csv(SSD / "Cerebellum_sample/analysis/diffexp/gene_expression_graphclust/differential_expression.csv")
mli = set(de["Feature Name"][(de["Cluster 14 Mean Counts"] >= 0.5) & ((de["Cluster 14 Mean Counts"] + 0.01) / (de["Cluster 33 Mean Counts"] + 0.01) >= 10)])
gsj = json.load(open(SSD / "compare/purkinje/gene_sets.json")); fr = lambda names: np.asarray(Xp[:, np.isin(genes, list(names))].sum(1)).ravel() / n
pz["f_pc"], pz["f_mli"] = fr(set(gsj["purkinje_A"]) | set(gsj["purkinje_B"])), fr(mli)
C = np.c_[np.ones(len(pz)), pz.f_pc, pz.f_mli, pz.f_gran, pz.f_glia, np.log(n)]
pz["zebrin"] = pz.aldoc - C @ np.linalg.lstsq(C, pz.aldoc, rcond=None)[0]                # ALDOC beyond purity and contamination
gs = json.load(open(SSD / "compare/purkinje/gene_sets.json")); excl = set(gs["glia"]) | set(gs["granule"]) | mli | {"ALDOC"}
keep = (np.asarray((Xp > 0).mean(0)).ravel() >= 0.10) & ~np.isin(genes, list(excl))
Y = np.log1p(1e4 * Xp[:, keep].toarray() / n[:, None])
D = np.c_[np.ones(len(pz)), (pz.zebrin - pz.zebrin.mean()) / pz.zebrin.std(), pz.f_pc, pz.f_mli, pz.f_gran, pz.f_glia, np.log(n)]
beta, *_ = np.linalg.lstsq(D, Y, rcond=None); res = Y - D @ beta; dof = len(pz) - D.shape[1]
s2 = (res ** 2).sum(0) / dof; se = np.sqrt(s2 * np.linalg.inv(D.T @ D)[1, 1]); t = beta[1] / se; p = 2 * stats.t.sf(np.abs(t), dof)
r = pd.DataFrame({"gene": genes[keep], "beta_per_sd": beta[1], "t": t, "p": p}); r["fdr"] = stats.false_discovery_control(r.p)
r = r.sort_values("p"); r.to_csv(Q / "zebrin_DE_covariates.csv", index=False); pz.to_parquet(Q / "purkinje_zebrin_v2.parquet", index=False)
print("genes tested", len(r), "FDR<0.05:", int((r.fdr < 0.05).sum()), "up", int(((r.fdr < 0.05) & (r.t > 0)).sum()), "down", int(((r.fdr < 0.05) & (r.t < 0)).sum()))
for g in ["SLC1A6", "PLCB3", "PLCB4", "HSPB1", "KCTD12", "TRPC3", "CA8", "PCP2", "CALB1", "GRM1", "EBF2", "NFIA", "MDGA1"]:
    if g in set(r.gene): print(g, r.set_index("gene").loc[g, ["beta_per_sd", "t", "fdr"]].round(4).to_dict())
print(r.head(30).round(4).to_string())
