"""Label-free segmentation metrics, computed from a transcript -> cell assignment (labels: int per transcript,
0 = no cell). Used by data_prep/12 (evaluation regions), data_prep/15 (cohort crops) and scg/benchmark_fullsample.py.

  mecr             mutually exclusive co-expression rate over cross-cell-type marker pairs, cells >= MIN_TX
  mecr_downsampled the same with every cell cut to MECR_N random transcripts (coverage-controlled)
  shell_coherence  cosine(off-nucleus profile, own on-nucleus profile) - cosine(.., nearest other cell's)
  score            coverage, cell count, median transcripts / cell, MECR, shell coherence
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MIN_TX = 5
MIN_SHELL = 5
MARKERS = {
    "neuron": ["STMN2", "RBFOX3", "SLC17A6", "GAD1", "GAD2"],
    "oligodendrocyte": ["MBP", "MOBP", "MOG", "MAG", "OPALIN", "ST18"],
    "astrocyte": ["AQP4", "SLC1A2", "GJA1"],
    "microglia": ["C1QC", "P2RY12", "PTPRC"],
}


def mecr(labels: np.ndarray, genes: np.ndarray) -> float:
    ok = labels > 0
    df = pd.DataFrame({"cell": labels[ok], "gene": genes[ok]})
    per_cell = df.groupby("cell").size()
    keep = per_cell.index[per_cell >= MIN_TX]
    present = df[df.cell.isin(keep)].drop_duplicates().assign(v=1).pivot_table(
        index="cell", columns="gene", values="v", fill_value=0)
    rates = []
    for (ta, ga), (tb, gb) in itertools.combinations([(t, g) for t, gs in MARKERS.items() for g in gs], 2):
        if ta == tb or ga not in present or gb not in present:
            continue
        a, b = present[ga].to_numpy() > 0, present[gb].to_numpy() > 0
        either = (a | b).sum()
        if either:
            rates.append((a & b).sum() / either)
    return float(np.mean(rates)) if rates else float("nan")


def shell_coherence(labels: np.ndarray, genes: np.ndarray, on_nucleus: np.ndarray,
                    xy: np.ndarray) -> tuple[float, float, int]:
    names, g = np.unique(genes, return_inverse=True)
    cells = np.unique(labels[labels > 0])
    core = np.zeros((len(cells), len(names)))
    shell = np.zeros((len(cells), len(names)))
    ok = labels > 0
    rows = np.searchsorted(cells, labels[ok])
    np.add.at(core, (rows[on_nucleus[ok]], g[ok][on_nucleus[ok]]), 1)
    np.add.at(shell, (rows[~on_nucleus[ok]], g[ok][~on_nucleus[ok]]), 1)
    centroid = pd.DataFrame(xy[ok], columns=["x", "y"]).groupby(rows).mean().reindex(range(len(cells))).to_numpy()
    usable = (core.sum(1) >= MIN_SHELL) & (shell.sum(1) >= MIN_SHELL)
    cand = np.flatnonzero(core.sum(1) >= MIN_SHELL)
    if usable.sum() == 0 or len(cand) < 2:
        return float("nan"), float("nan"), 0
    tree = cKDTree(centroid[cand])

    def cos(a, b):
        a, b = np.sqrt(a), np.sqrt(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    diffs = []
    for i in np.flatnonzero(usable):
        _, nn = tree.query(centroid[i], k=2)
        j = cand[nn[1]] if cand[nn[0]] == i else cand[nn[0]]
        diffs.append(cos(shell[i], core[i]) - cos(shell[i], core[j]))
    diffs = np.array(diffs)
    return float(np.median(diffs)), float((diffs > 0).mean()), len(diffs)


MECR_N = 20      # transcripts per cell for the coverage-controlled MECR


def mecr_downsampled(labels: np.ndarray, genes: np.ndarray, n: int = MECR_N, seed: int = 0) -> float:
    """MECR with every cell reduced to exactly n transcripts (random, fixed seed; cells with fewer are left out).
    Plain MECR rewards small, sparse cells: fewer transcripts per cell means fewer chances to hold two cell types'
    markers, so a method that leaves most transcripts unassigned scores low mixing without segmenting better.
    Equal transcripts per cell compares mixing on equal terms."""
    ok = np.flatnonzero(labels > 0)
    if ok.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    lab = labels[ok]
    order = np.lexsort((rng.random(ok.size), lab))           # random order within each cell
    lab_s = lab[order]
    start = np.r_[0, np.flatnonzero(np.diff(lab_s)) + 1]
    size = np.diff(np.r_[start, lab_s.size])
    rank = np.arange(lab_s.size) - np.repeat(start, size)
    keep_cell = np.repeat(size >= n, size)
    sel = ok[order][(rank < n) & keep_cell]
    ds = np.zeros_like(labels)
    ds[sel] = labels[sel]
    return mecr(ds, genes)


def score(name, labels, genes, on_nuc, xy) -> dict:
    ok = labels > 0
    per_cell = pd.Series(labels[ok]).value_counts()
    med, frac, n = shell_coherence(labels, genes, on_nuc, xy)
    return {"version": name, "pct_assigned": round(100 * ok.mean(), 1), "n_cells": int(per_cell.size),
            "median_tx_per_cell": float(per_cell.median()), "mecr": round(mecr(labels, genes), 4),
            f"mecr_n{MECR_N}": round(mecr_downsampled(labels, genes), 4),
            f"n_cells_ge{MECR_N}_tx": int((per_cell >= MECR_N).sum()),
            "shell_coherence_median": round(med, 3), "shell_coherence_pct_positive": round(100 * frac, 1),
            "n_cells_shell_scored": n}
