#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/celltype_pipeline_FINAL.py
#
# Coarse cell typing for the whole cohort. Clusters the combined object and assigns broad
# lineage labels from canonical markers, giving every downstream script a cell_type_coarse
# column to work with.
#
# The output directory is shared with every other cell typing script in the suite. That is
# load-bearing: X_pca.npy, X_tsne.npy, the typed h5ad and obs_celltype.csv all have to stay
# row-aligned, and they only do so because nothing writes them anywhere else.
#
# The motor-neuron split is provisional. Until Marcel's is_MN annotation lands on _final this
# splits motor neurons out on MNX1 alone, and uns['mn_split_method'] records 'MNX1-only' so
# nobody mistakes it for the real call. MNX1 is lowly expressed on this panel, so treat those
# labels as a placeholder and not a population.
#
# It asserts that transcript_counts and total_counts are present in obs before using them. A
# concat that quietly dropped either would otherwise fail much later with something
# unreadable. It also logs the fraction of cells under ten transcripts; the _gausss baseline
# was 38.6 percent, so a very different number is a signal that something upstream changed.
#
# The input object is opened read-only. Nothing in the typing pipeline is allowed to modify
# the cohort concat.
# ========================================================================================

"""celltype_pipeline_FINAL.py -- coarse cell-type clustering of the ALS SC-Xenium
MN-CORRECTED _FINAL cohort (Marcel's Ranger_procd_mw_final segmentation, 2026-07-19).

Adapted VERBATIM from the proven _gausss recipe
(CellTyping_coarse/code/celltype_pipeline.py). All clustering/scoring logic and gotchas
are preserved; only the crew's F1-F10 pre-review deltas are applied:

  F5 (paths): import final_config; RAW=cfg.COMBINED_H5AD (READ-ONLY), OUT=cfg.COARSE_TYPING_DIR
      (CellTyping_coarse_FINAL). NO hardcoded _gausss paths. OUT is identical across every
      _FINAL celltyping script so X_pca.npy / X_tsne.npy / the celltyped h5ad / obs_celltype.csv
      stay ROW-ALIGNED.
  F7: assert obs carries transcript_counts + total_counts before use (a concat that silently
      drops them would crash cryptically); LOG the _final under-10-tx junk fraction
      (gausss baseline was 38.6%).
  F3 (SCIENCE): the reference splits MotorNeuron ONLY via is_MN, so with is_MN pending
      (all-False on _final today per mn_annotation_join.py) it would silently yield ZERO
      MotorNeurons. Here: if obs.is_MN present AND sum>0 -> use is_MN (as reference); ELSE mark
      neuron-lineage cells with a positive MN lineage score (sc_MN) AND raw MNX1>0 as
      MotorNeuron, set uns['mn_split_method']='MNX1-only', log LOUDLY, and treat as PROVISIONAL
      (supersede with Marcel's is_MN + validate vs the nuclear MN mask, guardrail G14).

  KEEP (do not regress): normalize_total+log1p is correct for TYPING (raw kept in
      layers['counts']); umap init_pos='random' (spectral init segfaults at ~1M cells);
      leiden igraph seed 0.

Substrate: the UNFILTERED MN-corrected pre-QC concat (raw counts). Treated as READ-ONLY;
we never write back to it. We build a NEW processed object with:
  X = log1p(normalize_total)   (raw counts preserved in layers['counts'])
  obsm: X_pca, X_umap          (X_tsne added later by run_opentsne_FINAL.py from X_pca.npy)
  obs:  leiden_r{05,10}, cell_type_coarse
Coarse cell types assigned per leiden cluster by canonical marker score (argmax), with motor
neurons split out of the neuron block (F3).

Panel has no CHAT (MN relies on MNX1/is_MN) and only ABCC9 for mural -> those stay coarse.
PRE-QC / unfiltered on purpose (per PI): low-quality cells kept; expect a 'LowQuality' cluster.
Nothing here is a QC'd quantitative result. Env: pertpy_env (runner sets PYTHONNOUSERSITE=1)."""
import os, sys, argparse, numpy as np, pandas as pd, scanpy as sc, anndata as ad
# Drop user-site paths so the env's own anndata/scanpy win (belt-and-braces; runner also
# exports PYTHONNOUSERSITE=1). Then make final_config importable from this script's dir.
sys.path[:] = [p for p in sys.path if ".local/lib" not in p]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_config as cfg
sc.settings.verbosity = 1

RAW = str(cfg.COMBINED_H5AD)          # READ-ONLY MN-corrected pre-QC concat (F5)
OUT = str(cfg.COARSE_TYPING_DIR)      # CellTyping_coarse_FINAL (F5)

# canonical coarse marker sets restricted to what is IN the 480 panel (audited 2026-07-18)
MARKERS = {
    "Oligodendrocyte": ["MOBP","MOG","MAG","MBP","CLDN11","MYRF","OPALIN","ERMN","ST18","CNDP1","UGT8","MAL","KLK6"],
    "OPC":             ["PDGFRA","CSPG4","OLIG1","OLIG2","PTPRZ1","SOX10"],
    "Astrocyte":       ["AQP4","GJA1","SLC1A2","SOX9","FGFR3","GLIS3","GPC5","PHGDH","TTYH1"],
    "Microglia":       ["P2RY12","P2RY13","C1QC","TYROBP","TREM2","FCER1G","ITGAM","LAPTM5","CD68","C3AR1","CX3CR1"],
    "Macrophage_perivasc": ["CD163","LYVE1","MSR1","MRC1","F13A1"],
    "Neuron_excit":    ["SLC17A6","SLC17A7","RBFOX3","SNCG","STMN2"],
    "Neuron_inhib":    ["GAD1","GAD2","PVALB","SST","VIP","LAMP5","LHX6"],
    "Endothelial":     ["PECAM1","VWF","CAV1","EPAS1","ABCB1","NRP1","CLDN5"],
    "Mural":           ["ABCC9","PDGFRB","RGS5","NOTCH3","ACTA2"],
    "Fibroblast_VLMC": ["DCN","FBLN1","COL5A2","COL12A1","COL24A1","COL25A1","POSTN","SFRP2"],
    "Lymphoid":        ["PTPRC","CD4","THEMIS","IKZF1","TESPA1","SKAP1"],
}
MN_MARKERS = ["MNX1"]

# --- F3: PROVISIONAL MNX1-only MN-split parameters (used ONLY when is_MN is absent / all-False) ---
NEURON_BLOCK = ["Neuron_excit", "Neuron_inhib"]
# sc.tl.score_genes centres the score on a random reference gene set, so score>0 = MNX1-enriched
# vs background. Require a positive lineage score AND a raw MNX1 transcript (>0) in a neuron-block
# cell. Deliberately conservative; still PROVISIONAL and superseded by Marcel's is_MN.
MN_LINEAGE_SCORE_MIN = 0.0


def main(a):
    os.makedirs(OUT, exist_ok=True)
    print("loading...", RAW, flush=True)
    adata = sc.read_h5ad(RAW)
    print("loaded", adata.shape, "X max", float(adata.X.max()), flush=True)

    # --- F7: required QC columns must be present (a concat can silently drop them) ---
    for col in ("transcript_counts", "total_counts"):
        if col not in adata.obs.columns:
            raise KeyError(
                f"F7: obs is missing '{col}' -- the _final concat did not carry it; coarse typing "
                f"cannot proceed. obs cols (first 40): {list(adata.obs.columns)[:40]}")

    # --- exclude low-count junk from typing (raw object on disk stays untouched) ---
    tc = np.asarray(adata.obs["transcript_counts"], dtype=float)
    good = tc >= 10
    junk_frac = float((~good).mean()) * 100.0
    print(f"cells>=10 transcripts: {int(good.sum()):,} / {adata.n_obs:,} ({good.mean()*100:.1f}%); "
          f"{int((~good).sum()):,} low-count cells excluded from typing", flush=True)
    print(f"[F7] _final under-10-tx JUNK fraction = {junk_frac:.1f}%  (gausss baseline 38.6%)", flush=True)
    adata = adata[good].copy()

    # --- keep raw counts, build normalized working matrix (KEEP: correct for typing) ---
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)          # median library size, standard for panels
    sc.pp.log1p(adata)
    adata.raw = adata                     # lognorm snapshot for scoring/dotplots

    # --- embedding on scaled expression (all 480 genes; no HVG on a targeted panel) ---
    adata_s = adata.copy()
    sc.pp.scale(adata_s, max_value=10)
    sc.tl.pca(adata_s, n_comps=50)
    adata.obsm["X_pca"] = adata_s.obsm["X_pca"]
    np.save(f"{OUT}/X_pca.npy", adata.obsm["X_pca"])          # for openTSNE job (row order preserved)
    adata.obs[["sample"]].to_csv(f"{OUT}/obs_sample_order.csv")
    del adata_s
    print("pca done", flush=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    print("neighbors done", flush=True)
    for res, tag in [(0.5, "r05"), (1.0, "r10")]:
        sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{tag}",
                     flavor="igraph", n_iterations=2, directed=False, random_state=0)  # KEEP igraph seed 0
        print(f"leiden {tag}: {adata.obs[f'leiden_{tag}'].nunique()} clusters", flush=True)
    sc.tl.umap(adata, init_pos="random")   # KEEP: random init (spectral init crashes at ~1M+ cells)
    print("umap done", flush=True)

    # --- score canonical marker sets ---
    for ct, gs in MARKERS.items():
        gg = [g for g in gs if g in adata.raw.var_names]
        sc.tl.score_genes(adata, gg, score_name=f"sc_{ct}", use_raw=True)
    sc.tl.score_genes(adata, [g for g in MN_MARKERS if g in adata.raw.var_names],
                      score_name="sc_MN", use_raw=True)

    # --- assign coarse type per PRIMARY leiden (r10) by mean score argmax ---
    key = "leiden_r10"
    score_cols = [f"sc_{ct}" for ct in MARKERS]
    clmean = adata.obs.groupby(key, observed=True)[score_cols].mean()
    assign = clmean.idxmax(axis=1).str.replace("sc_", "", regex=False)
    # low-quality guard: clusters with low total_counts + weak best score -> Ambiguous
    tc_med = adata.obs.groupby(key, observed=True)["total_counts"].median()
    best = clmean.max(axis=1)
    lowq = (tc_med < 20) & (best < 0.05)
    assign[lowq] = "LowQuality"
    adata.obs["cell_type_coarse"] = adata.obs[key].map(assign).astype("category")

    # --- F3: split MotorNeuron out of the neuron block ------------------------------------
    is_neuron = adata.obs["cell_type_coarse"].isin(NEURON_BLOCK)
    adata.obs["cell_type_coarse"] = adata.obs["cell_type_coarse"].cat.add_categories(["MotorNeuron"])
    ismn = adata.obs.get("is_MN")
    ismn_b = ismn.map(lambda v: str(v).lower() in ("true", "1", "1.0")) if ismn is not None else None
    use_ismn = (ismn_b is not None) and (int(ismn_b.sum()) > 0)
    if use_ismn:
        mn_mask = (ismn_b & is_neuron).to_numpy()
        method = "is_MN"
        print(f"[F3] MN split via is_MN: {int(ismn_b.sum()):,} is_MN cells, "
              f"{int(mn_mask.sum()):,} in the neuron block -> MotorNeuron", flush=True)
    else:
        # PROVISIONAL MNX1-only degrade (is_MN absent or all-False on _final today) -------
        if "MNX1" in adata.var_names:
            col = adata.layers["counts"][:, adata.var_names.get_loc("MNX1")]
            mnx1_raw = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
        else:
            mnx1_raw = np.zeros(adata.n_obs)
        mnx1_pos = mnx1_raw > 0
        score_pos = np.asarray(adata.obs["sc_MN"], dtype=float) > MN_LINEAGE_SCORE_MIN
        mn_mask = is_neuron.to_numpy() & mnx1_pos & score_pos
        method = "MNX1-only"
        print(f"[F3] *** is_MN PENDING (absent or all-False) -> PROVISIONAL MNX1-only MN split. "
              f"neuron-block={int(is_neuron.sum()):,}, MNX1>0={int(mnx1_pos.sum()):,}, "
              f"sc_MN>{MN_LINEAGE_SCORE_MIN}={int(score_pos.sum()):,} -> "
              f"MotorNeuron={int(mn_mask.sum()):,}. PROVISIONAL: supersede with Marcel's is_MN "
              f"and validate vs the nuclear MN mask (guardrail G14). ***", flush=True)
    adata.obs.loc[mn_mask, "cell_type_coarse"] = "MotorNeuron"
    adata.obs["cell_type_coarse"] = adata.obs["cell_type_coarse"].cat.remove_unused_categories()
    adata.uns["mn_split_method"] = method
    print("coarse counts:\n", adata.obs["cell_type_coarse"].value_counts(), flush=True)

    # cluster->type table
    clmean.assign(assigned=assign, median_counts=tc_med, best_score=best).to_csv(
        f"{OUT}/leiden_r10_celltype_assignment.csv")
    # --- ranked marker genes per coarse type (for dotplots) ---
    sc.tl.rank_genes_groups(adata, "cell_type_coarse", method="wilcoxon", use_raw=True, n_genes=25)
    rg = sc.get.rank_genes_groups_df(adata, None)
    rg.to_csv(f"{OUT}/ranked_genes_per_celltype.csv", index=False)
    # --- save processed object (NEW file; raw object untouched) ---
    adata.write_h5ad(f"{OUT}/ALS_SCXenium_coarse_celltyped.h5ad")
    # lightweight obs export for downstream joins (heroes). is_MN only if present in obs.
    export_cols = ["sample", "cell_type_coarse", key] + (["is_MN"] if "is_MN" in adata.obs.columns else [])
    adata.obs[export_cols].to_csv(f"{OUT}/obs_celltype.csv")
    print("WROTE", f"{OUT}/ALS_SCXenium_coarse_celltyped.h5ad", "(mn_split_method=%s)" % method, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); a = ap.parse_args(); main(a)
