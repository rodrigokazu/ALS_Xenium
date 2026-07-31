# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/novae_downstream.py
#
# The biology behind the joint niche map: what each domain actually is, not where it is.
# Reproduces the downstream figures from the Novae paper (Blampey et al.) on our cohort.
#
# Seven blocks. Batch-corrected UMAP with an uncorrected comparison, domain proportions by
# sample and by status, differential expression between domains treated as spatially variable
# genes, per-slide SVG maps, PAGA, a per-domain pathway score heatmap over 19 curated gene
# sets, and a domain-by-sample clustermap.
#
# Three decisions in here are the ones to preserve. Differential expression runs on a
# normalised and logged derived copy while the canonical object keeps raw counts, so nothing
# downstream inherits a transformed matrix by accident. The cryptic exon probes CE_STMN2 and
# CE_UNC13A are excluded from domain markers because they read TDP-43 pathology rather than
# domain identity, though they stay in the object and in the TDP43 pathway sets so the scoring
# block still uses them. And the ALS gene set is named ALS_familial_and_GWAS_genes with its
# provenance recorded, after the earlier vaguer name invited the wrong reading.
#
# Scanpy version trap that cost a silent failure: rank_genes_groups_dotplot(return_fig=True)
# hands back a DotPlot whose .fig is None until it is materialised, so calling savefig on
# fig.fig writes nothing at all without erroring. Call DotPlot.savefig directly.
#
# Retained because no _FINAL successor exists. Point it at the _FINAL niche object when you
# port it.
# ========================================================================================

"""
novae_downstream.py
============================================================
Downstream analyses of the joint Novae niche/domain map on the ALS spinal-cord
Xenium cohort, reproducing the Novae paper's downstream figures
(Blampey et al., Methods 4.9 batch-effect / 4.10 SVG; Figs 2, 4, 5) on OUR data.

INPUT (read-only): the object produced by novae_persample_niches.py —
  /oak/.../Novae_persample_niches/novae_all_samples_domains.h5ad
  1,685,385 cells x 480-gene hSpinal Xenium panel, 20 samples (verified 2026-06-18).
  Primary domain key = 'novae_domains_5' (6 labels incl. 'unassigned').

The object already carries (verified on SCG, do NOT recompute):
  obs   : novae_domains_4/5/7 (categorical; 'unassigned' already folded in for the
          1413 neighbourhood-invalid cells), novae_sid, run_id (slide), sample (donor),
          status (control/c9/sporadic), seg_version, neighborhood_valid (bool).
  obsm  : novae_latent (N,64), spatial (TRUE microns), spatial_orig (true microns),
          spatial_grid_offset (the grid-offset coords kept for reference).
  layers: counts  (== X == RAW Gene Expression counts).
  obsp  : the Novae spatial graph (spatial_connectivities + distances).

ANALYSES (all best-effort; one failing block never kills the run):
  A. Batch-corrected UMAP (Fig 4c / 2b): novae.batch_effect_correction([adata], 'novae_domains_5')
     (in place; new obsm key detected by diffing obsm keys). Subsample ~250k (seeded) ->
     neighbors(use_rep=<corrected>) + umap, colour by domain/run_id/status/seg_version.
     PLUS a second UMAP on the UNcorrected novae_latent coloured by run_id (before/after).
  B. Domain proportions (Fig 5c): novae.plot.domains_proportions(slide_name_key='sample')
     + a custom pd.crosstab stacked bar of domain proportions by status.
  C. DEG between domains == SVG per 4.10 (Fig 5b): on a NORMALIZED copy
     (X<-counts, normalize_total, log1p; mito genes excluded) -> rank_genes_groups(wilcoxon)
     -> top-10 dotplot + full deg_by_domain.csv + per-domain deg_top10_by_domain.csv
     (+ seaborn top-10 marker heatmap fallback if the dotplot risks truncation).
  D. SVG spatial maps (Fig 5g): novae.plot.spatially_variable_genes, ONE slide/call,
     looped over 3 representative slides (largest non-reseg sample per status).
  E. PAGA / slide architecture (Fig 5d): novae.plot.paga, ONE slide/call, same 3 slides.
  F. Pathway scores per domain (Fig 5f): sc.tl.score_genes per pathway on the normalized copy
     -> mean per domain -> z-scored clustermap + CSV (19 curated panel-aware gene sets).
  G. Domain x slide clustermap (Fig 2a): seaborn.clustermap of domain proportions per sample.

OUTPUT (created): /oak/.../Novae_persample_niches/downstream/
    plots/   tables/   summary/run_manifest.json

Env: novae_env (Python 3.11 venv); module load python/3.11.1 gcc/13.3.0; NO GPU needed
(this is post-hoc analysis of an already-trained model). See run_novae_downstream.sh.

VERIFIED on SCG 2026-06-18 (novae 1.0.3):
  * novae.batch_effect_correction(adatas: list, obs_key: str) -> None  (in place).
  * novae.plot.domains_proportions(adata|list, obs_key=None, slide_name_key=None, figsize, show).
  * novae.plot.spatially_variable_genes(adata, obs_key=None, top_k=5, ..., show=True).
  * novae.plot.paga(adata, obs_key=None, show=True, **paga_plot_kwargs).
  * novae.plot.pathway_scores(adata, pathways: dict, obs_key=None, ..., return_df=False,
                              min_pathway_size=4, show=True) -> DataFrame|None.
  * Representative slides (largest sample per status, reseg-2026 excluded from control pool):
      control=SD01915_BG (181420), c9=SD01320_BG (115919), sporadic=SD03522_BG (114299).
      reseg-2026 samples (run_id contains '20260612'): SD00614_BG, SD02818_BG (both controls).
"""

import os
import json
import traceback
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import novae

# ============================================================
# CONFIG
# ============================================================
INPUT_H5AD = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_niches/novae_all_samples_domains.h5ad"
OUTPUT_DIR = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_niches/downstream"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summary")

DOMAIN_KEY = "novae_domains_5"     # PRIMARY domain key (6 labels incl. 'unassigned')
SLIDE_KEY = "run_id"               # one Xenium run = one slide (physical batch unit)
SAMPLE_KEY = "sample"              # donor / sample label
STATUS_KEY = "status"             # control / c9 / sporadic
SEG_KEY = "seg_version"           # segmentation-version annotation (confound aid)

LATENT_RAW = "novae_latent"        # uncorrected Novae embedding (64-d)
UNASSIGNED = "unassigned"          # Novae's NaN-domain label (already in the categorical)

# Reseg-2026 slides are identified by run_id containing this tag (NOT by sample name).
# Verified: only SD00614_BG and SD02818_BG (both controls) -> exclude from the control
# representative-slide pool so the control SVG/PAGA panels are NOT on a reseg outlier.
RESEG_2026_TAG = "20260612"

N_SUBSAMPLE = 250_000              # cells for the UMAP (speed); seeded
SEED = 0
N_NEIGHBORS = 15
TOP_K_SVG = 10                     # top SVG per domain for plot D (was 5)
DEG_TOP_N_DOTPLOT = 10            # top genes/domain in the rank_genes_groups dotplot (was 5)
DPI = 200

# Curated pathway gene sets (Fig 5f), refined by jarvis-spatial + 7 new panel-buildable ALS/SC sets.
# Panel reconciled 2026-06-18: object var_names == the 480-gene nominal panel EXACTLY (CFAP410 IS
# present; the earlier "478/CFAP410-absent" was a truncated relayed list, not a real object gap).
# Each set has >=4 panel genes. NOTE: scores are RELATIVE WITHIN THIS PANEL (background pool is
# the 478-gene panel, not the transcriptome) — not absolute pathway activity. 'Reactive_astrocyte_pan'
# is a panel-restricted proxy (no GFAP/VIM/S100B on panel), NOT a canonical Liddelow A1 signature.
GENE_SETS = {
    "Motor_neuron": ["STMN2", "SLC17A6", "MNX1", "SNCG", "RBFOX3", "TAC1", "NOS1"],
    "TDP43_cryptic_exon": ["CE_STMN2", "CE_UNC13A", "STMN2", "UNC13A"],
    # ALS_familial_and_GWAS_genes (renamed from 'ALS_risk_genes' for precision; jarvis-spatial):
    #   19 in-panel ALS-associated genes = familial/Mendelian ALS genes (C9orf72, SOD1, TARDBP,
    #   FUS, TBK1, OPTN, VCP, KIF5A, NEK1, PFN1, TUBA4A, ANXA11, CFAP410/C21orf2, GLT8D1) +
    #   GWAS/risk-modifier loci (UNC13A, ATXN2, NIPA1, SCFD1) + C9orf72-complex member SMCR8.
    #   Curated from the ALS gene-discovery literature (not one published list), intersected with
    #   the 480-gene panel. UNC13A here = the GWAS gene (distinct from the CE_UNC13A pathology probe).
    "ALS_familial_and_GWAS_genes": ["C9orf72", "SOD1", "TARDBP", "FUS", "TBK1", "OPTN", "VCP", "UNC13A",
                       "KIF5A", "NEK1", "ATXN2", "PFN1", "TUBA4A", "ANXA11", "NIPA1",
                       "SMCR8", "SCFD1", "GLT8D1", "CFAP410"],
    "Astrocyte": ["GJA1", "AQP4", "SLC1A2", "SOX9", "FGFR3", "GLIS3", "WDR49"],
    "Microglia_homeostatic": ["P2RY12", "P2RY13", "C1QC", "RGS10"],
    "Microglia_activated_DAM": ["TREM2", "TYROBP", "APOE", "CD68", "ITGAX", "GPNMB",
                                "MSR1", "CD163", "APOC1"],
    "Oligodendrocyte_myelin": ["MBP", "MOBP", "MOG", "MAG", "CLDN11", "MYRF", "OPALIN",
                               "ERMN", "CNDP1", "MAL", "UGT8", "ST18"],
    "OPC": ["PDGFRA", "CSPG4", "OLIG1", "OLIG2"],
    "Endothelial_vascular": ["PECAM1", "VWF", "EPAS1", "CAV1", "ABCB1"],
    "Inhibitory_neuron": ["GAD1", "GAD2", "PVALB", "SST", "VIP", "LHX6", "LAMP5"],
    "Excitatory_neuron": ["SLC17A7", "SLC17A6", "RORB", "CUX2", "FOXP2"],
    "Complement_neuroinflammation": ["C3", "C1QC", "C3AR1", "FCER1G", "CXCL14", "CXCL16",
                                     "CCL4", "TGFB1", "TGFB2", "SERPINA3", "CHI3L1", "TLR6"],
    # --- new sets (jarvis-spatial; all genes verified in-panel) ---
    "Reactive_astrocyte_pan": ["GJA1", "AQP4", "SLC1A2", "SOX9", "FGFR3", "SERPINA3",
                               "CHI3L1", "C3", "STAT3", "TF"],
    "Stress_proteostasis": ["BAG3", "SERPINH1", "HNRNPA1", "TARDBP", "FUS", "VCP", "ATXN2",
                            "DDI2", "GADD45G", "TP53", "STAT3"],
    "Synaptic_signaling": ["SNCG", "STMN2", "SYNPR", "SV2C", "HOMER2", "CADPS2", "CABP1",
                           "CCK", "NXPH1", "NXPH2", "CBLN2", "CLSTN2", "UNC13A", "UNC13C",
                           "STXBP2", "STX3", "NELL2", "FGF13"],
    "Cytotoxic_microglia_complement": ["C1QC", "C3", "C3AR1", "FCER1G", "TYROBP", "CD68",
                                       "ITGAM", "CTSB", "CTSD", "CTSC", "CTSH", "TREM2",
                                       "APOE", "SPI1", "PTPRC"],
    "Vascular_BBB": ["PECAM1", "VWF", "EPAS1", "CAV1", "ABCB1", "ABCC9", "CALCRL", "NRP1",
                     "ANGPT1", "COL5A2", "SLCO2B1"],
    "Oligo_lineage_TF": ["OLIG1", "OLIG2", "SOX10", "MYRF", "ST18", "PDGFRA", "CSPG4"],
    "TDP43_targets_extended": ["CE_STMN2", "CE_UNC13A", "STMN2", "UNC13A", "UNC13C",
                               "KCNQ2", "ATP2C2", "FGF13"],
}

# score_genes params for the panel-restricted background (jarvis-spatial WARNING: default
# ctrl_size=50/n_bins=25 exhausts a 478-gene pool). Lowered + seeded for reproducibility.
SCORE_CTRL_SIZE = 25
SCORE_N_BINS = 10

# ============================================================
# SETUP
# ============================================================
for d in (OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, SUMMARY_DIR):
    os.makedirs(d, exist_ok=True)


def log(m):
    print(m, flush=True)


# Manifest accumulates per-block status; written at the end no matter what.
MANIFEST = {
    "input_h5ad": INPUT_H5AD,
    "output_dir": OUTPUT_DIR,
    "domain_key": DOMAIN_KEY,
    "slide_key": SLIDE_KEY,
    "blocks": {},          # block_name -> {"status": ok|failed|skipped, "detail": ...}
    "warnings": [],
}


def record(block, status, **detail):
    MANIFEST["blocks"][block] = {"status": status, **detail}
    log(f"[manifest] {block}: {status}" + (f"  {detail}" if detail else ""))


def save_fig(fig, name):
    """Save and close a specific Figure."""
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def save_current(name):
    """Save whatever is on the current MPL figure (for novae.plot / scanpy fns that
    draw onto the active figure and return None)."""
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    return path


def write_manifest():
    path = os.path.join(SUMMARY_DIR, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(MANIFEST, f, indent=2, default=str)
    log(f"[manifest] wrote {path}")


# ============================================================
# LOAD
# ============================================================
log(f"[load] {INPUT_H5AD}")
adata = sc.read_h5ad(INPUT_H5AD)
log(f"[load] {adata.shape}  obs has domain_key={DOMAIN_KEY in adata.obs}  "
    f"obsm={list(adata.obsm.keys())}  layers={list(adata.layers.keys())}")

# Hard preconditions (fail loudly here — everything downstream depends on these).
assert DOMAIN_KEY in adata.obs, f"obs['{DOMAIN_KEY}'] missing"
assert LATENT_RAW in adata.obsm, f"obsm['{LATENT_RAW}'] missing"
assert "counts" in adata.layers, "layers['counts'] (raw counts) missing"
for k in (SLIDE_KEY, SAMPLE_KEY, STATUS_KEY):
    assert k in adata.obs, f"obs['{k}'] missing"

# seg_version: present from the upstream run; if somehow absent, derive it from run_id.
if SEG_KEY not in adata.obs:
    adata.obs[SEG_KEY] = np.where(
        adata.obs[SLIDE_KEY].astype(str).str.contains(RESEG_2026_TAG),
        "largecells_reseg_2026-06-12", "largecells_reseg_2025-12",
    )
    adata.obs[SEG_KEY] = adata.obs[SEG_KEY].astype("category")
    MANIFEST["warnings"].append(f"{SEG_KEY} was absent -> derived from {SLIDE_KEY}.")

MANIFEST["n_cells"] = int(adata.n_obs)
MANIFEST["n_genes"] = int(adata.n_vars)
MANIFEST["n_samples"] = int(adata.obs[SAMPLE_KEY].nunique())
MANIFEST["n_slides"] = int(adata.obs[SLIDE_KEY].nunique())
MANIFEST["domain_labels"] = sorted(adata.obs[DOMAIN_KEY].astype(str).unique())

# Identify the reseg-2026 samples (confound; exclude from the control representative pool).
reseg_mask = adata.obs[SLIDE_KEY].astype(str).str.contains(RESEG_2026_TAG)
reseg_samples = sorted(adata.obs.loc[reseg_mask, SAMPLE_KEY].astype(str).unique())
MANIFEST["reseg_2026_samples"] = reseg_samples
log(f"[seg] reseg-2026 samples (excluded from control representative): {reseg_samples}")

# Gene-set intersection with the panel (for the manifest + so pathway_scores is honest).
panel = set(map(str, adata.var_names))
gene_set_sizes = {}
for name, genes in GENE_SETS.items():
    present = [g for g in genes if g in panel]
    gene_set_sizes[name] = {"n_present": len(present), "n_total": len(genes),
                            "missing": [g for g in genes if g not in panel]}
MANIFEST["gene_set_sizes"] = gene_set_sizes
_under4 = [k for k, v in gene_set_sizes.items() if v["n_present"] < 4]
if _under4:
    MANIFEST["warnings"].append(
        f"gene sets with <4 panel genes (dropped by pathway_scores min_pathway_size=4): {_under4}")
log(f"[panel] gene-set intersection computed; sets with <4 present: {_under4 or 'none'}")

# --- explicit gene-set membership table (one row per pathway, present vs missing vs panel) ---
try:
    gs_rows = []
    for name, genes in GENE_SETS.items():
        present = [g for g in genes if g in panel]
        missing = [g for g in genes if g not in panel]
        gs_rows.append({"pathway": name, "n_present": len(present), "n_total": len(genes),
                        "genes_present": ";".join(present), "genes_missing": ";".join(missing)})
    pd.DataFrame(gs_rows).to_csv(os.path.join(TABLES_DIR, "gene_sets_membership.csv"), index=False)
    log(f"[panel] wrote gene_sets_membership.csv ({len(gs_rows)} pathways)")
except Exception as e:
    log(f"[panel] gene_sets_membership.csv failed: {type(e).__name__}: {e}")

# Full GENE_SETS dict (verbatim membership) into the manifest for provenance.
MANIFEST["gene_sets"] = {k: list(v) for k, v in GENE_SETS.items()}
MANIFEST["als_gene_set_provenance"] = (
    "ALS_familial_and_GWAS_genes: 19 in-panel ALS-associated genes — familial/Mendelian "
    "(C9orf72,SOD1,TARDBP,FUS,TBK1,OPTN,VCP,KIF5A,NEK1,PFN1,TUBA4A,ANXA11,CFAP410,GLT8D1) + "
    "GWAS/risk-modifier (UNC13A,ATXN2,NIPA1,SCFD1) + C9orf72-complex member SMCR8; "
    "curated from the ALS gene-discovery literature, intersected with the 480-gene panel."
)

# --- mitochondrial-gene removal (targeted panel; likely 0 MT genes — report honestly) ---
# Match MT- prefix (case-insensitive) plus an 'MT.'-prefix guard. Drop from the EXPRESSION
# analysis objects (DEG C, SVG D, pathway F) so they exclude mito genes when present. The
# domain-defining latent / UMAP / proportions are NOT affected (they don't read var_names).
_uv = adata.var_names.str.upper()
mt = _uv.str.match(r"^MT-") | _uv.str.startswith("MT.")
MITO_GENES = list(adata.var_names[mt])
N_MITO = int(mt.sum())
MANIFEST["n_mito_genes_removed"] = N_MITO
MANIFEST["mito_genes_removed"] = MITO_GENES
log(f"[mito] n_mito_genes={N_MITO}  genes={MITO_GENES or 'none'}")
if N_MITO:
    MANIFEST["warnings"].append(
        f"removed {N_MITO} mitochondrial gene(s) from DEG/SVG/pathway objects: {MITO_GENES}")

# Engineered TDP-43 cryptic-exon PROBES (jarvis-spatial WARNING): these are pathology readouts,
# not cell-type/domain identity markers. EXCLUDE them from per-domain DEG (C) + SVG (D) so they
# don't masquerade as domain markers — but KEEP them in the object and in the pathway gene sets
# (F), where they are the intended TDP-43 pathology axis.
EXCLUDE_FROM_MARKERS = [g for g in ["CE_STMN2", "CE_UNC13A"] if g in panel]
MANIFEST["excluded_from_markers"] = EXCLUDE_FROM_MARKERS
log(f"[markers] excluded from DEG/SVG marker analysis (kept for pathways): {EXCLUDE_FROM_MARKERS}")


def pick_representative_slides():
    """Largest sample (by n_cells) per status; reseg-2026 samples excluded from the
    CONTROL pool only (they are the seg-version outliers). Returns {status: sample}.
    Verified result on this object: control=SD01915_BG, c9=SD01320_BG, sporadic=SD03522_BG."""
    ct = pd.crosstab(adata.obs[SAMPLE_KEY], adata.obs[STATUS_KEY])
    chosen = {}
    for status in ct.columns:
        sizes = ct[status][ct[status] > 0].sort_values(ascending=False)
        if status == "control":
            sizes = sizes[~sizes.index.astype(str).isin(reseg_samples)]
        if len(sizes):
            chosen[str(status)] = str(sizes.index[0])
    return chosen


REP_SLIDES = pick_representative_slides()
MANIFEST["representative_slides"] = REP_SLIDES
log(f"[rep] representative slides per status: {REP_SLIDES}")

# --- 'unassigned' report (jarvis-spatial NOTE e): quantify before dropping it from DEG/PAGA/pathway ---
try:
    una = adata.obs[DOMAIN_KEY].astype(str) == UNASSIGNED
    una_tbl = pd.DataFrame({
        "n_unassigned": adata.obs[una].groupby(adata.obs.loc[una, SAMPLE_KEY], observed=True).size(),
    })
    una_by_status = adata.obs[una].groupby(adata.obs.loc[una, STATUS_KEY], observed=True).size()
    una_tbl.to_csv(os.path.join(TABLES_DIR, "unassigned_counts_by_sample.csv"))
    MANIFEST["n_unassigned_total"] = int(una.sum())
    MANIFEST["n_unassigned_by_status"] = {str(k): int(v) for k, v in una_by_status.items()}
    log(f"[unassigned] total={int(una.sum())} ({100*una.mean():.3f}%); by status={MANIFEST['n_unassigned_by_status']}")
except Exception as e:
    log(f"[unassigned] report failed: {type(e).__name__}: {e}")

# --- shared NORMALIZED copy (raw stays canonical; this is a DERIVED layer ONLY for DEG + score_genes,
#     per jarvis-spatial BLOCKER a). 'unassigned' cells removed (NOTE e). Built once, reused by C & F. ---
ADATA_NORM = None
def build_adata_norm():
    global ADATA_NORM
    if ADATA_NORM is not None:
        return ADATA_NORM
    log("[norm] building normalized+log copy (non-'unassigned' cells) for DEG + pathway scoring")
    keep = adata.obs[DOMAIN_KEY].astype(str) != UNASSIGNED
    an = adata[keep].copy()
    # exclude mito genes from the expression-analysis object (DEG + pathway scoring) when present.
    if N_MITO:
        an = an[:, ~an.var_names.isin(MITO_GENES)].copy()
        log(f"[norm] dropped {N_MITO} mito gene(s) -> {an.n_vars} genes")
    an.X = an.layers["counts"].copy()
    sc.pp.normalize_total(an, target_sum=1e4)
    sc.pp.log1p(an)
    an.obs[DOMAIN_KEY] = an.obs[DOMAIN_KEY].astype("category").cat.remove_unused_categories()
    ADATA_NORM = an
    return an


# ============================================================
# A. BATCH-CORRECTED UMAP  (Fig 4c / 2b) + uncorrected comparison
# ============================================================
def block_A_umap():
    # --- batch correction (in place; detect the new obsm key by diffing) ---
    corrected_key = None
    try:
        before = set(adata.obsm.keys())
        log(f"[A] novae.batch_effect_correction([adata], obs_key='{DOMAIN_KEY}') ...")
        novae.batch_effect_correction([adata], obs_key=DOMAIN_KEY)
        new_keys = sorted(set(adata.obsm.keys()) - before)
        log(f"[A] new obsm keys after correction: {new_keys}")
        if "novae_latent_corrected" in new_keys:
            corrected_key = "novae_latent_corrected"
        elif new_keys:
            corrected_key = new_keys[0]
        else:
            # Some versions correct in place on novae_latent; fall back to it.
            corrected_key = LATENT_RAW
            MANIFEST["warnings"].append(
                "batch_effect_correction added no new obsm key; using novae_latent for the corrected UMAP.")
    except Exception as e:
        record("A_batch_correction", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        corrected_key = None

    # --- seeded subsample for UMAP speed ---
    rng = np.random.default_rng(SEED)
    n_sub = min(N_SUBSAMPLE, adata.n_obs)
    idx = np.sort(rng.choice(adata.n_obs, size=n_sub, replace=False))
    sub = adata[idx].copy()
    sub.obsp.clear()  # drop sliced graph; we recompute neighbours on the chosen rep
    log(f"[A] subsampled {n_sub}/{adata.n_obs} cells (seed={SEED}) for UMAP")

    color_keys = [DOMAIN_KEY, SLIDE_KEY, STATUS_KEY, SEG_KEY]

    # --- corrected-latent UMAP ---
    if corrected_key is not None and corrected_key in sub.obsm:
        try:
            log(f"[A] neighbors+umap on corrected rep '{corrected_key}'")
            sc.pp.neighbors(sub, use_rep=corrected_key, n_neighbors=N_NEIGHBORS, random_state=SEED)
            sc.tl.umap(sub, random_state=SEED)
            for ck in color_keys:
                try:
                    fig = sc.pl.umap(sub, color=ck, show=False, return_fig=True,
                                     title=f"Batch-corrected UMAP ({corrected_key}) — {ck}")
                    save_fig(fig, f"A_umap_corrected__{ck}.png")
                except Exception as e:
                    log(f"[A] umap colour '{ck}' failed: {type(e).__name__}: {e}")
            record("A_umap_corrected", "ok", corrected_key=corrected_key, n_cells=int(n_sub))
        except Exception as e:
            record("A_umap_corrected", "failed", error=f"{type(e).__name__}: {e}")
            log(traceback.format_exc())
    else:
        record("A_umap_corrected", "skipped", reason="no corrected obsm key available")

    # --- uncorrected-latent UMAP (before/after batch-integration comparison) ---
    try:
        log(f"[A] neighbors+umap on UNcorrected '{LATENT_RAW}' (coloured by {SLIDE_KEY})")
        sub_u = sub.copy()
        if "neighbors" in sub_u.uns:
            del sub_u.uns["neighbors"]
        for ok in ("X_umap",):
            sub_u.obsm.pop(ok, None)
        sc.pp.neighbors(sub_u, use_rep=LATENT_RAW, n_neighbors=N_NEIGHBORS, random_state=SEED)
        sc.tl.umap(sub_u, random_state=SEED)
        for ck in (SLIDE_KEY, DOMAIN_KEY):
            try:
                fig = sc.pl.umap(sub_u, color=ck, show=False, return_fig=True,
                                 title=f"Uncorrected UMAP ({LATENT_RAW}) — {ck}")
                save_fig(fig, f"A_umap_uncorrected__{ck}.png")
            except Exception as e:
                log(f"[A] uncorrected umap colour '{ck}' failed: {type(e).__name__}: {e}")
        record("A_umap_uncorrected", "ok", n_cells=int(n_sub))
    except Exception as e:
        record("A_umap_uncorrected", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())

    return corrected_key


CORRECTED_KEY = block_A_umap()


# ============================================================
# B. DOMAIN PROPORTIONS  (Fig 5c)
# ============================================================
def block_B_proportions():
    # Novae's own per-slide proportions plot.
    try:
        log("[B] novae.plot.domains_proportions(...)")
        novae.plot.domains_proportions(adata, obs_key=DOMAIN_KEY, slide_name_key=SAMPLE_KEY, show=False)
        save_current("B_domains_proportions_novae.png")
        record("B_domains_proportions_novae", "ok")
    except Exception as e:
        record("B_domains_proportions_novae", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        plt.close("all")

    # Custom stacked bar of domain proportions by disease status.
    try:
        log("[B] custom stacked-bar of domain proportions by status")
        ct = pd.crosstab(adata.obs[DOMAIN_KEY], adata.obs[STATUS_KEY], dropna=False)
        ct.to_csv(os.path.join(TABLES_DIR, "domain_counts_by_status.csv"))
        prop = ct.div(ct.sum(axis=0), axis=1)  # columns=status, rows=domain
        groups = list(prop.columns)
        x = np.arange(len(groups))
        bottom = np.zeros(len(groups))
        cmap = plt.get_cmap("tab20")
        domains = list(prop.index.astype(str))
        colors = {d: ("0.8" if d == UNASSIGNED else cmap(i % 20)) for i, d in enumerate(domains)}
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(groups) + 2), 6))
        for d in domains:
            vals = prop.loc[d].values
            ax.bar(x, vals, bottom=bottom, color=colors[d], label=d)
            bottom += vals
        ax.set_ylabel("proportion of cells")
        ax.set_title(f"Domain proportions by status ({DOMAIN_KEY})")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=0)
        ax.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5), title="domain")
        ax.text(0.0, -0.14,
                "NOTE: the 2 control samples SD00614_BG / SD02818_BG use a different segmentation "
                "run (seg_version). Verify the seg-version confound before reading by-status "
                "differences as biology (leave-2-out sensitivity).",
                transform=ax.transAxes, fontsize=6, color="firebrick", wrap=True)
        save_fig(fig, "B_domain_proportions_by_status.png")
        record("B_proportions_by_status", "ok")
    except Exception as e:
        record("B_proportions_by_status", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        plt.close("all")


block_B_proportions()


# ============================================================
# C. DEG between domains == SVG per 4.10  (Fig 5b)
# ============================================================
def block_C_deg():
    try:
        an = build_adata_norm()   # normalized+log, 'unassigned' + mito already removed
        # drop CE_ cryptic-exon probes for the MARKER analysis only (local copy; the global
        # ADATA_NORM keeps them so block F's TDP-43 pathway scores still work).
        if EXCLUDE_FROM_MARKERS:
            an = an[:, ~an.var_names.isin(EXCLUDE_FROM_MARKERS)].copy()
            log(f"[C] excluded {EXCLUDE_FROM_MARKERS} from DEG -> {an.n_vars} genes")
        log(f"[C] rank_genes_groups(groupby='{DOMAIN_KEY}', method='wilcoxon') on {an.n_obs} cells")
        sc.tl.rank_genes_groups(an, groupby=DOMAIN_KEY, method="wilcoxon")

        deg_df = sc.get.rank_genes_groups_df(an, group=None)
        deg_path = os.path.join(TABLES_DIR, "deg_by_domain.csv")
        deg_df.to_csv(deg_path, index=False)
        log(f"[C] DEG table -> {deg_path}  ({len(deg_df)} rows)")

        # explicit per-domain top-10 table (top by score within each group).
        top10_path = os.path.join(TABLES_DIR, "deg_top10_by_domain.csv")
        try:
            groups = list(an.obs[DOMAIN_KEY].cat.categories)
            top_rows = []
            for grp in groups:
                gdf = sc.get.rank_genes_groups_df(an, group=grp)
                gdf = gdf.sort_values("scores", ascending=False).head(DEG_TOP_N_DOTPLOT)
                top_rows.append(pd.DataFrame({
                    "domain": grp,
                    "gene": gdf["names"].values,
                    "score": gdf["scores"].values,
                    "logfoldchanges": gdf["logfoldchanges"].values,
                    "pvals_adj": gdf["pvals_adj"].values,
                }))
            top10_df = pd.concat(top_rows, ignore_index=True)
            top10_df.to_csv(top10_path, index=False)
            log(f"[C] per-domain top-{DEG_TOP_N_DOTPLOT} DEG table -> {top10_path} ({len(top10_df)} rows)")
        except Exception as e:
            log(f"[C] top-{DEG_TOP_N_DOTPLOT} table failed: {type(e).__name__}: {e}")
            top10_df = None

        n_domains = len(an.obs[DOMAIN_KEY].cat.categories)
        # dotplot at 10 genes x n_domains can truncate -> pass explicit figsize + return_fig.
        try:
            dp_figsize = (max(14, 0.34 * n_domains * DEG_TOP_N_DOTPLOT + 3), 0.7 * n_domains + 3)
            dp_path = os.path.join(PLOTS_DIR, "C_deg_dotplot_top10.png")
            try:
                dp = sc.pl.rank_genes_groups_dotplot(
                    an, n_genes=DEG_TOP_N_DOTPLOT, show=False, return_fig=True, figsize=dp_figsize)
                # scanpy 1.9.x returns a DotPlot whose .fig is lazy/None until materialized;
                # DotPlot.savefig() builds + writes directly (verified in novae_env).
                if hasattr(dp, "savefig"):
                    dp.savefig(dp_path, dpi=DPI, bbox_inches="tight")
                    plt.close("all")
                elif getattr(dp, "fig", None) is not None:
                    save_fig(dp.fig, "C_deg_dotplot_top10.png")
                else:
                    save_current("C_deg_dotplot_top10.png")
            except TypeError:
                # older scanpy: no return_fig -> draw onto current figure, save it.
                sc.pl.rank_genes_groups_dotplot(an, n_genes=DEG_TOP_N_DOTPLOT, show=False, figsize=dp_figsize)
                save_current("C_deg_dotplot_top10.png")
        except Exception as e:
            log(f"[C] dotplot failed: {type(e).__name__}: {e}")
            plt.close("all")

        # clean seaborn fallback: mean log-norm expression of the union of top-10 markers x domains.
        try:
            if top10_df is not None and len(top10_df):
                marker_union = list(dict.fromkeys(top10_df["gene"].tolist()))  # ordered unique
                marker_union = [g for g in marker_union if g in an.var_names]
                if marker_union:
                    df_expr = sc.get.obs_df(an, keys=[DOMAIN_KEY] + marker_union)
                    mean_expr = df_expr.groupby(DOMAIN_KEY, observed=True)[marker_union].mean()
                    fh = max(10, 0.4 * len(marker_union) + 3)
                    fw = max(8, 0.6 * n_domains + 4)
                    fig, ax = plt.subplots(figsize=(fw, fh))
                    sns.heatmap(mean_expr.T, cmap="rocket", ax=ax,
                                cbar_kws={"label": "mean log-norm expression"})
                    ax.set_title(f"Top-{DEG_TOP_N_DOTPLOT} DEG markers per domain ({DOMAIN_KEY})")
                    ax.set_yticks(np.arange(len(marker_union)) + 0.5)
                    ax.set_yticklabels(marker_union, rotation=0, fontsize=6)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                    save_fig(fig, "C_deg_top10_heatmap.png")
        except Exception as e:
            log(f"[C] top-10 heatmap fallback failed: {type(e).__name__}: {e}")
            plt.close("all")

        record("C_deg", "ok", n_rows=int(len(deg_df)), table=deg_path,
               top10_table=top10_path, n_cells=int(an.n_obs))
    except Exception as e:
        record("C_deg", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        plt.close("all")


block_C_deg()


# ============================================================
# D. SVG spatial maps  (Fig 5g) — one slide per call, 3 representative slides
# ============================================================
def _subset_to_sample(sample_name, drop_mito=False):
    """Subset adata to one donor/sample. Keep obsp (the Novae spatial graph) so the
    per-slide plots that rely on it still work; slicing keeps the matching submatrix.
    drop_mito: when True, drop mito genes AND the CE_ cryptic-exon probes (for SVG block D, a
    gene-expression marker analysis); PAGA (E) leaves them in since it reads the latent/graph."""
    m = (adata.obs[SAMPLE_KEY].astype(str) == str(sample_name)).values
    sub = adata[m].copy()
    if drop_mito:
        drop = set(MITO_GENES) | set(EXCLUDE_FROM_MARKERS)
        if drop:
            sub = sub[:, ~sub.var_names.isin(drop)].copy()
    return sub


def block_D_svg_maps():
    results = {}
    for status, sample_name in REP_SLIDES.items():
        try:
            log(f"[D] spatially_variable_genes — {status} / {sample_name}")
            sub = _subset_to_sample(sample_name, drop_mito=True)  # SVG is gene-expression -> exclude mito
            novae.plot.spatially_variable_genes(sub, obs_key=DOMAIN_KEY, top_k=TOP_K_SVG, show=False)
            safe = str(sample_name).replace("/", "_")
            save_current(f"D_svg_spatial__{status}__{safe}.png")
            results[status] = {"sample": sample_name, "status": "ok"}
        except Exception as e:
            results[status] = {"sample": sample_name, "status": "failed",
                               "error": f"{type(e).__name__}: {e}"}
            log(f"[D] {status}/{sample_name} failed: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            plt.close("all")
    record("D_svg_maps", "ok" if any(r["status"] == "ok" for r in results.values()) else "failed",
           per_slide=results)


block_D_svg_maps()


# ============================================================
# E. PAGA / slide architecture  (Fig 5d) — one slide per call, same 3 slides
# ============================================================
def block_E_paga():
    results = {}
    for status, sample_name in REP_SLIDES.items():
        try:
            log(f"[E] novae.plot.paga — {status} / {sample_name}")
            sub = _subset_to_sample(sample_name)
            # drop 'unassigned' (jarvis-spatial NOTE e: junk hub distorts the trajectory)
            sub = sub[sub.obs[DOMAIN_KEY].astype(str) != UNASSIGNED].copy()
            sub.obs[DOMAIN_KEY] = sub.obs[DOMAIN_KEY].astype("category").cat.remove_unused_categories()
            novae.plot.paga(sub, obs_key=DOMAIN_KEY, show=False)
            safe = str(sample_name).replace("/", "_")
            save_current(f"E_paga__{status}__{safe}.png")
            results[status] = {"sample": sample_name, "status": "ok"}
        except Exception as e:
            results[status] = {"sample": sample_name, "status": "failed",
                               "error": f"{type(e).__name__}: {e}"}
            log(f"[E] {status}/{sample_name} failed: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            plt.close("all")
    record("E_paga", "ok" if any(r["status"] == "ok" for r in results.values()) else "failed",
           per_slide=results)


block_E_paga()


# ============================================================
# F. Pathway scores per domain  (Fig 5f)
# ============================================================
def block_F_pathways():
    # Primary: faithful Methods-4.10 implementation under our control — score_genes on the
    # NORMALIZED+log data (not raw), lowered ctrl_size/n_bins for the 478-gene background,
    # 'unassigned' already excluded; mean score per domain -> seaborn.clustermap.
    try:
        an = build_adata_norm()
        log(f"[F] score_genes for {len(GENE_SETS)} pathways "
            f"(ctrl_size={SCORE_CTRL_SIZE}, n_bins={SCORE_N_BINS}, seed={SEED})")
        score_cols = []
        for name, genes in GENE_SETS.items():
            present = [g for g in genes if g in an.var_names]
            if len(present) < 4:
                log(f"[F] skip pathway '{name}': only {len(present)} panel genes")
                continue
            col = f"PW_{name}"
            sc.tl.score_genes(an, gene_list=present, score_name=col,
                              ctrl_size=SCORE_CTRL_SIZE, n_bins=SCORE_N_BINS, random_state=SEED)
            score_cols.append(col)
        # mean pathway score per domain
        dfp = an.obs.groupby(DOMAIN_KEY, observed=True)[score_cols].mean()
        dfp.columns = [c[3:] for c in dfp.columns]  # strip PW_ prefix
        path = os.path.join(TABLES_DIR, "pathway_scores_by_domain.csv")
        dfp.to_csv(path)
        # record success on the TABLE first — a heatmap failure must not mask a valid CSV
        record("F_pathway_scores", "ok", table=path, shape=list(dfp.shape),
               note=f"score_genes on normalized data (ctrl_size={SCORE_CTRL_SIZE}, n_bins={SCORE_N_BINS}); relative within panel.")
        # heatmap: manual z-score across domains per pathway (seaborn z_score=0 has NO zero-std
        # guard -> a pathway flat across domains becomes an all-NaN row -> linkage ValueError).
        try:
            z = dfp.T.sub(dfp.T.mean(axis=1), axis=0).div(dfp.T.std(axis=1).replace(0, np.nan), axis=0)
            z = z.dropna(how="any")               # drop pathways constant across domains
            n_flat = dfp.shape[1] - z.shape[0]
            if n_flat:
                MANIFEST["warnings"].append(f"{n_flat} pathway(s) flat across domains; dropped from heatmap z-score.")
            n_pathways = z.shape[0]; n_domains = z.shape[1]
            # generous dynamic figsize so the title + all pathway labels never clip.
            w = max(8, 1.1 * n_domains + 5); h = max(7, 0.5 * n_pathways + 3)
            if n_pathways >= 2:
                g = sns.clustermap(
                    z, cmap="vlag", center=0, z_score=None, figsize=(w, h),
                    dendrogram_ratio=(0.12, 0.18),
                    cbar_pos=(0.02, 0.80, 0.03, 0.15),  # colorbar out of the heatmap area
                    cbar_kws={"label": "z-scored mean score (across domains)"})
                # show every pathway (row) label, no clipping; domains horizontal.
                g.ax_heatmap.set_yticks(np.arange(z.shape[0]) + 0.5)
                g.ax_heatmap.set_yticklabels(
                    [z.index[i] for i in g.dendrogram_row.reordered_ind], rotation=0, fontsize=7)
                plt.setp(g.ax_heatmap.get_xticklabels(), rotation=0)
                g.fig.suptitle(f"Pathway score per domain ({DOMAIN_KEY}) — relative within panel", y=1.02)
                g.fig.subplots_adjust(top=0.93)
                g.savefig(os.path.join(PLOTS_DIR, "F_pathway_scores_heatmap.png"), dpi=DPI, bbox_inches="tight")
                plt.close("all")
            else:
                # too few variable pathways to cluster -> plain heatmap of raw mean scores
                fig, ax = plt.subplots(figsize=(w, max(5, 0.35 * dfp.shape[1] + 2)))
                sns.heatmap(dfp.T, cmap="rocket", ax=ax, cbar_kws={"label": "mean score"})
                ax.set_title(f"Pathway mean score per domain ({DOMAIN_KEY})")
                save_fig(fig, "F_pathway_scores_heatmap.png")
        except Exception as e:
            log(f"[F] heatmap failed (table already saved): {type(e).__name__}: {e}")
            plt.close("all")
    except Exception as e:
        record("F_pathway_scores", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        plt.close("all")

    # Best-effort extra: Novae's own pathway_scores wrapper, ALSO on normalized data.
    try:
        an = build_adata_norm()
        novae.plot.pathway_scores(an, pathways=GENE_SETS, obs_key=DOMAIN_KEY,
                                  return_df=False, show=False)
        save_current("F_pathway_scores_heatmap_novae.png")
        record("F_pathway_scores_novae", "ok")
    except Exception as e:
        record("F_pathway_scores_novae", "failed", error=f"{type(e).__name__}: {e}")
        plt.close("all")


block_F_pathways()


# ============================================================
# G. Domain x slide clustermap  (Fig 2a)
# ============================================================
def block_G_clustermap():
    try:
        log("[G] seaborn.clustermap of domain proportions per sample")
        ct = pd.crosstab(adata.obs[SAMPLE_KEY], adata.obs[DOMAIN_KEY], dropna=False)  # rows=sample, cols=domain
        prop = ct.div(ct.sum(axis=1), axis=0)  # row-normalise -> proportions per sample
        prop.to_csv(os.path.join(TABLES_DIR, "domain_proportions_by_sample.csv"))
        # status annotation colours for the rows.
        samp_status = (adata.obs[[SAMPLE_KEY, STATUS_KEY]].drop_duplicates()
                       .set_index(SAMPLE_KEY)[STATUS_KEY].astype(str))
        status_pal = {"control": "#4C72B0", "c9": "#C44E52", "sporadic": "#DD8452"}
        row_colors = prop.index.to_series().map(
            lambda s: status_pal.get(samp_status.get(s, ""), "0.7"))
        n_samples = prop.shape[0]
        g_h = max(10, 0.4 * n_samples + 3)  # tall enough that all sample labels show
        g = sns.clustermap(prop, cmap="viridis", figsize=(8, g_h),
                           row_colors=row_colors.values,
                           col_cluster=True, row_cluster=True,
                           cbar_kws={"label": "proportion"})
        # show every sample (row) label.
        g.ax_heatmap.set_yticks(np.arange(prop.shape[0]) + 0.5)
        g.ax_heatmap.set_yticklabels(
            [prop.index[i] for i in g.dendrogram_row.reordered_ind], rotation=0, fontsize=7)
        g.fig.suptitle(f"Domain proportions per sample ({DOMAIN_KEY})", y=1.02)
        # status legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in status_pal.values()]
        g.ax_heatmap.legend(handles, list(status_pal.keys()), title="status",
                            bbox_to_anchor=(1.25, 1.0), loc="upper left", fontsize=7)
        path = os.path.join(PLOTS_DIR, "G_domain_by_sample_clustermap.png")
        g.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close("all")
        record("G_clustermap", "ok", plot=path)
    except Exception as e:
        record("G_clustermap", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        plt.close("all")


block_G_clustermap()


# ============================================================
# MANIFEST
# ============================================================
MANIFEST["corrected_obsm_key"] = CORRECTED_KEY
write_manifest()
log("Done.")
