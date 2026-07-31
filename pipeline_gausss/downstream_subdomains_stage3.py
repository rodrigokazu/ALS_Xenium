# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/downstream_subdomains_stage3.py
#
# Stage three. Characterises the sub-domains: differential expression, a pathway score
# heatmap, composition by sample and by status, a sub-domain by sample clustermap, and a
# faceted per-sample composition figure across all twenty sections. Everything saved as PDF
# and 300 dpi PNG.
#
# Reuses the analysis patterns from novae_downstream.py rather than reinventing them, which is
# why the two files look similar and why a fix in one usually belongs in the other.
#
# The biology it recovered was real structure rather than noise: a synaptic and motor-neuron
# sub-niche carrying the TDP-43 and ALS gene signal, a myelin-leaning grey matter sub-domain,
# a reactive astrocyte sub-domain, a vascular one, and a microglial and complement one. Five
# distinguishable sub-niches out of two parent domains.
#
# The by-status comparison is doubly confounded here, by segmentation version and by the fact
# that D1018 is itself disease-linked, so composition differences partly encode the parent
# split. Do not make an ALS-versus-control claim from this figure without the leave-two-out
# sensitivity run.
# ========================================================================================

"""
downstream_subdomains_stage3.py
============================================================
STAGE 3 of the ventral-horn (VH) sub-domain pipeline (sheff_als_spatial).

Goal: downstream characterisation of the Stage-2 Novae SUB-domains inside D1014+D1018:
DEG between sub-domains, pathway-score heatmap, sub-domain composition by sample / status,
sub-domain x sample clustermap, and a per-sample faceted composition figure across the
20 samples. All figures Nature-tier (nature_style) saved as BOTH .pdf and .png @300 dpi.

This REUSES the verified downstream patterns from novae_downstream.py (read 2026-06-18):
  * DEG: rank_genes_groups(wilcoxon) on a NORMALIZED+log COPY (raw stays canonical),
    'unassigned' dropped, mito genes excluded (no-op on this 0-MT panel but portable),
    CE_STMN2 / CE_UNC13A excluded from the per-domain MARKER analysis (engineered TDP-43
    cryptic-exon probes — pathology readouts, not domain markers; verified by DOMAIN).
    -> top-10 dotplot (DotPlot.savefig, which works in scanpy 1.9.8 where .fig is lazy)
       + deg_top10_by_subdomain.csv + full deg_by_subdomain.csv + seaborn fallback heatmap.
  * Pathways: the SAME GENE_SETS dict + score_genes(ctrl_size=25, n_bins=10, seed=0) on the
    normalized copy (verified safe against the ~480-gene background) -> z-scored clustermap
    (NaN-safe manual z-score; constant-across-subdomain pathways dropped) + CSV.
  * Composition: sub-domain by sample + by status (seg-version confound flagged IN-FIGURE)
    + sub-domain x sample seaborn clustermap (status row colours).
  * PER-SAMPLE: a sub-domain proportion table + a faceted grid figure across all samples.

PRIMARY sub-domain key auto-detect: read summary/stage2_manifest.json for
'primary_subdomain_key'; else pick the novae_subdomains_<L> obs key matching PRIMARY_LEVEL;
else the last (largest-level) novae_subdomains_* key present.

INPUT (read-only): ALT_DIR/novae_subset_domains.h5ad (Stage-2 output).
OUTPUT (created): ALT_DIR/downstream/{plots,tables,summary}/ + summary/stage3_manifest.json

Env: novae_env (Python 3.11 venv); module load python/3.11.1 gcc/13.3.0. CPU only.
See run_downstream_subdomains_stage3.sh.

ROBUSTNESS: matplotlib Agg; every analysis/plot block in try/except that logs+continues;
print(flush=True); preconditions asserted at load; raw counts canonical (normalize only a
derived copy for DEG + pathway scoring).
"""

import os
import re
import json
import math
import traceback
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import nature_style as ns

# ============================================================
# CONFIG
# ============================================================
ALT_DIR = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_D1014_D1018_subset"
INPUT_H5AD = os.path.join(ALT_DIR, "novae_subset_domains.h5ad")
STAGE2_MANIFEST = os.path.join(ALT_DIR, "summary", "stage2_manifest.json")

OUTPUT_DIR = os.path.join(ALT_DIR, "downstream")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summary")

SUBDOMAIN_PREFIX = "novae_subdomains_"   # stage-2 obs keys are novae_subdomains_<count>
PRIMARY_N_FALLBACK = 6                   # preferred sub-domain COUNT if stage2 manifest is absent

SAMPLE_KEY = "sample"
STATUS_KEY = "status"
SLIDE_KEY = "run_id"
SEG_KEY = "seg_version"
UNASSIGNED = "unassigned"

DEG_TOP_N = 10
SCORE_CTRL_SIZE = 25     # lowered for the ~480-gene panel background (verified safe)
SCORE_N_BINS = 10
SEED = 0

# Engineered TDP-43 cryptic-exon PROBES: pathology readouts, not domain identity markers.
# EXCLUDE from per-sub-domain DEG markers; KEEP in the object + in the pathway gene sets.
EXCLUDE_FROM_MARKERS_BASE = ["CE_STMN2", "CE_UNC13A"]

# Curated pathway gene sets (reused verbatim from novae_downstream.py; DOMAIN-validated,
# each >=4 panel genes; ALS set renamed to ALS_familial_and_GWAS_genes). Scores are RELATIVE
# WITHIN THIS PANEL (background pool is the panel, not the transcriptome).
GENE_SETS = {
    "Motor_neuron": ["STMN2", "SLC17A6", "MNX1", "SNCG", "RBFOX3", "TAC1", "NOS1"],
    "TDP43_cryptic_exon": ["CE_STMN2", "CE_UNC13A", "STMN2", "UNC13A"],
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

# ============================================================
# SETUP
# ============================================================
for d in (OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, SUMMARY_DIR):
    os.makedirs(d, exist_ok=True)


def log(m):
    print(m, flush=True)


MANIFEST = {
    "stage": 3,
    "input_h5ad": INPUT_H5AD,
    "output_dir": OUTPUT_DIR,
    "blocks": {},
    "warnings": [],
}


def record(block, status, **detail):
    MANIFEST["blocks"][block] = {"status": status, **detail}
    log(f"[manifest] {block}: {status}" + (f"  {detail}" if detail else ""))


def write_manifest():
    path = os.path.join(SUMMARY_DIR, "stage3_manifest.json")
    try:
        with open(path, "w") as f:
            json.dump(MANIFEST, f, indent=2, default=str)
        log(f"[manifest] wrote {path}")
    except Exception as e:
        log(f"[manifest] FAILED to write {path}: {type(e).__name__}: {e}")


def save_both_named(fig, name):
    return ns.save_both(fig, os.path.join(PLOTS_DIR, name))


# ============================================================
# LOAD + PRECONDITIONS
# ============================================================
log(f"[load] {INPUT_H5AD}")
adata = sc.read_h5ad(INPUT_H5AD)
log(f"[load] {adata.shape}  obs={list(adata.obs.columns)}  layers={list(adata.layers.keys())}")

assert "counts" in adata.layers, "layers['counts'] (raw) missing"
for k in (SAMPLE_KEY, STATUS_KEY):
    assert k in adata.obs, f"obs['{k}'] missing"

# ---- auto-detect the PRIMARY sub-domain key ----
subdomain_obs_keys = sorted(
    [c for c in adata.obs.columns if c.startswith(SUBDOMAIN_PREFIX)],
    key=lambda c: (int(re.sub(r"\D", "", c)) if re.sub(r"\D", "", c) else -1),
)
assert subdomain_obs_keys, f"no obs keys starting with '{SUBDOMAIN_PREFIX}' — Stage 2 must run first"
log(f"[detect] sub-domain obs keys present: {subdomain_obs_keys}")

primary_key = None
manifest_primary = None
if os.path.exists(STAGE2_MANIFEST):
    try:
        with open(STAGE2_MANIFEST) as f:
            m2 = json.load(f)
        manifest_primary = m2.get("primary_subdomain_key")
        if manifest_primary in adata.obs.columns:
            primary_key = manifest_primary
            log(f"[detect] primary from stage2 manifest: '{primary_key}'")
        elif manifest_primary:
            MANIFEST["warnings"].append(
                f"stage2 manifest primary_subdomain_key='{manifest_primary}' not in obs; falling back.")
    except Exception as e:
        MANIFEST["warnings"].append(f"could not read stage2 manifest: {type(e).__name__}: {e}")
        log(f"[detect] stage2 manifest unreadable: {type(e).__name__}: {e}")

if primary_key is None:
    # keys are novae_subdomains_<count>; pick the one whose count is nearest PRIMARY_N_FALLBACK
    def _count(k):
        try:
            return int(k.rsplit("_", 1)[-1])
        except Exception:
            return 10 ** 9
    primary_key = min(subdomain_obs_keys, key=lambda k: (abs(_count(k) - PRIMARY_N_FALLBACK), _count(k)))
    log(f"[detect] manifest absent; primary nearest n={PRIMARY_N_FALLBACK}: '{primary_key}'")

MANIFEST["primary_subdomain_key"] = primary_key
MANIFEST["manifest_primary_subdomain_key"] = manifest_primary
MANIFEST["subdomain_obs_keys"] = subdomain_obs_keys
MANIFEST["n_cells"] = int(adata.n_obs)
MANIFEST["n_genes"] = int(adata.n_vars)
MANIFEST["n_samples"] = int(adata.obs[SAMPLE_KEY].nunique())

subdomains_all = sorted(adata.obs[primary_key].astype(str).unique())
log(f"[detect] PRIMARY key='{primary_key}'  sub-domains={subdomains_all}")

# ---- panel / mito / CE_ bookkeeping (MT removal is a documented no-op on this 0-MT panel) ----
panel = set(map(str, adata.var_names))
_uv = adata.var_names.str.upper()
mt = _uv.str.match(r"^MT-") | _uv.str.startswith("MT.")
MITO_GENES = list(adata.var_names[mt])
N_MITO = int(mt.sum())
MANIFEST["n_mito_genes_removed"] = N_MITO
MANIFEST["mito_genes_removed"] = MITO_GENES
log(f"[mito] n_mito_genes={N_MITO} genes={MITO_GENES or 'none'}")

EXCLUDE_FROM_MARKERS = [g for g in EXCLUDE_FROM_MARKERS_BASE if g in panel]
MANIFEST["excluded_from_markers"] = EXCLUDE_FROM_MARKERS
log(f"[markers] excluded from DEG (kept for pathways): {EXCLUDE_FROM_MARKERS}")

# gene-set membership table (provenance)
try:
    gs_rows = []
    for name, genes in GENE_SETS.items():
        present = [g for g in genes if g in panel]
        gs_rows.append({"pathway": name, "n_present": len(present), "n_total": len(genes),
                        "genes_present": ";".join(present),
                        "genes_missing": ";".join([g for g in genes if g not in panel])})
    pd.DataFrame(gs_rows).to_csv(os.path.join(TABLES_DIR, "gene_sets_membership.csv"), index=False)
    MANIFEST["gene_sets"] = {k: list(v) for k, v in GENE_SETS.items()}
    log(f"[panel] wrote gene_sets_membership.csv ({len(gs_rows)} pathways)")
except Exception as e:
    log(f"[panel] gene_sets_membership.csv failed: {type(e).__name__}: {e}")

ns.apply_style()

# ---- shared NORMALIZED+log copy (raw canonical; derived ONLY for DEG + score_genes;
#      'unassigned' removed; mito removed when present). Built once, reused. ----
ADATA_NORM = None


def build_adata_norm():
    global ADATA_NORM
    if ADATA_NORM is not None:
        return ADATA_NORM
    log("[norm] building normalized+log copy (non-'unassigned' cells) for DEG + pathways")
    keep = adata.obs[primary_key].astype(str) != UNASSIGNED
    an = adata[keep].copy()
    if N_MITO:
        an = an[:, ~an.var_names.isin(MITO_GENES)].copy()
        log(f"[norm] dropped {N_MITO} mito gene(s) -> {an.n_vars} genes")
    an.X = an.layers["counts"].copy()
    sc.pp.normalize_total(an, target_sum=1e4)
    sc.pp.log1p(an)
    an.obs[primary_key] = an.obs[primary_key].astype("category").cat.remove_unused_categories()
    ADATA_NORM = an
    return an


# colours for the sub-domains (consistent across all figures)
SUBDOMAIN_COLORS = ns.domain_colors(subdomains_all, unassigned=UNASSIGNED)


# ============================================================
# A. DEG between sub-domains  (rank_genes_groups wilcoxon)
# ============================================================
def block_A_deg():
    try:
        an = build_adata_norm()
        if EXCLUDE_FROM_MARKERS:
            an = an[:, ~an.var_names.isin(EXCLUDE_FROM_MARKERS)].copy()
            log(f"[A] excluded {EXCLUDE_FROM_MARKERS} from DEG -> {an.n_vars} genes")
        n_groups = int(an.obs[primary_key].nunique())
        if n_groups < 2:
            record("A_deg", "skipped", reason=f"only {n_groups} sub-domain group after dropping 'unassigned'")
            return
        log(f"[A] rank_genes_groups(groupby='{primary_key}', wilcoxon) on {an.n_obs} cells, {n_groups} groups")
        sc.tl.rank_genes_groups(an, groupby=primary_key, method="wilcoxon")

        deg_df = sc.get.rank_genes_groups_df(an, group=None)
        deg_path = os.path.join(TABLES_DIR, "deg_by_subdomain.csv")
        deg_df.to_csv(deg_path, index=False)
        log(f"[A] full DEG table -> {deg_path} ({len(deg_df)} rows)")

        # per-sub-domain top-N table
        top_path = os.path.join(TABLES_DIR, "deg_top10_by_subdomain.csv")
        top_df = None
        try:
            rows = []
            for grp in list(an.obs[primary_key].cat.categories):
                gdf = sc.get.rank_genes_groups_df(an, group=grp)
                gdf = gdf.sort_values("scores", ascending=False).head(DEG_TOP_N)
                rows.append(pd.DataFrame({
                    "subdomain": grp, "gene": gdf["names"].values, "score": gdf["scores"].values,
                    "logfoldchanges": gdf["logfoldchanges"].values, "pvals_adj": gdf["pvals_adj"].values,
                }))
            top_df = pd.concat(rows, ignore_index=True)
            top_df.to_csv(top_path, index=False)
            log(f"[A] per-sub-domain top-{DEG_TOP_N} -> {top_path} ({len(top_df)} rows)")
        except Exception as e:
            log(f"[A] top-{DEG_TOP_N} table failed: {type(e).__name__}: {e}")

        # top-10 dotplot — DotPlot.savefig works in scanpy 1.9.8 (where .fig is lazy/None).
        try:
            dp_figsize = (max(10, 0.30 * n_groups * DEG_TOP_N + 3), 0.6 * n_groups + 3)
            dp = sc.pl.rank_genes_groups_dotplot(
                an, n_genes=DEG_TOP_N, show=False, return_fig=True, figsize=dp_figsize)
            base = os.path.join(PLOTS_DIR, "A_deg_dotplot_top10")
            if hasattr(dp, "savefig"):
                dp.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
                dp.savefig(base + ".png", dpi=300, bbox_inches="tight")
                plt.close("all")
            elif getattr(dp, "fig", None) is not None:
                ns.save_both(dp.fig, base)
            else:
                # last resort: whatever is on the current figure
                fig = plt.gcf()
                ns.save_both(fig, base)
            log(f"[A] dotplot -> {base}.pdf/.png")
        except Exception as e:
            log(f"[A] dotplot failed: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            plt.close("all")

        # seaborn fallback heatmap of mean log-norm expression of the top-N marker union
        try:
            if top_df is not None and len(top_df):
                marker_union = [g for g in dict.fromkeys(top_df["gene"].tolist()) if g in an.var_names]
                if marker_union:
                    df_expr = sc.get.obs_df(an, keys=[primary_key] + marker_union)
                    mean_expr = df_expr.groupby(primary_key, observed=True)[marker_union].mean()
                    fh = max(5, 0.18 * len(marker_union) + 2)
                    fw = max(5, 0.35 * n_groups + 3)
                    fig, ax = plt.subplots(figsize=(fw, fh))
                    sns.heatmap(mean_expr.T, cmap="rocket", ax=ax,
                                cbar_kws={"label": "mean log-norm expr"})
                    ax.set_title(f"Top-{DEG_TOP_N} DEG markers per sub-domain ({primary_key})")
                    ax.set_yticks(np.arange(len(marker_union)) + 0.5)
                    ax.set_yticklabels(marker_union, rotation=0, fontsize=5)
                    plt.setp(ax.get_xticklabels(), rotation=90)
                    save_both_named(fig, "A_deg_top10_heatmap")
        except Exception as e:
            log(f"[A] fallback heatmap failed: {type(e).__name__}: {e}")
            plt.close("all")

        record("A_deg", "ok", n_rows=int(len(deg_df)), table=deg_path, top_table=top_path,
               n_groups=n_groups, n_cells=int(an.n_obs))
    except Exception as e:
        record("A_deg", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        plt.close("all")


# ============================================================
# B. Pathway score heatmap per sub-domain
# ============================================================
def block_B_pathways():
    try:
        an = build_adata_norm()
        log(f"[B] score_genes for {len(GENE_SETS)} pathways "
            f"(ctrl_size={SCORE_CTRL_SIZE}, n_bins={SCORE_N_BINS}, seed={SEED})")
        score_cols = []
        for name, genes in GENE_SETS.items():
            present = [g for g in genes if g in an.var_names]
            if len(present) < 4:
                log(f"[B] skip pathway '{name}': only {len(present)} panel genes")
                continue
            col = f"PW_{name}"
            sc.tl.score_genes(an, gene_list=present, score_name=col,
                              ctrl_size=SCORE_CTRL_SIZE, n_bins=SCORE_N_BINS, random_state=SEED)
            score_cols.append(col)
        dfp = an.obs.groupby(primary_key, observed=True)[score_cols].mean()
        dfp.columns = [c[3:] for c in dfp.columns]
        path = os.path.join(TABLES_DIR, "pathway_scores_by_subdomain.csv")
        dfp.to_csv(path)
        # record table success FIRST (a heatmap failure must not mask a valid CSV)
        record("B_pathway_scores", "ok", table=path, shape=list(dfp.shape),
               note="score_genes on normalized data; relative within panel.")
        # NaN-safe manual z-score across sub-domains (seaborn z_score has no zero-std guard).
        try:
            z = dfp.T.sub(dfp.T.mean(axis=1), axis=0).div(dfp.T.std(axis=1).replace(0, np.nan), axis=0)
            z = z.dropna(how="any")
            n_flat = dfp.shape[1] - z.shape[0]
            if n_flat:
                MANIFEST["warnings"].append(f"{n_flat} pathway(s) flat across sub-domains; dropped from heatmap.")
            n_pw, n_sd = z.shape[0], z.shape[1]
            w = max(6, 0.9 * n_sd + 4)
            h = max(5, 0.35 * n_pw + 2)
            if n_pw >= 2 and n_sd >= 2:
                g = sns.clustermap(z, cmap="vlag", center=0, z_score=None, figsize=(w, h),
                                   dendrogram_ratio=(0.14, 0.18),
                                   cbar_pos=(0.02, 0.80, 0.03, 0.15),
                                   cbar_kws={"label": "z-scored mean score (across sub-domains)"})
                g.ax_heatmap.set_yticks(np.arange(z.shape[0]) + 0.5)
                g.ax_heatmap.set_yticklabels(
                    [z.index[i] for i in g.dendrogram_row.reordered_ind], rotation=0, fontsize=6)
                plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
                g.fig.suptitle(f"Pathway score per sub-domain ({primary_key}) — relative within panel", y=1.02)
                g.fig.subplots_adjust(top=0.93)
                base = os.path.join(PLOTS_DIR, "B_pathway_scores_heatmap")
                g.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
                g.savefig(base + ".png", dpi=300, bbox_inches="tight")
                plt.close("all")
            else:
                fig, ax = plt.subplots(figsize=(w, max(4, 0.3 * dfp.shape[1] + 2)))
                sns.heatmap(dfp.T, cmap="rocket", ax=ax, cbar_kws={"label": "mean score"})
                ax.set_title(f"Pathway mean score per sub-domain ({primary_key})")
                save_both_named(fig, "B_pathway_scores_heatmap")
        except Exception as e:
            log(f"[B] heatmap failed (table saved): {type(e).__name__}: {e}")
            log(traceback.format_exc())
            plt.close("all")
    except Exception as e:
        record("B_pathway_scores", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        plt.close("all")


# ============================================================
# C. Sub-domain composition by sample + by status (seg-confound flagged in-figure)
# ============================================================
def _stacked_prop(crosstab, title, name, note=None, rotate=90):
    prop = crosstab.div(crosstab.sum(axis=0), axis=1)   # cols=groups, rows=sub-domains
    groups = list(prop.columns)
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    fig, ax = plt.subplots(figsize=(max(3.4, 0.32 * len(groups) + 2.2), 3.4))
    for d in [str(i) for i in prop.index]:
        vals = prop.loc[d].values if d in prop.index else prop.loc[int(d)].values
        ax.bar(x, vals, bottom=bottom, color=SUBDOMAIN_COLORS.get(d, "0.6"), label=d, linewidth=0)
        bottom += vals
    ax.set_ylabel("proportion of cells")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=rotate, fontsize=6)
    ncol = 2 if prop.shape[0] > 8 else 1
    ax.legend(title="sub-domain", loc="center left", bbox_to_anchor=(1.01, 0.5),
              borderaxespad=0.0, ncol=ncol)
    if note:
        ax.text(0.0, -0.42, note, transform=ax.transAxes, fontsize=5, color="firebrick", wrap=True)
    save_both_named(fig, name)


def block_C_composition():
    try:
        ct_status = pd.crosstab(adata.obs[primary_key], adata.obs[STATUS_KEY], dropna=False)
        ct_status.to_csv(os.path.join(TABLES_DIR, "subdomain_counts_by_status.csv"))
        seg_note = ("TWO CONFOUNDS — do not read as biology without gating: (1) seg-version "
                    "(2 reseg controls); (2) COMPOSITION: D1018 (reactive) is itself disease-linked, "
                    "so the D1014:D1018 ratio entering this subset differs ALS-vs-control, biasing "
                    "sub-domain proportions. Needs leave-2-out + per-sample unit + per-sample subset-n.")
        _stacked_prop(ct_status, f"Sub-domain composition by status ({primary_key})",
                      "C_subdomain_proportions_by_status", note=seg_note, rotate=0)
        record("C_by_status", "ok")
    except Exception as e:
        record("C_by_status", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        plt.close("all")

    try:
        ct_sample = pd.crosstab(adata.obs[primary_key], adata.obs[SAMPLE_KEY], dropna=False)
        ct_sample.to_csv(os.path.join(TABLES_DIR, "subdomain_counts_by_sample.csv"))
        _stacked_prop(ct_sample, f"Sub-domain composition by sample ({primary_key})",
                      "C_subdomain_proportions_by_sample", rotate=90)
        record("C_by_sample", "ok")
    except Exception as e:
        record("C_by_sample", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        plt.close("all")


# ============================================================
# D. Sub-domain x sample clustermap (status row colours)
# ============================================================
def block_D_clustermap():
    try:
        ct = pd.crosstab(adata.obs[SAMPLE_KEY], adata.obs[primary_key], dropna=False)  # rows=sample
        prop = ct.div(ct.sum(axis=1), axis=0)
        prop.to_csv(os.path.join(TABLES_DIR, "subdomain_proportions_by_sample.csv"))
        if prop.shape[0] < 2 or prop.shape[1] < 2:
            record("D_clustermap", "skipped", reason=f"shape {prop.shape} too small to cluster")
            return
        samp_status = (adata.obs[[SAMPLE_KEY, STATUS_KEY]].drop_duplicates()
                       .set_index(SAMPLE_KEY)[STATUS_KEY].astype(str))
        status_pal = {"control": "#0072B2", "c9": "#D55E00", "sporadic": "#E69F00"}
        row_colors = prop.index.to_series().map(
            lambda s: status_pal.get(samp_status.get(s, ""), "0.7"))
        n_samples = prop.shape[0]
        g_h = max(5, 0.3 * n_samples + 2)
        g = sns.clustermap(prop, cmap="viridis", figsize=(max(6, 0.5 * prop.shape[1] + 4), g_h),
                           row_colors=row_colors.values, col_cluster=True, row_cluster=True,
                           cbar_kws={"label": "proportion"})
        g.ax_heatmap.set_yticks(np.arange(prop.shape[0]) + 0.5)
        g.ax_heatmap.set_yticklabels(
            [prop.index[i] for i in g.dendrogram_row.reordered_ind], rotation=0, fontsize=6)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
        g.fig.suptitle(f"Sub-domain proportions per sample ({primary_key})", y=1.02)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in status_pal.values()]
        g.ax_heatmap.legend(handles, list(status_pal.keys()), title="status",
                            bbox_to_anchor=(1.30, 1.0), loc="upper left", fontsize=6)
        base = os.path.join(PLOTS_DIR, "D_subdomain_by_sample_clustermap")
        g.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
        g.savefig(base + ".png", dpi=300, bbox_inches="tight")
        plt.close("all")
        record("D_clustermap", "ok", plot=base + ".pdf")
    except Exception as e:
        record("D_clustermap", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        plt.close("all")


# ============================================================
# E. PER-SAMPLE sub-domain proportion table + faceted grid figure
# ============================================================
def block_E_per_sample():
    # proportion table (rows=sample, cols=sub-domain)
    prop = None
    try:
        ct = pd.crosstab(adata.obs[SAMPLE_KEY], adata.obs[primary_key], dropna=False)
        prop = ct.div(ct.sum(axis=1), axis=0)
        prop.to_csv(os.path.join(TABLES_DIR, "per_sample_subdomain_proportions.csv"))
        ct.to_csv(os.path.join(TABLES_DIR, "per_sample_subdomain_counts.csv"))
        record("E_per_sample_table", "ok", n_samples=int(prop.shape[0]))
    except Exception as e:
        record("E_per_sample_table", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())

    # faceted grid: one small stacked bar per sample (a single bar of sub-domain proportions)
    try:
        if prop is None or prop.shape[0] == 0:
            record("E_per_sample_facet", "skipped", reason="no per-sample proportions")
            return
        samp_status = (adata.obs[[SAMPLE_KEY, STATUS_KEY]].drop_duplicates()
                       .set_index(SAMPLE_KEY)[STATUS_KEY].astype(str))
        # order samples by status then name so facets group disease state visually
        order = sorted(prop.index.astype(str),
                       key=lambda s: (str(samp_status.get(s, "zzz")), s))
        sub_cols = [str(c) for c in prop.columns]
        n = len(order)
        ncols = min(5, n) if n else 1
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(1.5 * ncols + 1.5, 1.7 * nrows + 0.8),
                                 squeeze=False)
        for i, samp in enumerate(order):
            ax = axes[i // ncols][i % ncols]
            bottom = 0.0
            row = prop.loc[samp] if samp in prop.index else prop.loc[int(samp)]
            for d in sub_cols:
                v = float(row[d]) if d in row.index else float(row[int(d)])
                ax.bar(0, v, bottom=bottom, width=0.8,
                       color=SUBDOMAIN_COLORS.get(d, "0.6"), linewidth=0)
                bottom += v
            st = samp_status.get(samp, "?")
            ax.set_title(f"{samp}\n({st})", fontsize=5)
            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            if i % ncols == 0:
                ax.set_ylabel("prop.", fontsize=6)
            else:
                ax.set_yticklabels([])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        # blank any unused panels
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        # shared legend outside, on the right
        handles = [plt.Rectangle((0, 0), 1, 1, color=SUBDOMAIN_COLORS.get(d, "0.6")) for d in sub_cols]
        fig.legend(handles, sub_cols, title="sub-domain", loc="center left",
                   bbox_to_anchor=(1.0, 0.5), fontsize=5, ncol=1)
        fig.suptitle(f"Per-sample sub-domain composition ({primary_key})", y=1.0, fontsize=8)
        fig.tight_layout(rect=(0, 0, 0.98, 0.98))
        save_both_named(fig, "E_per_sample_subdomain_facets")
        record("E_per_sample_facet", "ok", n_samples=n)
    except Exception as e:
        record("E_per_sample_facet", "failed", error=f"{type(e).__name__}: {e}")
        log(traceback.format_exc())
        plt.close("all")


# ============================================================
# RUN
# ============================================================
block_A_deg()
block_B_pathways()
block_C_composition()
block_D_clustermap()
block_E_per_sample()

write_manifest()
log("Stage 3 done.")
