#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/dual_pass/make_persample_notebooks.py
#
# The original generator that fanned the SD03522_BG notebook out to one notebook per sample.
# Do not regenerate from it. Its output predates both source patches, so a regenerated
# notebook loses the counts fix and the v6 relabel. Kept for provenance only.
# ========================================================================================

"""
Fan the SD03522_BG dual-pass QC notebook out into one notebook per sample.

Reads  : (RK)QC_per_sample_dualpass.ipynb
Writes : (RK)QC_per_sample_dualpass_<SAMPLE>.ipynb   (same folder, one per h5ad)

What changes in each copy
  - every SD03522_BG / SD03522BG string is swapped for that sample's id, so the
    input h5ad, the figure filenames and the plot titles all point at one sample
  - outputs and execution counts are stripped, so the copies are small
  - the hard-coded starting cell count (115371) becomes adata.n_obs
  - two new cells are inserted after the QC histograms: one draws the Novae
    domains, one turns them into a QC table
  - the trailing legacy section (SD01922_BI exploration and the whole-dataset
    pre-QC block) is dropped; the original notebook still has it
  - a short comment goes at the top of each code cell saying what it does
"""

import copy
import json
import re
from pathlib import Path

NB_DIR = Path("/home/rodrigok/Notebooks/Spatial_MN_correct")
SRC_NB = NB_DIR / "(RK)QC_per_sample_dualpass.ipynb"
H5_DIR = Path(
    "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/"
    "Novae_persample_INDEPENDENT_FINAL/per_sample"
)

REF_SAMPLE = "SD03522_BG"
REF_COMPACT = "SD03522BG"

# The per-sample pipeline runs from cell 0 to cell 47. Everything after that is
# leftover exploration of a different sample plus the old concatenated-object
# workflow, so the copies stop at 47.
KEEP_UPTO = 47

# The two new cells go in after cell 13 (the QC histograms), which is the first
# point where total_counts and n_genes_by_counts exist.
INSERT_AFTER = 13


# --------------------------------------------------------------------------- #
# Comments prepended to each original code cell.
# --------------------------------------------------------------------------- #
CELL_COMMENTS = {
    0: "Imports. scanpy and squidpy do the analysis, the scipy/sklearn bits are\n"
       "for the marker statistics and cluster agreement checks further down.",
    2: "Where figures go and how sharp they are. FIGDIR is also handed to scanpy\n"
       "so its own sc.pl.* calls save next to the hand-rolled matplotlib ones.",
    5: "Marker dictionary for the manual annotation. Each gene maps to one cell\n"
       "type and carries the reason it is there, so the choices stay auditable.",
    7: "Load the sample. This h5ad already carries the Novae domain labels and\n"
       "the spatial coordinates. A raw copy of X is stashed in layers['counts']\n"
       "so normalisation can be redone from scratch in pass 2.",
    8: "Sanity check that the object holds the one sample it should.",
    9: "Print the panel so gene names can be checked before markers are used.",
    11: "Work on a copy from here, and confirm the centroid columns are present.",
    13: "First look at the count distributions. The dashed lines are the Salas et\n"
        "al. cut-offs of 10 transcripts and 10 genes per cell, drawn but not yet\n"
        "applied. Read this before choosing the thresholds below.",
    14: "The same QC metrics painted on tissue coordinates. Low-count patches\n"
        "that follow tissue structure are biology. Low-count patches that follow\n"
        "a stripe or a blob are usually a smear or an out-of-focus region.",
    16: "Apply the transcript cut-off and redraw the tissue, so it is obvious\n"
        "whether the filter ate a real anatomical region.",
    18: "Preview of the density filter. Cells with fewer than 8 neighbours inside\n"
        "100 um are flagged in red. Nothing is removed yet. Isolated cells at the\n"
        "tissue edge or floating in the empty background show up here.",
    20: "Apply the density filter.",
    22: "Normalise to 100 counts per cell and log-transform, following Salas et\n"
        "al. for Xenium-scale panels. STMN2 is plotted afterwards as a spot check\n"
        "that the ventral horn still looks like a ventral horn.",
    24: "Scale and run PCA on all genes. A 480-gene panel has no HVG step to do.\n"
        "The geometric elbow is drawn for reference, but pass 1 keeps all 50 PCs.",
    26: "Leiden at three resolutions, shown on UMAP and on tissue side by side.\n"
        "Resolution 1.0 is what the smear hunt below uses. Resolution 1.5 becomes\n"
        "the working clustering.",
    27: "Save the pass-1 object, so the smear review can start from a clean read.",
    29: "Read the pass-1 object back.",
    30: "Per-cluster QC table sorted by median counts. Clusters at the top of this\n"
        "table are the smear candidates: many cells, few transcripts, few genes.",
    32: "Map the suspect clusters onto tissue. A real cell type sits where its\n"
        "anatomy sits. A smear sits in a band, an edge or a stripe.\n"
        "CHECK THIS PER SAMPLE. The cluster ids below were carried over from the\n"
        "reference notebook and almost certainly mean something else here.",
    34: "Drop the smear clusters and redraw.\n"
        "CHECK THIS PER SAMPLE before running, same reason as the cell above.",
    35: "Numbers for the methods section: what survived, and what fraction of the\n"
        "starting cells that is.",
    37: "Pass 2. Renormalise from the untouched counts layer now that the debris\n"
        "is gone, then redo PCA, neighbours and UMAP. Pass 1 embeddings were\n"
        "pulled towards the smear, which is the whole point of doing this twice.",
    39: "Check which canonical markers actually exist in the panel before scoring\n"
        "anything. A cell type with one surviving marker cannot be scored.",
    40: "Score each cell type and assign every cell to its highest-scoring one.\n"
        "This is a rough first pass, not a final annotation. Types with a single\n"
        "marker are printed separately and need a manual call.",
    42: "Wilcoxon markers per assigned type. If the dotplot does not recover the\n"
        "genes that were scored, the assignment is not holding up.",
    44: "Moran's I on the first 200 genes for a quick spatially variable gene\n"
        "list. Wrapped in a try block because it is a nice-to-have, not a gate.",
    45: "Object summary after pass 2.",
    46: "Save the pass-2 object.",
    47: "Cell types on UMAP and on tissue. The tissue panel is the one that\n"
        "matters here, since the annotation is only believable if the types land\n"
        "where the anatomy says they should.",
}


# --------------------------------------------------------------------------- #
# New cell 1: draw the Novae domains
# --------------------------------------------------------------------------- #
NOVAE_MD = """## Novae spatial domains

The domains were computed per sample by the independent Novae run and travel
with the h5ad. They are used here as a QC lens: a domain that is really a smear,
a fold or a detachment tends to look nothing like the rest of the tissue in the
count statistics, and it tends to sit in one contiguous blob.
"""

NOVAE_PLOT_CELL = '''# Draw the Novae domains that came with the file, one panel per resolution.
# Novae was run at four levels of granularity (n4, n6, n8, n10 domains). Cells
# it could not place are labelled 'unassigned' and are drawn in grey; they
# cluster wherever the neighbourhood graph is thin, which is itself a QC signal.
# The last panel shows neighborhood_valid, Novae's own flag for cells whose
# spatial neighbourhood was too sparse to build a niche from.

DOMAIN_KEYS = [k for k in ['novae_domains_n4', 'novae_domains_n6',
                           'novae_domains_n8', 'novae_domains_n10']
               if k in sample.obs.columns]
print(f"Novae domain columns found: {DOMAIN_KEYS}")

panels = list(DOMAIN_KEYS)
if 'neighborhood_valid' in sample.obs.columns:
    panels.append('neighborhood_valid')

ncol = 2
nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 6.5 * nrow), dpi=200)
axes = np.atleast_1d(axes).ravel()

x = sample.obs['x_centroid'].values
y = sample.obs['y_centroid'].values

for ax, key in zip(axes, panels):
    vals = sample.obs[key]

    if key == 'neighborhood_valid':
        ok = np.asarray(vals).astype(bool)
        ax.scatter(x[ok], y[ok], s=0.4, c='#1A56DB', edgecolors='none',
                   rasterized=True, label=f'valid (n={ok.sum():,})')
        ax.scatter(x[~ok], y[~ok], s=1.2, c='#C53030', edgecolors='none',
                   rasterized=True, label=f'sparse (n={(~ok).sum():,})')
        ax.set_title('Novae neighbourhood validity', fontsize=10,
                     fontweight='bold', pad=8)
        ax.legend(fontsize=7, frameon=False, markerscale=6, loc='upper right')
    else:
        cats = list(vals.cat.categories) if hasattr(vals, 'cat') else sorted(vals.unique())
        # 'unassigned' is always grey; everything else takes a tab20 colour.
        real = [c for c in cats if c != 'unassigned']
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(real), 1)))
        colour = {c: cmap[i] for i, c in enumerate(real)}
        colour['unassigned'] = (0.82, 0.82, 0.82, 1.0)

        for c in cats:
            m = (vals == c).values
            if not m.any():
                continue
            ax.scatter(x[m], y[m], s=0.4, c=[colour[c]], edgecolors='none',
                       rasterized=True, label=f'{c} ({m.sum():,})')
        ax.set_title(f"{key} ({len(real)} domains)", fontsize=10,
                     fontweight='bold', pad=8)
        ax.legend(fontsize=6, frameon=False, markerscale=6, ncol=1,
                  loc='upper left', bbox_to_anchor=(1.01, 1.0))

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

for ax in axes[len(panels):]:
    ax.axis('off')

plt.tight_layout()
plt.savefig(FIGDIR / '(RK)SD03522BG_novae_domains_spatial.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(FIGDIR / '(RK)SD03522BG_novae_domains_spatial.pdf',
            bbox_inches='tight', facecolor='white')
plt.show()

for key in DOMAIN_KEYS:
    counts = sample.obs[key].value_counts()
    print(f"\\n{key}")
    print((counts / counts.sum() * 100).round(2).to_string())
'''


# --------------------------------------------------------------------------- #
# New cell 2: domain statistics for QC
# --------------------------------------------------------------------------- #
NOVAE_STATS_CELL = '''# Turn the Novae domains into a QC table.
#
# The idea is simple. A domain that is tissue should look like tissue: decent
# transcript counts, a sensible number of genes, cell areas in the usual range,
# a small share of negative-control counts and neighbours all around it. A
# domain that is a smear, a fold, a detachment or an empty-field artefact fails
# several of those at once, and it fails them together in one place on the slide.
#
# Each metric below is turned into a z-score across domains (sign flipped where
# low is bad), and the mean of those z-scores becomes suspicion_score. High
# score means "look at this domain before you trust it". It is a ranking aid,
# not a verdict, so read it next to the maps in the previous cell.

DOMAIN_QC_KEY = 'novae_domains_n8'      # resolution used for the table
COMPUTE_NEIGHBOUR_DENSITY = True        # local density, the strongest smear cue
DENSITY_RADIUS_UM = 100

assert DOMAIN_QC_KEY in sample.obs.columns, f"{DOMAIN_QC_KEY} not in obs"

# QC metrics may not exist yet if this cell is run out of order.
if 'total_counts' not in sample.obs.columns or 'n_genes_by_counts' not in sample.obs.columns:
    sc.pp.calculate_qc_metrics(sample, percent_top=[20, 50, 100, 200],
                               log1p=False, inplace=True)

obs = sample.obs

# Share of each cell's counts that landed on negative controls rather than on
# real probes. Elevated in debris and in regions with poor optical quality.
neg_cols = [c for c in ['control_probe_counts', 'control_codeword_counts',
                        'genomic_control_counts', 'unassigned_codeword_counts',
                        'deprecated_codeword_counts'] if c in obs.columns]
if neg_cols:
    neg_total = obs[neg_cols].sum(axis=1)
    obs['neg_control_frac'] = neg_total / (obs['total_counts'] + neg_total).clip(lower=1)
else:
    obs['neg_control_frac'] = np.nan

# Nucleus to cell area ratio. Segmentation that has run away from the nucleus
# gives tiny ratios; cells with no nucleus at all give zero.
if {'nucleus_area', 'cell_area'}.issubset(obs.columns):
    obs['nucleus_ratio'] = obs['nucleus_area'] / obs['cell_area'].clip(lower=1e-9)
else:
    obs['nucleus_ratio'] = np.nan

# Neighbour count inside DENSITY_RADIUS_UM. Written to its own key so it does
# not tread on the spatial_100um graph the filter cells build later.
if COMPUTE_NEIGHBOUR_DENSITY:
    sq.gr.spatial_neighbors(sample, coord_type='generic',
                            radius=DENSITY_RADIUS_UM, key_added='novae_qc_density')
    obs['n_neighbours_qc'] = np.asarray(
        sample.obsp['novae_qc_density_connectivities'].sum(axis=1)).ravel()
else:
    obs['n_neighbours_qc'] = np.nan

agg = {
    'n_cells':          ('total_counts', 'size'),
    'median_counts':    ('total_counts', 'median'),
    'median_genes':     ('n_genes_by_counts', 'median'),
    'pct_lt10_counts':  ('total_counts', lambda s: (s < 10).mean() * 100),
    'pct_lt10_genes':   ('n_genes_by_counts', lambda s: (s < 10).mean() * 100),
    'median_neg_frac':  ('neg_control_frac', 'median'),
    'median_neighbours': ('n_neighbours_qc', 'median'),
}
if 'cell_area' in obs.columns:
    agg['median_cell_area'] = ('cell_area', 'median')
if 'nucleus_ratio' in obs.columns:
    agg['median_nucleus_ratio'] = ('nucleus_ratio', 'median')
if 'nucleus_count' in obs.columns:
    agg['pct_no_nucleus'] = ('nucleus_count', lambda s: (s == 0).mean() * 100)

domain_qc = obs.groupby(DOMAIN_QC_KEY, observed=True).agg(**agg)
domain_qc['pct_of_cells'] = domain_qc['n_cells'] / domain_qc['n_cells'].sum() * 100

# Fraction of each domain that Novae itself was unhappy about.
if 'neighborhood_valid' in obs.columns:
    domain_qc['pct_sparse_nbhd'] = (
        obs.groupby(DOMAIN_QC_KEY, observed=True)['neighborhood_valid']
           .apply(lambda s: (~s.astype(bool)).mean() * 100))

# Spatial compactness. sd_xy is the mean spread of a domain's cells around its
# own centre, scaled by the spread of the whole section. Values near 1 mean the
# domain is sprinkled over the section (usually a cell type). Small values mean
# it sits in one patch (a real anatomical region, or an artefact).
span = np.hypot(obs['x_centroid'].std(), obs['y_centroid'].std())
compact = obs.groupby(DOMAIN_QC_KEY, observed=True).apply(
    lambda d: np.hypot(d['x_centroid'].std(), d['y_centroid'].std()) / span)
domain_qc['spatial_spread'] = compact

# Composition, when the labels are around. A domain that is nearly all neurons
# or nearly no neurons is worth a second look either way.
for flag in ['is_neuron', 'is_MN']:
    if flag in obs.columns:
        domain_qc[f'pct_{flag}'] = (
            obs.groupby(DOMAIN_QC_KEY, observed=True)[flag]
               .apply(lambda s: np.asarray(s).astype(bool).mean() * 100))

# ---- suspicion score -------------------------------------------------------
def _z(v):
    v = pd.to_numeric(v, errors='coerce')
    sd = v.std(ddof=0)
    return pd.Series(0.0, index=v.index) if not np.isfinite(sd) or sd == 0 else (v - v.mean()) / sd

terms = []
terms.append(-_z(domain_qc['median_counts']))        # few transcripts is bad
terms.append(-_z(domain_qc['median_genes']))         # low complexity is bad
terms.append(_z(domain_qc['pct_lt10_counts']))       # many failing cells is bad
if 'median_neg_frac' in domain_qc:
    terms.append(_z(domain_qc['median_neg_frac']))   # negative controls climbing is bad
if COMPUTE_NEIGHBOUR_DENSITY:
    terms.append(-_z(domain_qc['median_neighbours']))  # isolated cells are bad
if 'median_nucleus_ratio' in domain_qc:
    terms.append(-_z(domain_qc['median_nucleus_ratio']))  # nucleus-free cells are bad
terms.append(-_z(domain_qc['spatial_spread']))       # one tight blob is suspicious

domain_qc['suspicion_score'] = pd.concat(terms, axis=1).mean(axis=1)
domain_qc = domain_qc.sort_values('suspicion_score', ascending=False)

cols = ['n_cells', 'pct_of_cells', 'median_counts', 'median_genes',
        'pct_lt10_counts', 'median_neighbours', 'median_neg_frac',
        'spatial_spread', 'suspicion_score']
cols = [c for c in cols if c in domain_qc.columns]
print(f"=== Novae domain QC — SD03522_BG — {DOMAIN_QC_KEY} ===")
print(domain_qc[cols].round(3).to_string())

domain_qc.round(4).to_csv(FIGDIR / '(RK)SD03522BG_novae_domain_qc.csv')

# ---- how the metrics look side by side -------------------------------------
heat_cols = [c for c in ['median_counts', 'median_genes', 'pct_lt10_counts',
                         'median_neighbours', 'median_neg_frac',
                         'median_cell_area', 'median_nucleus_ratio',
                         'pct_no_nucleus', 'spatial_spread', 'pct_sparse_nbhd']
             if c in domain_qc.columns]
heat = domain_qc[heat_cols].apply(_z)

fig, axes = plt.subplots(1, 2, figsize=(6 + 0.9 * len(heat_cols), 0.45 * len(domain_qc) + 3),
                         dpi=200, gridspec_kw={'width_ratios': [3, 1]})

sns.heatmap(heat, cmap='RdBu_r', center=0, ax=axes[0],
            cbar_kws={'label': 'z across domains', 'shrink': 0.6},
            linewidths=0.4, linecolor='white')
axes[0].set_title(f'Novae domain QC metrics ({DOMAIN_QC_KEY})',
                  fontsize=10, fontweight='bold', pad=8)
axes[0].set_ylabel('')
axes[0].tick_params(labelsize=7)
plt.setp(axes[0].get_xticklabels(), rotation=40, ha='right')

axes[1].barh(range(len(domain_qc)), domain_qc['suspicion_score'].values,
             color=['#C53030' if v > 0.5 else '#94A3B8'
                    for v in domain_qc['suspicion_score']])
axes[1].set_yticks(range(len(domain_qc)))
axes[1].set_yticklabels(domain_qc.index, fontsize=7)
axes[1].invert_yaxis()
axes[1].axvline(0.5, color='#C53030', linestyle='--', linewidth=0.8)
axes[1].set_title('Suspicion score', fontsize=10, fontweight='bold', pad=8)
axes[1].tick_params(labelsize=7)
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig(FIGDIR / '(RK)SD03522BG_novae_domain_qc.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(FIGDIR / '(RK)SD03522BG_novae_domain_qc.pdf',
            bbox_inches='tight', facecolor='white')
plt.show()

flagged = domain_qc.index[domain_qc['suspicion_score'] > 0.5].tolist()
print(f"\\nDomains above the 0.5 line: {flagged}")
print("Check them on the maps above before doing anything with them. "
      "To drop one:  sample = sample[~sample.obs[DOMAIN_QC_KEY].isin(flagged)].copy()")

# How stable the labels are across granularities. A domain that survives intact
# from n4 through n10 is a solid structure. One that shatters is a soft call.
other = [k for k in DOMAIN_KEYS if k != DOMAIN_QC_KEY]
if other:
    ref = obs[DOMAIN_QC_KEY].astype(str)
    for k in other:
        ct = pd.crosstab(ref, obs[k].astype(str), normalize='index') * 100
        purity = ct.max(axis=1)
        print(f"\\n{DOMAIN_QC_KEY} rows split across {k} columns (% of row)")
        print(ct.round(1).to_string())
        print(f"largest single overlap per domain:\\n{purity.round(1).to_string()}")
'''


# --------------------------------------------------------------------------- #
def blank_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def retarget(text: str, sample: str, compact: str) -> str:
    # "# Sample SD035/22 BG" is the only place the id is written the human way.
    text = text.replace("SD035/22 BG", sample)
    text = text.replace(REF_SAMPLE, sample)
    text = text.replace(REF_COMPACT, compact)
    text = re.sub(r"\b115371\b", "adata.n_obs", text)
    return text


def main() -> None:
    nb = json.loads(SRC_NB.read_text())
    samples = sorted(p.name.split("__")[0]
                     for p in H5_DIR.glob("*__niches_independent.h5ad"))
    print(f"{len(samples)} samples: {samples}")

    # Build the template once: trim, comment, insert the Novae cells.
    cells = copy.deepcopy(nb["cells"][: KEEP_UPTO + 1])

    for idx, comment in CELL_COMMENTS.items():
        cell = cells[idx]
        if cell["cell_type"] != "code":
            continue
        header = "".join(f"# {line}\n" if line else "#\n"
                         for line in comment.split("\n"))
        cell["source"] = (header + "\n" + "".join(cell["source"])).splitlines(keepends=True)

    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    new_cells = [md_cell(NOVAE_MD),
                 blank_code_cell(NOVAE_PLOT_CELL),
                 blank_code_cell(NOVAE_STATS_CELL)]
    cells = cells[: INSERT_AFTER + 1] + new_cells + cells[INSERT_AFTER + 1:]

    # Section title at the top so it is obvious which sample a copy belongs to.
    for sample in samples:
        compact = sample.replace("_", "")
        out = copy.deepcopy(cells)
        for cell in out:
            cell["source"] = retarget("".join(cell["source"]), sample, compact) \
                .splitlines(keepends=True)

        out.insert(0, md_cell(
            f"# Dual-pass QC — {sample}\n\n"
            f"Generated from `(RK)QC_per_sample_dualpass.ipynb`, which was written "
            f"against {REF_SAMPLE}. Paths, figure names and titles now point at "
            f"{sample}.\n\n"
            f"Two things still need a human before this runs end to end: the smear "
            f"cluster ids in the *Removing smear/debris* section, and the domains "
            f"the Novae QC table flags. Both were chosen for {REF_SAMPLE} and mean "
            f"nothing here until you look at the maps.\n"))

        doc = {"cells": out,
               "metadata": copy.deepcopy(nb["metadata"]),
               "nbformat": nb["nbformat"],
               "nbformat_minor": nb["nbformat_minor"]}

        dest = NB_DIR / f"(RK)QC_per_sample_dualpass_{sample}.ipynb"
        dest.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
        print(f"wrote {dest.name}  ({dest.stat().st_size/1024:.0f} KB, {len(out)} cells)")


if __name__ == "__main__":
    main()
