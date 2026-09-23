#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/dual_pass/fix_notebooks_at_source.py
#
# The fix for the worst bug of the QC. The Novae per-sample h5ads ship X as log1p values
# and keep the integer counts in layers['counts']. The notebooks applied the >=10
# transcripts filter to X, so they thresholded log values and discarded about 98% of cells.
#
# This script edits the 20 notebooks in place and stamps the marker SOURCE FIX 2026-08-17.
# It restores X from layers['counts'] and asserts row sums equal transcript_counts. It pins
# one PASS1_H5AD and PASS2_H5AD pair, and it makes smear removal read the domain verdicts
# and raise when decisions.csv is missing. The runner refuses any notebook without the
# marker. Already applied. The notebooks in notebooks/ are the patched state.
# ========================================================================================

"""
Correct the 20 per-sample dual-pass QC notebooks IN PLACE, at source.

The 2026-08-16 audit found the notebooks apply the Salas ">= 10 transcripts per
cell" filter to data that is not counts: the Novae per-sample h5ads ship X
already normalised + log1p'd, sc.pp.calculate_qc_metrics overwrites
obs['total_counts'] with log-normalised row sums, and the filter thresholds
that column. The corrected numbers were produced by patching the notebooks in
memory at run time (run_dualpass.py --fix-counts). This script instead writes
the corrections into the .ipynb files, so the notebooks are correct on disk and
in Jupyter, and a plain re-run reproduces them.

Three corrections, all from the audit, none of them a change to a science knob:

  1. counts restoration     -- adata.X = adata.layers['counts'] immediately
                               after the load, with a fail-loud assert that the
                               restored row sums equal obs['transcript_counts'].
  2. one path per object     -- PASS1_H5AD / PASS2_H5AD variables (env
                               overridable). Pass 1 wrote one filename and pass
                               2 read another ('_fisrt_pass_filtering' typo), so
                               pass 2 could silently resume from a stale object.
  3. smear removal by verdict -- the hardcoded suspect = SMEAR_CLUSTERS =
                               ['4','5'] (copied from SD03522_BG into all 20;
                               Leiden ids are not stable across samples) is
                               replaced by the validated, human-reviewed
                               Novae-domain KEEP/REVIEW/REMOVE verdicts. The
                               Leiden-level table is still computed, as a ranked
                               diagnostic only -- it deletes nothing.

Deliberately NOT touched (flagged by the audit, but each is a judgement call for
Rodrigo, not a typo): MIN_TRANSCRIPTS = 10 itself, the >= 8 neighbours/100um
density filter, N_PCS 50 vs 20 between passes, min_genes drawn but never
applied, and the QV >= 20 question with Marcel.

usage:  fix_notebooks_at_source.py [--dry-run] [SAMPLE ...]
"""

import json
import re
import shutil
import sys
from pathlib import Path

NBDIR   = Path("/home/rodrigok/Notebooks/Spatial_MN_correct")
CODEDIR = "/home/rodrigok/SLURM_jobs/Spatial/dualpass_rerun"
OAK     = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
MARKER  = "SOURCE FIX 2026-08-17"
BAK_EXT = ".bak_precountsfix"

SAMPLES = ["SD00614_BG", "SD01015_BG", "SD01115_BG", "SD01320_BG", "SD01413_BA",
           "SD01616_BI", "SD01620_BI", "SD01623_BI", "SD01915_BG", "SD01922_BI",
           "SD01923_BI", "SD02022_BI", "SD02622_BI", "SD02818_BG", "SD02913_BA",
           "SD03522_BG", "SD03614_BG", "SD03914_BG", "SD04219_BI", "SD05413_BG"]


# --------------------------------------------------------------- cell bodies ---
# Placeholders are @@TOKEN@@ rather than str.format fields: these cells are full
# of f-strings, so brace escaping would make them unreadable.

MD_FIX = """### Correction 2026-08-17 — X is log-normalised, the counts are in a layer

`sc.read` returns an object whose `X` has **already** been normalised and
log1p-transformed by Novae (float32, max ~2.6). The genuine integer transcript
counts are in `layers['counts']`; their row sums equal `obs['transcript_counts']`.

The cell above only stashes `X` into that layer when it is *absent*, so the layer
is intact — but every QC cell below reads `X`. `sc.pp.calculate_qc_metrics`
overwrites `obs['total_counts']` with log-normalised row sums, and the Salas
`>= 10 transcripts` filter then thresholds that column. Applied to log row sums
the filter kept **8.4%** of the cohort (median sample 2.2%) and reduced
SD01915_BG, SD01922_BI and SD01923_BI to **zero cells**, which crashes the
spatial-neighbours step.

The tell was in the notebook's own output: a printed median of **8.2 transcripts**
per cell alongside a median of **11 detected genes**. Eleven genes cannot be
detected from 8.2 transcripts.

The next cell restores the notebook's own assumption before anything reads `X`.
"""

FIX_CELL = '''# ── @@MARKER@@ — restore raw counts before any QC ─────────────
# X arrives normalised + log1p'd; layers['counts'] holds the integer counts.
# Restore them here so calculate_qc_metrics writes true totals and the Salas
# >= 10 transcript cut thresholds transcripts rather than log row sums.
import os as _os
import numpy as _np

assert 'counts' in adata.layers, "layers['counts'] absent -- cannot restore raw counts"
_pre  = float(_np.asarray(adata.X.sum(1)).ravel().mean())
adata.X = adata.layers['counts'].copy()
_post = _np.asarray(adata.X.sum(1)).ravel()
assert _np.allclose(_post, adata.obs['transcript_counts'].values, atol=1e-3), \\
    "restored X row sums != obs['transcript_counts'] -- do not trust this run"
print(f"X row-sum mean {_pre:.2f} (log-normalised)  ->  {_post.mean():.2f} (raw counts)")
print(f"median transcripts/cell {adata.obs['transcript_counts'].median():.0f}; "
      f"{(adata.obs['transcript_counts'] < 10).mean()*100:.1f}% below the >= 10 cut")

# One path per object. Pass 1 wrote one filename and pass 2 read another (the
# '_fisrt_pass_filtering' typo), so pass 2 could resume from a stale object
# without erroring. Env-overridable so a staging run cannot clobber canonical
# outputs.
SAMPLE_ID  = "@@SAMPLE@@"
SAMPLE_TAG = "@@TAG@@"
PASS1_H5AD = _os.environ.get(
    "DUALPASS_PASS1_H5AD",
    "@@OAK@@/Novae_IND_QC/ALS_SCXenium_@@TAG@@_postQC.h5ad")
PASS2_H5AD = _os.environ.get(
    "DUALPASS_PASS2_H5AD",
    "@@OAK@@/Ranger_procd/ALS_SCXenium_@@TAG@@_pass2.h5ad")
print("pass 1 ->", PASS1_H5AD)
print("pass 2 ->", PASS2_H5AD)

try:
    __TRACE__
except NameError:
    __TRACE__ = {}
__TRACE__.update({
    "sample": SAMPLE_ID,
    "fix_counts_applied": True,
    "true_median_transcripts": float(adata.obs['transcript_counts'].median()),
    "true_mean_transcripts": float(adata.obs['transcript_counts'].mean()),
    "true_pct_lt10": float((adata.obs['transcript_counts'] < 10).mean() * 100),
    "pass1_h5ad": PASS1_H5AD,
    "pass2_h5ad": PASS2_H5AD,
})
'''

MD_GATE = """### Correction 2026-08-17 — smear removal by domain verdict, not by Leiden id

The two cells below shipped with `suspect = SMEAR_CLUSTERS = ['4', '5']`, copied
verbatim from SD03522_BG into all 20 notebooks by the generator. Leiden ids are
not stable across samples, so those ids mean nothing here — and even on their
home sample clusters 4 and 5 are not the lowest-count clusters (six others are)
yet removing them deletes 19.9% of cells.

Smear is a property of a spatial **territory**, not of a transcriptional cluster:
astrocytes and oligodendrocytes are dispersed across the whole section by
design, so any spatial-coherence test scores a real dispersed cell type as
"incoherent". Porting the two-gate rule down to Leiden clusters flagged 44 of 46
clusters in SD01015_BG for deletion.

The removal therefore uses the lab's validated, human-reviewed **Novae-domain**
verdicts (`dualpass_qc/extracted/tables/<SAMPLE>__decisions.csv`, joined on
`novae_domains_n6`; cohort KEEP 77 / REVIEW 41 / REMOVE 22). The Leiden-level
table is still computed below, as a **ranked diagnostic for review only** — it
deletes nothing.
"""

GATE_CELL = '''# ── @@MARKER@@ — smear removal by validated Novae-domain verdict ──
import sys as _sys
if "@@CODEDIR@@" not in _sys.path:
    _sys.path.insert(0, "@@CODEDIR@@")
import json as _json1, os as _os1
import pandas as _pd
import scanpy as _sc
import smear_gate

DOMAIN_KEY_SMEAR = 'novae_domains_n6'      # primary resolution for these verdicts
DECISIONS_CSV = ("@@OAK@@/Novae_persample_INDEPENDENT_FINAL/dualpass_qc/"
                 "extracted/tables/@@SAMPLE@@__decisions.csv")

try:
    __TRACE__
except NameError:
    __TRACE__ = {}

if _os1.path.exists(DECISIONS_CSV) and DOMAIN_KEY_SMEAR in sample.obs.columns:
    _dec = _pd.read_csv(DECISIONS_CSV)
    _verdict = dict(zip(_dec['domain'].astype(str), _dec['verdict'].astype(str)))
    REMOVE_DOMAINS = sorted([d for d, v in _verdict.items() if v == 'REMOVE'])
    REVIEW_DOMAINS = sorted([d for d, v in _verdict.items() if v == 'REVIEW'])
    _mask = sample.obs[DOMAIN_KEY_SMEAR].astype(str).isin(REMOVE_DOMAINS).values
    print(_dec[['domain', 'n_cells', 'pct_cells', 'verdict', 'smear_score',
                'dominant_celltype', 'reason']].to_string(index=False))
    print()
    print(f"domain REMOVE: {REMOVE_DOMAINS}  -> {_mask.sum():,} cells")
    print(f"domain REVIEW: {REVIEW_DOMAINS}  (retained)")
    _n0 = sample.n_obs
    sample = sample[~_mask].copy()
    print(f"after domain smear removal: {_n0:,} -> {sample.n_obs:,}")
    __TRACE__['domain_decisions'] = _json1.loads(_dec.to_json(orient='records'))
    __TRACE__['domain_remove'] = REMOVE_DOMAINS
    __TRACE__['domain_review'] = REVIEW_DOMAINS
    __TRACE__['n_removed_by_domain'] = int(_mask.sum())
    __TRACE__['n_before_domain_removal'] = int(_n0)
else:
    raise FileNotFoundError(
        f"no domain verdicts at {DECISIONS_CSV} (or {DOMAIN_KEY_SMEAR} missing "
        f"from obs) -- refusing to fall back to hardcoded Leiden ids")

# Leiden-level table: ranked DIAGNOSTIC only. Nothing below deletes on it.
_sc.tl.rank_genes_groups(sample, 'leiden_1.0', method='wilcoxon',
                         key_added='rgg_leiden10', use_raw=False)
SMEAR_TABLE = smear_gate.cluster_gate_table(sample, 'leiden_1.0', 'rgg_leiden10')
SMEAR_TABLE = SMEAR_TABLE.sort_values('smear_rank_score', ascending=False)
print()
print("Leiden-level smear CANDIDATES (diagnostic; review against the maps):")
print(SMEAR_TABLE[['n_cells', 'pct_of_total', 'median_counts', 'count_ratio',
                   'n_strong_markers', 'largest_cc_frac', 'pct_MN',
                   'smear_rank_score', 'flag']].head(10).to_string())

DATA_DRIVEN_SMEAR = []          # removal already happened, by domain verdict
REVIEW_CLUSTERS = SMEAR_TABLE.index[SMEAR_TABLE.flag == 'CANDIDATE'].tolist()

_legacy = [c for c in ['4', '5'] if c in SMEAR_TABLE.index]
_legacy_n = int(SMEAR_TABLE.loc[_legacy, 'n_cells'].sum()) if _legacy else 0
print(f"\\nlegacy ['4','5'] present={_legacy} -> would have dropped {_legacy_n:,} cells")

__TRACE__['smear_table'] = _json1.loads(SMEAR_TABLE.to_json(orient='records'))
__TRACE__['leiden_candidates'] = REVIEW_CLUSTERS
__TRACE__['legacy_clusters'] = _legacy
__TRACE__['legacy_would_drop_n'] = _legacy_n
'''

TRACE_CELL = '''# ── @@MARKER@@ — machine-readable QC decision trace ───────────
# Print-only traces cannot be tested against disease group or spinal level.
import json as _json2, os as _os3
import numpy as _np2

def _f(x):
    try:
        return float(x)
    except Exception:
        return None

try:
    __TRACE__
except NameError:
    __TRACE__ = {}

__TRACE__['raw_cells'] = int(adata.n_obs)
__TRACE__['n_genes_panel'] = int(adata.n_vars)
__TRACE__['final_cells'] = int(sample.n_obs)
__TRACE__['raw_median_counts'] = _f(adata.obs['total_counts'].median())
__TRACE__['raw_median_genes'] = _f(adata.obs['n_genes_by_counts'].median())
__TRACE__['final_median_counts'] = _f(sample.obs['total_counts'].median())
__TRACE__['final_median_genes'] = _f(sample.obs['n_genes_by_counts'].median())
__TRACE__['pct_lt10_tx_raw'] = _f((adata.obs['total_counts'] < 10).mean() * 100)
__TRACE__['pct_lt10_genes_raw'] = _f((adata.obs['n_genes_by_counts'] < 10).mean() * 100)
__TRACE__['neg_frac_median'] = _f(globals().get('neg_frac_median'))
__TRACE__['snr_aggregate'] = _f(globals().get('snr_aggregate'))
__TRACE__['n_after_transcript'] = int(globals().get('n_after_transcript', -1))
__TRACE__['n_after_spatial'] = int(globals().get('n_after_spatial', -1))
if 'cell_type' in sample.obs.columns:
    __TRACE__['cell_type_counts'] = {str(k): int(v) for k, v in
                                     sample.obs['cell_type'].value_counts().items()}
if 'is_MN' in sample.obs.columns:
    __TRACE__['n_MN_final'] = int(__import__('pandas').Series(sample.obs['is_MN']).astype('boolean').fillna(False).sum())
if 'is_MN' in adata.obs.columns:
    __TRACE__['n_MN_raw'] = int(__import__('pandas').Series(adata.obs['is_MN']).astype('boolean').fillna(False).sum())
for _k in ['MIN_TRANSCRIPTS', 'MIN_NEIGHBOURS', 'N_PCS', 'ELBOW_PC', 'DOMAIN_QC_KEY']:
    if _k in globals():
        __TRACE__[_k] = globals()[_k]
if 'domain_qc' in globals():
    __TRACE__['n_domains_scored'] = int(domain_qc.shape[0])
if 'flagged' in globals():
    __TRACE__['flagged_domains'] = [str(_x) for _x in flagged]

_TRACE_DIR = _os3.environ.get("DUALPASS_TRACE_DIR",
                             "@@OAK@@/Novae_IND_QC/runs_srcfixed")
_os3.makedirs(_TRACE_DIR, exist_ok=True)
_TRACE_PATH = _os3.path.join(_TRACE_DIR, "trace_src_@@SAMPLE@@.json")
with open(_TRACE_PATH, "w") as _fh:
    _json2.dump(__TRACE__, _fh, indent=1, default=str)
print("TRACE WRITTEN ->", _TRACE_PATH)
'''


def render(body, sample):
    return (body.replace("@@MARKER@@", MARKER)
                .replace("@@SAMPLE@@", sample)
                .replace("@@TAG@@", sample.replace("_", ""))
                .replace("@@CODEDIR@@", CODEDIR)
                .replace("@@OAK@@", OAK))


def as_source(text):
    return text.splitlines(keepends=True)


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": as_source(text)}


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": as_source(text)}


def find_nb(sample):
    for name in (f"(RK)QC_per_sample_dualpass_{sample}.ipynb",
                 f"(RK)QC_per_sample_dualpass_{sample}-Copy1.ipynb"):
        p = NBDIR / name
        if p.exists():
            return p
    raise SystemExit(f"no notebook found for {sample}")


def patch(path, sample, dry_run=False):
    nb = json.loads(path.read_text())
    cells = nb["cells"]
    rep = {"nb": path.name, "n_cells_before": len(cells)}

    if any(MARKER in "".join(c.get("source", [])) for c in cells):
        rep["skipped"] = "already patched"
        return rep

    # 1 ── FIGDIR made env-overridable so a staging run keeps the current figures
    n_figdir = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        m = re.search(r"^FIGDIR\s*=\s*Path\((['\"])(.+?)\1\)\s*$", src, re.M)
        if m:
            new = src.replace(
                m.group(0),
                "import os as _os_fig\n"
                f"FIGDIR = Path(_os_fig.environ.get('DUALPASS_FIGDIR', '{m.group(2)}'))")
            c["source"] = as_source(new)
            n_figdir += 1
    rep["figdir_parameterised"] = n_figdir

    # 2 ── counts restoration, immediately after the load cell
    i_load = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and re.search(r"^\s*adata\s*=\s*sc\.read\(",
                                                  "".join(c["source"]), re.M):
            i_load = i
            break
    if i_load is None:
        raise RuntimeError(f"{path.name}: no 'adata = sc.read(' cell")
    cells.insert(i_load + 1, md_cell(render(MD_FIX, sample)))
    cells.insert(i_load + 2, code_cell(render(FIX_CELL, sample)))
    rep["fix_cell_at"] = i_load + 2

    # 3 ── one path per object
    pass1 = re.compile(r"(['\"])[^'\"]*Novae_IND_QC/[^'\"]*\.h5ad\1")
    pass2 = re.compile(r"(['\"])[^'\"]*Ranger_procd/[^'\"]*_pass2\.h5ad\1")
    n1 = n2 = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if MARKER in src:
            continue
        src, k1 = pass1.subn("PASS1_H5AD", src)
        src, k2 = pass2.subn("PASS2_H5AD", src)
        if k1 or k2:
            c["source"] = as_source(src)
        n1 += k1
        n2 += k2
    rep["pass1_paths"] = n1
    rep["pass2_paths"] = n2
    if n1 < 2:
        raise RuntimeError(f"{path.name}: expected a pass-1 write and read, found {n1}")
    if n2 != 1:
        raise RuntimeError(f"{path.name}: expected exactly 1 pass-2 write, found {n2}")

    # 4 ── smear: domain verdicts replace the copied Leiden ids
    i_suspect = i_smear = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if i_suspect is None and re.search(r"^\s*suspect\s*=\s*\[", src, re.M):
            i_suspect = i
        if i_smear is None and re.search(r"^\s*SMEAR_CLUSTERS\s*=\s*\[", src, re.M):
            i_smear = i
    if i_suspect is None or i_smear is None:
        raise RuntimeError(f"{path.name}: smear cells not found "
                           f"(suspect={i_suspect}, smear={i_smear})")

    # the plotting below does subplots(1, len(suspect)+1); an empty list returns a
    # bare Axes and the cell dies on axes[0], so always keep two ranked clusters.
    src = "".join(cells[i_suspect]["source"])
    cells[i_suspect]["source"] = as_source(re.sub(
        r"^\s*suspect\s*=\s*\[[^\]]*\]",
        "suspect = (DATA_DRIVEN_SMEAR + REVIEW_CLUSTERS) or "
        "SMEAR_TABLE.index[:2].tolist()   # ranked candidates, not hardcoded ids",
        src, count=1, flags=re.M))

    src = "".join(cells[i_smear]["source"])
    cells[i_smear]["source"] = as_source(re.sub(
        r"^\s*SMEAR_CLUSTERS\s*=\s*\[[^\]]*\]",
        "SMEAR_CLUSTERS = DATA_DRIVEN_SMEAR   # empty: removal is by domain verdict",
        src, count=1, flags=re.M))

    cells.insert(i_suspect, md_cell(render(MD_GATE, sample)))
    cells.insert(i_suspect + 1, code_cell(render(GATE_CELL, sample)))
    rep["gate_cell_at"] = i_suspect + 1

    # 5 ── trace
    cells.append(md_cell("### QC decision trace\n\nWritten to JSON so QC severity "
                         "can be tested against disease group and spinal level.\n"))
    cells.append(code_cell(render(TRACE_CELL, sample)))
    rep["n_cells_after"] = len(cells)

    if dry_run:
        rep["dry_run"] = True
        return rep

    bak = path.with_suffix(path.suffix + BAK_EXT)
    if not bak.exists():
        shutil.copy2(path, bak)
        rep["backup"] = bak.name
    else:
        rep["backup"] = f"{bak.name} (kept)"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return rep


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv[1:]
    todo = args or SAMPLES
    ok = 0
    for s in todo:
        rep = patch(find_nb(s), s, dry_run=dry)
        print(f"{s:12s} {rep}", flush=True)
        ok += 0 if "skipped" in rep else 1
    print(f"\n{ok}/{len(todo)} patched{' (dry run)' if dry else ''}")


if __name__ == "__main__":
    main()
