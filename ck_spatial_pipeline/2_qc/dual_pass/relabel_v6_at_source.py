#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/dual_pass/relabel_v6_at_source.py
#
# Injects Marcel's v6 neuron call into the 20 notebooks and stamps VH RELABEL V6 2026-08-18.
# The pinned snapshot is neuron_classification_tdp_v6__snapshot_20260818_1649.parquet. The
# md5 in the notebooks (d32b298e...) matches the file on OAK. The md5 in this docstring
# (e4f7ffc0...) is stale. Trust the notebook value.
#
# The join is by coordinate. It supersedes relabel_v4_at_source.py, which read the v2
# snapshot despite its name. The script was edited again on 08-19, after the notebooks were
# patched at 08-18 11:25, so the 08-19 edit never ran.
# ========================================================================================

"""Relabel the 20 per-sample dual-pass QC notebooks against Marcel's REAL v4 call file.

This supersedes relabel_v4_at_source.py, which despite its name read the **v2**
snapshot (neuron_classification_tdp_v2__snapshot_20260818_0851.parquet). Every
number produced by that run -- the 20 executed notebooks, the promoted pass-2
objects, cohort_v4/ALS_SCXenium_cohort_v4.h5ad, the scVI latent and the
cellannotator labels -- is v2 and is superseded.

Source of truth here:

    Novae_IND_QC/runs_srcfixed/annotation_snapshot/
        neuron_classification_tdp_v6__snapshot_20260818_1649.parquet
    md5 e4f7ffc0fb8ebe794c4f0c7b1bce4095

read through the pinned snapshot, never through the live file. The live
Ranger_procd_mw_final/annotation/neuron_classification_tdp_v6.parquet has been
rewritten repeatedly (last at 2026-08-18 06:28 UTC); reading it directly is how
you get numbers nobody can reproduce. The injected cell asserts the md5, so a
mid-run rewrite of the snapshot cannot change labels underneath a running array
job. Snapshot stamp rule: the SOURCE file's mtime rendered in UTC.

RESTORE FIRST. Each notebook on disk currently carries the v2 relabel (marker
"VH RELABEL V4 2026-08-18"). This script does NOT patch over it. It restores each
notebook from <nb>.bak_prev4relabel -- the pre-v2-relabel state, which still
carries the required "SOURCE FIX 2026-08-17" counts fix -- verifies the restored
text carries that prerequisite and carries NO relabel marker of any generation,
and only then injects. The current (v2-relabelled) file is preserved first under
the new suffix .bak_prev2relabel_20260818 so nothing is lost.

Restoring is not tidiness, it is correctness. obs['is_neuron_general'] is derived
from whatever obs['is_MN'] holds when the cell runs. On a pristine notebook that
is our native markers+size call (2,577 over the 20, per-sample in
EXPECTED_NATIVE_MN below). Patching on top of the v2 relabel would silently
redefine is_neuron_general as the v2 motor-neuron call.

WHAT CHANGED IN THE REAL v4, and what each change forced here
-------------------------------------------------------------

  Column renames. is_neuron -> is_neuron_v3, neuron_prob -> neuron_prob_v3.
  New columns: cell_type, pathologist_class, vh_source, STMN2, CE_STMN2.

  The recipe kept its shape but moved onto the refit neuron call:

      is_MN == is_neuron_v3 & ~is_GAD & in_VH & (cell_area_um2 >= 250)

  0 false positives and 0 false negatives on every one of the 22 sample keys
  individually, not merely cohort-wide. cell_type == 'motor_neuron' is identical
  to is_MN on all 22 keys, so cell_type carries no new information. Note the
  file only constrains the area gate to the window (249.0152, 251.5203] -- the
  largest rejected candidate is 249.0152 um2 and the smallest called MN is
  251.5203 um2. AREA_MIN = 250.0 sits inside that window and is the published
  number, so it is consistent-with the file rather than recovered uniquely
  from it. 252.0 gives 1 false positive, 249.0 gives 1 false negative.

  in_GM is NO LONGER a superset of in_VH: 7,763 cells cohort-wide (6,902 on our
  20, spread over 12 of them) are in_VH & ~in_GM. The v2 cell asserted the
  subset relation, so that assert would have killed 12 of 20 notebooks -- and
  because run_dualpass_src.py sets allow_errors=True it would not even have
  aborted, it would have written a half-populated contract that promote_pass2.py
  then refuses for all 20. The assert is replaced by a counted, traced,
  per-cell-flagged report. Every violation is in a robin_pathologist sample;
  all 8 auto_density_fixed samples have zero. 35 of the 738 called MN sit
  outside grey matter, which is anatomically impossible, so SD02913_BA (1,136
  violations = 69.8% of its in_VH) and SD03522_BG (2,028 = 35.2%) are flagged
  for mask-registration review before any VH-restricted figure is drawn.

  16 EXACT-DUPLICATE (sample, cell_index) PAIRS. v4 is not a 16-row superset of
  the v2 universe; deduplicating gives 2,063,633 rows, bit-exact the v2 and
  neuron_predictions.parquet count. It is the same Jul-22 universe with 16 rows
  duplicated, all 32 rows byte-identical across all 23 columns, all carrying
  vh_source='robin_pathologist' and a non-null pathologist_class -- so the
  duplication came from the pathologist-class merge, not from new segmentation.
  set_index("cell_index").reindex(...) raises ValueError on a duplicate index,
  so 4 of our 20 notebooks (SD01623_BI, SD01923_BI, SD03522_BG, SD04219_BI)
  would have died at the join. This cell asserts byte-identity first, then
  drops, then asserts the dropped count equals EXPECTED_DUPS.

  Consequently every headline count in the fact base is inflated. On our 20:
  is_MN 601 -> 599 unique cells, is_neuron_v3 6,037 -> 6,035, pathologist_class
  143 -> 130. All of the loss is in SD04219_BI (is_MN 38 -> 36, is_neuron_v3
  289 -> 287, n 99,671 -> 99,665). Publish 599, not 601.

  is_GAD and is_Interneuron are NO LONGER neuron subsets. 455 of 474 GAD+ and
  1,065 of 1,237 Interneuron cells on our 20 are ~is_neuron_v3. So the v2
  docstring line calling is_GAD "a neuron-only flag in this file" is now false.
  is_gabaergic is renamed is_gabaergic_neuron so the restriction it applies is
  visible in the column name, and both the unrestricted and the neuron-restricted
  counts go to the trace.

  NULLS. The fact base's claim that v4 is null-free apart from pathologist_class
  is wrong. Three columns carry nulls: pathologist_class 2,063,466,
  cord_level 299,367 (100% null on 4 whole samples, so the "unassigned" fallback
  in sample_meta is load-bearing, not belt-and-braces) and tdp_proba 281,859
  (all ALS, 24.4% of the ALS arm, 0 of Control). All seven boolean columns have
  zero nulls and dtype bool. The cell asserts zero nulls on the label columns
  and merely records the three that legitimately carry them.

THE JOIN GATE IS STILL THE COORDINATES
--------------------------------------
obs_names are "<sample>:<cells.parquet cell_id>" and cell_index == cell_id - 1.
We match obsm['spatial'] against the parquet x_centroid/y_centroid on that index
and require max |delta| < 1e-3 um. Re-measured on the v4 snapshot across all 20
samples: max |delta| = 1.82e-12 um (one float64 ULP), joined == n_obs on
1,694,362 of 1,694,362 cells, and the full 20x22 sweep gives a unique correct
argmin every time -- zero of the 420 wrong-key pairings put a single cell inside
1e-3 um, the closest wrong answer anywhere being 2,670.8 um median.

This is the only gate that can discriminate the two decoy pairs, and a row count
demonstrably cannot: our SD02022_BI raw 76,564 is closer to decoy SD02022 79,637
than to the true SD020_22_BI 80,618. Measured: SD02022_BI -> SD020_22_BI at
0.000 median / 1.8e-12 max, versus decoy SD02022 at 4,081.6 median / 11,365.4
max with 0.0% of cells inside tolerance. SD01620_BI -> SD016_20_BI likewise
(decoy 4,262.8 / 11,421.8). The decoys are different physical sections.

The -1 in cell_index was tested against both neighbours: +0 and -2 both give an
agreement fraction of exactly 0.000 on all 20 samples, so the coordinate
agreement is not an artefact of a permissive index map.

In these per-sample files obsm['spatial'], obsm['spatial_orig'] and
obsm['spatial_grid_offset'] are byte-identical to each other and to
obs.x_centroid/y_centroid (max abs diff exactly 0.0 on all 20), and
uns['spatial_note'] records that the grid offset is applied only in the
combined object. There is no 2e6 offset to strip here. Do NOT reuse this gate
unchanged against ALS_SCXenium_*_concatenated_offset_allsamples_preQC_*.h5ad.

Never build a join that matches on coordinates alone: 38,145 v4 rows (1.85%)
share an exact (x,y) with another cell in the same sample, the largest group
being 3,439 cells at (0,0) in SD01320_BG. Index-join, then verify with
coordinates.

THE TARGET BUILD IS _FINAL, AND THE WRONG CHOICE WOULD HAVE BEEN SILENT
----------------------------------------------------------------------
Novae_persample_INDEPENDENT_FINAL/per_sample (Jul-22) index-joins v4 at
100.0000% on all 20 with 599/599 motor neurons on the correct cell.
SC_MNcorrected_AUG5_h5ads joins 88.581%-98.649% (111,365 of 1,854,646 cells
wrong, 6.00%) and places only 207 of 599 motor neurons correctly -- yet all 599
is_MN cell_index values DO exist in the AUG5 objects, so an index join finds a
row for every motor neuron and nothing raises. Relabelling AUG5 would have
written 599 MN labels of which 392 sat on the wrong cell, with no error. Hence
assert_target_is_final() below and the hard coordinate assert. Do not downgrade
either to a warning.

ZERO-VH SAMPLES: A MISSING MASK, NOT A MEASUREMENT
--------------------------------------------------
SD01015_BG and SD01616_BI have in_GM True for 0 cells AND in_VH True for 0 cells
(0/51,912 and 0/80,312). There is no geometry to land off-tissue, because the
grey-matter mask is absent too. Every other auto_density_fixed sample has a real
mask on real tissue (SD00614_BG_ 32,118 in_GM / 12,883 in_VH with 5,305/5,305 VH
bins on occupied tissue). Under v2 these two were honestly <NA> on all 132,224
rows; v4 wrote every boolean as non-nullable bool, which mechanically converted
unknown to False, so is_MN reads "0 motor neurons" as if it were a count.

209 cells (71 in SD01015_BG, a Control; 138 in SD01616_BI, an ALS) satisfy
is_neuron_v3 & ~is_GAD & area >= 250 and would be MN candidates if a mask
existed. At the 599/2,393 = 0.2504 is_MN-per-eligible rate measured on the 18
masked cohort samples that is roughly 52 hidden motor neurons, about 8% of the
true pool, split across both arms.

vh_source is a PLAN tag, not an outcome. It appears in exactly one file on the
whole cluster (the v4 parquet), is documented nowhere, and is a per-sample
constant. Six of the eight auto_density_fixed samples have a real mask and two
have none, so 'auto_density_fixed' on those two is not evidence that an auto
path ran and succeeded. Mask presence is therefore DERIVED from the in_GM and
in_VH counts in the sample's own parquet slice, never from vh_source and never
from in_VH.isna(), which is now uniformly False.

Because pandas .sum() skips NA, an all-False column with 209 NA cells still sums
to 0. The only reduction-proof contract is is_MN = <NA> for EVERY cell in a
mask-absent section: then .mean() is NaN and .sum(min_count=1) is NaN. The cost
is that the genuine "definitely not a motor neuron" status of the other 51,703 /
80,174 cells moves to is_MN_eligible, which is where it belongs. is_MN_eligible
is written on every sample so an MN rate always has a denominator even where the
numerator is undetermined.

This is the one place where <NA> is deliberately introduced rather than carried:
v4 has no nulls in the label columns, and on the 18 masked samples this cell
writes none. The nullable dtypes and the .astype("boolean").fillna(False)
discipline throughout are kept anyway -- pd.NA is truthy, `== True` keeps <NA>,
and a future v5 could reintroduce nulls anywhere.

COLUMN CONTRACT WRITTEN TO obs
------------------------------
  is_neuron_v3        bool      v4 neuron call, 6,035 whole-section on our 20
  neuron_prob_v3      float32   v4 neuron probability
  is_neuron_prev      bool      whatever obs['is_neuron'] held before this cell
                                (the stale pre-existing column, 10,085 over the
                                20), kept so the refit can be diffed
  is_neuron_general   bool      our native markers+size call, preserved
                                unchanged; asserted against EXPECTED_NATIVE_MN
  in_GM, in_VH        boolean   verbatim v4; all-<NA> on a mask-absent section
  is_GAD              bool      verbatim v4. NOT neuron-restricted any more
  is_Interneuron      bool      verbatim v4, NOT in the MN rule, NOT restricted
  is_MN_v6_asdelivered bool     the raw parquet is_MN, unmodified, so the
                                delivery stays reproducible and the recipe gate
                                has something honest to test
  is_MN               boolean   THE CONTRACT: is_MN_v6_asdelivered, forced to
                                all-<NA> on a mask-absent section. UNGATED since
                                2026-08-19: 846 True pre-QC over our 20 sections,
                                774 surviving QC in the cohort. The 599/595 figures
                                in older notes were the area-250 view, now is_MN_a250
  is_MN_eligible      bool      is_neuron_v3 & ~is_GAD & area >= 250, NO in_VH
                                term. Written on every sample. The denominator
  is_MN_invh          boolean   is_neuron_v3 & in_VH -- GAD+ and small cells
                                kept, the set the size cut is taken from
  is_GADvh            boolean   is_GAD & in_VH
  is_gabaergic_neuron bool      is_neuron_v3 & is_GAD (renamed from
                                is_gabaergic: in v4 that restriction drops 455
                                of 474 GAD+ cells and the name must say so)
  in_VH_not_GM        bool      the mask-geometry violation, per cell
  vh_mask_empty       bool      constant within a sample; True <=> no mask
  has_vh_data         bool      kept for the existing column contract, but
                                REDEFINED as the measured flag (n_in_VH > 0 or
                                n_in_GM > 0). It used to be in_VH.notna(), which
                                under v4 is True for 20/20 and destroys the
                                sentinel
  cell_area_um2       float32   carried so the MN rule stays auditable in-object
  mn_signature_prob   float32   carried, deliberately NOT in the call, see below
  is_TDP              bool      carried. CONFOUNDED WITH DISEASE BY CONSTRUCTION
  tdp_proba           float32   carried; NaN on 281,859 ALS cells
  tdp_evaluated       bool      tdp_proba.notna(). is_TDP=False conflates
                                "TDP-negative" with "TDP-not-evaluated": 281,859
                                ALS cells have tdp_proba NaN and is_TDP False.
                                Any is_TDP rate must report this denominator
  cell_type_mw        category  v4 cell_type, RENAMED. The notebook assigns its
                                own obs['cell_type'] from marker-gene argmax
                                further down, and (RK)ALS_SCXenium_concat_
                                astrocytes_WDR49.ipynb selects
                                cell_type == 'Astrocytes' -- if v4's cell_type
                                won that name, that selection returns 0 cells.
                                v4's label set is also a misnomer: 2,056,939 of
                                2,063,649 cells are 'glia', which is simply
                                every non-neuron including endothelium,
                                pericytes, T cells and meninges
  pathologist_class_mw category  183 hand labels; "unlabelled" for the rest, NOT
                                <NA>, so a code/categories reader cannot hit a
                                -1 code
  vh_source           category  verbatim. A PLAN TAG, NOT AN OUTCOME
  STMN2_mw            int32     verbatim v4 counts, RENAMED: an obs column
  CE_STMN2_mw         int32     named STMN2 would shadow the panel gene of the
                                same name in every scanpy colour lookup
  neuron_class        category  one mutually exclusive label per cell

mn_signature_prob is carried but deliberately unused. Against the pathologist
ground truth its AUC is 0.5095, versus 1.0000 for neuron_prob_v3 and 0.9420 for
raw cell_area_um2 (class means 709.7 um2 neuronal vs 69.2 glial). The v4 MN call
is a size gate, not a signature gate.

uns
---
  sample_meta_v4      cord_level / disease_group / patient_id / vh_source, plain
                      strings, "unassigned" for missing -- never None, which
                      makes an h5ad unreadable in every anndata version
  vh_status           "present" or "absent_no_gm_no_vh"
  mn_call_valid       bool
  mn_call_note        plain string; on a mask-absent section it states that
                      is_MN=0 is a missing mask, not a count, and how many
                      eligible cells are undetermined
  vh_mask_metrics_v4  n_in_GM / n_in_VH / n_in_VH_not_GM / n_mn_eligible /
                      n_mn_undetermined / n_MN_outside_GM
  pathologist_scores_v4  raw tp/fn/fp/tn against is_neuron_v3 for this section
                      plus the in-sample caveat, see below
  is_TDP_caveat, pathologist_class_caveat, in_VH_not_GM_note, cell_type_mw_note
  v6_parquet, v6_parquet_md5, relabel_version

pathologist_class is the only ground truth in the file and it does not support a
generalisation claim. is_neuron_v3 scores 183/183 on it (sens 77/77, spec
106/106, neuron_prob_v3 AUC exactly 1.0000) while v2's is_neuron scores
66/11/3/103; all 14 disagreements resolve in v4's favour and 0 against, which is
the signature of an in-sample fit, and training-set membership could not be
verified from any file on the cluster. The labels are 0.00887% of the file with a
'neuronal' prevalence 129-fold above the file's, so a perfect specificity here
still only bounds the true specificity at >= 0.9658, which at cohort scale
permits on the order of 70,000 false-positive neurons. Separately, 40 of the 183
labels (including 35 of the 77 neuronal ones) sit on decoy key SD02022 and do
not transfer to our SD020_22_BI -- nearest counterpart 1.92 um minimum, 19.76 um
median -- so only 143 are usable for our 20, and 130 after dedupe. This cell
therefore reports raw per-sample tp/fn/fp/tn only, never a percentage, and never
writes "sensitivity 100%" anywhere a figure or a markdown cell could pick it up.
Because the parquet slice is taken by PARQUET_KEY, the 40 decoy labels can never
enter a notebook.

WHAT ELSE THIS SCRIPT TOUCHES (the consumer sweep)
--------------------------------------------------
Repointing the parquet is not enough; the v2 obs contract has consumers that
either hard-fail or, worse, keep working on stale data:

  notebook cell "Novae domains into a QC table": `for flag in ['is_neuron',
    'is_MN']` would build domain_qc['pct_is_neuron'] from the STALE
    obs['is_neuron'] (9,290 post-filter) rather than v4's (4,415) -- a 110%
    overstatement with no error. Repointed to is_neuron_v3, and this cell also
    DROPS the stale is_neuron / neuron_prob columns outright so no
    string-'is_neuron' consumer anywhere can silently read them.
  notebook QC decision trace: median(pct_is_MN) across Novae domains collapses
    to exactly 0.0 in 16 of 20 samples under v4, so "above-median MN content"
    fires for any domain holding a single motor neuron. Given a non-degenerate
    baseline and reworded.
  promote_pass2.py / concat_pass2_v4.py: REQUIRED and MARCEL_COLS updated to
    this contract; RELABEL_VERSION bumped so the 20 stale v2 traces are refused
    rather than promoted; concat asserts the stale is_neuron is absent.
  vh_visual_inspect.py: hard-fails on v4 (asks read_parquet for 'is_neuron').
    Repointed, and its has_vh sentinel moved off .notna() onto a measured count.
  plots_v4.py: per-sample panel title rendered "MN 0" for the two mask-absent
    sections. Now renders "no VH mask".
  cellannotator_v4.py: silently SHRANK its protection set when a contract column
    disappeared. Now fails on a missing marcel_column.
  fix_notebooks_at_source.py / run_dualpass.py: the notebook GENERATORS still
    emit int(np.asarray(...['is_MN']).astype(bool).sum()), which counts every
    <NA> as a motor neuron. Fixed at source so regenerating cannot reintroduce
    it.
  check_gad.py: throwaway diagnostic, hard-fails on the renames. Repointed.
  smear_gate.py: already NA-safe from the v2 run; left alone. Its
    PROTECT_MN_MULT logic was A/B tested with v2 vs v4 is_MN on all 20 promoted
    pass-2 objects -- 0 flag changes, 0 clusters losing protection, 0 cells
    newly flagged, including SD01915_BG where is_MN drops 337 -> 3. Nothing is
    removed on that table anyway (SMEAR_CLUSTERS = []); real removal reads
    decisions.csv files frozen at Jul 25. v4 removes zero extra cells. But
    dpqc_compute_FINAL.py:94 has its own PROTECT_MN_MULT at the Novae-DOMAIN
    level where removal IS real, so decisions.csv must not be regenerated
    against v4 is_MN without re-reviewing every REMOVE verdict.

usage: relabel_v4real_at_source.py [--dry-run] [--self-test] [SAMPLE ...]
       --self-test renders and statically checks the injected cells and exits
       without reading or writing a single notebook.
"""

import ast
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

NBDIR = Path("/home/rodrigok/Notebooks/Spatial_MN_correct")
CODEDIR = Path("/home/rodrigok/SLURM_jobs/Spatial/dualpass_rerun")

# The PINNED snapshot. Never the live file: Marcel rewrote the annotation six
# times on 2026-08-18 alone. Stamp = source mtime in UTC (16:49:58 -> 1649).
#
# v6, NOT v4. Migrated 2026-08-18 after Marcel shipped v5 then v6. What changed,
# all verified on SCG by probe_v5.py / probe_v6.py / probe_gaps.py:
#   v4 -> v5: 16 duplicate (sample, cell_index) rows FIXED (so EXPECTED_DUPS is
#     now empty); new bool column `vh_annotated`; `in_VH` and `is_MN` became
#     NULLABLE, NA on the two sections with no ventral-horn mask (is_MN NA only
#     on their 525 confirmed neurons, glia stay False); `cell_type` gained a 4th
#     level `neuron_MN_unknown`; `tdp_proba` became NaN for controls.
#   v5 -> v6: `is_MN` is UNGATED. The area term is gone from Marcel's rule, so
#     the recipe gate below asserts `is_neuron_v3 & ~is_GAD & in_VH` with NO area
#     term. Only `is_MN` and `cell_type` differ from v5; verified column by
#     column, row order identical on (sample, cell_index).
# Consequence for us: Marcel's `is_MN` now means "in-VH non-GAD neuron", 1,015
# cohort-wide (846 over our 20). It is carried as `is_MN_candidate`, and our
# `is_MN` stays the area-250 definition (599 over our 20) so nothing downstream
# silently changes meaning. `v6 is_MN & area >= 250` reproduces v5's `is_MN`
# exactly, verified, which is what makes that safe.
PARQUET = ("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_IND_QC/"
           "runs_srcfixed/annotation_snapshot/"
           "neuron_classification_tdp_v6__snapshot_20260818_1649.parquet")
PARQUET_MD5 = "d32b298e9662e023cc1794367680a0ce"

MARKER = "VH RELABEL V6 2026-08-18"
RELABEL_VERSION = "v6"
PREREQ_MARKER = "SOURCE FIX 2026-08-17"

# markers of every earlier relabel generation. If the file we are about to inject
# into carries any of them, something restored wrong and we stop.
STALE_MARKERS = ("VH RELABEL 2026-08-17", "VH RELABEL V4 2026-08-18",
                 "VH RELABEL V4REAL 2026-08-18")

RESTORE_FROM = ".bak_prev4relabel"          # pre-v2-relabel, carries PREREQ_MARKER
BAK = ".bak_prev2relabel_20260818"          # fresh backup of the v2-relabelled state

AREA_MIN = 250.0                            # um^2; see the area-window note above

SAMPLES = ["SD00614_BG", "SD01015_BG", "SD01115_BG", "SD01320_BG", "SD01413_BA",
           "SD01616_BI", "SD01620_BI", "SD01623_BI", "SD01915_BG", "SD01922_BI",
           "SD01923_BI", "SD02022_BI", "SD02622_BI", "SD02818_BG", "SD02913_BA",
           "SD03522_BG", "SD03614_BG", "SD03914_BG", "SD04219_BI", "SD05413_BG"]

# our label -> the key this sample has in the parquet. Re-verified on the v4
# snapshot 2026-08-18 by spatial-coordinate agreement over all 20x22 pairings:
# every one of these resolves at max |delta| 1.8e-12 um and every alternative at
# >= 2,670 um median. SD01620_BI and SD02022 also exist in the file as the
# SUPERSEDED originals; our cohort keeps the redos.
PARQUET_KEY = {
    "SD00614_BG": "SD00614_BG_",
    "SD02818_BG": "SD02818_BG_",
    "SD05413_BG": "Region_2",
    "SD01620_BI": "SD016_20_BI",
    "SD02022_BI": "SD020_22_BI",
}

# Duplicate (sample, cell_index) pairs, per OUR label. v4 shipped 16 of them
# (SD01623_BI 3, SD01923_BI 2, SD03522_BG 2, SD04219_BI 6, plus 3 on decoy
# SD02022). Marcel fixed that in v5 and v6 has ZERO, measured. The assert is kept
# and now demands zero, so a regression is visible instead of silently absorbed.
# Do NOT change this to a bare drop_duplicates(): 43,798 cohort cells share an
# exact (0,0) centroid and a cell_area_um2 of exactly 0.0, so an unkeyed dedupe
# destroys ~38k real cells. Any dedupe must be subset=["sample","cell_index"].
EXPECTED_DUPS = {}

# obs['is_MN'] on the PRISTINE niche input, i.e. our own markers+size call, which
# is_neuron_general must preserve. Measured 2026-08-18 directly from
# Novae_persample_INDEPENDENT_FINAL/per_sample/<S>__niches_independent.h5ad.
# The in-cell assert against this table is the tripwire for "somebody patched on
# top of a patch": if the notebook already carried a relabel, is_neuron_general
# would silently become that relabel's MN call instead.
EXPECTED_NATIVE_MN = {
    "SD00614_BG": 61, "SD01015_BG": 180, "SD01115_BG": 117, "SD01320_BG": 133,
    "SD01413_BA": 181, "SD01616_BI": 35, "SD01620_BI": 32, "SD01623_BI": 30,
    "SD01915_BG": 305, "SD01922_BI": 21, "SD01923_BI": 10, "SD02022_BI": 28,
    "SD02622_BI": 57, "SD02818_BG": 244, "SD02913_BA": 50, "SD03522_BG": 159,
    "SD03614_BG": 223, "SD03914_BG": 619, "SD04219_BI": 31, "SD05413_BG": 61,
}   # sums to 2,577

# in_GM True for 0 cells AND in_VH True for 0 cells. Listed here only so the
# patcher can print a warning next to the right samples; the notebook DERIVES
# mask presence from its own parquet slice and never reads this set, so a third
# section losing its mask in v5 is handled without editing this file.
KNOWN_NO_VH = {"SD01015_BG", "SD01616_BI"}

MD = """### Relabel v6 2026-08-18 - motor neurons from `neuron_classification_tdp_v6`

Cell calls come from `Ranger_procd_mw_final/annotation/neuron_classification_tdp_v6.parquet`,
read through the snapshot pinned at `neuron_classification_tdp_v6__snapshot_20260818_1649.parquet`
(md5 `e4f7ffc0fb8ebe794c4f0c7b1bce4095`, asserted below). **This replaces the relabel run
earlier today, which read the v2 file by mistake. Every number from that run is superseded.**

The v4 rule, exact (0 false positives, 0 false negatives) on each of the 22 sample keys
individually:

    is_MN = is_neuron_v3 & ~is_GAD & in_VH & (cell_area_um2 >= 250)

Four things about v4 that the v2 relabel got wrong or could not know:

- **`in_VH` is no longer a subset of `in_GM`.** 6,902 violations across 12 of our 20 sections,
  all of them in `robin_pathologist` samples. The v2 notebook asserted the subset relation;
  that assert is replaced here by a counted report written to `obs['in_VH_not_GM']` and to the
  trace. 35 of the 738 called motor neurons sit outside grey matter, which is anatomically
  impossible, so this is a mask-registration defect and not biology.
- **v4 ships 16 exactly-duplicated `(sample, cell_index)` pairs.** Deduplicated, the file is
  bit-exact the Jul-22 universe. The join drops them after asserting byte-identity. Over our
  20 sections the true counts are **599 motor neurons, not 601**, and **6,035 neurons, not
  6,037** -- all of the difference is in SD04219_BI.
- **`is_GAD` and `is_Interneuron` are no longer neuron subsets:** 455 of 474 GAD+ and 1,065 of
  1,237 Interneuron cells are non-neurons. `is_gabaergic` is therefore renamed
  `is_gabaergic_neuron`.
- **v4 has no nulls in any label column.** `has_vh_data = in_VH.notna()` would be True for all
  20 sections, which silently converts "no ventral-horn mask" into "zero motor neurons". It is
  redefined here as a measured count. The three columns that *do* carry nulls are
  `pathologist_class`, `cord_level` (100% null on 4 whole samples) and `tdp_proba` (281,859
  ALS cells, 0 Control).

`SD01015_BG` and `SD01616_BI` have `in_GM` True for **0** cells and `in_VH` True for **0**
cells. That is a missing mask, not a measurement: 71 and 138 cells respectively satisfy
everything except `in_VH`. Because `.sum()` skips `<NA>`, the only reduction-proof contract is
`is_MN = <NA>` for *every* cell in such a section, which is what this cell writes -- so a mean
is `NaN` and a bar chart has a gap rather than a zero. `is_MN_eligible` (no `in_VH` term) is
written on every section so an MN rate always has an honest denominator.

The join is gated on **spatial coordinates** (max |delta| < 1e-3 um; measured 1.8e-12 um), not
on row counts. A count picks the wrong member of the SD02022 pair. `is_TDP` and `tdp_proba` are
carried but confounded with disease by construction and must stay out of any contrast, and
`pathologist_class` scores are reported as raw counts only -- the 183 labels are very likely
in-sample and are 129-fold enriched for neurons relative to the file.
"""

CELL = r'''
# ── @@MARKER@@ ──────────────────────────────────────────────
# v4 cell calls. is_MN = is_neuron_v3 & ~is_GAD & in_VH & area >= 250um2.
#
# v4 has no nulls in the label columns, so nothing here NEEDS NA-safety today.
# The .astype("boolean").fillna(False) discipline is kept anyway: pd.NA is
# truthy, `== True` KEEPS <NA>, and this cell itself deliberately introduces
# <NA> on a section whose ventral-horn mask is missing.
import hashlib as _hashlib
import numpy as _np
import pandas as _pd

VH_PARQUET = ("@@PARQUET@@")
VH_PARQUET_MD5 = "@@PARQUETMD5@@"
VH_PARQUET_SAMPLE = "@@PKEY@@"      # verified by coordinate agreement, not by name
AREA_MIN = @@AREAMIN@@              # um^2, the v4 motor-neuron size gate. The file
                                    # only constrains it to (249.0152, 251.5203]
EXPECTED_DUP_ROWS = @@NDUPS@@       # exact-duplicate cell_index rows in this sample
EXPECTED_NATIVE_MN = @@NATIVEMN@@   # our markers+size call on the PRISTINE input
GAD_THRESHOLD = 8                   # only for the independent cross-check below

# md5 the snapshot. An array job takes minutes per sample; if somebody re-pins
# the snapshot mid-flight, the samples before and after would carry different
# labels in one cohort object and nothing would say so.
_h = _hashlib.md5()
with open(VH_PARQUET, "rb") as _fh:
    for _chunk in iter(lambda: _fh.read(1 << 22), b""):
        _h.update(_chunk)
assert _h.hexdigest() == VH_PARQUET_MD5, (
    f"{VH_PARQUET} md5 is {_h.hexdigest()}, expected {VH_PARQUET_MD5} -- the "
    f"pinned snapshot was rewritten; do not mix labels across one cohort")

_cols = ["cell_index", "x_centroid", "y_centroid",
         "is_neuron_v3", "neuron_prob_v3", "in_GM", "in_VH",
         "is_GAD", "is_Interneuron", "cell_area_um2", "mn_signature_prob",
         "is_MN", "cell_type", "is_TDP", "tdp_proba", "STMN2", "CE_STMN2",
         "pathologist_class", "vh_source",
         # vh_annotated is new in v5 and is READ by the VH-presence check below.
         # It has to be in this list: the slice is taken as _vh.loc[..., _cols],
         # so a column read but not listed here raises KeyError at runtime while
         # compiling and rendering perfectly. That cost a 20-task array.
         "vh_annotated",
         "cord_level", "disease_group", "patient_id"]
_vh = _pd.read_parquet(VH_PARQUET)
_missing = [c for c in _cols if c not in _vh.columns]
assert not _missing, (f"{VH_PARQUET} is missing {_missing}; the schema changed "
                      f"again -- it now has {list(_vh.columns)}")
_vh = _vh.loc[_vh["sample"] == VH_PARQUET_SAMPLE, _cols]
assert len(_vh), f"no rows for {VH_PARQUET_SAMPLE} in {VH_PARQUET}"

# ── v4 ships 16 exactly-duplicated (sample, cell_index) pairs ───
# reindex() raises ValueError on a duplicate index, so this kills 4 of the 20
# notebooks if left alone. Assert byte-identity BEFORE dropping: a conflicting
# pair would mean two different calls for one cell and keep="first" would pick
# one at random.
_dup_key = _vh.duplicated(subset=["cell_index"], keep=False)
_dup_all = _vh.duplicated(keep=False)
assert bool((_dup_key <= _dup_all).all()), (
    f"{VH_PARQUET_SAMPLE}: duplicated cell_index rows that DIFFER in some column "
    f"-- two different calls for one cell, refusing to pick one")
_n_dup = int(_vh.duplicated(subset=["cell_index"], keep="first").sum())
assert _n_dup == EXPECTED_DUP_ROWS, (
    f"{VH_PARQUET_SAMPLE}: {_n_dup} duplicate cell_index rows, expected "
    f"{EXPECTED_DUP_ROWS}. The duplication pattern changed -- update "
    f"EXPECTED_DUPS in relabel_v4real_at_source.py and re-check the counts")
_vh = _vh.drop_duplicates(subset=["cell_index"], keep="first")

# ── nulls: which columns are ALLOWED to carry them ──────────
# v4 had no nulls anywhere, so this asserted a blanket no-null contract. v5 and v6
# deliberately reintroduced them, and the blanket assert then killed exactly the
# two sections the nulls exist to protect (SD01015_BG in_VH 51,912 / is_MN 201;
# SD01616_BI in_VH 80,312 / is_MN 324). Allowed, with a reason each:
#   cord_level        100% null on 4 whole samples (metadata gap, not measured)
#   tdp_proba         NaN for every control by construction, plus 24.4% of ALS
#   pathologist_class 183 hand labels in the whole file, null elsewhere
#   in_VH, is_MN      NULL BY DESIGN on a section with no ventral-horn mask.
#                     This is the whole point of v5's vh_annotated: "not
#                     annotated" must not be encodable as False.
# Everything else must still be null-free, so a genuine schema slip is caught.
_nullable_ok = ("cord_level", "tdp_proba", "pathologist_class", "in_VH", "is_MN")
_nulls = {_c: int(_vh[_c].isna().sum()) for _c in _cols}
_bad_nulls = {_c: _n for _c, _n in _nulls.items()
              if _n and _c not in _nullable_ok}
assert not _bad_nulls, (f"{VH_PARQUET_SAMPLE}: nulls in columns that must not have "
                        f"them: {_bad_nulls}. in_VH/is_MN nulls are expected on a "
                        f"mask-less section; these are not those")
# and the nulls that ARE allowed still have to be self-consistent: in_VH is either
# fully null (no mask) or fully present, never partial, and is_MN nulls are a
# subset of the in_VH nulls
_n_invh_null = int(_vh["in_VH"].isna().sum())
assert _n_invh_null in (0, len(_vh)), (
    f"{VH_PARQUET_SAMPLE}: in_VH is null on {_n_invh_null} of {len(_vh)} rows. "
    f"A partial ventral-horn mask has no defined meaning here")
assert not bool((_vh["is_MN"].isna() & _vh["in_VH"].notna()).any()), (
    f"{VH_PARQUET_SAMPLE}: is_MN is null where in_VH is not; is_MN nulls must be "
    f"a subset of the mask-less rows")
print(f"nulls allowed and present: "
      f"{ {_c: _n for _c, _n in _nulls.items() if _n} }")

# obs_names are "<sample>:<cells.parquet cell_id>", and cell_index is cell_id - 1.
# The -1 was tested against both neighbours: +0 and -2 each give a coordinate
# agreement fraction of exactly 0.000 on all 20 samples.
_ci = _np.array([int(str(o).split(":")[1]) - 1 for o in adata.obs_names])
_m = _vh.set_index("cell_index").reindex(_ci)
_joined = int(_m["is_neuron_v3"].notna().sum())
print(f"joined {_joined:,} / {adata.n_obs:,} cells to {VH_PARQUET_SAMPLE} "
      f"({_n_dup} duplicate parquet rows dropped)")
assert _joined == adata.n_obs, (f"only {_joined} of {adata.n_obs} cells joined; "
                                f"cell_index is not aligned")

# ── THE JOIN GATE: coordinates, not counts ──────────────────
# A row count cannot prove a join, and here it actively misleads: our SD02022_BI
# raw 76,564 is closer to decoy SD02022 79,637 than to the true SD020_22_BI
# 80,618. The true sample agrees to 1.8e-12 um and the closest wrong key in the
# file is 2,670 um away in median, so this proves the join AND the sample-name
# mapping in one step. It is also the only thing that catches the wrong BUILD:
# the AUG5 objects index-join at 88-99% and place 392 of 599 motor neurons on
# the wrong cell with no index miss to raise on.
_xy = _np.asarray(adata.obsm["spatial"], dtype=float)
_dx = _np.abs(_xy[:, 0] - _m["x_centroid"].to_numpy(dtype=float))
_dy = _np.abs(_xy[:, 1] - _m["y_centroid"].to_numpy(dtype=float))
_maxd = float(_np.nanmax(_np.maximum(_dx, _dy)))
print(f"coordinate content gate: max |delta| = {_maxd:.3e} um")
assert _maxd < 1e-3, (f"spatial coordinates disagree by {_maxd:.3f}um -- this is "
                      f"either the WRONG parquet sample or the WRONG BUILD for "
                      f"this object. Do not relax this threshold")

# ── is the ventral-horn mask present at all? ────────────────
# v6 answers this three ways and we require all three to agree, because each one
# alone has failed at some point in this file's history:
#   (a) `vh_annotated`, Marcel's explicit flag, new in v5. Authoritative.
#   (b) in_VH.isna(), which is now meaningful (it was uniformly False in v4).
#   (c) DERIVED from the section's own geometry, which is what we used on v4 and
#       is the only one that keeps working if a future version drops the flag.
# NEVER vh_source: it reads 'auto_density_fixed' on both mask-less sections.
# NA-safe reads throughout: v6's in_VH is nullable, and .to_numpy(dtype=bool) on
# a nullable column containing pd.NA raises, so the bare v4 form would die here.
_in = _m["in_VH"].astype("boolean").fillna(False).to_numpy(dtype=bool)
_gm = _m["in_GM"].astype("boolean").fillna(False).to_numpy(dtype=bool)
_in_na = _m["in_VH"].isna().to_numpy()
_n_in_vh = int(_in.sum())
_n_in_gm = int(_gm.sum())

_flag_annotated = bool(_pd.Series(_m["vh_annotated"]).astype("boolean")
                       .fillna(False).all())
_na_says_present = not bool(_in_na.all())
_geom_says_present = bool(_n_in_vh > 0 or _n_in_gm > 0)
assert _flag_annotated == _na_says_present == _geom_says_present, (
    f"{VH_PARQUET_SAMPLE}: the three VH-presence signals disagree -- "
    f"vh_annotated={_flag_annotated}, in_VH-not-all-NA={_na_says_present}, "
    f"geometry={_geom_says_present}. One of them is now wrong and the MN call "
    f"for this section cannot be trusted until that is resolved by hand")
VH_MASK_PRESENT = _flag_annotated
VH_STATUS = "present" if VH_MASK_PRESENT else "absent_no_gm_no_vh"
# v6 also carries partial NA: is_MN is <NA> only on the confirmed neurons of a
# mask-less section, while its glia stay False. Recorded so the contract below is
# auditable against the delivery rather than reconstructed from it.
_n_isMN_na_delivered = int(_m["is_MN"].isna().sum())


def _nullable_vh(vals):
    """bool array -> pandas 'boolean'; all-<NA> when this section has no mask.

    All-<NA> and not just-the-candidates-<NA>: pandas .sum() SKIPS <NA>, so a
    column of False plus a handful of <NA> still sums to 0 and a reader sees a
    measurement. With every cell <NA>, .mean() is NaN and .sum(min_count=1) is
    NaN, which is what an undetermined section should look like.
    """
    if VH_MASK_PRESENT:
        return _pd.array(_np.asarray(vals, dtype=bool), dtype="boolean")
    return _pd.array([None] * len(vals), dtype="boolean")


# ── carry the columns ───────────────────────────────────────
# is_neuron_general must be captured BEFORE is_MN is overwritten. On a pristine
# notebook obs['is_MN'] is our own markers+size call; if this notebook had been
# patched on top of an earlier relabel it would be that relabel's MN call
# instead, which is what the assert below catches.
# read through astype("boolean").fillna(False), not np.asarray().astype(bool):
# the pristine column is plain bool with 0 nulls today (measured on all 20), but
# np.asarray on a nullable column turns pd.NA into True, i.e. into a motor neuron
adata.obs["is_neuron_general"] = (
    _pd.Series(adata.obs["is_MN"]).astype("boolean").fillna(False)
    .to_numpy(dtype=bool))
_n_gen = int(adata.obs["is_neuron_general"].sum())
assert _n_gen == EXPECTED_NATIVE_MN, (
    f"is_neuron_general is {_n_gen}, expected {EXPECTED_NATIVE_MN} from the "
    f"pristine markers+size call. This notebook was NOT restored from "
    f".bak_prev4relabel -- never patch on top of a patch")

adata.obs["is_neuron_prev"] = (
    _pd.Series(adata.obs["is_neuron"]).astype("boolean").fillna(False)
    .to_numpy(dtype=bool) if "is_neuron" in adata.obs.columns
    else _np.zeros(adata.n_obs, dtype=bool))
_n_prev = int(adata.obs["is_neuron_prev"].sum())

# DROP the stale is_neuron / neuron_prob. They already exist on the input object
# (10,085 over the 20 vs v4's 6,035), and writing v4 under the new names would
# leave every consumer keyed on the string 'is_neuron' reading the old column
# with no error -- measured as a 110% overstatement post-filter.
_dropped = [_c for _c in ("is_neuron", "neuron_prob") if _c in adata.obs.columns]
adata.obs = adata.obs.drop(columns=_dropped)

for _c in ("is_neuron_v3", "is_GAD", "is_Interneuron", "is_TDP"):
    adata.obs[_c] = _m[_c].to_numpy(dtype=bool)
# v6's is_MN is nullable (525 cohort-wide), so this must go through
# astype("boolean"), not .to_numpy(dtype=bool), which raises on pd.NA.
adata.obs["is_MN_v6_asdelivered"] = _pd.array(_m["is_MN"].to_numpy(), dtype="boolean")
for _c in ("cell_area_um2", "mn_signature_prob", "tdp_proba", "neuron_prob_v3"):
    adata.obs[_c] = _m[_c].to_numpy(dtype="float32")
for _c in ("STMN2", "CE_STMN2"):
    # renamed: an obs column called STMN2 would shadow the panel gene of the
    # same name in every scanpy colour lookup
    adata.obs[f"{_c}_mw"] = _m[_c].to_numpy(dtype="int32")

# in_GM / in_VH go <NA> on a mask-less section. in_VH's NAs now come FROM the
# file (v6 ships them); _nullable_vh reproduces the same thing and is kept so the
# behaviour is identical whether or not a future version supplies them.
adata.obs["in_GM"] = _nullable_vh(_gm)
adata.obs["in_VH"] = _nullable_vh(_in)

# has_vh_data was in_VH.notna(), which under v4 is True for 20/20 and destroys
# the sentinel. Redefined as the measured flag. Kept under the old name because
# promote_pass2.py and concat_pass2_v4.py require the column.
adata.obs["has_vh_data"] = _np.full(adata.n_obs, VH_MASK_PRESENT, dtype=bool)
adata.obs["vh_mask_empty"] = _np.full(adata.n_obs, not VH_MASK_PRESENT, dtype=bool)
adata.obs["vh_source"] = _pd.Categorical(
    _m["vh_source"].fillna("unassigned").astype(str).to_numpy())

# v4's cell_type carries no information (cell_type == 'motor_neuron' IS is_MN and
# 'other_neuron' IS is_neuron_v3 & ~is_MN, both exact on all 22 keys) and its
# 'glia' bucket is every non-neuron, endothelium and meninges included. Carried
# under _mw because this notebook assigns its OWN obs['cell_type'] from
# marker-gene argmax further down, and the astrocyte notebook selects
# cell_type == 'Astrocytes'.
adata.obs["cell_type_mw"] = _pd.Categorical(_m["cell_type"].astype(str).to_numpy())
# "unlabelled" rather than <NA> so a reader walking codes/categories directly
# cannot hit a -1 code
adata.obs["pathologist_class_mw"] = _pd.Categorical(
    _m["pathologist_class"].fillna("unlabelled").astype(str).to_numpy())
adata.obs["tdp_evaluated"] = _m["tdp_proba"].notna().to_numpy(dtype=bool)

_neu = adata.obs["is_neuron_v3"].to_numpy(dtype=bool)
_gad = adata.obs["is_GAD"].to_numpy(dtype=bool)
_int = adata.obs["is_Interneuron"].to_numpy(dtype=bool)
_mn_raw = (adata.obs["is_MN_v6_asdelivered"].astype("boolean")
           .fillna(False).to_numpy(dtype=bool))
_mn_na = adata.obs["is_MN_v6_asdelivered"].isna().to_numpy()
_area = adata.obs["cell_area_um2"].to_numpy()
_big = (_area >= AREA_MIN)

# ── the in_GM check, REPORTED not asserted ──────────────────
# v2 asserted `not (in_VH & ~in_GM).any()`. That is FALSE in v4 -- 6,902
# violations over our 20, in 12 of them, every one in a robin_pathologist
# sample and none in an auto_density_fixed one. Asserting it kills 12 of 20
# notebooks; with allow_errors=True it kills them QUIETLY, half-writing the
# contract. So it is counted, flagged per cell, and traced. in_GM is NOT applied
# to the MN call: the verified recipe does not use it.
adata.obs["in_VH_not_GM"] = (_in & ~_gm)
_n_vh_not_gm = int((_in & ~_gm).sum())
_n_mn_outside_gm = int((_mn_raw & ~_gm).sum())
if _n_vh_not_gm:
    print(f"in_VH outside in_GM: {_n_vh_not_gm:,} cells "
          f"({100.0 * _n_vh_not_gm / max(_n_in_vh, 1):.1f}% of in_VH); "
          f"{_n_mn_outside_gm} of them are called MN. Hand-drawn "
          f"robin_pathologist VH polygons extend past the auto GM mask -- a "
          f"motor neuron outside grey matter is anatomically impossible, so "
          f"treat this as mask registration, not biology")

# ── derived masks ───────────────────────────────────────────
# is_MN_eligible has NO in_VH term and is written on every section, so an MN
# rate has a denominator even where the numerator is undetermined.
adata.obs["is_MN_eligible"] = _neu & ~_gad & _big
adata.obs["is_MN_invh"] = _nullable_vh(_neu & _in)
adata.obs["is_GADvh"] = _nullable_vh(_gad & _in)
# renamed from is_gabaergic: in v4 is_GAD is NOT neuron-restricted, so this
# restriction discards most GAD+ cells and the name has to say so
adata.obs["is_gabaergic_neuron"] = _neu & _gad

# ── THE RECIPE GATE, v6: UNGATED ────────────────────────────
# v4/v5 baked an area >= 250 term into is_MN. v6 removes it, deliberately, so the
# threshold becomes our explicit downstream choice. The gate therefore asserts the
# UNGATED rule and would fire if the area term ever came back.
# Tested against the AS-DELIVERED column, on the non-NA rows only: a mask-less
# section has is_MN <NA> on its confirmed neurons, and pd.NA is truthy, so
# comparing those rows would silently pass whatever we asserted.
# No in_GM term: the verified recipe does not use it (and in v6 in_VH is not even
# a subset of in_GM -- see the counted report above).
_expect = _neu & ~_gad & _in
assert (_mn_raw[~_mn_na] == _expect[~_mn_na]).all(), (
    f"is_MN != is_neuron_v3 & ~is_GAD & in_VH on "
    f"{int((_mn_raw[~_mn_na] != _expect[~_mn_na]).sum())} of "
    f"{int((~_mn_na).sum())} determinate cells -- the v6 recipe changed. If the "
    f"area term has come back, this is a v5-style file and AREA_MIN belongs in "
    f"the rule again. Re-derive before trusting any motor-neuron count")
# and the NA pattern is itself a claim worth checking: <NA> should appear only on
# a mask-less section, and only on cells that are neurons
assert not bool(_mn_na.any()) or not VH_MASK_PRESENT, (
    f"is_MN has {int(_mn_na.sum())} <NA> cells on a section whose VH mask is "
    f"PRESENT -- that combination has no meaning")
assert not bool((_mn_na & ~_neu).any()), (
    f"is_MN is <NA> on {int((_mn_na & ~_neu).sum())} non-neurons; v6 leaves glia "
    f"in a mask-less section as False, so this is a different file than expected")

# ── the area threshold, now OURS to choose and to name ──────
# Marcel's v6 is_MN is the candidate pool. We keep an `is_MN` that means what it
# has always meant (area >= 250) so no downstream consumer silently changes
# meaning, and carry the alternatives beside it so a threshold question can be
# answered without regenerating anything.
#   250  Marcel's v4/v5 gate. `v6 is_MN & area >= 250` reproduces v5 exactly.
#   211  the 2-component mixture crossing on log10 area over the candidate pool
#        (bootstrap [188, 230]); also the Youden optimum against the pathologist
#        labels is a 204-220 plateau. Neither anchors to published human values;
#        see MN_AREA_THRESHOLD_RATIONALE.md.
# NOTE cell_area_um2 correlates with transcript count at rho ~0.82 inside this
# pool, so every area threshold is partly a depth filter. And 47 candidates
# cohort-wide have area exactly 0.0, which is a segmentation failure and not a
# small neuron, hence area_valid below rather than folding it into the cut.
adata.obs["is_MN_candidate"] = _nullable_vh(_expect)
adata.obs["area_valid"] = (_area > 0)
adata.obs["is_MN_a250"] = _nullable_vh(_expect & (_area >= 250.0))
adata.obs["is_MN_a211"] = _nullable_vh(_expect & (_area >= 211.0))
# UNGATED, 2026-08-19. The v6 annotations are canonical and human-inspected: QC
# filters may REMOVE such a cell, nothing may RE-LABEL one. The previous line was
# `_nullable_vh(_expect & _big)`, which turned 179 canonical MNs cohort-wide into
# not-MN on a 250 um2 threshold with no literature anchor -- a threshold separately
# shown to select ON the TDP phenotype (it drops 29.6% of control and 29.7% of ALS
# TDP-negative in-VH neurons but only 8.0% of TDP-positive, and restoring the 179
# moves the TDP+ MN fraction from 11.3% to 9.2%). The gated views survive as
# is_MN_a250 and is_MN_a211, which are opt-in.
adata.obs["is_MN"] = _nullable_vh(_expect)
assert int(_pd.Series(adata.obs["is_MN"]).astype("boolean").fillna(False).sum()) == \
    int(_pd.Series(adata.obs["is_MN_v6_asdelivered"]).astype("boolean")
        .fillna(False).sum()) or not VH_MASK_PRESENT, (
    "obs['is_MN'] must equal the v6 as-delivered call. If this fires, someone has "
    "re-introduced a gate; fix that rather than this assert")

# independent cross-check of their GAD call against our own transcript counts.
# The candidate set does NOT assume is_GAD is neuron-restricted, because in v4
# it is not.
_g = []
for _gene in ("GAD1", "GAD2"):
    _col = adata.layers["counts"][:, list(adata.var_names).index(_gene)]
    _g.append(_np.asarray(_col.todense()).ravel() if hasattr(_col, "todense")
              else _np.asarray(_col).ravel())
_ours = (_g[0] >= GAD_THRESHOLD) | (_g[1] >= GAD_THRESHOLD)
_cand = _neu & _in
print(f"GAD among in-VH neurons: parquet {int((_gad & _cand).sum())}, "
      f"our GAD1/GAD2 >= {GAD_THRESHOLD} rule {int((_ours & _cand).sum())}")
print(f"is_GAD {int(_gad.sum())} of which {int((_gad & ~_neu).sum())} are NON-neurons; "
      f"is_Interneuron {int(_int.sum())} of which {int((_int & ~_neu).sum())} are "
      f"NON-neurons  [neither flag is neuron-restricted in v4]")

# ── one mutually exclusive label per cell ───────────────────
# No "VH unknown" bucket is needed for the v4 nulls -- there are none. Two
# buckets exist for the mask-less sections because their neurons are genuinely
# unclassifiable with respect to the ventral horn, and calling them "outside VH"
# would be a claim the data does not support. GAD+ non-neurons get their own
# bucket rather than vanishing into "non-neuron": in v4 that is 455 of 474 GAD+
# cells cohort-wide, and folding them in would throw the GAD information away.
_cls = _np.full(adata.n_obs, "non-neuron", dtype=object)
_cls[~_neu & _gad] = "non-neuron, GAD+"
if VH_MASK_PRESENT:
    _cls[_neu & ~_in & ~_gad] = "neuron, outside VH"
    _cls[_neu & ~_in & _gad] = "GABAergic neuron, outside VH"
    _cls[_neu & _in & _gad] = "GABAergic neuron, in VH"
    # UNGATED, 2026-08-19: every in-VH non-GAD neuron the v6 annotation calls MN
    # reads MN here. The sub-threshold category is deliberately KEPT DECLARED but
    # empty, so a future regression shows up as a non-zero count in value_counts()
    # instead of hiding.
    _cls[_neu & _in & ~_gad] = "MN"
else:
    _cls[_neu & ~_gad] = "neuron, VH mask absent"
    _cls[_neu & _gad] = "GABAergic neuron, VH mask absent"
adata.obs["neuron_class"] = _pd.Categorical(
    _cls, categories=["MN", "neuron, in VH (sub-threshold)",
                      "GABAergic neuron, in VH", "neuron, outside VH",
                      "GABAergic neuron, outside VH", "neuron, VH mask absent",
                      "GABAergic neuron, VH mask absent", "non-neuron, GAD+",
                      "non-neuron"])
assert int(adata.obs["neuron_class"].isna().sum()) == 0, \
    "neuron_class has cells outside its category list -- the ladder is not total"
# The ladder's "MN" bucket is now the UNGATED canonical call, so this check ties it
# to obs['is_MN'], which is itself the as-delivered column. Before 2026-08-19 both
# sides were gated and agreed at 595 while 774 cells carried the canonical call.
_cls_mn = int((adata.obs["neuron_class"] == "MN").sum())
_our_mn = int(_pd.Series(adata.obs["is_MN"]).astype("boolean").fillna(False).sum())
assert _cls_mn == _our_mn, (
    f"neuron_class MN ({_cls_mn}) disagrees with obs['is_MN'] ({_our_mn})")
# and the two in-VH non-GAD buckets together must reconstitute the ungated
# candidate, which is what ties the ladder back to Marcel's delivery
if VH_MASK_PRESENT:
    _cls_cand = int(((adata.obs["neuron_class"] == "MN") |
                     (adata.obs["neuron_class"] == "neuron, in VH (sub-threshold)")).sum())
    assert _cls_cand == int(_mn_raw.sum()), (
        f"MN + sub-threshold ({_cls_cand}) != ungated v6 is_MN "
        f"({int(_mn_raw.sum())}); the ladder has drifted from the delivery")

# ── pathologist ground truth, raw counts only ───────────────
# The 183 labels almost certainly trained the v3/v4 classifier (is_neuron_v3
# scores 183/183; all 14 disagreements with v2 resolve in v4's favour and 0
# against) and training membership could not be verified from any file on the
# cluster. They are 0.00887% of the file with a neuronal prevalence 129-fold
# above it. So: raw tp/fn/fp/tn, no percentages, no "sensitivity" anywhere a
# figure could pick it up. The 40 labels on decoy SD02022 cannot reach here
# because the slice is taken by PARQUET_KEY.
_lab = adata.obs["pathologist_class_mw"].to_numpy(dtype=object)
_lab_neuronal = (_lab == "neuronal")
_lab_glial = (_lab == "glial")
_path = {"n_labelled": int(_lab_neuronal.sum() + _lab_glial.sum()),
         "tp_neuronal_called_neuron": int((_lab_neuronal & _neu).sum()),
         "fn_neuronal_called_glia": int((_lab_neuronal & ~_neu).sum()),
         "fp_glial_called_neuron": int((_lab_glial & _neu).sum()),
         "tn_glial_called_glia": int((_lab_glial & ~_neu).sum()),
         "caveat": ("183 hand labels cohort-wide, unevenly spread, 129-fold "
                    "enriched for neurons relative to the file. TRAINING-SET "
                    "MEMBERSHIP NOT VERIFIED -- treat as in-sample. Usable to "
                    "SCORE is_neuron_v3, not as a training or balancing set, "
                    "and never as a generalisation estimate.")}
if _path["n_labelled"]:
    print(f"pathologist_class vs is_neuron_v3 (raw counts, this section): "
          f"TP {_path['tp_neuronal_called_neuron']} "
          f"FN {_path['fn_neuronal_called_glia']} "
          f"FP {_path['fp_glial_called_neuron']} "
          f"TN {_path['tn_glial_called_glia']}  [in-sample, do not quote as a rate]")

# ── sample-level metadata and status, plain strings only ────
# A None anywhere in .uns makes the h5ad unreadable in every anndata version.
# The "unassigned" fallback is load-bearing: cord_level is 100% null on 4 of the
# 22 samples.
adata.uns["sample_meta_v4"] = {
    _k: (str(_m[_k].dropna().unique()[0]) if _m[_k].notna().any() else "unassigned")
    for _k in ("cord_level", "disease_group", "patient_id", "vh_source")}

_n_neu = int(_neu.sum())
_n_mn = int(_mn_raw.sum())
_n_elig = int(adata.obs["is_MN_eligible"].sum())
_n_undet = _n_elig if not VH_MASK_PRESENT else 0
_n_invh = int((_neu & _in).sum())
_n_sub = int((_neu & _in & ~_gad & ~_big).sum())
_n_gadvh = int((_gad & _in).sum())
_n_tdp = int(adata.obs["is_TDP"].sum())
_n_tdp_eval = int(adata.obs["tdp_evaluated"].sum())
_n_area_zero = int((_area <= 0).sum())

adata.uns["vh_status"] = VH_STATUS
adata.uns["mn_call_valid"] = bool(VH_MASK_PRESENT)
adata.uns["mn_call_note"] = (
    "motor-neuron call is valid for this section" if VH_MASK_PRESENT else
    (f"no GM or VH mask for this section: the v4 parquet ships is_MN=False for "
     f"all {adata.n_obs} cells and vh_source="
     f"{adata.uns['sample_meta_v4']['vh_source']}, but 0 cells are in_GM and 0 "
     f"in_VH, so is_MN=0 is a MISSING MASK, not a count. {_n_undet} MN-eligible "
     f"cells are UNDETERMINED. is_MN is <NA> for every cell here on purpose; "
     f"never render this section as 0 motor neurons."))
adata.uns["vh_mask_metrics_v4"] = {
    "n_in_GM": _n_in_gm, "n_in_VH": _n_in_vh,
    "n_in_VH_not_GM": _n_vh_not_gm, "n_MN_outside_GM": _n_mn_outside_gm,
    "n_mn_eligible": _n_elig, "n_mn_undetermined": _n_undet}
adata.uns["pathologist_scores_v4"] = _path
adata.uns["is_TDP_caveat"] = (
    "tdp_proba is exactly 0.0 for all 908,578 Control cells (max 0.0, zero "
    "nulls) and NaN for 281,859 ALS cells, so is_TDP is True in 0 of 10 "
    "controls BY CONSTRUCTION and False also means not-evaluated. Use "
    "tdp_evaluated as the denominator. Keep is_TDP out of every ALS-vs-control "
    "contrast and figure until Marcel explains the fit.")
adata.uns["pathologist_class_caveat"] = _path["caveat"]
adata.uns["in_VH_not_GM_note"] = (
    f"{_n_vh_not_gm} cells are in_VH and not in_GM in this section "
    f"({_n_mn_outside_gm} of them called MN). v4 broke the subset relation on "
    f"7,763 cells cohort-wide, all in robin_pathologist samples. in_GM is NOT "
    f"applied to the MN call; the verified recipe does not use it.")
adata.uns["cell_type_mw_note"] = (
    "cell_type_mw is v4's cell_type, renamed. It is fully derived: "
    "cell_type_mw == 'motor_neuron' equals is_MN and 'other_neuron' equals "
    "is_neuron_v3 & ~is_MN, exact on all 22 sample keys. Its 'glia' bucket is "
    "every non-neuron, endothelium and meninges included. obs['cell_type'] is "
    "this notebook's own marker-gene argmax and is a different thing.")
adata.uns["v6_parquet"] = VH_PARQUET
adata.uns["v6_parquet_md5"] = VH_PARQUET_MD5
adata.uns["relabel_version"] = "@@RELVER@@"

print(f"is_neuron_v3 {_n_neu:,} (stale obs is_neuron was {_n_prev:,}, dropped "
      f"{_dropped}; our markers+size call {_n_gen:,})  ->  in VH {_n_invh:,}  ->  "
      f"is_MN {_n_mn:,}")
print(f"  in-VH GAD+ {_n_gadvh:,} excluded, {_n_sub:,} excluded as < {AREA_MIN} um2; "
      f"MN-eligible ignoring VH {_n_elig:,}")
print(f"  is_TDP {_n_tdp:,} of {_n_tdp_eval:,} evaluated cells "
      f"({adata.n_obs - _n_tdp_eval:,} have tdp_proba NaN)  "
      f"[confounded with disease -- carried, not for contrasts]")
print(f"  cell_area_um2 == 0 on {_n_area_zero:,} cells (2.3% cohort-wide); never "
      f"take log(area) or an area-normalised density without masking area > 0")
print(f"  sample_meta_v4 {adata.uns['sample_meta_v4']}")
print(f"  vh_status {VH_STATUS}; in_GM {_n_in_gm:,}, in_VH {_n_in_vh:,}")
if not VH_MASK_PRESENT:
    print("NO VENTRAL-HORN MASK FOR THIS SECTION -- and no grey-matter mask "
          "either, so there is no geometry to have landed off-tissue.")
    print(f"  is_MN is <NA> for all {adata.n_obs:,} cells BY DESIGN. "
          f"{_n_undet} MN-eligible cells (is_neuron_v3 & ~is_GAD & area >= "
          f"{AREA_MIN}) are UNDETERMINED, not zero.")
    print("  vh_source says 'auto_density_fixed', which is a PLAN tag with no "
          "outcome semantics. Six of the eight auto_density_fixed sections have "
          "a real mask; this one does not.")
print(adata.obs["neuron_class"].value_counts()
      .reindex(adata.obs["neuron_class"].cat.categories).to_string())

try:
    __TRACE__
except NameError:
    __TRACE__ = {}
__TRACE__.update({
    "vh_relabel": True,
    "vh_relabel_version": "@@RELVER@@",
    "vh_parquet": VH_PARQUET,
    "vh_parquet_md5": VH_PARQUET_MD5,
    "vh_parquet_sample": VH_PARQUET_SAMPLE,
    "coord_gate_max_delta_um": _maxd,
    "mn_definition": ("v6 is_MN as delivered = is_neuron_v3 & ~is_GAD & in_VH, "
                      "UNGATED on area (changed 2026-08-19; the former "
                      f"cell_area_um2 >= {AREA_MIN} view is is_MN_a250)"),
    "mn_area_min_um2": AREA_MIN,
    "mn_area_window_um2": "(249.0152, 251.5203] -- 250 is inside it, not unique",
    "n_parquet_dup_rows_dropped": _n_dup,
    "parquet_nulls": _nulls,
    "dropped_stale_cols": list(_dropped),
    "n_is_neuron_general_raw": _n_gen,
    "n_is_neuron_v3_raw": _n_neu,
    "n_is_neuron_prev_raw": _n_prev,
    "n_is_MN_invh_raw": _n_invh,
    "n_is_MN_raw": _n_mn,
    "n_is_MN_eligible_raw": _n_elig,
    "n_mn_undetermined": _n_undet,
    "n_subthreshold_invh_raw": _n_sub,
    "n_is_GADvh_raw": _n_gadvh,
    "n_is_GAD_raw": int(_gad.sum()),
    "n_is_GAD_nonneuron_raw": int((_gad & ~_neu).sum()),
    "n_is_GAD_neuron_raw": int((_gad & _neu).sum()),
    "n_is_Interneuron_raw": int(_int.sum()),
    "n_is_Interneuron_nonneuron_raw": int((_int & ~_neu).sum()),
    "n_is_Interneuron_neuron_raw": int((_int & _neu).sum()),
    "n_is_TDP_raw": _n_tdp,
    "n_tdp_evaluated_raw": _n_tdp_eval,
    "n_in_GM_raw": _n_in_gm,
    "n_in_VH_raw": _n_in_vh,
    "n_inVH_not_inGM": _n_vh_not_gm,
    "n_MN_outside_GM": _n_mn_outside_gm,
    "n_cell_area_zero": _n_area_zero,
    "vh_mask_present": bool(VH_MASK_PRESENT),
    "vh_status": VH_STATUS,
    "mn_call_valid": bool(VH_MASK_PRESENT),
    "has_vh_data": bool(VH_MASK_PRESENT),
    "pathologist_scores_v4": {_k: _v for _k, _v in _path.items() if _k != "caveat"},
    "sample_meta_v4": dict(adata.uns["sample_meta_v4"]),
    "neuron_class_raw": {str(_k): int(_v) for _k, _v in
                         adata.obs["neuron_class"].value_counts().items()},
})
# sentinel: the trace cell asserts this, so a relabel that died half-way cannot
# write a trace that looks clean and get promoted
__RELABEL_V4REAL_OK__ = True
'''

TRACE_EXTRA = '''
# ── @@MARKER@@ - counts for the relabelled columns ───────────
# is_MN, in_VH, in_GM, is_MN_invh and is_GADvh are nullable: <NA> for every cell
# on a section with no ventral-horn mask. np.asarray(...).astype(bool) treats
# pd.NA as True and would turn every cell of such a section into a motor neuron,
# so every count goes through astype("boolean").fillna(False).
import pandas as _pd_tr

assert globals().get("__RELABEL_V4REAL_OK__") is True, \\
    "the v4REAL relabel cell did not complete -- refusing to write a trace that looks clean"


def _bool_mask(col):
    return _pd_tr.Series(col).astype("boolean").fillna(False).to_numpy(dtype=bool)


for _obj, _tag in ((adata, "raw"), (sample, "final")):
    for _c in ("is_MN", "is_MN_v6_asdelivered", "is_MN_invh", "is_MN_eligible",
               "in_VH", "in_GM", "in_VH_not_GM", "is_GAD", "is_GADvh",
               "is_gabaergic_neuron", "is_Interneuron", "is_TDP",
               "tdp_evaluated", "is_neuron_v3", "is_neuron_general",
               "is_neuron_prev", "has_vh_data", "vh_mask_empty"):
        if _c in _obj.obs.columns:
            __TRACE__[f"n_{_c}_{_tag}"] = int(_bool_mask(_obj.obs[_c]).sum())
    if "is_MN" in _obj.obs.columns:
        __TRACE__[f"n_is_MN_na_{_tag}"] = int(
            _pd_tr.Series(_obj.obs["is_MN"]).astype("boolean").isna().sum())
    for _c in ("neuron_class", "cell_type_mw", "pathologist_class_mw", "vh_source"):
        if _c in _obj.obs.columns:
            __TRACE__[f"{_c}_{_tag}"] = {
                str(_k): int(_v) for _k, _v in _obj.obs[_c].value_counts().items()}
# the stale column must be gone, not merely unused
assert "is_neuron" not in adata.obs.columns, \\
    "stale obs['is_neuron'] survived the relabel -- consumers keyed on that string " \\
    "would read the pre-v4 call (10,085 vs 6,035) with no error"
__TRACE__["mn_definition"] = ("v6 is_MN as delivered = is_neuron_v3 & ~is_GAD & "
                              "in_VH, UNGATED on area (2026-08-19). is_MN_a250 and "
                              "is_MN_a211 are the opt-in gated views. WARNING: "
                              "is_MN_eligible is area-gated with NO in_VH term, so it "
                              "EXCLUDES 179 of the canonical MNs and must not be used "
                              "as their denominator")
__TRACE__["published_cohort_counts"] = (
    "over our 20 sections, whole-section and AFTER dedupe: 599 is_MN, 6,035 "
    "is_neuron_v3, 130 pathologist labels. Summing the V4_FACTS table gives 601 "
    "and 6,037 because 16 rows are duplicated (all of the difference on our 20 "
    "is SD04219_BI). SD01015_BG and SD01616_BI are UNDETERMINED, never 0.")
'''


def render(body, sample):
    return (body.replace("@@MARKER@@", MARKER)
                .replace("@@PARQUETMD5@@", PARQUET_MD5)
                .replace("@@PARQUET@@", PARQUET)
                .replace("@@AREAMIN@@", repr(AREA_MIN))
                .replace("@@NDUPS@@", repr(EXPECTED_DUPS.get(sample, 0)))
                .replace("@@NATIVEMN@@", repr(EXPECTED_NATIVE_MN[sample]))
                .replace("@@RELVER@@", RELABEL_VERSION)
                .replace("@@PKEY@@", PARQUET_KEY.get(sample, sample)))


def src(text):
    return text.splitlines(keepends=True)


def code_cell(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src(t)}


def md_cell(t):
    return {"cell_type": "markdown", "metadata": {}, "source": src(t)}


def find_nb(sample):
    for nm in (f"(RK)QC_per_sample_dualpass_{sample}.ipynb",
               f"(RK)QC_per_sample_dualpass_{sample}-Copy1.ipynb"):
        p = NBDIR / nm
        if p.exists():
            return p
    raise SystemExit(f"no notebook for {sample}")


def preflight(text, label):
    """Parse an injected cell and resolve every module.attr against the real module.

    This is the gate for the 2026-08-17 bug: `_os1.exists(...)` instead of
    `_os1.path.exists(...)` is valid syntax, so compile() passed, and the
    AttributeError only surfaced at runtime -- after which the run carried on and
    wrote a plausible object with a whole analysis step silently missing.
    """
    tree = ast.parse(text)
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name.split(".")[0]] = a.name
    bad = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            modname = aliases[node.value.id]
            try:
                mod = __import__(modname, fromlist=["_"])
            except ImportError:
                continue                      # not importable here; runtime's problem
            if not hasattr(mod, node.attr):
                bad.append(f"{node.value.id}({modname}).{node.attr}")
    if bad:
        raise RuntimeError(f"{label}: unresolved module attributes {sorted(set(bad))}")
    return len(aliases)


def restore(path, dry=False):
    """Put the pre-v2-relabel notebook back, and refuse if it is not clean.

    The current file (v2-relabelled) is preserved under BAK first, so this is
    reversible; RESTORE_FROM is left untouched so it stays the one canonical
    pristine copy.
    """
    ref = Path(str(path) + RESTORE_FROM)
    if not ref.exists():
        raise RuntimeError(f"{path.name}: no {RESTORE_FROM} to restore from. That "
                           f"backup is the only pristine copy -- do not proceed by "
                           f"patching the current file")
    text = ref.read_text()
    if PREREQ_MARKER not in text:
        raise RuntimeError(f"{ref.name}: no '{PREREQ_MARKER}' cell. That backup is "
                           f"not counts-corrected, so X would stay log-normalised "
                           f"through QC. Refusing")
    stale = [m for m in STALE_MARKERS if m in text]
    if stale:
        raise RuntimeError(f"{ref.name}: carries {stale}. The 'pristine' backup is "
                           f"itself relabelled -- never patch on top of a patch. "
                           f"Find a clean copy before going further")
    if MARKER in text:
        raise RuntimeError(f"{ref.name}: already carries {MARKER!r}")
    bak = Path(str(path) + BAK)
    if not dry:
        if not bak.exists():
            shutil.copy2(path, bak)          # keep the v2-relabelled state
        shutil.copy2(ref, path)
    return {"restored_from": ref.name, "kept_current_as": bak.name}


def patch(path, sample, dry=False):
    rep = {"nb": path.name}

    # refuse to run twice: check the file as it stands BEFORE restoring
    if MARKER in path.read_text():
        rep["skipped"] = f"already carries {MARKER!r}; delete/restore to redo"
        return rep

    # The notebook is built ENTIRELY IN MEMORY from the pristine backup, and the
    # live file is not touched until the final write. restore() used to run here,
    # which meant any later failure (and there was one: see fix_consumers_in_nb)
    # left the operator with sample 1 reverted to pristine and samples 2-20 still
    # carrying the previous relabel, i.e. a mixed cohort after a run that reported
    # itself clean. Validate first, mutate last.
    #
    # restore(dry=True) performs every check (backup exists, carries PREREQ_MARKER,
    # carries no STALE_MARKERS) and copies nothing, so the validation still happens
    # in both modes -- a dry run that skipped it would not catch a contaminated
    # backup, which is the one thing this design cannot afford to miss.
    rep.update(restore(path, dry=True))
    nb = json.loads(Path(str(path) + RESTORE_FROM).read_text())
    cells = nb["cells"]
    rep["cells_before"] = len(cells)

    body = render(CELL, sample)
    rep["preflight_modules"] = preflight(body, f"{path.name}:relabel_v4real")

    i_fix = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "restore raw counts" in "".join(c["source"]):
            i_fix = i
            break
    if i_fix is None:
        raise RuntimeError(f"{path.name}: counts-restoration cell not found")
    cells.insert(i_fix + 1, md_cell(MD))
    cells.insert(i_fix + 2, code_cell(body))
    rep["relabel_cell_at"] = i_fix + 2

    rep.update(fix_consumers_in_nb(cells))

    extra = render(TRACE_EXTRA, sample)
    preflight(extra, f"{path.name}:trace_extra")
    rep["trace_extended"] = False
    for c in reversed(cells):
        if c["cell_type"] == "code" and "TRACE WRITTEN ->" in "".join(c["source"]):
            s = "".join(c["source"])
            anchor = "_TRACE_DIR = "
            assert anchor in s, f"{path.name}: trace cell has no {anchor!r} anchor"
            c["source"] = src(s.replace(anchor, extra + "\n" + anchor, 1))
            rep["trace_extended"] = True
            break
    if not rep["trace_extended"]:
        raise RuntimeError(f"{path.name}: no 'TRACE WRITTEN ->' cell; the trace "
                           f"sentinel would never be checked")

    rep["cells_after"] = len(cells)
    if dry:
        rep["dry_run"] = True
        return rep
    # everything above validated, so it is now safe to touch the live file:
    # restore() preserves the current state under BAK, then the patched notebook
    # is written over it.
    rep.update(restore(path, dry=False))
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return rep


# ── consumer fixes inside the notebook ──────────────────────────────────────
# Each entry is (label, kind, old, new, how_many_cells_must_match) with kind in
# {"literal", "regex"}. A pattern that matches fewer cells than required raises:
# the whole point of this pass is that a silent no-op here is exactly the failure
# mode being fixed. kind is explicit rather than sniffed from the pattern text.
NB_FIXES = [
    # domain_qc composition columns read the STALE obs['is_neuron'] (9,290
    # post-filter) instead of v4's (4,415) -- a 110% overstatement with no error.
    ("domain_qc_flag_list", "literal",
     "for flag in ['is_neuron', 'is_MN']:",
     "for flag in ['is_neuron_v3', 'is_MN']:", 1),

    # is_MN is nullable again, and np.asarray(...).astype(bool) counts pd.NA as
    # True, i.e. as a motor neuron
    ("domain_qc_pct_na_safe", "literal",
     ".apply(lambda s: np.asarray(s).astype(bool).mean() * 100))",
     ".apply(lambda s: pd.Series(s).astype('boolean').fillna(False)\n"
     "                          .to_numpy(dtype=bool).mean() * 100))", 1),

    # median(pct_is_MN) across Novae domains is exactly 0.0 in 16 of 20 samples
    # under v4, so "above-median MN content" fired for any domain holding one
    # motor neuron. Require a non-degenerate baseline and say what the test was.
    ("qc_trace_mn_note", "literal",
     '    hot = [d for d in flagged_domains if domain_qc.loc[d, "pct_is_MN"] > '
     'domain_qc["pct_is_MN"].median()]\n'
     "    if hot:\n"
     '        mn_note = f" ({\', \'.join(hot)} carry above-median MN content -- '
     'check MN somata before calling these smear)"\n',
     '    _mn_base = float(domain_qc["pct_is_MN"].median())\n'
     "    # under v4 that median is exactly 0.0 in 16 of 20 sections, which makes\n"
     "    # \"above median\" true for any domain holding a single motor neuron. Say\n"
     "    # which test actually applied rather than implying a distribution.\n"
     '    hot = [d for d in flagged_domains if domain_qc.loc[d, "pct_is_MN"] > _mn_base]\n'
     "    if hot:\n"
     '        _how = (f"above the {_mn_base:.3f}% domain median" if _mn_base > 0\n'
     '                else "any motor neurons at all (domain median is 0.000%)")\n'
     '        mn_note = f" ({\', \'.join(hot)} carry {_how} -- check MN somata '
     'before calling these smear)"\n', 1),

    # the notebook's own trace counts is_MN with the NA-unsafe idiom. Two sites,
    # one on `sample` and one on `adata`, hence the capture group.
    ("trace_is_MN_na_safe", "regex",
     r"int\(_np2\.asarray\((\w+)\.obs\['is_MN'\]\)\.astype\(bool\)\.sum\(\)\)",
     r"int(__import__('pandas').Series(\1.obs['is_MN'])"
     r".astype('boolean').fillna(False).sum())", 2),
]


def fix_consumers_in_nb(cells):
    """Apply NB_FIXES to every code cell that is not the injected one."""
    counts = {}
    for label, kind, old, new, need in NB_FIXES:
        assert kind in ("literal", "regex"), f"{label}: bad kind {kind!r}"
        n = 0
        for c in cells:
            if c["cell_type"] != "code":
                continue
            s = "".join(c["source"])
            if MARKER in s:
                continue
            # count OCCURRENCES, not cells. `need` is declared in occurrences and
            # self_test() measures occurrences over the concatenated body, so
            # counting cells here made the two disagree: trace_is_MN_na_safe has
            # both of its hits in ONE cell, so n was 1 against need 2 and every
            # run aborted on the first sample while --self-test reported green.
            if kind == "regex":
                out, k = re.subn(old, new, s)
            else:
                k = s.count(old)
                out = s.replace(old, new)
            if out != s:
                c["source"] = src(out)
                n += k
        if n < need:
            raise RuntimeError(f"notebook consumer fix {label!r} matched {n} occurrences, "
                               f"expected at least {need}: {old[:70]!r}. The "
                               f"notebooks moved -- re-locate the site by hand "
                               f"instead of letting this pass be a no-op")
        counts[label] = n
    return {"nb_consumer_fixes": counts}


# ── consumer fixes in the helper scripts ────────────────────────────────────
# Same rule: every pattern must match, and each file is backed up once under
# BAK before it is touched. SENTINEL is what makes a second run a no-op.
SCRIPT_FIXES = {
    # ── promote_pass2.py: the v4real column contract, and a version bump so the
    # 20 stale v2 traces sitting on disk are REFUSED rather than promoted.
    "promote_pass2.py": ("is_gabaergic_neuron", [
        ('REQUIRED = ["is_MN", "is_MN_invh", "is_neuron", "is_neuron_prev",\n'
         '            "is_neuron_general", "is_GADvh", "is_GAD", "is_Interneuron",\n'
         '            "in_VH", "in_GM", "has_vh_data", "is_gabaergic", "neuron_class",\n'
         '            "cell_area_um2", "mn_signature_prob", "is_TDP", "tdp_proba"]\n'
         'RELABEL_VERSION = "v4"',
         'REQUIRED = ["is_MN", "is_MN_v6_asdelivered", "is_MN_invh", "is_MN_eligible",\n'
         '            "is_neuron_v3", "neuron_prob_v3", "is_neuron_prev",\n'
         '            "is_neuron_general", "is_GADvh", "is_GAD", "is_Interneuron",\n'
         '            "in_VH", "in_GM", "in_VH_not_GM", "has_vh_data", "vh_mask_empty",\n'
         '            "vh_source", "is_gabaergic_neuron", "neuron_class",\n'
         '            "cell_type_mw", "pathologist_class_mw", "STMN2_mw", "CE_STMN2_mw",\n'
         '            "cell_area_um2", "mn_signature_prob", "is_TDP", "tdp_proba",\n'
         '            "tdp_evaluated"]\n'
         'RELABEL_VERSION = "v6"'),
    ]),

    # ── concat_pass2_v4.py: same contract; refuse the stale is_neuron outright;
    # report is_neuron_v3 rather than the pre-v4 column the old summary read.
    "concat_pass2_v4.py": ("is_gabaergic_neuron", [
        ('MARCEL_COLS = ["is_MN", "is_MN_invh", "is_neuron", "is_neuron_prev",\n'
         '               "is_neuron_general", "is_GADvh", "is_GAD", "is_Interneuron",\n'
         '               "in_VH", "in_GM", "has_vh_data", "is_gabaergic", "neuron_class",\n'
         '               "cell_area_um2", "mn_signature_prob", "is_TDP", "tdp_proba"]',
         'MARCEL_COLS = ["is_MN", "is_MN_v6_asdelivered", "is_MN_invh", "is_MN_eligible",\n'
         '               "is_neuron_v3", "neuron_prob_v3", "is_neuron_prev",\n'
         '               "is_neuron_general", "is_GADvh", "is_GAD", "is_Interneuron",\n'
         '               "in_VH", "in_GM", "in_VH_not_GM", "has_vh_data", "vh_mask_empty",\n'
         '               "vh_source", "is_gabaergic_neuron", "neuron_class",\n'
         '               "cell_type_mw", "pathologist_class_mw", "STMN2_mw",\n'
         '               "CE_STMN2_mw", "cell_area_um2", "mn_signature_prob",\n'
         '               "is_TDP", "tdp_proba", "tdp_evaluated"]'),
        ('    meta = a.uns.get("sample_meta_v4")\n',
         '    # the stale pre-v4 column must be GONE, not merely unused: MARCEL_COLS\n'
         '    # used to name it, so the contract gate passed on it and the summary\n'
         '    # reported 9,290 neurons where v4 says 4,415.\n'
         '    if "is_neuron" in a.obs.columns:\n'
         '        raise SystemExit(f"{s}: obs still carries the stale is_neuron; this "\n'
         '                         f"is a pre-v6 object")\n'
         '    if a.uns.get("relabel_version") != "v6":\n'
         '        raise SystemExit(f"{s}: uns relabel_version is "\n'
         '                         f"{a.uns.get(\'relabel_version\')!r}, expected \'v6\'")\n'
         '    meta = a.uns.get("sample_meta_v4")\n'),
        ('    for k in ("cord_level", "disease_group", "patient_id"):',
         '    for k in ("cord_level", "disease_group", "patient_id", "vh_source"):'),
        ('    print(f"  {s:12s} {a.n_obs:>8,} cells  is_MN "\n'
         '          f"{int(a.obs[\'is_MN\'].fillna(False).sum()):>5}"\n'
         '          f" (+{int(a.obs[\'is_MN\'].isna().sum()):>5} NA)"\n'
         '          f"  {meta[\'disease_group\']}/{meta[\'cord_level\']}", flush=True)',
         '    _valid = bool(a.uns.get("mn_call_valid", True))\n'
         '    _mn = ("UNDETERMINED (no VH mask)" if not _valid\n'
         '           else f"{int(a.obs[\'is_MN\'].fillna(False).sum()):>5}")\n'
         '    print(f"  {s:12s} {a.n_obs:>8,} cells  is_MN {_mn}"\n'
         '          f" (+{int(a.obs[\'is_MN\'].isna().sum()):>6} NA)"\n'
         '          f"  {meta[\'disease_group\']}/{meta[\'cord_level\']}"\n'
         '          f"  {a.uns.get(\'vh_status\', \'?\')}", flush=True)'),
        ('        "relabel_version": "v4",\n'
         '        "parquet_snapshot": ("neuron_classification_tdp_v2__snapshot_"\n'
         '                             "20260818_0851.parquet"),\n'
         '        "mn_definition": ("is_neuron & ~is_GAD & in_VH & cell_area_um2 >= 250"),',
         '        "relabel_version": "v6",\n'
         '        "parquet_snapshot": ("neuron_classification_tdp_v6__snapshot_"\n'
         '                             "20260818_1649.parquet"),\n'
         '        "parquet_md5": "d32b298e9662e023cc1794367680a0ce",\n'
         '        "mn_definition": ("parquet is_MN is UNGATED in v6 "\n'
         '                          "(is_neuron_v3 & ~is_GAD & in_VH) and is carried "\n'
         '                          "as is_MN_candidate. obs[is_MN] here is that "\n'
         '                          "AND cell_area_um2 >= 250, unchanged from v5, so "\n'
         '                          "downstream meaning is stable. is_MN_a211 and "\n'
         '                          "is_MN_a250 carry the alternatives; the 250 floor "\n'
         '                          "has no literature anchor, see "\n'
         '                          "MN_AREA_THRESHOLD_RATIONALE.md"),\n'
         '        "undetermined_sections": ("SD01015_BG and SD01616_BI have no GM or "\n'
         '                                  "VH mask, so is_MN is <NA> for every cell "\n'
         '                                  "there. Never render them as 0 MN; ~52 "\n'
         '                                  "motor neurons are hidden, 8% of the pool."),'),
        ('    print(f"  is_neuron {int(A.obs[\'is_neuron\'].sum()):,}   "\n'
         '          f"is_TDP {int(A.obs[\'is_TDP\'].sum()):,}")',
         '    print(f"  is_neuron_v3 {int(A.obs[\'is_neuron_v3\'].sum()):,}   "\n'
         '          f"is_MN_eligible {int(A.obs[\'is_MN_eligible\'].sum()):,}   "\n'
         '          f"is_TDP {int(A.obs[\'is_TDP\'].sum()):,} of "\n'
         '          f"{int(A.obs[\'tdp_evaluated\'].sum()):,} evaluated")'),
        ('               "is_neuron": int(A.obs["is_neuron"].sum()),',
         '               "is_neuron_v3": int(A.obs["is_neuron_v3"].sum()),\n'
         '               "is_MN_eligible": int(A.obs["is_MN_eligible"].sum()),'),
    ]),

    # ── vh_visual_inspect.py: hard-fails on v4 (asks read_parquet for a column
    # that no longer exists), and its has_vh sentinel is the .notna() one.
    "vh_visual_inspect.py": ("20260818_1649", [
        ('        "neuron_classification_tdp_v2__snapshot_20260818_0851.parquet")',
         '        "neuron_classification_tdp_v6__snapshot_20260818_1649.parquet")'),
        ('                                    "in_VH", "is_MN", "cell_area_um2", "is_neuron",\n'
         '                                    "is_GAD", "cord_level", "disease_group"])',
         '                                    "in_VH", "is_MN", "cell_area_um2",\n'
         '                                    "is_neuron_v3", "is_GAD", "vh_source",\n'
         '                                    "vh_annotated", "cell_index",\n'
         '                                    "cord_level", "disease_group"])\n'
         '# No dedupe. v4 shipped 16 exactly-duplicated (sample, cell_index) rows and\n'
         '# Marcel fixed that in v5; v6 has zero, measured. A bare drop_duplicates()\n'
         '# here would have been far worse than the problem: without cell_index in the\n'
         '# frame, 43,798 cohort cells share an exact (0,0) centroid and a\n'
         '# cell_area_um2 of exactly 0.0, so it collapsed ~38,000 genuinely distinct\n'
         '# cells (up to 2.7% of a section) and silently shrank every denominator.\n'
         '# cell_index is read anyway so any future dedupe can be keyed on it.\n'
         '_n_dup_check = int(df.duplicated(subset=["sample", "cell_index"]).sum())\n'
         'assert _n_dup_check == 0, (\n'
         '    f"{_n_dup_check} duplicate (sample, cell_index) rows in v6; the "\n'
         '    f"dedupe regression is back, key any fix on cell_index")'),
        ('    has_vh = bool(d["in_VH"].notna().any())',
         '    # NOT .notna(): v4 wrote every boolean non-nullable, so .notna() is True\n'
         '    # for all 20 and "no mask" becomes "zero motor neurons". Measure it.\n'
         '    has_vh = bool(vh.sum() > 0 or gm.sum() > 0)'),
        ('    ttl = f"{ours}  {grp}/{cord}"',
         '    _src = str(d["vh_source"].iloc[0]) if len(d) else "unassigned"\n'
         '    ttl = f"{ours}  {grp}/{cord}  [{_src}]"'),
        ('           if has_vh else f"NO VH MASK   is_MN <NA> x{n_na}")',
         '           if has_vh else "NO VH MASK   in_GM 0 / in_VH 0   "\n'
         '                          "is_MN UNDETERMINED, not 0")'),
        ('fig.suptitle("v4 ventral-horn masks and motor-neuron calls  —  "\n'
         '             "neuron_classification_tdp_v2, snapshot 2026-08-18 08:51",',
         'fig.suptitle("v4 ventral-horn masks and motor-neuron calls  —  "\n'
         '             "neuron_classification_tdp_v6, snapshot 2026-08-18 16:49 UTC",'),
    ]),

    # ── plots_v4.py: the per-sample panel title rendered "MN 0" for the two
    # sections that have no mask, which is the one place a reader takes it as a
    # measurement.
    "plots_v4.py": ("no VH mask", [
        ('            mn = A.obs["is_MN"].fillna(False).to_numpy(dtype=bool)[m]\n'
         '            axx.scatter(xy[mn, 0], xy[mn, 1], s=9, c="#c0143c",\n'
         '                        edgecolors="white", linewidths=0.25, zorder=5)\n'
         '            axx.set_title(f"{s}  MN {int(mn.sum())}", fontsize=10,\n'
         '                          fontweight="bold")',
         '            mn = A.obs["is_MN"].fillna(False).to_numpy(dtype=bool)[m]\n'
         '            axx.scatter(xy[mn, 0], xy[mn, 1], s=9, c="#c0143c",\n'
         '                        edgecolors="white", linewidths=0.25, zorder=5)\n'
         '            # is_MN is <NA> for every cell in a section with no ventral-horn\n'
         '            # mask. fillna(False) above makes it plottable; printing the sum\n'
         '            # as a count would publish a missing mask as "0 motor neurons".\n'
         '            _hv = bool(A.obs["has_vh_data"].to_numpy(dtype=bool)[m].all())\n'
         '            _lbl = f"MN {int(mn.sum())}" if _hv else "MN n/a - no VH mask"\n'
         '            axx.set_title(f"{s}  {_lbl}", fontsize=10,\n'
         '                          fontweight="bold")'),
    ]),

    # ── cellannotator_v4.py: it filtered its protection set to the columns that
    # happened to be present, so a contract column that vanished was dropped
    # with no error and then went unprotected.
    "cellannotator_v4.py": ("refusing to run unprotected", [
        ('PROTECTED_EXTRA = ["cell_type", "leiden_1.0", "leiden_1.2", "leiden_1.5",\n'
         '                   "sample", "disease_group", "cord_level", "patient_id"]',
         'PROTECTED_EXTRA = ["cell_type", "cell_type_mw", "pathologist_class_mw",\n'
         '                   "vh_source", "leiden_1.0", "leiden_1.2", "leiden_1.5",\n'
         '                   "sample", "disease_group", "cord_level", "patient_id"]'),
        ('    protected = list(A.uns.get("marcel_columns", [])) + PROTECTED_EXTRA\n'
         '    protected = [c for c in dict.fromkeys(protected) if c in A.obs.columns]',
         '    # The contract columns must all be present. Filtering to "whatever is\n'
         '    # there" meant a renamed or dropped column silently left the protection\n'
         '    # set and then went unchecked. PROTECTED_EXTRA stays soft: leiden_1.x\n'
         '    # legitimately does not exist on the scVI object.\n'
         '    _contract = list(dict.fromkeys(A.uns.get("marcel_columns", [])))\n'
         '    _absent = [c for c in _contract if c not in A.obs.columns]\n'
         '    if _absent:\n'
         '        sys.exit(f"uns[\'marcel_columns\'] names {_absent}, which are not in "\n'
         '                 f"obs -- refusing to run unprotected on a partial contract")\n'
         '    protected = list(dict.fromkeys(_contract + PROTECTED_EXTRA))\n'
         '    protected = [c for c in protected if c in A.obs.columns]'),
    ]),

    # ── the notebook GENERATORS still emit the NA-unsafe idiom, so regenerating
    # from source would reintroduce it. run_dualpass.py's copy lives inside an
    # f-string template, hence __import__ rather than a new import line: the
    # replacement must contain no braces.
    "fix_notebooks_at_source.py": ("__import__('pandas')", [
        ("    __TRACE__['n_MN_final'] = int(_np2.asarray(sample.obs['is_MN'])"
         ".astype(bool).sum())",
         "    __TRACE__['n_MN_final'] = int(__import__('pandas')"
         ".Series(sample.obs['is_MN']).astype('boolean').fillna(False).sum())"),
        ("    __TRACE__['n_MN_raw'] = int(_np2.asarray(adata.obs['is_MN'])"
         ".astype(bool).sum())",
         "    __TRACE__['n_MN_raw'] = int(__import__('pandas')"
         ".Series(adata.obs['is_MN']).astype('boolean').fillna(False).sum())"),
    ]),
    "run_dualpass.py": ("__import__('pandas')", [
        ("    __TRACE__['n_MN_final'] = int(_np2.asarray(sample.obs['is_MN'])"
         ".astype(bool).sum())",
         "    __TRACE__['n_MN_final'] = int(__import__('pandas')"
         ".Series(sample.obs['is_MN']).astype('boolean').fillna(False).sum())"),
        ("    __TRACE__['n_MN_raw'] = int(_np2.asarray(adata.obs['is_MN'])"
         ".astype(bool).sum())",
         "    __TRACE__['n_MN_raw'] = int(__import__('pandas')"
         ".Series(adata.obs['is_MN']).astype('boolean').fillna(False).sum())"),
    ]),

    # ── check_gad.py is a throwaway diagnostic that hard-fails on the renames
    # (and had in_VH spelled in_vh).
    "check_gad.py": ("is_neuron_v3", [
        ('    cand = df[df["is_neuron"] & (df["in_vh"] == True)].copy()\n'
         '    print(f"parquet: is_neuron {int(df[\'is_neuron\'].sum())} | "',
         '    cand = df[df["is_neuron_v3"] & (df["in_VH"] == True)].copy()\n'
         '    print(f"parquet: is_neuron_v3 {int(df[\'is_neuron_v3\'].sum())} | "'),
    ]),
}


def patch_scripts(dry=False):
    """Apply SCRIPT_FIXES. A pattern that does not match is an error, not a skip."""
    out = {}
    for name, (sentinel, pairs) in SCRIPT_FIXES.items():
        p = CODEDIR / name
        if not p.exists():
            out[name] = "MISSING -- not patched"
            continue
        s0 = p.read_text()
        if sentinel in s0:
            out[name] = "already v4real"
            continue
        s = s0
        for old, new in pairs:
            if old not in s:
                raise RuntimeError(
                    f"{name}: pattern not found, so this fix would be a silent "
                    f"no-op. Re-locate it by hand:\n  {old[:160]!r}")
            s = s.replace(old, new, 1)
        try:
            compile(s, str(p), "exec")
        except SyntaxError as e:
            raise RuntimeError(f"{name}: patched text does not compile: {e}")
        if not dry:
            bak = Path(str(p) + BAK)
            if not bak.exists():
                shutil.copy2(p, bak)
            p.write_text(s)
        out[name] = f"{len(pairs)} patches" + (" (dry run)" if dry else "")
    # smear_gate.py was made NA-safe by the v2 run and its MN logic was A/B
    # tested against v4 (0 flag changes on all 20). Left alone on purpose.
    sg = (CODEDIR / "smear_gate.py").read_text()
    out["smear_gate.py"] = ("already NA-safe, MN logic A/B-tested against v4, "
                            "not touched"
                            if 'astype("boolean")' in sg or "astype('boolean')" in sg
                            else "NOT NA-safe -- fix before running")
    # deck_figs2.py needs no change BECAUSE of the <NA> contract above: its
    # `mn_valid.any()` gate reads the is_MN null mask, which is now all-False on
    # exactly the two sections with no ventral-horn mask, so it prints
    # "is_MN <NA> no VH mask" for them and a count everywhere else.
    out["deck_figs2.py"] = ("correct as written once is_MN is <NA> on mask-less "
                            "sections; not touched")
    return out


def self_test():
    """Render and statically check every injected cell. Touches no notebook."""
    seen = set()
    for s in SAMPLES:
        body = render(CELL, s)
        extra = render(TRACE_EXTRA, s)
        for text, label in ((body, "relabel"), (extra, "trace")):
            assert "@@" not in text, f"{s}:{label} has an unrendered placeholder"
            compile(text, f"<{s}:{label}>", "exec")
            preflight(text, f"{s}:{label}")
        seen.add(PARQUET_KEY.get(s, s))
    assert len(seen) == len(SAMPLES), "PARQUET_KEY maps two samples to one key"
    assert sum(EXPECTED_NATIVE_MN.values()) == 2577, "native MN table changed"
    assert set(EXPECTED_NATIVE_MN) == set(SAMPLES), "native MN table is incomplete"
    assert set(EXPECTED_DUPS) <= set(SAMPLES), "EXPECTED_DUPS names a non-cohort key"

    # Confirm every notebook consumer-fix site still exists, read-only, against
    # the pristine backups. A pattern that has drifted must be found here rather
    # than half-way through a 20-notebook run.
    for s in SAMPLES:
        ref = Path(str(find_nb(s)) + RESTORE_FROM)
        if not ref.exists():
            print(f"  WARNING {ref.name} missing -- cannot check fix sites")
            continue
        body = "\n".join("".join(c.get("source", []))
                         for c in json.loads(ref.read_text())["cells"]
                         if c["cell_type"] == "code")
        for label, kind, old, _new, need in NB_FIXES:
            n = (len(re.findall(old, body)) if kind == "regex"
                 else body.count(old))
            if n < need:
                raise RuntimeError(f"{s}: fix site {label!r} found {n} times, "
                                   f"need {need}")
        for m in STALE_MARKERS + (MARKER,):
            assert m not in body, f"{ref.name} already carries {m!r}"
        assert PREREQ_MARKER in body, f"{ref.name} lacks {PREREQ_MARKER!r}"
    print(f"  all {len(NB_FIXES)} notebook fix sites present in all "
          f"{len(SAMPLES)} pristine backups")

    for name, (sentinel, pairs) in SCRIPT_FIXES.items():
        p = CODEDIR / name
        if not p.exists():
            print(f"  WARNING {name} missing")
            continue
        s0 = p.read_text()
        if sentinel in s0:
            print(f"  {name}: already v4real, fixes would be skipped")
            continue
        miss = [old[:70] for old, _ in pairs if old not in s0]
        if miss:
            print(f"  {name}: {len(pairs) - len(miss)}/{len(pairs)} patterns "
                  f"present  MISSING {miss}")
            continue
        # apply in memory and compile, so a broken replacement string is caught
        # here rather than half-way through the real run
        s = s0
        for old, new in pairs:
            s = s.replace(old, new, 1)
        compile(s, str(p), "exec")
        print(f"  {name}: {len(pairs)}/{len(pairs)} patterns present, "
              f"patched text compiles")
    print(f"self-test OK: {len(SAMPLES)} samples rendered, compiled and "
          f"module-attribute resolved")
    print(f"  marker         {MARKER}")
    print(f"  version        {RELABEL_VERSION}")
    print(f"  parquet        {PARQUET}")
    print(f"  md5            {PARQUET_MD5}")
    print(f"  restore from   <nb>{RESTORE_FROM}")
    print(f"  fresh backup   <nb>{BAK}")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    dry = "--dry-run" in sys.argv
    todo = [a for a in sys.argv[1:] if not a.startswith("--")] or SAMPLES
    unknown = [s for s in todo if s not in SAMPLES]
    if unknown:
        raise SystemExit(f"not cohort samples: {unknown}")
    for s in todo:
        rep = patch(find_nb(s), s, dry)
        note = ""
        if s in KNOWN_NO_VH:
            note = " [no GM/VH mask -> is_MN <NA> for every cell, NOT 0]"
        elif s in EXPECTED_DUPS:
            note = f" [{EXPECTED_DUPS[s]} duplicate parquet rows dropped]"
        print(f"{s:12s} {rep}{note}", flush=True)
    for k, v in patch_scripts(dry).items():
        print(f"  {k:32s} {v}")
    print(f"\n{len(todo)} notebooks{' (dry run)' if dry else ''}")
    print("Next: rerun the array, then promote_pass2.py (it now demands "
          f"relabel version {RELABEL_VERSION!r}, so the stale v2 traces are refused).")


if __name__ == "__main__":
    main()
