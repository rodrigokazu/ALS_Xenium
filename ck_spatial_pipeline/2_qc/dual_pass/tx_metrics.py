#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/dual_pass/tx_metrics.py
#
# Transcript-level QC straight from transcripts.parquet: % of reads at QV>=20, % of reads
# assigned to cells, and where the unassigned transcripts sit. cell_id == 0 marks an
# unassigned read. Report only. The % assigned column fails the Salas benchmark in all 20
# sections (median 33.8%), which points at segmentation under-capture.
# ========================================================================================

"""
Transcript-level QC for one sample, straight off transcripts.parquet.

The Novae h5ads carry only per-cell aggregates, so the two headline rows of the
LatchBio QC decision trace -- %reads QV>=20 and %reads assigned to cells -- cannot
be computed from them. This reads the Xenium Ranger transcript table instead.

Also runs the course's stated diagnostic for a low assignment rate: map the
unassigned transcripts. If they sit where the tissue is (rather than in
background), the segmentation is missing cells rather than the slide being empty.

usage: tx_metrics.py <SAMPLE>
"""

import json, os, sys
import numpy as np
import pyarrow.parquet as pq

RANGER = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Ranger_procd_mw_final"
OUT    = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_IND_QC/runs/tx"

# sample -> Ranger directory, taken from obs['run_id'] in each h5ad (NOT guessed
# from the sample name: three of them disagree with the naming convention).
RUNDIR = {
    "SD00614_BG": "SD00614_BG_", "SD01015_BG": "SD01015_BG", "SD01115_BG": "SD01115_BG",
    "SD01320_BG": "SD01320_BG",  "SD01413_BA": "SD01413_BA", "SD01616_BI": "SD01616_BI",
    "SD01620_BI": "SD016_20_BI", "SD01623_BI": "SD01623_BI", "SD01915_BG": "SD01915_BG",
    "SD01922_BI": "SD01922_BI",  "SD01923_BI": "SD01923_BI", "SD02022_BI": "SD020_22_BI",
    "SD02622_BI": "SD02622_BI",  "SD02818_BG": "SD02818_BG_", "SD02913_BA": "SD02913_BA",
    "SD03522_BG": "SD03522_BG",  "SD03614_BG": "SD03614_BG", "SD03914_BG": "SD03914_BG",
    "SD04219_BI": "SD04219_BI",  "SD05413_BG": "Region_2",
}

QV_MIN = 20.0
BIN_UM = 50.0          # grid for the unassigned-transcript diagnostic


def main():
    s = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    d = os.path.join(RANGER, RUNDIR[s])
    p = os.path.join(d, "transcripts.parquet")
    res = {"sample": s, "ranger_dir": d}

    if not os.path.exists(p):
        res["ERROR"] = "transcripts.parquet missing"
        json.dump(res, open(f"{OUT}/tx_{s}.json", "w"), indent=1)
        print(f"[{s}] MISSING {p}")
        return

    xj = os.path.join(d, "experiment.xenium")
    if os.path.exists(xj):
        j = json.load(open(xj))
        res["experiment"] = {k: j.get(k) for k in
                             ["run_name", "region_name", "num_cells", "num_transcripts",
                              "num_transcripts_high_quality", "fraction_transcripts_assigned",
                              "transcripts_per_cell", "region_area", "panel_name",
                              "analysis_sw_version", "preservation_method"]}

    t = pq.read_table(p, columns=["cell_id", "qv", "codeword_category",
                                  "x_location", "y_location", "overlaps_nucleus"])
    cid = t.column("cell_id").to_numpy()
    qv = t.column("qv").to_numpy()
    cat = np.asarray(t.column("codeword_category").to_pylist())
    n = len(cid)

    assigned = cid != 0                      # cell_id == 0 is the unassigned sentinel
    hq = qv >= QV_MIN
    is_neg = np.char.startswith(cat.astype(str), "negative_control")
    is_gene = cat == "custom_gene"

    res["n_transcripts"] = int(n)
    res["pct_qv20"] = round(100 * float(hq.mean()), 3)
    res["pct_assigned"] = round(100 * float(assigned.mean()), 3)
    res["pct_assigned_qv20"] = round(100 * float(assigned[hq].mean()), 3)
    res["pct_neg_control"] = round(100 * float(is_neg.mean()), 5)
    res["pct_overlaps_nucleus"] = round(
        100 * float(t.column("overlaps_nucleus").to_numpy().astype(bool).mean()), 3)
    res["median_qv"] = round(float(np.median(qv)), 3)

    # signal-to-noise: real-gene transcripts per negative-control transcript
    n_neg = int(is_neg.sum())
    res["n_neg_control"] = n_neg
    res["snr_transcript_level"] = round(float(is_gene.sum()) / max(n_neg, 1), 1)

    # ---- the LatchBio diagnostic: where do the unassigned transcripts sit? ----
    x = t.column("x_location").to_numpy()
    y = t.column("y_location").to_numpy()
    xb = ((x - x.min()) // BIN_UM).astype(np.int64)
    yb = ((y - y.min()) // BIN_UM).astype(np.int64)
    nx, ny = int(xb.max()) + 1, int(yb.max()) + 1
    flat = yb * nx + xb
    tot = np.bincount(flat, minlength=nx * ny).astype(float)
    una = np.bincount(flat[~assigned], minlength=nx * ny).astype(float)
    asg = np.bincount(flat[assigned], minlength=nx * ny).astype(float)

    # Two definitions of "tissue", because the obvious one is circular: unassigned
    # transcripts contribute to `tot`, so a bin containing nothing but unassigned
    # transcripts still counts as tissue under a total-density rule.
    tissue = tot >= 20                       # circular version, kept for comparison
    cellular = asg >= 5                      # non-circular: bins where cells WERE found
    res["n_bins"] = int(tissue.sum())
    res["n_cellular_bins"] = int(cellular.sum())
    res["bin_um"] = BIN_UM
    if tissue.sum() > 10:
        r = float(np.corrcoef(np.log1p(asg[tissue]), np.log1p(una[tissue]))[0, 1])
        res["corr_assigned_vs_unassigned_density"] = round(r, 4)
        res["median_unassigned_frac_in_tissue_bins"] = round(
            float(np.median(una[tissue] / tot[tissue])), 4)
        res["pct_unassigned_in_tissue_bins"] = round(
            100 * float(una[tissue].sum() / max(una.sum(), 1)), 2)
    if cellular.sum() > 10:
        # THE non-circular statement: what share of unassigned transcripts sit in
        # bins that already contain successfully segmented cells? High = the
        # segmentation is standing right next to the signal it failed to claim.
        res["pct_unassigned_in_cellular_bins"] = round(
            100 * float(una[cellular].sum() / max(una.sum(), 1)), 2)
        # and how lopsided is it inside those bins?
        res["median_unassigned_frac_in_cellular_bins"] = round(
            float(np.median(una[cellular] / tot[cellular])), 4)
        # background check: unassigned sitting where NO cell was found at all
        res["pct_unassigned_in_acellular_bins"] = round(
            100 * float(una[~cellular].sum() / max(una.sum(), 1)), 2)
    json.dump(res, open(f"{OUT}/tx_{s}.json", "w"), indent=1)
    print(f"[{s}] qv20={res['pct_qv20']}% assigned={res['pct_assigned']}% "
          f"snr={res['snr_transcript_level']} "
          f"unassigned_in_tissue={res.get('pct_unassigned_in_tissue_bins')}%")


if __name__ == "__main__":
    main()
