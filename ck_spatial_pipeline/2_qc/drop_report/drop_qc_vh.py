#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/drop_report/drop_qc_vh.py
#
# Asks whether the drop's ALS versus control imbalance also holds inside the ventral horn.
# Cohort-wide the drop removes 22.7% of C9 cells, 16.8% of sporadic and 4.3% of control.
# Section-level tests: Mann-Whitney between groups and Wilcoxon signed-rank for VH against
# non-VH within a section.
# ========================================================================================

"""Does the v3 exclusion's ALS-vs-control disparity ALSO happen inside the VH mask?

drop_qc.py established that the drop is 5x heavier on ALS cohort-wide (22.7% C9 /
16.8% sporadic vs 4.3% control). That is intended -- niche v1_3's disease association
is a fragmentation artefact at the tissue edge. The question here is whether the same
disparity reaches INSIDE the ventral horn, because every MN and VH composition result
is computed there. If VH is spared, VH analyses survive the drop; if it is not, VH
composition before and after are no more comparable than the cohort numbers.

DESIGN -- why this is a SECTION-LEVEL analysis
  A section is either control or ALS, so status is nested in section and a
  section-stratified MH/Fisher for an ALS-vs-control contrast is not available:
  pooling cells across donors is pseudo-replication, and that is exactly what
  invalidated the pooled Fisher on the sub-7 VH question (Tarone p=0.0001).
  So: each section contributes ONE VH drop rate and ONE non-VH drop rate.
    * between groups (control vs ALS)  -> Mann-Whitney on section rates
    * VH vs non-VH within a section    -> Wilcoxon signed-rank, PAIRED, valid
  Cell-level percentages are reported as descriptives only, never as a test.

VH evaluability: in_VH_v2 is NA on sections with no mask -- those are excluded from
every VH number and named in the output. Cf. vh_masks_v2 (SD02913_BA open question).
"""
import os, json, numpy as np, pandas as pd, h5py
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B   = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial"
OLD = f"{B}/NicheCompass_SC/runs_v3_noN3/NC_v2_Leiden_nichev1tagged.h5ad"
OUT = f"{B}/NicheCompass_SC/runs_v3_noN3"
log = lambda *a: print(*a, flush=True)

def cat(f, c):
    g = f["obs"][c]
    cc = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
    return np.array(cc, dtype=object)[g["codes"][:]].astype(str)
def nbool(f, c):
    g = f["obs"][c]; v = g["values"][:].astype(bool); m = g["mask"][:].astype(bool)
    o = np.empty(len(v), dtype=object); o[:] = v; o[m] = None; return o

log("reading obs")
with h5py.File(OLD, "r") as f:
    ns    = cat(f, "sample");   stat = cat(f, "status")
    dropr = cat(f, "dropped_v3"); v1  = cat(f, "niche_v1")
    invh  = nbool(f, "in_VH_v2"); mn  = nbool(f, "is_MN_v2")
    vhsrc = cat(f, "vh_v2_source") if "vh_v2_source" in f["obs"] else None
n = len(ns)
drop = dropr != "retained"
VH   = np.array([x is True  for x in invh])
NVH  = np.array([x is False for x in invh])
EVAL = VH | NVH
isMN = np.array([x is True for x in mn])
log(f"  {n:,} cells | dropped {int(drop.sum()):,} ({100*drop.mean():.2f}%)")
log(f"  VH evaluable {int(EVAL.sum()):,} ({100*EVAL.mean():.1f}%) | "
    f"in VH {int(VH.sum()):,} | outside VH {int(NVH.sum()):,} | NA {int((~EVAL).sum()):,}")

secs = sorted(set(ns))
noeval = [s for s in secs if not EVAL[ns == s].any()]
log(f"  sections with NO VH mask (excluded from all VH numbers): {noeval if noeval else 'none'}")

# ------------------------------------------------- 1. descriptive, cell level
log("\n=== cell-level descriptives (NOT a test -- donors are the unit) ===")
rows = []
for grp, m_grp in [("control", stat == "control"), ("sporadic", stat == "sporadic"),
                   ("c9", stat == "c9"), ("ALL ALS", stat != "control"),
                   ("COHORT", np.ones(n, bool))]:
    for zone, m_zone in [("in VH", VH), ("outside VH", NVH), ("all evaluable", EVAL)]:
        m = m_grp & m_zone
        if not m.any(): continue
        rows.append(dict(group=grp, zone=zone, cells=int(m.sum()),
                         dropped=int((m & drop).sum()),
                         pct_dropped=round(100 * (m & drop).mean() / max(m.mean(), 1e-12), 2)))
D = pd.DataFrame(rows)
D["pct_dropped"] = (100 * D.dropped / D.cells).round(2)
log(D.pivot(index="group", columns="zone", values="pct_dropped").to_string())
D.to_csv(f"{OUT}/drop_qc_vh_celllevel.csv", index=False)

# ------------------------------------------------- 2. per section, the real unit
per = []
for s_ in secs:
    m = ns == s_
    if not (m & EVAL).any(): continue
    mv, mo = m & VH, m & NVH
    r = dict(sample=s_, status=stat[m][0],
             n_vh=int(mv.sum()), n_nonvh=int(mo.sum()),
             drop_vh=int((mv & drop).sum()), drop_nonvh=int((mo & drop).sum()),
             pct_vh=round(100 * (mv & drop).sum() / max(mv.sum(), 1), 2),
             pct_nonvh=round(100 * (mo & drop).sum() / max(mo.sum(), 1), 2),
             n_MN=int((m & isMN).sum()), MN_dropped=int((m & isMN & drop).sum()))
    r["vh_minus_nonvh"] = round(r["pct_vh"] - r["pct_nonvh"], 2)
    r["ratio_vh_over_nonvh"] = round(r["pct_vh"] / r["pct_nonvh"], 3) if r["pct_nonvh"] > 0 else np.nan
    per.append(r)
P = pd.DataFrame(per)
P["als"] = P.status != "control"
P.to_csv(f"{OUT}/drop_qc_vh_per_sample.csv", index=False)
log("\n=== per section (the unit of analysis) ===")
log(P.drop(columns=["als"]).to_string(index=False))

# ------------------------------------------------- 3. the tests
log("\n=== TESTS ===")
ctl, als = P[~P.als], P[P.als]
res = {}
def mw(a, b, lab):
    if len(a) < 2 or len(b) < 2: return None
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    log(f"  {lab}: control median {np.median(b):.2f}% (n={len(b)}) vs "
        f"ALS median {np.median(a):.2f}% (n={len(a)})  U={u:.0f}  p={p:.4g}")
    return dict(als_median=float(np.median(a)), control_median=float(np.median(b)),
                n_als=len(a), n_control=len(b), U=float(u), p=float(p))

log("\n between groups (Mann-Whitney on section rates):")
res["vh_als_vs_control"]    = mw(als.pct_vh.values,    ctl.pct_vh.values,    "drop rate INSIDE VH ")
res["nonvh_als_vs_control"] = mw(als.pct_nonvh.values, ctl.pct_nonvh.values, "drop rate OUTSIDE VH")

log("\n VH vs non-VH within a section (Wilcoxon signed-rank, paired):")
for lab, sub in [("all sections", P), ("control only", ctl), ("ALS only", als)]:
    a, b = sub.pct_vh.values, sub.pct_nonvh.values
    if len(sub) < 3 or np.all(a == b): continue
    w, p = stats.wilcoxon(a, b)
    log(f"  {lab:<14} VH median {np.median(a):.2f}% vs non-VH {np.median(b):.2f}%  "
        f"W={w:.0f}  p={p:.4g}  | VH lower in {int((a < b).sum())}/{len(sub)} sections")
    res[f"paired_{lab.replace(' ','_')}"] = dict(
        vh_median=float(np.median(a)), nonvh_median=float(np.median(b)),
        W=float(w), p=float(p), n=len(sub), n_vh_lower=int((a < b).sum()))

log("\n the interaction -- is the ALS gap SMALLER inside VH?")
gap_vh    = np.median(als.pct_vh)    - np.median(ctl.pct_vh)
gap_nonvh = np.median(als.pct_nonvh) - np.median(ctl.pct_nonvh)
log(f"  ALS-minus-control gap  inside VH {gap_vh:+.2f} pts | outside VH {gap_nonvh:+.2f} pts")
d_als, d_ctl = als.vh_minus_nonvh.values, ctl.vh_minus_nonvh.values
u, p = stats.mannwhitneyu(d_als, d_ctl, alternative="two-sided")
log(f"  per-section (VH - non-VH) difference: ALS median {np.median(d_als):+.2f} vs "
    f"control {np.median(d_ctl):+.2f}  U={u:.0f}  p={p:.4g}")
res["interaction"] = dict(gap_inside_vh_pts=float(gap_vh), gap_outside_vh_pts=float(gap_nonvh),
                          als_median_vh_minus_nonvh=float(np.median(d_als)),
                          control_median_vh_minus_nonvh=float(np.median(d_ctl)),
                          U=float(u), p=float(p))

# ------------------------------------------------- 4. what gets dropped in VH
log("\n=== inside VH, WHAT is dropped? ===")
mvh = VH & drop
rr = pd.Series(dropr[mvh]).value_counts()
log((100 * rr / VH.sum()).round(3).to_frame("% of all VH cells").join(rr.to_frame("cells")).to_string())
log("\n  by v1 niche, inside VH:")
tv = pd.crosstab(pd.Series(v1[VH], name="niche_v1"), pd.Series(drop[VH], name="dropped"))
tv["pct_dropped"] = (100 * tv.get(True, 0) / tv.sum(1)).round(2)
log(tv.to_string())
tv.to_csv(f"{OUT}/drop_qc_vh_by_niche.csv")

log("\n=== motor neurons (is_MN_v2) ===")
log(f"  MN total {int(isMN.sum()):,} | dropped {int((isMN & drop).sum()):,} "
    f"({100*(isMN & drop).mean()/max(isMN.mean(),1e-12):.2f}%)")
if (isMN & drop).any():
    log("  dropped MN by reason:\n" + pd.Series(dropr[isMN & drop]).value_counts().to_string())

# ------------------------------------------------- 5. VH cohort shift
log("\n=== does the VH cohort composition shift? ===")
sh = []
for lab, m in [("before", EVAL & VH), ("after", EVAL & VH & ~drop)]:
    v = pd.Series(stat[m]).value_counts(normalize=True) * 100
    sh.append(v.rename(lab))
SH = pd.concat(sh, axis=1).round(2); SH["delta_pts"] = (SH["after"] - SH["before"]).round(2)
log("  VH only:\n" + SH.to_string())
sh2 = []
for lab, m in [("before", np.ones(n, bool)), ("after", ~drop)]:
    sh2.append((pd.Series(stat[m]).value_counts(normalize=True) * 100).rename(lab))
SH2 = pd.concat(sh2, axis=1).round(2); SH2["delta_pts"] = (SH2["after"] - SH2["before"]).round(2)
log("  whole cohort (for reference):\n" + SH2.to_string())
SH.to_csv(f"{OUT}/drop_qc_vh_cohort_shift.csv")

# ------------------------------------------------- 6. figure
fig, ax = plt.subplots(2, 2, figsize=(14.5, 9.4), facecolor="white")
C_CTL, C_ALS = "#0072B2", "#D55E00"
x = np.arange(len(P))
cols = [C_ALS if a else C_CTL for a in P.als]

a1 = ax[0, 0]
a1.bar(x - 0.2, P.pct_nonvh, 0.4, color="#b9c2bf", label="outside VH")
a1.bar(x + 0.2, P.pct_vh, 0.4, color=cols, label="inside VH (blue ctrl / orange ALS)")
a1.set_xticks(x); a1.set_xticklabels(P["sample"], rotation=90, fontsize=7.5)
a1.set_ylabel("% of cells dropped")
a1.set_title("Per section: the drop inside vs outside the VH mask", fontsize=12.5, loc="left")
a1.legend(frameon=False, fontsize=9)
for s in ("top", "right"): a1.spines[s].set_visible(False)

a2 = ax[0, 1]
for i, (lab, sub) in enumerate([("control", ctl), ("ALS", als)]):
    for _, r in sub.iterrows():
        a2.plot([i - 0.14, i + 0.14], [r.pct_nonvh, r.pct_vh], "-o", ms=4, lw=1.1,
                color=(C_CTL if lab == "control" else C_ALS), alpha=0.75)
a2.set_xticks([0, 1]); a2.set_xticklabels(["control", "ALS"])
a2.set_xlim(-0.5, 1.5); a2.set_ylabel("% of cells dropped")
a2.text(0.5, 0.97, "left point = outside VH, right = inside VH", transform=a2.transAxes,
        ha="center", va="top", fontsize=9, color="#444")
a2.set_title("Paired within section — does VH track the section?", fontsize=12.5, loc="left")
for s in ("top", "right"): a2.spines[s].set_visible(False)

a3 = ax[1, 0]
pp = res.get("vh_als_vs_control"); qq = res.get("nonvh_als_vs_control")
bx = a3.boxplot([ctl.pct_vh, als.pct_vh, ctl.pct_nonvh, als.pct_nonvh],
                positions=[0, 1, 2.6, 3.6], widths=0.62, patch_artist=True)
for b, c in zip(bx["boxes"], [C_CTL, C_ALS, C_CTL, C_ALS]):
    b.set_facecolor(c); b.set_alpha(0.55)
for med in bx["medians"]: med.set_color("#111")
for i, (v, c) in enumerate([(ctl.pct_vh, C_CTL), (als.pct_vh, C_ALS),
                            (ctl.pct_nonvh, C_CTL), (als.pct_nonvh, C_ALS)]):
    xx = [0, 1, 2.6, 3.6][i]
    a3.scatter(np.random.default_rng(i).normal(xx, 0.055, len(v)), v, s=16, color=c,
               edgecolor="white", linewidth=0.5, zorder=3)
a3.set_xticks([0, 1, 2.6, 3.6]); a3.set_xticklabels(["ctrl", "ALS", "ctrl", "ALS"])
a3.set_ylabel("% of section's cells dropped")
for xx, lab in [(0.5, "inside VH"), (3.1, "outside VH")]:
    a3.text(xx, -0.085, lab, transform=a3.get_xaxis_transform(), ha="center",
            va="top", fontsize=10.5)
ttl = "Section-level, the honest unit"
if pp and qq:
    ttl += f" — VH p={pp['p']:.3g}, non-VH p={qq['p']:.3g}"
a3.set_title(ttl, fontsize=12.5, loc="left")
for s in ("top", "right"): a3.spines[s].set_visible(False)

a4 = ax[1, 1]
o = P.sort_values("vh_minus_nonvh")
a4.barh(np.arange(len(o)), o.vh_minus_nonvh,
        color=[C_ALS if a else C_CTL for a in o.als])
a4.axvline(0, color="#111", lw=1.1)
a4.set_yticks(np.arange(len(o))); a4.set_yticklabels(o["sample"], fontsize=7.5)
a4.set_xlabel("VH drop % minus non-VH drop %  (negative = VH spared)")
nlow = int((P.vh_minus_nonvh < 0).sum())
a4.set_title(f"VH is spared relative to its own section in {nlow}/{len(P)} sections",
             fontsize=12.5, loc="left")
for s in ("top", "right"): a4.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(f"{OUT}/drop_qc_vh.png", dpi=145, facecolor="white")
log(f"\nwrote {OUT}/drop_qc_vh.png")

json.dump({"design": "section-level; status is nested in section so a section-stratified "
                     "ALS-vs-control test is unavailable. Cell-level % are descriptive only.",
           "n_cells": int(n), "n_vh": int(VH.sum()), "n_nonvh": int(NVH.sum()),
           "n_vh_na": int((~EVAL).sum()), "sections_without_mask": noeval,
           "cell_level_pct": D.to_dict("records"), "tests": res,
           "vh_cohort_shift_pts": SH["delta_pts"].to_dict(),
           "cohort_shift_pts": SH2["delta_pts"].to_dict(),
           "MN_total": int(isMN.sum()), "MN_dropped": int((isMN & drop).sum())},
          open(f"{OUT}/drop_qc_vh.json", "w"), indent=2)
log("wrote drop_qc_vh.json + drop_qc_vh_per_sample.csv + drop_qc_vh_celllevel.csv")
