# COHORT — GROUND TRUTH (sample metadata)
> SOURCE OF TRUTH for donor identity, disease group, sex/age/fixation/dv200, and **spinal level**.
> Parsed directly from `ALSsamples_TR19.24 final table.xlsx` (TR19/24 JCK, Cooper-Knock Lab) on 2026-07-08.
> This file OVERRIDES any level/demographic claim elsewhere (kernel, scratch pad, older memories, the abstract).
> The spreadsheet is the SELECTION table: each donor has 3 candidate blocks (Lumbar SC = BI, Cervical SC = BG,
> Brain BA4 = AD); the block flagged `x` (ALS) or `y` (control) is the one profiled on Xenium.

## CLASSIFICATION NOTE
- Spreadsheet lists **SD016/16 under C9ORF**, but the project applies a verified override **C9 → SPORADIC**.
  Effective working split = **Sporadic 6 / C9 4 / Control 10** (sheet-literal = Sporadic 5 / C9 5 / Control 10).
- ALS6 FFPE **brain** runs reuse C9 donors SD016/20 + SD020/22 = SEPARATE experiment (different region/preservation); NOT part of the 20-sample SC cohort.
- Region_2 (unknown donor in the 2026-03-31 PDF) = **SD054/13** (F, 51, cervical) — identity assigned later; matches exactly.
- SD028/18 + SD054/13 were "selected but not run" as of 2026-03-31; both later recovered into the run cohort (SD028/18 via June-2026 large-cell reseg; SD054/13 = Region_2).

## PER-DONOR TABLE (effective 20-sample SC cohort)
| Donor | Group | Sex | Age | Fix(d) | dv200 | Selected block | **Spinal level** |
|-------|-------|-----|-----|--------|-------|----------------|------------------|
| SD019/23 | Sporadic | M | 56 | 3 | 97 | BI | Lumbar |
| SD016/23 | Sporadic | F | 59 | 4 | 7  | BI | Lumbar |
| SD035/22 | Sporadic | M | 52 | 2 | 14 | BG | **Cervical** |
| SD026/22 | Sporadic | F | 70 | 2 | 8  | BI | Lumbar |
| SD019/22 | Sporadic | M | 66 | 1 | 9  | BI | Lumbar (worst QC — flag/sensitivity-only) |
| SD016/16 | Sporadic* | M | 58 | 1 | 14 | BI | Lumbar (*override C9→sporadic) |
| SD020/22 | C9ORF72 | F | 52 | 1 | 6  | BI | Lumbar |
| SD016/20 | C9ORF72 | F | 70 | 2 | 13 | BI | Lumbar |
| SD013/20 | C9ORF72 | F | 56 | 3 | 13 | BG | **Cervical** |
| SD042/19 | C9ORF72 | M | 66 | 4 | 6  | BI | Lumbar |
| SD028/18 | Control | F | 53 | 3 | 9  | BG | Cervical |
| SD019/15 | Control | M | 50 | 4 | 12 | BG | Cervical |
| SD011/15 | Control | M | 57 | 4 | 11 | BG | Cervical |
| SD010/15 | Control | F | 57 | 4 | 6  | BG | Cervical |
| SD039/14 | Control | M | 60 | 2 | 15 | BG | Cervical |
| SD036/14 | Control | M | 51 | 3 | 16 | BG | Cervical |
| SD006/14 | Control | M | 60 | 4 | 6  | BG | Cervical |
| SD054/13 | Control | F | 51 | 3 | 8  | BG | Cervical (= Region_2) |
| SD029/13 | Control | M | 58 | 3 | 7  | BA(block) | Cervical (elevated neg-ctrl 0.0193) |
| SD014/13 | Control | F | 74 | 2 | 13 | BA(block) | Cervical |

## THE TWO ABSTRACT-BREAKING FACTS (fix before submission)
1. **SPINAL LEVEL × DISEASE CONFOUND.** ALS = **8 Lumbar + 2 Cervical**; Controls = **10 Cervical (all)**.
   The abstract says "cervical spinal cord of 10 ALS and 10 controls" — TRUE only for controls; 8/10 ALS are LUMBAR.
   Cervical vs lumbar cords differ in MN pool size/density/somatotopy/ALS-vulnerability → every ALS-vs-control
   MN/CE contrast is entangled with a level effect. **Only level-matched ALS cases = SD035/22 (sporadic) + SD013/20 (C9)**
   → run these vs the cervical controls as the within-cervical sensitivity check for any ALS-vs-control claim.
2. **FIXATION.** Abstract says "<72 hours in formalin". Ground truth fixation runs 1–4 days; **6 samples are 4 days = 96 h > 72 h**
   (SD016/23, SD042/19, SD019/15, SD011/15, SD010/15, SD006/14); fix=3 samples are exactly 72 h (excluded by "<72h").
   The "<72 h" claim is not accurate for the cohort as run.

## DEMOGRAPHIC SUMMARY (effective split)
- Sporadic (6): 4M / 2F; age 52–70.
- C9ORF72 (4): 1M / 3F (female-skewed) — note for any sex-stratified analysis; age 52–70.
- Control (10): 6M / 4F; age 50–74.
- ALS total 5M/5F vs Control 6M/4F.

## dv200 caveat (from 2026-03-31 cross-corr)
dv200 does NOT predict Xenium yield here (SD019/23 dv200=97 but worst profile; SD026/22 dv200=8 excellent). Do not gate on dv200.
