# Cooper-Knock Spatial Pipeline

The code that took the 20 ALS spinal cord Xenium sections from Marcel's `_final`
segmentation to the objects behind the Spinal Cord Xenium Explorer and the Niche DE and
COZI Explorer. It is organised by tool: Novae, QC, cell annotation, NicheCompass and COZI.
Every script was pulled from SCG on 2026-09-23 and carries a commentary header. Read the
header before you run anything.

## The run, in the order it happened

| When | Tool | Step | Canonical output on OAK |
|---|---|---|---|
| 07-19 | `0_cohort_build` | Concatenate 20 MN-corrected sections, 1,854,568 cells pre-QC | `SC_MNcorrected_FINAL_h5ads/` |
| 07-22 to 07-25 | `1_novae` | One fresh Novae model per section, domains at n4, n6, n8, n10 | `Novae_persample_INDEPENDENT_FINAL/per_sample/` |
| 07-25 | `2_qc/domain_verdicts` | Score every n6 domain for smear, call KEEP, REVIEW or REMOVE | `dualpass_qc/extracted/tables/<S>__decisions.csv` |
| 08-06 | `1_novae` | Join Marcel's July neuron call onto the Novae objects | same per_sample h5ads, edited in place |
| 08-16 to 08-18 | `2_qc/dual_pass` | Patch 20 notebooks at source, run the dual-pass QC, verify, promote | `Ranger_procd/ALS_SCXenium_<TAG>_pass2.h5ad`, 1,097,669 cells |
| 08-18 | `2_qc/dual_pass` (notebook cells 51-52) | Base `cell_type` by marker score argmax, 9 classes | inside pass2 |
| 08-25 | `4_nichecompass` | Stage gene programmes, check how many survive the 480 panel | `NicheCompass_SC/gene_programs/` |
| 09-02 | `3_cell_annotation/motor_neurons` | Attach ventral-horn v2, make `is_MN_v2` (844 cells) the canonical `is_MN` | pass2, edited in place |
| 09-02 | `4_nichecompass` | Base integrated model, 1,097,669 cells, `latent_leiden_0.3` | `runs_v2/integrated_whole/20260902_ncv2/` |
| 09-04 | `3_cell_annotation/audit_v2_v3` | Audit and re-cluster Other neurons into 12 sub-clusters | `cell_audit/otherneurons_cells.csv.gz` |
| 09-04 | `5_cozi` | COZI and niche DE on the base model (what the COZI Explorer shows) | `cozi_whole/`, `cozi_vh/` |
| 09-09 | `3_cell_annotation/audit_v2_v3` | `cell_type_v2`: 48,063 cells move, Fibroblast added. v3 flags tested | `cell_audit/annotation_v2/` |
| 09-14 to 09-15 | `4_nichecompass` | Drop niches 3, 7, 8 and junk subs 2, 9, 10, retrain, label `niche_v2` | `runs_v3_noN3/NC_v3_Leiden_nichev2sub.h5ad`, 965,285 cells |
| 09-15 | `2_qc/drop_report` | Report what the drop did to depth, cohort-wide and inside the VH | `runs_v3_noN3/drop_qc*` |
| 09-15 to 09-17 | `5_cozi` | COZI on the canonical object, whole and VH, then networks | `cozi_als/20260915_022205_ncv3`, `cozi_nets/` |

The canonical object today is `NicheCompass_SC/runs_v3_noN3/NC_v3_Leiden_nichev2sub.h5ad`:
965,285 cells, `cell_type_v2` (11 classes), `niche_v2` v2_0 to v2_6, `niche_v2_sub` (v2_1 splits
in three), `is_MN_v2`, `in_VH_v2` and the NicheCompass latent.

## Folders

| Folder | What it holds |
|---|---|
| `common/` | `final_config`, `mn_annotation_join`, `nature_style`, `ps_io_FINAL_fix`, `jck_palette`. Imported by bare name. |
| `orchestration/` | `SUBMIT_ALL_FINAL.sh`, the July dependency graph for concat, Novae, coarse typing, verdicts and overlays. |
| `0_cohort_build/` | `VH_concatenate_FINAL`. The only step before the five tools. |
| `1_novae/` | Independent and joint Novae fits, the neuron join and the report-only AUG5 driver. |
| `2_qc/domain_verdicts/` | The dpqc family and the basic QC page. |
| `2_qc/dual_pass/` | Source patchers, runner, array, gates, promote, transcript QC and the 20 notebooks (outputs stripped). |
| `2_qc/drop_report/` | Depth and VH reports on the September drop. |
| `3_cell_annotation/coarse_reference/` | The July coarse typing. Its scores are evidence for the v2 moves, and its labels are not the base `cell_type`. |
| `3_cell_annotation/motor_neurons/` | The `is_MN_v2` attach and canonicalisation. |
| `3_cell_annotation/audit_v2_v3/` | Cell audit, `cell_type_v2`, the v3 tests and their corrections. |
| `4_nichecompass/` | Gene-programme staging, preflight, trainer, downstream, the three v3 steps and the env builder. |
| `5_cozi/` | Env builders, COZI, networks, niche DE and figures. |
| `side_imaging_overlays/` | Domains and MN close-ups on DAPI, gene co-localisation. Hangs off the joint Novae fit. |
| `side_astro_tsne_sweep/` | Astrocyte sanity checks and the 24-combination tSNE sweep. |
| `deploy/` | `manifest.tsv` and `deploy.sh`. |

## Two places where annotation lives outside its folder

The base `cell_type` comes from cells 51 and 52 of the dual-pass notebooks: `score_genes` on
`layers['log1p_norm']` for each marker set with at least two genes on the panel, then argmax.
`cell_type_mn` comes from `build_cell_type_mn()` in `4_nichecompass/NC_v2_train.py`: the
coarse Motor neurons class becomes Other neurons, then every `is_MN_v2` cell becomes Motor
neurons. Both stay where they ran, because moving them would break the notebooks and the
trainer.

## Deploying

The scripts carry absolute SCG paths, and cloning the repo does not give a runnable tree.
On SCG the code runs from flat folders (`final_rerun_code/`, `SLURM_jobs/Spatial/`,
`dualpass_rerun/`, `NicheCompasso_v2/`, `NicheDE/`, `VHv2/`). `deploy/manifest.tsv` records
the SCG folder, file name and md5 at ship time for every file here. Run
`deploy/deploy.sh` for a dry run that labels each file `same`, `shipped`, `DRIFTED` or
`absent`, then `--apply` to copy. A drifted file changed on SCG after 2026-09-23. Pull it
back into the repo before you overwrite it.

Two paths are load-bearing. Notebook cell 43 imports `smear_gate` from
`/home/rodrigok/SLURM_jobs/Spatial/dualpass_rerun`, and the Novae scripts insert
`final_rerun_code` into `sys.path` by absolute path.

## Left out on purpose

- The CellAnnotator LLM branch (`dualpass_rerun/cellannotator_v6/v7`, `fix_ca_labels`,
  `mn_override_ca`, `reannotate_lowconf`, `stage2_embed_v6`). It only ever wrote to
  `cohort_v6/ALS_SCXenium_cohort_v6_scvi.h5ad`, the canonical object carries no `ca_*`
  column and it rests on the stale 774-cell `is_MN_v1`. The July prototype sits in
  `archive/cellannotator_prototype/`.
- NicheCompass v1 (`NicheCompasso/` apart from the gene-programme staging) and the v2
  per-sample and VH configurations.
- The buggy first step 3 (`nc3_step3_subcluster.py`, `run_nc3_step3.sh`, `run_split_inventory.sh`).
- Marcel's `niche_seg/phase4_cozi.py`. It computes a COZI-like statistic on one crop with its
  own code and says so in its docstring.
- Probes and one-off diagnostics (`probe_*`, `count_*`, `inspect_*`, `nc_api_inspect*`,
  `nc_section6`, `niche_recon`, `junk_check`, `rev_*`, the `*.bak*` files).
- `mn_tdp_analysis.py`. It depends on `is_TDP`, which is unusable.

## Things that will bite you

- **Domain ids are per section** in the independent Novae run. Domain 3 in one section and
  domain 3 in another are unrelated. Use the joint fit to compare composition.
- **Niche numbering changes with every run.** The MN niche is 6 in NicheCompass v1, 1 in the
  20260902 base run and v2_5 in the canonical object. Check `niche_numbering.json`.
- **`X` is not counts** in the Novae per-sample h5ads. The counts sit in `layers['counts']`.
  That mistake once removed 98% of cells.
- **The Explorer COZI tabs show the 09-04 run** on the 1,097,669-cell base object. The
  canonical 965,285-cell runs are the 09-15 ones.
- **Spinal level confounds disease.** All ten controls are cervical and eight of ten ALS
  cases are lumbar.
