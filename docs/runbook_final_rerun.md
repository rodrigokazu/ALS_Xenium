# _FINAL RE-RUN — LAUNCH RUNBOOK

> **STATUS 2026-07-22 — DOMAINS-ONLY LAUNCHED (jobs 52199989–52200003).** Gates 2 (perms) CLEARED, 4 (run_ct_
> naming) FIXED; gate 1 partial (neuron layer landed, is_MN pending — off-by-one caught + patched + Marcel-
> confirmed, join-key CLOSED); gate 3 Region_2 boundary regenerated, other 19 cosmetic-pending. CONCAT
> COMPLETED clean (20 samples / 1,854,568 cells / join_overlap=1.0). To run the is_MN half when Marcel's MN
> stage lands: **Launch path 2** below (drop `--domains-only`). History below kept as the reusable procedure.

> "When Marcel's annotation lands, do X." Everything is STAGED; nothing has run. Code on OAK (HOME quota full):
> `/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/`. Full design: memory
> `project_mnfinal_pipeline_staging`; paths in `project_config.md` (_FINAL RE-RUN block).
> SSH only when `souls/ssh_cluster.flag` reads `connected` (DUO 2FA — never initiate).

## Pre-flight — clear the launch gates (none is a code defect)

### Gate A — `_final` raw-file permissions (CRITICAL; concat fails 3/20 today)
Owner-only (`-rwx------`, mw28) files break the loader on 3 KEPT samples: SD05413_BG (Region_2), SD020_22_BI,
SD03914_BG (cells.parquet/cells.csv.gz +/- barcodes/matrix). (SD01620_BI is also owner-only but is an
EXCLUDED duplicate-donor original — ignore.) Re-scan readability across all kept dirs:
```
ssh scg 'cd /oak/.../RK/Spatial/Ranger_procd_mw_final && for d in */ ; do d=${d%/}; \
  for f in "$d"/cell_feature_matrix/barcodes.tsv.gz "$d"/cell_feature_matrix/matrix.mtx.gz "$d"/cells.parquet "$d"/cells.csv.gz ; do \
    [ -r "$f" ] || echo "UNREADABLE: $f" ; done ; done'
```
If any KEPT sample prints: ask **Marcel** to `chmod g+r` those files (rodrigok cannot chmod mw28's files) OR
re-export group-readable. NFS-squash: OAK group is `nobody` -> g+r (644) works, o+r does NOT. Do not submit until clean.

### Gate B — is_MN annotation on `_final` (THE #1 tracked item)
Confirm Marcel produced the annotation ON `_final` (not `_gauss`) per sample, EITHER form:
`Ranger_procd_mw_final/<sample>/*_cell_classification_MN.csv` (+`*_MN_TDP.csv`) OR `CellAnnot/clustered_<LABEL>_res0.5.h5ad`.
```
ssh scg 'cd /oak/.../RK/Spatial/final_rerun_code && bash check_annotation_ready.sh'
```
- exit 0 (READY) = all 20 have annotation -> FULL launch.
- exit 1 (NOT READY) = prints missing list -> wait, or run `--domains-only` (below).
- JOIN KEY — CONFIRMED 2026-07-22 (was the #1 open item, now CLOSED). Marcel's annotation is keyed on a
  0-BASED `cell_index`; `_final` cells.parquet `cell_id` is the 1-BASED string -> **`cell_id == cell_index + 1`**
  (Marcel verified across ALL 22 samples vs features.csv; README has a "Join key" section). `mn_annotation_join.py`
  is PATCHED to emit a `cell_index+1` candidate (see `_candidate_source_keys`) — the +1 is applied automatically;
  the future is_MN/mn_prob column carries the same 0-based key so nothing changes. The N1 row-count guard still
  runs (necessary-not-sufficient); after concat, check the log for `is_MN_join_key='cell_index+1'`,
  is_MN ~150-320/sample, and `is_MN_provenance='row-count-ok'`. Still get Marcel's WRITTEN "built on _final"
  confirmation as the human backstop.

### Gate C — coarse cell-typing (RESOLVED 2026-07-19)
`celltype_*_FINAL.py` + `run_ct_*_FINAL.sh` are staged and compile-clean. At launch with is_MN pending they run
the MNX1-only PROVISIONAL MN split (uns['mn_split_method']='MNX1-only'); supersede + validate vs the nuclear MN
mask (G14) once is_MN lands. No `--skip-celltyping` needed anymore.

### Gate D — boundaries (polygon/hero FIGURES only; does NOT block coloc numbers)
`cell_/nucleus_boundaries` on `_final` are ~10-13% coverage + OLD gm-namespace. Filled-polygon "Xenium-Explorer"
figures need Marcel to re-export full-coverage integer-id boundaries. Until then boundary figures fall back to a
VALID cell-type/domain-coloured CENTROID map (guard auto-re-enables polygons with zero code change once fixed).
`transcripts.parquet` IS integer-namespace + complete, so the BCL6/MNX1/STMN2 coloc QUANTIFICATION runs fine now.

## Launch path 1 — domains + QC NOW (annotation not required)
After Gate A is clean. Runs concat -> Novae (JOINT+INDEP) -> astro+tSNE -> coarse typing -> dual-pass QC + PDFs,
with is_MN degraded to False (MN outputs emit honest placeholders / MNX1-only provisional).
```
ssh scg 'cd /oak/.../RK/Spatial/final_rerun_code && bash SUBMIT_ALL_FINAL.sh --domains-only --dry-run'   # preview
ssh scg 'cd /oak/.../RK/Spatial/final_rerun_code && bash SUBMIT_ALL_FINAL.sh --domains-only'
```
Jobids captured in `SUBMITTED_JOBS.txt`.

## Launch path 2 — FULL one-command (after Gates A+B clear)
```
ssh scg 'cd /oak/.../RK/Spatial/final_rerun_code && bash check_annotation_ready.sh && bash SUBMIT_ALL_FINAL.sh --dry-run'
ssh scg 'cd /oak/.../RK/Spatial/final_rerun_code && bash SUBMIT_ALL_FINAL.sh'
```
FULL refuses to submit unless `check_annotation_ready.sh` passes. `--no-astro-tsne` skips the 24-run sweep.
DAG: concat -> parallel {Novae JOINT + INDEP array + astro + celltyping} -> overlays/coloc + dpqc -> 4 PDFs.

## Submit order if driving by hand
```
C=$(sbatch --parsable run_VH_concatenate_FINAL.sh)
sbatch --dependency=afterok:$C run_novae_JOINT_FINAL.sh
I=$(sbatch --parsable --dependency=afterok:$C run_novae_INDEPENDENT_FINAL.sh)   # array 0-19%5 (+afterany aggregate in footer)
A=$(sbatch --parsable --dependency=afterok:$C run_astro_sanity_FINAL.sh)
P=$(sbatch --parsable --dependency=afterok:$A run_tsne_sweep_prep_FINAL.sh); sbatch --dependency=afterok:$P run_tsne_sweep_FINAL.sh
T=$(sbatch --parsable --dependency=afterok:$C run_ct_pipe_FINAL.sh); TT=$(sbatch --parsable --dependency=afterok:$T run_ct_tsne_FINAL.sh)
sbatch --dependency=afterok:$TT run_ct_plots_FINAL.sh; sbatch --dependency=afterok:$TT run_ct_dotpersample_FINAL.sh
sbatch run_dpqc_test_FINAL.sh; Q=$(sbatch --parsable run_dpqc_array_FINAL.sh); sbatch --dependency=afterany:$Q run_dpqc_aggregate_FINAL.sh
sbatch run_overlays_FINAL.sh                # array 0-19%6 (centroid fallback until boundaries re-exported)
sbatch run_gene_coloc_FINAL.sh             # array 0-2 = MNX1/BCL6/STMN2 ; then gene_coloc_compare_fig_FINAL.py  [is_MN-gated]
# hero_celltype + filled-polygon overlays: HOLD until Gate D (boundaries) cleared
```

## Monitoring
- `ssh scg 'squeue -u rodrigok'` ; logs `/oak/.../final_rerun_code/logs/%x_%j.{out,err}` ; jobids `SUBMITTED_JOBS.txt`.

## Expected outputs per stage
- Concat -> `SC_MNcorrected_FINAL_h5ads/ALS_SCXenium_MNcorrected_FINAL_concatenated_offset_allsamples_preQC_20260719.h5ad`
  + per-sample. Watch log: kept==20, split control10/sporadic6/c9 4, panel==480, join overlap ~1.0000/sample, is_MN readback dtype==bool.
- Novae JOINT -> `Novae_persample_niches_FINAL/`; INDEP -> `Novae_persample_INDEPENDENT_FINAL/per_sample/*__niches_independent.h5ad` (+ aggregate).
- Astro -> `astro_isolation_FINAL/` (+ tSNE sweep 24 runs). Coarse typing -> `CellTyping_coarse_FINAL/` (obs_celltype.csv + UMAP/tSNE/dotplots).
- Dual-pass QC -> `Novae_persample_INDEPENDENT_FINAL/dualpass_qc/` = Cohort/Meeting/Audio/Strategy PDFs (+ listening.io .txt/.pdf).
- Overlays/coloc -> under the Novae dirs; centroid maps until boundaries re-exported; coloc CSVs/figs valid now.

## Post-run
Fold any number into the manuscript ONLY with script path + log path + date, after @ADDRESSER verification.
Run the close-out sync (project memory + crew memory + mirror to jarvis_migration) per the standing directive.
