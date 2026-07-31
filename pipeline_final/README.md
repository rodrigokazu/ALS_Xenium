# pipeline_final

The current pipeline, running on Marcel's `_final` segmentation (`Ranger_procd_mw_final`,
delivered 2026-07-19). Deployed on SCG at
`/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/final_rerun_code/`.

This directory is flat because it has to be. Twenty-five of these scripts add their own
directory to `sys.path` and import siblings by bare name, so moving any of them into a
subfolder breaks imports at runtime. The grouping below is the map that the filesystem cannot
give you.

## Shared modules

Imported by nearly everything. Change these carefully.

- `final_config.py`: paths, the 20-sample discovery and dedup, the per-sample loader, the
  grid offset. Single source of truth.
- `mn_annotation_join.py`: the `is_MN` join, with the `cell_index + 1` key fix and the
  row-count provenance guard.
- `nature_style.py`: figure style, colourblind palette, scalebars, PDF plus PNG output.
- `ps_io_FINAL_fix.py`: h5py-level loader for the independent niche h5ads, with CSR/CSC dispatch.

## Stage 1, build the cohort object

- `VH_concatenate_FINAL.py`, `run_VH_concatenate_FINAL.sh`

## Stage 2, spatial domains

- `novae_JOINT_FINAL.py`, `run_novae_JOINT_FINAL.sh`: one model over all 20 slides, domains
  comparable across sections. Use for composition.
- `novae_INDEPENDENT_FINAL.py`, `run_novae_INDEPENDENT_FINAL.sh`: one model per sample, array
  0-19. QC only, domains not comparable.

## Stage 2b, astrocyte isolation and the tSNE sweep

- `astro_sanity_FINAL.py`, `run_astro_sanity_FINAL.sh`
- `tsne_sweep_prep_FINAL.py`, `run_tsne_sweep_prep_FINAL.sh`
- `tsne_sweep_run_FINAL.py`, `run_tsne_sweep_FINAL.sh`: 24 parameter combinations

## Stage 2c, coarse cell typing

- `celltype_pipeline_FINAL.py`, `run_ct_pipe_FINAL.sh`: clustering and lineage calls
- `run_opentsne_FINAL.py`, `run_ct_tsne_FINAL.sh`: cohort tSNE from the saved PCA
- `celltype_plots_FINAL.py`, `run_ct_plots_FINAL.sh`: combined figures and composition
- `celltype_dotplots_persample_FINAL.py`, `run_ct_dotpersample_FINAL.sh`: the only source of
  per-sample dotplots
- `hero_celltype_FINAL.py`, `run_ct_heroes_FINAL.sh`: cell-type close-ups

All of these write to the same output directory so `X_pca.npy`, `X_tsne.npy`, the typed h5ad
and `obs_celltype.csv` stay row-aligned. Do not redirect one of them.

## Stage 3, dual-pass QC

- `dpqc_compute_FINAL.py`: the scoring engine and the only file that decides anything
- `dpqc_figures_FINAL.py`, `dpqc_style_FINAL.py`: figures and the colour registries
- `dpqc_persample_FINAL.py`, `dpqc_report_FINAL.py`: one-sample driver and its PDF
- `dpqc_cohort_FINAL.py`, `dpqc_meeting_FINAL.py`, `dpqc_audio_FINAL.py`,
  `dpqc_strategy_FINAL.py`: cohort roll-up, meeting guide, audio edition, method white paper
- `qc_basic_FINAL.py`, `run_basicqc_all.sh`: the classic per-niche QC page
- `run_dpqc_array_FINAL.sh`, `run_dpqc_aggregate_FINAL.sh`, `run_dpqc_test_FINAL.sh`
- `build_merged_report.py`, `explain_report_FINAL.py`: merged PDF and the glossary

## Stage 4, imaging overlays and co-localisation

- `xenium_overlay_FINAL.py`: domains on DAPI, plus the loaders the rest build on
- `hero_crop_FINAL.py`: motor neuron close-ups
- `run_overlays_batch_FINAL.py`, `run_overlays_FINAL.sh`: batch rendering, array 0-19
- `gene_coloc_hero_FINAL.py`, `run_gene_coloc_FINAL.sh`: MNX1, BCL6 and STMN2, array 0-2
- `gene_coloc_compare_fig_FINAL.py`: the three genes compared across sections
- `matched_triple_panel_FINAL.py`: all three over one identical field

## Orchestration

- `SUBMIT_ALL_FINAL.sh`: the whole dependency graph. Run with `--dry-run` first.
- `check_annotation_ready.sh`: the gate in front of a full launch.
- `SUBMIT_WHEN_UP_FINAL.sh`: waits for the cluster, then submits.

## What is gated on motor neuron annotation

`is_MN` is False everywhere until Marcel's motor neuron stage lands. These degrade to honest
placeholders in the meantime: the hero crops, the gene co-localisation quantification and its
comparison figure, the matched triple panel, and the MN protection inside the dual-pass QC.
The cell typing splits motor neurons on MNX1 alone and records `mn_split_method` as
`MNX1-only` so nobody mistakes it for a real call.

Run `SUBMIT_ALL_FINAL.sh --domains-only` for the half that does not need the annotation.
