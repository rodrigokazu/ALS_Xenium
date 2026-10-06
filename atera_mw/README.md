# atera_mw: resegmentation and first analysis of the 10x Atera human cerebellum section

Handoff from Marcel, 6 October 2026. Everything here was run on Marcel's Mac (12 cores, 96 GB RAM) between 29 September and
5 October 2026. The data lives on Marcel's external SSD; section 6 lists every output path and what to copy to SCG. The
detailed method write-up is in [METHODS.md](METHODS.md). This file is the practical guide: what was done, what each
script does, which numbers came out, what to trust and what not.

## 1. Short version

- **Data.** One pre-production 10x Atera section of adult human cerebellum (AT1183_S4C), whole-transcriptome panel of
  18,050 genes, 3.42 billion decoded transcripts (3.01 billion pass qv >= 20 and are genes), 510,556 cells in the 10x
  (Xenium Ranger) segmentation. Shared by 10x on 29 September 2026 as pre-production data with "unresolved cell
  segmentation issues".
- **What we built.** Our spinal-cord segmentation pipeline (`paper_reseg_pipeline/splitmerge/rules.py`, unchanged) adapted
  to Atera: the 18S stain is combined with a transcript-density image, the tissue mask comes from transcript density,
  growth is gated by the local cell type read from the transcripts (I2), and every Purkinje soma is rebuilt as one cell
  (Purkinje repair). An optional final step takes glial, vascular and immune cells from Ranger (hybrid C).
- **Main result (Purkinje cells, large neurons).** Across 139 Purkinje somata in nine 500 µm windows, scored with
  Purkinje genes the method never saw: 0.77 of each soma's transcripts in a single cell against 0.51 for Ranger (median
  0.94 against 0.60; Wilcoxon p ~ 5e-20); somata split across >= 2 cells 7% against 28% (McNemar p ~ 5e-6); somata with
  no cell holding >= 10%: 13% against 22% (p ~ 2e-4). For the same nuclei, our Purkinje cells hold 1.46x the transcripts
  of the Ranger cells. Basket / stellate interneurons also improve (capture 0.92 against 0.85).
- **Overall.** Coverage 87.4% of QC transcripts in cells against 85.4% (whole section). Marker mixing (MECR) is level with
  Ranger. Glial cells are captured slightly worse than by Ranger, which is why the hybrid exists.
- **ResolVI** (Ergen & Yosef, Nature Methods 2026) is complementary, not a competitor: it removes misassigned transcripts
  after segmentation (about 21-25% less marker mixing on both segmentations) but cannot rejoin a split soma.
- **What was used for the MNDA / poster figure.** The original pipeline (without the hybrid step and without ResolVI):
  `compare/purkinje/Atera_future_work_MNDA_v2.pdf`, script `scripts/mnda_figure.py`.
- **Biology (showcase only, one section).** Purkinje dendritic mRNA ranking of 41 Purkinje-specific genes, replicates
  between section halves (rho 0.91); GJD2 (connexin 36) confined to basket / stellate cells. Zebrin bands: not supported.
  Rodrigo and Marcel agreed not to put new biology in the proposal or poster.

## 2. Folder layout

```
atera_mw/
  README.md                this file
  METHODS.md               detailed methods (paper style), all parameters
  config/                  gene sets, whole-sample statistics, window coordinates, stage-1 calibration
  run_scripts/             shell drivers (window runs, full-section run, ResolVI runs)
  scripts/                 every Python script used (flat folder; scripts import each other by name)
  scripts/baselines/       Cellpose / Baysor / mini-bundle helpers for the baseline comparison
  paper_reseg_pipeline/splitmerge/   snapshot of rules.py + metrics.py as used (imported via parents[1])
```

The scripts keep the absolute paths of Marcel's machine: `SSD = /Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB` (data and
outputs) and `RUNS = /private/tmp/claude-501/.../atera_local` (500 µm window runs). Replace those two roots to rerun. The
shared rules come from `paper_reseg_pipeline/splitmerge/` and were not edited for Atera; the copy here is the version used.

Environments: everything runs in the `spatial` conda env (repo `environment.yml`) except ResolVI and openTSNE, which run
in a separate env `resolvi` (python 3.11, `scvi-tools 1.4.2`, `scanpy`, `openTSNE 1.0.4`) so the shared env stays
untouched.

## 3. The pipeline, step by step (final version)

| Step | What it does | Script(s) | Key settings |
|---|---|---|---|
| 0 | Transcript-density image of the whole section (all QC transcripts, Gaussian sigma 0.5 µm, tx / µm², zarr, 4096 chunks) | `make_density_image.py` | sigma 0.5 µm |
| 0 | Whole-sample statistics (tissue percentiles of 18S, density, DAPI; computed once, never per crop) | `run_fullsample.py --stats-only` | `config/stats_density.json` |
| 1 | Base segmentation: `rules.segment_region` on a combined 18S + density signal, density tissue mask, seeded by 10x nuclei, split / merge rules, glial somata | `run_fullsample.py` (`--combined`, `--density`, `--density-tissue-thr 10`) | combined 18S (bg 48, hi 941, cut 0.459) + density (bg 5.815, hi 177.18, cut 0.793); `thr_value 0.5`, `thr_alt 0.6`, `glia_soma_thr_value 0.2`, `glia_soma_max_um 8` ("C500") |
| 2 | I2, identity-gated growth: free pixels go to a cell only if the local transcript composition confidently matches the cell's type; density watershed among same-type cells | `phase3.py` (windows), `full_post.py i2` (full section) | lfc 2, mean 0.3, sigma 2 µm, bins 4 px, conf 2.0 (log-likelihood gap), low-confidence pixels stay free, reach 4 µm (14 µm Purkinje / interneuron), density floor 40 tx / µm², min 30 typed transcripts |
| 3 | Purkinje repair: every Purkinje soma becomes one whole cell (also without a nucleus) | `pc_soma.py`, `pc_repair.py` (windows), `full_post.py pcr` | set-A Purkinje-gene density >= 0.534 tx / µm² (Otsu over all windows), 18S body (median >= 0.2), elongation <= 3, split at 2+ peaks >= 12 µm apart, grow <= 4 µm into 18S, protect foreign nuclei, absorb pieces >= 60% inside |
| 4 | Counts: QC transcripts (qv >= 20, genes) under the final labels -> cell x gene matrix | `full_post.py count` | |
| 5 (optional) | Hybrid C: cells our annotation calls glial / vascular / immune and that carry no Purkinje signal are replaced by the Ranger cells | `hybrid_seg.py` (windows), `full_post.py hybc` (full section) | Purkinje set-A share < 0.0164 (midpoint 10x cluster 33 vs 13) |
| 6 | QC, Leiden, UMAP, marker annotation | `atera_annot.py` (`SEG_DIR=` to choose the segmentation) | QC: >= 200 tx, >= 100 genes, >= 10 µm²; Leiden res 2.0 |
| 7 (optional) | ResolVI on the whole section (latent + corrected counts) | `resolvi_full.py` (env `resolvi`) | defaults, 50 epochs, batch 1024, 3,361 genes |

Full-section driver: `run_scripts/run_full_section.sh` (chunks: `make_chunks.py`, `run_chunk.sh`; stitching and post-steps:
`full_post.py stitch | i2 | pcr | count | hybc`). The base run took about 4 h for 147 chunks (7 parallel workers), I2 20 min,
Purkinje repair 7 min, counts 25 min; whole-section ResolVI 9.3 h on CPU.

## 4. Results with numbers

### 4.1 Whole section

| | Ranger (10x) | Ours (segmentation_full) | Ours + hybrid C |
|---|---|---|---|
| Cells | 510,556 | 507,376 | 507,543 |
| QC transcripts in cells | 2,573.7 M / 3,014.0 M (85.4%) | 2,631.0 M / 3,009.6 M (87.4%) | 2,585.2 M / 3,009.6 M (85.9%) |
| Median transcripts / genes per cell | 4,787 / 2,996 | 4,817 / 3,011 | 4,726 / 2,970 |
| Purkinje cells, same 10x nuclei (cluster 33) | 13,650 tx (822 nuclei) | 24,059 tx (523 matched), x1.46 | 25,812 tx (501 matched), x1.48 |
| Gene-defined Purkinje cells (>= 4.3% Purkinje genes) | | 945 | 969 |
| Nucleus-free Purkinje cells created by the repair | | 157 (169 in the repair log) | |

Per cell type, for the same nuclei, transcripts per cell are within x0.93-1.06 of Ranger except Purkinje cells
(`segmentation_full/transcripts_per_cell_by_type.csv`).

### 4.2 Benchmark on 500 µm windows (`config/windows_500.txt`)

Spot1 and spot2 were used for tuning; spot3 and held1-6 are held out. All metrics on the same QC transcripts; MECR =
mutually exclusive co-expression rate over the fixed marker panel of `splitmerge/metrics.py`; MECR@1000 = the same with
every cell cut to 1,000 random transcripts.

Held-out windows (mean of 7):

| | Coverage | MECR | MECR@1000 |
|---|---|---|---|
| Ranger | 89.9% | 0.0491 | 0.0166 |
| Ours (original pipeline) | 91.5% (7/7 better, p = 0.016) | 0.0490 | 0.0148 (6/7, p = 0.031) |
| Ours + hybrid C | 89.7% (equal) | 0.0474 (6/7, p = 0.031) | 0.0156 |

Purkinje somata (139 somata, 9 windows; scored with the held-out set-B Purkinje genes, soma from set A):

| | Capture (mean / median) | Split >= 2 cells | No cell >= 10% |
|---|---|---|---|
| Ranger | 0.51 / 0.60 | 28% | 22% |
| Ours | 0.77 / 0.94 | 7% | 13% |
| Ours + hybrid C | 0.77 / 0.94 | 7% | 13% |

Every 10x nucleus with >= 20 own-type specific transcripts (`compare/celltype_capture/`), capture in one cell (median):

| Type | n | Ranger | Ours | Hybrid C |
|---|---|---|---|---|
| Purkinje | 37 | 0.93 | 1.00 | 1.00 |
| Basket / stellate (MLI) | 123 | 0.85 | 0.92 | 0.92 |
| Astroglia | 825 | 0.75 | 0.69 | 0.72 |
| Oligodendroglia | 1,027 | 0.79 | 0.75 | 0.76 |
| Vascular | 429 | 0.81 | 0.78 | 0.77 |
| Immune | 288 | 0.88 | 0.85 | 0.85 |

Splits: astroglia 19% / 24% / 20%, oligodendroglia 29% / 36% / 25% (Ranger / ours / hybrid C).

### 4.3 ResolVI (`compare/resolvi/`)

ResolVI run with standard settings (unsupervised, 50 epochs) on each segmentation separately, 3,361 genes.

- Original pipeline vs Ranger, 9 benchmark windows: MECR 0.0498 -> 0.0356 (Ranger) and 0.0495 -> 0.0339 (ours). The
  difference after ResolVI was not significant (6/9), and on 20 additional random windows it vanished (27 windows:
  +0.2%, p = 0.91).
- Final pipeline (hybrid C) vs Ranger, 27 windows (20 random + 7 held out): before ResolVI ours is cleaner (-2.6%, 23/27,
  Wilcoxon p = 1.4e-4); after ResolVI level (+2.1%, 12/27, p = 0.34). ResolVI reduces MECR by 24.5% (Ranger) and 20.9%
  (ours).
- Caveat: part of every MECR gain from cleaning is mechanical, because MECR counts a single stray marker transcript as
  co-expression. After ResolVI the share of cells with any Purkinje-gene signal falls from 94% to 21-24% while the mean
  Purkinje share per cell does not change.

Whole-section ResolVI (final pipeline) finished: `segmentation_hybridC/resolvi/` (latent, corrected counts, model). Not
yet used for annotation or figures.

### 4.4 Figures

| Figure | File (under `SSD/compare/`) | Script |
|---|---|---|
| MNDA figure (Rodrigo's layout, original pipeline) | `purkinje/Atera_future_work_MNDA_v2.pdf` | `mnda_figure.py`, `panel_b.py` |
| Poster figure on hybrid C, with method schematic (a-g) | `purkinje/poster_figure_atera.pdf` | `mnda_figure2.py` |
| Benchmark, Ranger vs final pipeline (A-H) | `benchmark_atera_final.pdf` | `benchmark_atera.py` |
| Showcase: two held-out Purkinje cells + quantification | `purkinje/showcase_purkinje.png` | `showcase.py` |
| Panel b candidates and picks | `purkinje/panel_b_*.png` | `panel_b.py` (`CENTER=`, `PICK=`, `SET=`) |
| Purkinje gallery (12 random somata) | `purkinje/purkinje_gallery_v2_filtered.png` | `pc_gallery.py` |
| Early benchmark incl. baselines | `benchmark_figure.png` | `build_bench_data.py`, `bench_stats.py` |
| Transcripts per cell | `../segmentation_full/transcripts_per_cell.png` | `tx_per_cell.py` |
| UMAP / dot plot / t-SNE | `../segmentation_*/annotation/` | `atera_annot.py`, `embed_compare.py` |
| Biology summary | `../segmentation_full/open_questions/biological_findings_summary.png` | `findings_fig.py` |

## 5. What was tried, in order, and what came of it

This matters for anyone extending the work, because several attractive ideas did not hold up.

1. **18S only (option A, our spinal settings).** 96-99% of cells fell back to the 10x polygons; slightly worse than Ranger.
   The 18S signal is too uniform in the dense granule layer.
2. **Transcript density instead of 18S (option B).** With 60-170 transcripts / µm², density works as a cytoplasm stain.
   Density alone reached parity with Ranger on coverage and MECR@1000. The DAPI tissue mask covered only 45% of a window
   and capped growth; replacing it with a density mask (> 10 tx / µm²) fixed that.
3. **18S + density combined (C500).** Kept as the base. Original 18S pipeline on the large-cell review: it finds
   nucleus-free Purkinje profiles that Ranger misses entirely.
4. **Boundary-stain watershed (phase 1, `seg_v2.py`).** No gain; dropped.
5. **Identity-based assignment (phase 2, `phase2.py`, frozen "PF": reach 4 µm, floor 40, delta 0).** +5.8 points coverage,
   MECR@1000 equal, but plain MECR and the mixed-cell rate worse (also out of sample). Mostly density growth, identity
   only a weak gate.
6. **Identity-driven territory (phase 3, I2).** Confident-type pixels only, uncertain pixels stay free. +3 points coverage
   at plain MECR level with Ranger. Chosen as the growth step.
7. **Transcript-first pipeline (PLAN.md, stages 0-4).** Stage 0 (`feasibility.py`): local composition separates
   neighbouring cells of different type (edge-matched AUC up to 0.92) but not same-type neighbours (~0.55). Stage 1
   (`stage1_features.py`): per-pixel type posterior, calibrated temperature 3.5, 87-95% agreement with the 10x type of the
   nucleus. Stages 2-4 (`stage2_pipeline.py`, `stage_win.py`): on 500 µm windows every claiming rule lands on the same
   coverage-vs-MECR line as Ranger; no gain over I2. Not used further.
8. **Own nucleus detection (`nuclei_detect.py`).** A transcript-only nuclear score (per-gene nucleus log-odds) finds nuclei
   without DAPI (AUC 0.87-0.89); the detector itself under-detected. Adding its extra nuclei as seeds gave +0.2 points.
9. **Per-transcript cleaning.** Cuts MECR by 15% while removing 0.1% of transcripts: a MECR artefact, not better
   segmentation (border-swap recall 6-16%), and it helps Ranger equally. Not used; important caveat for MECR in general.
10. **Depth (z).** Neighbouring nuclei are not separated by z (AUC ~0.5); foreign transcripts sit at other depths only
    weakly (AUC 0.56-0.64). Not used.
11. **Purkinje repair.** v1 scored 0.94 capture but turned dendrite strips into cells and merged neighbouring somata (seen
    in the gallery). v2 adds the 18S-body, elongation and peak-split tests (cut-offs set on spot1 + spot2): 0.77, every
    created cell a real soma. Used.
12. **Mask shape (membrane landscape, 1 µm opening; `phase3.py land=membrane open_um=1`).** No measurable change. Not used.
13. **Nucleus image -> cell type (`nucfeat.py`, `nucmodel.py`).** Balanced accuracy 0.33 over 13 types (chance 0.08),
    Golgi vs granule 0.63; the image cannot replace transcripts for typing.
14. **Other cell types and the hybrid.** The gain is specific to neurons; glia are captured slightly worse because glial
    growth is deliberately short. Hybrid A (ours only for neurons) lost coverage; hybrid B (ours for neurons and granule
    cells, 10x cluster labels) lost Purkinje capture because 10x mislabels Purkinje-contaminated cells as glia; hybrid C
    (our own annotation + Purkinje set-A gate, rule fixed before scoring) keeps the Purkinje gain and brings glia back.
15. **ResolVI.** See 4.3.

## 6. Where the data is

On Marcel's SSD (`/Volumes/Extreme SSD/SPATIAL_FINAL/Atera_CB/`):

| Path | Content |
|---|---|
| `Cerebellum_sample/` | raw 10x bundle (also on SCG `.../RK/Spatial/Atera_CB/Cerebellum_sample/`) |
| `work/density/tx_density.zarr` | transcript-density image |
| `work/local_bundle/`, `work/local_bundle_heldout/` | window-sized transcript / boundary subsets used for the benchmark |
| `segmentation_full/` | original pipeline, whole section: `final_labels.zarr` (uint32, 42511 x 75409, chunks 4096; ids >= 3.5e9 = nucleus-free Purkinje cells), `base_labels.zarr`, `i2_labels.zarr`, `cells.parquet`, `base_cells.parquet` (join to 10x via `nucleus_id`), `cell_by_gene.npz` + `genes.csv`, `summary.json`, `tenx_summary.json`, `annotation/` (h5ad, tables, figures), `open_questions/`, `nucleus_model/` |
| `segmentation_hybridC/` | final pipeline incl. hybrid C: same files (ids >= 2e9 and < 3e9 = Ranger cells, Ranger cell index + 2e9), `annotation/` (h5ad, t-SNE), `resolvi/` (latent.parquet, corrected_counts.npy float16 497,002 x 3,361, cells.csv, genes.csv, model/) |
| `compare/` | all window comparisons: `option_*`, `purkinje/`, `purkinje_hybrid/`, `celltype_capture/`, `resolvi/`, `stage1/`, `stage2/`, `nuclei/`, benchmark figures |

The 500 µm window runs (`crops_*_500/<window>/final_labels.npy`, full-size sparse memmaps) sit in macOS `/private/tmp` and
are being cleaned by the system (their small json / parquet files are already gone). They can be regenerated from
`run_scripts/run_500.sh` (base), `run_i2_window.sh`, `pc_repair.py`, `hybrid_seg.py`.

SCG: an earlier attempt to copy everything to `.../RK/Spatial/Atera_CB_mw/handoff_2026-10-02/` broke twice when the SSH
master dropped; that folder is incomplete. To finish: copy `segmentation_full/`, `segmentation_hybridC/` (skip
`resolvi/model` if space matters) and `compare/` with `rsync --partial`, small files first.

## 7. Caveats

- One section of one donor. Everything (tuning windows, held-out windows, random windows) comes from the same section;
  held-out windows guard against overfitting the settings, not against section or donor effects.
- Pre-production data: 10x states that production data does not have these segmentation issues. The claim is "our
  pipeline also works on Atera and keeps large neurons whole", not "Ranger is bad".
- The Purkinje soma test uses different genes for defining and scoring the soma (set A vs set B), but the same soma
  region; the 10x-nucleus comparison (x1.46 transcripts) does not have that issue.
- MECR uses a fixed marker panel and counts single stray transcripts; small MECR differences are not meaningful, and
  anything that deletes a few marker transcripts improves it.
- The first-pass annotation is marker-based: the "Golgi" cluster (9k) is probably mostly granule cells, and the
  "Purkinje" cluster (11.5k) mixes Purkinje cells with Purkinje-contaminated cells; use the gene-defined Purkinje cells
  (>= 4.3% Purkinje genes) when it matters. On hybrid C the oligodendrocyte count drops (28.6k vs 44.9k) because Ranger
  glial cells carry more neighbour transcripts; 25k Ranger cells taken over in hybrid C are granule cells by 10x's own
  clustering.
- The dendritic mRNA ranking reads molecular-layer neuropil as Purkinje dendrite. It is restricted to Purkinje-specific
  genes for that reason, replicates between section halves and passes the interneuron / granule proximity control, but it
  is not validated by FISH or a second donor.

## 8. Script index

Grouped by purpose; every script has a docstring with its method and usage.

**Data and density.** `make_density_image.py`, `map_gene_density.py` (GJD2 etc.), `cut_examples.py`,
`cut_margin_transcripts.py`, `scripts/baselines/make_minibundle.py`, `make_chunks.py`.

**Segmentation core.** `run_fullsample.py` (tiled driver around `splitmerge.rules`, with `--density`, `--combined`,
`--density-tissue-thr`, `--v2`, `--stats`, `--param`, `--stat`), `seg_v2.py` (phase 1), `phase2.py`, `phase3.py` (I2,
`land=` / `open_um=` options), `pc_soma.py`, `pc_repair.py`, `hybrid_seg.py`, `full_post.py` (whole-section stitch / i2 /
pcr / hybc / count), `export_xenium.py` + `xenium_writer.py` (Xenium Explorer export, not run for Atera).

**Evaluation.** `compare_crop.py` (window vs 10x), `win_score.py` (window metrics for several runs),
`build_bench_data.py` + `bench_stats.py` (block bootstrap benchmark), `pc_eval.py` (Purkinje somata), `celltype_capture.py`,
`purkinje_blobs.py`, `purkinje_blobs2.py`, `plot_pareto.py`, `tx_per_cell.py`, `tenx_percell.py`, `tenx_total.py`,
`diag_object.py`.

**Transcript-first exploration.** `PLAN_2026-10-01.md`, `feasibility.py`, `stage1_features.py`, `stage2_pipeline.py`,
`stage_win.py`, `nuclei_detect.py`, `nuclei_fig.py`, `z_test.py`, `soma_shape.py`.

**Annotation and embeddings.** `atera_annot.py`, `embed_compare.py`, `nucfeat.py`, `nucmodel.py`.

**ResolVI.** `resolvi_inputs.py`, `resolvi_run.py`, `resolvi_full.py`; drivers in `run_scripts/`.

**Biology.** `oq_neuropil.py`, `oq_cells.py`, `oq_zebrin_de.py`, `oq_dendrites.py`, `oq_dendrites_robust.py`,
`oq_mli_check.py`, `findings_fig.py`.

**Figures.** `mnda_figure.py`, `mnda_figure2.py`, `panel_b.py`, `benchmark_atera.py`, `showcase.py`, `pc_gallery.py`,
`shape_compare.py`, `tile_1x3.py`, `full_overview.py`, `nice_spot.py`, `convince_fig.py`, `convince_fig_I2.py`,
`rows_merge_nuclei_masks.py`, `show_nuclei.py`, `review_large.py`, `review_vs_original.py`, `plot_compare.py`,
`find_rodrigo_tiles.py`.

**Baselines** (`scripts/baselines/`): Cellpose and Baysor crop runners, local run helpers. Baselines were run at their
standard settings, never tuned.
