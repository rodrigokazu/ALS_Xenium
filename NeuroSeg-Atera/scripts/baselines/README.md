# Paper baselines on the Atera cerebellum crops

The six baselines of the paper benchmark (Voronoi, Watershed, Cellpose, Baysor, BIDCell, Segger) on Marcel's three
400 x 400 um crops of the Atera whole-transcriptome cerebellum (`Atera_CB/Cerebellum_sample`, 18,050 genes,
3.4 B transcripts), scored with the benchmark's own metrics.

Nothing under `paper_reseg_pipeline/` is modified. The benchmark modules are imported unchanged by path from
`/home/mw28/spatial_dapi_pipeline/paper_reseg_pipeline/` (`scg/baselines/common.py`, `watershed_fullsample.run_tile`,
`cellpose_fullsample.norm99 / count_tile / pct_from_counts`, `baysor/run_baysor_fullsample.run_baysor / outer_ring`,
`segger/run_segger_fullsample.py` as a script, `splitmerge.metrics.score / mecr_downsampled`). The scripts here only
do the crop-sized input and output around them.

Outputs: `_work_reseg_atera/baselines/<method>/<crop>/` (`labels_window.npy` uint32 on the level-0 pixels of the
crop window, `assignment.parquet` transcript_id -> cell_id for the crop's QC transcripts, `cells.parquet`,
`run_info.json`), scores in `_work_reseg_atera/baselines/scores/` (`<method>_<crop>.json`, `all.csv`).

## Files

| file | what |
| --- | --- |
| `atera_common.py` | crop windows, pyarrow-filtered transcript reads, window-local pasting, output writer |
| `stats_watershed_threshold.py` | whole-sample 5th percentile of the watershed density image |
| `stats_cellpose_norm.py` | whole-sample p1 / p99 of the boundary and DAPI channels |
| `voronoi_crop.py`, `watershed_crop.py`, `cellpose_crop.py`, `baysor_crop.py` | per-crop drivers |
| `make_minibundle.py`, `segger_post.py` | crop-sized bundle for the unchanged Segger driver, and its conversion |
| `score_crops.py` | scoring (as `scg/benchmark_fullsample.py`) |
| `run_baseline.sbatch` | one SLURM entry point, `METHOD=...`, array index = crop |

## Scoring (same as `scg/benchmark_fullsample.py`)

- Transcripts: the crop window's QC transcripts, `is_gene & qv >= 20`, x in `[x0 * 0.2125, x1 * 0.2125)` um and
  the same for y (the scoring set of `atera_reseg/compare_crop.py`).
- Nucleus reference: the bundle's `overlaps_nucleus`.
- A method's assignment is joined on transcript_id and turned into integer codes as `benchmark_fullsample.cell_codes`.
- `splitmerge.metrics.score` unchanged. Added (approved by the user): `mecr_n1000 = mecr_downsampled(labels, genes,
  n=1000)` and `n_cells_ge1000_tx`, because at WTA depth `mecr_n20` is about 0 for every method.
- 10x's own assignment (bundle `cell_id`) is scored alongside as `xenium_original`.

## Changes against the benchmark scripts (all methods)

1. **Crop instead of whole sample.** Every method sees the crop window +- 25 um of context (`PAD_UM`, 118 px), so
   cells that straddle the window edge are segmented whole. Results are then cut to the window. Image methods also
   get the benchmark's own 800 px tile margin around that core, exactly like one tile of the whole-sample run.
2. **Transcript reads.** The 47 GB `transcripts.parquet` is never read whole: pyarrow filters on x / y (plus
   `is_gene` and `qv`) read only the region a method needs.
3. **Label images** are held for the tile window only, not the full 75409 x 42511 grid.
4. **Whole-sample statistics stay whole-sample.** Watershed's mask threshold and Cellpose's p1 / p99 normalization
   are computed over the whole Atera slide (two stats jobs), never from the crop.

## Per method

**Voronoi**: the whole-sample algorithm unchanged (seeds = every 10x nucleus of the slide, reflected at the 4 image
edges, clipped regions < 1 px^2 dropped, no distance cap). The label image is the nearest seed of each window pixel
centre. Because the seeds are the whole slide's nuclei, this is exactly the whole-sample result inside the window.

**Watershed**: `watershed_fullsample.run_tile` imported and called on the crop's tile (core = crop +- 25 um, one
4096 px tile, 800 px margin). The one implementation change: the whole-sample 5th-percentile threshold comes from
`stats_watershed_threshold.py`, which streams a per-pixel count image (uint16) and smooths it in 4096-row bands with
a 32-row pad (the driver's KPAD, more than the 24 px kernel radius). The driver itself would hold all 3.0 B QC
transcript coordinates in memory. It is the same statistic with the same percentile arithmetic.

**Cellpose**: the whole-sample per-tile step (cpsam_v2 default weights, `[ch0001 boundary, ch0000 DAPI]`,
`eval(channels=[1, 2], diameter=None, normalize=False)` after `norm99` at the whole-sample p1 / p99, loader =
`common.cells_from_labels`) on the crop's tile. It runs on `gpu_normal` (H200), the partition the benchmark's
`cellpose.sbatch` uses: `cellpose_env` (torch 2.13) only imports on that RHEL 9 node, while the `batch` GPU nodes run
CentOS 7 (`CXXABI_1.3.8 not found`).

**Baysor**: `run_baysor` imported unchanged (cpp-0.8.3, unmodified `configs/xenium.toml`, nucleus prior = the
bundle's `nucleus_boundaries.parquet` rows within the window +- 5 um, molecules `is_gene & qv >= 20`, fallback
`--scale` only on "Scale could not be determined", OMP_NUM_THREADS=1). Change: one Baysor window per crop (crop +-
25 um, about 12 M molecules), with every Baysor cell kept, as in the 108-crop benchmark runs. The whole-sample
1500 um tiles + 100 um margins are not used, because one 1700 um Atera tile would hold about 180 M molecules.
Polygons are rasterized at pixel centres (later polygon wins) and cut to the window.

**Segger**: the benchmark driver `segger/run_segger_fullsample.py` run unchanged as a script, on a crop-sized copy
of the bundle (`make_minibundle.py`: every column of every transcript in crop +- 25 um, and every vertex row of the
nuclei touching it), with the driver's own `--window` = crop +- 25 um. Runs on a `batch` GPU node, as the driver's
sbatch notes (segger_env is the CentOS 7 build). Tutorial settings are untouched, including `num_tx_tokens=500`.

**BIDCell**: not run, see the report / status below.

## Local runs (Mac), 2026-09-30

The SCG batch queue had ~1,500 jobs ahead, so the runs were moved to Marcel's Mac (`run_local.sh`, `ATERA_LOCAL=1`
switches `atera_common.py` to local paths). The drivers are the same. Differences from the SCG runs:

- Inputs: `SPATIAL_FINAL/Atera_CB/work/local_bundle/` (morphology, cells, boundaries, and a `transcripts.parquet`
  with every column for the three crops plus an 800 px margin). The per-crop reads and results are the same:
  Voronoi and Watershed reproduce the SCG runs' cell counts and assignment rates exactly on all three crops.
- Watershed's whole-sample threshold (`_stats/watershed_threshold.json`, thr = 1.4888e-4, zero fraction 0.51 %)
  comes from the SCG job 52657750 (value copied from its log). The local transcripts cover only the crops, so it
  cannot be recomputed here. The watershed tile's outermost 118 px of margin has no transcripts locally (the tile
  reaches 918 px past the crop, the local file 800 px). Cells in the core are unaffected: the SCG and local counts match.
- Cellpose's p1 / p99 were recomputed locally from the full image and are identical to SCG (boundary 0-131, DAPI
  0-257). Cellpose runs on MPS (Apple GPU), as the 108-crop benchmark did locally. The local `cellpose_env` holds
  only cellpose (no zarr, matplotlib, skimage or pyarrow), so `cellpose_crop.py` runs in three stages: image
  reading and `norm99` in the spatial env, `model.eval` in cellpose_env, then the loader in the spatial env. Same
  calls; on spot1, MPS gives 5249 native cells, the same as the H200 run (1821 vs 1820 kept after the loader).
- Baysor: local build `~/opt/Baysor` at the same commit d7077a7 and a byte-identical `configs/xenium.toml`.
- Segger: local `segger_env` (the crop benchmark's env, CPU; the driver picks cpu / precision 32 without CUDA).
- Threads: BLAS / OpenMP pinned to 1 thread, at most ~4 cores in use, one method at a time.

## Status (2026-09-30, 16:40)

- Voronoi, Watershed, Cellpose: done and scored on all three crops (locally; SCG runs of the same three agree).
- Baysor: running locally, one single-threaded process per crop. Loading plus ICA init takes about 20-70 min, and
  molecule clustering takes about 9 min per 100 iterations under the current load (spot1 was still clustering at
  iteration 900 after 3 h). `finish_local.sh` (detached) waits for all three runs, then scores and copies to the SSD.
- Segger: not run to completion. At the tutorial's `num_tx_tokens=500`, the model's `Embedding(500, 8)` cannot
  take this panel's gene tokens: segger's default `TranscriptEmbedding` is a LabelEncoder over all 18,050 genes, so
  token ids go up to 18,049 and the first forward pass fails with an index error. The standard config therefore
  cannot run on this panel. The only change that would make it run is `num_tx_tokens >= 18,050`, a setting
  change, so it was not made. (The one local attempt was SIGKILLed during segger's tile writing, before training.)
- BIDCell: not run. The benchmark's spinal reference was built from the Yadav 2023 human spinal cord snRNA atlas
  (GSE190442; see `notebooks/data/paper_methods/bidcell_reference/SOURCES.md`). No human cerebellum atlas is on
  SCG or on this Mac. The equivalent build needs a download, e.g. Lake et al. 2018 cerebellar hemisphere
  snDrop-seq (GEO GSE97930), which needs Marcel's approval. The 18k-gene panel is a second concern: BIDCell's model
  takes one input channel per reference gene.

## 500 um windows (2026-10-01, `run_local500.sh`)

Voronoi, Watershed and Cellpose rerun on three 500 um windows (list in `atera_local/baselines500/crops_500.tsv`), same
drivers and standard settings, whole-sample thr / p1 / p99 reused unchanged. Label images only, not scored here.
Only change: the tile margin. The local `transcripts.parquet` covers each window +- 400 px, so the benchmark's 800 px
margin cannot be used. The core pad stays at 118 px (25 um) as in the 400 um runs, and the tile margin is set to 282
px, so total context = 400 px. `atera_common.paper_path()` reads `ATERA_MARGIN_PX` and patches `common.MARGIN_PX` and
the default of `common.window()` (the paper file itself is untouched). Voronoi is unaffected (whole-slide seeds).
Outputs: SSD `Atera_CB/compare/baselines500/<method>/<window>/{labels_window.npy,run_info.json}`.
