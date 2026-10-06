# Methods: segmentation and analysis of the Atera human cerebellum section

Written as a methods section; parameters are the ones used for the reported results. Script names refer to
`atera_mw/scripts/`.

## Data

A single section of adult human cerebellum (AT1183_S4C) was profiled by 10x Genomics on the pre-production Atera platform
with a whole-transcriptome panel of 18,050 genes and shared with us as an unreleased dataset. The output bundle contained
3.42 billion decoded transcripts with positions, gene identity, quality value (qv) and the 10x cell assignment, a
four-channel morphology image (DAPI; ATP1A1 / CD45 / E-cadherin membrane stain; 18S rRNA; αSMA / vimentin) at
0.2125 µm per pixel (42,511 x 75,409 pixels), the Xenium Ranger segmentation (510,556 cells, cell and nucleus polygons)
and the 10x graph-based clustering (36 clusters) with per-cluster differential expression. Unless stated otherwise, all
analyses used transcripts with qv >= 20 that map to genes (3.01 billion; "QC transcripts").

## Transcript-density image and whole-sample statistics

QC transcripts were binned to the image grid and smoothed with a Gaussian of sigma 0.5 µm, giving a density image in
transcripts per µm² stored as a chunked zarr array (`make_density_image.py`). Thresholds and normalisation constants were
computed once over the tissue of the whole section, never per crop or tile: tissue percentiles of the 18S signal (p1 / p99:
48 / 941 after background subtraction) and of the density image (background anchor 5.815 tx / µm², high anchor 177.18 tx /
µm²), the DAPI tissue threshold and the nucleus reference intensities (`run_fullsample.py --stats-only`,
`config/stats_density.json`).

## Base segmentation

The base segmentation reused, unchanged, the rule-based split-and-merge segmentation developed for motor neurons in ALS
spinal cord (`paper_reseg_pipeline/splitmerge/rules.py`, function `segment_region`). In brief, the method thresholds a
smoothed enrichment signal to find cell bodies, assigns 10x nuclei to bodies, splits bodies that hold several nuclei along
watershed lines of the signal, separates soma from dendrite-like protrusions by shape, merges touching pieces with
compatible transcript profiles, adds glial somata around nuclei without a body, rescues nuclei missed by the default
segmentation and keeps the 10x polygon where no body is found.

For Atera, the 18S slot of the method received a combined signal: the normalised 18S signal and the normalised density
signal, each scaled between its whole-sample background and high anchor and cut at 0.459 and 0.793 of that range
respectively, then combined (`CombinedStack` in `run_fullsample.py`). The tissue mask was replaced by transcript density >
10 tx / µm², because the DAPI mask covered less than half of the tissue and capped growth. Settings that differ from the
spinal defaults: body threshold 0.5 of the normalised signal, alternative-signal threshold 0.6, glial soma threshold 0.2
and maximum glial soma radius 8 µm. These settings ("C500") were chosen on two 500 µm windows (spot1, spot2) and frozen.
The section was processed in 147 tiles of 4,096 x 4,096 pixels that contain tissue, each with a 400-pixel margin; a cell
is kept by the tile that contains its centroid, so each cell is decided once. Tiles were stitched by offsetting labels
(tile index x 10⁶); 302 pixels in total were claimed twice at seams and kept by the first tile.

## Identity-gated growth (I2)

Cell types were read from transcripts with a fixed reference: informative genes were those with log2 fold change >= 2,
mean count >= 0.3 and adjusted p < 0.01 in at least one 10x cluster (3,295 genes); the 36 clusters were grouped into six
types (granule, Purkinje / interneuron, astroglia, oligodendroglia, vascular, immune), and each type's gene profile was the
cell-count weighted mean of its clusters' mean counts. Each informative transcript contributed log p(gene | type) to its
4-pixel bin; per-type evidence was smoothed with a Gaussian of sigma 2 µm and interpolated bilinearly to the pixel grid.
A pixel's type was the type with the highest evidence, and its confidence the log-likelihood gap to the second-best type.
Each base cell was typed from its own informative transcripts (cells with >= 30 such transcripts).

Free pixels in tissue (density > 40 tx / µm²) were assigned only when confident (gap >= 2 and at least 0.5 smoothed
informative transcripts): for each type separately, a watershed on negative density was flooded from the cells of that
type, restricted to confident pixels of that type within 4 µm of such a cell (14 µm for Purkinje / interneuron cells).
Pixels without a confident type stayed unassigned. Only the part of a grown cell connected to its original body was kept,
and holes were filled (`phase3.py`, `full_post.py i2`; settings chosen on spot1 + spot2 among a grid of claim thresholds
and reaches, then frozen).

## Purkinje repair

Purkinje-specific genes (log2 fold change >= 3 and mean >= 1 in the 10x Purkinje cluster 33, mean <= 0.3 in every granule
cluster; 116 genes) were split at random (seed 0) into set A (58 genes, used to find and repair somata) and set B (58
genes, used only for scoring). A soma core was a connected region where the set-A density (Gaussian sigma 1.5 µm) was at or
above 0.534 tx / µm² (Otsu threshold on the pooled log density of tissue pixels in nine windows), within tissue, opened by
1 µm and >= 60 µm². A core was accepted as a soma only if it had an 18S body (median normalised 18S >= 0.2) and was compact
(major / minor axis <= 3); a core with two or more set-A density peaks >= 12 µm apart (each >= twice the threshold) was
split along the density valley into one soma per peak. These cut-offs were set on spot1 and spot2 after a gallery of the
first version showed dendrite strips and merged neighbours (`pc_soma.py`).

Each accepted core was grown by up to 4 µm into pixels that were 18S-bright (>= half the core's median 18S) and still
carried Purkinje signal (set-A density >= one third of the threshold); holes were filled. 10x nuclei lying less than 50%
inside the core were protected and kept their own cells. The cell holding most set-A transcripts in the extent (at least
20%) became the soma's cell; otherwise a new cell was created (nucleus-free Purkinje profiles). The soma's cell received the
whole extent minus protected nuclei, and other cells lying >= 60% inside the extent without a protected nucleus were
absorbed. Each existing cell could own at most one soma (`pc_repair.py`, `full_post.py pcr`; new cells get ids >= 3.5 x
10⁹ derived from the soma centroid so neighbouring tiles agree).

## Hybrid with the default segmentation (optional final step)

Because the method captured glial cells slightly worse than Xenium Ranger, a hybrid was built with a rule fixed before any
scoring: a cell of our segmentation was replaced by the Ranger cells covering it only if our transcript-based annotation
called it astroglia, oligodendroglia, vascular, immune or fibroblast / meningeal and its share of Purkinje set-A
transcripts was below 0.0164 (midpoint between the set-A share of the 10x Purkinje cluster, 0.0270, and of the highest
non-neuronal cluster, 0.0058). All other cells were kept from our segmentation and painted over the Ranger raster; Ranger
cells whose nucleus belongs to a kept cell were removed, and Ranger remnants < 10 µm² were dropped (`hybrid_seg.py`,
`full_post.py hybc`; Ranger cells get ids 2 x 10⁹ + Ranger row index).

## Expression matrix, QC and annotation

QC transcripts were assigned to the label under them and counted per cell and gene (`full_post.py count`). Cells passed QC
with >= 200 transcripts, >= 100 genes and >= 10 µm² (failing cells were flagged, not removed). Counts were normalised to the
median total count, log-transformed, reduced to 3,000 highly variable genes (Seurat flavour), scaled (max 10) and projected
on 50 principal components; a 15-nearest-neighbour graph gave Leiden clusters (resolution 2.0) and a UMAP (min_dist 0.3)
(`atera_annot.py`). Each cluster was named by the cell type of a fixed literature marker panel (granule, UBC, Purkinje,
Golgi, basket / stellate, Bergmann glia, astrocyte, oligodendrocyte, OPC, microglia, border macrophage, endothelial,
pericyte / smooth muscle, fibroblast / meningeal, lymphocyte) with the highest mean scaled expression, restricted to types
whose markers reach at least twice their median expression over clusters; a cluster whose best and second-best scores
differed by less than 0.25 was called mixed. Types were grouped into eight lineages for display. Gene-defined Purkinje cells
were cells with >= 200 transcripts of which >= 4.3% come from the 116 Purkinje genes (midpoint between the 10x Purkinje
cluster, 6.1%, and the interneuron cluster 14, 2.5%). A t-SNE was computed with openTSNE (FFT, perplexity 30, PCA
initialisation) on the same principal components (`embed_compare.py`).

## Benchmark

**Windows.** Nine 500 µm windows (2,353 x 2,353 pixels; `config/windows_500.txt`): spot1 and spot2 for tuning, spot3 and
held1-6 never used for any choice. For the ResolVI comparison, 20 further windows were drawn at random (seed 42) from
tissue (>= 80% of the window above 10 tx / µm²) without overlapping the nine.

**Coverage and marker mixing.** On the QC transcripts of each window: the share assigned to a cell; the mutually
exclusive co-expression rate (MECR) over the fixed cross-lineage marker panel of `splitmerge/metrics.py` (neuron, oligoden-
drocyte, astrocyte, microglia markers; cells with >= 5 transcripts); and MECR with every cell reduced to 1,000 random
transcripts (cells with fewer left out) to compare mixing at equal depth (`win_score.py`). Per-window differences to Ranger
were tested with the Wilcoxon signed-rank test (exact for n = 7). An earlier version of the benchmark with spatial block
bootstrap (2,000 replicates) and the baselines Voronoi, watershed and Cellpose at standard settings is in
`build_bench_data.py` / `bench_stats.py`.

**Purkinje somata.** Somata were found with set A as above in all nine windows (139 somata); each soma region was the core
dilated by 1 µm. For each method, the set-B transcripts in the region were tallied by cell: capture = share in the single
best cell, split = >= 2 cells hold >= 10%, missed = no cell holds >= 10%, contamination = share of granule and glia genes
in the best cell (`pc_eval.py`). Paired differences were tested with the Wilcoxon signed-rank test (capture) and the exact
McNemar test (split, missed).

**All cell types.** Each 10x nucleus was typed by its 10x cluster (eight groups). Specific genes per group: group mean >= 0.5
and >= 3 times the mean of every other group. A nucleus owned the transcripts of its group's specific genes that lie closer
to its centroid than to any other nucleus and within 10 µm. For nuclei owning >= 20 such transcripts, capture and split were
computed as for Purkinje somata (`celltype_capture.py`).

**Transcripts per cell for the same nuclei.** Our cells were matched to Ranger cells through the 10x nucleus that seeded
them; Ranger cells were recounted with the same QC filter (`tx_per_cell.py`, `tenx_percell.py`).

## ResolVI

ResolVI (scvi-tools 1.4.2) was run with its standard settings (unsupervised, 50 epochs, CPU) separately on each
segmentation, on cells with >= 20 QC transcripts and a fixed gene set of 3,361 genes (the informative genes, the MECR
markers and the Purkinje, granule and glia gene sets); cells with fewer than 5 counts on that set were removed as required
by the method. Spatial coordinates were the transcript centroids; windows were the batch (`resolvi_inputs.py`,
`resolvi_run.py`). Corrected counts were the median of three posterior samples of the corrected expression rate; MECR was
computed on corrected counts with presence defined as >= 0.5. For the whole section, ResolVI was trained on 497,002 cells
(batch size 1,024) and the latent representation and corrected counts were saved (`resolvi_full.py`).

## Exploratory analyses (not used in the final method)

- **Feasibility of transcript-based borders** (`feasibility.py`): pairs of nuclear transcripts in one 10x nucleus vs two
  neighbouring nuclei at matched distance (1-4 µm) and matched distance to the nucleus edge; features = cosine similarity of
  local composition (cluster-profile or unsupervised SVD embedding, sigma 0.5-2 µm), density difference, membrane stain on
  the connecting line; AUC per crop.
- **Pixel type posterior** (`stage1_features.py`): softmax of the per-type evidence with a temperature (3.5) fitted on spot1
  and spot2 nuclear transcripts.
- **Transcript nuclear score** (`nuclei_detect.py`): per-gene shrunken log-odds of lying in a nucleus, averaged around each
  pixel; combined with DAPI in a logistic model trained on spot1.
- **Depth** (`z_test.py`): |Δz| for nucleus pairs and z offset of foreign vs native transcripts inside cells.
- **Nucleus image to cell type** (`nucfeat.py`, `nucmodel.py`): 25 shape, DAPI texture, 18S / membrane ring and crowding
  features of 120,000 random nuclei; gradient boosting with five spatial folds (vertical strips).

## Biological readouts (showcase, one section)

- **Layers** (`oq_cells.py`): 25 µm bins classified from the annotated cells in a 3 x 3 neighbourhood (white matter >= 40%
  oligodendroglia; granule layer >= 60% granule cells; Purkinje layer = molecular side within 25 µm of the granule layer;
  molecular layer beyond).
- **Dendritic mRNA** (`oq_neuropil.py`, `oq_dendrites.py`, `oq_dendrites_robust.py`, `oq_mli_check.py`): unassigned QC
  transcripts per 25 µm bin and gene; for Purkinje-specific genes (cluster 33 >= 3 x every other cluster except 13 and 23,
  which carry Purkinje contamination; >= 2,000 soma transcripts) a dendritic index = log2((molecular-layer neuropil + 1) /
  (transcripts in the 945 gene-defined Purkinje cells + 1)), centred on the median and corrected for expression level by
  linear regression on log soma counts. Controls: replication between the two halves of the section; neuropil > 75 µm from
  the granule layer only; enrichment of neuropil transcripts within 4 µm of interneuron and granule cell bodies against
  the background of all molecular-layer neuropil.
- **Connexins**: share of cells with >= 1 transcript per cell type.
- **Zebrin** (`oq_zebrin_de.py`): ALDOC per 10k corrected for Purkinje purity and contamination; spatial coherence over the
  5 nearest Purkinje cells against 1,000 permutations; per-gene regression on the zebrin score with covariates.

## Software

Python 3.12 (`spatial` env: numpy, scipy, pandas, pyarrow, zarr, tifffile, scikit-image, scikit-learn, scanpy 1.12,
anndata 0.12, leidenalg, matplotlib) and Python 3.11 (`resolvi` env: scvi-tools 1.4.2, PyTorch, openTSNE 1.0.4). Figures
use Arial.
