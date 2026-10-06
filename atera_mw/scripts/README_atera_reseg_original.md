# Atera cerebellum resegmentation (separate copy)

Separate copy of the paper pipeline's full-sample scripts for the Atera whole-transcriptome cerebellum
(`Atera_CB/Cerebellum_sample`, 3.4 B transcripts). `paper_reseg_pipeline/` is not modified; the segmentation method
itself (`paper_reseg_pipeline/splitmerge/rules.py`) is imported unchanged.

Differences from `paper_reseg_pipeline/scg/`:
- `run_fullsample.py`: transcripts read in batches, filtered as read, categorical gene names, sorted by y so a tile
  takes its band by binary search; a `--window` run reads only the window plus the tile margin.
- `xenium_writer.py`: transcript -> cell assignment and the count matrix streamed in row batches.
  Both checked identical to the originals on SD03914_BG (14.1 M transcripts).
- `run_crop.sbatch`: crop windows (`crops_*.tsv`, level-0 px) with the whole-sample statistics from the
  `--stats-only` run in `_work_reseg_atera/_stats/`, never crop-local thresholds.
