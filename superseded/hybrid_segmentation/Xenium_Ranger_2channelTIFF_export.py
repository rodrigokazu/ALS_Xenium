# ========================================================================================
# ALS_Xenium repo commentary | superseded/hybrid_segmentation/Xenium_Ranger_2channelTIFF_export.py
#
# Exports the outlines as a two-channel TIFF so they can be viewed against a second channel.
#
# Built under the same constraints as its sibling: a hard-coded Windows Dropbox path, one
# sample, no argument parsing.
#
# The current way to look at segmentation against imaging is
# pipeline_final/xenium_overlay_FINAL.py. It reads the OME-TIFF pyramid directly and tests at
# runtime whether the boundaries can be trusted at all.
# ========================================================================================

# Getting cell outlines as a 2-channel TIFF from Xenium Ranger resegmentation of large cells
# Written by Dr Rodrigo Kazu @ JCK lab - University of Sheffield

import os
import pandas as pd
import numpy as np
import tifffile
from skimage.draw import polygon_perimeter

BASE = os.path.join(
    r"C:\Users\md1rss\Dropbox\NeuralPathways\Current_projects",
    r"Cooper-Knock_lab\Analysis_2025\Spatial",
    r"ALS2_SD01915_BG__largecells_reseg_20251215_064234_job49421202_4",
    r"outs"
)

cell_boundaries_path = os.path.join(BASE, "cell_boundaries.parquet")
morph_path = os.path.join(BASE, "morphology.ome.tif")
output_tif = os.path.join(BASE, "ranger_cells_edges_pixelspace.tif")

# ---- LOAD ----
cells = pd.read_parquet(cell_boundaries_path)
morph = tifffile.imread(morph_path)

# ---- Extract morphology 2D plane ----
if morph.ndim == 2:
    morph2d = morph
elif morph.ndim == 3:
    morph2d = morph[0]
elif morph.ndim == 4:
    morph2d = morph[0, 0]
else:
    raise ValueError(f"Unsupported morphology shape: {morph.shape}")

H, W = morph2d.shape

# ---- Pixel size (µm per pixel) ----
PX_SIZE = 0.3774  # adjust only if your dataset differs

# ---- Raster ----
edges = np.zeros((H, W), dtype=np.uint8)

for _, grp in cells.groupby("cell_id"):
    rr, cc = polygon_perimeter(
        (grp["vertex_y"] / PX_SIZE).astype(int),
        (grp["vertex_x"] / PX_SIZE).astype(int),
        shape=edges.shape,
        clip=True
    )
    edges[rr, cc] = 255

tifffile.imwrite(output_tif, edges)
print(f"Wrote pixel-aligned edges to: {output_tif}")

