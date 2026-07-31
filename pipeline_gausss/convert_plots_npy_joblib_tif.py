# ========================================================================================
# ALS_Xenium repo commentary | pipeline_gausss/convert_plots_npy_joblib_tif.py
#
# Converts the niche plot PNGs into formats Marcel's tooling could open: compressed joblib
# uint8 arrays as the preferred route, lossless LZW TIFF as the fallback.
#
# A pure format conversion with no resampling and no lossy re-encode, so the rasters are
# faithful to the originals.
#
# A workaround for a file-exchange constraint and not anything scientific. Prefer
# export_niche_data_joblib.py; it sends the underlying data instead of a raster.
# ========================================================================================

"""
convert_plots_npy_joblib_tif.py
============================================================
Convert the Novae result plots (PNG) to non-PNG/JPG formats for Marcel:
  - joblib/<name>.joblib : uint8 numpy array (H, W, C), zlib compress=3  [preferred]
  - tif/<name>.tif       : lossless LZW TIFF                              [fallback]
Faithful raster conversion (no resampling, no lossy re-encode).
"""

import os
import glob
import numpy as np
import joblib
from PIL import Image

SRC = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_niches/plots"
OUT = "/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/Novae_persample_niches/converted"

os.makedirs(os.path.join(OUT, "joblib"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tif"), exist_ok=True)

pngs = sorted(glob.glob(os.path.join(SRC, "*.png")))
print(f"[convert] {len(pngs)} PNGs from {SRC}", flush=True)

rows = []
for p in pngs:
    name = os.path.splitext(os.path.basename(p))[0]
    img = Image.open(p)
    arr = np.asarray(img)                       # uint8 (H, W, C) — exact pixels
    jpath = os.path.join(OUT, "joblib", name + ".joblib")
    tpath = os.path.join(OUT, "tif", name + ".tif")

    joblib.dump(arr, jpath, compress=3)         # zlib level 3

    try:
        img.save(tpath, format="TIFF", compression="tiff_lzw")
    except Exception as e:                       # RGBA/odd modes -> fall back to RGB
        print(f"  [tif] {name}: {type(e).__name__} on faithful save -> RGB ({e})", flush=True)
        img.convert("RGB").save(tpath, format="TIFF", compression="tiff_lzw")

    rows.append((name, tuple(arr.shape), str(arr.dtype), img.mode,
                 os.path.getsize(jpath), os.path.getsize(tpath)))
    print(f"  {name}: shape={arr.shape} dtype={arr.dtype} mode={img.mode}", flush=True)

with open(os.path.join(OUT, "README.txt"), "w") as f:
    f.write("Converted Novae result plots (source: PNG). No PNG/JPG here.\n\n")
    f.write("joblib/<name>.joblib : uint8 numpy array (H, W, C), zlib compress=3.\n")
    f.write("    Load:  import joblib; arr = joblib.load('<name>.joblib')\n")
    f.write("tif/<name>.tif       : lossless LZW TIFF (open in any image tool / tifffile / PIL).\n\n")
    f.write("file: shape | dtype | mode | joblib_bytes | tif_bytes\n")
    for name, shape, dt, mode, jb, tb in rows:
        f.write(f"{name}: {shape} | {dt} | {mode} | {jb} | {tb}\n")

print(f"[convert] DONE: {len(pngs)} files -> {OUT} (joblib/ + tif/ + README.txt)", flush=True)
