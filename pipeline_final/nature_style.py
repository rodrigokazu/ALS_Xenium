# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/nature_style.py
#
# Shared plotting helpers so every figure in the pipeline looks like it came from the same
# paper. Imported all over the place, so it stays free of project specific logic.
#
# What it gives you: apply_style() for the rcParams (7pt Arial, editable text via fonttype 42,
# vector friendly), despine_spatial() to strip axes and fix the aspect and y direction for
# tissue plots, add_scalebar() for a true-micron bar, cb_palette() for the Wong
# colourblind-safe sequence, domain_colors() which keeps 'unassigned' grey, and save_both()
# which writes a PDF and a 300 dpi PNG from one call.
#
# Call matplotlib.use("Agg") before importing this on a compute node. There is no display and
# the failure if you forget is not obvious.
#
# Rasterise the scatter in spatial plots. A section is over a hundred thousand points and a
# fully vector PDF of that is unopenable.
# ========================================================================================

"""
nature_style.py
============================================================
Shared publication-style helpers for the Novae ventral-horn sub-domain pipeline
(sheff_als_spatial). Imported by every plotting script in this directory:
  subset_domains_stage1.py / novae_resubset_stage2.py / downstream_subdomains_stage3.py

Provides:
  apply_style()                     -> set matplotlib rcParams for Nature-tier figures
  despine_spatial(ax)               -> hide all spines/ticks, equal aspect, invert y (tissue plots)
  add_scalebar(ax, xy_um, length_um=500, ...)  -> a µm scale bar (true-micron spatial plots)
  cb_palette(n=None)                -> colorblind-safe discrete colour list (Wong 2011 + extension)
  domain_colors(labels, unassigned="unassigned")  -> {label: colour}, grey for 'unassigned'
  save_both(fig, basepath, dpi=300) -> write <basepath>.pdf AND <basepath>.png @300 dpi

All callers must `import matplotlib; matplotlib.use("Agg")` BEFORE importing pyplot;
this module imports pyplot but does not set the backend (the caller owns that, so a
caller that already chose Agg is respected and no display is ever required).

NOTE: scatter calls in the plotting scripts pass rasterized=True so the vector PDF stays
small (point geometry is rasterized inside the otherwise-vector axes); text/axes remain
vector. pdf.fonttype=42 / svg.fonttype='none' keep text as editable font glyphs.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ------------------------------------------------------------
# (a) global publication rcParams
# ------------------------------------------------------------
def apply_style():
    """Set matplotlib rcParams for publication (Nature-tier). Idempotent."""
    rc = {
        # typography
        "font.size": 7,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.titlesize": 8,
        # lines / spines
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.5,
        # de-clutter
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        # output: vector-safe text, 300 dpi raster fallback
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "pdf.fonttype": 42,      # TrueType (editable text in Illustrator), not Type-3
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep text as <text>, not paths
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    }
    matplotlib.rcParams.update(rc)
    return rc


# ------------------------------------------------------------
# (b) despine a spatial (tissue) axis
# ------------------------------------------------------------
def despine_spatial(ax):
    """Tissue-map axis: hide ALL spines, remove ticks, equal aspect, invert y.
    invert_yaxis() so image/tissue coordinates (y grows downward) plot upright."""
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    # invert only if not already inverted (idempotent if called twice)
    ylim = ax.get_ylim()
    if ylim[0] < ylim[1]:
        ax.invert_yaxis()
    return ax


# ------------------------------------------------------------
# (c) micron scale bar
# ------------------------------------------------------------
def add_scalebar(ax, xy_um, length_um=500, lw=2.0, color="black",
                 label=None, fontsize=6, pad_frac=0.01):
    """Draw a horizontal scale bar of `length_um` microns with its LEFT end anchored
    at data-coordinate `xy_um=(x0, y0)` (true microns). A label (default '<n> µm') is
    placed just below the bar. Returns the line artist.

    Caller passes a data-space anchor (typically near the lower-left of the point cloud,
    computed from the cell coordinates) so the bar length is physically meaningful in µm.
    The y-axis is inverted on a despined spatial axis, so 'below' the bar means a LARGER
    data-y; we offset by a small fraction of the current y-range in the inverted sense."""
    x0, y0 = float(xy_um[0]), float(xy_um[1])
    line = Line2D([x0, x0 + length_um], [y0, y0], lw=lw, color=color,
                  solid_capstyle="butt", zorder=10)
    ax.add_line(line)
    if label is None:
        label = f"{int(length_um)} µm"
    # offset text away from the bar by a small fraction of the y span; respect inversion
    ylim = ax.get_ylim()                 # (top, bottom) when inverted -> ylim[0] > ylim[1]
    yspan = abs(ylim[1] - ylim[0])
    inverted = ylim[0] > ylim[1]
    dy = pad_frac * (yspan if yspan else 1.0) * 3.0
    ytext = y0 + dy if inverted else y0 - dy
    ax.text(x0 + length_um / 2.0, ytext, label, ha="center",
            va="top" if inverted else "bottom", fontsize=fontsize, color=color, zorder=10)
    return line


# ------------------------------------------------------------
# (d) colorblind-safe discrete palette
# ------------------------------------------------------------
# Wong (2011) Nature Methods 8:441 colorblind-safe 8-colour set, then extended with a
# few further distinguishable hues for when >8 categories are needed.
_WONG = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]
_EXTRA = [
    "#999999",  # grey
    "#7E2954",  # wine
    "#117733",  # dark green
    "#882255",  # mulberry
    "#44AA99",  # teal
    "#AA4499",  # purple
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#88CCEE",  # light blue
    "#661100",  # dark red
]
_PALETTE = _WONG + _EXTRA


def cb_palette(n=None):
    """Return a colorblind-safe discrete colour list. If n is None, return the full
    base palette; if n exceeds the curated list, cycle (with a stable order)."""
    if n is None:
        return list(_PALETTE)
    if n <= len(_PALETTE):
        return list(_PALETTE[:n])
    reps = int(np.ceil(n / len(_PALETTE)))
    return list((_PALETTE * reps)[:n])


def domain_colors(labels, unassigned="unassigned"):
    """Map an iterable of category labels to colorblind-safe colours. 'unassigned'
    (if present) is forced to a neutral light grey and does NOT consume a palette slot,
    so the meaningful domains keep stable, well-separated colours."""
    labels = [str(x) for x in labels]
    seen, ordered = set(), []
    for x in labels:                      # stable de-dup, preserve first-seen order
        if x not in seen:
            seen.add(x)
            ordered.append(x)
    non_un = [x for x in ordered if x != unassigned]
    pal = cb_palette(len(non_un))
    cmap = {lab: pal[i] for i, lab in enumerate(non_un)}
    if unassigned in ordered:
        cmap[unassigned] = "0.8"
    return cmap


# ------------------------------------------------------------
# (e) save both PDF and PNG @300 dpi
# ------------------------------------------------------------
def save_both(fig, basepath, dpi=300, close=True):
    """Save `fig` as BOTH <basepath>.pdf and <basepath>.png at `dpi`. `basepath` may
    include or omit an extension; any .pdf/.png/.svg suffix is stripped first. Returns
    (pdf_path, png_path). bbox_inches='tight' so legends placed outside the axes are kept."""
    bp = str(basepath)
    for ext in (".pdf", ".png", ".svg"):
        if bp.lower().endswith(ext):
            bp = bp[: -len(ext)]
            break
    pdf_path = bp + ".pdf"
    png_path = bp + ".png"
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return pdf_path, png_path
