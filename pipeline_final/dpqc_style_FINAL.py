#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/dpqc_style_FINAL.py
#
# One place for the dual-pass QC colour system, so no colour ever means two things inside the
# same legend.
#
# Four registries, each with a different scope. Cell type colours are global and fixed,
# because cell_type_coarse is the only label that genuinely means the same thing in every
# section, so it is the one thing that earns a hard-coded palette. Status colours cover
# control, sporadic and C9 as a grey baseline with a warm ALS pair. Verdict colours for KEEP /
# REVIEW / REMOVE are colourblind-safe and deliberately not red-green. Domain colours are
# generated per sample by cell count through nature_style.domain_colors.
#
# The name carries _FINAL only so this suite never overwrites the _gausss original. The
# content is the same.
# ========================================================================================

"""dpqc_style_FINAL.py -- single source of truth for the Dual-pass QC colour system + shared figure
helpers (_FINAL re-run; content identical to the proven _gausss dpqc_style, renamed so the _FINAL
suite never clobbers the _gausss originals). Four independent colour registries so a colour never
means two things within its legend:

  CELLTYPE_COLORS  GLOBAL, FIXED, biologically mnemonic (cell_type_coarse is the ONLY label
                   comparable across samples -> it gets the one hard-coded palette).
  STATUS_COLORS    disease status (control / sporadic / c9), a grey-baseline + warm-ALS family.
  FLAG_COLORS      KEEP / REVIEW / REMOVE verdict, deliberately colourblind-safe and NOT red-green.
  domain colours   PER-SAMPLE INDEPENDENT -> derived per sample by descending cell count via
                   nature_style.domain_colors (a fixed global domain palette would be a lie).

Imports nature_style for apply_style / despine_spatial / add_scalebar / save_both / cb_palette.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

VH = "/home/rodrigok/SLURM_jobs/Spatial/VH_isolation"
if VH not in sys.path:
    sys.path.insert(0, VH)
import nature_style as ns                       # apply_style, despine_spatial, add_scalebar, save_both, domain_colors, cb_palette

# ---------------------------------------------------------------- (1) cell types (global, fixed)
CELLTYPE_COLORS = {
    "Oligodendrocyte":     "#0072B2",   # blue  -- white-matter myelin anchor
    "OPC":                 "#56B4E9",   # light blue -- oligo lineage sibling
    "Astrocyte":           "#009E73",   # bluish green -- GFAP/AQP4 mnemonic
    "Microglia":           "#CC79A7",   # reddish purple -- immune
    "Macrophage_perivasc": "#661100",   # dark red -- perivascular myeloid
    "Lymphoid":            "#882255",   # mulberry -- other immune
    "Endothelial":         "#D55E00",   # vermillion -- blood / vessel
    "Mural":               "#7E2954",   # wine -- pericyte/vSMC
    "Fibroblast_VLMC":     "#DDCC77",   # sand -- meningeal / stroma
    "Neuron_excit":        "#E69F00",   # orange -- neuron (warm)
    "Neuron_inhib":        "#F0E442",   # yellow -- inhibitory neuron
    "MotorNeuron":         "#000000",   # black -- the hero cell, max contrast
    "Untyped":             "#CCCCCC",   # light grey -- ALWAYS shown, never dropped (diagnostic)
}
# canonical legend order (typed set first in a sensible biological order, Untyped last)
CELLTYPE_ORDER = ["Oligodendrocyte", "OPC", "Astrocyte", "Microglia", "Macrophage_perivasc",
                  "Lymphoid", "Endothelial", "Mural", "Fibroblast_VLMC", "Neuron_excit",
                  "Neuron_inhib", "MotorNeuron", "Untyped"]

# ---------------------------------------------------------------- (2) disease status
STATUS_COLORS = {"control": "#999999", "sporadic": "#E69F00", "c9": "#D55E00"}
STATUS_LABEL = {"control": "Control", "sporadic": "Sporadic ALS", "c9": "C9orf72 ALS"}

# ---------------------------------------------------------------- (3) keep/review/remove verdict
FLAG_COLORS = {"KEEP": "#44AA99", "REVIEW": "#E69F00", "REMOVE": "#AA4499"}
FLAG_ORDER = ["KEEP", "REVIEW", "REMOVE"]


def celltype_color(t):
    return CELLTYPE_COLORS.get(str(t), "#CCCCCC")


def status_color(s):
    return STATUS_COLORS.get(str(s).lower(), "#999999")


def status_label(s):
    return STATUS_LABEL.get(str(s).lower(), str(s))


def flag_color(v):
    return FLAG_COLORS.get(str(v).upper(), "#999999")


def get_color(kind, label):
    return {"celltype": celltype_color, "status": status_color, "flag": flag_color}[kind](label)


def domain_palette(domains_by_size):
    """Per-sample domain colour map. `domains_by_size` = iterable of domain labels ALREADY ordered
    by DESCENDING cell count (largest first). Uses nature_style.domain_colors so the largest domain
    takes palette slot 0 and 'unassigned' is forced grey without consuming a slot."""
    return ns.domain_colors(list(domains_by_size))


def celltype_handles(present=None):
    labs = [t for t in CELLTYPE_ORDER if (present is None or t in present)]
    return [Patch(facecolor=CELLTYPE_COLORS[t], edgecolor="none", label=t) for t in labs]


def flag_handles():
    return [Patch(facecolor=FLAG_COLORS[f], edgecolor="none", label=f.title()) for f in FLAG_ORDER]


def status_handles():
    return [Patch(facecolor=STATUS_COLORS[s], edgecolor="none", label=STATUS_LABEL[s])
            for s in ("control", "sporadic", "c9")]


def colour_key_figure(figsize=(8.27, 5.2)):
    """Render the four registries as one shared 'colour key' page for the PDFs."""
    ns.apply_style()
    fig = plt.figure(figsize=figsize); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.05, 0.03, 0.9, 0.9]); ax.axis("off")
    ax.text(0.0, 1.0, "Colour key", fontsize=15, fontweight="bold", va="top")

    def block(y0, title, items):
        ax.text(0.0, y0, title, fontsize=10, fontweight="bold", va="top")
        y = y0 - 0.055
        for lab, col in items:
            ax.add_patch(plt.Rectangle((0.02, y - 0.028), 0.05, 0.03, facecolor=col,
                                       edgecolor="0.4", lw=0.4, transform=ax.transAxes))
            ax.text(0.09, y - 0.012, lab, fontsize=8.5, va="center")
            y -= 0.042
        return y

    y = 0.90
    y = block(y, "Cell types (global, fixed)",
              [(t, CELLTYPE_COLORS[t]) for t in CELLTYPE_ORDER]) - 0.02
    # right column
    ax2 = fig.add_axes([0.52, 0.03, 0.45, 0.9]); ax2.axis("off")

    def block2(y0, title, items):
        ax2.text(0.0, y0, title, fontsize=10, fontweight="bold", va="top")
        y = y0 - 0.055
        for lab, col in items:
            ax2.add_patch(plt.Rectangle((0.02, y - 0.028), 0.09, 0.03, facecolor=col,
                                        edgecolor="0.4", lw=0.4, transform=ax2.transAxes))
            ax2.text(0.15, y - 0.012, lab, fontsize=8.5, va="center")
            y -= 0.05
        return y

    yy = 0.90
    yy = block2(yy, "Disease status", [(STATUS_LABEL[s], STATUS_COLORS[s])
                                       for s in ("control", "sporadic", "c9")]) - 0.03
    yy = block2(yy, "QC verdict (keep / review / remove)",
                [(f.title(), FLAG_COLORS[f]) for f in FLAG_ORDER]) - 0.03
    ax2.text(0.0, yy, "Novae domains", fontsize=10, fontweight="bold", va="top")
    ax2.text(0.02, yy - 0.05,
             "per-sample independent: colours are assigned by\n"
             "descending cell count WITHIN each sample and do\n"
             "NOT map to the same biology across samples.",
             fontsize=8, va="top")
    return fig


if __name__ == "__main__":
    fig = colour_key_figure()
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dpqc_colourkey.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
