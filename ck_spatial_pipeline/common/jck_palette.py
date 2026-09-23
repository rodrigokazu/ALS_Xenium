# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/common/jck_palette.py
#
# The multiome-paper palette, canonical for every plot in the project. NC_v2_downstream.py
# imports it flat from its own folder.
# ========================================================================================

"""
jck_palette.py
============================================================
The multiome-paper palette (Rodrigo, 2026-09-02) -- CANONICAL for every plot in
the ALS spinal-cord Xenium project (pairwise networks, Explorer artifact, decks).

Importable with no matplotlib at all (the dicts and the CSS helpers are plain
Python); the colormap helpers import matplotlib lazily and never touch pyplot,
so the caller keeps ownership of the backend (matplotlib.use("Agg") first).

    import jck_palette as jck
    jck.STATUS["control"]                 -> "#8FA7C7"
    jck.get_node_color("Motor neuron (flagged)")  -> "#00868B"  (legacy alias of "Motor neurons")
    jck.get_node_color("Something odd")   -> "#6d7773" + ONE logged warning (never a crash)
    jck.zscore_cmap()                     -> neutral RdBu_r copy (grey for masked cells)
    jck.wdr49_tier_cmap()                 -> ListedColormap of the 1 / 2 / 3 / 4+ tiers

Source palette keys are kept beside every mapped label so a reviewer can trace
each colour back to the multiome-paper figure legend.

Legacy colours that must not appear on any surface any more (Wong / Okabe-Ito
set used until 2026-09-02) are listed in LEGACY_HEX; legacy_hex_counts(text)
scans a file for them.
"""

from __future__ import annotations

import colorsys
import logging
import re

log = logging.getLogger("jck_palette")

PALETTE_NAME = "jck_multiome_2026-09-02"
PALETTE_VERSION = "2026-09-02"

# --------------------------------------------------------------------------- #
# Neutral
# --------------------------------------------------------------------------- #

NEUTRAL = "#6d7773"          # pooled / mixed / unknown -- the one sanctioned neutral grey

# --------------------------------------------------------------------------- #
# Status / genotype (obs['status'] values: control, c9, sporadic)
# --------------------------------------------------------------------------- #

STATUS = {
    "control": "#8FA7C7",    # source key "con"
    "c9": "#C9A08F",         # source key "c9_ALS"
    "sporadic": "#A5B7A1",   # source key "sals_ALS"
    "MIXED": NEUTRAL,        # pooled / mixed sections
    "pooled": NEUTRAL,
    "unknown": NEUTRAL,      # status could not be resolved: neutral, no warning (a known state)
}
STATUS_ORDER = ["control", "sporadic", "c9"]
STATUS_SOURCE_KEY = {"control": "con", "c9": "c9_ALS", "sporadic": "sals_ALS"}
# spellings seen across the project's CSVs / configs -> canonical status key
STATUS_ALIASES = {
    "control": "control", "ctrl": "control", "con": "control", "healthy": "control",
    "c9": "c9", "c9orf72": "c9", "c9_als": "c9", "c9-als": "c9", "c9als": "c9",
    "sporadic": "sporadic", "sals": "sporadic", "sals_als": "sporadic", "sporadic_als": "sporadic",
    "mixed": "MIXED", "pooled": "MIXED", "unknown": "unknown",
}

# --------------------------------------------------------------------------- #
# Cell types -- the 9 coarse spinal-cord classes on the source palette
# --------------------------------------------------------------------------- #

CELLTYPE = {
    "Astrocytes": "#9E78BA",            # ASC
    "Oligodendrocytes": "#6C4BA6",      # ODC
    "OPCs": "#8B8DC0",                  # OPC
    "Microglia": "#EEEEA8",             # MGC
    "Macrophages": "#C9752A",           # VLMC slot (no macrophage entry in the source palette)
    "Endothelial": "#F2D07A",           # Endo
    "Excitatory neurons": "#2E83B0",    # IT L5
    "Inhibitory neurons": "#E8627A",    # PVALB
    "Other neurons": "#F9B446",         # ET L5 -- coarse "Motor neurons" class minus the curated cells
    "WDR49+ Astrocyte": "#FF0000",      # ASC_WDR49 / "WDR49+ Astrocytes"
}
CELLTYPE_SOURCE_LABEL = {
    "Astrocytes": "ASC", "Oligodendrocytes": "ODC", "OPCs": "OPC", "Microglia": "MGC",
    "Macrophages": "VLMC", "Endothelial": "Endo", "Excitatory neurons": "IT L5",
    "Inhibitory neurons": "PVALB", "Other neurons": "ET L5", "WDR49+ Astrocyte": "ASC_WDR49",
}
WDR49_COLOR = CELLTYPE["WDR49+ Astrocyte"]

# --------------------------------------------------------------------------- #
# Motor-neuron trio (teal / magenta are absent from the source palette)
# --------------------------------------------------------------------------- #

MN = {
    "all": "#00868B",        # Motor neurons (is_MN_v2, all)          teal
    "tdppos": "#C71585",     # Motor neuron TDP+                      magenta (pathology pops)
    "tdpneg": "#7FCDCD",     # Motor neuron TDP-                      light teal (MN family)
}
# role spellings used by pairwise_v3 node_counts.csv / run configs / this script
MN_ROLE = {
    "mn": MN["all"], "all": MN["all"],
    "mn_pos": MN["tdppos"], "mn_tdppos": MN["tdppos"], "tdppos": MN["tdppos"],
    "mn_neg": MN["tdpneg"], "mn_tdpneg": MN["tdpneg"], "tdpneg": MN["tdpneg"],
}

# --------------------------------------------------------------------------- #
# Node labels (exact strings used by pairwise_v3 / pw_v3_stats_figs / Explorer)
# --------------------------------------------------------------------------- #

NODE = dict(CELLTYPE)
NODE.update({
    "WDR49+ Astrocytes": WDR49_COLOR,             # plural spelling (multiome legend)
    "Motor neurons": MN["all"],                   # is_MN_v2, all
    "Motor neuron (flagged)": MN["all"],          # LEGACY node name == Motor neurons (all)
    "Motor neuron TDP+": MN["tdppos"],
    "Motor neuron TDP−": MN["tdpneg"],       # U+2212 minus (canonical spelling)
    "Motor neuron TDP-": MN["tdpneg"],            # ASCII hyphen spelling
})

# --------------------------------------------------------------------------- #
# Sequential / diverging ramps
# --------------------------------------------------------------------------- #

RAMPS = {
    # WDR49 count tiers 1 / 2 / 3 / 4+
    "wdr49_tiers": ["#FDBFA8", "#F06B4F", "#FF0000", "#8B0000"],
    # expression dot ramps: light surface -> accent (surface = white in print)
    "mn_expression": ["#FFFFFF", MN["all"]],
    "wdr49_expression": ["#FFFFFF", WDR49_COLOR],
    # palette diverging option for z-score heatmaps (negative · zero · positive)
    "zscore_diverging": ["#2E83B0", "#FFFFFF", "#C71585"],
}
WDR49_TIER_LABELS = ["1", "2", "3", "4+"]
# neutral diverging map for z heatmaps: chosen where the semantic hues (red WDR49,
# teal / magenta / light-teal MN) are drawn ON the heatmap as strokes or labels,
# so no cell colour can be mistaken for a node colour
ZSCORE_NEUTRAL_CMAP = "RdBu_r"
MASKED_CELL = "#eeeeee"      # structural: "pair absent" cell in a heatmap

# --------------------------------------------------------------------------- #
# CSS tokens for the Explorer artifact (light values; dark variants via lighten)
# --------------------------------------------------------------------------- #

CSS_TOKENS = {
    "--wdr": WDR49_COLOR, "--mn": MN["all"], "--mn-tdppos": MN["tdppos"], "--mn-tdpneg": MN["tdpneg"],
    "--ctl": STATUS["control"], "--spo": STATUS["sporadic"], "--c9": STATUS["c9"],
    "--ct-astro": CELLTYPE["Astrocytes"], "--ct-oligo": CELLTYPE["Oligodendrocytes"],
    "--ct-opc": CELLTYPE["OPCs"], "--ct-micro": CELLTYPE["Microglia"],
    "--ct-macro": CELLTYPE["Macrophages"], "--ct-endo": CELLTYPE["Endothelial"],
    "--ct-exc": CELLTYPE["Excitatory neurons"], "--ct-inh": CELLTYPE["Inhibitory neurons"],
    "--ct-other": CELLTYPE["Other neurons"],
    "--wdr-t1": RAMPS["wdr49_tiers"][0], "--wdr-t2": RAMPS["wdr49_tiers"][1],
    "--wdr-t3": RAMPS["wdr49_tiers"][2], "--wdr-t4": RAMPS["wdr49_tiers"][3],
}
# the muted status colours are lightened ~10 % L on the dark theme; saturated ones stay
DARK_LIGHTEN_TOKENS = ("--ctl", "--spo", "--c9")

# --------------------------------------------------------------------------- #
# Legacy set (Wong / Okabe-Ito) that must disappear from every surface
# --------------------------------------------------------------------------- #

LEGACY_HEX = (
    "#0072B2",   # WDR49 blue
    "#D55E00",   # MN vermillion
    "#009E73",   # control green
    "#E69F00",   # orange (sporadic in the Explorer; TDP- MN in the old figure script)
    "#CC79A7",   # c9 pink
    "#56B4E9",   # sky blue (sporadic in the old figure script)
    "#3092d4", "#e26a1e",              # dark-mode variants
    "#9ecae1", "#3b8dc4", "#023858",   # WDR tier blues
)
LEGACY_CSS = ("rgb(213,94,0)", "rgb(213, 94, 0)")   # gene-dot vermillion ramp target


def legacy_hex_counts(text: str, skip_line_prefixes: tuple = ()) -> dict:
    """Case-insensitive count of every legacy colour in `text` (hex + CSS forms).
    Lines starting with any of `skip_line_prefixes` (e.g. base64 image blobs)
    are excluded from the scan."""
    if skip_line_prefixes:
        text = "\n".join(l for l in text.split("\n") if not l.startswith(skip_line_prefixes))
    low = text.lower()
    counts = {h: low.count(h.lower()) for h in LEGACY_HEX}
    for css in LEGACY_CSS:
        counts[css] = low.count(css.lower())
    return counts


# --------------------------------------------------------------------------- #
# Lookups (never crash; unknown -> NEUTRAL with ONE logged warning per label)
# --------------------------------------------------------------------------- #

_DASHES = dict.fromkeys(map(ord, "−–—‐‑"), "-")


def _norm(label) -> str:
    """Whitespace-collapsed, dash-unified, case-folded label for alias matching."""
    return " ".join(str(label).translate(_DASHES).split()).casefold()


_NODE_BY_NORM = {_norm(k): v for k, v in NODE.items()}
_CELLTYPE_BY_NORM = {_norm(k): v for k, v in CELLTYPE.items()}
_warned: set = set()


def _warn_once(kind: str, label, colour: str) -> None:
    key = (kind, str(label))
    if key not in _warned:
        _warned.add(key)
        log.warning("jck_palette: no %s colour for %r -> neutral grey %s", kind, label, colour)


def get_status_color(status, default: str = NEUTRAL) -> str:
    """control / sporadic / c9 (any project spelling) -> palette; MIXED / pooled /
    unknown -> neutral without a warning; anything else -> `default` + warning."""
    key = STATUS_ALIASES.get(_norm(status).replace(" ", "_"))
    if key is None:
        key = STATUS_ALIASES.get(_norm(status).replace(" ", "-"))
    if key is not None:
        return STATUS[key]
    _warn_once("status", status, default)
    return default


def get_celltype_color(label, default: str = NEUTRAL) -> str:
    col = CELLTYPE.get(str(label)) or _CELLTYPE_BY_NORM.get(_norm(label))
    if col is not None:
        return col
    _warn_once("cell-type", label, default)
    return default


def get_node_color(label, fallback: str | None = None, default: str = NEUTRAL) -> str:
    """Node colour precedence: exact node label -> palette (legacy 'Motor neuron
    (flagged)' == Motor neurons); dash/case-normalised alias -> palette; else the
    caller's `fallback` (e.g. a colour derived from the node's role) when given;
    else `default` (neutral grey) with a logged warning. Never raises."""
    col = NODE.get(str(label))
    if col is None:
        col = _NODE_BY_NORM.get(_norm(label))
    if col is not None:
        return col
    if fallback is not None:
        return fallback
    _warn_once("node", label, default)
    return default


def node_colors(labels) -> dict:
    return {l: get_node_color(l) for l in labels}


def status_colors(order=None) -> dict:
    return {s: get_status_color(s) for s in (order or STATUS_ORDER)}


def wdr49_tier_color(count) -> str | None:
    """WDR49 count -> tier colour (1 / 2 / 3 / 4+); None for 0, NaN or negative."""
    try:
        c = float(count)
    except (TypeError, ValueError):
        return None
    if not c >= 1:
        return None
    return RAMPS["wdr49_tiers"][min(int(c), 4) - 1]


# --------------------------------------------------------------------------- #
# Colour arithmetic (no matplotlib)
# --------------------------------------------------------------------------- #

def hex_to_rgb(hex_colour: str) -> tuple:
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, int(round(v)))) for v in rgb))


def css_rgb(hex_colour: str) -> str:
    return "rgb({},{},{})".format(*hex_to_rgb(hex_colour))


def lighten(hex_colour: str, dl: float = 0.10) -> str:
    """Raise HLS lightness by `dl` (absolute, clipped to [0, 1]); `dl` < 0 darkens."""
    r, g, b = (v / 255.0 for v in hex_to_rgb(hex_colour))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + dl))
    return rgb_to_hex(tuple(255 * v for v in colorsys.hls_to_rgb(h, l, s)))


def css_token_values(dark: bool = False, dl: float = 0.10) -> dict:
    """CSS variable -> hex for the Explorer's three token blocks (dark lightens
    the muted status tokens by `dl` L; every other token is theme-invariant)."""
    if not dark:
        return dict(CSS_TOKENS)
    return {k: (lighten(v, dl) if k in DARK_LIGHTEN_TOKENS else v) for k, v in CSS_TOKENS.items()}


def css_token_block(dark: bool = False, indent: str = "  ") -> str:
    return "\n".join(f"{indent}{k}: {v};" for k, v in css_token_values(dark).items())


# --------------------------------------------------------------------------- #
# matplotlib helpers (lazy import; pyplot never imported here)
# --------------------------------------------------------------------------- #

def _mpl_colors():
    import matplotlib.colors as mcolors  # noqa: WPS433 (lazy on purpose)
    return mcolors


def _mpl_cmap(name: str):
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except AttributeError:  # matplotlib < 3.5
        import matplotlib.cm as cm
        return cm.get_cmap(name)


def sequential_cmap(target: str, name: str | None = None, surface: str = "#FFFFFF", n: int = 256):
    """Light surface -> `target` accent (expression dot ramps and the like)."""
    mcolors = _mpl_colors()
    return mcolors.LinearSegmentedColormap.from_list(name or f"jck_seq_{target.lstrip('#')}",
                                                     [surface, target], N=n)


def mn_expression_cmap(surface: str = "#FFFFFF"):
    return sequential_cmap(MN["all"], "jck_mn_expression", surface)


def wdr49_expression_cmap(surface: str = "#FFFFFF"):
    return sequential_cmap(WDR49_COLOR, "jck_wdr49_expression", surface)


def wdr49_tier_cmap():
    """Discrete 4-step map for WDR49 count tiers (1 / 2 / 3 / 4+)."""
    return _mpl_colors().ListedColormap(RAMPS["wdr49_tiers"], name="jck_wdr49_tiers")


def status_listed_cmap(order=None):
    order = list(order or STATUS_ORDER)
    return _mpl_colors().ListedColormap([get_status_color(s) for s in order], name="jck_status")


def zscore_cmap(kind: str = "neutral", bad: str = MASKED_CELL):
    """Diverging map for z-score heatmaps.
    kind="neutral" (default): a copy of RdBu_r -- the choice wherever the
        semantic node hues (red WDR49, teal / magenta / light-teal MN) are drawn
        on the same heatmap as strokes or tick labels.
    kind="palette": blue #2E83B0 (negative) · white · magenta #C71585 (positive),
        the palette's own diverging option (avoid when a magenta TDP+ stroke sits
        on the same panel: the positive pole would swallow it).
    Masked / NaN cells render in `bad`."""
    mcolors = _mpl_colors()
    if kind == "palette":
        cmap = mcolors.LinearSegmentedColormap.from_list("jck_zscore_bwm", RAMPS["zscore_diverging"], N=256)
    elif kind == "neutral":
        cmap = _mpl_cmap(ZSCORE_NEUTRAL_CMAP).copy()
    else:
        raise ValueError(f"zscore_cmap kind must be 'neutral' or 'palette', got {kind!r}")
    cmap.set_bad(bad)
    return cmap


def register_colormaps(force: bool = True) -> list:
    """Register jck_* colormaps with matplotlib so cmap="jck_wdr49_tiers" etc.
    work in scanpy / pyplot calls. Returns the registered names; idempotent."""
    import matplotlib
    cmaps = [wdr49_tier_cmap(), mn_expression_cmap(), wdr49_expression_cmap(),
             zscore_cmap("palette"), status_listed_cmap()]
    names = []
    for cm_ in cmaps:
        try:
            matplotlib.colormaps.register(cm_, name=cm_.name, force=force)
        except (AttributeError, ValueError):  # old matplotlib or already present without force
            try:
                import matplotlib.cm as cm
                cm.register_cmap(name=cm_.name, cmap=cm_)
            except Exception:  # pragma: no cover
                continue
        names.append(cm_.name)
    return names


_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _selfcheck() -> None:
    for d in (STATUS, CELLTYPE, MN, NODE, CSS_TOKENS):
        for k, v in d.items():
            assert _HEX_RE.match(v), (k, v)
    for ramp in RAMPS.values():
        for v in ramp:
            assert _HEX_RE.match(v), ramp
    assert get_node_color("Motor neuron (flagged)") == MN["all"]
    assert get_node_color("Motor neuron TDP−") == get_node_color("Motor neuron TDP-") == MN["tdpneg"]
    assert get_status_color("c9_ALS") == STATUS["c9"] and get_status_color("con") == STATUS["control"]
    legacy = {h.lower() for h in LEGACY_HEX}
    used = {v.lower() for d in (STATUS, CELLTYPE, MN, NODE) for v in d.values()}
    used |= {v.lower() for ramp in RAMPS.values() for v in ramp}
    assert not (legacy & used), legacy & used


_selfcheck()

__all__ = [
    "PALETTE_NAME", "PALETTE_VERSION", "NEUTRAL", "STATUS", "STATUS_ORDER", "STATUS_ALIASES",
    "CELLTYPE", "WDR49_COLOR", "MN", "MN_ROLE", "NODE", "RAMPS", "WDR49_TIER_LABELS",
    "ZSCORE_NEUTRAL_CMAP", "MASKED_CELL", "CSS_TOKENS", "LEGACY_HEX", "LEGACY_CSS",
    "legacy_hex_counts", "get_status_color", "get_celltype_color", "get_node_color",
    "node_colors", "status_colors", "wdr49_tier_color", "hex_to_rgb", "rgb_to_hex", "css_rgb",
    "lighten", "css_token_values", "css_token_block", "sequential_cmap", "mn_expression_cmap",
    "wdr49_expression_cmap", "wdr49_tier_cmap", "status_listed_cmap", "zscore_cmap",
    "register_colormaps",
]
