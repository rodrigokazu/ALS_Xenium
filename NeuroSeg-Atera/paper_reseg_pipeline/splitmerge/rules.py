"""Full-resolution split/merge rules for the 18S-Vimentin resegmentation.

Everything here runs at Xenium level-0 resolution on the float difference
signal and every size is in um, so a crop's downsampling factor for the web
view never changes a result. The browser preview in artifact/index.html is a
live, slider-driven approximation of the first step only; this module is the
reference implementation the methods section describes.

Pipeline, per crop (driver: data_prep/10_build_splitmerge_rules.py):
  1. segment()               median filter -> percentile threshold -> fill holes -> min size, on the
                             18S - Vimentin signal (each channel scaled from its whole-sample tissue median to
                             its p99, sample_signal); run twice (main + sense check)
  2. rescue_nuclei()         DAPI blobs Xenium's nucleus segmentation missed become extra nuclei, only inside
                             an 18S mask (pale neuronal nuclei); never in the neuropil
  3. nucleus_owners()        a nucleus belongs to a mask if >= half of it lies inside, or the mask wraps
                             >= half of its outline
  4. compute_flags()         MULTI / SENSE_SPLIT / PARTIAL / NO_NUCLEUS / NONE + nucleus-type mismatch
  5. resolve()               somata from raw-18S peaks (half-maximum rule); extra nuclei in a soma become
                             satellites; dim multi-nucleus masks split per nucleus; waist split as fallback
  6. carve_satellites()      nuclei on the edge of a large mask (and unowned ones within 2 um) get their
                             own cell carved from it
  7. absorb_owned_nuclei()   every cell contains its whole nucleus
  8. classify_pieces()       nucleus status x shape (PCA elongation, inscribed radius) x transcript gate
  9. merge_dendrites()       a no-nucleus thin+long piece joins a nearby likely neuron unless glial/unlike it
 10. add_fallback_cells()    nuclei still without a cell get Xenium's own cell, capped at 3 um, clipped
 11. merge_touching_pieces() no-nucleus 18S pieces overlapping a fallback cell's nucleus join that cell
     segger_extend()         comparison only (not part of the final masks)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import expand_labels, watershed

UM_PER_PX = 0.2125

FLAG_NONE, FLAG_MULTI, FLAG_NO_NUCLEUS, FLAG_PARTIAL, FLAG_SENSE_SPLIT = 0, 1, 2, 3, 4
FLAG_NAMES = {FLAG_NONE: "none", FLAG_MULTI: "multi_nucleus", FLAG_NO_NUCLEUS: "no_nucleus",
              FLAG_PARTIAL: "partial_nucleus", FLAG_SENSE_SPLIT: "sense_split"}

SHAPE_SMALL, SHAPE_ROUND, SHAPE_THIN_LONG = 0, 1, 2
SHAPE_NAMES = {SHAPE_SMALL: "small", SHAPE_ROUND: "round", SHAPE_THIN_LONG: "thin_long"}

CLASS_CELL = "cell"                      # has a nucleus, not dendrite-shaped
CLASS_MASK_NO_NUCLEUS = "mask_no_nucleus"  # round, no nucleus: kept on shape alone, unconfirmed
CLASS_DENDRITE = "dendrite"              # thin+long, passed the transcript gate
CLASS_DENDRITE_NUCLEUS = "dendrite_with_nucleus"
CLASS_REJECT_SMALL = "rejected_small"
CLASS_REJECT_DENDRITE = "rejected_low_transcripts"
CLASS_FALLBACK = "xenium_fallback"        # nucleus with no 18S cell: Xenium's own cell polygon

GLIAL_MARKERS = {
    "oligodendrocyte": ["MBP", "MOBP", "MOG", "MAG", "OLIG1", "OLIG2", "SOX10", "ST18", "OPALIN"],
    "astrocyte": ["AQP4", "SLC1A2", "GJA1", "SOX9"],
    "microglia": ["C1QC", "P2RY12", "PTPRC"],
}
GLIAL_GENES = sorted({g for gs in GLIAL_MARKERS.values() for g in gs})


@dataclass(frozen=True)
class Params:
    median_px: int = 10                   # SEG_DEFAULTS in index.html, full-res px
    # percentiles of the WHOLE-SAMPLE tissue signal (scg/sample_stats), one pair for every sample; calibrated by
    # data_prep/16 to reproduce the reviewed crops (18S-mask Dice 0.97 / 0.98 on SD03914_BG / SD04219_BI, every
    # neuron-sized reviewed cell kept as one cell)
    thr_percentile: float = 94.25
    min_size_px: int = 200
    alt_median_px: int = 1                # sense check: no smoothing ...
    alt_thr_percentile: float = 97.25     # ... and a stricter percentile, on the float signal
    assign_frac: float = 0.5              # share of a nucleus (or alt component) that must lie inside to count
    partial_wrap_frac: float = 0.5        # share of a partial nucleus's boundary inside the mask -> it wraps
    neuron_nucleus_min_um2: float = 55.0  # see CHANGELOG: matches neuron_predictions on every crop nucleus >= 60um2
    glia_rim_um: float = 2.0              # territory a glial nucleus keeps when sharing a mask with a neuron
    glia_mask_over_nucleus: float = 4.0   # glial nucleus inside a mask >= this x its area -> mismatch, try waist
    dendrite_max_half_width_um: float = 3.0
    dendrite_min_elongation: float = 2.2
    small_area_um2: float = 15.0
    min_dendrite_transcripts: int = 10
    max_waist_rounds: int = 30
    # a waist cut is kept only where the mask really narrows: the cut's largest inscribed radius (distance to the
    # mask edge along the cut) at most this fraction of the smaller piece's own inscribed radius. A convex soma
    # has no such narrowing; without this test the distance-transform regrowth cut single somata along a straight
    # medial ridge (2026-09-30, straight cuts found at the pathologists' pins). None disables the test.
    waist_max_neck_frac: float | None = 0.5
    # per-component closing of the main 18S masks before holes are filled: granular (Nissl-clumped) somata
    # otherwise pass the threshold as a lacy mask. Each component is closed on its own, so neighbours are never
    # bridged. 0 disables it.
    soma_closing_um: float = 1.0
    # relative edge for large 18S masks (likely neurons, >= likely_neuron_soma_um2): the axon hillock and some
    # soma ends carry little Nissl, so their 18S sits far below the body's yet well above background, and the
    # whole-sample threshold cut them off with a ragged line (2026-09-30, SD016_20_BI). Such a mask grows into
    # connected pixels whose lightly smoothed raw 18S (above background) is >= soma_grow_frac of the mask's own
    # median, never below soma_grow_floor_frac of the sample's 18S range, at most soma_grow_max_um away, and never
    # into another mask or a nucleus it does not overlap. A region within soma_grow_clear_um of another mask or of
    # a nucleus the mask does not overlap is never added (its owner is ambiguous). Shape rule, two routes only:
    #   soma lobe          thick (survives an opening of soma_grow_lobe_um radius, >= soma_grow_min_lobe_um2),
    #                      attached broadly to the soma body (the mask after an opening of the dendrite
    #                      half-width): body contact length / sqrt(lobe area) >= soma_grow_min_body_contact, and
    #                      with not much around: >= soma_grow_isolation_um from any other mask or foreign nucleus
    #   process extension  thin and long (dendrite_max_half_width_um, dendrite_min_elongation), >= 5 um2, touching
    #                      the mask only at an existing process (the part an opening of the dendrite half-width
    #                      removes) over a short contact, i.e. it continues a process instead of lining an edge,
    #                      and, like a soma lobe, >= soma_grow_isolation_um from any other mask or foreign nucleus
    # so the blur rim around every bright soma, bumps on a soma's side and patches wrapped round a neighbour's
    # nucleus are left out. None disables it.
    soma_grow_frac: float | None = 0.2
    soma_grow_floor_frac: float = 0.15
    soma_grow_sigma_um: float = 0.75
    soma_grow_max_um: float = 15.0
    soma_grow_lobe_um: float = 1.5
    soma_grow_min_lobe_um2: float = 20.0
    soma_grow_clear_um: float = 1.0
    soma_grow_min_body_contact: float = 1.0
    soma_grow_isolation_um: float = 3.0
    # Vimentin-blind rescue (2026-10-01, pathologists' pins): the signal is 18S minus Vimentin, and the Vimentin channel of a
    # lipofuscin-laden neuron is filled with bright autofluorescent granules, so its soma (or the dim half of it) cancels out of
    # the signal and gets no mask, or only its bright part. The same threshold on 18S alone finds it. Only soma-sized, thick, compact
    # blobs of that 18S-only mask are taken (thin Vimentin-positive glial processes and cell bodies are far smaller and thinner):
    # >= vim_rescue_min_um2, holding a disk of vim_rescue_min_radius_um radius, solidity >= vim_rescue_min_solidity, mostly
    # (> 1 - vim_rescue_max_overlap) not yet masked; what is added is the part outside every mask and foreign nucleus, cleaned by
    # an opening of vim_rescue_open_um radius, joined to the one mask it touches (two masks -> skipped, ambiguous). None disables.
    vim_rescue_min_um2: float | None = None
    vim_rescue_min_radius_um: float = 5.0
    vim_rescue_min_solidity: float = 0.75
    vim_rescue_max_overlap: float = 0.6
    vim_rescue_open_um: float = 2.5
    vim_rescue_min_add_um2: float = 100.0
    # which nucleus rules the soma-resolution steps use (2026-09-30, pathologists' pins); each switchable so its
    # effect on the reviewed regions can be measured on its own:
    #   basin_nucleus        soma-peak basins without a nucleus of their own join their neighbour: "off", "any"
    #                        (any nucleus >= 50% inside counts) or "enclosed" (only enclosed_nuclei count)
    #   soma_keeper          nucleus that keeps a multi-nucleus soma: "largest_neuron" (the neuron-sized one, else
    #                        the most central), "interior_neuron" (the same among nuclei >= radius + 1 um from the
    #                        edge) or "deepest" (the enclosed one deepest beyond its radius, any size)
    #   neuron_split_enclosed two neurons only for two ENCLOSED neuron-sized nuclei (judged before the extension)
    #   satellite_keep_deepest carve_satellites never carves a piece's deepest enclosed owned nucleus
    basin_nucleus: str = "enclosed"
    soma_keeper: str = "interior_neuron"
    neuron_split_enclosed: bool = True
    satellite_keep_deepest: bool = False
    soma_peak_sigma_um: float = 1.5        # smoothing of the raw 18S channel for soma peaks
    soma_candidate_h: float = 0.05         # candidate peaks (h-maxima, share of the mask's p99 raw 18S)
    soma_max_saddle_frac: float = 0.5      # half-maximum rule: two somata are separate only if the signal
                                           # between them drops below half the dimmer one's peak
    likely_neuron_soma_um2: float = 150.0  # cell mask this large counts as a likely neuron even without a neuron-sized nucleus
    merge_max_gap_um: float = 3.0          # "unless they are distant to each other"
    merge_max_glial_frac: float = 0.3      # dendrite with >= this share of glial-marker transcripts is glial, not merged
    merge_min_cosine: float = 0.2          # sqrt-count profile similarity dendrite vs neuron
    merge_min_transcripts_for_profile: int = 5  # below this the profile checks are skipped (proximity only)
    rescue_dapi_frac: float = 0.5         # missed-nucleus rescue: DAPI >= this x median DAPI inside Xenium nuclei
    rescue_min_um2: float = 10.0
    rescue_max_um2: float = 150.0
    rescue_max_elongation: float = 2.5
    rescue_min_solidity: float = 0.9      # ~ the 10th percentile of Xenium's own nuclei (0.91-0.93); drops crescents
    contrast_ring_um: float = 3.0         # QC only: ring for the DAPI contrast of Xenium's nuclei (stats)
    # SNR-anchored main threshold (hippocampus, 2026-09-30): the 18S cut at a fixed multiple of the sample's own
    # 18S background (tissue median) instead of a fixed percentile. The percentile is taken over a signal scaled to
    # the 18S p99, so where the background is a much smaller share of the range (hippocampus: 18S p99 / background
    # 70x vs 25x in spinal cord) the same percentile cuts far higher above background (33x vs 10.5x). None keeps
    # thr_percentile; 10.5 is the spinal cohort's median at thr_percentile.
    thr_bg_multiple: float | None = None
    # the same cuts as explicit values of the normalized signal (share of the 18S range, background to p99), used
    # instead of thr_percentile / thr_bg_multiple and glia_soma_bg_multiple when set: a background of a few counts
    # (integer images, most tissue pixels 0) is too coarse to multiply (SD012-18: tissue median 0)
    thr_value: float | None = None
    glia_soma_thr_value: float | None = None
    # glial somata from 18S (hippocampus, 2026-09-30): a nucleus left without an 18S cell grows into the 18S signal
    # above glia_soma_bg_multiple x background (same signal as the main mask), within glia_soma_max_um of its own
    # edge, splitting contested pixels with neighbouring orphan nuclei by distance, never into an existing cell.
    # Threads thinner than glia_soma_open_um (radius) are cut off, so processes are not followed. A nucleus that
    # gains < glia_soma_min_um2 of cytoplasm keeps the Xenium fallback below. None disables it.
    glia_soma_bg_multiple: float | None = None
    glia_soma_max_um: float = 4.0
    glia_soma_open_um: float = 0.5
    glia_soma_min_um2: float = 3.0
    fallback_max_expansion_um: float = 3.0  # glial cells have little cytoplasm; 3 um beat 5 um (Xenium's standard) on
                                            # mixing at ~1 point less coverage (step 13); these runs' polygons reach 15 um

    def to_dict(self) -> dict:
        return asdict(self)


def um2_to_px(a_um2: float) -> float:
    return a_um2 / (UM_PER_PX ** 2)


def um_to_px(d_um: float) -> float:
    return d_um / UM_PER_PX


# ---------------------------------------------------------------- 1. segmentation

def normalized_diff(crop: np.ndarray, rna_lo, rna_hi, vim_lo, vim_hi) -> np.ndarray:
    """Whole-sample-normalized 18S - Vimentin signal (as render_panels() encodes into seg_input.png, kept as
    float). Vimentin is never stretched over a smaller range than 18S: in a sample with a weak Vimentin stain
    (whole-sample p99 below 18S's, e.g. SD00614_BG_ 203 vs 481) scaling to its own p99 lifts the faint
    autofluorescence of motor-neuron somata to ~0.5 and cancels them out of the signal. Samples whose Vimentin
    range is at least 18S's are unchanged."""
    def norm(img, lo, hi):
        return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    vim_hi = vim_lo + max(vim_hi - vim_lo, rna_hi - rna_lo)
    return norm(crop[2], rna_lo, rna_hi) - norm(crop[3], vim_lo, vim_hi)


def sample_signal(crop: np.ndarray, pct, bg18: float, bgvim: float) -> np.ndarray:
    """The method's 18S - Vimentin signal: each channel scaled linearly from its whole-sample tissue background
    (median) to its whole-sample 99th percentile and clipped to [0, 1], Vimentin never over a smaller range than
    18S (normalized_diff). The background anchor subtracts a diffuse stain: with the 1st percentile (0 in every
    sample) as the lower anchor, a Vimentin background of ~25% of its range (Region_2) pushed motor-neuron somata
    under the threshold. Where the background is a few % of the range (most samples) the change is small."""
    return normalized_diff(crop, bg18, pct[1], bgvim, pct[3])


def smoothed_signal(diff: np.ndarray, valid: np.ndarray, median_px: int) -> tuple[np.ndarray, np.ndarray]:
    positive = valid & (diff > 0)
    filled = np.where(positive, diff, 0.0)
    return (ndi.median_filter(filled, size=median_px) if median_px > 1 else filled), positive


def close_components(lab: np.ndarray, radius_px: float) -> np.ndarray:
    """Binary closing of every labelled component on its own (disk of radius_px), keeping only added pixels that
    belong to no other component, so two neighbours are never bridged. Holes are filled per component after."""
    r = int(round(radius_px))
    if r < 1 or not lab.any():
        return lab
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (yy ** 2 + xx ** 2) <= r * r
    out = lab.copy()
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(0, s.start - r - 1), min(dim, s.stop + r + 1)) for s, dim in zip(sl, lab.shape))
        m = ndi.binary_fill_holes(ndi.binary_closing(lab[sl] == i, structure=disk))
        add = m & (out[sl] == 0)
        out[sl][add] = i
    return out


def segment(diff: np.ndarray, valid: np.ndarray, median_px: int, thr_percentile: float,
            min_size_px: int, thr: float | None = None, closing_px: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """thr: the whole-sample threshold when this is one tile of a sample; if None it is the percentile over
    this region's positive pixels. closing_px > 0: close each component on its own (close_components)."""
    filt, positive = smoothed_signal(diff, valid, median_px)
    thr = float(np.percentile(filt[positive], thr_percentile)) if thr is None else float(thr)
    binary = ndi.binary_fill_holes(positive & (filt > thr))
    lab, _ = ndi.label(binary)
    if closing_px > 0:
        lab = close_components(lab, closing_px)
    sizes = np.bincount(lab.ravel())
    too_small = sizes < min_size_px
    too_small[0] = False
    lab[too_small[lab]] = 0
    lab, _ = ndi.label(lab > 0)
    return lab, filt, thr


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy ** 2 + xx ** 2) <= r * r


def grow_dim_somata(comp: np.ndarray, raw18: np.ndarray, nuc: np.ndarray, valid: np.ndarray, bg18: float,
                    p99_18: float, p: Params) -> np.ndarray:
    """Relative edge for large masks with the shape rule (see Params.soma_grow_frac). raw18: the raw 18S channel
    smoothed with soma_grow_sigma_um. Masks are grown one at a time, largest first; a grown mask blocks the later
    ones."""
    if p.soma_grow_frac is None or not comp.any():
        return comp
    out = comp.copy()
    cap = um_to_px(p.soma_grow_max_um)
    pad = int(np.ceil(cap)) + 2
    areas = np.bincount(comp.ravel())
    big = [int(i) for i in np.argsort(-areas) if i > 0 and areas[i] >= um2_to_px(p.likely_neuron_soma_um2)]
    slices = ndi.find_objects(comp)
    floor = p.soma_grow_floor_frac * (p99_18 - bg18)
    lobe_disk = _disk(max(1, int(round(um_to_px(p.soma_grow_lobe_um)))))
    dend_disk = _disk(max(1, int(round(um_to_px(p.dendrite_max_half_width_um)))))
    min_lobe = um2_to_px(p.soma_grow_min_lobe_um2)
    min_process = um2_to_px(5.0)
    max_contact = 2 * um_to_px(p.dendrite_max_half_width_um)
    clear = um_to_px(p.soma_grow_clear_um)
    isolation = um_to_px(p.soma_grow_isolation_um)
    for i in big:
        sl = tuple(slice(max(0, s.start - pad), min(dim, s.stop + pad)) for s, dim in zip(slices[i - 1], comp.shape))
        m = out[sl] == i
        v = raw18[sl] - bg18
        thr = max(p.soma_grow_frac * float(np.median(v[m])), floor)
        n = nuc[sl]
        foreign = (n > 0) & ~np.isin(n, np.unique(n[m]))
        others = ((out[sl] > 0) & ~m) | foreign
        dist_others = ndi.distance_transform_edt(~others)
        free = (out[sl] == 0) & valid[sl] & ~foreign & (dist_others > clear)
        lab, _ = ndi.label(m | (free & (v >= thr) & (ndi.distance_transform_edt(~m) <= cap)))
        reach = np.isin(lab, np.unique(lab[m])) & ~m
        if not reach.any():
            continue
        g = m.copy()
        # soma lobes: broad attachment to the soma body, nothing else nearby
        thick = ndi.binary_opening(reach, structure=lobe_disk)
        lab, _ = ndi.label(thick)
        body = ndi.binary_opening(m, structure=dend_disk)
        body_edge = body & ~ndi.binary_erosion(body)
        for j in range(1, int(lab.max()) + 1):
            lobe = lab == j
            a = int(lobe.sum())
            if a < min_lobe or dist_others[lobe].min() < isolation:
                continue
            contact = int((body_edge & ndi.binary_dilation(lobe, iterations=3)).sum())
            if contact / np.sqrt(a) >= p.soma_grow_min_body_contact:
                g |= lobe
        g = ndi.binary_fill_holes(g)
        # process extensions: thin, long, and attached only at an existing process of the mask
        process_part = m & ~ndi.binary_opening(m, structure=dend_disk)
        lab, _ = ndi.label(reach & ~ndi.binary_dilation(thick, iterations=1) & ~g)
        rim = ndi.binary_dilation(m, iterations=1) & ~m
        for j in range(1, int(lab.max()) + 1):
            piece = lab == j
            if (piece.sum() < min_process or dist_others[piece].min() < isolation
                    or classify_shape(piece, p)[0] != SHAPE_THIN_LONG):
                continue
            contact = piece & rim
            if not contact.any() or contact.sum() > max_contact:
                continue
            at_process = ndi.binary_dilation(contact, iterations=1) & m
            if process_part[at_process].mean() >= 0.8:
                g |= piece
        add = g & ~m & (out[sl] == 0)
        out[sl][add] = i
    return out




# ---------------------------------------------------------------- nuclei

def vim_rescue(comp: np.ndarray, comp_nv: np.ndarray, nuc: np.ndarray, valid: np.ndarray, p: Params) -> np.ndarray:
    """Soma-sized blobs of the Vimentin-blind 18S mask that the main mask mostly misses (see Params.vim_rescue_min_um2)."""
    from skimage.morphology import convex_hull_image
    out = comp.copy()
    if p.vim_rescue_min_um2 is None or not comp_nv.any():
        return out
    lab_nv, n = ndi.label(comp_nv > 0)
    open_disk = _disk(max(1, int(round(um_to_px(p.vim_rescue_open_um)))))
    for j, sl in enumerate(ndi.find_objects(lab_nv), start=1):
        if sl is None:
            continue
        pad = tuple(slice(max(0, x.start - 3), x.stop + 3) for x in sl)
        c = lab_nv[pad] == j
        a = int(c.sum())
        if a < um2_to_px(p.vim_rescue_min_um2):
            continue
        cm = out[pad]
        if float(ndi.distance_transform_edt(c).max()) < um_to_px(p.vim_rescue_min_radius_um):
            continue
        if a / max(1, convex_hull_image(c).sum()) < p.vim_rescue_min_solidity:
            continue
        ids, cnt = np.unique(cm[c & (cm > 0)], return_counts=True)
        sizes = {int(k): int((cm == k).sum()) for k in ids}
        host = int(ids[np.argmax(cnt)]) if len(ids) else 0
        n_sl = nuc[pad]
        # masks largely inside the blob: a small one without a nucleus is a fragment of the same soma and is absorbed; one with a
        # nucleus or >= the likely-neuron size is a second cell, and then the blob is two cells, not one soma with a missing part
        inside = [int(k) for k, n_in in zip(ids, cnt) if int(k) != host and n_in >= 0.3 * sizes[int(k)]]
        own = set(np.unique(n_sl[(cm == host) & (n_sl > 0)])) if host else set()
        foreign = (n_sl > 0) & ~np.isin(n_sl, list(own))
        # a small mask that overlaps or touches a nucleus the host does not own is a satellite cell, not a fragment of the soma
        if any(sizes[k] >= um2_to_px(p.likely_neuron_soma_um2) or (ndi.binary_dilation(cm == k, iterations=3) & foreign).any() for k in inside):
            continue
        new = c & (cm == 0) & valid[pad] & ~ndi.binary_dilation(foreign, iterations=1)
        new = ndi.binary_opening(new, structure=open_disk)
        if new.sum() < max(um2_to_px(p.vim_rescue_min_add_um2), (1 - p.vim_rescue_max_overlap) * 0.5 * a):
            continue
        lab_id = host or int(out.max()) + 1
        view = out[pad]
        for k in inside:
            view[view == k] = lab_id
        view[new] = lab_id
        view[ndi.binary_fill_holes(view == lab_id) & (view == 0)] = lab_id
    return out


def polygon_areas_um2(nb: pd.DataFrame) -> pd.Series:
    """Shoelace area per cell_id of a nucleus_boundaries table (rows grouped contiguously per cell)."""
    if nb.empty:
        return pd.Series(dtype=float, name="nucleus_area_um2")
    codes, uniques = pd.factorize(nb["cell_id"].to_numpy())
    if np.any(np.diff(codes) < 0):
        order = np.argsort(codes, kind="stable")
        codes = codes[order]
        nb = nb.iloc[order]
    n = len(codes)
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    ends = np.r_[starts[1:], n] - 1
    nxt = np.arange(n) + 1
    nxt[ends] = starts
    x, y = nb["vertex_x"].to_numpy(), nb["vertex_y"].to_numpy()
    area = 0.5 * np.abs(np.bincount(codes, weights=x * y[nxt] - x[nxt] * y, minlength=len(uniques)))
    return pd.Series(area, index=uniques, name="nucleus_area_um2")


def rasterize_nuclei(nb: pd.DataFrame, fx0: int, fy0: int, shape: tuple[int, int]
                     ) -> tuple[np.ndarray, list[str]]:
    """Label image of individual Xenium nuclei in crop-local level-0 px (label k -> cell_ids[k-1])."""
    H, W = shape
    x0_um, y0_um = fx0 * UM_PER_PX, fy0 * UM_PER_PX
    x1_um, y1_um = (fx0 + W) * UM_PER_PX, (fy0 + H) * UM_PER_PX
    sub = nb[nb.vertex_x.between(x0_um, x1_um) & nb.vertex_y.between(y0_um, y1_um)]
    labels = np.zeros(shape, dtype=np.int32)
    cell_ids: list[str] = []
    for cid, grp in sub.groupby("cell_id", sort=False):
        if len(grp) < 3:
            continue
        poly = np.column_stack([grp["vertex_y"].to_numpy() / UM_PER_PX - fy0,
                                grp["vertex_x"].to_numpy() / UM_PER_PX - fx0])
        r0, r1 = max(0, int(np.floor(poly[:, 0].min()))), min(H, int(np.ceil(poly[:, 0].max())) + 1)
        c0, c1 = max(0, int(np.floor(poly[:, 1].min()))), min(W, int(np.ceil(poly[:, 1].max())) + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        yy, xx = np.mgrid[r0:r1, c0:c1]
        inside = MplPath(poly).contains_points(np.column_stack([yy.ravel(), xx.ravel()])).reshape(r1 - r0, c1 - c0)
        if not inside.any():
            continue
        cell_ids.append(cid)
        labels[r0:r1, c0:c1][inside] = len(cell_ids)
    return labels, cell_ids


def nucleus_contrast(dapi_smooth: np.ndarray, labels: np.ndarray, ids: np.ndarray, ring_um: float) -> np.ndarray:
    """DAPI contrast of each labelled nucleus: median smoothed DAPI inside / median in a ring of ring_um around
    it. labels holds every nucleus of the region, so the ring never includes another nucleus."""
    if len(ids) == 0:
        return np.empty(0)
    ring = expand_labels(labels, um_to_px(ring_um))
    ring[labels > 0] = 0
    inner = np.asarray(ndi.median(dapi_smooth, labels, ids), dtype=float)
    outer = np.asarray(ndi.median(dapi_smooth, ring, ids), dtype=float)
    return inner / np.maximum(outer, 1.0)


def rescue_nuclei(dapi: np.ndarray, nuc: np.ndarray, valid: np.ndarray, p: Params,
                  ref: float | None = None, inside: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Nuclei Xenium's segmentation missed: DAPI blobs >= rescue_dapi_frac of the median DAPI inside Xenium's
    own nuclei (so the reference is the nuclei themselves, not a pixel percentile), not within 1 um of any
    Xenium nucleus, nucleus-sized, roughly round, and lying (>= assign_frac of their area) inside an 18S mask
    (inside). A soma is evidence that a cell and its nucleus are there; the pale, euchromatic nuclei of neurons
    are what Xenium misses. In the neuropil DAPI would be the only evidence, and there textured background passes
    the intensity test in some samples, so no nucleus is added there. Appended as labels after the Xenium ones.
    ref: the whole-sample reference when this is one tile of a sample."""
    d = ndi.gaussian_filter(dapi.astype(float), um_to_px(0.5))
    if ref is None:
        ref = float(np.median(d[nuc > 0])) if (nuc > 0).any() else float("inf")
    near_known = ndi.distance_transform_edt(nuc == 0) <= um_to_px(1.0)
    cand = valid & (d >= p.rescue_dapi_frac * ref) & ~near_known
    cand = ndi.binary_opening(cand, iterations=2)
    lab, n = ndi.label(cand)
    out = nuc.copy()
    nxt = int(nuc.max())
    added = 0
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab[sl] == i
        a = m.sum() * UM_PER_PX ** 2
        if not (p.rescue_min_um2 <= a <= p.rescue_max_um2) or elongation_ratio(m) > p.rescue_max_elongation:
            continue
        if regionprops(m.astype(np.uint8))[0].solidity < p.rescue_min_solidity:
            continue
        if inside is not None and inside[sl][m].mean() < p.assign_frac:
            continue
        nxt += 1
        added += 1
        out[sl][m] = nxt
    return out, added


def nucleus_contrast(dapi_smooth: np.ndarray, labels: np.ndarray, ids: np.ndarray, ring_um: float) -> np.ndarray:
    """QC only: DAPI contrast of each labelled nucleus, median smoothed DAPI inside / median in a ring of
    ring_um around it (labels holds every nucleus, so no ring covers another nucleus)."""
    if len(ids) == 0:
        return np.empty(0)
    ring = expand_labels(labels, um_to_px(ring_um))
    ring[labels > 0] = 0
    inner = np.asarray(ndi.median(dapi_smooth, labels, ids), dtype=float)
    outer = np.asarray(ndi.median(dapi_smooth, ring, ids), dtype=float)
    return inner / np.maximum(outer, 1.0)


# ---------------------------------------------------------------- 2./3. assignment + flags

def overlap_table(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    """Pixel counts for every (a_label, b_label) pair with both > 0."""
    m = (a > 0) & (b > 0)
    if not m.any():
        return pd.DataFrame(columns=["a", "b", "px"])
    pairs = pd.DataFrame({"a": a[m], "b": b[m]})
    return pairs.value_counts().rename("px").reset_index()


def assign_nuclei(comp: np.ndarray, nuc: np.ndarray, assign_frac: float
                  ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """assigned[c] = nuclei with >= assign_frac of their area inside c; touching[c] = nuclei overlapping
    c by less than that. A nucleus is assigned to at most one component by construction."""
    nuc_area = np.bincount(nuc.ravel())
    ov = overlap_table(comp, nuc)
    assigned: dict[int, list[int]] = {}
    touching: dict[int, list[int]] = {}
    for c, k, px in ov.itertuples(index=False):
        frac = px / nuc_area[k]
        (assigned if frac >= assign_frac else touching).setdefault(int(c), []).append(int(k))
    return assigned, touching


def boundary_wrap_fraction(comp_mask: np.ndarray, nuc_mask: np.ndarray) -> float:
    edge = nuc_mask & ~ndi.binary_erosion(nuc_mask)
    return float((edge & comp_mask).sum() / max(edge.sum(), 1))


def nucleus_owners(labels: np.ndarray, nuc: np.ndarray, p: Params) -> tuple[dict[int, int], dict[int, list[int]]]:
    """owner[k] = the label that owns nucleus k: the one holding >= assign_frac of it, else the first label
    whose mask wraps >= partial_wrap_frac of its boundary. Also returns touching nuclei per label."""
    assigned, touching = assign_nuclei(labels, nuc, p.assign_frac)
    owner = {k: lab for lab, ks in assigned.items() for k in ks}
    slices = ndi.find_objects(labels)
    for lab in sorted(touching):
        sl = slices[lab - 1]
        m = labels[sl] == lab
        for k in touching[lab]:
            if k not in owner and boundary_wrap_fraction(m, nuc[sl] == k) >= p.partial_wrap_frac:
                owner[k] = lab
    return owner, touching


def _merge_basins(basins: np.ndarray, vn: np.ndarray, p: Params) -> np.ndarray:
    """Merge neighbouring basins until every remaining pair is resolved (saddle < soma_max_saddle_frac of the
    dimmer peak) and every basin is round; the least-resolved pair / a non-round basin's best-connected
    neighbour goes first."""
    basins = basins.copy()
    while int(len(np.unique(basins[basins > 0]))) >= 2:
        ids = [int(i) for i in np.unique(basins[basins > 0])]
        peak = {i: float(vn[basins == i].max()) for i in ids}
        grown = {i: ndi.binary_dilation(basins == i) for i in ids}
        best, best_ratio = None, -1.0
        for a in ids:
            for b in ids:
                if b <= a:
                    continue
                contact = (grown[a] & (basins == b)) | (grown[b] & (basins == a))
                if contact.any():
                    ratio = float(vn[contact].max()) / min(peak[a], peak[b])
                    if ratio > best_ratio:
                        best, best_ratio = (a, b), ratio
        not_round = [i for i in ids if classify_shape(basins == i, p)[0] != SHAPE_ROUND]
        if best is not None and best_ratio >= p.soma_max_saddle_frac:
            a, b = best
        elif not_round:
            a = not_round[0]
            nbr = {}
            for b in ids:
                if b != a:
                    contact = (grown[a] & (basins == b)) | (grown[b] & (basins == a))
                    if contact.any():
                        nbr[b] = float(vn[contact].max())
            if not nbr:
                return basins
            b = max(nbr, key=nbr.get)
        else:
            return basins
        basins[basins == b] = a
    return basins


def enclosed_nuclei(mask: np.ndarray, nuc_local: np.ndarray) -> set[int]:
    """Nuclei fully inside mask: the largest distance of their pixels to the mask edge is at least their own
    equivalent radius, so no part reaches or crosses the edge. A soma's own nucleus is enclosed (in a thin
    soma only just); a glial nucleus pressed onto the soma's edge is not."""
    depth = ndi.distance_transform_edt(mask)
    npx = np.bincount(nuc_local.ravel())
    out = set()
    for k in np.unique(nuc_local[mask]):
        if k and float(depth[(nuc_local == k) & mask].max(initial=0)) >= np.sqrt(npx[k] / np.pi):
            out.add(int(k))
    return out


def _merge_nucleusless_basins(basins: np.ndarray, vn: np.ndarray, nuc_local: np.ndarray,
                              enclosed_only: bool = True) -> np.ndarray:
    """A soma-peak basin that holds no nucleus of its own (>= 50% of a nucleus inside it) joins the neighbour
    it shares the highest saddle with, as long as some basin of the mask has a nucleus: intensity texture
    within one soma (clumped Nissl, a bright end) otherwise reads as a second soma (2026-09-30, pathologists'
    pins). A mask with no nucleus at all keeps its basins. Only enclosed nuclei count (enclosed_nuclei): a
    glial nucleus on the mask's edge does not make a basin a soma."""
    basins = basins.copy()
    npx = np.bincount(nuc_local.ravel())
    own = enclosed_nuclei(basins > 0, nuc_local) if enclosed_only else set(np.unique(nuc_local[basins > 0]).tolist())

    def has_nucleus(i):
        m = basins == i
        return any(((nuc_local == k) & m).sum() * 2 >= npx[k] for k in np.unique(nuc_local[m]) if k in own)

    while True:
        ids = [int(i) for i in np.unique(basins[basins > 0])]
        if len(ids) < 2:
            return basins
        with_nuc = {i: has_nucleus(i) for i in ids}
        lonely = [i for i in ids if not with_nuc[i]]
        if not lonely or len(lonely) == len(ids):
            return basins
        a = lonely[0]
        grown = ndi.binary_dilation(basins == a)
        nbr = {}
        for b in ids:
            if b != a:
                contact = (grown & (basins == b)) | (ndi.binary_dilation(basins == b) & (basins == a))
                if contact.any():
                    nbr[b] = float(vn[contact].max())
        if not nbr:
            return basins
        basins[basins == a] = max(nbr, key=nbr.get)


def soma_peaks(mask: np.ndarray, raw_local: np.ndarray, p: Params, nuc_local: np.ndarray | None = None) -> np.ndarray | None:
    """Somata inside one mask, from the RAW 18S channel (the normalized signal is clipped at the sample's
    99th percentile, so two bright somata pressed together read as one flat plateau there). Candidate peaks
    are merged until every remaining pair is resolved by the half-maximum rule and every soma is round; with
    nuc_local, a basin without a nucleus of its own then joins its best-connected neighbour
    (_merge_nucleusless_basins). Returns a label image of >= 2 somata, else None."""
    if mask.sum() < 2 * um2_to_px(p.small_area_um2):
        return None
    v = np.where(mask, raw_local, 0.0)
    top = np.percentile(v[mask], 99)
    if top <= 0:
        return None
    vn = np.clip(v / top, 0, 1)
    peaks, n = ndi.label(h_maxima(vn, p.soma_candidate_h) & mask)
    if n < 2:
        return None
    basins = _merge_basins(watershed(-vn, peaks, mask=mask), vn, p)
    if nuc_local is not None and p.basin_nucleus != "off":
        basins = _merge_nucleusless_basins(basins, vn, np.where(mask, nuc_local, 0), p.basin_nucleus == "enclosed")
    ids = np.unique(basins[basins > 0])
    if len(ids) < 2:
        return None
    out = np.zeros_like(basins)
    for i, b in enumerate(ids, start=1):
        out[basins == b] = i
    return out


def compute_flags(comp: np.ndarray, nuc: np.ndarray, alt: np.ndarray, raw18: np.ndarray,
                  nucleus_is_neuron: np.ndarray, p: Params) -> pd.DataFrame:
    n_comp = int(comp.max())
    owner, touching = nucleus_owners(comp, nuc, p)
    own_by: dict[int, list[int]] = {}
    for k, c in owner.items():
        own_by.setdefault(c, []).append(k)
    alt_assigned, _ = assign_nuclei(comp, alt, p.assign_frac)
    areas = np.bincount(comp.ravel(), minlength=n_comp + 1)
    nuc_area = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1)
    slices = ndi.find_objects(comp)
    rows = []
    for c in range(1, n_comp + 1):
        own = own_by.get(c, [])
        t = [k for k in touching.get(c, []) if owner.get(k) != c]
        n_alt = len(alt_assigned.get(c, []))
        sl = slices[c - 1]
        somata = soma_peaks(comp[sl] == c, raw18[sl], p, nuc[sl])
        n_somata = 0 if somata is None else int(somata.max())
        if len(own) >= 2:
            flag = FLAG_MULTI
        elif n_alt >= 2 or n_somata >= 2:
            flag = FLAG_SENSE_SPLIT
        elif own:
            flag = FLAG_NONE
        elif t:
            flag = FLAG_PARTIAL
        else:
            flag = FLAG_NO_NUCLEUS
        mismatch = False
        if len(own) == 1 and flag != FLAG_SENSE_SPLIT:
            k = own[0]
            if not nucleus_is_neuron[k] and areas[c] >= p.glia_mask_over_nucleus * nuc_area[k]:
                mismatch = True
        rows.append({"comp": c, "flag": flag, "n_touching": len(t), "n_alt": n_alt, "cores": alt_assigned.get(c, []),
                     "n_somata": n_somata,
                     "own_nuclei": own, "mismatch": mismatch, "area_px": int(areas[c])})
    cols = ["comp", "flag", "n_touching", "n_alt", "cores", "n_somata", "own_nuclei", "mismatch", "area_px"]
    return pd.DataFrame(rows, columns=cols).set_index("comp")   # a region without 18S masks gives no rows


# ---------------------------------------------------------------- shape

def elongation_ratio(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 4:
        return 1.0
    cov = np.cov(np.vstack([xs, ys]))
    ev = np.sort(np.linalg.eigvalsh(cov))
    return float(np.sqrt(ev[1] / max(ev[0], 1e-6)))


def classify_shape(mask: np.ndarray, p: Params) -> tuple[int, dict]:
    area_um2 = mask.sum() * UM_PER_PX ** 2
    half_width_um = float(ndi.distance_transform_edt(np.pad(mask, 1)).max()) * UM_PER_PX
    elong = elongation_ratio(mask)
    if half_width_um <= p.dendrite_max_half_width_um and elong >= p.dendrite_min_elongation:
        shape = SHAPE_THIN_LONG
    elif area_um2 < p.small_area_um2:
        shape = SHAPE_SMALL
    else:
        shape = SHAPE_ROUND
    return shape, {"area_um2": area_um2, "half_width_um": half_width_um, "elongation": elong}


# ---------------------------------------------------------------- 4. resolution

def nucleus_seeded_split(mask: np.ndarray, nuc_local: np.ndarray, own: list[int], elevation: np.ndarray) -> list[np.ndarray]:
    """One piece per nucleus, flooding along the 18S signal (cuts land in its valleys)."""
    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, k in enumerate(own, start=1):
        markers[(nuc_local == k) & mask] = i
    pieces = watershed(-elevation, markers, mask=mask)
    return [pieces == i for i in range(1, len(own) + 1) if (pieces == i).any()]


def soma_nuclei_pieces(soma: np.ndarray, nuc_local: np.ndarray, own: list[int], elevation: np.ndarray,
                       nucleus_is_neuron: np.ndarray, bright: bool, p: Params,
                       inner: set[int] | None = None, base: np.ndarray | None = None) -> tuple[list[np.ndarray], str]:
    """A soma holding >= 2 nuclei. If it contains a bright 18S core it is one RNA-rich soma with satellite
    glia: the soma keeps its deepest enclosed nucleus (the one lying furthest inside relative to its own radius,
    whatever its size) and each other nucleus gets only itself plus a glia_rim_um rim. Two or more enclosed
    neuron-sized nuclei are two neurons. A dim mask with several nuclei is a cluster of small (glial) cells and
    is split one piece per nucleus. inner / base: the enclosed nuclei (enclosed_nuclei) and the 18S mask, both
    from before the mask was extended by its nuclei's pixels (which makes an edge nucleus look interior);
    computed from soma if None."""
    nuc_px = {k: int(((nuc_local == k) & soma).sum()) for k in own}
    if not bright:
        return nucleus_seeded_split(soma, nuc_local, own, elevation), "nucleus_seeded"
    inner = enclosed_nuclei(soma, nuc_local) if inner is None else inner
    # only enclosed neuron-sized nuclei are neurons of this soma; one on the edge is a satellite however large
    neuron = [k for k in own if nucleus_is_neuron[k] and (k in inner or not p.neuron_split_enclosed)]
    if len(neuron) >= 2:
        # two neuron nuclei are two neurons: split between them, then resolve each one's satellites
        parts = []
        for part in nucleus_seeded_split(soma, nuc_local, neuron, elevation):
            sub = [k for k in own if ((nuc_local == k) & part).sum() * 2 >= (nuc_local == k).sum()]
            if len(sub) >= 2:
                parts += soma_nuclei_pieces(part, nuc_local, sub, elevation, nucleus_is_neuron, bright, p, inner,
                                            None if base is None else base & part)[0]
            else:
                parts.append(part)
        return parts, "neurons+satellite_rim"
    depth = ndi.distance_transform_edt(soma)
    # the soma keeps its deepest enclosed nucleus (depth beyond its own radius), whatever its size: an edge or
    # side-lobe nucleus never takes the soma from the central one (2026-09-30: a large nucleus on the edge or in
    # a side lobe took the soma and the neuron's own central nucleus was carved out)
    if p.soma_keeper == "deepest":
        inside = [k for k in own if k in inner] or own
        main = max(inside, key=lambda k: float(depth[(nuc_local == k) & soma].max(initial=0)) - np.sqrt(max(nuc_px[k], 1) / np.pi))
    else:
        if p.soma_keeper == "interior_neuron":
            b = soma if base is None else base
            depth_b = ndi.distance_transform_edt(b)
            inside = [k for k in own if float(depth_b[(nuc_local == k) & b].max(initial=0))
                      >= np.sqrt(max(nuc_px[k], 1) / np.pi) + um_to_px(1.0)] or own
        else:
            inside = own
        neuron_in = [k for k in inside if nucleus_is_neuron[k]]
        main = max(neuron_in or inside, key=lambda k: (nuc_px[k] if neuron_in else depth[nuc_local == k].mean()))
    rim_px = um_to_px(p.glia_rim_um)
    rest = soma.copy()
    rims = []
    for k in own:
        if k == main:
            continue
        rim = soma & (ndi.distance_transform_edt(nuc_local != k) <= rim_px)
        rim &= rest
        rims.append(rim)
        rest &= ~rim
    frag, _ = ndi.label(rest)
    main_frag = np.unique(frag[(nuc_local == main) & rest])
    body = np.isin(frag, main_frag[main_frag > 0]) if len(main_frag) else rest
    # bits cut off from the soma by a rim go to the nearest rim instead of floating free
    orphan = rest & ~body
    if orphan.any() and rims:
        lab = np.zeros(soma.shape, dtype=np.int32)
        for i, r in enumerate(rims, start=1):
            lab[r] = i
        grown = watershed(-elevation, lab, mask=orphan | (lab > 0))
        rims = [grown == i for i in range(1, len(rims) + 1)]
    return [body] + [r for r in rims if r.any()], "satellite_rim"


def _first_waist(core: np.ndarray) -> np.ndarray | None:
    cur = core
    while True:
        cur = ndi.binary_erosion(cur)
        if not cur.any():
            return None
        frag, n = ndi.label(cur)
        if n >= 2:
            return frag


def _is_cell_body(m: np.ndarray, p: Params) -> bool:
    """A piece the waist split may separate as its own cell: not small and not elongated. SHAPE_ROUND alone is
    not enough: a thick proximal dendrite (half-width above dendrite_max_half_width_um but many times longer than
    wide) is 'round' by that test, and cutting it off would split a neuron from its own process."""
    shape, sm = classify_shape(m, p)
    return shape == SHAPE_ROUND and sm["elongation"] < p.dendrite_min_elongation


def _undo_fake_necks(final: np.ndarray, ids: list[int], mask: np.ndarray, p: Params) -> list[np.ndarray]:
    """Rejoin pieces separated by a cut that is not a neck: neck radius (distance to the mask edge, max along the
    cut) > waist_max_neck_frac x the smaller piece's inscribed radius. Returns the remaining groups as masks."""
    if p.waist_max_neck_frac is None:
        return [final == i for i in ids]
    dt = ndi.distance_transform_edt(mask)
    rad = {i: float(ndi.distance_transform_edt(final == i).max()) for i in ids}
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a_i, i in enumerate(ids):
        grown_i = ndi.binary_dilation(final == i)
        for j in ids[a_i + 1:]:
            cut = grown_i & (final == j)
            if not cut.any():
                continue
            if float(dt[cut].max()) > p.waist_max_neck_frac * min(rad[i], rad[j]):
                parent[find(j)] = find(i)
    groups: dict[int, np.ndarray] = {}
    for i in ids:
        r = find(i)
        groups[r] = groups.get(r, np.zeros_like(mask)) | (final == i)
    return list(groups.values())


def waist_split(mask: np.ndarray, p: Params) -> list[np.ndarray] | None:
    """Marcel: "we identify the split area where the signal is missing, if this is thin and long
    (dendritic) then we attach it to the larger segment and cut at the smaller, if it is just a bridge
    we cut at the thinner point." Erode until the shape first breaks, grow the fragments back over the
    distance transform (so the cut lands on the neck), fold any piece that is not a cell body (small or
    elongated, _is_cell_body) back into the core and keep drilling; accept only when >= 2 cell bodies exist at once."""
    core = mask
    for _ in range(p.max_waist_rounds):
        frag = _first_waist(core)
        if frag is None:
            return None
        grown = watershed(-ndi.distance_transform_edt(core), frag, mask=core)
        ids = [i for i in range(1, int(grown.max()) + 1) if (grown == i).any()]
        round_ids = [i for i in ids if _is_cell_body(grown == i, p)]
        if len(round_ids) >= 2:
            seeds = np.where(np.isin(grown, round_ids), grown, 0)
            final = watershed(-ndi.distance_transform_edt(mask), seeds, mask=mask)
            groups = _undo_fake_necks(final, round_ids, mask, p)
            return groups if len(groups) >= 2 else None
        if len(round_ids) == 1:
            core = grown == round_ids[0]
            continue
        return None
    return None


def resolve(comp: np.ndarray, nuc: np.ndarray, alt: np.ndarray, raw18: np.ndarray, filt: np.ndarray,
            flags: pd.DataFrame, nucleus_is_neuron: np.ndarray, p: Params) -> tuple[np.ndarray, pd.DataFrame]:
    """Returns a piece label image plus, per piece, which component it came from and how.
    MULTI / SENSE_SPLIT: split into somata at the raw-18S peaks first (soma_peaks); a soma still holding
    >= 2 nuclei is one cell with satellite glia if it contains a bright sense-check core, else a cluster of
    small cells split one piece per nucleus (soma_nuclei_pieces). A SENSE_SPLIT or mismatch component with
    no usable peaks falls back to the geometric waist split."""
    out = np.zeros_like(comp)
    rows = []
    slices = ndi.find_objects(comp)
    next_id = 0
    for c, f in flags.iterrows():
        sl = slices[c - 1]
        mask = comp[sl] == c
        methods: list[str] = []
        pieces = [mask]
        base18 = mask                                                                  # before the extension
        inner18 = enclosed_nuclei(mask, nuc[sl]) if f.flag == FLAG_MULTI else None
        if f.flag == FLAG_MULTI:
            mask = mask | (np.isin(nuc[sl], f.own_nuclei) & (comp[sl] == 0) & (out[sl] == 0))
            pieces = [mask]
        if f.flag in (FLAG_MULTI, FLAG_SENSE_SPLIT):
            basins = soma_peaks(mask, raw18[sl], p, nuc[sl])
            if basins is not None:
                pieces = [basins == i for i in range(1, int(basins.max()) + 1) if (basins == i).any()]
                methods.append("somata")
        if f.flag == FLAG_MULTI:
            resolved = []
            for soma in pieces:
                own = [k for k in f.own_nuclei if ((nuc[sl] == k) & soma).sum() * 2 >= (nuc[sl] == k).sum()]
                if len(own) >= 2:
                    bright = bool(((alt[sl] > 0) & soma).any())
                    parts, how = soma_nuclei_pieces(soma, nuc[sl], own, filt[sl], nucleus_is_neuron, bright, p,
                                                    inner=inner18, base=base18 & soma)
                    resolved += parts
                    methods.append(how)
                else:
                    resolved.append(soma)
            pieces = resolved
        elif not methods and (f.flag == FLAG_SENSE_SPLIT or f.mismatch):
            split = waist_split(mask, p)
            if split:
                pieces = split
                methods.append("waist")
            else:
                methods.append("waist_not_found")
        method = "+".join(dict.fromkeys(methods)) or "unchanged"
        for piece in pieces:
            next_id += 1
            out[sl][piece] = next_id
            rows.append({"piece": next_id, "comp": c, "comp_flag": FLAG_NAMES[f.flag],
                         "mismatch": bool(f.mismatch), "method": method, "n_pieces_from_comp": len(pieces)})
    return out, pd.DataFrame(rows, columns=["piece", "comp", "comp_flag", "mismatch", "method",
                                            "n_pieces_from_comp"]).set_index("piece")


# ---------------------------------------------------------------- 5. nuclei belong to their cell

def absorb_owned_nuclei(pieces: np.ndarray, nuc: np.ndarray, p: Params) -> tuple[np.ndarray, dict[int, int]]:
    """Unclaimed pixels of an owned nucleus join its owner, so every cell contains its whole nucleus."""
    owner, _ = nucleus_owners(pieces, nuc, p)
    out = pieces.copy()
    for k, pid in owner.items():
        out[(nuc == k) & (out == 0)] = pid
    return out, owner


# ---------------------------------------------------------------- 6. classification

def transcript_pixel_labels(tx: pd.DataFrame, labels: np.ndarray, fx0: int, fy0: int) -> np.ndarray:
    col = np.floor(tx["x_location"].to_numpy() / UM_PER_PX - fx0).astype(int)
    row = np.floor(tx["y_location"].to_numpy() / UM_PER_PX - fy0).astype(int)
    H, W = labels.shape
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    lab = np.zeros(len(tx), dtype=np.int32)
    lab[ok] = labels[row[ok], col[ok]]
    return lab


def classify_pieces(pieces: np.ndarray, nuc: np.ndarray, owner: dict[int, int], tx_piece: np.ndarray,
                    piece_info: pd.DataFrame, nucleus_is_neuron: np.ndarray, p: Params) -> pd.DataFrame:
    own_by_piece: dict[int, list[int]] = {}
    for k, pid in owner.items():
        own_by_piece.setdefault(pid, []).append(k)
    nuc_px = np.bincount(nuc.ravel())
    tx_counts = np.bincount(tx_piece, minlength=int(pieces.max()) + 1)
    slices = ndi.find_objects(pieces)
    rows = []
    for pid in piece_info.index:
        mask = pieces[slices[pid - 1]] == pid
        own = own_by_piece.get(pid, [])
        shape, sm = classify_shape(mask, p)
        n_tx = int(tx_counts[pid])
        has_nucleus = bool(own)
        # A piece that is mostly nucleus is a cell body whatever its outline (e.g. a glial rim).
        nucleus_dominated = has_nucleus and mask.sum() < p.glia_mask_over_nucleus * sum(nuc_px[k] for k in own)
        if shape == SHAPE_THIN_LONG and not nucleus_dominated:
            if n_tx >= p.min_dendrite_transcripts:
                cls = CLASS_DENDRITE_NUCLEUS if has_nucleus else CLASS_DENDRITE
            else:
                cls = CLASS_REJECT_DENDRITE
        elif has_nucleus:
            cls = CLASS_CELL
        elif shape == SHAPE_SMALL:
            cls = CLASS_REJECT_SMALL
        else:
            cls = CLASS_MASK_NO_NUCLEUS
        rows.append({"piece": pid, "class": cls, "shape": SHAPE_NAMES[shape], "n_nuclei": len(own),
                     "nucleus_type": ("neuron" if any(nucleus_is_neuron[k] for k in own) else "glia") if own else "",
                     "n_transcripts": n_tx, **{k: round(v, 3) for k, v in sm.items()}})
    return piece_info.join(pd.DataFrame(rows, columns=["piece", "class", "shape", "n_nuclei", "nucleus_type", "n_transcripts",
                                                       "area_um2", "half_width_um", "elongation"]).set_index("piece"))


# ---------------------------------------------------------------- 7. merge

def gene_matrix(tx_piece: np.ndarray, genes: np.ndarray, n_labels: int) -> tuple[np.ndarray, list[str]]:
    names, gidx = np.unique(genes, return_inverse=True)
    m = np.zeros((n_labels + 1, len(names)), dtype=np.int32)
    np.add.at(m, (tx_piece, gidx), 1)
    return m, list(names)


def merge_dendrites(pieces: np.ndarray, table: pd.DataFrame, tx_piece: np.ndarray, genes: np.ndarray,
                    p: Params) -> tuple[np.ndarray, pd.DataFrame]:
    """Marcel: "if there is likely neuron and we have a dendrite very close by (long, thin, no nucleus) and
    then we check the transcripts and they are not badly different or very much oligo/glia like, then we
    merge them, unless they are distant to each other." Likely neuron = cell with a neuron-sized nucleus or
    a mask >= likely_neuron_soma_um2. Candidate dendrite = no-nucleus thin+long piece (including ones that
    failed the transcript gate alone). Each candidate goes to its nearest likely neuron within
    merge_max_gap_um; the glial-share and profile-similarity vetoes apply only when the dendrite has enough
    transcripts to judge."""
    neurons = table.index[(table["class"] == CLASS_CELL) &
                          ((table.nucleus_type == "neuron") | (table.area_um2 >= p.likely_neuron_soma_um2))]
    dends = table.index[(table["shape"] == "thin_long") & (table.n_nuclei == 0)]
    rows = []
    merged = pieces.copy()
    if len(neurons) == 0 or len(dends) == 0:
        return merged, pd.DataFrame(rows, columns=["dendrite", "neuron", "gap_um", "glial_frac", "cosine",
                                                    "decision"])
    neuron_mask = np.isin(pieces, neurons)
    dist, (ri, ci) = ndi.distance_transform_edt(~neuron_mask, return_indices=True)
    nearest = pieces[ri, ci]
    counts, names = gene_matrix(tx_piece, genes, int(pieces.max()))
    glial_cols = [i for i, g in enumerate(names) if g in set(GLIAL_GENES)]
    slices = ndi.find_objects(pieces)
    for d in dends:
        sl = slices[d - 1]
        m = pieces[sl] == d
        dd, nn = dist[sl][m], nearest[sl][m]
        j = int(np.argmin(dd))
        gap_um, n = float(dd[j]) * UM_PER_PX, int(nn[j])
        vd, vn = counts[d].astype(float), counts[n].astype(float)
        n_tx = int(vd.sum())
        glial_frac = float(vd[glial_cols].sum() / n_tx) if n_tx else float("nan")
        a, b = np.sqrt(vd), np.sqrt(vn)
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) if n_tx and vn.sum() else float("nan")
        if gap_um > p.merge_max_gap_um:
            decision = "too_far"
        elif n_tx < p.merge_min_transcripts_for_profile:
            decision = "merged_proximity_only"
        elif glial_frac >= p.merge_max_glial_frac:
            decision = "vetoed_glial"
        elif cosine < p.merge_min_cosine:
            decision = "vetoed_dissimilar"
        else:
            decision = "merged"
        if decision.startswith("merged"):
            merged[sl][m] = n
        rows.append({"dendrite": int(d), "neuron": n, "gap_um": round(gap_um, 2), "n_transcripts": n_tx,
                     "glial_frac": round(glial_frac, 3), "cosine": round(cosine, 3), "decision": decision})
    return merged, pd.DataFrame(rows, columns=["dendrite", "neuron", "gap_um", "n_transcripts", "glial_frac", "cosine",
                                               "decision"])


# ---------------------------------------------------------------- 8. fallback

def rasterize_by_id(df: pd.DataFrame, id_to_label: dict[str, int], fx0: int, fy0: int,
                    shape: tuple[int, int]) -> np.ndarray:
    """Label image where the polygon of cell_id c (df: cell_id, vertex_x, vertex_y in um) gets id_to_label[c]."""
    H, W = shape
    out = np.zeros(shape, dtype=np.int32)
    sub = df[df.cell_id.isin(id_to_label.keys())]
    for cid, grp in sub.groupby("cell_id", sort=False):
        poly = np.column_stack([grp["vertex_y"].to_numpy() / UM_PER_PX - fy0,
                                grp["vertex_x"].to_numpy() / UM_PER_PX - fx0])
        r0, r1 = max(0, int(np.floor(poly[:, 0].min()))), min(H, int(np.ceil(poly[:, 0].max())) + 1)
        c0, c1 = max(0, int(np.floor(poly[:, 1].min()))), min(W, int(np.ceil(poly[:, 1].max())) + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        yy, xx = np.mgrid[r0:r1, c0:c1]
        inside = MplPath(poly).contains_points(np.column_stack([yy.ravel(), xx.ravel()])).reshape(r1 - r0, c1 - c0)
        out[r0:r1, c0:c1][inside] = id_to_label[cid]
    return out


def add_glial_somata(cells: np.ndarray, nuc: np.ndarray, owner: dict[int, int], n_nuclei: int, filt: np.ndarray,
                     valid: np.ndarray, thr: float, p: Params) -> tuple[np.ndarray, dict[int, int]]:
    """Orphan nuclei (no 18S cell) take the 18S signal around them as their soma (glia_soma_* in Params): pixels
    with filt > thr, free, on tissue, within glia_soma_max_um of the nucleus and nearer to it than to any other
    orphan nucleus, opened by glia_soma_open_um and connected to the nucleus. Nuclei gaining less than
    glia_soma_min_um2 are left to the fallback. Returns the label image and {new_label: nucleus_label}."""
    orphans = np.array([k for k in range(1, n_nuclei + 1) if k not in owner], dtype=int)
    if not len(orphans):
        return cells, {}
    seeds = np.where(np.isin(nuc, orphans), nuc, 0)
    territory = expand_labels(seeds, distance=um_to_px(p.glia_soma_max_um))
    other_nuc = (nuc > 0) & ~np.isin(nuc, orphans)
    cand = (filt > thr) & valid & (cells == 0) & ~other_nuc
    r = int(round(um_to_px(p.glia_soma_open_um)))
    out = cells.copy()
    nxt = int(cells.max())
    new: dict[int, int] = {}
    min_px = um2_to_px(p.glia_soma_min_um2)
    slices = ndi.find_objects(territory, max_label=int(orphans.max()))
    for k in orphans:
        sl = slices[k - 1]
        if sl is None:
            continue
        own = (seeds[sl] == k) & (out[sl] == 0)
        mine = territory[sl] == k
        m = (cand[sl] & mine) | own
        if r >= 1:   # nucleus and cytoplasm opened together: a thin rim survives, a thin thread does not
            m = ndi.binary_opening(m, structure=_disk(r)) | own
        lab, _ = ndi.label(m)
        keep = np.unique(lab[own])
        m = ndi.binary_fill_holes(np.isin(lab, keep[keep > 0])) & mine & (out[sl] == 0) & ~other_nuc[sl]
        if (m & ~own).sum() < min_px:
            continue
        nxt += 1
        out[sl][m] = nxt
        new[nxt] = int(k)
    return out, new


def add_fallback_cells(cells: np.ndarray, nuc: np.ndarray, owner: dict[int, int], nucleus_cell_ids: list[str],
                       cell_boundaries: pd.DataFrame, fx0: int, fy0: int, p: Params) -> tuple[np.ndarray, dict[int, int]]:
    """Every nucleus without an 18S cell gets its Xenium cell (same cell_id, Xenium's nucleus expansion),
    cut back to fallback_max_expansion_um from the nucleus and minus any pixel already in an 18S cell; its
    own nucleus pixels are always kept. Returns the final label image and {new_label: nucleus_label}."""
    orphans = [k for k in range(1, len(nucleus_cell_ids) + 1) if k not in owner]
    xen = rasterize_by_id(cell_boundaries, {nucleus_cell_ids[k - 1]: k for k in orphans}, fx0, fy0, cells.shape)
    no_poly = [k for k in orphans if not (xen == k).any()]
    if no_poly:
        grown = expand_labels(np.where(np.isin(nuc, no_poly), nuc, 0), distance=um_to_px(p.fallback_max_expansion_um))
        xen = np.where((xen == 0) & (grown > 0), grown, xen)
    # inside a Xenium cell the nearest nucleus is (almost always) its own, so one global EDT is enough
    xen[ndi.distance_transform_edt(nuc == 0) > um_to_px(p.fallback_max_expansion_um)] = 0
    out = cells.copy()
    free = out == 0
    nxt = int(cells.max())
    new: dict[int, int] = {}
    xen_sl = ndi.find_objects(np.maximum(xen, np.where(np.isin(nuc, orphans), nuc, 0)))
    for k in orphans:
        sl = xen_sl[k - 1] if k - 1 < len(xen_sl) else None
        if sl is None:
            continue
        m = free[sl] & ((xen[sl] == k) | (nuc[sl] == k))
        # clipping can cut a Xenium cell into fragments: keep only what is attached to its own nucleus
        frag, _ = ndi.label(m)
        attached = np.unique(frag[(nuc[sl] == k) & m])
        m = np.isin(frag, attached[attached > 0])
        if m.any():
            full = np.zeros_like(free)
            full[sl] = m
            m = full
            nxt += 1
            out[m] = nxt
            free &= ~m
            new[nxt] = k
    return out, new


# ---------------------------------------------------------------- 9. Segger extension

def _hull_mask(pts_rc: np.ndarray, shape: tuple[int, int]) -> tuple[tuple[slice, slice], np.ndarray] | None:
    from scipy.spatial import ConvexHull, QhullError
    H, W = shape
    try:
        hull = pts_rc[ConvexHull(pts_rc).vertices]
    except (QhullError, ValueError):
        return None
    r0, r1 = max(0, int(np.floor(hull[:, 0].min()))), min(H, int(np.ceil(hull[:, 0].max())) + 1)
    c0, c1 = max(0, int(np.floor(hull[:, 1].min()))), min(W, int(np.ceil(hull[:, 1].max())) + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    yy, xx = np.mgrid[r0:r1, c0:c1]
    inside = MplPath(hull).contains_points(np.column_stack([yy.ravel(), xx.ravel()])).reshape(r1 - r0, c1 - c0)
    return (slice(r0, r1), slice(c0, c1)), inside


def segger_extend(final: np.ndarray, cell_of_nucleus: dict[str, int], seg: pd.DataFrame, fx0: int, fy0: int,
                  min_transcripts: int = 5) -> tuple[np.ndarray, pd.DataFrame]:
    """Grow final cells into free space using Segger's transcript assignment (seg: x_location, y_location,
    segger_cell_id; '-nx' = unassigned). A Segger cell's region is the convex hull of its transcripts (the
    convention seg_methods_utils.segger_polygons_px uses for the benchmark figure). If the Segger cell is
    anchored to a nucleus we already have a cell for, the free part of its hull that touches that cell is
    added to it; otherwise the free part becomes a new cell. Existing pixels are never reassigned."""
    seg = seg[~seg.segger_cell_id.isna() & ~seg.segger_cell_id.astype(str).str.endswith("-nx")]
    out = final.copy()
    nxt = int(final.max())
    rows = []
    for sid, g in seg.groupby("segger_cell_id", sort=False):
        if len(g) < min_transcripts:
            continue
        pts = np.column_stack([g.y_location.to_numpy() / UM_PER_PX - fy0, g.x_location.to_numpy() / UM_PER_PX - fx0])
        hm = _hull_mask(pts, out.shape)
        if hm is None:
            continue
        sl, inside = hm
        free = inside & (out[sl] == 0)
        target = cell_of_nucleus.get(str(sid))
        if target is not None:
            grown = free | (out[sl] == target)
            lab, _ = ndi.label(grown)
            ids = np.unique(lab[out[sl] == target])
            add = free & np.isin(lab, ids[ids > 0])
            kind = "extended"
        else:
            lab, n = ndi.label(free)
            if n == 0:
                continue
            biggest = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
            add = lab == biggest
            nxt += 1
            target = nxt
            kind = "new"
        if not add.any():
            continue
        out[sl][add] = target
        rows.append({"segger_cell_id": sid, "cell": target, "kind": kind, "n_transcripts": len(g),
                     "added_um2": round(float(add.sum()) * UM_PER_PX ** 2, 2)})
    return out, pd.DataFrame(rows, columns=["segger_cell_id", "cell", "kind", "n_transcripts", "added_um2"])


def merge_touching_pieces(cells: np.ndarray, no_nucleus_ids, fallback_nucleus: dict[int, int],
                          nuc: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """A no-nucleus 18S piece that overlaps the nucleus of a fallback cell is that cell's RNA-rich cytoplasm or
    process, not a separate object: it joins that cell (the nucleus it overlaps most, if several). Returns the
    relabelled image and {piece: cell it joined}."""
    cell_of_nucleus = {k: lbl for lbl, k in fallback_nucleus.items()}
    ov = overlap_table(np.where(np.isin(cells, list(no_nucleus_ids)), cells, 0), nuc)
    ov = ov[ov.b.isin(cell_of_nucleus.keys())].sort_values("px", ascending=False).drop_duplicates("a")
    out = cells.copy()
    joined = {}
    for piece, k, _px in ov.itertuples(index=False):
        out[cells == piece] = cell_of_nucleus[int(k)]
        joined[int(piece)] = cell_of_nucleus[int(k)]
    return out, joined


def carve_satellites(pieces: np.ndarray, info: pd.DataFrame, nuc: np.ndarray, elevation: np.ndarray,
                     p: Params) -> tuple[np.ndarray, pd.DataFrame]:
    """Nuclei on the edge of a large 18S piece are separate cells, not that piece's nucleus. A soma's own
    nucleus is surrounded by cytoplasm, so its depth in the mask (distance to the mask edge) exceeds its own
    radius; a nucleus with depth < radius + 1 um in a piece >= glia_mask_over_nucleus x its area -- or an
    unowned nucleus within glia_rim_um of such a piece -- gets its own cell: itself plus the piece's pixels
    within glia_rim_um of it. The piece keeps its central nucleus if it has one, otherwise continues without
    an in-plane nucleus. Bits cut off from the body by a satellite join the nearest satellite."""
    owner, _ = nucleus_owners(pieces, nuc, p)
    nuc_px = np.bincount(nuc.ravel())
    rim_px = um_to_px(p.glia_rim_um)
    out = pieces.copy()
    nxt = int(pieces.max())
    new_rows = []
    slices = ndi.find_objects(pieces)
    for pid in info.index:
        sl0 = slices[pid - 1] if pid - 1 < len(slices) else None
        if sl0 is None:
            continue
        pad = int(rim_px) + 2
        sl = tuple(slice(max(0, s.start - pad), min(dim, s.stop + pad)) for s, dim in zip(sl0, pieces.shape))
        m = out[sl] == pid
        if not m.any():
            continue
        depth = ndi.distance_transform_edt(m) * UM_PER_PX
        near = ndi.distance_transform_edt(~m) <= rim_px
        cand = [int(k) for k in np.unique(nuc[sl][near | m]) if k]
        # the piece's own nucleus: of its owned, enclosed nuclei the one deepest beyond its radius. It is never a
        # satellite, so a thin soma whose nucleus fills most of its width keeps it (2026-09-30)
        inner = enclosed_nuclei(m, nuc[sl])
        keeper = None if not p.satellite_keep_deepest else max((k for k in cand if owner.get(k) == pid and k in inner),
                     key=lambda k: float(depth[(nuc[sl] == k) & m].max(initial=0)) - np.sqrt(nuc_px[k] / np.pi) * UM_PER_PX,
                     default=None)
        sats = []
        for k in cand:
            if k == keeper:
                continue
            nk = nuc[sl] == k
            if m.sum() < p.glia_mask_over_nucleus * nuc_px[k]:
                continue
            o = owner.get(k)
            if o == pid:
                radius = np.sqrt(nuc_px[k] / np.pi) * UM_PER_PX
                if float(depth[nk & m].max(initial=0)) < radius + 1.0:
                    sats.append(k)
            elif o is None:
                sats.append(k)
        if not sats:
            continue
        body = m.copy()
        claimed = []
        for k in sats:
            nk = nuc[sl] == k
            region = (body & (ndi.distance_transform_edt(~nk) <= rim_px)) | (nk & (out[sl] == 0))
            if not region.any():
                continue
            body &= ~region
            nxt += 1
            out[sl][region] = nxt
            claimed.append(nxt)
            new_rows.append({"piece": nxt, "comp": int(info.at[pid, "comp"]), "comp_flag": info.at[pid, "comp_flag"],
                             "mismatch": False, "method": "satellite_edge", "n_pieces_from_comp": 1})
        frag, n = ndi.label(body)
        if n > 1 and claimed:
            keep_nuc = [k for k in cand if k not in sats and owner.get(k) == pid]
            anchor = np.unique(frag[np.isin(nuc[sl], keep_nuc) & body]) if keep_nuc else []
            anchor = [a for a in anchor if a] or [1 + int(np.argmax(np.bincount(frag.ravel())[1:]))]
            loose = body & ~np.isin(frag, anchor)
            if loose.any():
                seeds = np.where(np.isin(out[sl], claimed), out[sl], 0)
                grown = watershed(-elevation[sl], seeds, mask=loose | (seeds > 0))
                out[sl][loose] = grown[loose]
    extra = pd.DataFrame(new_rows).set_index("piece") if new_rows else None
    info = pd.concat([info, extra]) if extra is not None else info
    # a piece can be carved away entirely (all of it within 2 um of its satellites); drop it from the table
    present = set(np.unique(out[out > 0]).tolist())
    return out, info[info.index.isin(present)]


# ---------------------------------------------------------------- whole region

def tissue_mask(dapi: np.ndarray, thr: float | None = None) -> np.ndarray:
    """DAPI above the 5th percentile of its non-zero values (thr: the whole-sample value for a tile), holes
    filled, dilated 5 px."""
    if thr is None:
        nz = dapi[dapi > 0]
        thr = float(np.percentile(nz, 5)) if nz.size else 0.0
    m = ndi.binary_dilation(ndi.binary_fill_holes(dapi > thr), iterations=5)
    return m if m.any() else np.ones_like(m)


def segment_region(crop: np.ndarray, pct: tuple[float, float, float, float], nb: pd.DataFrame,
                   cb: pd.DataFrame, tx: pd.DataFrame, fx1: int, fy1: int, p: Params,
                   sample_stats: dict | None = None) -> dict:
    """The whole method on one region. crop: (4, H, W) level-0 channels DAPI / boundary / 18S / Vimentin;
    pct: whole-sample (18S p1, 18S p99, Vimentin p1, Vimentin p99), the p99s are the signal's upper anchors
    (lower anchors: the tissue medians in sample_stats); nb / cb: Xenium nucleus / cell
    boundaries (cell_id, vertex_x, vertex_y in um); tx: QC-passed transcripts (x_location, y_location,
    feature_name in um); fx1 / fy1: the region's level-0 origin; sample_stats: whole-sample "thr", "thr_alt",
    "dapi_ref" and "dapi_tissue_thr" when the region is one tile of a sample (see scg/run_fullsample.py), else they are taken
    from the region itself. Returns every intermediate and final label image plus the per-piece and per-cell
    tables."""
    st = dict(sample_stats or {})
    # the whole-sample quantile tables (when present) give the thresholds at p's percentiles, so a stats file stays
    # valid whatever percentile Params sets; the stored "thr" / "thr_alt" are only used without them
    for key, table, q in (("thr", "filt_quantiles", p.thr_percentile), ("thr_alt", "raw_quantiles", p.alt_thr_percentile)):
        if table in st and f"{q:g}" in st[table]:
            st[key] = st[table][f"{q:g}"]
    valid = tissue_mask(crop[0], st.get("dapi_tissue_thr"))
    # channel backgrounds: whole-sample tissue medians (the region's own only when run without sample stats)
    bg18 = st.get("bg_18s") or float(np.median(crop[2][valid]))
    bgvim = st.get("bg_vim") or float(np.median(crop[3][valid]))
    diff = sample_signal(crop, pct, bg18, bgvim)
    bg_unit = bg18 / max(float(pct[1]) - bg18, 1e-6)  # one 18S background in the normalized signal's units
    if p.thr_value is not None:
        st["thr"] = p.thr_value
    elif p.thr_bg_multiple is not None:
        if bg18 <= 0:
            raise ValueError("thr_bg_multiple needs a positive 18S background (bg_18s is 0); set thr_value instead")
        st["thr"] = p.thr_bg_multiple * bg_unit

    comp, filt, thr = segment(diff, valid, p.median_px, p.thr_percentile, p.min_size_px, st.get("thr"),
                              closing_px=um_to_px(p.soma_closing_um))
    alt, _, thr_alt = segment(diff, valid, p.alt_median_px, p.alt_thr_percentile, p.min_size_px, st.get("thr_alt"))

    nuc, cell_ids = rasterize_nuclei(nb, fx1, fy1, comp.shape)
    n_xenium_nuclei = len(cell_ids)
    nuc, n_rescued = rescue_nuclei(crop[0], nuc, valid, p, st.get("dapi_ref"), inside=comp > 0)
    cell_ids = cell_ids + [f"dapi_rescue_{i}" for i in range(1, n_rescued + 1)]
    if p.vim_rescue_min_um2 is not None:
        blind = np.clip((crop[2] - bg18) / max(float(pct[1]) - bg18, 1e-6), 0, 1)   # 18S alone, same scaling
        comp_nv, _, _ = segment(blind, valid, p.median_px, p.thr_percentile, p.min_size_px, st.get("thr"), closing_px=um_to_px(p.soma_closing_um))
        comp = vim_rescue(comp, comp_nv, nuc, valid, p)
    if p.soma_grow_frac is not None:
        grow18 = ndi.gaussian_filter(crop[2].astype(float), um_to_px(p.soma_grow_sigma_um))
        comp = grow_dim_somata(comp, grow18, nuc, valid, bg18, float(pct[1]), p)
    poly_area = polygon_areas_um2(nb).reindex(cell_ids[:n_xenium_nuclei]).to_numpy()
    rescued_area = np.bincount(nuc.ravel(), minlength=len(cell_ids) + 1)[n_xenium_nuclei + 1:] * UM_PER_PX ** 2
    nuc_area_um2 = np.r_[0.0, poly_area, rescued_area]
    is_neuron = nuc_area_um2 >= p.neuron_nucleus_min_um2
    is_neuron[0] = False

    raw18 = ndi.gaussian_filter(crop[2].astype(float), um_to_px(p.soma_peak_sigma_um))
    flags = compute_flags(comp, nuc, alt, raw18, is_neuron, p)
    pieces, info = resolve(comp, nuc, alt, raw18, filt, flags, is_neuron, p)
    pieces, info = carve_satellites(pieces, info, nuc, filt, p)
    pieces, owner = absorb_owned_nuclei(pieces, nuc, p)

    genes = tx["feature_name"].astype(str).to_numpy()
    tx_piece = transcript_pixel_labels(tx, pieces, fx1, fy1)
    table = classify_pieces(pieces, nuc, owner, tx_piece, info, is_neuron, p)

    merged, merges = merge_dendrites(pieces, table, tx_piece, genes, p)

    rejected = table.index[table["class"].isin({CLASS_REJECT_SMALL, CLASS_REJECT_DENDRITE})]
    keep18 = np.where(np.isin(merged, rejected), 0, merged)
    # a nucleus whose piece was rejected has no cell yet, so it is eligible for the fallback
    owner_kept = {k: v for k, v in owner.items() if v not in set(rejected)}
    glia: dict[int, int] = {}
    with_glia = keep18
    if p.glia_soma_thr_value is not None or p.glia_soma_bg_multiple is not None:
        if p.glia_soma_thr_value is None and bg18 <= 0:
            raise ValueError("glia_soma_bg_multiple needs a positive 18S background; set glia_soma_thr_value instead")
        glia_thr = p.glia_soma_thr_value if p.glia_soma_thr_value is not None else p.glia_soma_bg_multiple * bg_unit
        with_glia, glia = add_glial_somata(keep18, nuc, owner_kept, len(cell_ids), filt, valid, glia_thr, p)
    owner_all = {**owner_kept, **{k: lbl for lbl, k in glia.items()}}
    final, fallback = add_fallback_cells(with_glia, nuc, owner_all, cell_ids, cb, fx1, fy1, p)
    no_nuc_18s = set(np.unique(keep18[keep18 > 0])) - set(owner_kept.values())
    final, joined = merge_touching_pieces(final, no_nuc_18s, {**glia, **fallback}, nuc)
    final_18s = np.where(keep18 > 0, final, 0)  # 18S pixels only, labelled as in the final cell set

    has_nuc = set(owner_kept.values())
    tx_final = transcript_pixel_labels(tx, final, fx1, fy1)
    final_ids = np.unique(final[final > 0])
    area_um2 = np.bincount(final.ravel(), minlength=int(final.max()) + 1) * UM_PER_PX ** 2
    n_tx_final = np.bincount(tx_final, minlength=int(final.max()) + 1)
    cells = pd.DataFrame({
        "source": ["xenium_fallback" if i in fallback else "glia_18S" if i in glia else "18S" for i in final_ids],
        "has_nucleus": [i in fallback or i in glia or i in has_nuc for i in final_ids],
        "xenium_cell_id": [cell_ids[fallback[i] - 1] if i in fallback else cell_ids[glia[i] - 1] if i in glia else ""
                           for i in final_ids],
        "area_um2": area_um2[final_ids].round(2),
        "n_transcripts": n_tx_final[final_ids],
        "n_18s_pieces_joined": [sum(1 for v in joined.values() if v == i) for i in final_ids],
    }, index=pd.Index(final_ids, name="cell"))

    return dict(comp=comp, alt=alt, filt=filt, thr=thr, thr_alt=thr_alt, nuc=nuc, cell_ids=cell_ids,
                n_xenium_nuclei=n_xenium_nuclei, n_rescued=n_rescued, is_neuron=is_neuron, flags=flags,
                pieces=pieces, owner=owner, table=table, merged=merged, merges=merges, keep18=keep18,
                owner_kept=owner_kept, final=final, fallback=fallback, glia=glia, joined=joined, final_18s=final_18s,
                cells=cells, genes=genes, tx_piece=tx_piece, tx_final=tx_final)
