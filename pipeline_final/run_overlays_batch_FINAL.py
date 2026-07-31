#!/usr/bin/env python
# ========================================================================================
# ALS_Xenium repo commentary | pipeline_final/run_overlays_batch_FINAL.py
#
# Driver that renders the domain overlays, detail crops and motor-neuron heroes for the
# cohort. --array-index N picks the Nth discovered sample for one array task, --all iterates
# everything.
#
# The fix to know about is bundle resolution. It delegates to final_config.discover_samples()
# instead of matching directory names locally. The old name-normalising matcher could return
# the wrong directory for the two duplicated donors, because the superseded original and the
# redo both normalise to the same string, so it could silently render the sample we
# deliberately excluded.
#
# Reads the per-sample Novae h5ads. They do not exist until the joint Novae run finishes.
# Respect the dependency; running early gives you missing-file errors on all twenty tasks.
#
# CURATED lists the four sections used as worked examples.
# ========================================================================================

"""run_overlays_batch_FINAL.py -- render Xenium domain overlays (overview+detail+MN heroes)
for the _final cohort. Read-only on bundles; writes to OUT.

Adapted from run_overlays_batch.py. Deltas:
  * bundle resolution is DELEGATED to final_config.discover_samples() (the authoritative
    20-sample dedup: drops superseded originals SD01620_BI/SD02022, keeps the redos,
    maps Region_2 -> SD05413_BG, strips trailing '_'). The old norm()-match resolve_bundle
    could return the WRONG dir for the duplicate donors (original vs redo both norm-match).
  * paths point at Ranger_procd_mw_final + Novae_persample_niches_FINAL (via final_config).
  * --array-index N selects the Nth sorted discovered label -> one sample per SLURM array
    task (see run_overlays_FINAL.sh). --all iterates all 20.

The per-sample Novae h5 (<label>__niches.h5ad under NOVAE_JOINT_DIR/per_sample) does not
exist until novae_persample_niches_FINAL runs -> such samples are SKIPPED with a clear
message (expected while staging). Env: xenium_vistools."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_config as fc
import xenium_overlay_FINAL as xo
import hero_crop_FINAL as hc

HDIR = str(fc.NOVAE_JOINT_DIR / "per_sample")
OUT  = str(fc.NOVAE_JOINT_DIR / "xenium_overlays")

# A short curated set for quick spot-checks (must be a subset of the discovered labels).
CURATED = ["SD03614_BG", "SD01915_BG", "SD01320_BG", "SD03522_BG"]


def _sample_dir_map(verbose=False):
    """{label -> bundle dir str} from the authoritative discovery (handles all dedup)."""
    return {lab: str(p) for lab, p in fc.discover_samples(verbose=verbose)}


def sorted_labels():
    return sorted(_sample_dir_map().keys())


def resolve_bundle(sample, dmap=None):
    dmap = dmap or _sample_dir_map()
    return dmap.get(sample)


def main(samples, out=OUT, genes=None, hero_only=False, topk=4, min_sep=450.0):
    os.makedirs(out, exist_ok=True)
    dmap = _sample_dir_map(verbose=False)
    for s in samples:
        b = resolve_bundle(s, dmap)
        h5 = f"{HDIR}/{s}__niches.h5ad"
        if b is None or not os.path.exists(h5):
            print(f"[SKIP] {s}: bundle={b} h5exists={os.path.exists(h5)} "
                  f"(Novae _FINAL per-sample h5 pending?)"); continue
        print(f"\n===== {s}  (bundle {os.path.basename(b)}) =====", flush=True)
        if not hero_only:
            try:
                xo.main(s, b, h5, out, genes=genes)
            except Exception as e:
                print(f"[ERR overview/detail] {s}: {e}")
        try:
            hc.main(s, b, h5, out, half=120, topk=topk, genes=genes, min_sep=min_sep)
        except Exception as e:
            print(f"[ERR hero] {s}: {e}")
    print("\nALL DONE ->", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    out, genes, hero_only, topk, min_sep = OUT, None, False, 4, 450.0
    array_index = None
    if "--out" in args:
        i = args.index("--out"); out = args[i+1]; del args[i:i+2]
    if "--genes" in args:
        i = args.index("--genes"); genes = args[i+1].split(","); del args[i:i+2]
    if "--topk" in args:
        i = args.index("--topk"); topk = int(args[i+1]); del args[i:i+2]
    if "--min-sep" in args:
        i = args.index("--min-sep"); min_sep = float(args[i+1]); del args[i:i+2]
    if "--array-index" in args:
        i = args.index("--array-index"); array_index = int(args[i+1]); del args[i:i+2]
    if "--hero-only" in args:
        hero_only = True; args.remove("--hero-only")

    if array_index is not None:
        labs = sorted_labels()
        if array_index < 0 or array_index >= len(labs):
            print(f"[array] index {array_index} out of range 0..{len(labs)-1} "
                  f"({len(labs)} discovered samples) -- nothing to do."); sys.exit(0)
        samples = [labs[array_index]]
    elif args and args[0] == "--all":
        samples = sorted_labels()
    elif args:
        samples = args
    else:
        samples = CURATED
    print("Samples:", samples, "| out:", out, "| genes:", genes, "| hero_only:", hero_only,
          "| topk:", topk, "| min_sep:", min_sep)
    main(samples, out=out, genes=genes, hero_only=hero_only, topk=topk, min_sep=min_sep)
