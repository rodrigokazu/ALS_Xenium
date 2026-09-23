#!/usr/bin/env python3
# ========================================================================================
# ALS_Xenium repo commentary | ck_spatial_pipeline/2_qc/dual_pass/verify_traces_v6.py
#
# "status: OK" in a trace only means no cell raised. This check compares each trace's raw
# motor-neuron count on the unfiltered object with the count derived from the parquet. A
# mismatch means the relabel joined the wrong rows.
# ========================================================================================

"""Verify each completed v6 trace against ground truth derived from the parquet.

"status: OK" only means no cell raised. It does not mean the labels are right. The
gate here is that the trace's RAW motor-neuron count (measured on the unfiltered
object, before QC and smear removal) equals the count computed independently from
the pinned v6 parquet for that sample. EXPECTED_RAW below came from
preflight_exec.py, which executed the injected cell against the parquet and the
real niche h5ad; those numbers sum to 599, matching a direct parquet query.

The FINAL count (post QC + smear removal) is reported but not asserted: filtering
legitimately removes cells, so final <= raw is the only invariant available.

usage: verify_traces_v6.py            # every trace present so far
"""
import json
import sys
from pathlib import Path

V = Path("/oak/stanford/scg/lab_mpsnyder/johnck/Projects/RK/Spatial/"
         "Novae_IND_QC/runs_v6")

# (raw is_MN, raw is_MN <NA>) per sample, from preflight_exec.py on the pinned v6
# snapshot md5 d32b298e9662e023cc1794367680a0ce
EXPECTED_RAW = {
    "SD00614_BG": (56, 0),      "SD01015_BG": (0, 47270),
    "SD01115_BG": (30, 0),      "SD01320_BG": (49, 0),
    "SD01413_BA": (30, 0),      "SD01616_BI": (0, 70274),
    "SD01620_BI": (96, 0),      "SD01623_BI": (40, 0),
    "SD01915_BG": (3, 0),       "SD01922_BI": (4, 0),
    "SD01923_BI": (12, 0),      "SD02022_BI": (35, 0),
    "SD02622_BI": (8, 0),       "SD02818_BG": (69, 0),
    "SD02913_BA": (10, 0),      "SD03522_BG": (29, 0),
    "SD03614_BG": (31, 0),      "SD03914_BG": (53, 0),
    "SD04219_BI": (36, 0),      "SD05413_BG": (8, 0),
}
EXPECT_TOTAL = sum(v[0] for v in EXPECTED_RAW.values())     # 599
NO_VH = {"SD01015_BG", "SD01616_BI"}


def main():
    traces = sorted(V.glob("trace_src_*.json"))
    if not traces:
        print(f"no traces in {V} yet")
        return 0
    print(f"{'sample':<12}{'status':<12}{'raw MN':>8}{'exp':>6}{'raw NA':>9}"
          f"{'exp':>8}{'final MN':>10}{'cells':>10}  verdict")
    print("-" * 100)
    bad, inprog, seen_raw = [], [], 0
    for t in traces:
        b = json.loads(t.read_text())
        s = b.get("sample")
        run = b.get("run", {})
        st = run.get("status")
        raw = b.get("n_is_MN_raw")
        rawna = b.get("n_is_MN_na_raw")
        fin = b.get("n_is_MN_final")
        finna = b.get("n_is_MN_na_final")
        cells = b.get("final_cells")
        exp, expna = EXPECTED_RAW.get(s, (None, None))

        # A trace with no `run` block, or an explicit INCOMPLETE flag, is being
        # written right now: the notebook's own trace cell lands the body first and
        # run_dualpass_src.py adds `run` when it finishes. Treating that as a
        # FAILURE produced a false alarm at 11/20 that cleared on re-read. A
        # monitor that cries wolf gets ignored, so this is IN-PROGRESS, not bad.
        if run == {} or b.get("INCOMPLETE") or run.get("status") is None:
            inprog.append(s)
            print(f"{s:<12}{'(writing)':<12}{'-':>8}{str(exp):>6}{'-':>9}"
                  f"{str(expna):>8}{'-':>10}{'-':>10}  in progress, re-check")
            continue

        probs = []
        if st != "OK":
            probs.append(f"status={st}")
        if run.get("cell_errors"):
            probs.append(f"{len(run['cell_errors'])} cell errors")
        if b.get("vh_relabel_version") != "v6":
            probs.append(f"version={b.get('vh_relabel_version')!r}")
        cg = b.get("coord_gate_max_delta_um")
        if cg is None or cg >= 1e-3:
            probs.append(f"coord_gate={cg}")
        if exp is not None and raw != exp:
            probs.append(f"RAW MN {raw} != expected {exp}")
        if expna is not None and rawna != expna:
            probs.append(f"RAW NA {rawna} != expected {expna}")
        if fin is not None and raw is not None and fin > raw:
            probs.append(f"final {fin} > raw {raw}")
        if s in NO_VH and (fin or 0) != 0:
            probs.append(f"mask-less section reports {fin} MN, must be 0 with NA")
        if raw is not None:
            seen_raw += raw
        print(f"{s:<12}{str(st):<12}{str(raw):>8}{str(exp):>6}{str(rawna):>9}"
              f"{str(expna):>8}{str(fin):>10}{str(cells):>10}  "
              f"{'OK' if not probs else '; '.join(probs)}")
        if probs:
            bad.append((s, probs))
    print("-" * 100)
    n_done = len(traces) - len(inprog)
    print(f"{len(traces)}/20 traces present ({n_done} complete"
          + (f", {len(inprog)} still writing: {inprog}" if inprog else "")
          + f"); raw MN summed over the complete ones = {seen_raw}")
    if n_done == 20:
        print(f"cohort raw MN total {seen_raw} (expected {EXPECT_TOTAL})"
              f"  {'MATCH' if seen_raw == EXPECT_TOTAL else 'MISMATCH'}")
    if bad:
        print(f"\n{len(bad)} SAMPLE(S) FAILED VERIFICATION:")
        for s, p in bad:
            print(f"  {s}: {'; '.join(p)}")
        return 2
    print("all present traces verify against the parquet-derived ground truth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
