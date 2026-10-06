#!/usr/bin/env python3
"""Paired spatial-block bootstrap on the precomputed benchmark data, and the benchmark figure.
python bench_stats.py <chosen.pkl> <heldout.pkl or -> <out png> [ours method name] [ref method name]"""
import sys, pickle, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
B_REP = 2000
chosen = pickle.load(open(sys.argv[1], "rb")); held = pickle.load(open(sys.argv[2], "rb")) if sys.argv[2] != "-" else None
OUT = Path(sys.argv[3]); OURS = sys.argv[4] if len(sys.argv) > 4 else "ours"; REF = sys.argv[5] if len(sys.argv) > 5 else "10x"
MG, MT = chosen["markers"], chosen["types"]; types = sorted(set(MT)); tix = np.array([types.index(t) for t in MT])
cross = [(i, j) for i in range(len(MG)) for j in range(i + 1, len(MG)) if MT[i] != MT[j]]
ci_, cj_ = np.array([p[0] for p in cross]), np.array([p[1] for p in cross])

def mecr(A, w):
    Af = A.astype(np.float32); C = Af.T @ (w[:, None] * Af); cnt = np.diag(C)
    both = C[ci_, cj_]; either = cnt[ci_] + cnt[cj_] - both; ok = (cnt[ci_] > 0) & (cnt[cj_] > 0) & (either > 0)
    return float(np.mean(both[ok] / either[ok])) if ok.any() else np.nan

def mixed_frac(A, w):
    if len(A) == 0 or w.sum() == 0: return np.nan
    m = np.zeros((len(A), len(types)), bool)
    for t in range(len(types)): m[:, t] = A[:, tix == t].any(1)
    return float((w * (m.sum(1) >= 2)).sum() / w.sum())

def metrics(data, method, W):                       # W: {window: weight per block}
    cov_n = cov_d = 0.0; A_full, w_full, A_ds, w_ds = [], [], [], []
    for wn, d in data["data"].items():
        m = d["methods"][method]; w = W[wn]
        cov_n += float((w * m["assigned_by_block"]).sum()); cov_d += float((w * d["total_by_block"]).sum())
        wc = w[m["cell_block"]]; A_full.append(m["pres"]); w_full.append(wc)
        big = m["big"]; A_ds.append(m["pres_ds"][big]); w_ds.append(wc[big])
    Af, wf, Ad, wd = np.vstack(A_full), np.concatenate(w_full), np.vstack(A_ds), np.concatenate(w_ds)
    return {"recovered": 100 * cov_n / cov_d, "mecr1000": mecr(Ad, wd), "mecr": mecr(Af, wf), "mixed": 100 * mixed_frac(Ad, wd)}

def run(data, methods, seed=1):
    rng = np.random.default_rng(seed); wins = list(data["data"])
    ones = {wn: np.ones(data["data"][wn]["n_blocks"]) for wn in wins}
    est = {m: metrics(data, m, ones) for m in methods}
    units = {wn: np.flatnonzero(data["data"][wn]["total_by_block"] > 0) for wn in wins}
    reps = {m: [] for m in methods}
    for _ in range(B_REP):
        W = {}
        for wn in wins:
            u = rng.choice(units[wn], size=len(units[wn]), replace=True); W[wn] = np.bincount(u, minlength=data["data"][wn]["n_blocks"]).astype(float)
        for m in methods: reps[m].append(metrics(data, m, W))
    return est, {m: pd.DataFrame(r) for m, r in reps.items()}

def diff_test(reps, a, b, col):
    d = (reps[a][col] - reps[b][col]).dropna().to_numpy()
    lo, hi = np.percentile(d, [2.5, 97.5]); p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d))
    return float(np.mean(d)), float(lo), float(hi), float(p)

def stars(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

methods = list(next(iter(chosen["data"].values()))["methods"])
CACHE = OUT.with_suffix('.cache.pkl')
if CACHE.exists():
    est, reps, hest, hreps = pickle.load(open(CACHE, 'rb'))
else:
    est, reps = run(chosen, methods); hest = hreps = None
rows = []
for m in methods:
    for col in ("recovered", "mecr1000", "mecr", "mixed"):
        lo, hi = np.percentile(reps[m][col].dropna(), [2.5, 97.5])
        r = {"method": m, "metric": col, "estimate": est[m][col], "ci_lo": lo, "ci_hi": hi}
        if m != REF: r["diff_vs_ref"], r["diff_lo"], r["diff_hi"], r["p"] = diff_test(reps, m, REF, col)
        rows.append(r)
res = pd.DataFrame(rows); res.to_csv(OUT.with_suffix(".csv"), index=False)
pd.set_option("display.width", 220); print(res.round(4).to_string(index=False))

BLUE, ORANGE, GREY, DARK = "#2a78d6", "#eb6834", "#9a9890", "#0b0b0b"
col_of = lambda m: ORANGE if m == REF else BLUE if m == OURS else GREY
LAB = {"recovered": "Transcripts recovered\n(% of QC transcripts in a cell, higher = more)", "mecr1000": "Cell purity: MECR at 1,000 tx / cell\n(lower = cleaner)",
       "mecr": "Cell purity: MECR, all cells ≥ 5 tx\n(lower = cleaner)", "mixed": "Cells mixing ≥ 2 cell types\n(% of cells at 1,000 tx, lower = cleaner)"}
nrow = 2 if held else 1
fig = plt.figure(figsize=(20, 5.6 * nrow + 0.5), dpi=140)
gs = fig.add_gridspec(nrow, 4, hspace=0.75, wspace=0.55, left=0.12, right=0.985, top=0.9, bottom=0.12)
order = [REF] + [m for m in methods if m not in (REF, OURS)] + [OURS]
for k, col in enumerate(("recovered", "mecr1000", "mecr", "mixed")):
    ax = fig.add_subplot(gs[0, k]); ys = np.arange(len(order))[::-1]
    for y, m in zip(ys, order):
        e = res[(res.method == m) & (res.metric == col)].iloc[0]
        ax.plot([e.ci_lo, e.ci_hi], [y, y], color=col_of(m), lw=2.2, solid_capstyle="round"); ax.plot(e.estimate, y, "o", color=col_of(m), ms=8, mec="white", mew=0.8, zorder=3)
        if m != REF:
            ax.text(max(e.ci_hi, e.estimate), y, "  " + stars(e.p), va="center", fontsize=9, color=DARK)
    ax.set_yticks(ys); ax.set_yticklabels(order if k == 0 else [""] * len(order), fontsize=10)
    ax.set_title(LAB[col], loc="left", fontsize=10.5); ax.grid(axis="x", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.tick_params(colors="#52514e")
    if k == 0: [t.set_color(col_of(m)) for t, m in zip(ax.get_yticklabels(), order)]
fig.text(0.005, 0.965, f"Benchmark on three 500 µm Atera cerebellum windows (pooled; point = estimate, bar = 95% interval of a paired spatial-block bootstrap, 50 µm blocks, {B_REP:,} resamples). "
         f"Stars: bootstrap p of the difference to {REF} (* p<0.05, ** p<0.01, *** p<0.001).", fontsize=10.5, color="#52514e", ha="left")
if held:
    hm = [m for m in next(iter(held["data"].values()))["methods"]]; hwins = list(held["data"])
    if hreps is None:
        hest, hreps = run(held, hm, seed=2); pickle.dump((est, reps, hest, hreps), open(CACHE, 'wb'))
    # per-window point estimates
    ones = lambda d: {wn: np.ones(d["data"][wn]["n_blocks"]) for wn in d["data"]}
    pw = {m: {wn: metrics({"data": {wn: held["data"][wn]}}, m, {wn: np.ones(held["data"][wn]["n_blocks"])}) for wn in hwins} for m in hm}
    for k, col in enumerate(("recovered", "mecr1000", "mecr", "mixed")):
        ax = fig.add_subplot(gs[1, k]); dif = np.array([pw[OURS][wn][col] - pw[REF][wn][col] for wn in hwins])
        better = (dif > 0) if col == "recovered" else (dif < 0)
        ax.axvline(0, color="#52514e", lw=0.9)
        for i, (wn, dv) in enumerate(zip(hwins, dif)): ax.plot(dv, len(hwins) - i, "o", ms=9, color=BLUE if better[i] else ORANGE, mec="white", zorder=3)
        ax.set_yticks(range(1, len(hwins) + 1)); ax.set_yticklabels(hwins[::-1] if k == 0 else [""] * len(hwins), fontsize=9)
        d, lo, hi, p = diff_test(hreps, OURS, REF, col)
        from scipy.stats import binomtest
        sp = binomtest(int(better.sum()), len(dif), 0.5).pvalue
        ax.plot([lo, hi], [0.1, 0.1], color=DARK, lw=3, solid_capstyle="butt"); ax.plot(d, 0.1, "D", color=DARK, ms=7)
        ax.set_ylim(-0.2, len(hwins) + 0.8)
        short = {"recovered": "Δ transcripts recovered (points)", "mecr1000": "Δ MECR at 1,000 tx / cell", "mecr": "Δ MECR, all cells", "mixed": "Δ cells mixing ≥ 2 types (points)"}[col]
        ax.set_title(short + f"\nours − 10x; better in {int(better.sum())}/{len(dif)} windows, sign test p = {sp:.3f}", loc="left", fontsize=9.5)
        ax.text(0.0, 1.0 - 0.0, "", transform=ax.transAxes)
        lab = f"pooled Δ {d:+.3g}  [{lo:+.3g}, {hi:+.3g}]   bootstrap p = {p:.3g} ({stars(p)})"
        ax.text(0.5, -0.17, lab, transform=ax.transAxes, ha="center", va="top", fontsize=9, color=DARK)
        ax.grid(axis="x", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True); ax.tick_params(colors="#52514e")
    fig.text(0.005, 0.485, f"Out-of-sample check: six held-out 500 µm windows (random positions, never used for tuning); each dot = one window, ours minus 10x on identical transcripts "
             f"(blue = ours better, orange = worse); black diamond and bar = pooled difference with 95% bootstrap interval.", fontsize=10.5, color="#52514e", ha="left")
fig.savefig(OUT, facecolor="white"); print("saved", OUT)
