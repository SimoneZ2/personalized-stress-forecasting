#!/usr/bin/env python3
"""RQ1 statistics on the rows of Table 1.

1. Paired comparisons A - B of the absolute error (negative = A better): pooled
   difference, 95% CI by bootstrap over users (2,000 resamples), Wilcoxon test on
   the per-user mean differences, share of users where A is better.
2. Mean signed error (prediction - true) on peak and non-peak weeks, with a 95%
   bootstrap CI over users and the share of underestimated weeks.

Usage:
  python evaluation/rq1_tests.py --npz ces_stress_forecast.npz \
      --wf-pattern results/wf_seed{s}/predictions_wf.csv \
      --luo-pattern results/luo_seed{s}/predictions_luo.csv [--seeds 42 43 44]
The signed error is computed on the first seed.
"""

import argparse
import os

import numpy as np
from scipy.stats import wilcoxon

from common import load_predictions

N_BOOT = 2000
SYSTEM = "y_pred_A_peak_only"
COMPARISONS = [("MAE", "LSTM - Personal mean", "y_pred_lstm", "pm_base", False, 1),
               ("MAE", "LSTM - Last EMA", "y_pred_lstm", "last_base", False, 1),
               ("MAE", "System - LSTM", SYSTEM, "y_pred_lstm", False, 1),
               ("MAE_peak", "System - LSTM", SYSTEM, "y_pred_lstm", True, 3),
               ("MAE_peak", "System - Last EMA", SYSTEM, "last_base", True, 3)]
SIGNED = [("LSTM", "y_pred_lstm"), ("Full system", SYSTEM),
          ("Personal mean (sparse)", "pm_base"), ("Last EMA (sparse)", "last_base")]


def paired(d, a, b, peak_only, min_peaks, rng):
    v = d[d["is_peak"] == 1] if peak_only else d
    diff = (v["y_true"] - v[a]).abs() - (v["y_true"] - v[b]).abs()
    g = diff.groupby(v["uid"]).agg(["sum", "count", "mean"])
    g = g[g["count"] >= min_peaks]
    S, N = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(S), size=(N_BOOT, len(S)))
    lo, hi = np.percentile(S[idx].sum(1) / N[idx].sum(1), [2.5, 97.5])
    return S.sum() / N.sum(), lo, hi, wilcoxon(g["mean"]).pvalue, (g["mean"] < 0).mean(), len(g)


def user_bootstrap(err, uid, rng):
    users = np.unique(uid)
    rows = {u: np.flatnonzero(uid == u) for u in users}
    s = np.array([err[rows[u]].sum() for u in users])
    n = np.array([len(rows[u]) for u in users])
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(users), len(users))
        boot.append(s[i].sum() / n[i].sum())
    return np.percentile(boot, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf-pattern", required=True)
    ap.add_argument("--luo-pattern", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = ap.parse_args()
    patterns = [("WF", args.wf_pattern), ("LUO", args.luo_pattern)]

    rng = np.random.default_rng(0)
    for scheme, pat in patterns:
        print(f"\n===== {scheme}")
        print(f"  {'metric':<9}{'comparison':<22}{'seed':>5}{'diff':>9}{'95% CI':>20}{'p Wilcoxon':>12}"
              f"{'users A better':>17}{'users':>8}")
        for metric, label, a, b, peak_only, min_peaks in COMPARISONS:
            for s in args.seeds:
                d = load_predictions(os.path.expanduser(pat.format(s=s)), scheme, args.npz)
                diff, lo, hi, p, share, n = paired(d, a, b, peak_only, min_peaks, rng)
                print(f"  {metric:<9}{label:<22}{s:>5}{diff:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]{p:>12.2g}"
                      f"{share:>16.0%}{n:>8}")

    rng = np.random.default_rng(0)
    for scheme, pat in patterns:
        v = load_predictions(os.path.expanduser(pat.format(s=args.seeds[0])), scheme, args.npz, with_age=True)
        print(f"\n== {scheme}  rows {len(v)}  users {v.uid.nunique()}  peak weeks {int((v.is_peak == 1).sum())}")
        print("   %-32s %-9s %8s %20s %10s %8s" % ("predictor", "weeks", "signed", "95% CI (users)", "% under", "MAE"))
        for label, col in SIGNED:
            for weeks, m in [("peak", v.is_peak == 1), ("non-peak", v.is_peak == 0), ("all", v.is_peak.notna())]:
                d = v[m & v[col].notna()]
                e = (d[col] - d.y_true).values
                lo, hi = user_bootstrap(e, d.uid.values, rng)
                print("   %-32s %-9s %+8.3f   [%+7.3f, %+7.3f] %9.1f%% %8.3f"
                      % (label, weeks, e.mean(), lo, hi, 100 * (e < 0).mean(), np.abs(e).mean()))
            print()


if __name__ == "__main__":
    main()
