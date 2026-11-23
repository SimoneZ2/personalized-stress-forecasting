#!/usr/bin/env python3
"""Table 2 (RQ2): the peak classifier as an alert, on the rows of Table 1.

- AUC of the GBM probability, of the LSTM forecast and of the last collected self-report
  (both as z-scores w.r.t. the sparse personal mean/std), with 95% CIs of the AUC
  differences by bootstrap over users (2,000 resamples).
- Tolerance metrics with a window of k calendar weeks: precision over alerts (an alert is
  correct if a peak occurs in its target week or within the following k weeks), recall
  over peaks (a peak is detected if an alert targets it or one of the k preceding weeks),
  F1 from the two. LSTM and last-EMA alerts flag the same share of weeks as the GBM flag.
- Random alert: the GBM flags permuted within each user; p = share of permutations with
  a value >= the real one.

Usage: python evaluation/rq2_table.py --wf predictions_wf.csv --luo predictions_luo.csv --npz ces_stress_forecast.npz [--n-perm 1000]
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import load_predictions

N_BOOT = 2000
KS = [0, 1, 2, 4]
METRICS = ["F1", "Precision", "Recall"]


def tolerance(wk, flag, peak, groups, k):
    n_al = n_al_ok = n_pk = n_pk_ok = 0
    for idx in groups:
        w = wk[idx]
        al = np.sort(w[flag[idx]])
        pk = np.sort(w[peak[idx]])
        if len(al):
            n_al += len(al)
            if len(pk):
                j = np.searchsorted(pk, al, side="left")
                n_al_ok += int(((j < len(pk)) & (pk[np.minimum(j, len(pk) - 1)] <= al + k)).sum())
        if len(pk):
            n_pk += len(pk)
            if len(al):
                j = np.searchsorted(al, pk, side="right") - 1
                n_pk_ok += int(((j >= 0) & (al[np.maximum(j, 0)] >= pk - k)).sum())
    p = n_al_ok / n_al if n_al else 0.0
    r = n_pk_ok / n_pk if n_pk else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return f, p, r


def all_k(wk, flag, peak, groups):
    return np.array([tolerance(wk, flag, peak, groups, k) for k in KS]).T   # [metric, k]


def analyse(v, scheme, n_perm, rng):
    n0 = len(v)
    v = v[v.is_peak.notna()].sort_values(["uid", "week_start"]).reset_index(drop=True)
    ws = pd.to_datetime(v.week_start)
    v["wk"] = ((ws - ws.min()).dt.days / 7).round().astype(int)
    groups = [g.index.values for _, g in v.groupby("uid", sort=False)]
    gap = v.groupby("uid").wk.diff().dropna()

    ps = v.personal_std_sparse
    ok = ps.notna() & (ps > 0.01)
    v["lstm_z"] = np.where(ok, (v.y_pred_lstm - v.personal_mean_sparse) / ps, np.nan)
    v["last_z"] = np.where(ok, (v.last_base - v.personal_mean_sparse) / ps, np.nan)
    v["flag_gbm"] = (v.flag_peak_active == 1).astype(int)
    rate = v.flag_gbm.mean()
    for score, fc in [("lstm_z", "flag_lstm"), ("last_z", "flag_last")]:
        thr = np.nanquantile(v[score], 1 - rate)
        v[fc] = (v[score] >= thr).fillna(False).astype(int)
    peak = (v.is_peak == 1).values
    wk = v.wk.values

    a = v[ok & v.y_prob_peak.notna()]
    ya = (a.is_peak == 1).values
    auc = {c: roc_auc_score(ya, a[c]) for c in ["y_prob_peak", "lstm_z", "last_z"]}
    users = a.uid.unique()
    rows = {u: np.flatnonzero(a.uid.values == u) for u in users}
    diffs = {"lstm_z": [], "last_z": []}
    for _ in range(N_BOOT):
        idx = np.concatenate([rows[u] for u in rng.choice(users, len(users))])
        yb = ya[idx]
        if yb.min() == yb.max():
            continue
        ag = roc_auc_score(yb, a.y_prob_peak.values[idx])
        for c in diffs:
            diffs[c].append(ag - roc_auc_score(yb, a[c].values[idx]))

    res = {c: all_k(wk, v[c].values == 1, peak, groups) for c in ["flag_gbm", "flag_lstm", "flag_last"]}
    shuf = np.zeros((n_perm, len(METRICS), len(KS)))
    fg = v.flag_gbm.values
    for i in range(n_perm):
        fs = fg.copy()
        for idx in groups:
            fs[idx] = rng.permutation(fs[idx])
        shuf[i] = all_k(wk, fs == 1, peak, groups)
    pval = (np.sum(shuf >= res["flag_gbm"][None], axis=0) + 1) / (n_perm + 1)

    print(f"\n== {scheme}  rows {len(v)} (dropped without is_peak: {n0 - len(v)})  users {v.uid.nunique()}  "
          f"peak prevalence {peak.mean():.3f}  flag rate {rate:.3f}  "
          f"consecutive rows more than 1 week apart: {(gap > 1).mean():.3f}")
    print("AUC  GBM %.3f   LSTM %.3f   Last EMA %.3f   (rows for AUC %d)"
          % (auc["y_prob_peak"], auc["lstm_z"], auc["last_z"], len(a)))
    for c, label in [("lstm_z", "LSTM forecast"), ("last_z", "last EMA")]:
        dd = np.array(diffs[c])
        print(f"AUC GBM - {label}: {auc['y_prob_peak'] - auc[c]:+.3f}  95% CI "
              f"[{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]")
    for mi, met in enumerate(METRICS):
        print(f"\n  {met}")
        print("  %-22s" % "alert" + "".join(" %8s" % f"k={k}" for k in KS))
        for fc, label in [("flag_gbm", "GBM (system)"), ("flag_lstm", "LSTM forecast"), ("flag_last", "Last EMA")]:
            print("  %-22s" % label + "".join(" %8.3f" % x for x in res[fc][mi]))
        print("  %-22s" % "Random (mean)" + "".join(" %8.3f" % x for x in shuf[:, mi].mean(0)))
        print("  %-22s" % "p GBM vs random" + "".join(" %8.3f" % x for x in pval[mi]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", required=True)
    ap.add_argument("--luo", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for scheme, path in [("WF", args.wf), ("LUO", args.luo)]:
        analyse(load_predictions(path, scheme, args.npz, with_age=True), scheme, args.n_perm, rng)


if __name__ == "__main__":
    main()
