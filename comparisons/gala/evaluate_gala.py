#!/usr/bin/env python3
"""
RQ3: PHQ-4 severity classification evaluated with the protocol of Gala et al.

Evaluated weeks: those with at least four previous responses of the same user (position
of the row in the user's full series in the dataset), since Gala et al.'s model uses the
four previous weekly observations.
Metrics: accuracy within each user, averaged over the users of each fold and then over
folds (Gala et al.'s accuracy); pooled accuracy, per-class F1 and macro-F1; 95% CI of the
severe-class F1 by bootstrap over users (2,000 resamples).

Usage:
  python comparisons/gala/evaluate_gala.py --npz ces_phq4_forecast.npz \
      --run sparse=results/phq4_sparse_seed42/predictions_severity.csv \
      --run weekly=results/phq4_weekly_seed42/predictions_severity.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

MIN_PREV = 4
N_BOOT = 2000
GALA = ("Gala et al. (XGBoost)", 0.777, None, 0.884, 0.638, 0.425, 0.434)   # values reported in their paper


def positions(npz):
    """Position of each row in the user's full series of the dataset."""
    z = np.load(npz, allow_pickle=True)
    r = pd.DataFrame({"uid": z["uids"].astype(str), "anchor": pd.to_datetime(z["anchor_dates"]),
                      "y": z["y"].astype(float)})
    r = r[~np.isnan(r["y"])].sort_values(["uid", "anchor"], kind="stable")
    r["pos"] = r.groupby("uid").cumcount()
    return r.set_index(["uid", "anchor"])["pos"]


def evaluate(d, pos):
    key = pd.MultiIndex.from_arrays([d["uid"].astype(str), pd.to_datetime(d["anchor_date"])])
    d = d.assign(n_prev_rows=pos.reindex(key).values)
    if d["n_prev_rows"].isna().any():
        raise SystemExit("rows not found in the dataset: wrong .npz?")
    d = d.sort_values(["uid", "anchor_date"], kind="stable").copy()
    s = d[d["n_prev_rows"] >= MIN_PREV]
    yt, yp = s["y_true_class"].astype(int), s["y_pred_class_stack"].astype(int)
    f = f1_score(yt, yp, labels=[0, 1, 2, 3], average=None, zero_division=0)
    per_fold = [g.groupby("uid").apply(lambda u: (u["y_true_class"] == u["y_pred_class_stack"]).mean()).mean()
                for _, g in s.groupby("fold")]
    rng = np.random.default_rng(0)
    groups = list(s.groupby("uid").indices.values())
    ya, pa = yt.values, yp.values
    boot = []
    for _ in range(N_BOOT):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boot.append(f1_score(ya[ix], pa[ix], labels=[3], average="macro", zero_division=0))
    return dict(boot=boot, n=len(s), users=s["uid"].nunique(), acc_gala=float(np.mean(per_fold)),
                acc_pool=accuracy_score(yt, yp), f1=f, mf1=f.mean(),
                dist=np.bincount(yt, minlength=4) / len(yt))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="name=predictions_severity.csv")
    ap.add_argument("--npz", required=True, help="PHQ-4 dataset (positions in the users' series)")
    args = ap.parse_args()
    pos = positions(args.npz)
    print(f"Weeks with at least {MIN_PREV} previous responses of the same user\n")
    print(f"{'configuration':<24}{'rows':>7}{'users':>7}{'Acc':>7}{'AccPool':>8}"
          f"{'Normal':>8}{'Mild':>7}{'Mod':>7}{'Sev':>7}{'MF1':>7}   class distribution")
    for r in args.run:
        name, path = r.split("=", 1)
        m = evaluate(pd.read_csv(path), pos)
        f = m["f1"]
        print(f"{name:<24}{m['n']:>7,}{m['users']:>7}{m['acc_gala']:>7.3f}{m['acc_pool']:>8.3f}"
              f"{f[0]:>8.3f}{f[1]:>7.3f}{f[2]:>7.3f}{f[3]:>7.3f}{m['mf1']:>7.3f}   "
              + " ".join(f"{x:.1%}" for x in m["dist"]))
        lo, hi = np.percentile(m["boot"], [2.5, 97.5])
        print(f"{'':<24}  severe-class F1 95% CI, bootstrap over users [{lo:.3f}, {hi:.3f}]  (Gala 0.434)")
    g = GALA
    print(f"{g[0]:<24}{'':>7}{'121':>7}{g[1]:>7.3f}{'':>8}{g[3]:>8.3f}{g[4]:>7.3f}{g[5]:>7.3f}{g[6]:>7.3f}"
          f"{np.mean(g[3:]):>7.3f}   62.9% 26.0% 7.2% 3.9%")


if __name__ == "__main__":
    main()
