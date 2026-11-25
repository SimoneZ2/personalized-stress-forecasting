#!/usr/bin/env python3
"""RQ3: comparison with I-HOPE on next-week stress (WF).

I-HOPE estimates the stress of week W from the sensing of that week's self-report day
(nowcasting); its row for week W is matched to our row whose target falls in week W
(forecast issued at the anchor of the previous week). Evaluated rows: rows of Table 1
present in both systems with identical ground truth.

Usage: python evaluation/rq3_ihope.py --ihope ihope_predictions.csv --wf predictions_wf.csv --npz ces_stress_forecast.npz
"""

import argparse
import os

import numpy as np
import pandas as pd

from common import ccc, load_predictions, per_user_rho
from rq1_tests import paired

SYSTEM = "y_pred_A_peak_only"
PREDICTORS = [("Personal mean (sparse)", "pm_base"), ("Last EMA (sparse)", "last_base"),
              ("Personalized LSTM", "y_pred_lstm"), ("LSTM+GBM (full system)", SYSTEM),
              ("I-HOPE", "y_pred_ihope")]
TESTS = [("MAE", "LSTM - I-HOPE", "y_pred_lstm", "y_pred_ihope", False, 1),
         ("MAE", "System - I-HOPE", SYSTEM, "y_pred_ihope", False, 1),
         ("MAE_peak", "LSTM - I-HOPE", "y_pred_lstm", "y_pred_ihope", True, 3),
         ("MAE_peak", "System - I-HOPE", SYSTEM, "y_pred_ihope", True, 3)]


def monday(s):
    s = pd.to_datetime(s).dt.normalize()
    return s - pd.to_timedelta(s.dt.weekday, unit="D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ihope", required=True)
    ap.add_argument("--wf", required=True)
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()

    d = load_predictions(args.wf, "WF", args.npz)
    d["tw"] = monday(d["target_date"])
    print(f"target in the week after the anchor: {(d['tw'] == d['week_start'] + pd.Timedelta(days=7)).mean():.1%} of the rows")

    ih = pd.read_csv(os.path.expanduser(args.ihope))
    ih["uid"] = ih["uid"].astype(str)
    ih["tw"] = monday(ih["week_start"])
    print(f"I-HOPE: {len(ih):,} rows, {ih['uid'].nunique()} users, duplicated (uid, week): {ih.duplicated(['uid', 'tw']).sum()}")
    print(f"I-HOPE: week_start always a Monday: {(pd.to_datetime(ih['week_start']).dt.weekday == 0).all()}")
    ih = ih.drop_duplicates(["uid", "tw"])[["uid", "tw", "y_true", "y_pred"]] \
           .rename(columns={"y_true": "y_true_ih", "y_pred": "y_pred_ihope"})

    print(f"our rows (Table 1, WF): {len(d):,}, {d['uid'].nunique()} users")
    m = d.merge(ih, on=["uid", "tw"], how="inner")
    same = np.isclose(m["y_true"], m["y_true_ih"], atol=1e-6)
    print(f"matched on (uid, target week): {len(m):,} rows, {m['uid'].nunique()} users")
    print(f"  identical y_true: {same.sum():,} ({same.mean():.1%})")
    v = m[same].copy()
    pk = v["is_peak"] == 1
    print(f"\nEVALUATED ROWS: {len(v):,} predictions, {v['uid'].nunique()} users, {int(pk.sum()):,} peak weeks")

    print(f"  {'Predictor':<28}{'MAE':>7}{'MAE_pk':>8}{'rho_pool':>9}{'CCC':>7}{'rho_pu':>8}")
    for label, col in PREDICTORS:
        e = (v["y_true"] - v[col]).abs()
        print(f"  {label:<28}{e.mean():>7.3f}{e[pk].mean():>8.3f}{v['y_true'].corr(v[col]):>9.3f}"
              f"{ccc(v['y_true'], v[col]):>7.3f}{per_user_rho(v, col):>8.3f}")

    print("\nTESTS (A - B, difference of absolute error; negative = A better)")
    rng = np.random.default_rng(0)
    for metric, label, a, b, peak_only, min_n in TESTS:
        diff, lo, hi, p, share, n = paired(v, a, b, peak_only, min_n, rng)
        print(f"  {metric:<9}{label:<18}{diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  Wilcoxon p={p:.2g}"
              f"  A better for {share:.0%} of {n} users")


if __name__ == "__main__":
    main()
