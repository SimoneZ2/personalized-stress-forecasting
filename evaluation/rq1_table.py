#!/usr/bin/env python3
"""Table 1 (RQ1): next-week stress forecasting in WF and LUO.

Usage: python evaluation/rq1_table.py --wf predictions_wf.csv --luo predictions_luo.csv --npz ces_stress_forecast.npz
"""

import argparse
import warnings

from common import ccc, load_predictions, per_user_rho

warnings.filterwarnings("ignore")
PREDICTORS = [("Personal mean (sparse)", "pm_base"), ("Last EMA (sparse)", "last_base"),
              ("Personalized LSTM", "y_pred_lstm"), ("LSTM+GBM (full system)", "y_pred_A_peak_only")]


def table1(d, scheme):
    pk = d["is_peak"] == 1
    print(f"\nTABLE 1 — {scheme}: {len(d):,} predictions, {d['uid'].nunique()} users, "
          f"{int(pk.sum()):,} peak weeks")
    print(f"  {'Predictor':<34}{'MAE':>7}{'MAE_pk':>8}{'rho_pool':>9}{'CCC':>7}{'rho_pu':>8}")
    for label, col in PREDICTORS:
        e = (d["y_true"] - d[col]).abs()
        print(f"  {label:<34}{e.mean():>7.3f}{e[pk].mean():>8.3f}{d['y_true'].corr(d[col]):>9.3f}"
              f"{ccc(d['y_true'], d[col]):>7.3f}{per_user_rho(d, col):>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", required=True)
    ap.add_argument("--luo", required=True)
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()
    for scheme, path in [("WF", args.wf), ("LUO", args.luo)]:
        table1(load_predictions(path, scheme, args.npz), scheme)


if __name__ == "__main__":
    main()
