#!/usr/bin/env python3
"""
PHQ-4 severity classification (RQ3, comparison with Gala et al.).

A HistGradientBoosting classifier maps the system's outputs to the four PHQ-4 severity
bands (0-3 normal, 4-6 mild, 7-9 moderate, 10-12 severe). Features: corrected forecast,
peak probability, peak flag, personal mean and std, and the forecast's deviation from the
personal mean in personal std units. Leave-one-user-out: each user is classified by a
model trained on all the other users. Class weights 1/sqrt(frequency).

--stats sparse uses the personal statistics of the collected self-reports only;
--stats dense those of all weekly self-reports (weekly configuration).

Usage:
  python comparisons/gala/severity_stacking.py --predictions results/phq4_weekly_seed42/predictions_luo_all.csv \
      --stats dense --out results/phq4_weekly_seed42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

BAND_EDGES = [-0.5, 3.5, 6.5, 9.5, 12.5]
FEATURES = ["y_pred_A_peak_only", "y_prob_peak", "flag_peak_active",
            "personal_mean", "personal_std", "peak_pred_dev_z"]
HGB_PARAMS = dict(max_iter=200, max_depth=None, min_samples_leaf=20, l2_regularization=0.0,
                  validation_fraction=0.1)


def to_band(values):
    return pd.cut(values, bins=BAND_EDGES, labels=[0, 1, 2, 3]).astype(float)


def sqrt_balanced_weights(y):
    counts = pd.Series(y).value_counts()
    raw = pd.Series(y).map(lambda c: 1.0 / np.sqrt(counts[c])).values
    return raw / raw.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="predictions_luo_all.csv of the PHQ-4 run")
    ap.add_argument("--stats", required=True, choices=["sparse", "dense"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    if args.stats == "sparse":
        df["personal_mean"] = df["personal_mean_sparse"]
        df["personal_std"] = df["personal_std_sparse"]
    ps = df["personal_std"].where(df["personal_std"] > 0.01)
    df["peak_pred_dev_z"] = np.clip((df["y_pred_A_peak_only"] - df["personal_mean"]) / ps, -8, 8)
    df = df.dropna(subset=["y_true"]).copy()
    df["y_true_class"] = to_band(df["y_true"])
    df = df.dropna(subset=["y_true_class"])
    df["y_true_class"] = df["y_true_class"].astype(int)

    results = []
    for uid in df["uid"].unique():
        train, test = df[df["uid"] != uid], df[df["uid"] == uid].copy()
        if train["y_true_class"].nunique() < 2:
            continue
        y_train = train["y_true_class"].values
        clf = HistGradientBoostingClassifier(random_state=42, class_weight=None, **HGB_PARAMS)
        clf.fit(train[FEATURES].values.astype(float), y_train, sample_weight=sqrt_balanced_weights(y_train))
        test["y_pred_class_stack"] = clf.predict(test[FEATURES].values.astype(float))
        results.append(test)

    out = pd.concat(results, ignore_index=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.out) / "predictions_severity.csv", index=False)
    print(f"saved {Path(args.out) / 'predictions_severity.csv'} | {len(out):,} rows | {out['uid'].nunique()} users")


if __name__ == "__main__":
    main()
