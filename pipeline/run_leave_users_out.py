#!/usr/bin/env python3
"""
Leave-Users-Out (LUO) evaluation.

Users are split into folds. For each fold the population LSTM, the peak classifier
and the calibrations (beta, adaptive-threshold parameters) use only the training users'
rows whose target precedes the cutoff; the test users are simulated as new users from
the cutoff onwards, starting from the population model.

Stress (RQ1-RQ2, default options):
  PYTHONHASHSEED=0 python pipeline/run_leave_users_out.py --data ces_stress_forecast.npz --out results/luo_seed42

PHQ-4, comparison with Gala et al. (RQ3): five folds over the restricted population, no
temporal cutoff, PHQ-4 range; --self-reports weekly gives the system all weekly responses
for the personal statistics and the classifier's self-report features (fine-tuning still
follows the policy):
  PYTHONHASHSEED=0 python pipeline/run_leave_users_out.py --data ces_phq4_forecast.npz \
      --target phq4 --folds 5 --no-cutoff --users comparisons/gala/phq4_users.txt \
      --self-reports weekly --all-rows --out results/phq4_weekly_seed42
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from correction import (ALPHA, DECAY, THR_MAX, THR_MIN, AdaptiveFlag, calibrate_flag_params,
                        corrected_forecast)
from dataset import load_forecast_dataset
from lstm_model import finetune_user, predict_one, train_global
from peak_classifier import (PEAK_K, PeakClassifierData, fit_peak_classifier,
                             peak_probabilities, row_key)
from policy import ONBOARDING, recent_stats, request_mask

MIN_ROWS = 20                 # users with fewer rows are excluded
CUTOFF = "2019-11-08"
LSTM_PARAMS = dict(hidden_size=64, dropout=0.2, lr=5e-4, max_epochs=100, batch_size=256, patience=15)
TARGETS = {   # output range and Huber delta (same share of the range)
    "stress": {"y_range": (1.0, 5.0), "huber_delta": 1.0},
    "phq4": {"y_range": (0.0, 12.0), "huber_delta": 3.0},
}


def simulate_user(uid, X, y, weeks, anchors, history, global_model, col_med, device,
                  probs, peak_size, params, fold, target, weekly):
    """history: the user's self-reports before the test window, used only for the
    evaluation labels (is_peak) and never by the system."""
    requested = request_mask(weeks)
    flag = AdaptiveFlag(params["decay"], params["thr_min"], params["thr_max"])
    observed = list(history)
    collected = []
    user_model = global_model
    results = []
    for t in range(len(y)):
        y_true = float(y[t])
        pm, ps = recent_stats(observed)
        pm_s, ps_s = recent_stats(collected)
        pm_c, ps_c = (pm, ps) if weekly else (pm_s, ps_s)
        y_lstm = predict_one(user_model, X[t], col_med, device, target["y_range"])
        p_peak = probs.get(row_key(uid, anchors[t]), 0.0)
        fv, active = flag.update(p_peak)
        y_full = corrected_forecast(y_lstm, fv, active, pm_c, ps_c, peak_size, params["alpha"], target["y_range"])
        is_peak = int((y_true - pm) / ps >= PEAK_K) if not np.isnan(ps) and ps > 0.01 else np.nan

        results.append({
            "uid": uid, "week_start": pd.Timestamp(str(weeks[t])), "fold": fold,
            "anchor_date": str(pd.Timestamp(str(anchors[t])).date()),
            "ema_requested": int(requested[t]),
            "y_true": y_true, "y_pred_lstm": y_lstm, "y_pred_A_peak_only": y_full,
            "y_prob_peak": p_peak, "flag_peak": fv, "flag_peak_active": int(active), "is_peak": is_peak,
            "personal_mean": pm, "personal_std": ps,
            "personal_mean_sparse": pm_s, "personal_std_sparse": ps_s,
        })

        observed.append(y_true)
        if requested[t]:
            collected.append(y_true)
            user_model = finetune_user(user_model, X[t], y[t], col_med, device,
                                       huber_delta=target["huber_delta"])
    return results


def restrict_users(data, path):
    keep = set(open(path).read().split())
    m = np.isin(data.uids.astype(str), list(keep))
    for name in ["X", "y", "weeks", "uids", "anchors", "targets", "gbm_X"]:
        setattr(data, name, getattr(data, name)[m])
    print(f"users in the list {len(keep)} | found {len(set(data.uids.astype(str)))} | rows {int(m.sum()):,}")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help=".npz from build_forecast_dataset.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--target", default="stress", choices=list(TARGETS))
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--cutoff", default=CUTOFF)
    ap.add_argument("--no-cutoff", action="store_true", help="classic LUO: all weeks for training and test")
    ap.add_argument("--users", default=None, help="file with the user ids to keep")
    ap.add_argument("--self-reports", default="sparse", choices=["sparse", "weekly"])
    ap.add_argument("--all-rows", action="store_true", help="also save the rows without evaluation label")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = TARGETS[args.target]
    weekly = args.self_reports == "weekly"
    cutoff = None if args.no_cutoff else pd.Timestamp(args.cutoff)

    data = load_forecast_dataset(args.data)
    if args.users:
        data = restrict_users(data, args.users)
    peak_data = PeakClassifierData(data, luo_cutoff=cutoff)
    counts = {u: (data.uids == u).sum() for u in np.unique(data.uids)}
    valid_uids = np.array([u for u, n in counts.items() if n >= MIN_ROWS])
    splits = list(KFold(n_splits=args.folds, shuffle=True, random_state=42).split(valid_uids))
    print(f"rows {len(data.y):,} | eligible users {len(valid_uids)} | target {args.target} | "
          f"cutoff {cutoff.date() if cutoff is not None else 'none'} | self-reports {args.self_reports}")

    all_results, fold_params = [], []
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_uids, test_order = set(valid_uids[train_idx]), valid_uids[test_idx]
        test_uids = set(test_order)

        train_mask = np.isin(data.uids, list(train_uids))
        test_mask = np.isin(data.uids, list(test_uids))
        if cutoff is not None:
            cutoff_np = np.datetime64(cutoff)
            train_mask &= (data.weeks < cutoff_np) & (data.targets < cutoff_np)
            test_mask &= data.weeks >= cutoff_np
        X_tr, y_tr = data.X[train_mask], data.y[train_mask]
        X_tr, y_tr = X_tr[~np.isnan(y_tr)], y_tr[~np.isnan(y_tr)]
        global_model, col_med, val_loss = train_global(X_tr, y_tr, data.X.shape[2], args.device,
                                                       huber_delta=target["huber_delta"], **LSTM_PARAMS)

        frame = peak_data.feature_frame(set() if weekly else test_uids)
        model, cols, med = fit_peak_classifier(frame, train_uids, cutoff)
        probs = peak_probabilities(model, frame, cols, med)
        peak_size = peak_data.peak_size(train_uids, cutoff)
        best = calibrate_flag_params(peak_data.calibration_users(probs, train_uids, cutoff, MIN_ROWS),
                                     peak_size, target["y_range"])
        params = best or {"decay": DECAY, "thr_min": THR_MIN, "thr_max": THR_MAX, "alpha": ALPHA}
        fold_params.append({"fold": fold, "decay": params["decay"], "thr_min": params["thr_min"],
                            "thr_max": params["thr_max"], "alpha": params["alpha"], "tgt_std_peak": peak_size})

        fold_results = []
        for uid in test_order:
            m = test_mask & (data.uids == uid) & ~np.isnan(data.y)
            order = np.argsort(data.weeks[m])
            X_u, y_u = data.X[m][order], data.y[m][order]
            w_u, a_u = data.weeks[m][order], data.anchors[m][order]
            if len(y_u) < ONBOARDING + 2:
                continue
            m_pre = (data.uids == uid) & (data.weeks < w_u[0])
            history = data.y[m_pre][np.argsort(data.weeks[m_pre], kind="stable")].astype(float)
            history = history[~np.isnan(history)]
            fold_results.extend(simulate_user(uid, X_u, y_u, w_u, a_u, history, global_model, col_med,
                                              args.device, probs, peak_size, params, fold, target, weekly))
        all_results.extend(fold_results)
        if fold_results:
            fd = pd.DataFrame(fold_results)
            print(f"fold {fold + 1}/{args.folds} | params {params['decay']}/{params['thr_min']}/"
                  f"{params['thr_max']}/{params['alpha']} | test rows {len(fd)} | "
                  f"MAE LSTM {np.abs(fd.y_true - fd.y_pred_lstm).mean():.4f} | "
                  f"MAE system {np.abs(fd.y_true - fd.y_pred_A_peak_only).mean():.4f}")

    pd.DataFrame(fold_params).to_csv(out_dir / "fold_params.csv", index=False)
    out = pd.DataFrame(all_results)
    if args.all_rows:
        out.to_csv(out_dir / "predictions_luo_all.csv", index=False)
    out = out[out["is_peak"].notna()]   # weeks without an evaluation label are not reported
    out.to_csv(out_dir / "predictions_luo.csv", index=False)
    print(f"saved {out_dir / 'predictions_luo.csv'} | {len(out):,} rows | {out['uid'].nunique()} users")


if __name__ == "__main__":
    main()
