#!/usr/bin/env python3
"""
Walk-Forward (WF) evaluation: expanding training window, one fold every 16 weeks.

At each fold cutoff the population LSTM and the peak classifier are retrained on the
rows whose target precedes the cutoff. Each user active in the test window is then
personalized by replaying, in order, the fine-tuning steps on the self-reports the
policy collected before the cutoff, and simulated week by week until the next cutoff.

Usage (PYTHONHASHSEED=0 is required: test users are iterated in a Python set):
  PYTHONHASHSEED=0 python pipeline/run_walk_forward.py --data ces_stress_forecast.npz --out results/wf_seed42
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from correction import ALPHA, DECAY, THR_MAX, THR_MIN, AdaptiveFlag, corrected_forecast
from dataset import load_forecast_dataset
from lstm_model import finetune_user, predict_one, train_global
from peak_classifier import (PEAK_K, PeakClassifierData, fit_peak_classifier,
                             peak_probabilities, row_key)
from policy import STATS_MIN, STATS_WINDOW, recent_stats, request_mask

STEP_WEEKS = 16
MIN_TRAIN_WEEKS = 50
LSTM_PARAMS = dict(hidden_size=64, dropout=0.2, lr=5e-4, max_epochs=100, batch_size=256, patience=15)
SHRINKAGE_K = 5.0   # weight of the user's own peaks in beta: n / (n + 5)


def user_rows(uid, data):
    """The user's rows sorted by anchor day."""
    m = (data.uids == uid) & ~np.isnan(data.y)
    order = np.argsort(data.anchors[m], kind="stable")
    return (data.X[m][order], data.y[m][order], data.weeks[m][order],
            data.anchors[m][order], data.targets[m][order])


def personal_peak_deviations(y):
    devs = []
    for i in range(STATS_WINDOW, len(y)):
        h = y[max(0, i - STATS_WINDOW):i]
        h = h[~np.isnan(h)]
        if len(h) < STATS_MIN:
            continue
        mu, sig = np.mean(h), np.std(h)
        if sig < 0.01:
            continue
        dev = (y[i] - mu) / sig
        if dev >= PEAK_K:
            devs.append(dev)
    return devs


def simulate_user(uid, data, global_model, col_med, device, cutoff, test_end, probs,
                  population_peak_size, flag, fold):
    Xu, yu, wu, au, tu = user_rows(uid, data)
    before = wu < np.datetime64(cutoff)
    in_test = (wu >= np.datetime64(cutoff)) & (wu < np.datetime64(test_end))
    n_before, n_test = int(before.sum()), int(in_test.sum())
    if n_test == 0:
        return [], flag

    requested = request_mask(wu)
    req_before, req_test = requested[:n_before], requested[n_before:n_before + n_test]
    y_before = yu[before]
    y_before_policy = y_before[req_before].astype(float)

    # beta: population value shrunk towards the user's own peaks among the policy self-reports
    devs = personal_peak_deviations(y_before_policy)
    if devs:
        weight = len(devs) / (len(devs) + SHRINKAGE_K)
        peak_size = weight * float(np.mean(devs)) + (1 - weight) * population_peak_size
    else:
        weight, peak_size = 0.0, population_peak_size

    user_model = copy.deepcopy(global_model)
    user_model.to(device)
    Xb = Xu[before]
    for i in range(n_before):
        if req_before[i]:
            user_model = finetune_user(user_model, Xb[i], y_before[i], col_med, device, inplace=True)

    if flag is None:
        flag = AdaptiveFlag(DECAY, THR_MIN, THR_MAX)
    observed = list(y_before.astype(float))           # all self-reports: evaluation labels only
    collected = list(y_before_policy.astype(float))   # self-reports available to the system

    Xt, yt, wt, at, tt = Xu[in_test], yu[in_test], wu[in_test], au[in_test], tu[in_test]
    results = []
    for j in range(n_test):
        y_true = float(yt[j])
        pm, ps = recent_stats(observed)
        pm_s, ps_s = recent_stats(collected)
        y_lstm = predict_one(user_model, Xt[j], col_med, device)
        p_peak = probs.get(row_key(uid, at[j]), 0.0)
        fv, active = flag.update(p_peak)
        y_full = corrected_forecast(y_lstm, fv, active, pm_s, ps_s, peak_size, ALPHA)
        is_peak = int((y_true - pm) / ps >= PEAK_K) if not np.isnan(ps) and ps > 0.01 else np.nan

        results.append({
            "uid": uid, "week_start": pd.Timestamp(str(wt[j])), "fold": fold,
            "anchor_date": str(pd.Timestamp(at[j]).date()), "target_date": str(pd.Timestamp(tt[j]).date()),
            "t_rel_in_test": j, "abs_ema_index": n_before + j, "n_ema_history_pre_cutoff": n_before,
            "ema_requested": int(req_test[j]),
            "y_true": y_true, "y_pred_lstm": y_lstm, "y_pred_A_peak_only": y_full,
            "y_prob_peak": p_peak, "flag_peak": fv, "flag_peak_active": int(active), "is_peak": is_peak,
            "personal_mean": pm, "personal_std": ps,
            "personal_mean_sparse": pm_s, "personal_std_sparse": ps_s,
            "tgt_std_peak_used": peak_size, "n_personal_peak_pre_cutoff": len(devs),
            "shrinkage_weight_peak": weight,
        })

        if req_test[j]:
            user_model = finetune_user(user_model, Xt[j], y_true, col_med, device, inplace=True)
            collected.append(y_true)
        observed.append(y_true)
    return results, flag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help=".npz from build_forecast_dataset.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_forecast_dataset(args.data)
    peak_data = PeakClassifierData(data)
    all_weeks = np.sort(np.unique(data.weeks))
    fold_starts = list(range(MIN_TRAIN_WEEKS, len(all_weeks), STEP_WEEKS))
    print(f"rows {len(data.y):,} | users {len(np.unique(data.uids))} | folds {len(fold_starts)}")

    flags, all_results = {}, []
    for fold, start in enumerate(fold_starts):
        cutoff = pd.Timestamp(all_weeks[start])
        end = min(start + STEP_WEEKS, len(all_weeks))
        test_end = pd.Timestamp(all_weeks[end]) if end < len(all_weeks) else pd.Timestamp(all_weeks[-1]) + pd.Timedelta(weeks=1)

        train_mask = (data.weeks < np.datetime64(cutoff)) & (data.targets < np.datetime64(cutoff))
        test_mask = (data.weeks >= np.datetime64(cutoff)) & (data.weeks < np.datetime64(test_end))
        train_uids = set(np.unique(data.uids[train_mask]))
        test_uids = set(np.unique(data.uids[test_mask]))

        X_tr, y_tr = data.X[train_mask], data.y[train_mask]
        X_tr, y_tr = X_tr[~np.isnan(y_tr)], y_tr[~np.isnan(y_tr)]
        if len(X_tr) < 50:
            continue
        global_model, col_med, val_loss = train_global(X_tr, y_tr, data.X.shape[2], args.device, **LSTM_PARAMS)

        frame = peak_data.feature_frame(test_uids)
        model, cols, med = fit_peak_classifier(frame, train_uids, cutoff)
        probs = peak_probabilities(model, frame, cols, med)
        population_peak_size = peak_data.peak_size(train_uids, cutoff)

        fold_results = []
        for uid in test_uids:
            res, flags[uid] = simulate_user(uid, data, global_model, col_med, args.device, cutoff, test_end,
                                            probs, population_peak_size, flags.get(uid), fold)
            fold_results.extend(res)
        all_results.extend(fold_results)
        if not fold_results:
            continue
        fd = pd.DataFrame(fold_results)
        print(f"fold {fold + 1}/{len(fold_starts)} cutoff {cutoff.date()} | test rows {len(fd)} | "
              f"MAE LSTM {np.abs(fd.y_true - fd.y_pred_lstm).mean():.4f} | "
              f"MAE system {np.abs(fd.y_true - fd.y_pred_A_peak_only).mean():.4f}")

    out = pd.DataFrame(all_results)
    out.to_csv(out_dir / "predictions_wf.csv", index=False)
    print(f"saved {out_dir / 'predictions_wf.csv'} | {len(out):,} rows | {out['uid'].nunique()} users")


if __name__ == "__main__":
    main()
