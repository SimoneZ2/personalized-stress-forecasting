#!/usr/bin/env python3
"""Mean ± standard deviation across seeds of the Table 1 entries and of the alert AUC
and row-based forward-tolerance F1 (used for the seed-stability sentence).

The naive baselines do not depend on the seed, so their standard deviation is ~0.

Usage:
  python evaluation/seed_stability.py --npz ces_stress_forecast.npz --seeds 42 43 44 \
      --wf-pattern results/wf_seed{s}/predictions_wf.csv \
      --luo-pattern results/luo_seed{s}/predictions_luo.csv [--out seeds.csv]
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ccc, load_predictions, per_user_rho
from rq1_table import PREDICTORS

KS = [0, 1, 2, 4]


def table1_rows(d):
    pk = d["is_peak"] == 1
    out = []
    for label, col in PREDICTORS:
        e = (d["y_true"] - d[col]).abs()
        out.append({"predictor": label, "MAE": e.mean(), "MAE_peak": e[pk].mean(),
                    "rho_pooled": d["y_true"].corr(d[col]), "CCC": ccc(d["y_true"], d[col]),
                    "rho_per_user": per_user_rho(d, col)})
    return out


def _groups(d, flags):
    out = []
    for _, idx in d.groupby("uid", sort=False).indices.items():
        yt = d["is_peak"].values[idx]
        yt = np.where(np.isnan(yt), -1, yt).astype(int)
        out.append((yt, flags[idx].astype(int)))
    return out


def _f1_forward(groups, k):
    """Forward tolerance counted in rows: alert in t correct if a peak in t..t+k;
    peak in t missed if no alert in t-k..t."""
    tp = fp = fn = 0
    for yt, yp in groups:
        T = len(yt)
        if k == 0:
            v = yt != -1
            tp += int(((yt == 1) & (yp == 1) & v).sum())
            fp += int(((yt == 0) & (yp == 1)).sum())
            fn += int(((yt == 1) & (yp == 0)).sum())
            continue
        ar = np.arange(T)
        c = np.concatenate([[0], np.cumsum(yt == 1)])
        anyp = (c[np.minimum(ar + k + 1, T)] - c[ar]) > 0
        f = yp == 1
        tp += int((f & anyp).sum())
        fp += int((f & ~anyp).sum())
        cf = np.concatenate([[0], np.cumsum(yp)])
        anyf = (cf[ar + 1] - cf[np.maximum(ar - k, 0)]) > 0
        fn += int(((yt == 1) & ~anyf).sum())
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def _top_share(score, share, rng):
    s = np.asarray(score, float)
    order = np.lexsort((rng.random(len(s)), -s))
    flags = np.zeros(len(s), dtype=int)
    flags[order[:int(round(share * len(s)))]] = 1
    return flags


def alert_rows(d, n_perm, seed):
    d = d.sort_values(["uid", "week_start"], kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    share = d["flag_peak_active"].mean()
    y = d["is_peak"].astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_l = np.where(d["ps_base"] > 0.01, (d["y_pred_lstm"] - d["pm_base"]) / d["ps_base"], np.nan)
        z_e = np.where(d["ps_base"] > 0.01, (d["last_base"] - d["pm_base"]) / d["ps_base"], np.nan)
    z_l, z_e = np.nan_to_num(z_l, nan=0.0), np.nan_to_num(z_e, nan=0.0)
    rows = [("Peak classifier (full system)", d["y_prob_peak"].values, d["flag_peak_active"].values),
            ("LSTM forecast", z_l, _top_share(z_l, share, rng)),
            ("Last EMA", z_e, _top_share(z_e, share, rng))]
    out = []
    for label, score, flags in rows:
        g = _groups(d, flags)
        r = {"alert": label, "AUC": roc_auc_score(y, score)}
        r.update({f"F1_k{k}": _f1_forward(g, k) for k in KS})
        out.append(r)
    base = d["flag_peak_active"].values
    idx = list(d.groupby("uid", sort=False).indices.values())
    acc = np.zeros(len(KS))
    for _ in range(n_perm):
        perm = base.copy()
        for ix in idx:
            perm[ix] = rng.permutation(base[ix])
        g = _groups(d, perm)
        acc += [_f1_forward(g, k) for k in KS]
    r = {"alert": "Random", "AUC": 0.5}
    r.update({f"F1_k{k}": v for k, v in zip(KS, acc / n_perm)})
    out.append(r)
    info = {"rows": len(d), "users": d["uid"].nunique(), "peaks": int(y.sum()),
            "flag_share": share, "prevalence": y.mean()}
    return out, info


def show(df, key, cols, title):
    print(f"\n{title}")
    g = df.groupby(key, sort=False)[cols]
    m, s = g.mean(), g.std(ddof=1)
    print(f"  {key:<36}" + "".join(f"{c:>17}" for c in cols))
    for name in m.index:
        print(f"  {name:<36}" + "".join(f"{m.loc[name, c]:>9.3f} ± {s.loc[name, c]:<5.3f}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--wf-pattern", required=True)
    ap.add_argument("--luo-pattern", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--out", default=None, help="CSV with the values of every seed")
    args = ap.parse_args()

    all1, all2 = [], []
    for scheme, pat in [("WF", args.wf_pattern), ("LUO", args.luo_pattern)]:
        infos = []
        for s in args.seeds:
            d = load_predictions(pat.format(s=s), scheme, args.npz)
            all1 += [{"scheme": scheme, "seed": s, **r} for r in table1_rows(d)]
            rows2, info = alert_rows(d, args.n_perm, seed=s)
            all2 += [{"scheme": scheme, "seed": s, **r} for r in rows2]
            infos.append(info)
        inf = pd.DataFrame(infos)
        print(f"\n===== {scheme}: seeds {args.seeds} | rows {inf['rows'].min():,}-{inf['rows'].max():,}, "
              f"users {inf['users'].min()}-{inf['users'].max()}, peaks {inf['peaks'].min():,}-{inf['peaks'].max():,}, "
              f"flagged share {inf['flag_share'].mean():.1%} ± {inf['flag_share'].std(ddof=1):.1%}, "
              f"prevalence {inf['prevalence'].mean():.1%}")
        show(pd.DataFrame([r for r in all1 if r["scheme"] == scheme]), "predictor",
             ["MAE", "MAE_peak", "rho_pooled", "CCC", "rho_per_user"],
             f"TABLE 1 — {scheme} (mean ± std across seeds)")
        show(pd.DataFrame([r for r in all2 if r["scheme"] == scheme]), "alert",
             ["AUC"] + [f"F1_k{k}" for k in KS], f"TABLE 2 — {scheme} (mean ± std across seeds)")

    if args.out:
        pd.concat([pd.DataFrame(all1).assign(table=1), pd.DataFrame(all2).assign(table=2)]).to_csv(args.out, index=False)
        print(f"\nper-seed values saved in {args.out}")


if __name__ == "__main__":
    main()
