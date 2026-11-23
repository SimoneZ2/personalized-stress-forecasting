#!/usr/bin/env python3
"""Figures of RQ1 and RQ2 (rows of Table 1).

fig_rq1_maturation: LUO, MAE, MAE_peak and CCC per interval of weeks since the start of
  the user's test period, with 95% bootstrap bands over users (500 resamples).
fig_rq2_peak_users: per-user peak-week error of the personalized LSTM (x) and of the full
  system (y), users with at least 3 peak weeks, WF and LUO.

Usage: python evaluation/figures.py --wf predictions_wf.csv --luo predictions_luo.csv --npz ces_stress_forecast.npz --out-dir figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ccc, load_predictions

N_BOOT = 500
EDGES = [1, 8, 16, 32, 64, np.inf]
LABELS = ["1-7", "8-15", "16-31", "32-63", "64+"]
PREDICTORS = [("y_pred_lstm", "Personalized LSTM", "C0", "o"),
              ("y_pred_A_peak_only", "Full system", "C3", "D"),
              ("pm_base", "Personal mean", "C1", "s"),
              ("last_base", "Last EMA", "C2", "^")]
METRICS = ["MAE", "MAE_peak", "CCC"]
MIN_PEAKS = 3


def maturation(v, out):
    rng = np.random.default_rng(0)
    res = {c: {m: [] for m in METRICS} for c, *_ in PREDICTORS}
    n_users = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = v[(v.age >= lo) & (v.age < hi)]
        users = s.uid.unique()
        n_users.append(len(users))
        groups = {u: g for u, g in s.groupby("uid")}
        boot_idx = [rng.choice(users, len(users)) for _ in range(N_BOOT)]
        for c, *_ in PREDICTORS:
            mae = (s.y_true - s[c]).abs().mean()
            cc = ccc(s.y_true, s[c])
            maep = (s.y_true - s[c]).abs().values[s.is_peak.values == 1].mean()
            bm, bp, bc = [], [], []
            for idx in boot_idx:
                b = np.concatenate([np.c_[groups[u].y_true.values, groups[u][c].values,
                                          groups[u].is_peak.values] for u in idx])
                e = np.abs(b[:, 0] - b[:, 1])
                bm.append(e.mean())
                bp.append(e[b[:, 2] == 1].mean())
                bc.append(ccc(b[:, 0], b[:, 1]))
            res[c]["MAE"].append((mae, *np.percentile(bm, [2.5, 97.5])))
            res[c]["MAE_peak"].append((maep, *np.percentile(bp, [2.5, 97.5])))
            res[c]["CCC"].append((cc, *np.percentile(bc, [2.5, 97.5])))

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 10, "font.family": "serif"})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.2, 2.9))
    x = np.arange(len(LABELS))
    for ax, metric in zip(axes, METRICS):
        for c, name, col, mk in PREDICTORS:
            arr = np.array(res[c][metric])
            ax.plot(x, arr[:, 0], color=col, marker=mk, ms=4, lw=1.4, label=name)
            ax.fill_between(x, arr[:, 1], arr[:, 2], color=col, alpha=0.12, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS)
        ax.set_xlabel("Weeks since the start of the test period")
        ylab = {"MAE": "MAE", "MAE_peak": r"MAE$_{\mathrm{peak}}$", "CCC": "CCC"}[metric]
        ax.set_ylabel(ylab + (" (higher is better)" if metric == "CCC" else " (lower is better)"))
        ax.grid(alpha=0.3, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(PREDICTORS), frameon=False, fontsize=8)
    fig.suptitle("LUO", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)

    print("interval  users   " + "   ".join(f"{name[:17]:>32}" for _, name, *_ in PREDICTORS))
    for i, label in enumerate(LABELS):
        row = "   ".join(f"MAE {res[c]['MAE'][i][0]:.3f} pk {res[c]['MAE_peak'][i][0]:.3f} "
                         f"CCC {res[c]['CCC'][i][0]:.3f}" for c, *_ in PREDICTORS)
        print(f"{label:>6} {n_users[i]:>7}   {row}")
    print(f"saved {out}.pdf")


def peak_users(data, out):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    per_user, lim = {}, 0.0
    for scheme, v in data.items():
        pk = v[v["is_peak"] == 1]
        g = pk.assign(el=(pk.y_true - pk.y_pred_lstm).abs(), es=(pk.y_true - pk.y_pred_A_peak_only).abs()) \
              .groupby("uid").agg(n=("el", "size"), el=("el", "mean"), es=("es", "mean"))
        per_user[scheme] = g[g["n"] >= MIN_PEAKS]
        lim = max(lim, per_user[scheme]["el"].max(), per_user[scheme]["es"].max())
    lim = np.ceil(lim * 10) / 10 + 0.1

    for ax, (scheme, title) in zip(axes, [("WF", "Walk-Forward"), ("LUO", "LUO")]):
        g = per_user[scheme]
        below = (g["es"] < g["el"]).mean()
        ax.scatter(g["el"], g["es"], s=10, alpha=0.6, color="C0", lw=0)
        ax.plot([0, lim], [0, lim], ls="--", color="grey", lw=0.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{title} ({len(g)} users)")
        ax.set_xlabel(r"MAE$_{\mathrm{peak}}$ per user, personalized LSTM")
        ax.text(0.95, 0.05, f"{below:.0%} of users\nbelow the diagonal", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)
        ax.grid(alpha=0.3, lw=0.5)
        print(f"{scheme}: {len(g)} users with >= {MIN_PEAKS} peaks, {below:.1%} below the diagonal")
    axes[0].set_ylabel(r"MAE$_{\mathrm{peak}}$ per user, full system")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"saved {out}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", required=True)
    ap.add_argument("--luo", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"WF": load_predictions(args.wf, "WF", args.npz, with_age=True),
            "LUO": load_predictions(args.luo, "LUO", args.npz, with_age=True)}
    maturation(data["LUO"], out_dir / "fig_rq1_maturation")
    peak_users(data, out_dir / "fig_rq2_peak_users")


if __name__ == "__main__":
    main()
