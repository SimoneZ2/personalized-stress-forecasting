"""Shared loading and metrics for the evaluation scripts.

load_predictions returns the rows of Table 1: weeks with an evaluation label (is_peak)
where both naive baselines are defined.
  - Personal mean (sparse): mean of the last (up to) 12 collected self-reports; in WF
    the value computed by the simulation (personal_mean_sparse).
  - Last EMA (sparse): last collected self-report before the row.
  - Collected self-reports: WF from the simulation (ema_requested); LUO from the
    position of the row among the user's rows from the cutoff (first two, then one
    every four rows).
"""

import os

import numpy as np
import pandas as pd

CUTOFF = "2019-11-08"
LUO_ONBOARDING, LUO_EVERY = 2, 4


def ccc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(), p.var()
    cov = ((y - my) * (p - mp)).mean()
    den = vy + vp + (my - mp) ** 2
    return 2 * cov / den if den > 0 else np.nan


def _luo_requested(d, npz, cutoff):
    z = np.load(os.path.expanduser(npz), allow_pickle=True)
    u = z["uids"].astype(str)
    w = pd.to_datetime(z["weeks"]).values
    y = z["y"].astype(float)
    m = (w >= np.datetime64(pd.Timestamp(cutoff))) & ~np.isnan(y)
    t = pd.DataFrame({"uid": u[m], "week_start": pd.to_datetime(w[m])})
    t = t.sort_values(["uid", "week_start"], kind="stable")
    t["t"] = t.groupby("uid").cumcount()
    t["requested"] = (t["t"] < LUO_ONBOARDING) | ((t["t"] - LUO_ONBOARDING) % LUO_EVERY == 0)
    out = d.merge(t[["uid", "week_start", "requested"]], on=["uid", "week_start"], how="left")
    missing = int(out["requested"].isna().sum())
    if missing:
        print(f"  warning: {missing} LUO rows not found in the dataset")
    out["requested"] = out["requested"].fillna(False).astype(bool)
    return out


def _add_baselines(d):
    d = d.sort_values(["uid", "week_start"], kind="stable").copy()
    pm, ps, last = np.full(len(d), np.nan), np.full(len(d), np.nan), np.full(len(d), np.nan)
    pos = 0
    for _, g in d.groupby("uid", sort=False):
        obs = []
        for yt, req in zip(g["y_true"].values, g["requested"].values):
            if obs:
                pm[pos] = np.mean(obs[-12:])
                last[pos] = obs[-1]
                if len(obs) >= 2:
                    ps[pos] = np.std(obs[-12:])
            if req:
                obs.append(yt)
            pos += 1
    d["pm_base"], d["ps_base"], d["last_base"] = pm, ps, last
    return d


def load_predictions(path, scheme, npz, cutoff=CUTOFF, with_age=False):
    """Rows of Table 1. with_age adds 'age': calendar weeks since the user's first test week."""
    d = pd.read_csv(os.path.expanduser(path))
    d["uid"] = d["uid"].astype(str)
    d["week_start"] = pd.to_datetime(d["week_start"]).dt.normalize()
    first = d.groupby("uid")["week_start"].min()
    if scheme == "WF":
        d["requested"] = d["ema_requested"].astype(bool)
    else:
        d = _luo_requested(d, npz, cutoff)
    d = _add_baselines(d)
    if scheme == "WF":
        d["pm_base"] = d["personal_mean_sparse"]
        d["ps_base"] = d["personal_std_sparse"]
    d = d[d["is_peak"].notna() & d["pm_base"].notna() & d["last_base"].notna()]
    if with_age:
        d = d.copy()
        d["age"] = ((d["week_start"] - d["uid"].map(first)).dt.days // 7).astype(int)
        d = d.sort_values(["uid", "week_start"]).copy()
    return d


def _user_corr(g, col, min_n):
    if len(g) < min_n or g[col].nunique() < 2 or g["y_true"].nunique() < 2:
        return np.nan
    return g["y_true"].corr(g[col], method="pearson")


def per_user_rho(d, col, min_n=5):
    """Mean over users (at least min_n rows) of the Pearson correlation."""
    r = d.groupby("uid").apply(lambda g: _user_corr(g, col, min_n))
    return np.nanmean(r)
