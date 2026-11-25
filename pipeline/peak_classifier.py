"""Peak Classifier: GBM estimating whether next week's self-report is a stress peak.

Each row of the dataset is (user, anchor day D, target self-report of the next week).
Sensing and academic-period features are read from the dataset (gbm_X); the
self-report features depend on which responses are available to the model and are
computed here for every fold:
  - training users: all their responses;
  - test users: only the responses collected by the policy (in LUO, from the cutoff).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from policy import STATS_MIN, STATS_WINDOW, request_mask

PEAK_K = 1.0              # peak: target >= mean + 1 std of the previous self-reports
EMA_MAX_LAG_DAYS = 28     # self-report features use responses up to 28 days old
DEFAULT_PEAK_SIZE = 1.877
META_COLS = ["uid", "week_start", "anchor_date", "target_date", "is_peak_next"]
DAY = np.timedelta64(1, "D")


class PeakClassifierData:
    def __init__(self, data, luo_cutoff=None):
        rows = pd.DataFrame({
            "uid": np.asarray(data.uids).astype(str),
            "anchor_date": pd.to_datetime(data.anchors),
            "target_date": pd.to_datetime(data.targets),
            "week_start": pd.to_datetime(data.weeks),
            "y": np.asarray(data.y, dtype=float),
        })
        if rows["y"].isna().any():
            raise ValueError("dataset rows without target")
        rows = rows.sort_values(["uid", "anchor_date"], kind="stable")
        order = rows.index.values
        rows = rows.reset_index(drop=True)

        gbm_X = np.asarray(data.gbm_X, dtype=float)[order]
        names = list(data.gbm_names)
        sens = [i for i, n in enumerate(names) if n.startswith(("trend_", "point_"))]
        ctx = [i for i, n in enumerate(names) if n.startswith("ctx_")]
        self.sensing = pd.DataFrame(gbm_X[:, sens], columns=[names[i] for i in sens])
        self.context = pd.DataFrame(gbm_X[:, ctx], columns=[names[i] for i in ctx])

        rows["requested"] = False
        for _, g in rows.groupby("uid", sort=False):
            rows.loc[g.index, "requested"] = request_mask(g["week_start"].values)
        self.rows = rows
        self.luo_cutoff = pd.Timestamp(luo_cutoff) if luo_cutoff is not None else None
        self._deviation, self.pm, self.ps = self._peak_deviation()
        self._ema_cache = {}

    def _peak_deviation(self):
        """(target - mean) / std of the user's previous targets (dense: training labels)."""
        n = len(self.rows)
        dev, pm, ps = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for _, g in self.rows.groupby("uid", sort=False):
            yv, idx = g["y"].values, g.index.values
            for i in range(len(yv)):
                h = yv[max(0, i - STATS_WINDOW):i]
                h = h[~np.isnan(h)]
                if len(h) < STATS_MIN:
                    continue
                mu, sd = float(np.mean(h)), float(np.std(h))
                pm[idx[i]], ps[idx[i]] = mu, sd
                if sd > 0.01:
                    dev[idx[i]] = (yv[i] - mu) / sd
        return dev, pm, ps

    def labels(self):
        lab = np.full(len(self.rows), np.nan)
        ok = ~np.isnan(self._deviation)
        lab[ok] = (self._deviation[ok] >= PEAK_K).astype(float)
        return lab

    def _ema_features(self, test_uids):
        """Last available self-report (up to D, at most 28 days old), weeks elapsed (0-4)
        and its z-score against the available responses."""
        key = frozenset(str(u) for u in test_uids)
        if key in self._ema_cache:
            return self._ema_cache[key]
        n = len(self.rows)
        last, since, z = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        max_lag = np.timedelta64(EMA_MAX_LAG_DAYS, "D")
        latest_used = None

        for uid, g in self.rows.groupby("uid", sort=False):
            if uid not in key:
                avail = g
            elif self.luo_cutoff is None:
                avail = g[g["requested"]]
            else:
                post = g[g["week_start"] >= self.luo_cutoff]
                avail = post[request_mask(post["week_start"].values)]
            av_d, av_y = avail["target_date"].values, avail["y"].values
            o = np.argsort(av_d, kind="stable")
            av_d, av_y = av_d[o], av_y[o]
            for ridx, D in zip(g.index.values, g["anchor_date"].values):
                hi = np.searchsorted(av_d, D, side="right")
                if hi == 0:
                    continue
                ld = av_d[hi - 1]
                if D - ld > max_lag:
                    continue
                gap = ld - D
                latest_used = gap if latest_used is None or gap > latest_used else latest_used
                last[ridx] = av_y[hi - 1]
                since[ridx] = min(4, max(0, int(round((D - ld) / (7 * DAY)))))
                h = av_y[max(0, hi - STATS_WINDOW):hi]
                h = h[~np.isnan(h)]
                if len(h) >= STATS_MIN:
                    mu, sd = float(np.mean(h)), float(np.std(h))
                    if sd > 0.01:
                        z[ridx] = float(np.clip((av_y[hi - 1] - mu) / sd, -8, 8))

        if latest_used is not None and latest_used > np.timedelta64(0, "D"):
            raise RuntimeError("self-report used after the anchor day")
        out = pd.DataFrame({"last_ema_stress": last, "weeks_since_ema": since, "ema_stress_z": z})
        self._ema_cache[key] = out
        return out

    def feature_frame(self, test_uids):
        base = self.rows[["uid", "week_start", "anchor_date", "target_date"]].copy()
        base["is_peak_next"] = self.labels()
        return pd.concat([base, self.sensing, self._ema_features(test_uids), self.context], axis=1)

    def peak_size(self, train_uids, cutoff):
        """Mean deviation of peak weeks among training users before the cutoff (beta)."""
        r = self.rows[self.rows["uid"].isin({str(u) for u in train_uids})]
        if cutoff is not None:
            r = r[(r["week_start"] < cutoff) & (r["target_date"] < cutoff)]
        devs = []
        for _, g in r.groupby("uid", sort=False):
            s = g["y"].values
            for i in range(STATS_WINDOW, len(s)):
                h = s[max(0, i - STATS_WINDOW):i]
                h = h[~np.isnan(h)]
                if len(h) < STATS_MIN:
                    continue
                mu, sig = np.mean(h), np.std(h)
                if sig < 0.01:
                    continue
                dev = (s[i] - mu) / sig
                if dev >= PEAK_K:
                    devs.append(dev)
        return float(np.mean(devs)) if len(devs) >= 50 else DEFAULT_PEAK_SIZE

    def calibration_users(self, probs, train_uids, cutoff, min_rows):
        """Per-user sequences (probability, target, personal mean/std) of the training
        users before the cutoff, for the calibration of the adaptive thresholds."""
        r = self.rows.assign(pm=self.pm, ps=self.ps)
        r = r[r["uid"].isin({str(u) for u in train_uids})]
        if cutoff is not None:
            r = r[(r["week_start"] < cutoff) & (r["target_date"] < cutoff)]
        users = []
        for uid, g in r.groupby("uid", sort=False):
            if len(g) < min_rows:
                continue
            users.append({
                "probs": np.array([probs.get((uid, a), 0.0) for a in g["anchor_date"]], dtype=float),
                "y_true": g["y"].values.astype(float),
                "pm": g["pm"].values.astype(float),
                "ps": g["ps"].values.astype(float),
            })
        return users


def fit_peak_classifier(frame, train_uids, cutoff):
    """GBM on the training users' rows whose target precedes the cutoff (all rows if None), with
    class-balanced weights and missing values imputed with the training medians."""
    feat_cols = [c for c in frame.columns if c not in META_COLS]
    train = frame[frame["uid"].isin({str(u) for u in train_uids})]
    if cutoff is not None:
        train = train[(train["week_start"] < cutoff) & (train["target_date"] < cutoff)]
    train = train[train["is_peak_next"].notna()]
    if len(train) < 50 or train["is_peak_next"].sum() < 5:
        return None, feat_cols, None

    X_tr = train[feat_cols].values.copy()
    y_tr = train["is_peak_next"].values.astype(int)
    col_med = np.nanmedian(X_tr, axis=0)
    for ci, med in enumerate(col_med):
        X_tr[np.isnan(X_tr[:, ci]), ci] = med

    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, random_state=42)
    pos_rate = y_tr.mean()
    w = np.where(y_tr == 1, 1.0 / (pos_rate + 1e-6), 1.0 / (1 - pos_rate + 1e-6))
    model.fit(X_tr, y_tr, sample_weight=w / w.mean())
    return model, feat_cols, col_med


def peak_probabilities(model, frame, feat_cols, col_med):
    """Peak probability of every row, keyed by (uid, anchor day). Empty without a model."""
    if model is None:
        return {}
    X = frame[feat_cols].values.astype(float).copy()
    r_idx, c_idx = np.where(np.isnan(X))
    X[r_idx, c_idx] = np.asarray(col_med, dtype=float)[c_idx]
    p = model.predict_proba(X)[:, 1]
    keys = zip(frame["uid"].astype(str), pd.to_datetime(frame["anchor_date"]).dt.normalize())
    return {k: float(v) for k, v in zip(keys, p)}


def row_key(uid, anchor):
    return str(uid), pd.Timestamp(anchor).normalize()
