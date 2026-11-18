#!/usr/bin/env python3
"""
Build the forecasting dataset shared by the LSTM and the GBM peak classifier.

One row per (user, anchor):
  anchor D : day of the user's last response in calendar week t (Mon-Sun)
  target   : the user's last response in calendar week t+1, provided it falls
             at least 5 days after D (5-13 days, median 7)

Weeks whose following calendar week has no response produce no row, so every
response is the target of at most one row.

For each row the file stores the fold-independent inputs of both models:
  X      (LSTM) 14 days of daily sensing (D-14 ... D-1), each signal z-scored
         with mean/std of the user's previous 84 days with data
  gbm_X  (GBM)  trend (slope over D-28 ... D-1) and level (mean over D-7 ... D-1)
         of daily signals z-scored with median/MAD of the previous 84 days with
         data, plus the academic period of D (one-hot)
The fold-dependent GBM inputs (EMA features, policy) are computed in the simulation.

Usage:
  python build_forecast_dataset.py --ema dataset/EMA/general_ema.csv \
      --sens dataset/Sensing/sensing.csv --out ces_stress_forecast.npz
  python build_forecast_dataset.py --ema dataset/EMA/general_ema.csv \
      --sens dataset/Sensing/sensing.csv --target-col phq4_score --out ces_phq4_forecast.npz
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MIN_QUALITY_HOURS = 6.0      # sensing days with less activity data are discarded
SEQ_DAYS = 14                # LSTM input window
BASELINE_DAYS = 84           # z-score baseline: previous 84 days with data
MIN_BASELINE_DAYS = 7        # minimum baseline size to compute a z-score
MIN_RESPONSE_WEEKS = 21      # users need >= 20 weeks with a response + 1 forecast week
MIN_HORIZON_DAYS = 5         # minimum distance between anchor and target
GBM_TREND_DAYS = 28          # GBM trend window
GBM_LEVEL_DAYS = 7           # GBM level window
DAY = np.timedelta64(1, "D")

# ---------------------------------------------------------------------------
# LSTM signals: output name -> raw columns (averaged when more than one).
# Epochs: ep_0 = whole day, ep_2 = afternoon. None = derived below.
# ---------------------------------------------------------------------------
LSTM_SIGNALS = {
    "loc_max_campus_dist": ["loc_max_dis_from_campus_ep_0", "loc_max_dis_from_campus_ep_1",
                            "loc_max_dis_from_campus_ep_2", "loc_max_dis_from_campus_ep_3"],
    "loc_study_dur": ["loc_study_dur"],
    "loc_self_dorm": ["loc_self_dorm_dur"],
    "loc_food_dur": ["loc_food_dur"],
    "loc_social_dur": ["loc_social_dur"],
    "loc_home_dur": ["loc_home_dur"],
    "loc_visit_num_pm": ["loc_visit_num_ep_2"],
    "act_in_vehicle_day": ["act_in_vehicle_ep_0"],
    "act_still_pm": ["act_still_ep_2"],
    "act_walking_pm": ["act_walking_ep_2"],
    "sleep_dur": ["sleep_duration"],
    "sleep_start_hour": None,
    "audio_play_num": ["other_playing_num_ep_0", "other_playing_num_ep_1",
                       "other_playing_num_ep_2", "other_playing_num_ep_3"],
    "unlock_num_day": ["unlock_num_ep_0"],
}
LSTM_ZERO_IS_MISSING = {"loc_max_campus_dist"}   # zero = sensor off
HOME_RATIO_PLACES = ["loc_home_dur", "loc_self_dorm_dur", "loc_study_dur",
                     "loc_social_dur", "loc_food_dur", "loc_leisure_dur"]

# ---------------------------------------------------------------------------
# GBM signals: output name -> raw columns, always averaged over the 4 epochs.
# ---------------------------------------------------------------------------
GBM_SIGNALS = {
    "loc_self_dorm": ["loc_self_dorm_dur"],
    "loc_study_dur": ["loc_study_dur"],
    "loc_max_campus": ["loc_max_dis_from_campus_ep_0", "loc_max_dis_from_campus_ep_1",
                       "loc_max_dis_from_campus_ep_2", "loc_max_dis_from_campus_ep_3"],
    "loc_visit_num": ["loc_visit_num_ep_0", "loc_visit_num_ep_1", "loc_visit_num_ep_2", "loc_visit_num_ep_3"],
    "act_vehicle": ["act_in_vehicle_ep_0", "act_in_vehicle_ep_1", "act_in_vehicle_ep_2", "act_in_vehicle_ep_3"],
    "loc_dist": ["loc_dist_ep_0", "loc_dist_ep_1", "loc_dist_ep_2", "loc_dist_ep_3"],
    "sleep_dur": ["sleep_duration"],
    "loc_social_dur": ["loc_social_dur"],
    "unlock_num": ["unlock_num_ep_0", "unlock_num_ep_1", "unlock_num_ep_2", "unlock_num_ep_3"],
    "act_still": ["act_still_ep_0", "act_still_ep_1", "act_still_ep_2", "act_still_ep_3"],
    "loc_home_dur": ["loc_home_dur"],
}
GBM_TREND_SIGNALS = ["loc_self_dorm", "loc_study_dur", "loc_max_campus", "loc_visit_num",
                     "act_vehicle", "loc_dist", "sleep_dur", "loc_social_dur", "unlock_num"]
GBM_LEVEL_SIGNALS = GBM_TREND_SIGNALS + ["act_still", "loc_home_dur"]
GBM_ZERO_IS_MISSING = {"loc_dist", "loc_max_campus", "loc_visit_num", "act_vehicle"}

# ---------------------------------------------------------------------------
# Dartmouth academic calendar (term type, year, first day, last day)
# ---------------------------------------------------------------------------
DARTMOUTH_TERMS = [
    ("fall", 2017, "2017-09-11", "2017-11-25"), ("fall", 2018, "2018-09-10", "2018-11-24"),
    ("fall", 2019, "2019-09-16", "2019-11-30"), ("fall", 2020, "2020-09-14", "2020-11-28"),
    ("fall", 2021, "2021-09-13", "2021-11-27"),
    ("winter", 2018, "2018-01-03", "2018-03-10"), ("winter", 2019, "2019-01-07", "2019-03-09"),
    ("winter", 2020, "2020-01-06", "2020-03-07"), ("winter", 2021, "2021-01-04", "2021-03-13"),
    ("winter", 2022, "2022-01-03", "2022-03-12"),
    ("spring", 2018, "2018-03-26", "2018-06-02"), ("spring", 2019, "2019-03-25", "2019-06-01"),
    ("spring", 2020, "2020-03-23", "2020-05-30"), ("spring", 2021, "2021-03-29", "2021-06-05"),
    ("spring", 2022, "2022-03-28", "2022-06-04"),
    ("summer", 2018, "2018-06-18", "2018-09-08"), ("summer", 2019, "2019-06-17", "2019-09-07"),
    ("summer", 2020, "2020-06-15", "2020-09-05"), ("summer", 2021, "2021-06-14", "2021-09-11"),
]
ACADEMIC_PERIODS = ["pre_reentry", "reentry", "midterm", "finals", "normal", "break"]


# ===========================================================================
# EMA responses and rows
# ===========================================================================
def load_weekly_responses(path, target_col):
    """Keep the last response of each (user, calendar week)."""
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["day"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", target_col, "uid"])
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    df["week_start"] = df["date"].dt.to_period("W").dt.start_time
    df = df.sort_values(["uid", "date"]).reset_index(drop=True)

    n_raw = len(df)
    per_week = df.groupby(["uid", "week_start"]).size()
    log.info("Responses: %d | users %d | weeks with more than one response: %.1f%%",
             n_raw, df["uid"].nunique(), 100 * (per_week > 1).mean())

    df = df.groupby(["uid", "week_start"], sort=False).last().reset_index()
    df = df.sort_values(["uid", "date"]).reset_index(drop=True)
    log.info("Last response per week: %d kept, %.1f%% discarded", len(df), 100 * (n_raw - len(df)) / n_raw)

    n_weeks = df.groupby("uid")["week_start"].nunique().to_dict()
    df = df.rename(columns={target_col: "y"})
    return df[["uid", "date", "y", "week_start"]], n_weeks


def build_rows(responses, n_weeks, lstm_z, z_cols):
    """One row per anchor whose next calendar week has a response >= MIN_HORIZON_DAYS later."""
    sens_index = {}
    for uid, grp in lstm_z.groupby("uid"):
        grp = grp.sort_values("date")
        sens_index[uid] = dict(zip(grp["date"].dt.date, grp[z_cols].to_numpy(np.float32)))

    X, y, weeks, uids, anchors, targets = [], [], [], [], [], []
    n_missing_days = 0
    for uid, grp in responses.groupby("uid"):
        if n_weeks.get(uid, 0) < MIN_RESPONSE_WEEKS or uid not in sens_index:
            continue
        user_sens = sens_index[uid]
        rows = grp.sort_values("date")[["date", "y", "week_start"]].values.tolist()
        for (d_anchor, _, week), (d_target, y_target, week_target) in zip(rows[:-1], rows[1:]):
            d_anchor, d_target = pd.Timestamp(d_anchor), pd.Timestamp(d_target)
            week = pd.Timestamp(week)
            if pd.Timestamp(week_target) != week + pd.Timedelta(days=7):
                continue
            if d_target < d_anchor + pd.Timedelta(days=MIN_HORIZON_DAYS):
                continue
            seq = np.full((SEQ_DAYS, len(z_cols)), np.nan, dtype=np.float32)
            for d in range(SEQ_DAYS):
                day = (d_anchor - pd.Timedelta(days=SEQ_DAYS - d)).date()
                if day in user_sens:
                    seq[d] = user_sens[day]
                else:
                    n_missing_days += 1
            X.append(seq)
            y.append(float(y_target))
            weeks.append(str(week.date()))
            uids.append(str(uid))
            anchors.append(str(d_anchor.date()))
            targets.append(str(d_target.date()))

    if not X:
        raise RuntimeError("no rows built")
    log.info("Rows: %d | users %d | missing LSTM input days %.1f%%",
             len(X), len(set(uids)), 100 * n_missing_days / (len(X) * SEQ_DAYS))
    horizon = (pd.to_datetime(targets) - pd.to_datetime(anchors)).days
    log.info("Horizon (days): median %.0f | min %d | max %d", np.median(horizon), horizon.min(), horizon.max())
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(weeks),
            np.array(uids), np.array(anchors), np.array(targets))


# ===========================================================================
# Sensing
# ===========================================================================
def read_sensing(path):
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["day"].astype(str), format="%Y%m%d", errors="coerce")
    return df.dropna(subset=["date", "uid"])


def lstm_daily_signals(raw):
    """Daily LSTM signals (raw scale), one row per (user, day)."""
    quality = pd.to_numeric(raw["quality_activity"], errors="coerce")
    df = raw[~(quality < MIN_QUALITY_HOURS)].copy()

    out = df[["uid", "date"]].copy()
    for name, cols in LSTM_SIGNALS.items():
        if cols is None:  # sleep onset hour from Unix time; hours 0-5 mapped to 24-29
            hour = (pd.to_numeric(df["sleep_start"], errors="coerce") % 86400) / 3600
            out[name] = hour.where(hour >= 6, hour + 24).values
            continue
        present = [c for c in cols if c in df.columns]
        if not present:
            raise KeyError(f"no sensing column found for {name}: {cols}")
        values = df[present].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        if name in LSTM_ZERO_IS_MISSING:
            values = values.replace(0, np.nan)
        out[name] = values.values

    places = df.reindex(columns=HOME_RATIO_PLACES).apply(pd.to_numeric, errors="coerce")
    total = places.sum(axis=1).replace(0, np.nan)
    out["loc_home_ratio"] = (places["loc_home_dur"] / total).clip(0, 1).values
    return out


def add_mean_std_zscores(sens, feature_cols):
    """LSTM z-score: each day against mean/std of the previous BASELINE_DAYS days with data."""
    parts = []
    for _, grp in sens.groupby("uid"):
        grp = grp.sort_values("date").copy()
        for feat in feature_cols:
            series = grp[feat].values.astype(float)
            z = np.full(len(series), np.nan)
            for i in range(len(series)):
                base = series[max(0, i - BASELINE_DAYS):i]
                base = base[~np.isnan(base)]
                if len(base) < MIN_BASELINE_DAYS:
                    continue
                mu, sigma = base.mean(), base.std()
                if sigma < 1e-6:
                    z[i] = 0.0
                else:
                    z[i] = (series[i] - mu) / sigma if not np.isnan(series[i]) else np.nan
            grp[f"{feat}_z"] = z
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def _median_mad_zscore(s):
    """GBM z-score: median/MAD of the previous BASELINE_DAYS days with data, clipped to +-8."""
    roll = s.rolling(window=BASELINE_DAYS, min_periods=MIN_BASELINE_DAYS)
    med = roll.median().shift(1)
    mad = roll.apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True).shift(1) * 1.4826
    return ((s - med) / mad.clip(lower=1e-6)).clip(-8, 8)


def gbm_daily_zscores(raw):
    """Daily GBM signals z-scored per user, one row per (user, day)."""
    df = raw[raw["quality_activity"].fillna(0) >= MIN_QUALITY_HOURS]
    daily = df[["uid", "date"]].copy()
    for name, cols in GBM_SIGNALS.items():
        daily[name] = df[[c for c in cols if c in df.columns]].mean(axis=1).values
        if name in GBM_ZERO_IS_MISSING:
            daily.loc[daily[name] == 0, name] = np.nan
    daily = daily.sort_values(["uid", "date"]).reset_index(drop=True)
    z = {f"{name}_z": daily.groupby("uid", group_keys=False)[name].apply(_median_mad_zscore) for name in GBM_SIGNALS}
    out = pd.concat([daily[["uid", "date"]], pd.DataFrame(z)], axis=1)
    out["uid"] = out["uid"].astype(str)
    return out


def _slopes(window):
    """Least-squares slope of each column over the available days (rows) of the window."""
    slopes = np.full(window.shape[1], np.nan)
    for j in range(window.shape[1]):
        vals = window[:, j]
        mask = ~np.isnan(vals)
        if mask.sum() < 2:
            continue
        xv, yv = np.arange(len(vals), dtype=float)[mask], vals[mask]
        xm, ym = xv.mean(), yv.mean()
        denom = ((xv - xm) ** 2).sum()
        if denom < 1e-12:
            continue
        slopes[j] = ((xv - xm) * (yv - ym)).sum() / denom
    return slopes


def gbm_sensing_features(gbm_z, uids, anchors):
    """Trend over [D-28, D-1] and level over [D-7, D-1] for each row."""
    zt = [f"{s}_z" for s in GBM_TREND_SIGNALS]
    zl = [f"{s}_z" for s in GBM_LEVEL_SIGNALS]
    trend = np.full((len(uids), len(zt)), np.nan)
    level = np.full((len(uids), len(zl)), np.nan)
    by_uid = {u: g for u, g in gbm_z.groupby("uid", sort=False)}
    anchors = pd.to_datetime(anchors).values
    t_days, l_days = np.timedelta64(GBM_TREND_DAYS, "D"), np.timedelta64(GBM_LEVEL_DAYS, "D")

    for uid in np.unique(uids):
        ud = by_uid.get(uid)
        if ud is None:
            continue
        dates = ud["date"].values
        ZT, ZL = ud[zt].values.astype(float), ud[zl].values.astype(float)
        for i in np.where(uids == uid)[0]:
            D = anchors[i]
            hi = np.searchsorted(dates, D - DAY, side="right")
            lo_t = np.searchsorted(dates, D - t_days, side="left")
            lo_l = np.searchsorted(dates, D - l_days, side="left")
            if hi - lo_t >= 2:
                trend[i] = _slopes(ZT[lo_t:hi])
            if hi - lo_l > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    level[i] = np.nanmean(ZL[lo_l:hi], axis=0)

    names = [f"trend_{s}" for s in GBM_TREND_SIGNALS] + [f"point_{s}" for s in GBM_LEVEL_SIGNALS]
    return np.hstack([trend, level]), names


# ===========================================================================
# Academic calendar
# ===========================================================================
def academic_weeks():
    """Monday of each week -> academic period."""
    records = []
    for term_type, year, start, end in DARTMOUTH_TERMS:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        monday = start - pd.Timedelta(days=start.weekday())
        total_weeks = int((end - start).days / 7) + 1
        week_num, cursor = 0, monday
        while cursor <= end:
            week_num += 1
            if week_num == 1:
                period = "reentry"
            elif week_num in (5, 6):
                period = "midterm"
            elif week_num >= total_weeks - 1:
                period = "finals"
            else:
                period = "normal"
            records.append((cursor, period))
            cursor += pd.Timedelta(weeks=1)

    covered = {w for w, _ in records}
    for _, _, start, _ in DARTMOUTH_TERMS:  # the 3 weeks before each term
        start = pd.Timestamp(start)
        monday = start - pd.Timedelta(days=start.weekday())
        for w in range(1, 4):
            week = monday - pd.Timedelta(weeks=w)
            if week not in covered:
                records.append((week, "pre_reentry"))
                covered.add(week)

    periods = {}
    for week, period in records:  # first assignment wins
        periods.setdefault(week, period)
    return periods


def academic_period_features(anchors):
    """One-hot academic period of the week containing D (break outside terms)."""
    periods = academic_weeks()
    d = pd.to_datetime(anchors)
    mondays = (d - pd.to_timedelta(d.weekday, unit="D")).normalize()
    labels = np.array([periods.get(m, "break") for m in mondays])
    onehot = np.stack([(labels == p).astype(float) for p in ACADEMIC_PERIODS], axis=1)
    return onehot, [f"ctx_{p}" for p in ACADEMIC_PERIODS]


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ema", required=True, help="general_ema.csv")
    ap.add_argument("--sens", required=True, help="sensing.csv")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--target-col", default="stress", help="EMA column to forecast (stress or phq4_score)")
    args = ap.parse_args()

    responses, n_weeks = load_weekly_responses(args.ema, args.target_col)
    raw = read_sensing(args.sens)

    lstm_daily = lstm_daily_signals(raw)
    lstm_cols = list(lstm_daily.columns[2:])
    lstm_z = add_mean_std_zscores(lstm_daily, lstm_cols)
    z_cols = [f"{f}_z" for f in lstm_cols]
    X, y, weeks, uids, anchors, targets = build_rows(responses, n_weeks, lstm_z, z_cols)

    gbm_sens, gbm_sens_names = gbm_sensing_features(gbm_daily_zscores(raw), uids, anchors)
    gbm_ctx, gbm_ctx_names = academic_period_features(anchors)
    gbm_X = np.hstack([gbm_sens, gbm_ctx])
    gbm_names = gbm_sens_names + gbm_ctx_names
    for name, miss in zip(gbm_names, np.isnan(gbm_X).mean(axis=0)):
        log.info("  GBM %-22s missing %.1f%%", name, 100 * miss)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Configuration metadata kept with the original keys for the downstream scripts.
    np.savez_compressed(out, X=X, y=y, weeks=weeks, uids=uids, anchor_dates=anchors, target_dates=targets,
                        feature_names=np.array(z_cols), gbm_X=gbm_X, gbm_feature_names=np.array(gbm_names),
                        seq_days=np.array([SEQ_DAYS]), baseline_days=np.array([BASELINE_DAYS]),
                        feature_set=np.array(["lagged"]), forecast_weeks=np.array([1]), all_ema=np.array([0]),
                        target_next_only=np.array([1]), target_next_week=np.array([1]))
    log.info("Saved %s | X %s | gbm_X %s | y %.1f-%.1f", out, X.shape, gbm_X.shape, y.min(), y.max())


if __name__ == "__main__":
    main()
