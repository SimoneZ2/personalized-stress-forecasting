#!/usr/bin/env python3
"""
Build the forecasting dataset shared by the LSTM and the GBM peak classifier.

One row per (user, anchor):
  anchor D : day of the user's last response in calendar week t (Mon-Sun)
  target   : the user's last response in calendar week t+1, provided it falls
             at least 5 days after D (5-13 days, median 7)
  X        : 14 days of daily sensing features (D-14 ... D-1), each feature
             z-scored against the user's own previous days of data

Weeks whose following calendar week has no response produce no row, so every
response is the target of at most one row.

Usage:
  python build_forecast_dataset.py --ema dataset/EMA/general_ema.csv \
      --sens dataset/Sensing/sensing.csv --out daily_lagged_forecast_t1_v5.npz
  PHQ-4 target: add --target-col phq4_score
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MIN_QUALITY_HOURS = 6.0      # sensing days with less activity data are discarded
SEQ_DAYS = 14                # input window length
BASELINE_DAYS = 84           # z-score baseline: previous 84 available days
MIN_BASELINE_DAYS = 7        # minimum baseline size to compute a z-score
MIN_RESPONSE_WEEKS = 21      # users need >= 20 weeks with a response + 1 forecast week
MIN_HORIZON_DAYS = 5         # minimum distance between anchor and target

# Daily features: output name -> raw sensing columns (averaged when more than one).
# Epochs: ep_0 = whole day, ep_2 = afternoon. None = derived below.
SENSING_FEATURES = {
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

# Features for which a zero value means "sensor off" and is treated as missing.
ZERO_IS_MISSING = {"loc_max_campus_dist"}

# Place durations used for loc_home_ratio = home time / time in all these places.
HOME_RATIO_PLACES = ["loc_home_dur", "loc_self_dorm_dur", "loc_study_dur",
                     "loc_social_dur", "loc_food_dur", "loc_leisure_dur"]


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


def load_daily_sensing(path):
    """Daily sensing features (raw scale), one row per (user, day)."""
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["day"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "uid"])

    quality = pd.to_numeric(df["quality_activity"], errors="coerce")
    low = quality < MIN_QUALITY_HOURS
    log.info("Sensing days discarded for low quality: %d (%.1f%%)", low.sum(), 100 * low.mean())
    df = df[~low].copy()

    out = df[["uid", "date"]].copy()
    for name, cols in SENSING_FEATURES.items():
        if cols is None:  # sleep onset hour from Unix time; hours 0-5 mapped to 24-29
            hour = (pd.to_numeric(df["sleep_start"], errors="coerce") % 86400) / 3600
            out[name] = hour.where(hour >= 6, hour + 24).values
            continue
        present = [c for c in cols if c in df.columns]
        if not present:
            raise KeyError(f"no sensing column found for feature {name}: {cols}")
        values = df[present].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        if name in ZERO_IS_MISSING:
            values = values.replace(0, np.nan)
        out[name] = values.values

    places = df.reindex(columns=HOME_RATIO_PLACES).apply(pd.to_numeric, errors="coerce")
    total = places.sum(axis=1).replace(0, np.nan)
    out["loc_home_ratio"] = (places["loc_home_dur"] / total).clip(0, 1).values

    for name in out.columns[2:]:
        log.info("  %-22s missing %.1f%%", name, 100 * out[name].isna().mean())
    return out


def add_rolling_zscores(sens, feature_cols):
    """Per-user causal z-score: each day against the previous BASELINE_DAYS available days."""
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


def build_rows(responses, n_weeks, sens_z, z_cols):
    """One row per anchor whose next calendar week has a response >= MIN_HORIZON_DAYS later."""
    sens_index = {}
    for uid, grp in sens_z.groupby("uid"):
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
    log.info("Rows: %d | users %d | missing input days %.1f%%",
             len(X), len(set(uids)), 100 * n_missing_days / (len(X) * SEQ_DAYS))
    horizon = (pd.to_datetime(targets) - pd.to_datetime(anchors)).days
    log.info("Horizon (days): median %.0f | min %d | max %d", np.median(horizon), horizon.min(), horizon.max())
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(weeks),
            np.array(uids), np.array(anchors), np.array(targets))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ema", required=True, help="general_ema.csv")
    ap.add_argument("--sens", required=True, help="sensing.csv")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--target-col", default="stress", help="EMA column to forecast (stress or phq4_score)")
    args = ap.parse_args()

    responses, n_weeks = load_weekly_responses(args.ema, args.target_col)
    sens = load_daily_sensing(args.sens)
    feature_cols = list(sens.columns[2:])
    sens_z = add_rolling_zscores(sens, feature_cols)
    z_cols = [f"{f}_z" for f in feature_cols]
    X, y, weeks, uids, anchors, targets = build_rows(responses, n_weeks, sens_z, z_cols)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Configuration metadata kept with the original keys for the downstream scripts.
    np.savez_compressed(out, X=X, y=y, weeks=weeks, uids=uids, anchor_dates=anchors, target_dates=targets,
                        feature_names=np.array(z_cols), seq_days=np.array([SEQ_DAYS]),
                        baseline_days=np.array([BASELINE_DAYS]), feature_set=np.array(["lagged"]),
                        forecast_weeks=np.array([1]), all_ema=np.array([0]),
                        target_next_only=np.array([1]), target_next_week=np.array([1]))
    log.info("Saved %s | X %s | y %.1f-%.1f", out, X.shape, y.min(), y.max())


if __name__ == "__main__":
    main()
