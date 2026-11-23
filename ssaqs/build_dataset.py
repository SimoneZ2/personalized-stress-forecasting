"""Build the SSAQS (RQ4) model inputs from the raw per-user folders.

One row per user and day with a stress self-report. Outputs:
  ssaqs_lstm_input_ces.csv  LSTM input: 13 sensing z-scores, sparse EMA lags, sparse personal stats
  ssaqs_gbm_input_ces.csv   GBM input: sensing trend/point features, sparse EMA features, day of week

Peak label (is_peak): self-report >= 1 personal std above the mean of the previous 21 rows (dense).
Self-report collection policy (CES-aligned): every answer in the first 14 rows, then one every 14 rows.
Sparse personal stats: last 12 collected answers before the row (at least 4).

Usage: python3 build_dataset.py --data <SSAQS folder with one sub-folder per user> --out-dir .
"""
import argparse
import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PEAK_K = 1.0
PEAK_WINDOW = 21
ZSCORE_WINDOW = 14
MIN_ZSCORE_DAYS = 7
MIN_EMA_DAYS = 25
TREND_DAYS = 14
POINT_DAYS = 5
MIN_TREND_OBS = 7
ONBOARDING = 14
INTERVAL = 14
STATS_WINDOW = 12
STATS_MIN = 4

SENSING = ['hrv_rmssd_mean', 'hrv_rmssd_std', 'hrv_lf_mean', 'hrv_hf_mean', 'hrv_lf_hf_ratio',
           'sleep_score', 'deep_sleep_min', 'spo2_mean', 'spo2_std',
           'steps_total', 'steps_std', 'act_active_frac', 'act_sedentary_frac']


def daily(path, agg):
    """Read a sensor file and aggregate it per calendar day of its timestamp."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['date'] = pd.to_datetime(df['timestamp'].dt.date)
    return agg(df)


def hrv_agg(df):
    out = df.groupby('date').agg(hrv_rmssd_mean=('rmssd', 'mean'), hrv_rmssd_std=('rmssd', 'std'),
                                 hrv_lf_mean=('low_frequency', 'mean'), hrv_hf_mean=('high_frequency', 'mean')).reset_index()
    out['hrv_lf_hf_ratio'] = out['hrv_lf_mean'] / (out['hrv_hf_mean'] + 1e-6)
    return out


def sleep_agg(df):
    return df.groupby('date').agg(sleep_score=('overall_score', 'mean'),
                                  deep_sleep_min=('deep_sleep_in_minutes', 'mean')).reset_index()


def spo2_agg(df):
    df = df[df['value'] >= 85]
    return df.groupby('date').agg(spo2_mean=('value', 'mean'), spo2_std=('value', 'std')).reset_index()


def steps_agg(df):
    out = df.groupby('date')['steps'].agg(['sum', 'std']).reset_index()
    out.columns = ['date', 'steps_total', 'steps_std']
    return out


def activity_agg(df):
    counts = df.groupby(['date', 'level']).size().unstack(fill_value=0).reset_index()
    total = counts.drop('date', axis=1).sum(axis=1)
    active = [c for c in counts.columns if c not in ['date', 'SEDENTARY', 'total']]
    out = counts[['date']].copy()
    out['act_sedentary_frac'] = counts['SEDENTARY'] / (total + 1e-6) if 'SEDENTARY' in counts.columns else 0.0
    out['act_active_frac'] = counts[active].sum(axis=1) / (total + 1e-6)
    return out


def load_user(user_dir):
    """Daily self-reports (date in America/Mexico_City) left-joined with the daily sensor aggregates."""
    q_path = user_dir / 'daily_questions.csv'
    if not q_path.exists():
        return None
    q = pd.read_csv(q_path)
    if 'stress' not in q.columns or q['stress'].notna().sum() < 10:
        return None
    q['date'] = pd.to_datetime(pd.to_datetime(q['timeStampScheduled'], unit='s', utc=True)
                               .dt.tz_convert('America/Mexico_City').dt.date)
    df = q.groupby('date').agg({'stress': 'mean'}).reset_index().rename(columns={'stress': 'stress_raw'})
    for name, agg in [('hrv.csv', hrv_agg), ('sleep.csv', sleep_agg), ('oxygen.csv', spo2_agg),
                      ('steps.csv', steps_agg), ('activity_level.csv', activity_agg)]:
        sub = daily(user_dir / name, agg)
        if sub is not None:
            df = df.merge(sub, on='date', how='left')
    df = df.sort_values('date').reset_index(drop=True)
    df['uid'] = int(user_dir.name)
    return df


def add_sensing_zscores(df):
    """Causal z-score of each sensing signal against the previous 14 rows (at least 7)."""
    for f in SENSING:
        if f not in df.columns:
            df[f'{f}_z'] = np.nan
            continue
        roll = df[f].rolling(window=ZSCORE_WINDOW, min_periods=MIN_ZSCORE_DAYS)
        mu, sig = roll.mean().shift(1), roll.std().shift(1)
        df[f'{f}_z'] = ((df[f] - mu) / (sig + 1e-8)).where(sig >= 1e-4, 0.0)
    return df


def add_peak_label(df):
    roll = df['stress_raw'].rolling(window=PEAK_WINDOW, min_periods=7)
    mu, sig = roll.mean().shift(1), roll.std().shift(1).clip(lower=1.0)
    df['personal_mean_dense'], df['personal_std_dense'] = mu, sig
    valid = mu.notna() & df['stress_raw'].notna()
    df['is_peak'] = np.nan
    df.loc[valid, 'is_peak'] = ((df.loc[valid, 'stress_raw'] - mu[valid]) / sig[valid] >= PEAK_K).astype(float)
    return df


def collected_rows(stress):
    return [i for i in range(len(stress)) if (i < ONBOARDING or (i - ONBOARDING) % INTERVAL == 0) and not np.isnan(stress[i])]


def add_policy_features(df):
    """Sparse personal stats and EMA features from the self-reports collected before each row."""
    s = df['stress_raw'].values.astype(float); n = len(s)
    idx = collected_rows(s)
    cols = {c: np.full(n, np.nan) for c in ['personal_mean', 'personal_std', 'ema_lag1', 'ema_lag2', 'ema_lag3',
                                            'ema_trend', 'days_since_last_ema', 'gbm_ema_lag2', 'ema_z']}
    cols['ema_collected'] = np.zeros(n); cols['n_ema_available'] = np.zeros(n)
    for i in range(n):
        past = [j for j in idx if j < i]
        hist = [s[j] for j in past][-STATS_WINDOW:]
        if len(hist) >= STATS_MIN:
            cols['personal_mean'][i], cols['personal_std'][i] = np.mean(hist), np.std(hist)
        pm, ps = cols['personal_mean'][i], cols['personal_std'][i]
        cols['n_ema_available'][i] = len(past)
        if i in idx:
            cols['ema_collected'][i] = 1.0
        if past:
            cols['ema_lag1'][i] = s[past[-1]] / 100.0
            cols['days_since_last_ema'][i] = i - past[-1]
            cols['gbm_ema_lag2'][i] = s[past[-1]] / 100.0          # GBM: last answer until a second one exists
            if not np.isnan(pm) and not np.isnan(ps) and ps > 0:
                cols['ema_z'][i] = (s[past[-1]] - pm) / ps
        if len(past) >= 2:
            cols['ema_lag2'][i] = cols['gbm_ema_lag2'][i] = s[past[-2]] / 100.0
            recent = [s[j] / 100.0 for j in past[-5:]]
            cols['ema_trend'][i] = cols['ema_lag1'][i] - np.mean(recent[:-1])
        if len(past) >= 3:
            cols['ema_lag3'][i] = s[past[-3]] / 100.0
    for c, v in cols.items():
        df[c] = v
    return df


def csv_round_trip(df):
    """Write and re-read a table as CSV text. The original build ran in three steps that exchanged CSV files,
    and pandas' default CSV parser is not round-trip exact for floats: repeating the exchanges at the same
    points keeps every value bit-identical to the inputs used for the paper."""
    return pd.read_csv(io.StringIO(df.to_csv(index=False)), parse_dates=['date'])


def ols_slope(y):
    valid = ~np.isnan(y)
    if valid.sum() < MIN_TREND_OBS:
        return np.nan
    x = np.arange(len(y), dtype=float)[valid]; y = y[valid]
    denom = ((x - x.mean()) ** 2).sum()
    return np.nan if denom < 1e-10 else float(((x - x.mean()) * (y - y.mean())).sum() / denom)


def add_gbm_sensing(df):
    """Trend (OLS slope over the previous 14 rows) and point (mean of the previous 5 rows) of each z-score,
    then per-user forward fill of z-scores, trends and points."""
    n = len(df)
    for f in SENSING:
        z = df[f'{f}_z'].values
        trend, point = np.full(n, np.nan), np.full(n, np.nan)
        for i in range(n):
            trend[i] = ols_slope(z[max(0, i - TREND_DAYS):i])
            w = z[max(0, i - POINT_DAYS):i]; w = w[~np.isnan(w)]
            if len(w) >= 1:
                point[i] = float(w.mean())
        df[f'trend_{f}'], df[f'point_{f}'] = trend, point
    fill = [f'{f}_z' for f in SENSING] + [f'trend_{f}' for f in SENSING] + [f'point_{f}' for f in SENSING]
    df[fill] = df[fill].ffill()
    dow = df['date'].dt.dayofweek
    df['dow_sin'], df['dow_cos'] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    df['is_peak_next'] = df['is_peak'].shift(-1)
    return df


DENSE_COLS = (['uid', 'date', 'stress_raw', 'is_peak', 'personal_mean_dense', 'personal_std_dense']
              + [f'{f}_z' for f in SENSING])
LSTM_COLS = (['uid', 'date', 'stress_raw', 'is_peak', 'personal_mean', 'personal_std'] + [f'{f}_z' for f in SENSING]
             + ['ema_lag1', 'ema_lag2', 'ema_lag3', 'ema_trend', 'ema_collected', 'days_since_last_ema',
                'personal_mean_dense', 'personal_std_dense'])
GBM_COLS = (['uid', 'date', 'stress_raw', 'is_peak', 'is_peak_next', 'personal_mean', 'personal_std']
            + [f'{p}{f}{s}' for p, s in [('', '_z'), ('trend_', ''), ('point_', '')] for f in SENSING]
            + ['ema_lag2', 'ema_z', 'ema_trend', 'n_ema_available', 'dow_sin', 'dow_cos',
               'personal_mean_dense', 'personal_std_dense'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out-dir', default='.')
    a = p.parse_args()
    user_dirs = sorted([d for d in Path(a.data).iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: int(d.name))
    dense = []
    for d in user_dirs:
        df = load_user(d)
        if df is None or df['stress_raw'].notna().sum() < MIN_EMA_DAYS:
            print(f'  uid={d.name}: skipped')
            continue
        dense.append(add_peak_label(add_sensing_zscores(df))[DENSE_COLS])
    dense = csv_round_trip(pd.concat(dense, ignore_index=True))
    by_user = lambda df, fn: pd.concat([fn(g.reset_index(drop=True)) for _, g in df.groupby('uid', sort=False)], ignore_index=True)
    lstm = by_user(dense, add_policy_features)[LSTM_COLS]
    gbm = by_user(csv_round_trip(by_user(dense, add_gbm_sensing)), add_policy_features)
    gbm['ema_lag2'] = gbm['gbm_ema_lag2']
    gbm = gbm[GBM_COLS]
    lstm['date'] = lstm['date'].dt.strftime('%Y-%m-%d'); gbm['date'] = gbm['date'].dt.strftime('%Y-%m-%d')
    for df, name in [(lstm, 'ssaqs_lstm_input_ces.csv'), (gbm, 'ssaqs_gbm_input_ces.csv')]:
        df.to_csv(os.path.join(a.out_dir, name), index=False)
        print(f'{name}: {df.shape}, {df.uid.nunique()} users')
    per_user = lstm.groupby('uid')['ema_collected'].sum()
    print(f'peaks {int(lstm.is_peak.sum())} ({100 * lstm.is_peak.mean():.1f}%), collected answers per user: median '
          f'{per_user.median():.0f} min {per_user.min():.0f} max {per_user.max():.0f}')


if __name__ == '__main__':
    main()
