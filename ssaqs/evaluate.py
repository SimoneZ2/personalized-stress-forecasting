"""SSAQS (RQ4): full system and personalized LSTM against the Fitbit Stress Score and naive baselines, on the same
rows, for one validation schema and one or more seeds (one predictions file per seed).

Predictors:
  y_pred_ens   full system (LSTM + peak classifier + correction)
  y_pred_lstm  personalized LSTM
  fitbit_raw   100 - Fitbit STRESS_SCORE of day D-1 (no older value)
  fitbit_cal   population linear regression STRESS_SCORE(D-1) -> stress(D), trained on the users of the other
               folds (LUO) or on the pairs before the start of the fold (WF)
  dummy_pop    population mean of the same training set
  pm_sparse    mean of the self-reports collected by the policy before D
  last_sparse  last self-report collected by the policy before D
  lag1_dense   self-report of day D-1 (reference: needs a self-report every day)

Metrics: MAE, MAE_peak, pooled rho, pooled CCC, per-user rho (users with >= 5 rows).
Tests: per-user differences of mean absolute error, two-sided Wilcoxon and 95% bootstrap CI over users;
MAE_peak on users with >= 3 peaks.
Detection: AUC of the GBM probability and of alternative scores; F1 with forward tolerance k = 0..3 days for the
system flag and the Fitbit flag, against random alerts (flags permuted within user, same number of flags).

Output in --out: rq4_rows_<schema>_<seed folder>__<file>.csv (rows of the comparison) and
rq4_{metrics,tests,detection}_<schema>.csv, read by rq4_table.py.

Usage: python3 evaluate.py --schema LUO --data <SSAQS folder> --input ssaqs_lstm_input_ces.csv \
           --preds runs/seed42/ensemble_forecast_luo_adaptive_preds.csv runs/seed43/... --out rq4_eval
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

MIN_ROWS_RHO = 5
N_BOOT, N_PERM = 2000, 1000
# Fitbit peak: score at least 1 std below the mean of the previous 21 values (high score = low stress)
FITBIT_PEAK_K, FITBIT_WINDOW, FITBIT_MIN_PERIODS, FITBIT_SIGMA_FLOOR = 1.0, 21, 7, 1.0
DAY = pd.Timedelta(days=1)

p = argparse.ArgumentParser()
p.add_argument('--schema', required=True, choices=['WF', 'LUO'])
p.add_argument('--preds', nargs='+', required=True)
p.add_argument('--data', required=True, help='SSAQS folder with <uid>/stress.csv')
p.add_argument('--input', default='ssaqs_lstm_input_ces.csv', help='stress_raw and ema_collected of every row')
p.add_argument('--out', default='rq4_eval')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(0)

# ---------------- self-reports and naive baselines ----------------
cal = pd.read_csv(a.input, parse_dates=['date'])[['uid', 'date', 'stress_raw', 'ema_collected']]
cal['date'] = cal['date'].astype('datetime64[ns]')
cal = cal.sort_values(['uid', 'date']).reset_index(drop=True)
obs = cal['stress_raw'].where(cal['ema_collected'] == 1)
cal['pm_sparse'] = obs.groupby(cal['uid']).transform(lambda x: x.expanding().mean().shift(1))
cal['last_sparse'] = obs.groupby(cal['uid']).transform(lambda x: x.ffill().shift(1))
prev = cal[['uid', 'date', 'stress_raw']].assign(date=lambda d: d['date'] + DAY).rename(columns={'stress_raw': 'lag1_dense'})
cal = cal.merge(prev, on=['uid', 'date'], how='left')


# ---------------- Fitbit Stress Score ----------------
def load_fitbit(uid):
    """Successful calculations, one per day (the most recently updated)."""
    df = pd.read_csv(f'{a.data}/{uid}/stress.csv')
    df = df[df['CALCULATION_FAILED'].astype(str).str.strip().str.lower() == 'false'].copy()
    df['date'] = pd.to_datetime(pd.to_datetime(df['DATE'], format='ISO8601').dt.date)
    df['upd'] = pd.to_datetime(df['UPDATED_AT'], format='ISO8601')
    n0 = len(df)
    df = df.sort_values(['date', 'upd']).drop_duplicates('date', keep='last')
    df['uid'] = uid
    return df[['uid', 'date', 'STRESS_SCORE']].reset_index(drop=True), n0 - len(df)


def fitbit_peak_flag(df):
    roll = df['STRESS_SCORE'].rolling(window=FITBIT_WINDOW, min_periods=FITBIT_MIN_PERIODS)
    mu, sig = roll.mean().shift(1), roll.std().shift(1).clip(lower=FITBIT_SIGMA_FLOOR)
    valid = mu.notna() & df['STRESS_SCORE'].notna()
    flag = pd.Series(np.nan, index=df.index)
    flag.loc[valid] = ((mu[valid] - df.loc[valid, 'STRESS_SCORE']) / sig[valid] >= FITBIT_PEAK_K).astype(float)
    return flag


fb, n_dup = [], 0
for x in sorted(cal['uid'].unique()):
    try:
        f, nd = load_fitbit(x)
    except FileNotFoundError:
        continue
    n_dup += nd
    f['fitbit_peak_flag'] = fitbit_peak_flag(f).values
    fb.append(f)
fb = pd.concat(fb, ignore_index=True)
fb['date'] = pd.to_datetime(fb['date']).astype('datetime64[ns]')
fb = fb.dropna(subset=['STRESS_SCORE'])
assert not fb.duplicated(['uid', 'date']).any()
print(f'Fitbit: {n_dup} duplicated same-day calculations removed (most recent UPDATED_AT kept)')
fbp = fb[['uid', 'date', 'STRESS_SCORE', 'fitbit_peak_flag']].assign(date=lambda d: d['date'] + DAY)  # day D-1 on row D
cal = cal.merge(fbp, on=['uid', 'date'], how='left')
cal['fitbit_raw'] = 100 - cal['STRESS_SCORE']
pairs = cal.dropna(subset=['STRESS_SCORE', 'stress_raw'])
print(f'Users with Fitbit: {fb.uid.nunique()}  rows with Fitbit(D-1): {cal.STRESS_SCORE.notna().mean() * 100:.1f}%')


# ---------------- metrics and tests ----------------
def corr(x, y):
    return np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan


def ccc(x, y):
    c = ((x - x.mean()) * (y - y.mean())).mean(); d = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return 2 * c / d if d > 0 else np.nan


def user_rho(v, col):
    r = {uid: corr(g['y_true'].values.astype(float), g[col].values.astype(float)) for uid, g in v.groupby('uid') if len(g) >= MIN_ROWS_RHO}
    return pd.Series(r).dropna()


def metrics(v, col):
    yt = v['y_true'].values.astype(float); yp = v[col].values.astype(float); pk = v['is_peak'].values == 1
    e = np.abs(yt - yp); ru = user_rho(v, col)
    return dict(MAE=e.mean(), MAE_peak=e[pk].mean(), rho_pooled=corr(yt, yp), CCC=ccc(yt, yp),
                rho_user=ru.mean() if len(ru) else np.nan, n_users_rho=len(ru))


def paired(v, x, y, min_n=1):
    g = v.assign(dd=(v['y_true'] - v[x]).abs() - (v['y_true'] - v[y]).abs()).groupby('uid')['dd'].agg(['sum', 'count', 'mean'])
    g = g[g['count'] >= min_n]; S = g['sum'].values; N = g['count'].values
    bs = [S[i].sum() / N[i].sum() for i in (rng.integers(0, len(S), len(S)) for _ in range(N_BOOT))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pv = wilcoxon(g['mean']).pvalue if len(g) >= 6 else np.nan
    return dict(diff=S.sum() / N.sum(), ci_lo=lo, ci_hi=hi, n_users=len(g), pct_first_better=100 * (g['mean'] < 0).mean(), p=pv)


def tol_f1(v, flag, k):
    """Precision over alerts (a peak within k days after the alert), recall over peaks (an alert within k days before)."""
    tp_p = nf = tp_r = npk = 0
    for _, g in v.groupby('uid'):
        d = g['date'].values; fd = d[g[flag].values == 1]; pd_ = d[g['target'].values == 1]
        kk = np.timedelta64(k, 'D')
        nf += len(fd); tp_p += sum(np.any((pd_ >= t) & (pd_ <= t + kk)) for t in fd)
        npk += len(pd_); tp_r += sum(np.any((fd >= t - kk) & (fd <= t)) for t in pd_)
    pr = tp_p / nf if nf else np.nan; rc = tp_r / npk if npk else np.nan
    return 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0, pr, rc


def perm_f1(v, flag, k):
    out = []
    for _ in range(N_PERM):
        w = v.copy(); w[flag] = w.groupby('uid')[flag].transform(lambda s: rng.permutation(s.values))
        out.append(tol_f1(w, flag, k)[0])
    return np.mean(out), np.percentile(out, 95)


PRED = [('y_pred_ens', 'Full system'), ('y_pred_lstm', 'Personalized LSTM'), ('fitbit_raw', 'Fitbit Stress Score (raw)'),
        ('fitbit_cal', 'Fitbit Stress Score (calibrated)'), ('dummy_pop', 'Population mean'),
        ('pm_sparse', 'Personal mean (sparse)'), ('last_sparse', 'Last EMA (sparse)'), ('lag1_dense', 'Previous-day self-report (dense)')]
TESTS = [('fitbit_raw', 'MAE'), ('fitbit_cal', 'MAE'), ('dummy_pop', 'MAE'), ('pm_sparse', 'MAE'), ('last_sparse', 'MAE'), ('y_pred_lstm', 'MAE'), ('lag1_dense', 'MAE'),
         ('fitbit_raw', 'MAE_peak'), ('fitbit_cal', 'MAE_peak'), ('pm_sparse', 'MAE_peak'), ('y_pred_lstm', 'MAE_peak')]

rows_m, rows_t, rows_d = [], [], []
for f in a.preds:
    tag = os.path.basename(os.path.dirname(os.path.abspath(f))) + '__' + os.path.splitext(os.path.basename(f))[0]
    d = pd.read_csv(f, parse_dates=['date']); d['date'] = d['date'].astype('datetime64[ns]')
    d = d.merge(cal.drop(columns=['ema_collected']), on=['uid', 'date'], how='left')
    m = d['stress_raw'].notna()
    assert np.allclose(d.loc[m, 'y_true'], d.loc[m, 'stress_raw'], atol=1e-4), 'y_true differs from stress_raw'
    d['fitbit_cal'] = np.nan; d['dummy_pop'] = np.nan
    test_uids = set(d['uid'])
    for fold, g in d.groupby('fold'):
        tr = pairs[pairs['date'] < g['date'].min()] if a.schema == 'WF' else pairs[pairs['uid'].isin(test_uids - set(g['uid']))]
        if len(tr) < 30:
            continue
        idx = g.index[g['STRESS_SCORE'].notna()]
        reg = LinearRegression().fit(tr[['STRESS_SCORE']], tr['stress_raw'])
        d.loc[idx, 'fitbit_cal'] = reg.predict(d.loc[idx, ['STRESS_SCORE']])
        d.loc[g.index, 'dummy_pop'] = tr['stress_raw'].mean()
    v = d.dropna(subset=[c for c, _ in PRED] + ['y_true', 'is_peak']).copy()
    print(f'\n=== {a.schema} {tag}: predictions {len(d)}  common rows {len(v)}  users {v.uid.nunique()}  peaks {int((v.is_peak == 1).sum())}')
    for c, lab in PRED:
        r = metrics(v, c); rows_m.append(dict(file=tag, predictor=lab, **r))
        print(f'  {lab:34s} MAE {r["MAE"]:6.2f}  MAE_pk {r["MAE_peak"]:6.2f}  rho {r["rho_pooled"]:6.3f}  CCC {r["CCC"]:6.3f}  rho_u {r["rho_user"]:6.3f} ({r["n_users_rho"]})')
    for c, met in TESTS:
        t = paired(v if met == 'MAE' else v[v['is_peak'] == 1], 'y_pred_ens', c, min_n=1 if met == 'MAE' else 3)
        rows_t.append(dict(file=tag, comparison=f'Full system - {dict(PRED)[c]}', metric=met, **t))
        print(f'  {met:8s} system - {dict(PRED)[c]:32s} {t["diff"]:+6.2f} CI95 [{t["ci_lo"]:+6.2f},{t["ci_hi"]:+6.2f}] users {t["n_users"]:2d} system better {t["pct_first_better"]:3.0f}%  p={t["p"]:.2g}')
    re, rf = user_rho(v, 'y_pred_ens'), user_rho(v, 'fitbit_raw'); cu = re.index.intersection(rf.index)
    p_rr = wilcoxon(re[cu] - rf[cu]).pvalue; p_r0 = wilcoxon(re).pvalue
    rows_t.append(dict(file=tag, comparison='rho per-user: Full system - Fitbit raw', metric='rho_user', diff=(re[cu] - rf[cu]).mean(), n_users=len(cu), p=p_rr))
    rows_t.append(dict(file=tag, comparison='rho per-user: Full system vs 0', metric='rho_user', diff=re.mean(), n_users=len(re), p=p_r0))
    print(f'  rho per-user system - Fitbit raw {(re[cu] - rf[cu]).mean():+.3f} (users {len(cu)}, p={p_rr:.2g});  system vs 0: p={p_r0:.2g}')
    # detection: flag of row D against the peak of row D
    w = v.assign(target=v['is_peak']).dropna(subset=['fitbit_peak_flag', 'flag_active', 'prob_peak'])
    print(f'  detection: rows {len(w)}  peaks {int(w.target.sum())}')
    for sc, sl in [('prob_peak', 'GBM probability'), ('fitbit_raw', 'Fitbit raw'), ('y_pred_lstm', 'LSTM forecast'), ('last_sparse', 'Last EMA (sparse)')]:
        auc = roc_auc_score(w['target'], w[sc]); rows_d.append(dict(file=tag, kind='AUC', detector=sl, value=auc))
        print(f'    AUC {sl:20s} {auc:.3f}')
    for fl, fl_lab in [('flag_active', 'Full system flag'), ('fitbit_peak_flag', 'Fitbit flag')]:
        rate = w[fl].mean()
        for k in range(4):
            f1, pr, rc = tol_f1(w, fl, k); ch, ch95 = perm_f1(w, fl, k)
            rows_d.append(dict(file=tag, kind=f'F1_k{k}', detector=fl_lab, value=f1, precision=pr, recall=rc, flag_rate=rate, chance_f1=ch, chance_f1_p95=ch95))
            print(f'    {fl_lab:17s} k={k} F1 {f1:.3f} (P {pr:.3f} R {rc:.3f})  random {ch:.3f} [p95 {ch95:.3f}]  flag rate {rate:.3f}')
    v.to_csv(os.path.join(a.out, f'rq4_rows_{a.schema}_{tag}.csv'), index=False)

for X, n in [(rows_m, 'metrics'), (rows_t, 'tests'), (rows_d, 'detection')]:
    pd.DataFrame(X).to_csv(os.path.join(a.out, f'rq4_{n}_{a.schema}.csv'), index=False)
print(f'\nSaved in {a.out}/')
