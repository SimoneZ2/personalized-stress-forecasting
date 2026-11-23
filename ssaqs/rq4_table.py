"""RQ4 table (SSAQS, next-day forecast) from the files written by evaluate.py for WF and LUO. Trains nothing.

Adds, from the rq4_rows files of each seed, two tests that evaluate.py does not compute:
  - difference of pooled CCC (full system - Fitbit raw), 95% bootstrap CI over users
  - difference of peak AUC (GBM probability - Fitbit raw), 95% bootstrap CI over users
Reported values: mean +- sd over seeds; for the tests, mean difference over seeds, CI of seed 42 (main seed,
fixed before the results) and number of seeds where the test is significant.

Output in --dir: rq4_table.md and rq4_table_{regression,tests,detection}_{WF,LUO}.csv.
Usage: python3 rq4_table.py --dir rq4_eval
"""
import argparse
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOT = 2000
p = argparse.ArgumentParser()
p.add_argument('--dir', required=True)
a = p.parse_args()
rng = np.random.default_rng(0)
ORDER = ['Full system', 'Personalized LSTM', 'Fitbit Stress Score (raw)', 'Fitbit Stress Score (calibrated)',
         'Population mean', 'Personal mean (sparse)', 'Last EMA (sparse)', 'Previous-day self-report (dense)']


def ccc(x, y):
    c = ((x - x.mean()) * (y - y.mean())).mean(); d = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return 2 * c / d if d > 0 else np.nan


def boot_users(v, stat):
    groups = [g for _, g in v.groupby('uid')]
    obs = stat(v); bs = []
    for _ in range(N_BOOT):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        try:
            bs.append(stat(s))
        except ValueError:          # resample without peaks or without non-peaks
            continue
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return obs, lo, hi


def fmt(m, s, d=2):
    return f'{m:.{d}f} ± {s:.{d}f}' if s > 0 else f'{m:.{d}f}'


def first_ci(lo, hi):
    """CI of the first seed (42), the main seed declared before the results."""
    return lo[0], hi[0]


md = ['# RQ4 - SSAQS, next-day forecast\n']
for schema in ['WF', 'LUO']:
    M = pd.read_csv(f'{a.dir}/rq4_metrics_{schema}.csv')
    T = pd.read_csv(f'{a.dir}/rq4_tests_{schema}.csv')
    D = pd.read_csv(f'{a.dir}/rq4_detection_{schema}.csv')
    seeds = list(M['file'].unique())
    rows_files = sorted(glob.glob(f'{a.dir}/rq4_rows_{schema}_*.csv'))
    assert len(rows_files) == len(seeds), f'{schema}: {len(rows_files)} file rows contro {len(seeds)} seed'
    n_rows = pd.read_csv(rows_files[0]).shape[0]

    # ---- regression ----
    g = M.groupby('predictor')[['MAE', 'MAE_peak', 'CCC', 'rho_user']].agg(['mean', 'std'])
    reg = pd.DataFrame({k: [fmt(g.loc[pr, (k, 'mean')], g.loc[pr, (k, 'std')], 2 if 'MAE' in k else 3) for pr in ORDER]
                        for k in ['MAE', 'MAE_peak', 'CCC', 'rho_user']}, index=ORDER)
    reg.loc['Population mean', 'rho_user'] = 'n/a'          # constant prediction per user
    reg.to_csv(f'{a.dir}/rq4_table_regression_{schema}.csv')

    # ---- tests: full system vs Fitbit raw and baselines ----
    tests = []
    for comp, met in [('Full system - Fitbit Stress Score (raw)', 'MAE'), ('Full system - Fitbit Stress Score (raw)', 'MAE_peak'),
                      ('rho per-user: Full system - Fitbit raw', 'rho_user'),
                      ('Full system - Fitbit Stress Score (calibrated)', 'MAE'), ('Full system - Fitbit Stress Score (calibrated)', 'MAE_peak'),
                      ('Full system - Personal mean (sparse)', 'MAE'), ('Full system - Last EMA (sparse)', 'MAE'),
                      ('Full system - Personalized LSTM', 'MAE'), ('Full system - Personalized LSTM', 'MAE_peak')]:
        t = T[(T['comparison'] == comp) & (T['metric'] == met)]
        lo, hi = (first_ci(t['ci_lo'].values, t['ci_hi'].values) if 'ci_lo' in t and t['ci_lo'].notna().all() else (np.nan, np.nan))
        tests.append(dict(comparison=comp.replace('rho per-user: ', ''), metric=met, diff=t['diff'].mean(), ci_lo=lo, ci_hi=hi,
                          p_min=t['p'].min(), p_max=t['p'].max(), sig_seeds=f"{int((t['p'] < 0.05).sum())}/{len(t)}",
                          n_users=int(t['n_users'].iloc[0])))
    ccc_r, auc_r = [], []
    for f in rows_files:
        v = pd.read_csv(f)
        ccc_r.append(boot_users(v, lambda s: ccc(s['y_true'].values, s['y_pred_ens'].values) - ccc(s['y_true'].values, s['fitbit_raw'].values)))
        w = v.dropna(subset=['prob_peak', 'is_peak'])
        auc_r.append(boot_users(w, lambda s: roc_auc_score(s['is_peak'], s['prob_peak']) - roc_auc_score(s['is_peak'], s['fitbit_raw'])))
    for lab, met, r in [('Full system - Fitbit Stress Score (raw)', 'CCC', ccc_r), ('GBM probability - Fitbit Stress Score (raw)', 'AUC', auc_r)]:
        r = np.array(r); lo, hi = first_ci(r[:, 1], r[:, 2])
        tests.append(dict(comparison=lab, metric=met, diff=r[:, 0].mean(), ci_lo=lo, ci_hi=hi, p_min=np.nan, p_max=np.nan,
                          sig_seeds=f'{int(((r[:, 1] > 0) | (r[:, 2] < 0)).sum())}/{len(r)} (CI excl. 0)',
                          n_users=pd.read_csv(rows_files[0])['uid'].nunique()))
    tests = pd.DataFrame(tests); tests.to_csv(f'{a.dir}/rq4_table_tests_{schema}.csv', index=False)

    # ---- detection ----
    det = []
    auc = D[D['kind'] == 'AUC'].groupby('detector')['value'].agg(['mean', 'std'])
    for dl in ['GBM probability', 'Fitbit raw', 'LSTM forecast', 'Last EMA (sparse)']:
        det.append(dict(detector=dl, metric='AUC', value=fmt(auc.loc[dl, 'mean'], auc.loc[dl, 'std'], 3), chance='0.500', flag_rate=''))
    for dl in ['Full system flag', 'Fitbit flag']:
        for k in range(4):
            x = D[(D['kind'] == f'F1_k{k}') & (D['detector'] == dl)]
            above = int((x['value'] > x['chance_f1_p95']).sum())
            det.append(dict(detector=dl, metric=f'F1 (k={k})', value=fmt(x['value'].mean(), x['value'].std(), 3),
                            chance=f"{x['chance_f1'].mean():.3f} (p95 {x['chance_f1_p95'].mean():.3f}; above p95 in {above}/{len(x)})",
                            flag_rate=f"{x['flag_rate'].mean():.3f}"))
    det = pd.DataFrame(det); det.to_csv(f'{a.dir}/rq4_table_detection_{schema}.csv', index=False)

    # ---- markdown ----
    md.append(f'\n## {schema} - {n_rows} days, {pd.read_csv(rows_files[0])["uid"].nunique()} users, {len(seeds)} seeds (mean ± sd)\n')
    md.append('| Predictor | MAE | MAE_peak | CCC | per-user ρ |\n|---|---|---|---|---|')
    md += [f'| {pr} | ' + ' | '.join(reg.loc[pr]) + ' |' for pr in ORDER]
    md.append('\n| Comparison | Metric | Δ (mean) | 95% CI (seed 42) | Wilcoxon p (range) | Significant |\n|---|---|---|---|---|---|')
    for _, t in tests.iterrows():
        pr = '' if np.isnan(t['p_min']) else f"{t['p_min']:.2g} - {t['p_max']:.2g}"
        ci = '' if np.isnan(t['ci_lo']) else f"[{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}]" if t['metric'] in ('CCC', 'AUC', 'rho_user') else f"[{t['ci_lo']:+.2f}, {t['ci_hi']:+.2f}]"
        dd = f"{t['diff']:+.3f}" if t['metric'] in ('CCC', 'AUC', 'rho_user') else f"{t['diff']:+.2f}"
        md.append(f"| {t['comparison']} | {t['metric']} | {dd} | {ci} | {pr} | {t['sig_seeds']} |")
    md.append('\n| Detector | Metric | Value | Chance level | Flag rate |\n|---|---|---|---|---|')
    md += [f"| {r.detector} | {r.metric} | {r.value} | {r.chance} | {r.flag_rate} |" for r in det.itertuples()]

open(f'{a.dir}/rq4_table.md', 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
print(f'\nSaved: {a.dir}/rq4_table.md and rq4_table_{{regression,tests,detection}}_{{WF,LUO}}.csv')
