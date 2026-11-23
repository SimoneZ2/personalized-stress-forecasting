"""RQ4 figure: daily population mean of self-reported stress, full system and Fitbit Stress Score (100 - score of
day D-1) on the comparison rows of the table (leave-users-out, seed 42), every day with at least one user.
Missing days break the lines; the legend reports the correlation of each series with the mean self-report.

Output in --out: rq4_pop_calendar_raw_luo.{png,pdf}.
Usage: python3 rq4_figure.py --rows rq4_eval/rq4_rows_LUO_seed42__ensemble_forecast_luo_adaptive_preds.csv --out rq4_figs
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_USERS = 1
COL = {'y_true': ('black', '-', 'Self-reported stress (EMA)'), 'y_pred_ens': ('#1f77b4', '--', 'Full system'),
       'fitbit_raw': ('#ff7f0e', ':', 'Fitbit Stress Score (100 - score, day D-1)')}

p = argparse.ArgumentParser()
p.add_argument('--rows', required=True)
p.add_argument('--out', default='rq4_figs')
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'savefig.dpi': 300})


def corr(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    return np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 2 else np.nan


v = pd.read_csv(a.rows, parse_dates=['date'])
cols = list(COL)
agg = v.groupby('date').agg(n=('uid', 'nunique'), **{f'{c}_m': (c, 'mean') for c in cols},
                            **{f'{c}_s': (c, 'std') for c in cols}).reset_index()
agg = agg[agg['n'] >= MIN_USERS]
full = pd.date_range(agg['date'].min(), agg['date'].max(), freq='D')
agg = agg.set_index('date').reindex(full).rename_axis('date').reset_index()
fig, ax = plt.subplots(figsize=(11, 4.2))
for c in cols:
    colr, ls, lab = COL[c]
    r = corr(agg['y_true_m'].values, agg[f'{c}_m'].values)
    ax.plot(agg['date'], agg[f'{c}_m'], color=colr, ls=ls, lw=1.6, label=lab + ('' if c == 'y_true' else f'  (r = {r:.2f})'))
ax.set_ylabel('Stress (0-100)')
ax.set_ylim(0, 100)
ax.legend(loc='upper left', fontsize=8, frameon=False)
ax.set_title('Daily population mean - SSAQS, leave-users-out', fontsize=11)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(os.path.join(a.out, f'rq4_pop_calendar_raw_luo.{ext}'), bbox_inches='tight')
plt.close(fig)
print(f'rq4_pop_calendar_raw_luo: {int(agg.n.notna().sum())} days with >= {MIN_USERS} users, median users per day {agg.n.median():.0f}; '
      f'r with the mean self-report: system {corr(agg.y_true_m.values, agg.y_pred_ens_m.values):.2f}, '
      f'Fitbit {corr(agg.y_true_m.values, agg.fitbit_raw_m.values):.2f}')
