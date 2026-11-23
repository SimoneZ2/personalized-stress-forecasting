"""SSAQS (RQ4): next-day stress forecast with the personalized LSTM, the GBM peak classifier and the correction,
evaluated with walk-forward (WF) and leave-users-out (LUO). Same design as the CES pipeline:

- LSTM: population model on the 14 previous rows (sensing z-scores) plus sparse EMA lags; per-user fine-tuning
  on rows 6 and 13 (onboarding, on the rows accumulated so far) and then on every self-report collected by the
  policy (all in the first 14 rows, then one every 14 rows).
- GBM: probability that the row is a peak, computed from the features of the previous row of the same user.
- Correction: adaptive flag on the GBM probability; when active, the LSTM forecast is moved towards
  personal_mean + tgt_std * personal_std (sparse personal stats), weighted by the flag value.
- WF: expanding cutoffs every 14 calendar days after the first 70; LSTM and GBM trained on rows before the
  cutoff, flag with the CES defaults, tgt_std personal with shrinkage towards the population value.
- LUO: 5 user folds; flag parameters calibrated on the training users (minimum MAE_peak, at most 45% flagged).

WF and LUO run in this order in one process: the LUO results depend on the random state left by WF.
Output in --out: ensemble_forecast_wf_adaptive_preds.csv, ensemble_forecast_luo_adaptive_preds.csv.

Usage: python3 run_ensemble.py --lstm-input ssaqs_lstm_input_ces.csv --gbm-input ssaqs_gbm_input_ces.csv --seed 42 --out runs/seed42
"""
import argparse
import copy
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

# LSTM
SEQ_LEN, HIDDEN_SIZE, DROPOUT = 14, 64, 0.3
LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, BATCH_SIZE, HUBER_DELTA = 1e-3, 1e-4, 100, 15, 64, 1.0
FT_LR, FT_STEPS = 5e-3, 10
ONBOARDING, INTERVAL = 14, 14
ONBOARDING_FT_ROWS = (6, 13)

# GBM
GBM_PARAMS = dict(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=20)
GBM_FEATURES = ['ema_lag2', 'ema_z', 'ema_trend', 'n_ema_available',
                'point_act_sedentary_frac', 'point_act_active_frac', 'point_steps_total', 'dow_sin',
                'point_deep_sleep_min', 'point_spo2_mean', 'point_hrv_lf_hf_ratio',
                'trend_deep_sleep_min', 'trend_hrv_rmssd_mean']

# Flag and correction
DECAY, THR_MIN, THR_MAX, ALPHA, FLAG_WINDOW = 0.75 ** (1 / 7), 0.20, 0.50, 1.0, 28
PEAK_K, SHRINK, TGT_DEFAULT, PS_MIN = 1.0, 5.0, 1.877, 0.25
GRID_DECAY = [round(x ** (1 / 7), 4) for x in (0.5, 0.75, 0.9)]
GRID_THR_MIN, GRID_THR_MAX, GRID_ALPHA = (0.20, 0.30, 0.40), (0.50, 0.60, 0.70), (0.5, 0.75, 1.0)
MAX_FLAGGED, MIN_OBS_USER = 0.45, 20

# Validation
K_FOLDS, WF_MIN_TRAIN, WF_STEP = 5, 70, 14

SENSING = [f'{f}_z' for f in ['hrv_rmssd_mean', 'hrv_rmssd_std', 'hrv_lf_mean', 'hrv_hf_mean', 'hrv_lf_hf_ratio',
                              'sleep_score', 'deep_sleep_min', 'spo2_mean', 'spo2_std',
                              'steps_total', 'steps_std', 'act_active_frac', 'act_sedentary_frac']]
EMA = ['ema_lag1', 'ema_lag2', 'ema_lag3', 'ema_trend']
DEVICE = torch.device('cpu')


class StressLSTM(nn.Module):
    def __init__(self, n_sensing, n_ema):
        super().__init__()
        self.lstm = nn.LSTM(n_sensing, HIDDEN_SIZE, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE + n_ema, 1)

    def forward(self, x_seq, x_ema):
        out, _ = self.lstm(x_seq)
        return self.fc(torch.cat([self.drop(out[:, -1, :]), x_ema], dim=1)).squeeze(-1)


def build_sequences(df):
    """Per user: input = 14 previous rows of sensing + EMA features of the row, target = self-report of the row."""
    seqs = {}
    for uid, g in df.groupby('uid'):
        g = g.sort_values('date').reset_index(drop=True)
        S = g[SENSING].fillna(0.).values.astype(np.float32)
        E = g[EMA].fillna(0.).values.astype(np.float32)
        T = g['stress_raw'].fillna(g['stress_raw'].mean()).values.astype(np.float32)
        n = len(g)
        if n < SEQ_LEN + 1:
            continue
        rows = range(SEQ_LEN, n)
        seqs[uid] = dict(
            onb=tuple((S[:i].copy(), E[i].copy(), float(T[i] / 100.)) for i in ONBOARDING_FT_ROWS if 1 <= i < n),
            X=np.array([S[i - SEQ_LEN:i] for i in rows], np.float32), E=np.array([E[i] for i in rows], np.float32),
            y=np.array([T[i] / 100. for i in rows], np.float32),
            dates=np.array([pd.Timestamp(g['date'].iloc[i]).date() for i in rows]),
            is_peak=g['is_peak'].values.astype(np.float32)[SEQ_LEN:],
            personal_mean=g['personal_mean'].values.astype(np.float32)[SEQ_LEN:],
            personal_std=g['personal_std'].fillna(1.).values.astype(np.float32)[SEQ_LEN:])
    return seqs


def truncate_sequences(seqs, uids, cutoff):
    """Training samples with target date before the cutoff (causal WF training)."""
    c = pd.Timestamp(cutoff).date()
    out = {}
    for u in uids:
        if u not in seqs:
            continue
        d = seqs[u]
        m = np.array([pd.Timestamp(x).date() < c for x in d['dates']], dtype=bool)
        if m.any():
            out[u] = {k: (np.asarray(v)[m] if isinstance(v, (list, np.ndarray)) and len(v) == len(m) else v) for k, v in d.items()}
    return out


def train_lstm(seqs, uids):
    model = StressLSTM(len(SENSING), len(EMA)).to(DEVICE)
    crit = nn.HuberLoss(delta=HUBER_DELTA)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    used = [u for u in uids if u in seqs]
    if not used:
        return model
    X = np.concatenate([seqs[u]['X'] for u in used]); E = np.concatenate([seqs[u]['E'] for u in used])
    y = np.concatenate([seqs[u]['y'] for u in used])
    idx = np.random.permutation(len(y)); n_tr = int(.8 * len(y))
    tr, va = idx[:n_tr], idx[n_tr:]
    t = lambda a, i: torch.from_numpy(a[i]).to(DEVICE)
    best, wait, best_state = float('inf'), 0, copy.deepcopy(model.state_dict())
    for _ in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = crit(model(t(X, tr[b]), t(E, tr[b])), t(y, tr[b]))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(t(X, va), t(E, va)), t(y, va)).item()
        if vl < best:
            best, wait, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    return model


def predict(model, x_seq, x_ema):
    model.eval()
    with torch.no_grad():
        return float(model(torch.from_numpy(x_seq[np.newaxis]).to(DEVICE), torch.from_numpy(x_ema[np.newaxis]).to(DEVICE)).cpu().item() * 100.)


def finetune(model, x_seq, x_ema, y_scaled):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=FT_LR)
    crit = nn.HuberLoss(delta=HUBER_DELTA)
    xt, et = torch.from_numpy(x_seq[np.newaxis]).to(DEVICE), torch.from_numpy(x_ema[np.newaxis]).to(DEVICE)
    yt = torch.tensor([y_scaled], dtype=torch.float32).to(DEVICE)
    for _ in range(FT_STEPS):
        opt.zero_grad(); loss = crit(model(xt, et), yt)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
    model.eval()
    return model


def fit_gbm(X, y, seed):
    """Class-balanced GBM; None when there are too few rows or peaks."""
    if len(X) < 20 or y.sum() < 3 or (1 - y).sum() < 3:
        return None
    w = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (1 - y).sum()))
    return GradientBoostingClassifier(**GBM_PARAMS, random_state=seed).fit(X, y, sample_weight=w)


def gbm_matrix(df, cols, med):
    return np.nan_to_num(df[cols].fillna(med).values.astype(float))


def next_row_date(df):
    """Date of the next row of the same user (df sorted by uid, date): the GBM row t-1 serves row t."""
    return pd.to_datetime(df.groupby('uid')['date'].shift(-1))


class AdaptiveFlag:
    """Decaying peak probability, active above a threshold that grows with the recent flag activity."""
    def __init__(self, decay=DECAY, thr_min=THR_MIN, thr_max=THR_MAX, w=FLAG_WINDOW):
        self.decay, self.thr_min, self.thr_max, self.w = decay, thr_min, thr_max, w
        self.value, self.hist = 0., []

    def update(self, prob):
        self.value = max(prob, self.value * self.decay)
        self.hist.append(self.value)
        recent = self.hist[-self.w:]
        activity = sum(1 for v in recent if v > self.thr_min) / len(recent)
        return self.value, self.value > self.thr_min + (self.thr_max - self.thr_min) * activity


def peak_deviations(s):
    """Standardized deviations >= PEAK_K of each value from the previous 12 (at least 4), as in CES."""
    devs = []
    for i in range(12, len(s)):
        hist = s[max(0, i - 12):i]; hist = hist[~np.isnan(hist)]
        if len(hist) < 4:
            continue
        mu, sig = np.mean(hist), np.std(hist)
        if sig < PS_MIN:
            continue
        dev = (s[i] - mu) / sig
        if dev >= PEAK_K:
            devs.append(dev)
    return devs


def tgt_std_population(df_lstm, train_uids, cutoff=None):
    """Mean standardized deviation of the labelled peaks of the training users (dense stats of the label)."""
    d = df_lstm[df_lstm['uid'].isin(train_uids)]
    if cutoff is not None:
        d = d[pd.to_datetime(d['date']) < pd.Timestamp(cutoff)]
    d = d[(d['is_peak'] == 1) & d['personal_mean_dense'].notna() & (d['personal_std_dense'] > PS_MIN)]
    devs = ((d['stress_raw'] - d['personal_mean_dense']) / d['personal_std_dense']).values
    return float(np.mean(devs)) if len(devs) >= 50 else TGT_DEFAULT


def tgt_std_personal(df_lstm, uid, cutoff, pop):
    """WF: peak deviations of the self-reports collected before the cutoff, shrunk towards the population value."""
    g = df_lstm[(df_lstm['uid'] == uid) & (pd.to_datetime(df_lstm['date']) < pd.Timestamp(cutoff))].sort_values('date')
    devs = peak_deviations(g.loc[g['ema_collected'] == 1, 'stress_raw'].dropna().values.astype(float))
    if not devs:
        return pop
    w = len(devs) / (len(devs) + SHRINK)
    return w * float(np.mean(devs)) + (1 - w) * pop


def calibrate_flag_luo(df_lstm, gbm_train, gbm_model, cols, med, train_uids, tgt):
    """LUO: flag parameters with the lowest MAE_peak on the training users, at most 45% of days flagged.
    The correction is simulated on the personal mean (dense) as baseline."""
    P = gbm_train[['uid', 'date']].copy()
    P['p'] = gbm_model.predict_proba(gbm_matrix(gbm_train, cols, med))[:, 1]
    P = P.sort_values(['uid', 'date']); P['date'] = next_row_date(P); P = P.dropna(subset=['date'])
    L = df_lstm[df_lstm['uid'].isin(train_uids)].copy(); L['date'] = pd.to_datetime(L['date'])
    L = L.merge(P, on=['uid', 'date'], how='left')
    users = []
    for _, g in L.sort_values(['uid', 'date']).groupby('uid'):
        g = g.dropna(subset=['stress_raw'])
        if len(g) < MIN_OBS_USER:
            continue
        pm, y = g['personal_mean_dense'].values.astype(float), g['stress_raw'].values.astype(float)
        users.append(dict(p=g['p'].fillna(0.).values, y=y, pm=pm, ps=g['personal_std_dense'].values.astype(float),
                          pk=(g['is_peak'].values == 1), yl=np.where(np.isnan(pm), np.nanmean(y), pm)))
    best = None
    for decay in GRID_DECAY:
        for tmin in GRID_THR_MIN:
            for tmax in GRID_THR_MAX:
                for alpha in GRID_ALPHA:
                    err_pk, n_flag, n_tot = [], 0, 0
                    for u in users:
                        fl = AdaptiveFlag(decay=decay, thr_min=tmin, thr_max=tmax)
                        yp = np.empty(len(u['y'])); fg = np.zeros(len(u['y']), dtype=bool)
                        for i in range(len(yp)):
                            fv, act = fl.update(u['p'][i]); fg[i] = act; c = 0.
                            if act and not np.isnan(u['ps'][i]) and u['ps'][i] > PS_MIN:
                                c = alpha * (u['pm'][i] + tgt * u['ps'][i] - u['yl'][i]) * fv
                            yp[i] = np.clip(u['yl'][i] + c, 0., 100.)
                        err_pk += list(np.abs(u['y'] - yp)[u['pk']]); n_flag += int(fg.sum()); n_tot += len(fg)
                    if n_tot == 0 or not err_pk or n_flag / n_tot > MAX_FLAGGED:
                        continue
                    if best is None or np.mean(err_pk) < best['mae_peak']:
                        best = dict(decay=decay, thr_min=tmin, thr_max=tmax, alpha=alpha, mae_peak=float(np.mean(err_pk)),
                                    pct_flagged=n_flag / n_tot)
    print(f'    flag calibration: {best if best else "no combination within 45%, defaults"}')
    return best or {}


def simulate_user(uid, seq, gbm_rows, lstm_model, tgt_std, flag_params, test_dates=None):
    """Day-by-day forecast for one user: predict, correct, then fine-tune when the policy collects the answer."""
    model = copy.deepcopy(lstm_model); model.eval()
    g = gbm_rows[gbm_rows['uid'] == uid].sort_values('date').copy()
    g['_date'] = next_row_date(g).dt.date
    g = g[g['_date'].notna()]
    prob = dict(zip(g['_date'], g['prob_peak']))
    flag = AdaptiveFlag(decay=flag_params.get('decay', DECAY), thr_min=flag_params.get('thr_min', THR_MIN),
                        thr_max=flag_params.get('thr_max', THR_MAX))
    alpha = flag_params.get('alpha', ALPHA)
    for x, e, y in seq['onb']:
        model = finetune(model, x, e, y)
    out = []
    for t in range(len(seq['y'])):
        d, pm, ps = seq['dates'][t], seq['personal_mean'][t], seq['personal_std'][t]
        y_lstm = predict(model, seq['X'][t], seq['E'][t])
        p = float(prob.get(d, 0.0))
        fv, active = flag.update(p)
        corr = alpha * fv * (pm + tgt_std * ps - y_lstm) if active and not np.isnan(pm) and ps > PS_MIN else 0.
        y_ens = float(np.clip(y_lstm + corr, 0., 100.))
        row = t + SEQ_LEN
        if row < ONBOARDING or (row - ONBOARDING) % INTERVAL == 0:
            model = finetune(model, seq['X'][t], seq['E'][t], float(seq['y'][t]))
        if test_dates is None or d in test_dates:
            out.append(dict(uid=uid, date=str(d), y_true=float(seq['y'][t] * 100.), y_pred_lstm=y_lstm, y_pred_ens=y_ens,
                            prob_peak=p, flag_active=int(active), is_peak=float(seq['is_peak'][t])))
    return out


def run_wf(df_lstm, df_gbm, seqs, seed):
    dates = sorted(df_lstm['date'].unique())
    rows, fold = [], 0
    for ci in range(WF_MIN_TRAIN, len(dates) - WF_STEP + 1, WF_STEP):
        cutoff, test_end = dates[ci], dates[min(ci + WF_STEP - 1, len(dates) - 1)]
        tr_uids = [u for u, d in df_lstm.groupby('uid') if (d['date'] < cutoff).any()]
        te_uids = [u for u, d in df_lstm.groupby('uid')
                   if ((d['date'] >= cutoff) & (d['date'] <= test_end)).any() and (d['date'] < cutoff).any()]
        if len(tr_uids) < 3 or not te_uids:
            continue
        print(f'  WF {fold + 1}  cutoff={pd.Timestamp(cutoff).date()}  train={len(tr_uids)}  test={len(te_uids)}')
        lstm_model = train_lstm(truncate_sequences(seqs, tr_uids, cutoff), tr_uids)
        gbm_train = df_gbm[df_gbm['uid'].isin(tr_uids) & (df_gbm['date'] < cutoff)]
        cols = [f for f in GBM_FEATURES if f in gbm_train.columns]
        med = gbm_train[cols].median()
        lab = gbm_train['is_peak_next'].notna()
        gbm_model = fit_gbm(gbm_matrix(gbm_train, cols, med)[lab], gbm_train.loc[lab, 'is_peak_next'].values.astype(int), seed)
        if gbm_model is None:
            print('    GBM: not enough data, fold skipped')
            continue
        tgt_pop = tgt_std_population(df_lstm, tr_uids, cutoff)
        gbm_rows = df_gbm[df_gbm['uid'].isin(te_uids)].copy()
        gbm_rows['prob_peak'] = gbm_model.predict_proba(gbm_matrix(gbm_rows, cols, med))[:, 1]
        for uid in te_uids:
            if uid not in seqs:
                continue
            u = df_lstm[df_lstm['uid'] == uid]
            test_dates = set(pd.Timestamp(d).date() for d in u.loc[(u['date'] >= cutoff) & (u['date'] <= test_end), 'date'].values)
            r = simulate_user(uid, seqs[uid], gbm_rows, lstm_model, tgt_std_personal(df_lstm, uid, cutoff, tgt_pop), {}, test_dates)
            rows += [dict(x, fold=fold) for x in r]
        fold += 1
    return rows


def run_luo(df_lstm, df_gbm, seqs, seed):
    uids = np.array(list(seqs.keys()))
    rows = []
    for fi, (tr_idx, te_idx) in enumerate(KFold(n_splits=K_FOLDS, shuffle=True, random_state=42).split(uids)):
        tr_uids, te_uids = list(uids[tr_idx]), list(uids[te_idx])
        print(f'  LUO fold {fi + 1}/{K_FOLDS}  train={len(tr_uids)}  test={len(te_uids)}')
        lstm_model = train_lstm(seqs, tr_uids)
        gbm_train = df_gbm[df_gbm['uid'].isin(tr_uids)]
        cols = [f for f in GBM_FEATURES if f in gbm_train.columns]
        med = gbm_train[cols].median()
        lab = gbm_train['is_peak_next'].notna().values
        gbm_model = fit_gbm(gbm_matrix(gbm_train, cols, med)[lab], gbm_train['is_peak_next'].values[lab].astype(int), seed)
        if gbm_model is None:
            print('    GBM: not enough data, fold skipped')
            continue
        tgt = tgt_std_population(df_lstm, tr_uids)
        flag_params = calibrate_flag_luo(df_lstm, gbm_train, gbm_model, cols, med, tr_uids, tgt)
        gbm_rows = df_gbm[df_gbm['uid'].isin(te_uids)].copy()
        gbm_rows['prob_peak'] = gbm_model.predict_proba(gbm_matrix(gbm_rows, cols, med))[:, 1]
        for uid in te_uids:
            if uid in seqs:
                rows += [dict(x, fold=fi) for x in simulate_user(uid, seqs[uid], gbm_rows, lstm_model, tgt, flag_params)]
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lstm-input', required=True)
    p.add_argument('--gbm-input', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    df_lstm = pd.read_csv(a.lstm_input, parse_dates=['date']).sort_values(['uid', 'date']).reset_index(drop=True)
    df_gbm = pd.read_csv(a.gbm_input, parse_dates=['date']).sort_values(['uid', 'date']).reset_index(drop=True)
    seqs = build_sequences(df_lstm)
    print(f'LSTM rows {len(df_lstm)}, GBM rows {len(df_gbm)}, users with sequences {len(seqs)}, seed {a.seed}')
    for name, run in [('wf', run_wf), ('luo', run_luo)]:
        rows = pd.DataFrame(run(df_lstm, df_gbm, seqs, a.seed))
        f = os.path.join(a.out, f'ensemble_forecast_{name}_adaptive_preds.csv')
        rows.to_csv(f, index=False)
        print(f'{name.upper()}: {len(rows)} predictions, {rows.uid.nunique()} users -> {f}')


if __name__ == '__main__':
    main()
