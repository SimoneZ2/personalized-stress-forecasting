#!/usr/bin/env python3
"""
Reimplementation of I-HOPE (Roy Chowdhury et al., CHASE '25) as a per-user stress
regressor, on calendar folds (RQ3).

Stage 1: five behavioural scores (leisure, me, phone, sleep, social) from the raw sensing
of the self-report day; feature weights from a random forest and min-max scaling fitted
on the user's training weeks only.
Stage 2: per-user MLP (3 x 64 ReLU, linear output, Huber loss) on the five scores,
retrained for every (user, fold). Early stopping uses the last 20% of the user's training
weeks; with fewer than 10 training weeks it trains for 30 epochs without validation.
Target: the last stress self-report of the calendar week, clipped to [1, 5].

Folds and rows are those of the WF run used in the paper, read from ihope_rows.csv
(uid, week_start, fold): a week's cutoff is the first week of its fold, and only the
listed (user, week) pairs are used.

Predictions are appended user by user, and users already in the output are skipped, so the
run can be resumed. TensorFlow slows down as models accumulate in one process: --max-users
ends the process after N users (exit code 3) so that a shell loop can restart it. Results do
not depend on the restarts, since every model resets its seed before training.

Usage (CPU, several hours):
  while true; do python comparisons/ihope/ihope_regressor.py --sensing dataset/Sensing/sensing.csv \
      --ema dataset/EMA/general_ema.csv --rows comparisons/ihope/ihope_rows.csv --out results/ihope \
      --max-users 15; [ $? -eq 3 ] || break; done
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

STRESS_MIN, STRESS_MAX = 1.0, 5.0
MIN_SAMPLES = 20               # users with fewer weeks are excluded
MIN_TRAIN_SAMPLES = 5          # (user, fold) with fewer training weeks are skipped
MIN_ROWS_FOR_INTERNAL_VAL = 10
INTERNAL_VAL_FRACTION = 0.2
FIXED_EPOCHS_NO_VAL = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RAW_COLS = [
    "act_on_bike_ep_0", "act_on_foot_ep_0", "act_running_ep_0",
    "act_still_ep_0", "act_walking_ep_0", "audio_convo_duration_ep_0",
    "call_in_num_ep_0", "call_out_num_ep_0",
    "call_in_duration_ep_0", "call_out_duration_ep_0",
    "loc_food_audio_voice", "loc_home_audio_voice", "loc_social_audio_voice",
    "loc_other_dorm_audio_voice", "loc_self_dorm_audio_voice", "loc_study_audio_voice",
    "loc_food_convo_duration", "loc_home_convo_duration",
    "loc_other_dorm_convo_duration", "loc_social_convo_duration",
    "loc_study_convo_duration", "loc_self_dorm_convo_duration",
    "loc_home_dur", "loc_leisure_dur", "loc_other_dorm_dur",
    "loc_self_dorm_dur", "loc_social_dur", "loc_study_dur", "loc_workout_dur",
    "loc_home_unlock_num", "loc_home_unlock_duration",
    "loc_other_dorm_unlock_num", "loc_other_dorm_unlock_duration",
    "loc_self_dorm_unlock_num", "loc_self_dorm_unlock_duration",
    "loc_social_unlock_num", "loc_social_unlock_duration",
    "loc_study_unlock_num", "loc_study_unlock_duration",
    "sleep_duration", "sleep_end", "sleep_start",
    "unlock_num_ep_0", "unlock_duration_ep_0", "sleep_heathkit_dur",
]

LABEL_FEATURES = {
    "leisure": [
        "act_on_bike_ep_0", "act_on_foot_ep_0", "act_running_ep_0",
        "act_walking_ep_0", "audio_convo_duration_ep_0", "call_ratio",
        "loc_social_audio_voice", "loc_other_dorm_audio_voice",
        "loc_other_dorm_convo_duration", "loc_social_convo_duration",
        "loc_leisure_dur", "loc_workout_dur",
        "loc_home_unlock_ratio", "loc_other_dorm_unlock_ratio",
    ],
    "me": [
        "act_on_bike_ep_0", "act_on_foot_ep_0", "act_running_ep_0",
        "act_still_ep_0", "act_walking_ep_0", "loc_home_audio_voice",
        "loc_self_dorm_audio_voice", "loc_study_audio_voice",
        "loc_home_convo_duration", "loc_study_convo_duration",
        "loc_self_dorm_convo_duration", "loc_self_dorm_dur", "loc_study_dur",
        "loc_workout_dur", "loc_home_unlock_ratio",
        "loc_self_dorm_unlock_ratio", "loc_study_unlock_ratio",
    ],
    "phone": [
        "audio_convo_duration_ep_0", "call_ratio", "loc_home_audio_voice",
        "loc_social_audio_voice", "loc_self_dorm_audio_voice",
        "loc_home_convo_duration", "loc_self_dorm_convo_duration",
        "loc_home_unlock_ratio", "loc_other_dorm_unlock_ratio",
        "loc_self_dorm_unlock_ratio", "loc_social_unlock_ratio",
        "loc_study_unlock_ratio", "unlock_ratio_ep0",
    ],
    "sleep": [
        "loc_self_dorm_dur", "sleep_duration", "sleep_end_minus_start",
        "sleep_heathkit_dur",
    ],
    "social": [
        "act_on_foot_ep_0", "act_walking_ep_0", "audio_convo_duration_ep_0",
        "loc_food_audio_voice", "loc_social_audio_voice",
        "loc_other_dorm_audio_voice", "loc_study_audio_voice",
        "loc_food_convo_duration", "loc_other_dorm_convo_duration",
        "loc_social_convo_duration", "loc_study_convo_duration",
        "loc_leisure_dur", "loc_other_dorm_dur", "loc_social_dur",
        "loc_other_dorm_unlock_ratio", "loc_social_unlock_ratio",
    ],
}
LABELS = list(LABEL_FEATURES)


def safe_division(numerator, denominator):
    return (numerator / denominator.replace(0, np.nan)).fillna(0)


def load_sensing(path):
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["day"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "uid"])
    present = [c for c in RAW_COLS if c in df.columns]
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        log.warning("sensing columns not found (ignored): %s", missing)
    data = df[["uid", "date"] + present].copy()
    data[present] = data[present].fillna(0)

    def g(col):
        return data[col] if col in data.columns else pd.Series(0.0, index=data.index)

    data["call_ratio"] = safe_division(g("call_in_num_ep_0") + g("call_out_num_ep_0"),
                                       g("call_in_duration_ep_0") + g("call_out_duration_ep_0"))
    for place in ["home", "other_dorm", "self_dorm", "social", "study"]:
        data[f"loc_{place}_unlock_ratio"] = safe_division(g(f"loc_{place}_unlock_num"),
                                                          g(f"loc_{place}_unlock_duration"))
    data["sleep_end_minus_start"] = g("sleep_end") - g("sleep_start")
    data["unlock_ratio_ep0"] = safe_division(g("unlock_num_ep_0"), g("unlock_duration_ep_0"))
    return data


def load_weekly_stress(path):
    """Last stress self-report of each (user, calendar week)."""
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["day"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "uid"])
    df["stress"] = pd.to_numeric(df["stress"], errors="coerce")
    df = df.dropna(subset=["stress"])
    df["week_start"] = df["date"].dt.to_period("W").dt.start_time
    df = df.sort_values(["uid", "date"]).reset_index(drop=True)
    df = df.sort_values(["uid", "date"]).groupby(["uid", "week_start"], sort=False).last().reset_index()
    return df[["uid", "date", "stress", "week_start"]]


def behavioural_scores(train_raw, apply_raw):
    """Stage 1, fitted on the training weeks only."""
    train_scores = pd.DataFrame(index=train_raw.index)
    apply_scores = pd.DataFrame(index=apply_raw.index)
    for label in LABELS:
        feats = [f for f in LABEL_FEATURES[label] if f in train_raw.columns]
        if not feats:
            train_scores[f"{label}_score"] = 0.0
            apply_scores[f"{label}_score"] = 0.0
            continue
        z_train = StandardScaler().fit_transform(train_raw[feats].values)
        score_train = (z_train > 0).sum(axis=1).astype(float)
        weights = {f: 1.0 / len(feats) for f in feats}
        if pd.Series(score_train).nunique() > 1 and len(train_raw) >= 5:
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            rf.fit(z_train, score_train)
            weights = dict(zip(feats, rf.feature_importances_))
        mm = MinMaxScaler().fit(train_raw[feats].values)
        w = np.array([weights[f] for f in feats])
        train_scores[f"{label}_score"] = (mm.transform(train_raw[feats].values) * w).sum(axis=1)
        apply_scores[f"{label}_score"] = (mm.transform(apply_raw[feats].values) * w).sum(axis=1)
    return train_scores, apply_scores


def build_model(n_features, seed):
    tf.keras.utils.set_random_seed(seed)
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss=Huber(delta=1.0), metrics=["mae"])
    return model


def predict_fold(train_raw, test_raw, y_train, seed):
    train_scores, test_scores = behavioural_scores(train_raw, test_raw)
    cols = [f"{l}_score" for l in LABELS]
    X_train, X_test = train_scores[cols].values, test_scores[cols].values
    y_train = y_train.astype(float)

    n = len(y_train)
    model = build_model(X_train.shape[1], seed)
    if n >= MIN_ROWS_FOR_INTERNAL_VAL:
        n_val = max(1, int(round(n * INTERNAL_VAL_FRACTION)))
        es = EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True)
        model.fit(X_train[:-n_val], y_train[:-n_val], validation_data=(X_train[-n_val:], y_train[-n_val:]),
                  epochs=50, batch_size=8, callbacks=[es], verbose=0)
    else:
        model.fit(X_train, y_train, epochs=FIXED_EPOCHS_NO_VAL, batch_size=8, verbose=0)

    y_pred = np.clip(model.predict(X_test, verbose=0).flatten(), STRESS_MIN, STRESS_MAX)
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return y_pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensing", required=True)
    ap.add_argument("--ema", required=True)
    ap.add_argument("--rows", required=True, help="ihope_rows.csv: uid, week_start, fold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-users", type=int, default=0, help="stop after N users (0 = no limit)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(args.rows)
    rows["week_start"] = pd.to_datetime(rows["week_start"])
    week_to_fold = {w: int(min(f)) for w, f in rows.groupby("week_start")["fold"].unique().items()}
    fold_cutoff = rows.groupby("fold")["week_start"].min().to_dict()
    allowed = set(zip(rows["uid"].astype(str), rows["week_start"]))

    merged = load_weekly_stress(args.ema).merge(load_sensing(args.sensing), on=["uid", "date"], how="inner")
    merged["stress"] = merged["stress"].clip(STRESS_MIN, STRESS_MAX)
    keys = list(zip(merged["uid"].astype(str), merged["week_start"]))
    merged = merged[[k in allowed for k in keys]]
    merged["fold"] = merged["week_start"].map(week_to_fold)
    merged = merged.dropna(subset=["fold"])
    merged["fold"] = merged["fold"].astype(int)

    counts = merged.groupby("uid").size()
    users = counts[counts >= MIN_SAMPLES].index.tolist()
    log.info("weeks %d | users included %d / %d", len(merged), len(users), merged["uid"].nunique())

    out_path = out_dir / "ihope_predictions.csv"
    done = set(pd.read_csv(out_path, usecols=["uid"])["uid"].astype(str)) if out_path.exists() else set()
    todo = [u for u in users if str(u) not in done]
    log.info("users already done %d | to do %d", len(users) - len(todo), len(todo))

    feat_cols = sorted({f for feats in LABEL_FEATURES.values() for f in feats})
    for n_done, uid in enumerate(todo, start=1):
        predictions = []
        df_u = merged[merged["uid"] == uid].sort_values("week_start").reset_index(drop=True)
        y_u = df_u["stress"].values.astype(float)
        for fold in sorted(int(f) for f in df_u["fold"].unique()):
            train = df_u["week_start"] < fold_cutoff[fold]
            test = df_u["fold"] == fold
            if train.sum() < MIN_TRAIN_SAMPLES or test.sum() == 0:
                continue
            y_train = y_u[train.values]
            if y_train.std() == 0:
                continue
            y_pred = predict_fold(df_u.loc[train, feat_cols], df_u.loc[test, feat_cols], y_train, seed=42 + fold)
            predictions.append(pd.DataFrame({
                "uid": uid, "fold": fold, "week_start": df_u.loc[test, "week_start"].dt.strftime("%Y-%m-%d").tolist(),
                "y_true": y_u[test.values], "y_pred": y_pred}))
        if predictions:
            pd.concat(predictions, ignore_index=True).to_csv(out_path, mode="a", index=False,
                                                             header=not out_path.exists())
        log.info("user %s done (%d/%d)", uid, len(users) - len(todo) + n_done, len(users))
        if args.max_users and n_done >= args.max_users and n_done < len(todo):
            log.info("stopping after %d users, restart to continue", n_done)
            sys.exit(3)

    out = pd.read_csv(out_path)
    log.info("saved %s | %d predictions | %d users", out_path, len(out), out["uid"].nunique())


if __name__ == "__main__":
    main()
