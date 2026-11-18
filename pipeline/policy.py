"""Sparse self-report collection policy and personal statistics."""

import numpy as np
import pandas as pd

ONBOARDING = 2          # self-reports requested at the start
POLICY_DAYS = 28        # then one every 4 calendar weeks (weeks compared by their Monday)
STATS_WINDOW = 12       # personal mean/std over the last 12 self-reports...
STATS_MIN = 4           # ...when at least 4 are available


def request_mask(week_starts):
    """True for the rows whose self-report is collected. If the scheduled week has
    no response, the next week with a response is used."""
    mask = np.zeros(len(week_starts), dtype=bool)
    last = None
    for i, week in enumerate(week_starts):
        week = pd.Timestamp(week)
        mask[i] = i < ONBOARDING or (week - last).days >= POLICY_DAYS
        if mask[i]:
            last = week
    return mask


def recent_stats(values):
    """Mean and std of the last STATS_WINDOW values (NaN if fewer than STATS_MIN)."""
    arr = np.array(values[-STATS_WINDOW:], dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < STATS_MIN:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr))
