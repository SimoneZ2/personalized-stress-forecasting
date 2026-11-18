"""Load the forecasting dataset written by features/build_forecast_dataset.py."""

from types import SimpleNamespace

import numpy as np
import pandas as pd


def load_forecast_dataset(path):
    d = np.load(path, allow_pickle=True)
    return SimpleNamespace(
        X=d["X"].astype(np.float32),
        y=d["y"].astype(np.float32),
        weeks=pd.to_datetime(d["weeks"]).values,
        uids=d["uids"],
        anchors=pd.to_datetime(d["anchor_dates"]).values,
        targets=pd.to_datetime(d["target_dates"]).values,
        gbm_X=d["gbm_X"],
        gbm_names=[str(n) for n in d["gbm_feature_names"]],
    )
