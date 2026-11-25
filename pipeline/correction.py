"""Adaptive Thresholds and Correction Module."""

import numpy as np

from peak_classifier import PEAK_K

DECAY, THR_MIN, THR_MAX, ALPHA = 0.75, 0.20, 0.50, 1.0   # used when not calibrated
FLAG_WINDOW = 4
GRID_DECAY = [0.5, 0.75, 0.9]
GRID_THR_MIN = [0.20, 0.30, 0.40]
GRID_THR_MAX = [0.50, 0.60, 0.70]
GRID_ALPHA = [0.5, 0.75, 1.0]
MAX_FLAGGED = 0.45


class AdaptiveFlag:
    """flag(t) = max(p(t), flag(t-1) * decay); active when flag(t) exceeds a threshold
    that rises from thr_min to thr_max with the share of recent flagged weeks."""

    def __init__(self, decay=DECAY, thr_min=THR_MIN, thr_max=THR_MAX, w=FLAG_WINDOW):
        self.decay, self.thr_min, self.thr_max, self.w = decay, thr_min, thr_max, w
        self.value = 0.0
        self._hist = []

    def update(self, prob):
        self.value = max(prob, self.value * self.decay)
        self._hist.append(self.value)
        recent = self._hist[-self.w:]
        activity = sum(1 for v in recent if v > self.thr_min) / len(recent)
        threshold = self.thr_min + (self.thr_max - self.thr_min) * activity
        return self.value, self.value > threshold


def corrected_forecast(y_lstm, flag_value, active, pm, ps, peak_size, alpha, y_range=(1, 5)):
    """Move the LSTM forecast towards pm + peak_size * ps, in proportion to the flag."""
    correction = 0.0
    if active and not np.isnan(ps) and ps > 0.01:
        target_peak = pm + peak_size * ps
        correction = alpha * (target_peak - y_lstm) * flag_value
    return float(np.clip(y_lstm + correction, *y_range))


def calibrate_flag_params(users, peak_size, y_range=(1, 5)):
    """Grid search of decay, thresholds and alpha on the training users (LUO): lowest
    peak-week error with at most 45% of flagged weeks. The personal mean stands in for
    the LSTM forecast. Returns None if no combination is admissible."""
    if not users:
        return None
    for u in users:
        u["y_lstm_approx"] = np.where(np.isnan(u["pm"]), np.nanmean(u["y_true"]), u["pm"])

    best = None
    for decay in GRID_DECAY:
        for thr_min in GRID_THR_MIN:
            for thr_max in GRID_THR_MAX:
                if thr_min >= thr_max:
                    continue
                for alpha in GRID_ALPHA:
                    errors, peak_errors, n_flagged, n_total = [], [], 0, 0
                    for u in users:
                        flag = AdaptiveFlag(decay, thr_min, thr_max)
                        y_pred = np.empty(len(u["probs"]))
                        flagged = np.zeros(len(u["probs"]), dtype=bool)
                        for i, p in enumerate(u["probs"]):
                            fv, active = flag.update(p)
                            flagged[i] = active
                            y_pred[i] = corrected_forecast(u["y_lstm_approx"][i], fv, active,
                                                           u["pm"][i], u["ps"][i], peak_size, alpha, y_range)
                        err = np.abs(u["y_true"] - y_pred)
                        errors.extend(err.tolist())
                        with np.errstate(invalid="ignore", divide="ignore"):
                            z = (u["y_true"] - u["pm"]) / u["ps"]
                        is_peak = (z >= PEAK_K) & ~np.isnan(z)
                        if is_peak.any():
                            peak_errors.extend(err[is_peak].tolist())
                        n_flagged += int(flagged.sum())
                        n_total += len(flagged)

                    if n_total == 0 or not peak_errors:
                        continue
                    share = n_flagged / n_total
                    mae_peak = float(np.mean(peak_errors))
                    if share > MAX_FLAGGED:
                        continue
                    if best is None or mae_peak < best["mae_peak"]:
                        best = {"decay": decay, "thr_min": thr_min, "thr_max": thr_max, "alpha": alpha,
                                "mae": float(np.mean(errors)), "mae_peak": mae_peak, "pct_flagged": share}
    return best
