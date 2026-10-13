"""Predictors that turn past days into an expected call rate for every minute of the day.

Both predictors return an array of shape (n_functions, 1440): the expected number of calls
in each minute. The scheduler turns that into P(at least one call in the next L minutes)
with 1 - exp(-sum of rates), which assumes calls arrive as a Poisson process.
"""
from __future__ import annotations

import numpy as np

MINUTES = 1440


class MeanRatePredictor:
    """Average the same minute over the training days, smoothed by +/- smooth minutes."""
    name = "mean"

    def fit_predict(self, train_days, functions) -> np.ndarray:
        stack = np.stack(train_days).astype(float)          # days, functions, minutes
        est = np.zeros((stack.shape[1], MINUTES))
        for fi, f in enumerate(functions):
            s = f.smooth
            acc = np.zeros(MINUTES)
            for k in range(-s, s + 1):
                acc += np.roll(stack[:, fi, :], -k, axis=1).mean(axis=0)
            est[fi] = acc / (2 * s + 1)
        return est


class GBMRatePredictor:
    """Gradient boosting (Poisson loss) on time-of-day features, one model per function."""
    name = "gbm"

    def __init__(self, max_iter: int = 150, learning_rate: float = 0.08, random_state: int = 0):
        self.max_iter, self.learning_rate, self.random_state = max_iter, learning_rate, random_state

    @staticmethod
    def _features() -> np.ndarray:
        m = np.arange(MINUTES)
        ang = 2 * np.pi * m / MINUTES
        return np.column_stack([m / 60.0, np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang),
                                m % 60, m % 30, m % 15, m % 10, m % 5])

    def fit_predict(self, train_days, functions) -> np.ndarray:
        from sklearn.ensemble import HistGradientBoostingRegressor

        X1 = self._features()
        stack = np.stack(train_days)
        X = np.vstack([X1] * stack.shape[0])
        est = np.zeros((stack.shape[1], MINUTES))
        for fi in range(stack.shape[1]):
            y = np.concatenate([d[fi] for d in stack]).astype(float)
            model = HistGradientBoostingRegressor(loss="poisson", max_iter=self.max_iter,
                                                  learning_rate=self.learning_rate,
                                                  min_samples_leaf=5, random_state=self.random_state)
            model.fit(X, y)
            est[fi] = np.clip(model.predict(X1), 0, None)
        return est


PREDICTORS = {"mean": MeanRatePredictor, "gbm": GBMRatePredictor}


def prefix_sums(est: np.ndarray) -> np.ndarray:
    """Cumulative sums over two repeated days so windows can wrap past midnight."""
    doubled = np.concatenate([est, est], axis=1)
    pre = np.zeros((est.shape[0], 2 * MINUTES + 1))
    pre[:, 1:] = np.cumsum(doubled, axis=1)
    return pre


def evaluate_predictor(est: np.ndarray, test_day: np.ndarray) -> dict:
    """Simple accuracy numbers on the test day (mean absolute error and calls-in-next-2-min F1)."""
    mae = float(np.abs(est - test_day).mean())
    pre = prefix_sums(est)
    tp = fp = fn = 0
    for fi in range(est.shape[0]):
        for m in range(MINUTES):
            p = 1 - np.exp(-(pre[fi, m + 2] - pre[fi, m]))
            actual = test_day[fi, m] + test_day[fi, (m + 1) % MINUTES] > 0
            pred = p >= 0.5
            tp += pred and actual
            fp += pred and not actual
            fn += (not pred) and actual
    tp, fp, fn = int(tp), int(fp), int(fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"mae_per_minute": mae, "precision": prec, "recall": rec, "f1": f1}
