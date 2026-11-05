"""Calibration and diagnostics for binary parking arrival risks."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


_LOGIT_EPS = 1e-6
_BIN_EDGES = np.asarray([i / 10 for i in range(11)])


def _vector(values, name):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric vector") from error
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return array


def _probabilities(values):
    array = _vector(values, "probabilities")
    if ((array < 0) | (array > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    return array


def _scoring_pair(labels, probabilities):
    y = _vector(labels, "labels")
    p = _probabilities(probabilities)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be binary (0 or 1)")
    if len(y) != len(p):
        raise ValueError("labels and probabilities must have equal lengths")
    return y, p


def _logit(probabilities):
    p = np.clip(probabilities, _LOGIT_EPS, 1 - _LOGIT_EPS)
    return np.log(p) - np.log1p(-p)


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """A fitted logistic calibration or an explicitly reported identity map."""

    status: str
    slope: float | None = None
    intercept: float | None = None

    def predict(self, raw):
        p = _probabilities(raw)
        if self.slope is None:
            return p.copy()
        scores = self.intercept + self.slope * _logit(p)
        return 1 / (1 + np.exp(-np.clip(scores, -700, 700)))


def fit_calibrator(raw, labels, *, min_per_class=5):
    """Fit a regularized Platt map on a separate chronological segment.

    The caller supplies scores from a base model that has never trained on
    these labels. Too few examples, no variation in scores, or a fitted map
    that would invert rankings leave probabilities unchanged.
    """
    if not isinstance(min_per_class, (int, np.integer)) or min_per_class < 1:
        raise ValueError("min_per_class must be a positive integer")
    y, p = _scoring_pair(labels, raw)
    if len(y) == 0:
        return ProbabilityCalibrator("identity_no_rows")
    positives = int(y.sum())
    if min(positives, len(y) - positives) < min_per_class:
        return ProbabilityCalibrator("identity_insufficient_classes")
    x = _logit(p).reshape(-1, 1)
    if np.ptp(x) <= 1e-12:
        return ProbabilityCalibrator("identity_constant_scores")
    model = LogisticRegression(C=1.0, solver="lbfgs").fit(x, y)
    slope = float(model.coef_[0, 0])
    if slope <= 0:
        return ProbabilityCalibrator("identity_nonpositive_slope")
    return ProbabilityCalibrator("platt", slope, float(model.intercept_[0]))


def risk_metrics(labels, probabilities):
    """Score probability forecasts; discrimination is undefined for one class."""
    y, p = _scoring_pair(labels, probabilities)
    n = len(y)
    events = int(y.sum())
    both_classes = 0 < events < n
    return {
        "n": n,
        "events": events,
        "brier": float(np.mean((p - y) ** 2)) if n else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if n else None,
        "average_precision": float(average_precision_score(y, p)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
    }


def reliability_bins(labels, probabilities):
    """Summarize held-out reliability in fixed deciles, keeping empty bins.

    The boundaries are [0, .1), [.1, .2), ..., [.9, 1]. Compare each bin's
    mean forecast to its observed event rate, with the count for context.
    """
    y, p = _scoring_pair(labels, probabilities)
    indices = np.minimum(np.searchsorted(_BIN_EDGES, p, side="right") - 1, 9)
    result = []
    for index in range(10):
        members = indices == index
        n = int(members.sum())
        events = int(y[members].sum())
        result.append({
            "bin_lower": float(_BIN_EDGES[index]),
            "bin_upper": float(_BIN_EDGES[index + 1]),
            "n": n,
            "events": events,
            "mean_predicted": float(p[members].mean()) if n else None,
            "event_rate": events / n if n else None,
        })
    return result
