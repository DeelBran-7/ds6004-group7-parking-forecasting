"""Paired comparisons that keep observations from the same day together."""

from numbers import Integral

import numpy as np
import pandas as pd


def paired_day_inference(days, candidate, reference, *, repetitions=10_000, seed=20_261_105):
    """Compare matched candidate and reference outcomes, grouped by local day.

    Inputs are aligned, finite, one-dimensional vectors. ``days`` must
    contain a nonmissing local calendar-day key for each paired observation;
    the caller chooses which pairs qualify and maps timestamps to days.

    The estimand is the observation-weighted mean difference. The 95%
    percentile interval resamples whole days with replacement, pooling all
    observations of each sampled day. The two-sided Monte Carlo test flips
    the signs of entire days, uses the absolute mean difference as the
    statistic, and counts the observed assignment via an add-one correction.
    The bootstrap treats days as representative, independent blocks. The
    sign-flip test additionally assumes independent day effects whose signs
    are exchangeable under the null (e.g. symmetric day differences).
    Neither procedure accounts for dependence between neighboring days.
    No interval or p-value is supplied if there are fewer than two days or
    all observed day differences are zero; that does not prove equivalence.
    """
    day_keys = np.asarray(days)
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if any(values.ndim != 1 for values in (day_keys, candidate, reference)):
        raise ValueError("days, candidate and reference must be one-dimensional")
    if not len(day_keys) == len(candidate) == len(reference):
        raise ValueError("days, candidate and reference must have equal lengths")
    if pd.isna(day_keys).any() or not (np.isfinite(candidate).all() and np.isfinite(reference).all()):
        raise ValueError("day keys and outcomes must be nonmissing; outcomes must be finite")
    if isinstance(repetitions, bool) or not isinstance(repetitions, Integral) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")

    n_pairs = len(candidate)
    indices, distinct_days = pd.factorize(day_keys, sort=True)
    n_days = len(distinct_days)
    result = dict(n_pairs=n_pairs, n_days=n_days, candidate_mean=None, reference_mean=None,
                  estimate=None, ci_low=None, ci_high=None, p_value=None, status="no_pairs")
    if not n_pairs:
        return result

    deltas = candidate - reference
    day_deltas = np.bincount(indices, weights=deltas, minlength=n_days)
    counts = np.bincount(indices, minlength=n_days)
    result.update(candidate_mean=float(candidate.mean()), reference_mean=float(reference.mean()),
                  estimate=float(deltas.mean()), status="insufficient_days")
    if n_days < 2:
        return result
    if np.all(day_deltas == 0):
        result["status"] = "no_day_level_variation"
        return result

    bootstrap_seed, sign_flip_seed = np.random.SeedSequence(seed).spawn(2)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    sign_rng = np.random.default_rng(sign_flip_seed)
    estimates = np.empty(repetitions)
    observed_abs_sum = abs(day_deltas.sum())
    at_least_as_extreme = 0
    batch_size = max(1, min(256, 262_144 // n_days))
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        shape = (stop - start, n_days)
        sampled = bootstrap_rng.integers(0, n_days, size=shape)
        estimates[start:stop] = day_deltas[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        signs = 2 * sign_rng.integers(0, 2, size=shape, dtype=np.int8) - 1
        # The denominator is the same for every sign assignment.
        totals = np.abs((signs * day_deltas).sum(axis=1))
        at_least_as_extreme += np.count_nonzero(totals >= observed_abs_sum)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    result.update(ci_low=float(lower), ci_high=float(upper),
                  p_value=float((at_least_as_extreme + 1) / (repetitions + 1)), status="ok")
    return result


def holm_adjust(p_values):
    """Return Holm-adjusted p-values in input order, keeping None as None."""
    p_values = list(p_values)
    adjusted = [None] * len(p_values)
    ranked = []
    for index, value in enumerate(p_values):
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("p-values must be in [0, 1] or None")
        ranked.append((value, index))
    previous = 0.0
    for rank, (value, index) in enumerate(sorted(ranked)):
        previous = min(1.0, max(previous, value * (len(ranked) - rank)))
        adjusted[index] = previous
    return adjusted
