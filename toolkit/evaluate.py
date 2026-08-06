"""Scoring with the uncertainty attached.

A single cross-validation mean is a number without an error bar, and most of the
comparisons in this repository are close enough that the error bar decides them.
Everything here returns the spread alongside the estimate.

Author: Elyes Lounissi
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values, *, level: float = 0.95, n_boot: int = 10_000,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap interval for the mean of `values`.

    Returns (mean, low, high). Used instead of a t-interval because fold scores
    are neither independent nor normal, and the percentile method makes fewer
    promises about either.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(values.mean()), float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))


def paired_ci(a, b, *, level: float = 0.95, n_boot: int = 10_000,
              seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap interval for the paired difference a - b.

    Pairing matters when both models saw the same folds: the fold-to-fold
    variation is shared, and ignoring that makes a real difference look like noise.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match in shape, got {a.shape} and {b.shape}")
    return bootstrap_ci(a - b, level=level, n_boot=n_boot, seed=seed)


def expected_calibration_error(y_true, y_prob, *, n_bins: int = 15) -> float:
    """ECE with equal-width bins.

    Equal-width rather than equal-count, because the sparsely populated
    high-confidence bins are exactly the ones worth seeing.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)

    error = 0.0
    for b in range(n_bins):
        mask = which == b
        if not mask.any():
            continue
        error += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(error)


def reliability_curve(y_true, y_prob, *, n_bins: int = 15):
    """Points for a reliability diagram: (mean predicted, observed rate, count)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)

    predicted, observed, counts = [], [], []
    for b in range(n_bins):
        mask = which == b
        if not mask.any():
            continue
        predicted.append(y_prob[mask].mean())
        observed.append(y_true[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(predicted), np.array(observed), np.array(counts)
