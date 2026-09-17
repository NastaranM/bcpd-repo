"""Design matrices for the trend and seasonal components.

Kept separate from the inference code so a different basis can be swapped in
without touching the recursion.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "constant_design",
    "trend_design",
    "seasonal_design",
    "joint_design",
    "harmonic_nested_sizes",
]


def constant_design(n: int) -> np.ndarray:
    """Intercept only: each segment has a level and nothing else.

    Use this when the changes you care about are shifts in level.  It is the
    most sensitive choice for that purpose, because nothing else in the model
    can absorb a step.

    Prefer it over ``trend_design`` unless you positively want within-segment
    slopes.  A slope can explain a moderate level shift as gradual drift, so
    including one makes the model *less* likely to call a step a changepoint --
    correctly, in the sense that one ramp really can fit a modest step about as
    well as two levels while using fewer parameters, but usually not what is
    wanted from a detector.
    """
    return np.ones((n, 1))


def trend_design(n: int) -> np.ndarray:
    """Piecewise-linear trend design: intercept and slope.

    Returns an (n, 2) matrix.  Time runs 1..n, matching ``seq_along`` in R.
    """
    t = np.arange(1, n + 1, dtype=float)
    return np.column_stack([np.ones(n), t])


def seasonal_design(n: int, period: float, harmonic_order: int) -> np.ndarray:
    """Fourier design for a periodic component.

    Returns an (n, 2 * harmonic_order) matrix with columns
    [sin(2*pi*l*t/P), cos(2*pi*l*t/P)] for l = 1..harmonic_order.

    ``period`` is measured in observations, so it depends on the sampling rate:
    a daily cycle is 24 for hourly data and 1440 for minutely data.
    """
    if harmonic_order < 1:
        raise ValueError("harmonic_order must be >= 1")
    t = np.arange(1, n + 1, dtype=float)
    cols = []
    for l in range(1, harmonic_order + 1):
        angle = 2.0 * np.pi * l * t / period
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.column_stack(cols)


def joint_design(n: int, period: float, harmonic_order: int) -> np.ndarray:
    """Trend and seasonal terms as one design matrix.

    Columns are ``[1, t, sin(2 pi t/P), cos(2 pi t/P), ...]``: the two trend
    columns first, then ``harmonic_order`` sin/cos pairs.  Shape (n, 2 + 2L).

Estimating trend and periodic terms in one design is preferable to the common
    alternative of fitting a trend, detrending, and fitting the periodic part on
    the residual.  Staged estimation treats the detrended series as data when it
    is an estimate carrying its own uncertainty, so the second stage comes out
    overconfident.  It also does not survive the move to multiple series, since
    detrending each separately presupposes series-specific trend changepoints
    and contradicts a shared-changepoint model.

    The cost is that a changepoint now marks a change of regime rather than
    being labelled trend or seasonal.  That label is recoverable after the fact
    by comparing posterior coefficient blocks either side of a boundary: a
    moved slope is a trend break, moved Fourier terms a seasonal one.
    """
    if harmonic_order < 0:
        raise ValueError("harmonic_order must be >= 0")
    parts = [trend_design(n)]
    if harmonic_order > 0:
        parts.append(seasonal_design(n, period, harmonic_order))
    return np.column_stack(parts)


def harmonic_nested_sizes(harmonic_order: int) -> tuple[int, ...]:
    """Nested column counts for ``joint_design``, for a prior on L.

    Returns ``(2, 4, ..., 2 + 2 * harmonic_order)``: trend only, trend plus one
    harmonic, and so on.  The leading entry lets a segment carry no seasonal
    component at all.
    """
    return tuple(2 + 2 * l for l in range(harmonic_order + 1))
