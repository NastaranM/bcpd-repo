"""Design matrices for the trend and seasonal components.

Corresponds to Eq. (4.2) and Eq. (4.3) of the thesis.  Kept separate from the
inference code so that the multivariate extension can swap in a different
design without touching the recursion.
"""

from __future__ import annotations

import numpy as np

__all__ = ["trend_design", "seasonal_design"]


def trend_design(n: int) -> np.ndarray:
    """Piecewise-linear trend design: intercept and slope, Eq. (4.2).

    Returns an (n, 2) matrix.  Time runs 1..n, matching ``seq_along`` in R.
    """
    t = np.arange(1, n + 1, dtype=float)
    return np.column_stack([np.ones(n), t])


def seasonal_design(n: int, period: float, harmonic_order: int) -> np.ndarray:
    """Fourier design for the seasonal component, Eq. (4.3).

    Returns an (n, 2 * harmonic_order) matrix with columns
    [sin(2*pi*l*t/P), cos(2*pi*l*t/P)] for l = 1..harmonic_order.

    Note the column count drives the 2**m enumeration in the segment
    likelihood: the thesis default of harmonic_order = 4 gives m = 8, hence 256
    inclusion patterns per segment.
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
