"""High-level interface: the three-step algorithm as a single call."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .design import seasonal_design, trend_design
from .marginal import PriorSpec, SegmentLikelihood
from .recursion import ChangepointPosterior, forward_recursion, sample_changepoints

__all__ = ["detect", "DetectionResult", "decompose"]


@dataclass
class DetectionResult:
    posterior: ChangepointPosterior
    segments: np.ndarray  # log marginal likelihood table
    forward: np.ndarray
    likelihood: SegmentLikelihood

    @property
    def n_changepoints(self) -> np.ndarray:
        return self.posterior.n_changepoints

    @property
    def location_prob(self) -> np.ndarray:
        return self.posterior.location_prob


def detect(
    y: np.ndarray,
    X: np.ndarray,
    *,
    k_max: int = 4,
    min_distance: int = 96,
    prior: PriorSpec | None = None,
    nested_sizes: tuple[int, ...] | None = None,
    n_samples: int = 500,
    rng: np.random.Generator | None = None,
) -> DetectionResult:
    """Run initialisation, forward recursion and stochastic backtrace.

    Parameters
    ----------
    y : (N,) array
        The series.
    X : (N, m) array
        Design matrix. Use ``trend_design`` or ``seasonal_design``.
    k_max : int
        Maximum number of changepoints.
    min_distance : int
        Minimum number of observations between adjacent changepoints.  The
        thesis uses 96 (one day at 15-minute resolution).
    prior : PriorSpec, optional
        Defaults to the thesis settings (c1 = c2 = 0.01).
    nested_sizes : tuple of int, optional
        Nested design sizes to average over.  For a Fourier design, pass
        ``(2, 4, ..., 2 * L_max)`` to put a prior on the harmonic order.
    """
    y = np.asarray(y, dtype=float).ravel()
    if prior is None:
        prior = PriorSpec()

    like = SegmentLikelihood(y, X, prior, nested_sizes=nested_sizes)
    P = like.segment_table(min_length=min_distance)
    F = forward_recursion(P, k_max)
    post = sample_changepoints(like, P, F, k_max, n_samples=n_samples, rng=rng)
    return DetectionResult(posterior=post, segments=P, forward=F, likelihood=like)


def decompose(
    y: np.ndarray,
    *,
    period: float = 96.0,
    harmonic_order: int = 4,
    k_max: int = 4,
    min_distance: int = 96,
    n_samples: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[DetectionResult, DetectionResult, np.ndarray]:
    """Two-stage pipeline: trend changepoints, detrend, then seasonal.

    This reproduces the thesis pipeline faithfully, including its known
    approximation: Eq. (4.4) defines the model jointly over
    ``X = [X_trend, X_seasonal]``, but inference runs the two components in
    sequence rather than jointly.  Retained here as the reproduction baseline;
    the joint formulation is the intended replacement.

    Returns ``(trend_result, seasonal_result, detrended)``.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)

    trend = detect(
        y,
        trend_design(n),
        k_max=k_max,
        min_distance=min_distance,
        n_samples=n_samples,
        rng=rng,
    )

    fitted = _fitted_values(y, trend, trend_design(n), min_distance)
    detrended = y - fitted

    seasonal = detect(
        detrended,
        seasonal_design(n, period, harmonic_order),
        k_max=k_max,
        min_distance=min_distance,
        nested_sizes=tuple(2 * l for l in range(1, harmonic_order + 1)),
        n_samples=n_samples,
        rng=rng,
    )
    return trend, seasonal, detrended


def _fitted_values(
    y: np.ndarray, result: DetectionResult, X: np.ndarray, min_distance: int
) -> np.ndarray:
    """Posterior-mean fit implied by the modal changepoint configuration."""
    n = len(y)
    k = result.posterior.map_count()
    if k == 0:
        boundaries = [0, n]
    else:
        prob = result.location_prob.copy()
        picked: list[int] = []
        for _ in range(k):
            idx = int(np.argmax(prob))
            if prob[idx] <= 0:
                break
            picked.append(idx)
            lo = max(0, idx - min_distance)
            hi = min(n, idx + min_distance)
            prob[lo:hi] = 0.0
        boundaries = [0, *sorted(p + 1 for p in picked), n]

    fitted = np.zeros(n)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop <= start:
            continue
        beta = result.likelihood.posterior_beta(start, stop)
        q = len(beta)
        fitted[start:stop] = X[start:stop, :q] @ beta
    return fitted
