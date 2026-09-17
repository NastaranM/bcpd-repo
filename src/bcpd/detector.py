"""High-level interface: the three-step algorithm as a single call."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .marginal import PriorSpec, SegmentLikelihood
from .multivariate import MultivariateSegmentLikelihood, MVPriorSpec
from .recursion import ChangepointPosterior, forward_recursion, sample_changepoints

__all__ = ["detect", "detect_multivariate", "DetectionResult"]


def _resolve_k_max(min_distance: int, n: int, k_max: int) -> int:
    """Validate the spacing and clamp k_max to what the series can support.

    k changepoints need k + 1 segments of at least ``min_distance`` each, so
    anything above ``n // min_distance - 1`` is unreachable: the recursion would
    spend its time on rows of the table that are entirely -inf and silently
    report a posterior over counts it could never have produced.

    A spacing too large for even one changepoint is an outright error, since it
    almost always means a spacing tuned for one sampling rate has been carried
    onto a series with another.  A merely optimistic k_max is clamped instead,
    because it is an upper bound rather than a requirement.
    """
    if min_distance < 1:
        raise ValueError(f"min_distance must be >= 1; got {min_distance}")
    if 2 * min_distance > n:
        raise ValueError(
            f"min_distance={min_distance} leaves no room for a changepoint in a "
            f"series of length {n}: two segments alone need {2 * min_distance} "
            "observations. Check that min_distance matches this series' sampling "
            "rate rather than another's."
        )
    feasible = n // min_distance - 1
    return min(k_max, feasible)


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
    min_distance: int,
    k_max: int = 4,
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
    min_distance : int
        Minimum number of observations between adjacent changepoints.  Required,
        with no default, because the right value depends on the sampling rate
        and on how close two changes could plausibly be.
    k_max : int
        Upper bound on the number of changepoints; clamped to what the series
        can support.
    prior : PriorSpec, optional
        Defaults to weakly informative settings.
    nested_sizes : tuple of int, optional
        Nested design sizes to average over.  For a Fourier design, pass
        ``(2, 4, ..., 2 * L_max)`` to put a prior on the harmonic order.
    """
    y = np.asarray(y, dtype=float).ravel()
    k_max = _resolve_k_max(min_distance, len(y), k_max)
    if prior is None:
        prior = PriorSpec()

    like = SegmentLikelihood(y, X, prior, nested_sizes=nested_sizes)
    P = like.segment_table(min_length=min_distance)
    F = forward_recursion(P, k_max)
    post = sample_changepoints(
        like, P, F, k_max, min_distance, n_samples=n_samples, rng=rng
    )
    return DetectionResult(posterior=post, segments=P, forward=F, likelihood=like)


def detect_multivariate(
    Y: np.ndarray,
    X: np.ndarray,
    *,
    min_distance: int,
    k_max: int = 4,
    prior: MVPriorSpec | None = None,
    nested_sizes: tuple[int, ...] | None = None,
    n_samples: int = 500,
    rng: np.random.Generator | None = None,
) -> DetectionResult:
    """Detect changepoints shared across d series.

    Identical to ``detect`` apart from the segment likelihood: the recursion and
    the backtrace are untouched, because shared changepoints leave the segment
    structure exactly as it was.

    Parameters
    ----------
    Y : (N, d) array
        The series, one column each.
    X : (N, q) array
        Shared design matrix; ``joint_design`` is the usual choice.
    prior : MVPriorSpec, optional
        Defaults to ``MVPriorSpec.vague_for(Y)``.
    nested_sizes : tuple of int, optional
        Nested design sizes to average over.  With ``joint_design``, pass
        ``harmonic_nested_sizes(L_max)`` for a prior on the harmonic order.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    k_max = _resolve_k_max(min_distance, Y.shape[0], k_max)
    if prior is None:
        prior = MVPriorSpec.vague_for(Y)

    like = MultivariateSegmentLikelihood(Y, X, prior, nested_sizes=nested_sizes)
    P = like.segment_table(min_length=min_distance)
    F = forward_recursion(P, k_max)
    post = sample_changepoints(
        like, P, F, k_max, min_distance, n_samples=n_samples, rng=rng
    )
    return DetectionResult(posterior=post, segments=P, forward=F, likelihood=like)
