"""High-level interface: the three-step algorithm as a single call."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .marginal import PriorSpec, SegmentLikelihood
from .multivariate import MultivariateSegmentLikelihood, MVPriorSpec
from .recursion import ChangepointPosterior, forward_recursion, sample_changepoints

__all__ = ["detect", "detect_multivariate", "DetectionResult"]


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


def detect_multivariate(
    Y: np.ndarray,
    X: np.ndarray,
    *,
    k_max: int = 4,
    min_distance: int = 96,
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
    if prior is None:
        prior = MVPriorSpec.vague_for(Y)

    like = MultivariateSegmentLikelihood(Y, X, prior, nested_sizes=nested_sizes)
    P = like.segment_table(min_length=min_distance)
    F = forward_recursion(P, k_max)
    post = sample_changepoints(like, P, F, k_max, n_samples=n_samples, rng=rng)
    return DetectionResult(posterior=post, segments=P, forward=F, likelihood=like)
