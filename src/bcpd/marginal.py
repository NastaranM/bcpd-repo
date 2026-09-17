"""Segment marginal likelihood for the Bayesian changepoint model.

The conjugate linear-Gaussian segment model:

    y = X beta + e,          e ~ N(0, sigma^2 I)
    beta | sigma^2, nu ~ N(0, sigma^2 nu I_q)
    sigma^2 ~ InvGamma(c1, c2)
    nu      ~ InvGamma(c3, c4)

A single shared scale ``nu`` shrinks every coefficient by the same amount.
There is no spike-and-slab component: predictors are never switched off, so
the segment likelihood is one closed-form evaluation rather than a sum over
2**q inclusion patterns.

Integrating out both beta and sigma^2 leaves

    log p(y_seg | nu) = -n/2 log(2 pi)
                        - q/2 log(nu)
                        - 1/2 log|X'X + nu^-1 I_q|
                        + c1 log(c2) - lgamma(c1)
                        + lgamma(c1 + n/2)
                        - (c1 + n/2) log(c2 + S/2)

with S = y'y - (X'y)' (X'X + nu^-1 I_q)^-1 (X'y).

Numerical validation
--------------------
This expression is checked in ``tests/test_marginal.py`` against numerical
integration of beta and sigma^2, so the implementation is pinned to the
integral rather than to a transcription of it.

Nested designs
--------------
``nested_sizes`` averages the segment likelihood over a nested family of
designs -- the leading 2, 4, ..., q columns -- under a uniform prior.  Its main
use is Fourier terms, where it amounts to a prior on the harmonic order: it
respects the ordering of harmonics and keeps each sin/cos pair intact, at a cost
of one evaluation per family member rather than 2**q.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

__all__ = ["PriorSpec", "SegmentLikelihood"]


@dataclass(frozen=True)
class PriorSpec:
    """Hyperparameters of the segment model.

    Parameters
    ----------
    c1, c2
        Shape and scale of the inverse-gamma prior on the error variance.
        Small values are weakly informative; 0.01 is a common choice.
    nu
        Scale of the normal prior on the coefficients: larger means less
        shrinkage.  Held fixed rather than sampled.
    c3, c4
        Shape and scale of a hyper-prior on nu.  Recorded for callers who want
        to update nu themselves; unused while nu is held fixed.
    """

    c1: float = 0.01
    c2: float = 0.01
    nu: float = 100.0
    c3: float = 0.02
    c4: float = 0.02


class SegmentLikelihood:
    """Log marginal likelihood of any contiguous segment of a series.

    Cumulative sufficient statistics are precomputed, so the cost of a segment
    query is independent of its length: O(q**3), not O(n q**2).

    Parameters
    ----------
    y : (N,) array
    X : (N, q) array
        Design matrix, rows aligned with ``y``.
    prior : PriorSpec
    nested_sizes : sequence of int, optional
        Column counts of a nested family of designs to average over, under a
        uniform prior.  For a Fourier design with L_max = 4 harmonics, pass
        ``(2, 4, 6, 8)`` for a prior on the harmonic order.  Each entry must be
        <= X.shape[1].  Defaults to using all columns.
    """

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        prior: PriorSpec,
        nested_sizes: tuple[int, ...] | None = None,
    ):
        y = np.asarray(y, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")

        self.y = y
        self.X = X
        self.prior = prior
        self.N, self.q_full = X.shape

        if nested_sizes is None:
            self.nested_sizes = (self.q_full,)
        else:
            sizes = tuple(int(s) for s in nested_sizes)
            if not sizes:
                raise ValueError("nested_sizes must not be empty")
            if any(s < 1 or s > self.q_full for s in sizes):
                raise ValueError(f"nested_sizes must lie in 1..{self.q_full}")
            self.nested_sizes = sizes
        self._log_prior_size = -np.log(len(self.nested_sizes))

        # Cumulative sufficient statistics, padded so that segment [start, stop)
        # is cum[stop] - cum[start].
        self._XtX = np.zeros((self.N + 1, self.q_full, self.q_full))
        self._XtX[1:] = np.cumsum(X[:, :, None] * X[:, None, :], axis=0)
        self._Xty = np.zeros((self.N + 1, self.q_full))
        self._Xty[1:] = np.cumsum(X * y[:, None], axis=0)
        self._yty = np.zeros(self.N + 1)
        self._yty[1:] = np.cumsum(y**2)

        self._log_const = prior.c1 * np.log(prior.c2) - gammaln(prior.c1)

    def suff_stats(self, start: int, stop: int):
        """Sufficient statistics for y[start:stop] (half-open, 0-based)."""
        XtX = self._XtX[stop] - self._XtX[start]
        Xty = self._Xty[stop] - self._Xty[start]
        yty = self._yty[stop] - self._yty[start]
        return XtX, Xty, yty

    def _log_marginal_at_size(
        self, XtX: np.ndarray, Xty: np.ndarray, yty: float, n: int, q: int
    ) -> float:
        """Segment log marginal likelihood for the leading ``q`` columns."""
        p = self.prior
        A = XtX[:q, :q] + np.eye(q) / p.nu
        b = Xty[:q]

        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            raise np.linalg.LinAlgError("non-positive-definite Gram matrix")

        # S = y'y - (X'y)' A^-1 (X'y), the residual sum of squares after
        # shrinkage.  Positive by construction; clipped only against roundoff.
        S = yty - b @ np.linalg.solve(A, b)
        S = max(S, 0.0)

        shape = p.c1 + n / 2.0
        scale = p.c2 + 0.5 * S
        return (
            -0.5 * n * np.log(2.0 * np.pi)
            - 0.5 * q * np.log(p.nu)
            - 0.5 * logdet
            + self._log_const
            + gammaln(shape)
            - shape * np.log(scale)
        )

    def log_marginal(self, start: int, stop: int) -> float:
        """Log marginal likelihood of y[start:stop].

        When ``nested_sizes`` holds more than one entry, the result is the
        average over that nested family under a uniform prior.
        """
        n = stop - start
        if n <= 0:
            raise ValueError("empty segment")
        XtX, Xty, yty = self.suff_stats(start, stop)

        if len(self.nested_sizes) == 1:
            return self._log_marginal_at_size(XtX, Xty, yty, n, self.nested_sizes[0])

        terms = [
            self._log_marginal_at_size(XtX, Xty, yty, n, q) for q in self.nested_sizes
        ]
        return float(logsumexp(terms) + self._log_prior_size)

    def log_marginal_by_size(self, start: int, stop: int) -> np.ndarray:
        """Log marginal likelihood at each entry of ``nested_sizes``.

        Normalise these to read off the posterior over the harmonic order on a
        segment.
        """
        n = stop - start
        XtX, Xty, yty = self.suff_stats(start, stop)
        return np.array(
            [self._log_marginal_at_size(XtX, Xty, yty, n, q) for q in self.nested_sizes]
        )

    def posterior_size(self, start: int, stop: int) -> np.ndarray:
        """Posterior over ``nested_sizes`` on a segment.

        For a Fourier design this is the posterior over the harmonic order.
        """
        logp = self.log_marginal_by_size(start, stop) + self._log_prior_size
        return np.exp(logp - logsumexp(logp))

    def posterior_beta(self, start: int, stop: int, q: int | None = None) -> np.ndarray:
        """Posterior mean of beta on a segment.

        ``V X'y`` with ``V = (X'X + nu^-1 I_q)^-1``.
        """
        if q is None:
            q = self.nested_sizes[-1]
        XtX, Xty, _ = self.suff_stats(start, stop)
        A = XtX[:q, :q] + np.eye(q) / self.prior.nu
        return np.linalg.solve(A, Xty[:q])

    def segment_table(self, min_length: int) -> np.ndarray:
        """Log marginal likelihood of every admissible segment.

        Returns an (N, N) array ``P`` with ``P[i, j]`` the log marginal
        likelihood of ``y[i : j + 1]``, and -inf where the segment is shorter
        than ``min_length``.  This is the initialisation step of the recursion.
        """
        P = np.full((self.N, self.N), -np.inf)
        for i in range(self.N):
            for j in range(i + min_length - 1, self.N):
                P[i, j] = self.log_marginal(i, j + 1)
        return P
