"""Forward recursion and stochastic backtrace.

Steps two and three of the recursive Bayesian changepoint algorithm
(Meftahi 2021, Sec. 4.4; Ruggieri & Lawrence 2012).

The forward pass fills a table ``F`` where ``F[k, j]`` is the log probability of
the first ``j + 1`` observations containing exactly ``k + 1`` changepoints,
marginalising over all admissible locations:

    p(y_1..j | k) = sum_tau p(y_1..tau | k - 1) p(y_tau+1..j)      Eq. (4.41)

The backward pass then draws changepoint counts and locations directly from the
joint posterior, without MCMC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .marginal import SegmentLikelihood

__all__ = ["forward_recursion", "ChangepointPosterior", "sample_changepoints"]


def forward_recursion(P: np.ndarray, k_max: int) -> np.ndarray:
    """Run the forward recursion over the segment table.

    Parameters
    ----------
    P : (N, N) array
        ``P[i, j]`` is the log marginal likelihood of ``y[i : j + 1]``; -inf for
        inadmissible segments.  Produced by ``SegmentLikelihood.segment_table``.
    k_max : int
        Maximum number of changepoints to consider.

    Returns
    -------
    (k_max, N) array
        ``F[k, j]`` = log p(y[0 : j + 1] with exactly k + 1 changepoints).
    """
    N = P.shape[0]
    F = np.full((k_max, N), -np.inf)

    # k = 0 (one changepoint): split the prefix at every admissible tau.
    for j in range(1, N):
        tau = np.arange(j)
        terms = P[0, tau] + P[tau + 1, j]
        if np.any(np.isfinite(terms)):
            F[0, j] = logsumexp(terms)

    # k >= 1: reuse the previous row.
    for k in range(1, k_max):
        for j in range(k + 1, N):
            tau = np.arange(j)
            terms = F[k - 1, tau] + P[tau + 1, j]
            if np.any(np.isfinite(terms)):
                F[k, j] = logsumexp(terms)
    return F


@dataclass
class ChangepointPosterior:
    """Posterior over the number and locations of changepoints."""

    n_changepoints: np.ndarray  # (k_max + 1,) probability of 0..k_max changepoints
    location_counts: np.ndarray  # (N,) times each index was drawn as a changepoint
    draws: list[list[int]]  # the sampled changepoint sets
    n_samples: int

    @property
    def location_prob(self) -> np.ndarray:
        """Posterior probability that each index is a changepoint."""
        return self.location_counts / self.n_samples

    def map_count(self) -> int:
        return int(np.argmax(self.n_changepoints))


def _count_posterior(P: np.ndarray, F: np.ndarray, k_max: int) -> np.ndarray:
    """Posterior over the number of changepoints, Eq. (4.43).

    A uniform prior is placed on {0, ..., k_max}: mass 1/2 on "no changepoint"
    and 1/2 spread over the k_max positive counts, matching the R original.
    """
    N = P.shape[1]
    log_k = np.empty(k_max + 1)
    log_k[0] = P[0, N - 1] + np.log(0.5)
    for k in range(1, k_max + 1):
        log_k[k] = F[k - 1, N - 1] + np.log(0.5) - np.log(k_max)
    return np.exp(log_k - logsumexp(log_k))


def sample_changepoints(
    seg: SegmentLikelihood,
    P: np.ndarray,
    F: np.ndarray,
    k_max: int,
    n_samples: int = 500,
    rng: np.random.Generator | None = None,
) -> ChangepointPosterior:
    """Draw from the joint posterior of changepoint number and locations.

    Draws are independent (no burn-in, no convergence diagnostics), which is the
    main practical advantage of the recursive approach over RJMCMC.

    Unlike the R original, the generator is created once and threaded through,
    so the draws actually differ from one another.
    """
    if rng is None:
        rng = np.random.default_rng()
    N = P.shape[1]

    count_post = _count_posterior(P, F, k_max)
    location_counts = np.zeros(N)
    draws: list[list[int]] = []

    for _ in range(n_samples):
        k = int(rng.choice(k_max + 1, p=count_post))
        locations: list[int] = []
        end = N - 1
        # Peel off changepoints from the right, Eq. (4.44).
        for level in range(k, 0, -1):
            tau = np.arange(end)
            prefix = P[0, tau] if level == 1 else F[level - 2, tau]
            terms = prefix + P[tau + 1, end]
            finite = np.isfinite(terms)
            if not finite.any():
                break
            probs = np.zeros_like(terms)
            probs[finite] = np.exp(terms[finite] - logsumexp(terms[finite]))
            cp = int(rng.choice(end, p=probs))
            locations.append(cp)
            location_counts[cp] += 1
            end = cp
        draws.append(sorted(locations))

    return ChangepointPosterior(
        n_changepoints=count_post,
        location_counts=location_counts,
        draws=draws,
        n_samples=n_samples,
    )
