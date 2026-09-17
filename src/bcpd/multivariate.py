"""Multivariate segment marginal likelihood, shared changepoints.

Generalises ``bcpd.marginal`` from one series to d series that share segment
boundaries.  For a segment of length n:

    Y = X B + E,              rows of E iid N_d(0, Sigma)
    B | Sigma ~ MN(0, nu I_q, Sigma)
    Sigma     ~ InverseWishart(Psi, nu0)

Y is n x d, X is n x q (the same design as the univariate case), B is q x d.
Each series gets its own coefficients; the segment boundaries are common.  The
off-diagonal entries of Sigma carry the cross-series correlation, which is the
reason to model the series jointly rather than d times separately.

The prior on B is conditioned on Sigma.  That is what makes the pair conjugate
-- both exponents must carry the same Sigma^-1 to merge into one quadratic form
-- and it is the direct analogue of beta | sigma^2 ~ N(0, sigma^2 nu I_q) in the
univariate model.  It does assert something substantive: series whose noise is
correlated are given correlated coefficient priors.

Integrating out B and Sigma leaves a matrix-t:

    p(Y) = pi^(-nd/2)
           * |Lambda_0|^(d/2) / |Lambda_n|^(d/2)
           * |Psi|^(nu0/2)    / |Psi_n|^(nu_n/2)
           * Gamma_d(nu_n/2)  / Gamma_d(nu0/2)

with Lambda_0 = I_q / nu, Lambda_n = X'X + Lambda_0, nu_n = nu0 + n, and

    S     = Y'Y - (X'Y)' Lambda_n^-1 (X'Y)
    Psi_n = Psi + S

S is the d x d analogue of the univariate residual scalar, minus sign and all.

At d = 1, setting nu0 = 2 c1 and Psi = 2 c2 recovers the univariate marginal
exactly; ``tests/test_multivariate.py`` checks this against
``bcpd.marginal.SegmentLikelihood``, and checks d > 1 against Monte Carlo over
the prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

__all__ = ["MVPriorSpec", "MultivariateSegmentLikelihood", "log_multivariate_gamma"]


def log_multivariate_gamma(a: float, d: int) -> float:
    """log of the multivariate gamma function Gamma_d(a).

    Gamma_d(a) = pi^(d(d-1)/4) prod_{j=1..d} Gamma(a + (1 - j)/2),
    finite only for a > (d - 1)/2.
    """
    if a <= (d - 1) / 2.0:
        raise ValueError(f"Gamma_{d} requires a > {(d - 1) / 2}; got {a}")
    j = np.arange(1, d + 1)
    return 0.25 * d * (d - 1) * np.log(np.pi) + np.sum(gammaln(a + (1.0 - j) / 2.0))


@dataclass(frozen=True)
class MVPriorSpec:
    """Hyperparameters of the multivariate segment model.

    Parameters
    ----------
    nu
        Scale of the matrix-normal prior on B across predictors: the row
        covariance is ``nu I_q``.  Same role as ``nu`` in the univariate model.
    nu0
        Degrees of freedom of the inverse-Wishart prior on Sigma.  Must exceed
        d - 1 for the prior to be proper, and d + 1 for its mean to exist.
        Unlike the univariate c1, this cannot be made arbitrarily small: the
        dimension sets a floor.  ``vague_for`` picks a defensible default.
    psi_scale
        Sigma's prior scale matrix is ``psi_scale * I_d``.  With short segments
        this genuinely influences the estimated correlations, so it is worth
        setting from the data rather than leaving at 1.
    """

    nu: float = 100.0
    nu0: float = 3.0
    psi_scale: float = 1.0

    @classmethod
    def vague_for(cls, Y: np.ndarray, nu: float = 100.0) -> "MVPriorSpec":
        """A weakly informative default for d series.

        Takes nu0 = d + 2, the smallest value giving the prior a finite mean
        with a little slack, and scales Psi by the mean marginal variance so
        the prior sits at the data's order of magnitude.
        """
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        if Y.shape[0] == 1:
            Y = Y.T
        d = Y.shape[1]
        scale = float(np.mean(np.var(Y, axis=0))) or 1.0
        return cls(nu=nu, nu0=d + 2.0, psi_scale=scale)


class MultivariateSegmentLikelihood:
    """Log marginal likelihood of any contiguous segment of d series.

    Cumulative sufficient statistics are precomputed, so a segment query costs
    O(q^3 + d^3) independent of segment length.

    Parameters
    ----------
    Y : (N, d) array
        The series, one column each.  A 1-D array is treated as d = 1.
    X : (N, q) array
        Shared design matrix.
    prior : MVPriorSpec
    nested_sizes : sequence of int, optional
        Column counts of a nested design family to average over under a uniform
        prior.  With ``joint_design``, pass ``(2, 4, ..., 2 + 2 L_max)`` to put
        a prior on the harmonic order while always retaining the trend columns.
    """

    def __init__(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        prior: MVPriorSpec,
        nested_sizes: tuple[int, ...] | None = None,
    ):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but Y has {Y.shape[0]}")

        self.Y = Y
        self.X = X
        self.prior = prior
        self.N, self.d = Y.shape
        self.q_full = X.shape[1]

        if prior.nu0 <= self.d - 1:
            raise ValueError(
                f"nu0 must exceed d - 1 = {self.d - 1} for a proper "
                f"inverse-Wishart prior; got {prior.nu0}"
            )

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

        # Cumulative sufficient statistics; segment [start, stop) is
        # cum[stop] - cum[start].
        self._XtX = np.zeros((self.N + 1, self.q_full, self.q_full))
        self._XtX[1:] = np.cumsum(X[:, :, None] * X[:, None, :], axis=0)
        self._XtY = np.zeros((self.N + 1, self.q_full, self.d))
        self._XtY[1:] = np.cumsum(X[:, :, None] * Y[:, None, :], axis=0)
        self._YtY = np.zeros((self.N + 1, self.d, self.d))
        self._YtY[1:] = np.cumsum(Y[:, :, None] * Y[:, None, :], axis=0)

        self._Psi = prior.psi_scale * np.eye(self.d)
        self._log_det_Psi = self.d * np.log(prior.psi_scale)
        self._log_gamma_0 = log_multivariate_gamma(prior.nu0 / 2.0, self.d)

    def suff_stats(self, start: int, stop: int):
        """Sufficient statistics for rows [start, stop)."""
        XtX = self._XtX[stop] - self._XtX[start]
        XtY = self._XtY[stop] - self._XtY[start]
        YtY = self._YtY[stop] - self._YtY[start]
        return XtX, XtY, YtY

    def _log_marginal_at_size(
        self, XtX: np.ndarray, XtY: np.ndarray, YtY: np.ndarray, n: int, q: int
    ) -> float:
        p = self.prior
        d = self.d

        Lam0_logdet = -q * np.log(p.nu)  # |I_q / nu| = nu^-q
        Lam_n = XtX[:q, :q] + np.eye(q) / p.nu
        sign, Lam_n_logdet = np.linalg.slogdet(Lam_n)
        if sign <= 0:
            raise np.linalg.LinAlgError("non-positive-definite Lambda_n")

        B = XtY[:q, :]
        S = YtY - B.T @ np.linalg.solve(Lam_n, B)
        Psi_n = self._Psi + 0.5 * (S + S.T)  # symmetrise against roundoff

        sign, Psi_n_logdet = np.linalg.slogdet(Psi_n)
        if sign <= 0:
            raise np.linalg.LinAlgError("non-positive-definite Psi_n")

        nu_n = p.nu0 + n
        return (
            -0.5 * n * d * np.log(np.pi)
            + 0.5 * d * Lam0_logdet
            - 0.5 * d * Lam_n_logdet
            + 0.5 * p.nu0 * self._log_det_Psi
            - 0.5 * nu_n * Psi_n_logdet
            + log_multivariate_gamma(nu_n / 2.0, d)
            - self._log_gamma_0
        )

    def log_marginal(self, start: int, stop: int) -> float:
        """Log marginal likelihood of rows [start, stop)."""
        n = stop - start
        if n <= 0:
            raise ValueError("empty segment")
        XtX, XtY, YtY = self.suff_stats(start, stop)

        if len(self.nested_sizes) == 1:
            return self._log_marginal_at_size(XtX, XtY, YtY, n, self.nested_sizes[0])
        terms = [
            self._log_marginal_at_size(XtX, XtY, YtY, n, q) for q in self.nested_sizes
        ]
        return float(logsumexp(terms) + self._log_prior_size)

    def log_marginal_by_size(self, start: int, stop: int) -> np.ndarray:
        n = stop - start
        XtX, XtY, YtY = self.suff_stats(start, stop)
        return np.array(
            [self._log_marginal_at_size(XtX, XtY, YtY, n, q) for q in self.nested_sizes]
        )

    def posterior_size(self, start: int, stop: int) -> np.ndarray:
        """Posterior over ``nested_sizes`` on a segment."""
        logp = self.log_marginal_by_size(start, stop) + self._log_prior_size
        return np.exp(logp - logsumexp(logp))

    def posterior_B(self, start: int, stop: int, q: int | None = None) -> np.ndarray:
        """Posterior mean of the coefficient matrix: Lambda_n^-1 X'Y, q x d.

        Comparing this across a changepoint says which series moved, and which
        columns of the design moved with them.
        """
        if q is None:
            q = self.nested_sizes[-1]
        XtX, XtY, _ = self.suff_stats(start, stop)
        Lam_n = XtX[:q, :q] + np.eye(q) / self.prior.nu
        return np.linalg.solve(Lam_n, XtY[:q, :])

    def segment_table(self, min_length: int) -> np.ndarray:
        """Log marginal likelihood of every admissible segment, (N, N)."""
        P = np.full((self.N, self.N), -np.inf)
        for i in range(self.N):
            for j in range(i + min_length - 1, self.N):
                P[i, j] = self.log_marginal(i, j + 1)
        return P
