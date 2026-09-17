"""Validate the segment likelihood against independent numerical integration.

Eq. (4.38) as printed in the thesis contains transcription errors (see the
module docstring of ``bcpd.marginal``).  These tests pin the corrected form by
integrating beta and sigma^2 numerically and comparing.
"""

import numpy as np
import pytest
from scipy.integrate import dblquad, quad
from scipy.special import gammaln

from bcpd import PriorSpec, SegmentLikelihood, seasonal_design, trend_design


def brute_force_2d(y, X, prior):
    """Integrate beta and sigma^2 numerically. Only viable for q = 1."""
    n = len(y)
    x = X[:, 0]

    def integrand(beta, s2):
        log_lik = -0.5 * n * np.log(2 * np.pi * s2) - np.sum((y - x * beta) ** 2) / (2 * s2)
        log_prior_b = -0.5 * np.log(2 * np.pi * s2 * prior.nu) - beta**2 / (2 * s2 * prior.nu)
        log_prior_s = (
            prior.c1 * np.log(prior.c2)
            - gammaln(prior.c1)
            - (prior.c1 + 1) * np.log(s2)
            - prior.c2 / s2
        )
        return np.exp(log_lik + log_prior_b + log_prior_s)

    # Bound both axes around where the mass actually is: a fixed wide box
    # leaves the peak too narrow for the quadrature grid to resolve.
    a = float(x @ x + 1.0 / prior.nu)
    b_hat = float(x @ y) / a
    S = float(y @ y) - float(x @ y) ** 2 / a
    s2_mode = (prior.c2 + 0.5 * S) / (prior.c1 + n / 2.0 + 1.0)
    s2_lo, s2_hi = 1e-4 * s2_mode, 60.0 * s2_mode
    spread = 12.0 * np.sqrt(s2_hi / a)

    val, _ = dblquad(
        integrand,
        s2_lo,
        s2_hi,
        lambda _: b_hat - spread,
        lambda _: b_hat + spread,
        epsabs=1e-13,
        epsrel=1e-11,
    )
    return np.log(val)


def brute_force_sigma_only(y, X, prior):
    """Integrate sigma^2 numerically with beta done analytically. Any q."""
    n, q = X.shape
    A = X.T @ X + np.eye(q) / prior.nu
    b = X.T @ y
    S = y @ y - b @ np.linalg.solve(A, b)
    logdet = np.linalg.slogdet(A)[1]

    def integrand(s2):
        lp = (
            -0.5 * n * np.log(2 * np.pi * s2)
            - 0.5 * q * np.log(s2 * prior.nu)
            + 0.5 * q * np.log(s2)
            - 0.5 * logdet
            - S / (2 * s2)
            + prior.c1 * np.log(prior.c2)
            - gammaln(prior.c1)
            - (prior.c1 + 1) * np.log(s2)
            - prior.c2 / s2
        )
        return np.exp(lp)

    # The integrand is a sharply peaked inverse-gamma kernel.  Naive quadrature
    # over a wide interval misses most of the mass, so split around the mode.
    mode = (prior.c2 + 0.5 * S) / (prior.c1 + n / 2.0 + 1.0)
    edges = [1e-12, 0.05 * mode, 0.5 * mode, 2.0 * mode, 20.0 * mode, 1e5 * mode]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        part, _ = quad(integrand, lo, hi, limit=400)
        total += part
    return np.log(total)


def test_matches_full_numerical_integration_q1():
    """Strongest check: both beta and sigma^2 integrated numerically."""
    rng = np.random.default_rng(0)
    n = 8
    X = np.ones((n, 1))
    y = 2.0 + rng.normal(scale=0.6, size=n)
    prior = PriorSpec(nu=2.0)

    like = SegmentLikelihood(y, X, prior)
    np.testing.assert_allclose(
        like.log_marginal(0, n), brute_force_2d(y, X, prior), rtol=1e-4
    )


@pytest.mark.parametrize("start,stop", [(0, 30), (5, 45), (12, 60)])
def test_matches_numerical_sigma_integration(start, stop):
    rng = np.random.default_rng(1)
    y = rng.normal(size=60)
    X = trend_design(60)
    prior = PriorSpec(nu=50.0)
    like = SegmentLikelihood(y, X, prior)

    expected = brute_force_sigma_only(y[start:stop], X[start:stop], prior)
    np.testing.assert_allclose(like.log_marginal(start, stop), expected, rtol=1e-5)


def test_sufficient_statistics_match_direct():
    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    X = trend_design(60)
    like = SegmentLikelihood(y, X, PriorSpec())

    XtX, Xty, yty = like.suff_stats(10, 40)
    np.testing.assert_allclose(XtX, X[10:40].T @ X[10:40])
    np.testing.assert_allclose(Xty, X[10:40].T @ y[10:40])
    np.testing.assert_allclose(yty, y[10:40] @ y[10:40])


def test_posterior_beta_recovers_known_slope():
    n = 200
    t = np.arange(1, n + 1, dtype=float)
    y = 3.0 + 0.5 * t
    like = SegmentLikelihood(y, trend_design(n), PriorSpec(nu=1e6))
    np.testing.assert_allclose(like.posterior_beta(0, n), [3.0, 0.5], atol=1e-2)


def test_nested_sizes_average_lies_between_members():
    """A uniform average must sit between the smallest and largest member."""
    n = 300
    rng = np.random.default_rng(4)
    t = np.arange(1, n + 1, dtype=float)
    y = 2.0 * np.sin(2 * np.pi * t / 96) + rng.normal(scale=0.3, size=n)
    X = seasonal_design(n, 96, 4)
    like = SegmentLikelihood(y, X, PriorSpec(), nested_sizes=(2, 4, 6, 8))

    per_size = like.log_marginal_by_size(0, n)
    total = like.log_marginal(0, n)
    assert per_size.min() <= total <= per_size.max()


def test_posterior_harmonic_order_favours_the_truth():
    """A pure first harmonic should put most posterior mass on L = 1."""
    n = 400
    rng = np.random.default_rng(5)
    t = np.arange(1, n + 1, dtype=float)
    y = 3.0 * np.sin(2 * np.pi * t / 96) + rng.normal(scale=0.4, size=n)
    X = seasonal_design(n, 96, 4)
    like = SegmentLikelihood(y, X, PriorSpec(), nested_sizes=(2, 4, 6, 8))

    post = like.posterior_size(0, n)
    np.testing.assert_allclose(post.sum(), 1.0, rtol=1e-10)
    assert post[0] > 0.5, f"expected L=1 to dominate, got {post}"


def test_posterior_harmonic_order_detects_higher_harmonic():
    """Signal with a third harmonic should not collapse to L = 1."""
    n = 400
    rng = np.random.default_rng(6)
    t = np.arange(1, n + 1, dtype=float)
    y = (
        3.0 * np.sin(2 * np.pi * t / 96)
        + 2.5 * np.cos(2 * np.pi * 3 * t / 96)
        + rng.normal(scale=0.4, size=n)
    )
    X = seasonal_design(n, 96, 4)
    like = SegmentLikelihood(y, X, PriorSpec(), nested_sizes=(2, 4, 6, 8))

    post = like.posterior_size(0, n)
    assert post[2] + post[3] > 0.9, f"expected L>=3, got {post}"


def test_nested_sizes_are_validated():
    y = np.zeros(20)
    X = trend_design(20)
    with pytest.raises(ValueError, match="must lie in"):
        SegmentLikelihood(y, X, PriorSpec(), nested_sizes=(2, 9))
