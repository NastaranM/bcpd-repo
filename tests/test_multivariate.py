"""Multivariate segment likelihood and shared-changepoint detection.

Two independent checks pin the derivation:

* at d = 1 the matrix-normal / inverse-Wishart marginal must reproduce the
  univariate ``SegmentLikelihood`` exactly, with nu0 = 2 c1 and Psi = 2 c2;
* for d > 1 it is compared against Monte Carlo over the inverse-Wishart prior,
  using Y | Sigma ~ MN(0, I_n + nu X X', Sigma).  That identity follows from
  vec(Y) = (I_d kron X) vec(B) + vec(E) and does not go through the
  completing-the-square step being tested.
"""

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import invwishart

from bcpd import (
    MultivariateSegmentLikelihood,
    MVPriorSpec,
    PriorSpec,
    SegmentLikelihood,
    detect_multivariate,
    harmonic_nested_sizes,
    joint_design,
    log_multivariate_gamma,
    trend_design,
)


def monte_carlo_log_marginal(Y, X, prior, n_samples=400_000, seed=0):
    """E over Sigma of MN(Y; 0, I_n + nu X X', Sigma)."""
    n, d = Y.shape
    V = np.eye(n) + prior.nu * X @ X.T
    G = Y.T @ np.linalg.solve(V, Y)
    ldV = np.linalg.slogdet(V)[1]

    S = invwishart.rvs(
        df=prior.nu0, scale=prior.psi_scale * np.eye(d), size=n_samples, random_state=seed
    ).reshape(n_samples, d, d)
    quad = np.einsum("mij,ji->m", np.linalg.inv(S), G)
    ldS = np.linalg.slogdet(S)[1]
    logw = -0.5 * n * d * np.log(2 * np.pi) - 0.5 * d * ldV - 0.5 * n * ldS - 0.5 * quad
    return logsumexp(logw) - np.log(n_samples)


def test_log_multivariate_gamma_reduces_to_gammaln():
    from scipy.special import gammaln

    for a in (1.5, 3.0, 7.25):
        np.testing.assert_allclose(log_multivariate_gamma(a, 1), gammaln(a))


def test_log_multivariate_gamma_rejects_small_argument():
    with pytest.raises(ValueError, match="requires a >"):
        log_multivariate_gamma(0.4, 3)


@pytest.mark.parametrize("start,stop", [(0, 20), (5, 45), (30, 80)])
def test_d1_collapses_to_univariate_trend(start, stop):
    rng = np.random.default_rng(0)
    N = 80
    y = rng.normal(size=N) + 0.02 * np.arange(N)
    c1, c2, nu = 0.37, 1.9, 40.0

    uni = SegmentLikelihood(y, trend_design(N), PriorSpec(c1=c1, c2=c2, nu=nu))
    mv = MultivariateSegmentLikelihood(
        y, trend_design(N), MVPriorSpec(nu=nu, nu0=2 * c1, psi_scale=2 * c2)
    )
    np.testing.assert_allclose(
        mv.log_marginal(start, stop), uni.log_marginal(start, stop), rtol=1e-12
    )


def test_d1_collapses_on_joint_design():
    """Same check with q = 8, which exercises the determinant terms."""
    rng = np.random.default_rng(1)
    N = 80
    y = rng.normal(size=N)
    X = joint_design(N, 96, 3)
    c1, c2, nu = 0.37, 1.9, 40.0

    uni = SegmentLikelihood(y, X, PriorSpec(c1=c1, c2=c2, nu=nu))
    mv = MultivariateSegmentLikelihood(
        y, X, MVPriorSpec(nu=nu, nu0=2 * c1, psi_scale=2 * c2)
    )
    np.testing.assert_allclose(mv.log_marginal(10, 70), uni.log_marginal(10, 70), rtol=1e-12)


@pytest.mark.parametrize(
    "n,q,d,nu,nu0,psi", [(6, 1, 2, 3.0, 5.0, 1.4), (10, 2, 3, 5.0, 7.0, 1.2)]
)
def test_matches_monte_carlo_over_prior(n, q, d, nu, nu0, psi):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n, q))
    Y = rng.normal(size=(n, d)) * 0.8
    prior = MVPriorSpec(nu=nu, nu0=nu0, psi_scale=psi)

    analytic = MultivariateSegmentLikelihood(Y, X, prior).log_marginal(0, n)
    np.testing.assert_allclose(
        analytic, monte_carlo_log_marginal(Y, X, prior, seed=7), atol=0.05
    )


def test_improper_prior_is_rejected():
    Y = np.zeros((20, 4))
    with pytest.raises(ValueError, match="nu0 must exceed"):
        MultivariateSegmentLikelihood(Y, trend_design(20), MVPriorSpec(nu0=2.0))


def test_vague_for_scales_with_data():
    rng = np.random.default_rng(3)
    Y = rng.normal(scale=5.0, size=(200, 3))
    prior = MVPriorSpec.vague_for(Y)
    assert prior.nu0 == 5.0  # d + 2
    assert 10.0 < prior.psi_scale < 50.0  # near var = 25


def test_correlated_series_share_a_changepoint():
    """Three correlated series breaking together at t = 150."""
    rng = np.random.default_rng(11)
    n, d = 300, 3
    shared = np.concatenate([np.zeros(150), np.full(150, 6.0)])
    noise = rng.multivariate_normal(
        np.zeros(d), 0.6 * np.eye(d) + 0.4 * np.ones((d, d)), size=n
    )
    Y = shared[:, None] + noise

    result = detect_multivariate(
        Y,
        trend_design(n),
        k_max=3,
        min_distance=60,
        n_samples=200,
        rng=np.random.default_rng(5),
    )
    assert result.posterior.map_count() == 1
    prob = result.location_prob
    assert prob[140:160].sum() > 0.8, f"mass near t=150 was {prob[140:160].sum():.3f}"


def test_no_false_detection_multivariate():
    rng = np.random.default_rng(12)
    n, d = 250, 3
    Y = rng.multivariate_normal(np.zeros(d), np.eye(d), size=n)

    result = detect_multivariate(
        Y,
        trend_design(n),
        k_max=3,
        min_distance=60,
        n_samples=200,
        rng=np.random.default_rng(6),
    )
    assert result.posterior.map_count() == 0


def test_harmonic_nested_sizes_include_trend_only():
    sizes = harmonic_nested_sizes(4)
    assert sizes == (2, 4, 6, 8, 10)
    assert sizes[0] == 2  # a segment may carry no seasonal component


def test_posterior_B_has_one_column_per_series():
    rng = np.random.default_rng(4)
    n, d = 200, 3
    Y = rng.normal(size=(n, d))
    X = joint_design(n, 96, 2)
    like = MultivariateSegmentLikelihood(Y, X, MVPriorSpec.vague_for(Y))
    assert like.posterior_B(0, n).shape == (X.shape[1], d)
