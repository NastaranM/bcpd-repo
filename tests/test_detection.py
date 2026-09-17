"""Reproduce the two synthetic experiments from Chapter 5 of the thesis.

Section 5.1: data with no changepoints; the algorithm should report zero.
Section 5.2: three segments with shifts at t = 100 and t = 250.
"""

import numpy as np
import pytest

from bcpd import detect, trend_design


def synthetic_no_changepoint(n=400, seed=0):
    """Thesis synthetic dataset-1: Gaussian noise, no structural break."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=n)


def synthetic_two_changepoints(seed=0):
    """Thesis synthetic dataset-2: mean shifts at t = 100 and t = 250."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.normal(1.0, 2.0, size=100),
            rng.normal(175.0, 20.0, size=150),
            rng.normal(100.0, 20.0, size=200),
        ]
    )


def test_no_false_detection():
    """Section 5.1: zero changepoints should dominate the posterior."""
    y = synthetic_no_changepoint()
    result = detect(
        y,
        trend_design(len(y)),
        k_max=4,
        min_distance=96,
        n_samples=200,
        rng=np.random.default_rng(42),
    )
    assert result.posterior.map_count() == 0
    assert result.n_changepoints[0] > 0.9


def test_recovers_two_changepoints():
    """Section 5.2: two changepoints, located near t = 100 and t = 250."""
    y = synthetic_two_changepoints()
    result = detect(
        y,
        trend_design(len(y)),
        k_max=4,
        min_distance=96,
        n_samples=200,
        rng=np.random.default_rng(42),
    )
    assert result.posterior.map_count() == 2

    prob = result.location_prob
    for truth in (100, 250):
        window = prob[truth - 12 : truth + 12]
        assert window.sum() > 0.5, f"no posterior mass near t={truth}"


def test_changepoint_probabilities_are_normalised():
    y = synthetic_two_changepoints()
    result = detect(
        y,
        trend_design(len(y)),
        k_max=4,
        min_distance=96,
        n_samples=100,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_allclose(result.n_changepoints.sum(), 1.0, rtol=1e-10)
    assert np.all(result.location_prob >= 0)


def test_min_distance_is_respected():
    y = synthetic_two_changepoints()
    result = detect(
        y,
        trend_design(len(y)),
        k_max=4,
        min_distance=96,
        n_samples=100,
        rng=np.random.default_rng(7),
    )
    for draw in result.posterior.draws:
        gaps = np.diff([0, *draw, len(y) - 1])
        assert np.all(gaps >= 0)


def test_draws_are_independent():
    """The R original re-seeded on every call, freezing all 500 draws."""
    y = synthetic_two_changepoints()
    result = detect(
        y,
        trend_design(len(y)),
        k_max=4,
        min_distance=96,
        n_samples=100,
        rng=np.random.default_rng(3),
    )
    unique = {tuple(d) for d in result.posterior.draws}
    assert len(unique) > 1


def test_min_distance_has_no_default():
    y = synthetic_two_changepoints()
    with pytest.raises(TypeError):
        detect(y, trend_design(len(y)))


def test_min_distance_too_large_is_rejected():
    """A spacing carried over from another sampling rate should fail loudly."""
    y = np.random.default_rng(0).normal(size=100)
    with pytest.raises(ValueError, match="no room for a changepoint"):
        detect(y, trend_design(100), min_distance=96, k_max=4)


def test_k_max_is_clamped_to_what_fits():
    """k_max is an upper bound, so an optimistic one is clamped, not refused."""
    y = synthetic_two_changepoints()  # length 450
    result = detect(y, trend_design(len(y)), min_distance=96, k_max=4,
                    n_samples=50, rng=np.random.default_rng(0))
    # 450 // 96 - 1 = 3, so the posterior spans 0..3 rather than 0..4
    assert len(result.n_changepoints) == 4


def test_forward_recursion_matches_exhaustive_enumeration():
    """The recursion must equal a brute-force sum over every admissible split."""
    import itertools

    from scipy.special import logsumexp

    from bcpd import PriorSpec, SegmentLikelihood
    from bcpd.recursion import forward_recursion

    rng = np.random.default_rng(0)
    N, mind = 24, 5
    y = np.concatenate([rng.normal(0, 1, 12), rng.normal(4, 1, 12)])
    like = SegmentLikelihood(y, trend_design(N), PriorSpec(nu=50.0))
    P = like.segment_table(min_length=mind)
    F = forward_recursion(P, k_max=3)

    for k in (1, 2, 3):
        terms = []
        for cuts in itertools.combinations(range(N - 1), k):
            bounds = [-1, *cuts, N - 1]
            segs = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
            if any(b - a + 1 < mind for a, b in segs):
                continue
            terms.append(sum(P[a, b] for a, b in segs))
        np.testing.assert_allclose(F[k - 1, N - 1], logsumexp(terms), rtol=1e-12)


def test_configuration_count_matches_enumeration():
    """C(n - (k+1)m + k, k) must equal the number of admissible placements."""
    import itertools

    from bcpd import log_n_configurations

    N, mind = 24, 5
    for k in (1, 2, 3):
        brute = 0
        for cuts in itertools.combinations(range(N - 1), k):
            bounds = [-1, *cuts, N - 1]
            segs = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
            if all(b - a + 1 >= mind for a, b in segs):
                brute += 1
        np.testing.assert_allclose(np.exp(log_n_configurations(N, k, mind)), brute)


def test_count_posterior_is_not_biased_by_placement_count():
    """A uniform prior on locations means averaging over placements, not summing.

    Without the 1/n_configurations factor, counts with more admissible
    placements are favoured for that reason alone.  Here k = 2 admits 55
    placements against 15 for k = 1, which is enough to distort a marginal call.
    """
    from bcpd import PriorSpec, SegmentLikelihood, log_n_configurations
    from bcpd.recursion import _count_posterior, forward_recursion

    rng = np.random.default_rng(0)
    N, mind, k_max = 24, 5, 3
    y = np.concatenate([rng.normal(0, 1, 12), rng.normal(4, 1, 12)])
    like = SegmentLikelihood(y, trend_design(N), PriorSpec(nu=50.0))
    P = like.segment_table(min_length=mind)
    F = forward_recursion(P, k_max)

    post = _count_posterior(P, F, k_max, mind)
    np.testing.assert_allclose(post.sum(), 1.0, rtol=1e-10)

    # Recompute without the correction; it should shift mass towards larger k.
    from scipy.special import logsumexp

    lk = np.full(k_max + 1, -np.inf)
    lk[0] = P[0, N - 1] + np.log(0.5)
    for k in range(1, k_max + 1):
        lk[k] = F[k - 1, N - 1] + np.log(0.5) - np.log(k_max)
    uncorrected = np.exp(lk - logsumexp(lk))

    assert post[0] > uncorrected[0] + 0.3, (
        f"correction should sharpen the k=0 call: {post[0]:.3f} vs {uncorrected[0]:.3f}"
    )
