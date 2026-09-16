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
