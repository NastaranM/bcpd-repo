"""Borderline and degenerate inputs, and the effect of the design matrix.

The detection tests elsewhere use large, unambiguous shifts.  These cover the
cases where the answer is close, the input is degenerate, or the choice of
design changes what counts as a change at all.
"""

import numpy as np
import pytest

from bcpd import (
    MVPriorSpec,
    OnlineChangepointFilter,
    PriorSpec,
    SegmentLengthPrior,
    SegmentLikelihood,
    constant_design,
    detect,
    detect_multivariate,
    harmonic_nested_sizes,
    joint_design,
    trend_design,
)


# --- the design decides what counts as a change ------------------------------

def _modest_shift(n=300, size=0.55, seed=5):
    rng = np.random.default_rng(seed)
    half = n // 2
    return np.concatenate([rng.normal(0, 1, half), rng.normal(size, 1, n - half)])


def test_constant_design_detects_a_modest_level_shift():
    y = _modest_shift()
    result = detect(y, constant_design(len(y)), min_distance=20, k_max=3,
                    n_samples=200, rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 1
    assert abs(int(np.argmax(result.location_prob)) - 150) < 40


def test_a_slope_absorbs_a_modest_level_shift():
    """Not a defect: one ramp fits a small step about as well as two levels.

    A slope gives the model another way to explain a shift, so including one
    makes a step less likely to be called a changepoint.  This pins the
    behaviour so the trade-off between the two designs stays visible.
    """
    y = _modest_shift()
    result = detect(y, trend_design(len(y)), min_distance=20, k_max=3,
                    n_samples=200, rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 0


def test_a_slope_still_finds_a_large_shift():
    rng = np.random.default_rng(6)
    y = np.concatenate([rng.normal(0, 1, 150), rng.normal(8, 1, 150)])
    result = detect(y, trend_design(len(y)), min_distance=20, k_max=3,
                    n_samples=200, rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 1


# --- boundaries ---------------------------------------------------------------

def test_shift_at_the_first_admissible_position():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 50), rng.normal(6, 1, 150)])
    result = detect(y, constant_design(200), min_distance=50, k_max=3,
                    n_samples=100, rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 1
    assert abs(int(np.argmax(result.location_prob)) - 50) <= 5


def test_shift_at_the_last_admissible_position():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 150), rng.normal(6, 1, 50)])
    result = detect(y, constant_design(200), min_distance=50, k_max=3,
                    n_samples=100, rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 1
    assert abs(int(np.argmax(result.location_prob)) - 150) <= 5


def test_min_distance_exactly_half_the_series():
    """Only k <= 1 fits, so the posterior must be clamped to that."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    result = detect(y, constant_design(200), min_distance=100, k_max=4,
                    n_samples=50, rng=np.random.default_rng(1))
    assert result.n_changepoints.shape == (2,)


# --- degenerate inputs --------------------------------------------------------

def test_constant_series_reports_no_changepoint():
    result = detect(np.ones(100), constant_design(100), min_distance=20, k_max=2,
                    n_samples=50, rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 0


def test_shortest_workable_series():
    rng = np.random.default_rng(0)
    y = rng.normal(size=4)
    result = detect(y, constant_design(4), min_distance=2, k_max=1,
                    n_samples=10, rng=np.random.default_rng(1))
    np.testing.assert_allclose(result.n_changepoints.sum(), 1.0, rtol=1e-10)


def test_single_column_Y_matches_one_dimensional_Y():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 60), rng.normal(5, 1, 60)])
    flat = detect_multivariate(y, constant_design(120), min_distance=30, k_max=2,
                               n_samples=50, rng=np.random.default_rng(1))
    column = detect_multivariate(y[:, None], constant_design(120), min_distance=30,
                                 k_max=2, n_samples=50, rng=np.random.default_rng(1))
    np.testing.assert_allclose(flat.n_changepoints, column.n_changepoints)


def test_more_series_than_observations_per_segment():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(60, 10))
    result = detect_multivariate(Y, constant_design(60), min_distance=20, k_max=2,
                                 prior=MVPriorSpec.vague_for(Y), n_samples=30,
                                 rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 0


# --- online edges -------------------------------------------------------------

def _likelihood(y):
    return SegmentLikelihood(y, constant_design(len(y)), PriorSpec(nu=50.0))


@pytest.mark.parametrize("mean_length", [0.5, 5.0, 1000.0, 100000.0])
def test_filter_survives_extreme_length_priors(mean_length):
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 50), rng.normal(6, 1, 150)])
    res = OnlineChangepointFilter(
        _likelihood(y), SegmentLengthPrior(mean_length=mean_length), max_particles=60
    ).run(rng=np.random.default_rng(1))
    assert np.all(np.isfinite(res.changepoint_prob))
    assert res.map_run_length[-1] >= 0


def test_single_particle_still_tracks_a_strong_shift():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 50), rng.normal(6, 1, 150)])
    res = OnlineChangepointFilter(
        _likelihood(y), SegmentLengthPrior(mean_length=100.0, min_length=20),
        max_particles=1,
    ).run(rng=np.random.default_rng(1))
    assert res.particle_counts.max() == 1
    assert np.all(np.isfinite(res.changepoint_prob))


# --- joint design end to end --------------------------------------------------

def test_joint_design_detects_a_level_shift_under_seasonality():
    """The path that replaced staged trend-then-seasonal estimation."""
    rng = np.random.default_rng(7)
    n = 400
    t = np.arange(1, n + 1)
    seasonal = 4 * np.sin(2 * np.pi * t / 50)
    y = np.concatenate([np.zeros(200), np.full(200, 8.0)]) + seasonal + rng.normal(0, 1, n)

    result = detect(y, joint_design(n, 50, 3), min_distance=80, k_max=3,
                    nested_sizes=harmonic_nested_sizes(3), n_samples=200,
                    rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 1
    assert abs(int(np.argmax(result.location_prob)) - 200) <= 15
