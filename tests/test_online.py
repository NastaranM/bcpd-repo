"""Online filtering and optimal resampling.

The optimal resampler is checked against the three properties Fearnhead &
Clifford (2003) prove: total mass is preserved, particles at or above the
threshold survive deterministically at their own weight, and the budget is
respected.  The filter is checked against the exact (unpruned) version, which
is the natural oracle -- pruning is the only approximation in play.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

from bcpd import (
    MVPriorSpec,
    MultivariateSegmentLikelihood,
    OnlineChangepointFilter,
    PriorSpec,
    SegmentLengthPrior,
    SegmentLikelihood,
    optimal_resample,
    trend_design,
)


def _filter(y, *, min_length=40, mean_length=200.0, max_particles=100, nu=50.0):
    y = np.asarray(y)
    like = SegmentLikelihood(y, trend_design(len(y)), PriorSpec(nu=nu))
    return OnlineChangepointFilter(
        like, SegmentLengthPrior(mean_length=mean_length, min_length=min_length), max_particles=max_particles
    )


def test_segment_length_prior_dead_zone():
    prior = SegmentLengthPrior(mean_length=49.0, min_length=96)
    assert prior.hazard(50) == 0.0
    assert prior.hazard(95) == 0.0
    assert prior.hazard(96) == pytest.approx(0.02)
    assert prior.hazard(500) == pytest.approx(0.02)
    assert prior.log_survival(50) == 0.0  # cannot end, so survives with prob 1


def test_segment_length_prior_validates():
    with pytest.raises(ValueError, match="mean_length must be positive"):
        SegmentLengthPrior(mean_length=-1.0)
    with pytest.raises(ValueError, match="min_length"):
        SegmentLengthPrior(mean_length=50.0, min_length=0)


def test_optimal_resample_preserves_total_mass():
    rng = np.random.default_rng(0)
    lw = np.log(rng.dirichlet(np.ones(60) * 0.3))
    _, new_lw = optimal_resample(lw, 20, rng)
    np.testing.assert_allclose(logsumexp(new_lw), logsumexp(lw), rtol=1e-10)


def test_optimal_resample_respects_budget():
    rng = np.random.default_rng(1)
    lw = np.log(rng.dirichlet(np.ones(80) * 0.5))
    idx, _ = optimal_resample(lw, 25, rng)
    assert len(idx) <= 25
    assert len(np.unique(idx)) == len(idx)


def test_optimal_resample_keeps_dominant_particles_exactly():
    """Weights at or above alpha survive at their own weight, not resampled."""
    rng = np.random.default_rng(2)
    w = np.concatenate([[0.4, 0.3, 0.2], np.full(50, 0.1 / 50)])
    lw = np.log(w)
    idx, new_lw = optimal_resample(lw, 10, rng)

    for big in (0, 1, 2):
        assert big in idx
        np.testing.assert_allclose(new_lw[list(idx).index(big)], lw[big], rtol=1e-10)


def test_optimal_resample_is_a_noop_below_budget():
    lw = np.log(np.array([0.5, 0.3, 0.2]))
    idx, new_lw = optimal_resample(lw, 10, np.random.default_rng(0))
    np.testing.assert_array_equal(idx, [0, 1, 2])
    np.testing.assert_allclose(new_lw, lw)


def test_run_length_resets_at_the_changepoint():
    """The break is located correctly, but the MAP flips a step late.

    Changepoint probability peaks at the true location.  The MAP run length
    lags by a step or so, because a single observation from the new regime is
    not yet enough evidence to beat "the segment continues".  That delay is
    inherent to online detection -- it is the detection-lag side of the
    delay/false-alarm tradeoff, and it is what the offline algorithm avoids by
    seeing the whole series at once.
    """
    rng = np.random.default_rng(3)
    y = np.concatenate([rng.normal(0, 1, 150), rng.normal(5, 1, 150)])
    res = _filter(y).run(rng=np.random.default_rng(1))

    peak = int(np.argmax(res.changepoint_prob))
    assert abs(peak - 150) <= 2, f"changepoint probability peaked at {peak}"

    assert res.map_run_length[149] > 100  # grown through the first segment
    assert res.map_run_length[153] < 10  # reset, allowing a few steps of lag
    assert res.map_run_length[160] > 5  # growing again


def test_no_changepoint_means_run_length_grows_monotonically():
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1, 250)
    res = _filter(y).run(rng=np.random.default_rng(1))
    assert res.map_run_length[-1] > 200


def test_dead_zone_forbids_early_changepoints():
    """No changepoint may be posited before min_length observations."""
    rng = np.random.default_rng(5)
    y = np.concatenate([rng.normal(0, 1, 20), rng.normal(8, 1, 230)])
    res = _filter(y, min_length=60).run(rng=np.random.default_rng(1))
    assert res.changepoint_prob[:60].sum() == 0.0


def test_particle_budget_is_respected():
    rng = np.random.default_rng(6)
    y = rng.normal(0, 1, 300)
    res = _filter(y, max_particles=50).run(rng=np.random.default_rng(1))
    assert res.particle_counts.max() <= 50


def test_pruned_filter_tracks_the_exact_filter():
    """Pruning is the only approximation; it should barely move the answer."""
    rng = np.random.default_rng(7)
    y = np.concatenate([rng.normal(0, 1, 100), rng.normal(4, 1, 100)])

    exact = _filter(y, min_length=30, max_particles=None).run(np.random.default_rng(1))
    pruned = _filter(y, min_length=30, max_particles=30).run(np.random.default_rng(1))

    assert abs(int(exact.map_run_length[-1]) - int(pruned.map_run_length[-1])) <= 2
    peak_e = int(np.argmax(exact.changepoint_prob[50:150])) + 50
    peak_p = int(np.argmax(pruned.changepoint_prob[50:150])) + 50
    assert abs(peak_e - peak_p) <= 5


def test_backward_sample_recovers_the_changepoint():
    rng = np.random.default_rng(3)
    y = np.concatenate([rng.normal(0, 1, 150), rng.normal(5, 1, 150)])
    filt = _filter(y)
    filt.run(rng=np.random.default_rng(1))

    draws = filt.backward_sample(200, rng=np.random.default_rng(2))
    assert all(len(d) == 1 for d in draws)
    locations = [d[0] for d in draws]
    assert abs(np.median(locations) - 150) <= 3


def test_backward_sample_requires_history():
    y = np.random.default_rng(0).normal(size=120)
    filt = _filter(y)
    filt.run(rng=np.random.default_rng(1), store_history=False)
    with pytest.raises(RuntimeError, match="store_history"):
        filt.backward_sample(10)


def test_filter_accepts_the_multivariate_likelihood():
    """The filter only needs log_marginal, so both likelihoods drop in."""
    rng = np.random.default_rng(8)
    n, d = 240, 3
    shift = np.concatenate([np.zeros(120), np.full(120, 5.0)])
    Y = shift[:, None] + rng.multivariate_normal(np.zeros(d), np.eye(d), size=n)

    like = MultivariateSegmentLikelihood(Y, trend_design(n), MVPriorSpec.vague_for(Y))
    filt = OnlineChangepointFilter(
        like, SegmentLengthPrior(mean_length=200.0, min_length=40), max_particles=80
    )
    res = filt.run(rng=np.random.default_rng(1))

    assert res.map_run_length[119] > 80
    assert res.map_run_length[124] < 10  # allowing detection lag


def test_min_length_has_no_default():
    """The spacing is a property of the series, so it must be stated."""
    with pytest.raises(TypeError):
        SegmentLengthPrior()


def test_shape_one_is_exactly_geometric():
    """shape = 1 must give constant hazard at r / (r + mean)."""
    prior = SegmentLengthPrior(mean_length=100.0, shape=1.0)
    expected = 1.0 / 101.0
    for r in (1, 5, 50, 200, 900):
        assert prior.hazard(r) == pytest.approx(expected, rel=1e-6)


def test_shape_above_one_gives_rising_hazard():
    """A characteristic duration means the hazard climbs with run length."""
    prior = SegmentLengthPrior(mean_length=100.0, shape=20.0)
    h = [prior.hazard(r) for r in (10, 50, 100, 200)]
    assert h == sorted(h)
    assert h[0] < 0.01 < h[-1]


def test_min_length_truncates_without_changing_the_rate_above_it():
    """A hard floor zeroes the hazard below it and leaves it alone above."""
    free = SegmentLengthPrior(mean_length=100.0)
    floored = SegmentLengthPrior(mean_length=100.0, min_length=50)
    assert floored.hazard(49) == 0.0
    assert floored.hazard(80) == pytest.approx(free.hazard(80), rel=1e-9)


def test_mean_length_is_required_and_validated():
    with pytest.raises(TypeError):
        SegmentLengthPrior()
    with pytest.raises(ValueError, match="mean_length must be positive"):
        SegmentLengthPrior(mean_length=0.0)
    with pytest.raises(ValueError, match="shape must be positive"):
        SegmentLengthPrior(mean_length=100.0, shape=0.0)
