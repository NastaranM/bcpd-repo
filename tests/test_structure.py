"""Periodicity and cross-series dependence.

Two ways a changepoint detector can go wrong on structured data: mistaking a
recurring pattern for a sequence of changes, and missing a change that is only
visible in how series move together.
"""

import numpy as np
import pytest

from bcpd import (
    MVPriorSpec,
    PriorSpec,
    SegmentLikelihood,
    constant_design,
    detect,
    detect_multivariate,
    harmonic_nested_sizes,
    joint_design,
)

N, PERIOD = 480, 60
T = np.arange(1, N + 1)


def _seasonal(seed=11, amplitude=5.0):
    rng = np.random.default_rng(seed)
    return (
        amplitude * np.sin(2 * np.pi * T / PERIOD)
        + 2 * np.cos(2 * np.pi * 2 * T / PERIOD)
        + rng.normal(0, 0.8, N)
    )


def test_periodicity_alone_is_not_called_a_changepoint():
    """A repeating pattern with no real break must not be segmented."""
    y = _seasonal()
    result = detect(y, joint_design(N, PERIOD, 3), min_distance=70, k_max=4,
                    nested_sizes=harmonic_nested_sizes(3), n_samples=200,
                    rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 0


def test_a_mismatched_period_manufactures_changepoints():
    """A basis that cannot represent the signal gets compensated by segmenting.

    With period 120 and three harmonics the basis covers periods 120, 60 and 40,
    but the signal also has a period-30 component that cannot be represented.
    The recursion explains that leftover structure by chopping the series up.
    This is the main failure mode to watch for when the period is uncertain.
    """
    y = _seasonal()
    result = detect(y, joint_design(N, 120, 3), min_distance=70, k_max=4,
                    nested_sizes=harmonic_nested_sizes(3), n_samples=200,
                    rng=np.random.default_rng(1))
    assert result.posterior.map_count() > 0


def test_enough_harmonics_repairs_a_mismatched_period():
    """Widening the basis until it spans the signal removes the spurious breaks.

    Period 120 with six harmonics reaches period 30, so the signal is
    representable again and the fragmentation stops -- which is why the
    posterior over nested sizes is worth inspecting when breaks look suspicious.
    """
    y = _seasonal()
    like = SegmentLikelihood(y, joint_design(N, 120, 6), PriorSpec(nu=50.0),
                             nested_sizes=harmonic_nested_sizes(6))
    one = like.log_marginal(0, N)
    split = like.log_marginal(0, 240) + like.log_marginal(240, N)
    assert one > split, "one segment should win once the basis spans the signal"


def test_a_level_shift_under_seasonality_is_still_found():
    rng = np.random.default_rng(12)
    y = (5 * np.sin(2 * np.pi * T / PERIOD)
         + np.concatenate([np.zeros(240), np.full(240, 6.0)])
         + rng.normal(0, 0.8, N))
    result = detect(y, joint_design(N, PERIOD, 3), min_distance=70, k_max=4,
                    nested_sizes=harmonic_nested_sizes(3), n_samples=200,
                    rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 1
    assert abs(int(np.argmax(result.location_prob)) - 240) <= 15


def test_a_change_in_seasonal_amplitude_is_found():
    """The mean is unchanged; only the size of the swing differs."""
    rng = np.random.default_rng(13)
    y = np.concatenate([
        2.0 * np.sin(2 * np.pi * T[:240] / PERIOD),
        9.0 * np.sin(2 * np.pi * T[240:] / PERIOD),
    ]) + rng.normal(0, 0.8, N)
    result = detect(y, joint_design(N, PERIOD, 3), min_distance=70, k_max=4,
                    nested_sizes=harmonic_nested_sizes(3), n_samples=200,
                    rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 1
    assert abs(int(np.argmax(result.location_prob)) - 240) <= 15


# --- cross-series dependence --------------------------------------------------

def _correlated(shift_size, n=400, d=4, rho=0.9, seed=31):
    """Series with strongly common-mode noise and an anti-correlated shift.

    The shift lies in a low-variance direction of Sigma, so it is small relative
    to each series' own spread but large relative to how the series usually move
    together.
    """
    rng = np.random.default_rng(seed)
    Sigma = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
    Y = rng.multivariate_normal(np.zeros(d), Sigma, size=n)
    Y[n // 2:] += np.array([1.0, -1.0] * (d // 2)) * shift_size
    return Y


@pytest.mark.parametrize("shift", [0.4, 0.2])
def test_correlation_reveals_a_shift_invisible_in_any_single_series(shift):
    Y = _correlated(shift)
    n = len(Y)
    joint = detect_multivariate(Y, constant_design(n), min_distance=60, k_max=3,
                                prior=MVPriorSpec.vague_for(Y), n_samples=150,
                                rng=np.random.default_rng(1))
    assert joint.n_changepoints[1] > 0.9
    assert abs(int(np.argmax(joint.location_prob)) - n // 2) <= 5

    best_alone = max(
        detect(Y[:, k], constant_design(n), min_distance=60, k_max=3,
               n_samples=150, rng=np.random.default_rng(1)).n_changepoints[1]
        for k in range(Y.shape[1])
    )
    assert best_alone < 0.2, f"expected the marginal view to miss it, got {best_alone:.3f}"


def test_a_change_in_covariance_alone_is_detected():
    """Means are identical either side; only the dependence structure changes.

    Sigma is estimated per segment under an unstructured inverse-Wishart prior,
    so its off-diagonal entries are free to differ between segments.  A model
    holding Sigma diagonal, or fixed across segments, could not see this.
    """
    n, d, rho = 300, 4, 0.85
    rng = np.random.default_rng(32)
    uncorrelated = rng.multivariate_normal(np.zeros(d), np.eye(d), size=n // 2)
    correlated = rng.multivariate_normal(
        np.zeros(d), (1 - rho) * np.eye(d) + rho * np.ones((d, d)), size=n // 2
    )
    Y = np.vstack([uncorrelated, correlated])

    # The per-series means barely move, so this is not a mean shift in disguise.
    assert np.max(np.abs(uncorrelated.mean(0) - correlated.mean(0))) < 0.3

    result = detect_multivariate(Y, constant_design(n), min_distance=40, k_max=3,
                                 prior=MVPriorSpec.vague_for(Y), n_samples=200,
                                 rng=np.random.default_rng(1))
    assert result.posterior.map_count() == 1
    assert abs(int(np.argmax(result.location_prob)) - n // 2) <= 15
