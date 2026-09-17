"""Online changepoint detection: the Fearnhead & Liu (2007) particle filter.

Where ``bcpd.recursion`` works offline -- it sees all N observations, builds the
full segment table and recurses over it -- this module processes observations
one at a time and maintains a posterior over the changepoint in progress.

The segment model is unchanged.  Everything here needs from the likelihood is
``log_marginal(start, stop)``, so ``SegmentLikelihood`` and
``MultivariateSegmentLikelihood`` both drop in as they are.  The one-step
predictive the filter requires is a ratio of two segment likelihoods:

    p(y_t | y_{s:t-1}) = P(y_{s:t}) / P(y_{s:t-1})

which in logs is a single subtraction.  The cumulative sufficient statistics
already in those classes make each such call O(q^3) regardless of how long the
segment has grown, which is what keeps the filter affordable.

What does change is the prior.  The offline model puts a uniform prior on the
*number* of changepoints, which cannot be evaluated until all the data is in.  Here, segment lengths are iid draws from g(.), a renewal
prior (Fearnhead 2006), which can be evaluated as data arrives.

Relationship to Adams & MacKay
------------------------------
Both papers appeared in 2007 and derive the same filtering recursion
independently.  They differ in two ways that matter:

* Adams & MacKay prune particles below a probability threshold.  Fearnhead &
  Liu use the optimal resampling of Fearnhead & Clifford (2003), which keeps
  every sufficiently large weight exactly and stratified-samples the rest --
  minimising expected squared error rather than discarding by fiat.
* Adams & MacKay return the run-length posterior.  Fearnhead & Liu add a
  backward simulation step that draws complete changepoint configurations from
  the joint posterior, which is what makes online output directly comparable
  with the offline algorithm's.

References
----------
Fearnhead, P. & Liu, Z. (2007). On-line inference for multiple changepoint
problems. JRSS-B 69(4), 589-605.
Fearnhead, P. & Clifford, P. (2003). On-line inference for hidden Markov models
via particle filters. JRSS-B 65(4), 887-899.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "SegmentLengthPrior",
    "OnlineChangepointFilter",
    "OnlineResult",
    "optimal_resample",
]


@dataclass(frozen=True)
class SegmentLengthPrior:
    """Negative-binomial prior on how long a segment lasts.

    The filter consumes this through its *hazard*: the chance the current
    segment ends now, given it has lasted this long.

        h(l) = Pr(L = l | L >= l)

    Two knobs, both of which describe the data rather than the algorithm.

    ``mean_length`` is the typical number of observations between changes.  It
    has no default because only you know it, and it is not scale-free: a value
    that is right for a daily series is wrong for one sampled every second.  If
    you have no idea, a fraction of the series length is a defensible starting
    point, but say so and check how much the answer moves when you vary it.

    ``shape`` says whether segments have a characteristic duration.

    * ``shape = 1`` gives a geometric prior: constant hazard, memoryless.  A
      segment that has run for a long time is no more likely to end than a
      fresh one.  This is the least committal choice and the default.  It suits
      changes that arrive at random.
    * ``shape > 1`` gives a rising hazard: short segments become improbable and
      a segment that has already outlasted ``mean_length`` is increasingly
      likely to end.  This suits changes that recur on a cadence.  Larger values
      concentrate lengths more tightly around ``mean_length``.

    ``min_length`` is separate and does a different job.  ``shape`` makes short
    segments *improbable*; ``min_length`` makes them *impossible*.  Use it only
    for a genuine constraint -- a resolution limit, a physical minimum dwell
    time -- rather than as a way to express a mere preference.  It defaults to 1,
    meaning no constraint.

    Parameters
    ----------
    mean_length : float
        Typical segment length, in observations.  Must be positive.
    shape : float
        Negative-binomial shape.  1 is geometric; larger concentrates lengths
        around ``mean_length``.  Must be positive.
    min_length : int
        Hard lower bound on segment length.  1 means unconstrained.
    max_run_length : int
        Size of the precomputed hazard table.  Beyond it the hazard is held at
        its limiting value, which the negative binomial approaches from below.
    """

    mean_length: float
    shape: float = 1.0
    min_length: int = 1
    max_run_length: int = 10_000

    def __post_init__(self):
        if self.mean_length <= 0:
            raise ValueError(f"mean_length must be positive; got {self.mean_length}")
        if self.shape <= 0:
            raise ValueError(f"shape must be positive; got {self.shape}")
        if self.min_length < 1:
            raise ValueError(f"min_length must be >= 1; got {self.min_length}")
        object.__setattr__(self, "_p", self.shape / (self.shape + self.mean_length))
        object.__setattr__(self, "_table", self._build_table())

    def _build_table(self) -> np.ndarray:
        """Hazard at each run length, computed in logs to survive the tail.

        With ``L = k + 1`` for ``k`` negative-binomial on 0, 1, 2, ...:

            Pr(L = l)  = pmf(l - 1)
            Pr(L >= l) = Pr(k > l - 2) = sf(l - 2)

        so ``h(l) = exp(logpmf(l - 1) - logsf(l - 2))``.  Taking the survival
        from ``logsf`` rather than from ``log(1 - cdf)`` matters: the latter
        underflows to zero once the cdf rounds to 1, which silently kills the
        hazard in the tail.  Where even ``logsf`` underflows, the hazard has
        already converged to its limit ``p``, which it approaches from below.

        Truncation at ``min_length`` does not alter the hazard above it: both
        numerator and denominator of h(l) are rescaled by the same constant, so
        the conditional rate is unchanged and only the zeros below move.
        """
        from scipy.stats import nbinom

        l = np.arange(1, self.max_run_length + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_pmf = nbinom.logpmf(l - 1, self.shape, self._p)
            log_sf = nbinom.logsf(l - 2, self.shape, self._p)
            h = np.exp(log_pmf - log_sf)

        # Past the point where the survival underflows, hold the limiting rate.
        h = np.where(np.isfinite(h), h, self._p)
        h[: self.min_length - 1] = 0.0
        return np.clip(h, 0.0, 1.0 - 1e-12)

    def hazard(self, run_length: int) -> float:
        """Pr(L = r | L >= r): the chance this segment ends now."""
        if run_length < self.min_length:
            return 0.0
        if run_length > self.max_run_length:
            return float(self._table[-1])
        return float(self._table[run_length - 1])

    def log_hazard(self, run_length: int) -> float:
        h = self.hazard(run_length)
        return -np.inf if h <= 0.0 else float(np.log(h))

    def log_survival(self, run_length: int) -> float:
        """log Pr(L > r | L >= r): the factor for surviving one more step."""
        h = self.hazard(run_length)
        return 0.0 if h <= 0.0 else float(np.log1p(-h))


def optimal_resample(
    log_weights: np.ndarray, n_keep: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Fearnhead & Clifford (2003) stratified optimal resampling.

    Reduces M weighted particles to at most ``n_keep``, minimising the expected
    sum of squared errors in the weights.  The result is unbiased: each
    surviving particle's expected weight equals its original weight.

    The algorithm finds the threshold ``alpha`` solving

        sum_i min(w_i / alpha, 1) = n_keep

    Particles at or above ``alpha`` are kept deterministically **at their own
    weight** -- no sampling noise is introduced on the particles that carry the
    mass.  Those below are stratified-sampled to fill the remaining slots and
    each survivor is assigned weight exactly ``alpha``.

    This is the substantive difference from Adams & MacKay, who instead drop
    particles whose weight falls below a fixed threshold.  Thresholding is
    biased -- the discarded mass is simply lost -- and it can drop a particle
    that later matters.

    Parameters
    ----------
    log_weights : (M,) array
        Unnormalised log weights.
    n_keep : int
        Target particle count.
    rng : numpy Generator

    Returns
    -------
    indices : (K,) int array
        Positions of the surviving particles, K <= n_keep.
    new_log_weights : (K,) array
        Their log weights, on the same scale as the input (the total is
        preserved, so downstream normalisation is unaffected).
    """
    M = len(log_weights)
    if M <= n_keep:
        return np.arange(M), log_weights.copy()

    total = logsumexp(log_weights)
    w = np.exp(log_weights - total)  # normalised, sums to 1

    order = np.argsort(w)[::-1]
    w_sorted = w[order]
    cum = np.cumsum(w_sorted)

    # Find how many particles sit at or above the threshold.  For a candidate
    # count k, the remaining mass 1 - cum[k-1] is spread over n_keep - k slots,
    # giving alpha; k is correct when w_(k) >= alpha > w_(k+1).
    alpha = 1.0 / n_keep
    n_big = 0
    for k in range(n_keep):
        remaining = 1.0 - (cum[k - 1] if k > 0 else 0.0)
        slots = n_keep - k
        if slots <= 0:
            break
        candidate = remaining / slots
        if w_sorted[k] < candidate:
            alpha, n_big = candidate, k
            break
    else:
        alpha = (1.0 - cum[n_keep - 1]) / 1.0 if n_keep < M else alpha
        n_big = n_keep

    kept = list(order[:n_big])
    kept_logw = [log_weights[i] for i in kept]

    # Stratified pass over the remainder: walk the cumulative weight of the
    # small particles and take one sample per interval of width alpha.
    small = order[n_big:]
    if len(small) and n_keep - n_big > 0:
        w_small = w[small]
        csum = np.cumsum(w_small)
        u = rng.uniform(0.0, alpha)
        n_slots = n_keep - n_big
        targets = u + alpha * np.arange(n_slots)
        picks = np.searchsorted(csum, targets)
        picks = picks[picks < len(small)]
        for i in np.unique(picks):
            kept.append(small[i])
            kept_logw.append(np.log(alpha) + total)

    idx = np.array(kept, dtype=int)
    return idx, np.array(kept_logw, dtype=float)


@dataclass
class OnlineResult:
    """Output of a pass of the online filter.

    Attributes
    ----------
    run_length_prob : list of dict
        One entry per time step, mapping segment start index to its posterior
        probability.  Sparse because the filter prunes.
    map_run_length : (N,) int array
        Most probable run length at each step.  A drop back towards zero is the
        filter announcing a changepoint.
    changepoint_prob : (N,) array
        Posterior probability at each step that a changepoint occurred there,
        i.e. that the current segment had just started.
    particle_counts : (N,) int array
        Surviving particles per step; useful for checking that resampling is
        actually binding.
    """

    run_length_prob: list[dict[int, float]] = field(default_factory=list)
    map_run_length: np.ndarray | None = None
    changepoint_prob: np.ndarray | None = None
    particle_counts: np.ndarray | None = None


class OnlineChangepointFilter:
    """Sequential changepoint filter over a segment likelihood.

    Parameters
    ----------
    likelihood
        Anything exposing ``log_marginal(start, stop)`` for a half-open row
        range -- ``SegmentLikelihood`` or ``MultivariateSegmentLikelihood``.
    prior : SegmentLengthPrior
    max_particles : int
        Particle budget.  The exact filter grows one particle per step and so
        costs O(N^2); capping it at ``max_particles`` makes the cost linear at
        the price of a small approximation.  Set to ``None`` to run exactly,
        which is worth doing on short series to measure what the approximation
        costs.
    """

    def __init__(
        self,
        likelihood,
        prior: SegmentLengthPrior,
        max_particles: int | None = 200,
    ):
        self.likelihood = likelihood
        self.prior = prior
        self.max_particles = max_particles
        self.N = likelihood.N

    def _log_predictive(self, start: int, t: int) -> float:
        """log p(y_t | y_{start:t-1}), as a ratio of segment likelihoods."""
        numer = self.likelihood.log_marginal(start, t + 1)
        if start == t:
            return numer  # new segment: no conditioning data yet
        return numer - self.likelihood.log_marginal(start, t)

    def run(
        self, rng: np.random.Generator | None = None, store_history: bool = True
    ) -> OnlineResult:
        """Filter the whole series, one observation at a time.

        Set ``store_history`` to keep every step's particle set, which
        ``backward_sample`` needs.  Turn it off for a pure streaming pass where
        only the run-length posterior matters.
        """
        if rng is None:
            rng = np.random.default_rng()
        prior = self.prior

        starts = np.array([0])  # segment start index per particle
        logw = np.array([0.0])  # log P(C_t = s, y_0:t), unnormalised

        history: list[tuple[np.ndarray, np.ndarray]] = []
        run_length_prob: list[dict[int, float]] = []
        map_run_length = np.zeros(self.N, dtype=int)
        changepoint_prob = np.zeros(self.N)
        particle_counts = np.zeros(self.N, dtype=int)

        for t in range(self.N):
            if t == 0:
                new_starts = np.array([0])
                new_logw = np.array([self._log_predictive(0, 0)])
            else:
                run_lengths = t - starts  # length as at time t-1

                # (a) every particle's segment continues and absorbs y_t
                grow_logw = np.array(
                    [
                        logw[i]
                        + self._log_predictive(int(starts[i]), t)
                        + prior.log_survival(int(run_lengths[i]))
                        for i in range(len(starts))
                    ]
                )

                # (b) some segment ended at t-1 and a new one starts at t;
                #     all old particles contribute to this single new one
                end_logw = np.array(
                    [
                        logw[i] + prior.log_hazard(int(run_lengths[i]))
                        for i in range(len(starts))
                    ]
                )
                if np.any(np.isfinite(end_logw)):
                    cp_logw = logsumexp(end_logw) + self._log_predictive(t, t)
                    new_starts = np.append(starts, t)
                    new_logw = np.append(grow_logw, cp_logw)
                else:
                    # still inside the dead zone: no changepoint is admissible
                    new_starts, new_logw = starts, grow_logw

            if self.max_particles is not None and len(new_starts) > self.max_particles:
                idx, new_logw = optimal_resample(new_logw, self.max_particles, rng)
                new_starts = new_starts[idx]

            starts, logw = new_starts, new_logw

            norm = logw - logsumexp(logw)
            probs = np.exp(norm)
            run_length_prob.append({int(s): float(p) for s, p in zip(starts, probs)})
            map_run_length[t] = t - int(starts[np.argmax(probs)])
            # At t = 0 the single particle trivially starts at 0; that is the
            # series beginning, not a detected changepoint.
            changepoint_prob[t] = 0.0 if t == 0 else float(probs[starts == t].sum())
            particle_counts[t] = len(starts)

            if store_history:
                history.append((starts.copy(), logw.copy()))

        self._history = history if store_history else None
        return OnlineResult(
            run_length_prob=run_length_prob,
            map_run_length=map_run_length,
            changepoint_prob=changepoint_prob,
            particle_counts=particle_counts,
        )

    def backward_sample(
        self, n_samples: int = 500, rng: np.random.Generator | None = None
    ) -> list[list[int]]:
        """Draw complete changepoint configurations from the joint posterior.

        The filter alone gives a run-length posterior -- how long since the last
        change.  This recovers the same object the offline algorithm produces: a
        sample of full changepoint sets, so the two can be compared directly.

        Starting from the final particle set, draw the current segment's start,
        then step back.  Conditioning on a segment starting at ``s`` means the
        previous segment ended *exactly* at ``s - 1``, so its stored weight --
        which carries "length at least r" -- is multiplied by the hazard to
        convert it to "length exactly r".

        This is the same stochastic backtrace as the offline algorithm, run over
        the filter's stored weights rather than the offline segment table.

        Requires ``run(store_history=True)`` to have been called first.
        """
        if getattr(self, "_history", None) is None:
            raise RuntimeError("call run(store_history=True) before backward_sample")
        if rng is None:
            rng = np.random.default_rng()

        draws: list[list[int]] = []
        final_starts, final_logw = self._history[-1]
        final_p = np.exp(final_logw - logsumexp(final_logw))

        for _ in range(n_samples):
            locations: list[int] = []
            s = int(final_starts[rng.choice(len(final_starts), p=final_p)])

            while s > 0:
                locations.append(s)
                starts, logw = self._history[s - 1]
                run_lengths = s - starts
                adj = np.array(
                    [
                        logw[i] + self.prior.log_hazard(int(run_lengths[i]))
                        for i in range(len(starts))
                    ]
                )
                finite = np.isfinite(adj)
                if not finite.any():
                    break
                p = np.zeros(len(adj))
                p[finite] = np.exp(adj[finite] - logsumexp(adj[finite]))
                s = int(starts[rng.choice(len(starts), p=p)])

            draws.append(sorted(locations))
        return draws
