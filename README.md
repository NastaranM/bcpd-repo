# bcpd — Exact Bayesian Changepoint Detection

Find the points where a time series changes regime, with calibrated uncertainty
about how many changes there are and where each one sits.

Two algorithms over one shared segment model:

- **Offline** — dynamic programming over all segmentations. Exact, no MCMC, and
  the posterior draws are independent, so there is no burn-in and no convergence
  to diagnose.
- **Online** — a particle filter that processes one observation at a time and
  maintains a posterior over how long the current regime has lasted.

Both work on a single series or on several series that change together.

## Install

```bash
pip install -e ".[dev]"
pytest
```

Requires Python 3.10+, numpy and scipy.

## Quick start

```python
import numpy as np
from bcpd import detect, trend_design

y = np.loadtxt("series.txt")
result = detect(y, trend_design(len(y)), min_distance=50, k_max=4)

result.n_changepoints        # posterior over 0..k_max changes
result.location_prob         # per-observation probability of a change
result.posterior.map_count() # most probable number of changes
```

Online, for streaming data:

```python
from bcpd import OnlineChangepointFilter, SegmentLengthPrior, SegmentLikelihood, PriorSpec

likelihood = SegmentLikelihood(y, trend_design(len(y)), PriorSpec())
filt = OnlineChangepointFilter(
    likelihood, SegmentLengthPrior(mean_length=200), max_particles=200
)
res = filt.run()

res.map_run_length    # observations since the last change, per step
res.changepoint_prob  # probability of a change at each step
filt.backward_sample(500)  # complete changepoint configurations
```

## The model

Each segment is a linear-Gaussian model with conjugate priors:

    y = X beta + e,   e ~ N(0, sigma^2 I)
    beta | sigma^2 ~ N(0, sigma^2 nu I),   sigma^2 ~ InvGamma(c1, c2)

Both parameters integrate out, leaving a closed-form marginal likelihood per
segment. That closed form is what makes exact inference possible: no sampling is
needed inside a segment, so the algorithms only have to reason about where the
boundaries are.

You supply `X`. `constant_design` gives an intercept, `trend_design` adds a
slope, `seasonal_design` gives Fourier terms, and `joint_design` combines trend
and seasonal. Any design matrix works, so the same machinery covers a change in
level, in trend, in seasonal amplitude, or in regression coefficients.

### The design decides what counts as a change

This is the choice that most affects your results, and it is easy to get wrong.

A segment can only be distinguished from its neighbour by something the design
cannot already explain. Give each segment a slope, and a moderate level shift
can be explained as gradual drift within one segment instead — so it will not be
flagged.

The effect is not subtle. On 300 observations with a 0.55 sd level shift at the
midpoint:

| Design | Posterior on one changepoint |
| --- | --- |
| `constant_design` | 0.998 |
| `trend_design` | 0.001 |

Neither answer is wrong. One straight line genuinely does fit a small step about
as well as two levels, using half the parameters, and the marginal likelihood
prefers the simpler account. But if what you want is a level-shift detector,
`trend_design` is the wrong tool.

**Use `constant_design` unless your segments genuinely trend** and you want
changes in slope detected.

### Periodicity is not a changepoint

A recurring pattern is not a sequence of changes, and the way to say so is to
put the periodicity in the design. `joint_design` adds Fourier terms, so a
segment can carry a cycle and the recursion has no reason to chop it up.

This only works if the basis can actually represent the signal. When it cannot,
the recursion explains the leftover structure the one way it can — by
segmenting. On a period-60 signal:

| Basis | Changepoints found |
| --- | --- |
| period 60, 3 harmonics | 0 (correct) |
| period 120, 3 harmonics | 4 (spurious) |
| period 120, 6 harmonics | 0 (correct) |

The middle row fails because periods 120, 60 and 40 cannot reach the signal's
period-30 component. Widening the basis until it spans the signal fixes it.

**So spurious-looking changepoints are a symptom worth diagnosing rather than
accepting.** Check `posterior_size`: if segments are consistently choosing the
largest available harmonic order, the basis is too narrow and the breaks may be
an artefact of that rather than anything in the data.

Genuine seasonal changes are still found — a level shift underneath a cycle, or a
change in the amplitude of the cycle with no change in mean.

## Multiple series

`detect_multivariate` handles d series that change at the same times:

    Y = X B + E,   rows of E iid N_d(0, Sigma)
    B | Sigma ~ MN(0, nu I, Sigma),   Sigma ~ InverseWishart(Psi, nu0)

`Sigma` is estimated per segment under an unstructured inverse-Wishart prior, so
its off-diagonal entries are free: the model uses how the series move together,
not just each one's own spread. Two consequences.

**A change can be invisible in every series and obvious jointly.** With four
series at correlation 0.9, and a shift that pushes them in opposite directions —
a direction the common-mode noise rarely produces:

| Shift | Joint model | Best single series |
| --- | --- | --- |
| 0.6 sd | 1.000 | 0.990 |
| 0.4 sd | 1.000 | 0.052 |
| 0.2 sd | 1.000 | 0.001 |

**A change in dependence alone is detectable.** If the series become correlated
partway through while every mean stays put, that is a changepoint here. A model
holding `Sigma` diagonal, or fixed across segments, would see nothing.

```python
from bcpd import detect_multivariate, MVPriorSpec, joint_design

X = joint_design(len(Y), period=24, harmonic_order=3)
result = detect_multivariate(Y, X, min_distance=50, prior=MVPriorSpec.vague_for(Y))
result.likelihood.posterior_B(start, stop)   # q x d coefficients per segment
```

`nu0` cannot be made arbitrarily vague the way a univariate shape parameter can:
an inverse-Wishart needs `nu0 > d - 1` to be proper and `nu0 > d + 1` for its
mean to exist, so the dimension sets a floor. `MVPriorSpec.vague_for` takes
`nu0 = d + 2` and scales `Psi` by the mean marginal variance.

## Choosing the settings

Three parameters have no defaults, because no default could be right for every
series. They describe your data, not the algorithm.

**`min_distance`** (offline) — the closest two changes could plausibly be, in
observations. A value right for one sampling rate is wrong for another. A
spacing too large to admit even one change raises an error rather than silently
returning an empty posterior; a `k_max` larger than the series can support is
clamped to what fits.

**`mean_length`** (online) — the typical number of observations between changes.
If you have no idea, a fraction of the series length is a defensible starting
point, but say so and check how much the answer moves when you vary it.

**`shape`** (online) — whether regimes have a characteristic duration.

| `shape` | Hazard | Use when |
| --- | --- | --- |
| 1 (default) | constant — memoryless | changes arrive at random |
| > 1 | rises with run length | changes recur on a cadence |

At `shape = 1` the prior is geometric and a segment that has run a long time is
no more likely to end than a fresh one. Above 1, short segments become
improbable and a segment that has outlasted `mean_length` is increasingly likely
to end. Larger values concentrate lengths more tightly.

`min_length` is separate and does a different job: `shape` makes short segments
*improbable*, `min_length` makes them *impossible*. Use it only for a genuine
constraint such as a resolution limit, not to express a preference.

## Nested designs

`nested_sizes` averages the segment likelihood over a nested family of designs
under a uniform prior, so different segments can use different model
complexities. The main use is Fourier terms, where it becomes a prior on the
harmonic order:

```python
from bcpd import detect, joint_design, harmonic_nested_sizes

X = joint_design(len(y), period=24, harmonic_order=4)
result = detect(y, X, min_distance=50, nested_sizes=harmonic_nested_sizes(4))
result.likelihood.posterior_size(start, stop)   # posterior over harmonic order
```

This respects the ordering of harmonics and keeps each sin/cos pair intact, at a
cost of one evaluation per family member.

## Online versus offline

They answer different questions and the difference matters.

Offline gives a posterior over the complete set of changepoints, using all the
data for every one of them. Online gives a run-length posterior — how long since
the last change — updated as data arrives.

The online filter detects at the right location but flips a step or two late:
one observation from a new regime is not yet enough evidence to outweigh "the
current segment continues". That delay is inherent to sequential detection and
is the price of not waiting for the whole series.

`backward_sample` recovers complete changepoint configurations from the online
filter, so its output can be compared directly with the offline algorithm's.

The two also differ in their prior. Offline places a uniform prior on the
*number* of changepoints, which cannot be evaluated until all data is in. Online
places a renewal prior on segment *lengths*, which can.

## Approximation and its cost

The only approximation in either algorithm is the online particle budget. The
exact filter grows one particle per observation and so costs O(N²); capping at
`max_particles` makes it linear.

The pruning uses stratified optimal resampling (Fearnhead & Clifford 2003)
rather than dropping particles below a fixed threshold. Particles at or above the
threshold survive deterministically at their own weight, so no sampling noise is
introduced on the particles carrying the mass; the rest are stratified-sampled
and total mass is preserved. Thresholding is biased — the discarded mass is
simply lost — and can drop a particle that later matters.

Set `max_particles=None` to run exactly, which is worth doing on short series to
measure what the approximation costs.

## Validation

The segment marginal likelihood is checked against numerical integration of both
parameters, so the implementation is pinned to the integral rather than to a
transcription of it. The multivariate version is checked two ways: at d = 1 it
must reproduce the univariate likelihood exactly, and for d > 1 it is compared
against Monte Carlo over the prior using an identity that does not go through the
step being tested. The pruned online filter is checked against the exact filter.

## References

- Barry, D. & Hartigan, J. A. (1992). Product partition models for change point
  problems. *The Annals of Statistics* 20(1), 260–279.
- Fearnhead, P. (2006). Exact and efficient Bayesian inference for multiple
  changepoint problems. *Statistics and Computing* 16(2), 203–213.
- Fearnhead, P. & Liu, Z. (2007). On-line inference for multiple changepoint
  problems. *JRSS-B* 69(4), 589–605.
- Fearnhead, P. & Clifford, P. (2003). On-line inference for hidden Markov models
  via particle filters. *JRSS-B* 65(4), 887–899.
- Adams, R. P. & MacKay, D. J. C. (2007). Bayesian online changepoint detection.
  arXiv:0710.3742.
- Meftahi, N. (2021). *Changepoint Detection in KPI Time-Series Data — a
  Bayesian Approach*. Master's thesis, Linköping University.
  LIU-IDA/STAT-A--21/046--SE.
- Ruggieri, E. & Lawrence, C. E. (2014). The Bayesian change point and variable
  selection algorithm. *JCGS* 23(1), 87–110.

## Acknowledgements

The conjugate segment model follows the author's 2021 master's thesis at
Linköping University. The inference algorithms are not original to this package:
the offline recursion is Fearnhead (2006), building on the product partition
models of Barry & Hartigan (1992), with the variable-selection treatment of
Ruggieri & Lawrence (2014); the online filter and its optimal resampling are
Fearnhead & Liu (2007) and Fearnhead & Clifford (2003). See the references above.

Developed with assistance from Claude (Anthropic).

## Licence

MIT. See [LICENSE](LICENSE).
