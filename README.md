# bcpd — Recursive Bayesian Changepoint Detection

Python port of the algorithm from *Changepoint Detection in KPI Time-Series Data — a
Bayesian Approach* (Meftahi, Linköping University, 2021), which follows Ruggieri &
Lawrence (2012, 2014) and Ruggieri & Antonellis (2016).

Exact, offline, sampling-free inference over the number and locations of changepoints.
Posterior draws are independent — no burn-in, no convergence diagnostics.

## Setup

```bash
git clone <your-repo-url> bcpd && cd bcpd
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

For VS Code: install it from https://code.visualstudio.com, add the Microsoft **Python**
extension, open this folder, and pick `.venv` via *Python: Select Interpreter* in the
command palette. `.vscode/settings.json` already wires up the interpreter path and the
test runner.

To start the git history:

```bash
git init && git add . && git commit -m "Port recursive Bayesian CPD from R to Python"
git remote add origin <your-repo-url>
git push -u origin main
```

## Usage

```python
import numpy as np
from bcpd import detect, trend_design

y = np.loadtxt("series.txt")
result = detect(y, trend_design(len(y)), k_max=4, min_distance=96)

print(result.n_changepoints)        # posterior over 0..k_max changepoints
print(result.location_prob)         # per-index posterior probability
print(result.posterior.map_count())
```

The two-stage trend-then-seasonal pipeline from the thesis:

```python
from bcpd import decompose

trend, seasonal, detrended = decompose(y, period=96, harmonic_order=4)
```

The seasonal stage places a uniform prior on the harmonic order rather than fixing
L = 4, so each segment can use as many harmonics as the data support:

```python
seasonal.likelihood.posterior_size(start, stop)   # posterior over L = 1..4
```

## Layout

| Path | Contents |
| --- | --- |
| `src/bcpd/marginal.py` | Segment marginal likelihood, Eq. (4.38), corrected |
| `src/bcpd/recursion.py` | Forward recursion (4.41) and stochastic backtrace (4.43)–(4.45) |
| `src/bcpd/design.py` | Trend (4.2) and Fourier seasonal (4.3) design matrices |
| `src/bcpd/detector.py` | High-level `detect` and `decompose` |
| `tests/` | Reproduces the synthetic experiments of Chapter 5 |

## Differences from the R original

The R code had several defects that the port does not reproduce. They are listed here so
the divergence is on the record rather than silent.

1. **Spike-and-slab.** The R `DP_exact` enumerated all 2^q predictor include/exclude
   patterns with precisions `k_i = 0.01` and `k_e = 100`. That came across with the
   Ruggieri & Lawrence reference implementation; Chapter 4 of the thesis specifies a
   single shared prior scale ν (Eq. 4.13) and no inclusion indicators. This package
   implements the thesis model.
2. **Frozen RNG.** `set.seed(12345)` sat *inside* `pick_k1`, `sample_model_given_chgpt_sim`
   and `partition_fn`, re-seeding on every call, so all 500 posterior draws used the same
   uniform variate. Here a single `numpy.random.Generator` is threaded through and can be
   seeded by the caller for reproducibility.
3. **Uniform sampling of inclusion patterns.** `sample_model_given_chgpt_sim` computed the
   posterior over the 2^m patterns and then discarded it, calling
   `sample(1:length(PROB), 1)`. The port samples from the computed posterior.
4. **`XtX` in place of `Xty`.** In `AllStepsTogether.R`, the `num_chgpts > 1` branch set
   `Xty <- t(X[...]) %*% X[...]`, while the equivalent `kk == 1` branch used `%*% y`.
5. **Inconsistent prior scale.** `DP_exact` added `sig0 + v0*sig0` to `s_n` in its first
   leaf; `sample_model_given_chgpt_sim` added only `v0*sig0`. Moot now that the
   inverse-gamma prior follows Eq. (4.15) directly.

Because of these, the R output is not a reliable oracle. The tests validate against a
direct evaluation of the derivation instead.

## Corrections to Eq. (4.38)

Eq. (4.37) takes the normalising constant of the beta integral as (σ²)^(1/2) rather
than (σ²)^(q/2). Four consequences propagate into Eq. (4.38) and (4.39). This package
implements the corrected form, checked against numerical integration of both β and σ²
(`tests/test_marginal.py`).

| As printed | Should be |
| --- | --- |
| exponent `c₁ + (q + n − 1)/2` | `c₁ + n/2` — the q cancels against the prior |
| `1 / \|XᵀX + νI_q\|` | `\|XᵀX + ν⁻¹I_q\|^(−1/2)` |
| `y'y/2 + (Xᵀy)ᵀV(Xᵀy)` | `y'y/2 − ½(Xᵀy)ᵀV(Xᵀy)` |
| no `c₂` in the denominator | `+ c₂` from the inverse-gamma prior |

The derivation up to Eq. (4.37) is sound, and the R `post_beta` has the sign right, so
these are transcription slips rather than modelling errors.

## Harmonic order

Chapter 6 fixes L = 4 on every seasonal segment and notes that a prior on L_k would be
preferable. `nested_sizes` supplies one: the segment likelihood is averaged over the
nested designs using the first 2, 4, …, 2·L_max columns, under a uniform prior on L.

This is the right shape for Fourier terms, where spike-and-slab is not. Harmonics are
ordered — real signals use 1…L rather than arbitrary subsets — and sin/cos at one
frequency form a pair, so switching one off without the other just forces an arbitrary
phase. Averaging over nested designs respects both, at L_max evaluations per segment
instead of 2^q.

## Scaling

Cumulative sufficient statistics are precomputed, so a segment query costs O(q³)
independent of segment length. The remaining O(N²) Python loop in
`SegmentLikelihood.segment_table` is the obvious next optimisation target.

## Known approximation

Eq. (4.4) specifies the model jointly over `X = [X_trend, X_seasonal]`, but `decompose`
estimates the trend, detrends, and only then estimates the seasonal component. This is
the thesis pipeline, retained as a reproduction baseline. The joint formulation is not
yet implemented.

## References

- Ruggieri, E. & Lawrence, C. E. (2012). On efficient calculations for Bayesian variable
  selection. *Computational Statistics & Data Analysis*, 56(6), 1319–1332.
- Ruggieri, E. & Lawrence, C. E. (2014). The Bayesian change point and variable selection
  algorithm. *Journal of Computational and Graphical Statistics*, 23(1), 87–110.
- Ruggieri, E. & Antonellis, M. (2016). An exact approach to Bayesian sequential change
  point detection. *Computational Statistics & Data Analysis*, 97, 71–86.
- Zhao, K. et al. (2019). Detecting change-point, trend, and seasonality in satellite time
  series data (BEAST). *Remote Sensing of Environment*, 232, 111–181.
