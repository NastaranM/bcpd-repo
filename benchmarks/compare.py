"""Compare bcpd against Adams & MacKay BOCPD, Prophet and ruptures.

    python benchmarks/compare.py

Prophet and ruptures are used if importable and skipped otherwise. The
Adams & MacKay implementation is included here rather than imported, so the
core comparison needs no installs and is not confounded by a third-party
package's own design choices.

Read the two tables together. Runtime alone is misleading because these tools
do not all answer the same question:

* bcpd offline returns a posterior over the number of changepoints and over each
  location.
* bcpd online and BOCPD return a run-length posterior, updated per observation.
* ruptures returns a point estimate of the locations, given the number of them.
* Prophet returns a MAP trend with kinks at a fixed candidate grid; it does not
  quantify uncertainty about changepoints at all.

Only the first quantifies "how many changes are there", which is what the O(N^2)
enumeration buys.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import gammaln

from bcpd import (
    OnlineChangepointFilter,
    PriorSpec,
    SegmentLengthPrior,
    SegmentLikelihood,
    constant_design,
    detect,
)

# --------------------------------------------------------------------------
# Reference implementation: Adams & MacKay (2007), vectorised.
# --------------------------------------------------------------------------


def bocpd_normal(y: np.ndarray, hazard: float, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    """Bayesian online changepoint detection with a Normal-inverse-gamma model.

    This is the implementation style BOCPD is usually benchmarked in: the whole
    run-length vector is updated with array operations, and the Student-t
    predictive comes from sufficient statistics maintained in O(1) per run
    length. No linear algebra per run length, which is where its speed advantage
    over a design-matrix formulation comes from.

    Returns the run-length posterior matrix, (N + 1, N + 1).
    """
    n = len(y)
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1.0

    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])

    for t in range(n):
        # Student-t predictive for every current run length at once.
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        z = (y[t] - mu) / scale
        df = 2.0 * alpha
        log_pred = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(np.pi * df)
            - np.log(scale)
            - (df + 1.0) / 2.0 * np.log1p(z**2 / df)
        )
        pred = np.exp(log_pred)

        growth = R[: t + 1, t] * pred * (1.0 - hazard)
        cp = np.sum(R[: t + 1, t] * pred * hazard)

        R[1 : t + 2, t + 1] = growth
        R[0, t + 1] = cp
        total = R[:, t + 1].sum()
        if total > 0:
            R[:, t + 1] /= total

        # Conjugate Normal-inverse-gamma updates, prepending the fresh-segment
        # prior. beta uses the *old* mu, so compute all new arrays before
        # overwriting any of them.
        new_mu = (kappa * mu + y[t]) / (kappa + 1.0)
        new_beta = beta + kappa * (y[t] - mu) ** 2 / (2.0 * (kappa + 1.0))
        new_alpha = alpha + 0.5
        new_kappa = kappa + 1.0

        mu = np.concatenate([[mu0], new_mu])
        beta = np.concatenate([[beta0], new_beta])
        alpha = np.concatenate([[alpha0], new_alpha])
        kappa = np.concatenate([[kappa0], new_kappa])

    return R


def bocpd_changepoints(R: np.ndarray, min_drop: int = 5) -> list[int]:
    """Steps where the most probable run length collapses.

    Not a threshold on ``R[0, t]``: at a changepoint the mass moves to run
    length *one* at the following step rather than to zero at the current one,
    so ``R[0, t]`` stays near the hazard rate and a threshold on it detects
    nothing. The drop in the modal run length is the usual read-out, and it
    carries the same one-step detection lag as any sequential method.
    """
    modal = R.argmax(axis=0)
    return [
        t - 1
        for t in range(2, R.shape[1])
        if modal[t] < modal[t - 1] - min_drop
    ]


# --------------------------------------------------------------------------
# Data and scoring
# --------------------------------------------------------------------------


def make_series(n: int, n_cp: int = 3, jump: float = 5.0, seed: int = 0):
    """Piecewise-constant mean with evenly spaced changes."""
    rng = np.random.default_rng(seed)
    cuts = [int(n * (i + 1) / (n_cp + 1)) for i in range(n_cp)]
    y = rng.normal(0, 1, n)
    for i, c in enumerate(cuts):
        y[c:] += jump * (-1) ** i
    return y, cuts


def location_error(found: list[int], truth: list[int]) -> float:
    """Mean distance from each true changepoint to the nearest detection."""
    if not found:
        return float("inf")
    return float(np.mean([min(abs(f - t) for f in found) for t in truth]))


# --------------------------------------------------------------------------


def run_speed(sizes=(200, 400, 800, 1600)):
    print("RUNTIME (seconds)\n")
    header = f"{'N':>6} {'bcpd offline':>14} {'bcpd online':>13} {'BOCPD (A&M)':>13} {'ratio':>8}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        y, _ = make_series(n)
        X = constant_design(n)

        t0 = time.perf_counter()
        detect(y, X, min_distance=max(20, n // 12), k_max=4, n_samples=100,
               rng=np.random.default_rng(0))
        t_off = time.perf_counter() - t0

        like = SegmentLikelihood(y, X, PriorSpec())
        filt = OnlineChangepointFilter(
            like, SegmentLengthPrior(mean_length=n / 4.0), max_particles=100
        )
        t0 = time.perf_counter()
        filt.run(rng=np.random.default_rng(0), store_history=False)
        t_on = time.perf_counter() - t0

        t0 = time.perf_counter()
        bocpd_normal(y, hazard=4.0 / n)
        t_bo = time.perf_counter() - t0

        print(f"{n:6d} {t_off:14.3f} {t_on:13.3f} {t_bo:13.3f} {t_on / max(t_bo, 1e-9):7.1f}x")

    print("\nThe last column is the honest one: bcpd online and BOCPD share the")
    print("same O(N x particles) complexity, so that gap is entirely constant")
    print("factors. BOCPD updates sufficient statistics in O(1) per run length")
    print("with pure array operations; bcpd solves a q x q system per particle")
    print("inside a Python loop, which is wasted work when q = 1.")
    print()
    print("The gap is an implementation gap, not an algorithmic one. It buys")
    print("generality: bcpd takes an arbitrary design matrix, so the same code")
    print("detects changes in trend, seasonal amplitude or regression")
    print("coefficients, and extends to correlated multivariate series. The")
    print("BOCPD above handles a change in mean and variance and nothing else.")


def run_accuracy(n=800):
    print("\n\nDETECTION QUALITY (N = %d, 3 changepoints)\n" % n)
    y, truth = make_series(n)
    X = constant_design(n)
    print(f"  true locations: {truth}")

    res = detect(y, X, min_distance=n // 12, k_max=6, n_samples=300,
                 rng=np.random.default_rng(0))
    k = res.posterior.map_count()
    top = sorted(int(i) for i in np.argsort(res.location_prob)[::-1][:k])
    print(f"\n  bcpd offline : {k} changepoints, P = {res.n_changepoints[k]:.3f}")
    print(f"                 at {top}, mean error {location_error(top, truth):.1f}")
    print("                 (only method here that reports how many there are)")

    like = SegmentLikelihood(y, X, PriorSpec())
    filt = OnlineChangepointFilter(
        like, SegmentLengthPrior(mean_length=n / 4.0), max_particles=150
    )
    filt.run(rng=np.random.default_rng(0))
    draws = filt.backward_sample(200, rng=np.random.default_rng(1))
    counts = np.bincount([len(d) for d in draws])
    online_k = int(np.argmax(counts))
    locs = sorted({c for d in draws for c in d})
    print(f"\n  bcpd online  : modal {online_k} changepoints")
    print(f"                 mean error {location_error(locs, truth):.1f}")

    R = bocpd_normal(y, hazard=4.0 / n)
    found = bocpd_changepoints(R)
    print(f"\n  BOCPD (A&M)  : {len(found)} changepoints")
    print(f"                 mean error {location_error(found, truth):.1f}")

    try:
        import ruptures as rpt

        t0 = time.perf_counter()
        algo = rpt.Pelt(model="l2", min_size=n // 12).fit(y)
        found = [c for c in algo.predict(pen=np.log(n) * 4) if c < n]
        dt = time.perf_counter() - t0
        print(f"\n  ruptures PELT: {len(found)} changepoints in {dt:.3f}s")
        print(f"                 mean error {location_error(found, truth):.1f}")
        print("                 (point estimate; no posterior over the count)")
    except ImportError:
        print("\n  ruptures     : not installed  (pip install ruptures)")

    try:
        import pandas as pd
        from prophet import Prophet

        df = pd.DataFrame(
            {"ds": pd.date_range("2020-01-01", periods=n, freq="D"), "y": y}
        )
        t0 = time.perf_counter()
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                    daily_seasonality=False)
        m.fit(df)
        dt = time.perf_counter() - t0
        deltas = np.abs(m.params["delta"].mean(axis=0))
        idx = np.argsort(deltas)[::-1][:3]
        found = sorted(int(np.where(df["ds"] == m.changepoints.iloc[i])[0][0]) for i in idx)
        print(f"\n  Prophet      : fitted in {dt:.3f}s")
        print(f"                 3 largest trend kinks at {found}")
        print(f"                 mean error {location_error(found, truth):.1f}")
        print("                 (MAP trend on a fixed candidate grid;")
        print("                  no posterior over number or location)")
    except ImportError:
        print("\n  Prophet      : not installed  (pip install prophet)")


if __name__ == "__main__":
    run_speed()
    run_accuracy()
