"""Reproduce the synthetic experiments of Chapter 5.

Run from the project root with the venv active:

    python examples/synthetic.py
"""

import numpy as np

from bcpd import detect, trend_design


def main() -> None:
    rng = np.random.default_rng(0)

    # Section 5.1 - no changepoints, testing for false detection.
    flat = rng.normal(0.0, 1.0, size=400)
    res = detect(flat, trend_design(len(flat)), k_max=4, min_distance=96,
                 n_samples=500, rng=np.random.default_rng(42))
    print("Section 5.1  posterior over changepoint count:")
    print("  ", np.round(res.n_changepoints, 3))

    # Section 5.2 - shifts at t = 100 and t = 250.
    shifted = np.concatenate([
        rng.normal(1.0, 2.0, size=100),
        rng.normal(175.0, 20.0, size=150),
        rng.normal(100.0, 20.0, size=200),
    ])
    res = detect(shifted, trend_design(len(shifted)), k_max=4, min_distance=96,
                 n_samples=500, rng=np.random.default_rng(42))
    print("\nSection 5.2  posterior over changepoint count:")
    print("  ", np.round(res.n_changepoints, 3))

    prob = res.location_prob
    top = np.argsort(prob)[::-1][:4]
    print("  most probable locations:")
    for idx in sorted(top):
        if prob[idx] > 0.01:
            print(f"    t = {idx:4d}   p = {prob[idx]:.3f}")


def multivariate_demo() -> None:
    """Three correlated series sharing one changepoint at t = 150."""
    from bcpd import detect_multivariate, trend_design

    rng = np.random.default_rng(11)
    n, d = 300, 3
    shared = np.concatenate([np.zeros(150), np.full(150, 6.0)])
    cov = 0.6 * np.eye(d) + 0.4 * np.ones((d, d))
    Y = shared[:, None] + rng.multivariate_normal(np.zeros(d), cov, size=n)

    res = detect_multivariate(
        Y, trend_design(n), k_max=3, min_distance=60,
        n_samples=500, rng=np.random.default_rng(5),
    )
    print("\nMultivariate, 3 correlated series:")
    print("  posterior over changepoint count:", np.round(res.n_changepoints, 3))
    top = np.argsort(res.location_prob)[::-1][:3]
    for idx in sorted(top):
        if res.location_prob[idx] > 0.01:
            print(f"    t = {idx:4d}   p = {res.location_prob[idx]:.3f}")


def harmonic_order_demo() -> None:
    """Posterior over harmonic order, the Chapter 6 future-work item."""
    from bcpd import PriorSpec, SegmentLikelihood, seasonal_design

    n, rng = 400, np.random.default_rng(5)
    t = np.arange(1, n + 1, dtype=float)
    cases = {
        "first harmonic only": 3.0 * np.sin(2 * np.pi * t / 96),
        "first + third": 3.0 * np.sin(2 * np.pi * t / 96)
        + 2.5 * np.cos(2 * np.pi * 3 * t / 96),
    }
    print("\nPosterior over harmonic order L:")
    for label, signal in cases.items():
        y = signal + rng.normal(scale=0.4, size=n)
        like = SegmentLikelihood(
            y, seasonal_design(n, 96, 4), PriorSpec(), nested_sizes=(2, 4, 6, 8)
        )
        post = like.posterior_size(0, n)
        shown = "  ".join(f"L={i + 1}: {p:.3f}" for i, p in enumerate(post))
        print(f"  {label:22s} {shown}")


if __name__ == "__main__":
    main()
    harmonic_order_demo()
    multivariate_demo()
