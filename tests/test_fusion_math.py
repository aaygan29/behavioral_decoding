"""Empirical validation of the inverse-variance fusion inequality.

The fusion claim in docs/biosignal_fusion.md rests on one inequality:

    sigma^2_fused = 1 / sum_m (1 / sigma_m^2)  <=  min_m sigma_m^2

with equality only when a single family carries all the precision. These tests
check it two ways that do not depend on the algebra being stated correctly:

1. Algebraically, over many random positive variance vectors (exact, to fp tol).
2. Operationally, by Monte Carlo: build independent noisy estimators of a shared
   signal, combine them by inverse-variance weights, and confirm the realised
   mean squared error matches the predicted fused variance and beats the best
   single estimator.

Empirical checks over reasoning: if the inequality were mis-stated, these fail.
"""

from __future__ import annotations

import numpy as np
import pytest


def fused_variance(sigma2: np.ndarray) -> float:
    """Inverse-variance-combined variance 1 / sum(1/sigma_m^2)."""
    return 1.0 / np.sum(1.0 / sigma2)


def test_inequality_holds_over_random_configs():
    rng = np.random.default_rng(0)
    for _ in range(100_000):
        m = rng.integers(1, 8)
        sigma2 = rng.uniform(1e-3, 10.0, size=m)
        assert fused_variance(sigma2) <= sigma2.min() + 1e-12


def test_equality_only_with_one_informative_family():
    # One tiny variance dominates: fused variance approaches that minimum.
    sigma2 = np.array([1e-6, 5.0, 3.0, 8.0])
    assert fused_variance(sigma2) == pytest.approx(sigma2.min(), rel=1e-3)
    # All equal: fused variance is sigma^2 / m, strictly below the min for m > 1.
    eq = np.full(4, 2.0)
    assert fused_variance(eq) == pytest.approx(2.0 / 4)
    assert fused_variance(eq) < eq.min()


def test_monte_carlo_matches_predicted_fused_variance():
    """Realised MSE of the inverse-variance combination matches the prediction."""
    rng = np.random.default_rng(1)
    sigma2 = np.array([0.5, 1.0, 2.0, 4.0])
    sigma = np.sqrt(sigma2)
    w = (1.0 / sigma2) / np.sum(1.0 / sigma2)  # inverse-variance weights

    n = 400_000
    ell = rng.normal(0.0, 1.0, size=n)                       # shared signal
    eps = rng.normal(0.0, 1.0, size=(n, sigma2.size)) * sigma  # independent noise
    estimates = ell[:, None] + eps
    fused = estimates @ w

    realised = np.var(fused - ell)
    predicted = fused_variance(sigma2)
    # Monte Carlo agreement to ~1%.
    assert realised == pytest.approx(predicted, rel=0.02)
    # And the operational payoff: fused MSE beats the best single estimator.
    best_single = np.min([np.var(estimates[:, k] - ell) for k in range(sigma2.size)])
    assert realised < best_single


def test_correlated_noise_erodes_the_gain():
    """Negative control: if the families' noise is perfectly correlated, fusion
    gives no variance reduction. This is why the doc requires independence."""
    rng = np.random.default_rng(2)
    sigma2 = np.array([1.0, 1.0, 1.0])
    sigma = np.sqrt(sigma2)
    w = (1.0 / sigma2) / np.sum(1.0 / sigma2)

    n = 400_000
    ell = rng.normal(0.0, 1.0, size=n)
    shared_noise = rng.normal(0.0, 1.0, size=n)               # identical across families
    estimates = ell[:, None] + shared_noise[:, None] * sigma
    fused = estimates @ w

    realised = np.var(fused - ell)
    best_single = np.min([np.var(estimates[:, k] - ell) for k in range(sigma2.size)])
    # No gain: fused variance equals the single-family variance (within MC error).
    assert realised == pytest.approx(best_single, rel=0.02)
