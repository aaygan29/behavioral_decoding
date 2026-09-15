"""Synthetic ground-truth checks for the decision-geometry evaluation utilities.

Covers `behavioral_decoding.evaluation.choice_psychometrics` (Takacs et al.
2026 style logistic psychometrics) and
`behavioral_decoding.evaluation.decoding_geometry` (Li et al. 2026 style
cross-condition generalization, variance decomposition, subspace alignment).
Every check plants a known effect in synthetic data and verifies the utility
recovers it, so the tests do not depend on the surrounding narrative being
correct.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from behavioral_decoding.evaluation.choice_psychometrics import (
    additivity_test,
    bias_sensitivity_ci,
    fit_psychometric,
)
from behavioral_decoding.evaluation.decoding_geometry import (
    cross_condition_generalization,
    subspace_alignment,
    variance_decomposition,
)


def _simulate_choices(rng, n, bias, slope, evidence_scale=1.0):
    evidence = rng.normal(0.0, evidence_scale, size=n)
    logit = bias + slope * evidence
    p = 1.0 / (1.0 + np.exp(-logit))
    choice = (rng.uniform(size=n) < p).astype(int)
    return evidence, choice


# ---------------------------------------------------------------------------
# choice_psychometrics
# ---------------------------------------------------------------------------


def test_bias_shift_recovered_without_slope_change():
    """A pure bias shift (Takacs et al. SC-inactivation style) moves the
    fitted intercept but leaves the fitted slope roughly unchanged."""
    rng = np.random.default_rng(0)
    n = 4000
    true_slope = 2.0

    evidence_ctrl, choice_ctrl = _simulate_choices(rng, n, bias=0.0, slope=true_slope)
    evidence_shift, choice_shift = _simulate_choices(rng, n, bias=-1.5, slope=true_slope)

    fit_ctrl = fit_psychometric(evidence_ctrl, choice_ctrl)
    fit_shift = fit_psychometric(evidence_shift, choice_shift)

    assert fit_ctrl["bias"] == pytest.approx(0.0, abs=0.15)
    assert fit_shift["bias"] == pytest.approx(-1.5, abs=0.15)
    # The bias moved by ~1.5 log-odds units, but slope stayed put.
    assert abs(fit_shift["bias"] - fit_ctrl["bias"]) > 1.0
    assert abs(fit_shift["slope"] - fit_ctrl["slope"]) < 0.25
    assert fit_ctrl["slope"] == pytest.approx(true_slope, rel=0.15)
    assert fit_shift["slope"] == pytest.approx(true_slope, rel=0.15)


def test_bias_sensitivity_ci_brackets_truth_and_separates_metrics():
    rng = np.random.default_rng(1)
    evidence, choice = _simulate_choices(rng, 3000, bias=0.8, slope=1.5)
    subjects = np.repeat(np.arange(30), 100)

    out = bias_sensitivity_ci(evidence, choice, groups=subjects, n_boot=300, seed=2)

    assert out["bias"]["lo"] < 0.8 < out["bias"]["hi"]
    assert out["slope"]["lo"] < 1.5 < out["slope"]["hi"]
    # Reported as two independent intervals, not one collapsed number.
    assert "bias" in out and "slope" in out and out["bias"] is not out["slope"]


def test_additive_manipulations_show_near_zero_interaction():
    """Two manipulations that each shift bias by a fixed additive amount in
    log-odds space (Takacs et al. bilateral-additive style) should yield a
    small, non-significant interaction coefficient."""
    rng = np.random.default_rng(3)
    n = 6000
    manip_a = rng.integers(0, 2, size=n).astype(float)
    manip_b = rng.integers(0, 2, size=n).astype(float)
    evidence = rng.normal(0.0, 1.0, size=n)

    true_slope = 1.2
    effect_a = -1.0
    effect_b = -1.0
    logit = 0.2 + true_slope * evidence + effect_a * manip_a + effect_b * manip_b
    p = 1.0 / (1.0 + np.exp(-logit))
    choice = (rng.uniform(size=n) < p).astype(int)

    result = additivity_test(evidence, choice, manip_a, manip_b)
    assert abs(result["interaction_coef"]) < 0.3
    assert result["p_value"] > 0.05


def test_interacting_manipulations_show_large_significant_interaction():
    """When the true generative model has a genuine interaction term, the
    additivity test should detect it as significant."""
    rng = np.random.default_rng(4)
    n = 6000
    manip_a = rng.integers(0, 2, size=n).astype(float)
    manip_b = rng.integers(0, 2, size=n).astype(float)
    evidence = rng.normal(0.0, 1.0, size=n)

    true_slope = 1.0
    logit = 0.0 + true_slope * evidence - 0.5 * manip_a - 0.5 * manip_b + 3.0 * (manip_a * manip_b)
    p = 1.0 / (1.0 + np.exp(-logit))
    choice = (rng.uniform(size=n) < p).astype(int)

    result = additivity_test(evidence, choice, manip_a, manip_b)
    assert result["interaction_coef"] > 1.5
    assert result["p_value"] < 0.01


# ---------------------------------------------------------------------------
# decoding_geometry
# ---------------------------------------------------------------------------


def _make_two_class_gaussians(rng, n_per_class, mean_sep, dim, noise_sd=1.0):
    mean0 = np.zeros(dim)
    mean1 = np.zeros(dim)
    mean1[0] = mean_sep
    X0 = rng.normal(size=(n_per_class, dim)) * noise_sd + mean0
    X1 = rng.normal(size=(n_per_class, dim)) * noise_sd + mean1
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    return X, y


def test_aligned_code_generalizes_across_conditions():
    """If the discriminating direction is identical in both conditions, a
    decoder trained on one condition should generalize to the other, close to
    its within-condition accuracy (Li et al. 2026 aligned-subspace case)."""
    rng = np.random.default_rng(5)
    X_a, y_a = _make_two_class_gaussians(rng, 150, mean_sep=4.0, dim=10)
    X_b, y_b = _make_two_class_gaussians(rng, 150, mean_sep=4.0, dim=10)

    estimator = LogisticRegression(max_iter=1000)
    result = cross_condition_generalization(estimator, X_a, y_a, X_b, y_b, n_perm=200, seed=0)

    assert result["a_to_b"] > 0.9
    assert result["b_to_a"] > 0.9
    assert abs(result["a_to_b"] - result["within_a"]) < 0.1
    assert result["p_value_a_to_b"] < 0.01
    assert result["p_value_b_to_a"] < 0.01


def test_orthogonal_rotated_code_drops_to_chance_across_conditions():
    """If condition B's discriminating direction is rotated to be orthogonal
    to condition A's, a decoder trained on A should fail on B even though it
    succeeds within each condition on its own."""
    rng = np.random.default_rng(6)
    dim = 10
    X_a, y_a = _make_two_class_gaussians(rng, 150, mean_sep=4.0, dim=dim)

    # Condition B: separate along a different, orthogonal axis.
    mean0 = np.zeros(dim)
    mean1 = np.zeros(dim)
    mean1[1] = 4.0  # orthogonal to axis 0 used in condition A
    X0 = rng.normal(size=(150, dim)) + mean0
    X1 = rng.normal(size=(150, dim)) + mean1
    X_b = np.vstack([X0, X1])
    y_b = np.concatenate([np.zeros(150), np.ones(150)])

    estimator = LogisticRegression(max_iter=1000)
    result = cross_condition_generalization(estimator, X_a, y_a, X_b, y_b, n_perm=200, seed=0)

    assert result["within_a"] > 0.9
    assert result["within_b"] > 0.9
    # Cross-condition generalization collapses toward chance.
    assert result["a_to_b"] < 0.65
    assert result["b_to_a"] < 0.65


def test_variance_decomposition_attributes_planted_separation_increase():
    """Two datasets with identical within-class noise but a larger gap
    between class means should show a higher between-class variance and a
    near-identical within-class variance."""
    rng = np.random.default_rng(7)
    X_small_sep, y_small = _make_two_class_gaussians(rng, 300, mean_sep=1.0, dim=5)
    X_large_sep, y_large = _make_two_class_gaussians(rng, 300, mean_sep=4.0, dim=5)

    small = variance_decomposition(X_small_sep, y_small)
    large = variance_decomposition(X_large_sep, y_large)

    assert large["between_class_variance"] > small["between_class_variance"] * 2
    # Within-class variance along the LDA axis should stay comparable since
    # only the mean separation was changed, not the noise.
    ratio = large["within_class_variance"] / small["within_class_variance"]
    assert 0.5 < ratio < 2.0
    assert large["variance_ratio"] > small["variance_ratio"]


def test_principal_angles_near_zero_for_aligned_and_near_90_for_orthogonal():
    rng = np.random.default_rng(8)
    dim = 8
    X_a, y_a = _make_two_class_gaussians(rng, 200, mean_sep=5.0, dim=dim)
    X_b_aligned, y_b_aligned = _make_two_class_gaussians(rng, 200, mean_sep=5.0, dim=dim)

    aligned = subspace_alignment(X_a, y_a, X_b_aligned, y_b_aligned)
    assert aligned["mean_coding_vector_correlation"] > 0.8
    assert all(angle < 25.0 for angle in aligned["principal_angles_deg"])

    mean0 = np.zeros(dim)
    mean1 = np.zeros(dim)
    mean1[1] = 5.0
    X0 = rng.normal(size=(200, dim)) + mean0
    X1 = rng.normal(size=(200, dim)) + mean1
    X_b_orth = np.vstack([X0, X1])
    y_b_orth = np.concatenate([np.zeros(200), np.ones(200)])

    orthogonal = subspace_alignment(X_a, y_a, X_b_orth, y_b_orth)
    assert abs(orthogonal["mean_coding_vector_correlation"]) < 0.3
    assert all(angle > 65.0 for angle in orthogonal["principal_angles_deg"])
