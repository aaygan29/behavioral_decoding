"""Probability-calibration metrics.

The multimodal ensemble reconciles predicted probabilities, so how well those
probabilities match observed frequencies is part of the claim, not an
afterthought. These tests pin the Brier score, the Expected Calibration Error,
and the reliability-curve data.
"""

from __future__ import annotations

import numpy as np

from behavioral_decoding.evaluation.metrics import (
    calibration_curve_points,
    classification_report,
    expected_calibration_error,
)


def test_perfectly_calibrated_probabilities_have_low_ece():
    rng = np.random.default_rng(0)
    # Draw true probabilities, then sample labels from them: by construction the
    # predicted probability equals the outcome frequency, so ECE -> 0.
    p = rng.uniform(0, 1, size=5000)
    y = (rng.uniform(0, 1, size=5000) < p).astype(int)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.05


def test_overconfident_probabilities_have_high_ece():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0, 1, size=5000)
    y = (rng.uniform(0, 1, size=5000) < p_true).astype(int)
    # Push probabilities toward the extremes: same ranking, worse calibration.
    p_over = np.clip((p_true - 0.5) * 3 + 0.5, 0, 1)
    assert expected_calibration_error(y, p_over) > expected_calibration_error(y, p_true)


def test_report_includes_brier_and_ece():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(0, 1, size=200)
    report = classification_report(y, p)
    assert "brier" in report and 0.0 <= report["brier"] <= 1.0
    assert "ece" in report and 0.0 <= report["ece"] <= 1.0


def test_calibration_curve_shapes_and_diagonal():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, size=3000)
    y = (rng.uniform(0, 1, size=3000) < p).astype(int)
    curve = calibration_curve_points(y, p, n_bins=10)
    n = len(curve["mean_predicted"])
    assert n == len(curve["observed_frequency"]) == len(curve["count"])
    # A calibrated model tracks the diagonal: predicted ~ observed per bin.
    x = np.array(curve["mean_predicted"])
    yv = np.array(curve["observed_frequency"])
    assert np.mean(np.abs(x - yv)) < 0.1


def test_quantile_strategy_runs():
    rng = np.random.default_rng(4)
    p = np.concatenate([rng.uniform(0, 0.1, 900), rng.uniform(0.9, 1.0, 100)])
    y = (rng.uniform(0, 1, size=1000) < p).astype(int)
    curve = calibration_curve_points(y, p, n_bins=5, strategy="quantile")
    assert curve["strategy"] == "quantile"
    assert len(curve["mean_predicted"]) >= 1
