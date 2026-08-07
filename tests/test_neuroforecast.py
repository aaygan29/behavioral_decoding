"""Aggregate forecasting, and the ground-truth dissociation the framework must recover."""

from __future__ import annotations

import numpy as np
import pytest

from behavioral_decoding.evaluation.neuroforecast import (
    aggregate_block,
    aggregate_by_stimulus,
    compare_forecast_arms,
    cross_validate_ensemble,
    forecast_market,
)
from behavioral_decoding.synthetic import SyntheticConfig, make_synthetic_dataset

pytest.importorskip("imblearn")


def test_aggregate_by_stimulus_pools_correctly():
    values = np.array([1.0, 3.0, 10.0, 20.0])
    stimuli = np.array(["a", "a", "b", "b"])
    ids, pooled = aggregate_by_stimulus(values, stimuli, statistic="mean")
    assert list(ids) == ["a", "b"]
    assert np.allclose(pooled, [2.0, 15.0])


def test_aggregate_block_returns_one_row_per_stimulus(small_dataset):
    dataset, _ = small_dataset
    ids, pooled = aggregate_block(dataset.blocks["fmri"])
    assert len(ids) == dataset.n_stimuli
    assert pooled.shape == (dataset.n_stimuli, dataset.blocks["fmri"].n_features)


def test_trimmed_mean_resists_a_single_outlier():
    values = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 500.0])
    stimuli = np.array(["a"] * 10)
    _, mean_pooled = aggregate_by_stimulus(values, stimuli, statistic="mean")
    _, trimmed = aggregate_by_stimulus(values, stimuli, statistic="trimmed_mean")
    assert trimmed[0] < mean_pooled[0]
    assert trimmed[0] < 5.0


def test_forecast_market_reports_negative_r2_when_features_are_noise():
    """An honest out-of-sample R^2 goes negative on noise. It must not be clipped."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = rng.normal(size=60)
    result = forecast_market(X, y, n_splits=5, task="regression", seed=0)
    assert result["r2_out_of_sample"] < 0.15
    assert result["n_stimuli"] == 60.0


def test_forecast_market_recovers_a_planted_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.3 * rng.normal(size=80)
    result = forecast_market(X, y, n_splits=5, task="regression", seed=0)
    assert result["r2_out_of_sample"] > 0.8
    assert result["pearson_p"] < 1e-6


def test_forecast_market_refuses_a_sample_too_small_for_the_folds():
    X = np.zeros((3, 2))
    y = np.zeros(3)
    with pytest.raises(ValueError, match="cannot run"):
        forecast_market(X, y, n_splits=5)


def test_compare_arms_requires_aggregate_outcomes(small_dataset):
    from behavioral_decoding.io.base import MultimodalDataset

    dataset, _ = small_dataset
    stripped = MultimodalDataset(
        blocks=dataset.blocks,
        y_individual=dataset.y_individual,
        subject_ids=dataset.subject_ids,
        stimulus_ids=dataset.stimulus_ids,
        y_aggregate=None,
    )
    with pytest.raises(ValueError, match="no y_aggregate"):
        compare_forecast_arms(stripped)


def test_brain_forecasts_the_market_better_than_self_report():
    """The core dissociation, on data where it is true by construction.

    This is the framework's positive control. If it stops passing, the pipeline
    can no longer detect the effect it exists to detect, and nothing it reports
    on real data should be believed until this is green again.
    """
    dataset, _ = make_synthetic_dataset(SyntheticConfig(n_subjects=16, n_stimuli=50, seed=0))
    arms = compare_forecast_arms(dataset, task="regression", n_splits=5, seed=0)

    assert "brain_only" in arms and "behavior_only" in arms
    brain = arms["brain_only"]["r2_out_of_sample"]
    behaviour = arms["behavior_only"]["r2_out_of_sample"]

    assert brain > behaviour, f"brain oos R2 {brain:.3f} did not exceed behaviour {behaviour:.3f}"
    assert brain > 0.0


def test_the_dissociation_holds_across_seeds():
    """One seed is an anecdote. Check the ordering is not a lucky draw."""
    wins = 0
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        dataset, _ = make_synthetic_dataset(
            SyntheticConfig(n_subjects=14, n_stimuli=45, seed=seed)
        )
        arms = compare_forecast_arms(dataset, task="regression", n_splits=5, seed=seed)
        if arms["brain_only"]["r2_out_of_sample"] > arms["behavior_only"]["r2_out_of_sample"]:
            wins += 1
    assert wins >= 4, f"brain beat behaviour in only {wins}/{len(seeds)} seeds"


def test_nested_cv_reports_weight_stability(small_dataset):
    dataset, _ = small_dataset
    result = cross_validate_ensemble(
        dataset, n_splits_outer=3, n_splits_inner=2, seed=0
    )

    assert not np.isnan(result["oof_proba"]).any()
    assert len(result["folds"]) == 3
    stability = result["weight_stability"]
    assert set(stability) == set(dataset.modalities)
    for stats in stability.values():
        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["sd"] >= 0.0


def test_nested_cv_beats_chance_on_data_with_planted_signal(small_dataset):
    dataset, _ = small_dataset
    result = cross_validate_ensemble(dataset, n_splits_outer=3, n_splits_inner=2, seed=0)
    assert result["pooled_metrics"]["balanced_accuracy"] > 0.55
