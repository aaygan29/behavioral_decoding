"""Reconciliation behaviour: bagging, majority voting, and accuracy weighting."""

from __future__ import annotations

import numpy as np
import pytest

from behavioral_decoding.models.ensemble import MultimodalEnsemble
from behavioral_decoding.models.modality_models import build_modality_model

pytest.importorskip("imblearn")


def test_modality_model_is_a_bag_of_resampling_pipelines():
    model = build_modality_model("fmri", y=np.array([0] * 80 + [1] * 20), n_bags=7)
    assert model.n_estimators == 7
    steps = dict(model.estimator.steps)
    assert "scaler" in steps
    assert "estimator" in steps
    # Resampling must sit between scaling and the learner, inside each bag.
    assert list(dict(model.estimator.steps)) == ["scaler", "resample", "estimator"]
    assert model.bd_spec_["modality"] == "fmri"


def test_bagging_reduces_variance_across_seeds(small_dataset):
    """More bags should give more stable predictions across reseeds."""
    dataset, _ = small_dataset
    X = dataset.blocks["face"].X
    y = dataset.y_individual

    def spread(n_bags: int) -> float:
        preds = []
        for seed in range(3):
            model = build_modality_model("face", y=y, n_bags=n_bags, seed=seed)
            model.fit(X, y)
            preds.append(model.predict_proba(X)[:, 1])
        return float(np.mean(np.std(np.vstack(preds), axis=0)))

    assert spread(20) < spread(2)


def test_weights_sum_to_one_and_below_chance_modality_is_dropped(small_dataset):
    dataset, _ = small_dataset
    ensemble = MultimodalEnsemble(n_splits=3, seed=0).fit(dataset)

    assert pytest.approx(sum(ensemble.weights_.values()), abs=1e-9) == 1.0
    assert set(ensemble.weights_) == set(dataset.modalities)

    for modality, weight in ensemble.weights_.items():
        balacc = ensemble.modality_scores_[modality]["balanced_accuracy"]
        if balacc <= 0.5:
            assert weight == 0.0, f"{modality} is at or below chance but kept weight {weight}"


def test_accuracy_weighting_favours_the_stronger_modality(small_dataset):
    """fMRI carries the strongest signal by construction, so it should win."""
    dataset, _ = small_dataset
    ensemble = MultimodalEnsemble(n_splits=3, seed=0).fit(dataset)
    best = max(ensemble.weights_, key=lambda m: ensemble.weights_[m])
    assert best == "fmri"


def test_weights_track_out_of_fold_not_training_accuracy(small_dataset):
    """Weight ordering must match out-of-fold ordering, exactly."""
    dataset, _ = small_dataset
    ensemble = MultimodalEnsemble(n_splits=3, seed=0).fit(dataset)

    by_weight = sorted(ensemble.weights_, key=lambda m: -ensemble.weights_[m])
    by_oof = sorted(
        ensemble.modality_scores_,
        key=lambda m: -ensemble.modality_scores_[m]["balanced_accuracy"],
    )
    # Modalities dropped to zero weight all tie, so compare only the survivors.
    survivors = [m for m in by_weight if ensemble.weights_[m] > 0]
    assert survivors == [m for m in by_oof if ensemble.weights_[m] > 0]


def test_reconciliation_strategies_produce_valid_probabilities(small_dataset):
    dataset, _ = small_dataset
    for strategy in ("majority", "soft", "accuracy_weighted"):
        ensemble = MultimodalEnsemble(reconciliation=strategy, n_splits=3, seed=0).fit(dataset)
        proba = ensemble.predict_proba(dataset)
        assert proba.shape == (dataset.n_trials, 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert set(np.unique(ensemble.predict(dataset))) <= {0, 1}


def test_majority_vote_ignores_confidence_and_soft_vote_does_not():
    ensemble = MultimodalEnsemble(reconciliation="majority")
    ensemble.classes_ = np.array([0, 1])

    confident = {"a": np.array([0.99]), "b": np.array([0.99]), "c": np.array([0.01])}
    marginal = {"a": np.array([0.51]), "b": np.array([0.51]), "c": np.array([0.49])}
    assert ensemble.reconcile(confident) == ensemble.reconcile(marginal)

    ensemble.reconciliation = "soft"
    assert ensemble.reconcile(confident) != ensemble.reconcile(marginal)


def test_majority_vote_breaks_ties_with_mean_confidence():
    ensemble = MultimodalEnsemble(reconciliation="majority")
    ensemble.classes_ = np.array([0, 1])
    tied = {"a": np.array([0.9]), "b": np.array([0.1])}
    result = ensemble.reconcile(tied)
    assert pytest.approx(result[0], abs=1e-6) == 0.5


def test_predicting_with_a_missing_modality_raises(small_dataset):
    """Silently dropping a modality would leave the weights unnormalised."""
    from behavioral_decoding.io.base import MultimodalDataset

    dataset, _ = small_dataset
    ensemble = MultimodalEnsemble(n_splits=3, seed=0).fit(dataset)

    reduced = MultimodalDataset(
        blocks={k: v for k, v in dataset.blocks.items() if k != "eeg"},
        y_individual=dataset.y_individual,
        subject_ids=dataset.subject_ids,
        stimulus_ids=dataset.stimulus_ids,
    )
    with pytest.raises(KeyError, match="missing modality"):
        ensemble.predict_proba(reduced)


def test_unfitted_ensemble_raises_clearly(small_dataset):
    dataset, _ = small_dataset
    with pytest.raises(RuntimeError, match="not fitted"):
        MultimodalEnsemble().predict_proba(dataset)


def test_run_record_is_serialisable(small_dataset):
    import json

    dataset, _ = small_dataset
    ensemble = MultimodalEnsemble(n_splits=3, seed=0).fit(dataset)
    payload = json.loads(json.dumps(ensemble.to_record()))
    assert set(payload["weights"]) == set(dataset.modalities)
    assert payload["reconciliation"] == "accuracy_weighted"
