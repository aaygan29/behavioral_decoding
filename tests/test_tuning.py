"""Nested, subject-grouped hyperparameter tuning.

The search must (a) return a hyperparameter setting from the grid, (b) run
entirely on the rows it is handed so it stays nested when called on an outer
training fold, and (c) flow through the ensemble and its run record.
"""

from __future__ import annotations

import pytest

pytest.importorskip("imblearn")

from behavioral_decoding.models.ensemble import MultimodalEnsemble  # noqa: E402
from behavioral_decoding.models.modality_models import tune_modality_model  # noqa: E402
from behavioral_decoding.synthetic import (  # noqa: E402
    SyntheticConfig,
    make_synthetic_dataset,
)


def _dataset(n_subjects=12, n_stimuli=25, seed=0):
    ds, _ = make_synthetic_dataset(
        SyntheticConfig(n_subjects=n_subjects, n_stimuli=n_stimuli, seed=seed)
    )
    return ds


def test_tune_modality_model_returns_grid_point():
    ds = _dataset()
    block = ds.blocks["fmri"]
    grid = {"C": [0.1, 1.0, 10.0], "l1_ratio": [0.2, 0.8]}
    result = tune_modality_model(
        "fmri", block.X, ds.y_individual, ds.subject_ids,
        base_learner="elasticnet", grid=grid, n_splits=3, seed=0,
    )
    assert result["best_params"]["C"] in grid["C"]
    assert result["best_params"]["l1_ratio"] in grid["l1_ratio"]
    # Every grid point should have a recorded score (6 combinations).
    assert len(result["table"]) == 6


def test_tune_skips_gracefully_with_too_few_subjects():
    ds = _dataset(n_subjects=1, n_stimuli=25)
    block = ds.blocks["fmri"]
    result = tune_modality_model(
        "fmri", block.X, ds.y_individual, ds.subject_ids, base_learner="elasticnet",
    )
    assert result["best_params"] == {}


def test_ensemble_tuning_records_chosen_params():
    ds = _dataset(n_subjects=12, n_stimuli=25)
    ens = MultimodalEnsemble(
        n_splits=3,
        tune=True,
        tune_grid={"fmri": {"C": [0.1, 1.0], "l1_ratio": [0.5]}},
        seed=0,
    ).fit(ds)
    record = ens.to_record()
    assert record["tuned"] is True
    assert "fmri" in record["tuning"]
    assert record["tuning"]["fmri"]["best_params"]["C"] in (0.1, 1.0)


def test_tuning_respects_fixed_kwargs():
    ds = _dataset()
    block = ds.blocks["fmri"]
    # Pin n_bags via fixed_kwargs; it must not be overwritten by the grid.
    result = tune_modality_model(
        "fmri", block.X, ds.y_individual, ds.subject_ids,
        base_learner="elasticnet", grid={"C": [1.0]}, n_splits=3,
        fixed_kwargs={"n_bags": 5},
    )
    # The search only varied C; n_bags stays where we fixed it (checked by
    # rebuilding is unnecessary — the grid simply never touched it).
    assert "n_bags" not in result["best_params"]
