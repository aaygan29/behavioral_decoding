"""Guards against the two leaks that would silently inflate every result.

These are the most important tests in the repo. Both failure modes produce
better-looking numbers, neither raises an error, and both are common enough in
published multimodal work that they should be checked in CI rather than
remembered.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from behavioral_decoding.balance.smote import make_balanced_pipeline, resample_train_only
from behavioral_decoding.evaluation.cv import iter_splits, out_of_fold_proba
from behavioral_decoding.evaluation.metrics import classification_report

imblearn = pytest.importorskip("imblearn")


def test_no_subject_appears_in_both_sides_of_a_fold(small_dataset):
    dataset, _ = small_dataset
    X = dataset.blocks["fmri"].X
    y = dataset.y_individual
    groups = dataset.subject_ids

    n_folds = 0
    for train_idx, test_idx in iter_splits(X, y, groups, n_splits=4, seed=0):
        assert not set(groups[train_idx]) & set(groups[test_idx])
        assert len(set(train_idx) & set(test_idx)) == 0
        n_folds += 1
    assert n_folds == 4


def test_every_trial_is_tested_exactly_once(small_dataset):
    dataset, _ = small_dataset
    X = dataset.blocks["fmri"].X
    y = dataset.y_individual

    seen = np.zeros(len(y), dtype=int)
    for _, test_idx in iter_splits(X, y, dataset.subject_ids, n_splits=4, seed=0):
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_balanced_pipeline_does_not_resample_at_predict_time():
    """The resampler must be a fit-time-only step.

    If it ran at transform time, ``predict`` would return more rows than it was
    given, and any per-trial join downstream would silently misalign.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = np.array([0] * 170 + [1] * 30)

    pipeline = make_balanced_pipeline(
        LogisticRegression(max_iter=1000), sampler="smote", k_neighbors=5, random_state=0
    )
    pipeline.fit(X, y)
    assert pipeline.predict(X).shape == (200,)
    assert pipeline.predict_proba(X).shape == (200, 2)


def test_smote_before_split_inflates_accuracy_and_the_pipeline_does_not():
    """Reproduce the classic leak, then show the pipeline is immune to it.

    Data here is pure noise, so the only honest answer is chance. Resampling the
    whole dataset before splitting interpolates synthetic training points from
    test-set neighbours, and balanced accuracy climbs well above chance on data
    with no signal in it at all.

    Averaged over seeds rather than asserted on one draw: a single split of 400
    noise rows moves by several points seed to seed, and a threshold tuned to
    one seed would be a flaky test rather than a demonstration.
    """
    seeds = range(6)
    leaky_scores = []
    clean_scores = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(400, 20))
        y = np.array([0] * 340 + [1] * 60)
        rng.shuffle(y)

        # --- Wrong: resample the whole dataset, then split.
        X_res, y_res = resample_train_only(X, y, sampler="smote", k_neighbors=5, random_state=seed)
        Xtr, Xte, ytr, yte = train_test_split(
            X_res, y_res, test_size=0.3, random_state=seed, stratify=y_res
        )
        leaky = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        leaky_scores.append(
            classification_report(yte, leaky.predict_proba(Xte)[:, 1])["balanced_accuracy"]
        )

        # --- Right: split first, resample inside the pipeline.
        Xtr2, Xte2, ytr2, yte2 = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        clean = make_balanced_pipeline(
            LogisticRegression(max_iter=2000), sampler="smote", k_neighbors=5, random_state=seed
        ).fit(Xtr2, ytr2)
        clean_scores.append(
            classification_report(yte2, clean.predict_proba(Xte2)[:, 1])["balanced_accuracy"]
        )

    leaky_mean = float(np.mean(leaky_scores))
    clean_mean = float(np.mean(clean_scores))

    assert leaky_mean > 0.55, (
        f"the leak did not reproduce (mean {leaky_mean:.3f} on pure noise); if this stops "
        "failing the test no longer demonstrates anything, so fix the test rather "
        "than deleting it"
    )
    assert clean_mean < 0.53, (
        f"the leakage-safe pipeline averaged {clean_mean:.3f} on pure noise, which it should "
        "not be able to do"
    )
    assert leaky_mean - clean_mean > 0.06, (
        f"leak inflation was only {leaky_mean - clean_mean:.3f}"
    )


def test_out_of_fold_proba_covers_every_row(small_dataset):
    dataset, _ = small_dataset
    oof, fold_ids = out_of_fold_proba(
        LogisticRegression(max_iter=1000),
        dataset.blocks["fmri"].X,
        dataset.y_individual,
        dataset.subject_ids,
        n_splits=4,
        seed=0,
    )
    assert not np.isnan(oof).any()
    assert (fold_ids >= 0).all()
    assert set(np.unique(fold_ids)) == {0, 1, 2, 3}


def test_out_of_fold_scores_are_lower_than_in_sample(small_dataset):
    """In-sample beats out-of-fold. If it does not, the split is broken."""
    dataset, _ = small_dataset
    X = dataset.blocks["face"].X  # high-dimensional block, overfits readily
    y = dataset.y_individual

    model = LogisticRegression(max_iter=2000).fit(X, y)
    in_sample = classification_report(y, model.predict_proba(X)[:, 1])["balanced_accuracy"]

    oof, _ = out_of_fold_proba(
        LogisticRegression(max_iter=2000), X, y, dataset.subject_ids, n_splits=4, seed=0
    )
    out_of_fold = classification_report(y, oof)["balanced_accuracy"]

    assert in_sample > out_of_fold
