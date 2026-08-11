"""Base-learner factory: the estimators each modality can use.

These check that every named learner instantiates with the right regularisation,
exposes ``predict_proba`` (the ensemble weights on out-of-fold probabilities, so
a learner without it is unusable), and survives a fit inside the leakage-safe
pipeline on imbalanced data.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from behavioral_decoding.balance.smote import make_balanced_pipeline
from behavioral_decoding.models.modality_models import build_modality_model, make_base_learner

pytest.importorskip("imblearn")


def _imbalanced(seed: int = 0, n: int = 160, d: int = 8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.array([0] * (n - n // 5) + [1] * (n // 5))
    # Plant a little separable signal so predict_proba is not degenerate.
    X[y == 1] += 0.6
    return X, y


def test_elasticnet_is_l1_l2_logistic_with_saga():
    est = make_base_learner("elasticnet")
    assert isinstance(est, LogisticRegression)
    assert est.penalty == "elasticnet"
    assert est.solver == "saga"
    assert 0.0 < est.l1_ratio < 1.0


def test_linear_svm_is_linear_kernel_with_probability():
    est = make_base_learner("linear_svm")
    assert isinstance(est, SVC)
    assert est.kernel == "linear"
    assert est.probability is True


def test_unknown_learner_lists_the_valid_names():
    with pytest.raises(ValueError, match="elasticnet.*linear_svm"):
        make_base_learner("transformer")


@pytest.mark.parametrize("kind", ["logistic", "elasticnet", "linear_svm", "random_forest"])
def test_learner_fits_and_predicts_proba_in_pipeline(kind):
    X, y = _imbalanced()
    pipe = make_balanced_pipeline(
        make_base_learner(kind, class_weight="balanced"),
        sampler="smote",
        k_neighbors=5,
        random_state=0,
    )
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    # Above chance on planted signal, so the column is informative.
    assert proba[y == 1, 1].mean() > proba[y == 0, 1].mean()


def test_kwargs_override_defaults():
    est = make_base_learner("elasticnet", l1_ratio=0.2, C=0.5)
    assert est.l1_ratio == 0.2
    assert est.C == 0.5


@pytest.mark.parametrize("kind", ["elasticnet", "linear_svm"])
def test_build_modality_model_accepts_new_learners(kind):
    y = np.array([0] * 80 + [1] * 20)
    model = build_modality_model("fmri", y=y, base_learner=kind, n_bags=5, seed=0)
    assert model.n_estimators == 5
    assert model.bd_spec_["base_learner"] == kind
    # class_weight balanced is passed through for both (neither is gradient boosting).
    assert model.bd_spec_["class_weight"] == "balanced"
