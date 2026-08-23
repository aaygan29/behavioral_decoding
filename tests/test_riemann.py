"""Riemannian tangent-space mapping and the `riemann` estimator path.

The transformer maps flattened EEG covariance features into the log-Euclidean
tangent space, where the framework's scaler/SMOTE/logistic become valid again.
These tests check the vectorisation is isometric and invertible, the log map
survives rank-deficient covariances, the reference mean is leakage-safe, and the
wired-up `riemann` model classifies covariance structure above chance.

Note on scope: on clean synthetic data the raw covariance entries are usually
already linearly separable, so these tests deliberately do NOT assert that
riemann beats a plain logistic. The tangent map's advantage is on ill-conditioned
real EEG; claiming a synthetic win would be cherry-picking. See docs/estimators.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from behavioral_decoding.models.riemann import (
    RiemannianTangentSpace,
    flatten_spd,
    n_channels_from_flat,
    unflatten_spd,
)

pytest.importorskip("imblearn")


def _spd(n, k, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, k, k))
    return A @ A.transpose(0, 2, 1) + k * np.eye(k)


def _covariance_dataset(n=180, k=6, seed=0, samples=200, noise=0.0, short=False):
    """Two classes differing in which channel pair is correlated.

    ``noise`` adds channel-wise measurement noise (a heavier diagonal, weaker
    off-diagonal structure), and ``short`` shrinks the epoch length so the
    per-epoch covariances become rank-deficient and ill-conditioned. Both make
    the problem look more like real EEG.
    """
    rng = np.random.default_rng(seed)
    subjects = np.repeat(np.arange(9), n // 9)
    y = rng.integers(0, 2, size=n)
    t = k - 1 if short else samples  # short epochs => singular covariance
    mats = []
    for i in range(n):
        W = rng.normal(size=(k, k)) * 0.3 + np.eye(k)
        if y[i] == 1:
            W[0, 1] = W[1, 0] = 0.9
        else:
            W[2, 3] = W[3, 2] = 0.9
        sig = W @ rng.normal(size=(k, t))
        if noise:
            sig = sig + rng.normal(scale=noise, size=sig.shape)
        # bias=True keeps the estimate defined even when t < k (rank-deficient).
        mats.append(np.cov(sig, bias=True))
    return flatten_spd(np.stack(mats)), y, subjects


# ---------------------------------------------------------------- vectorisation


def test_n_channels_inferred_from_triangular_length():
    assert n_channels_from_flat(21) == 6  # 6*7/2
    assert n_channels_from_flat(3) == 2
    assert n_channels_from_flat(1) == 1


def test_non_triangular_length_is_rejected():
    with pytest.raises(ValueError, match="not a flattened covariance|not k"):
        n_channels_from_flat(20)


def test_flatten_is_invertible():
    A = _spd(5, 4)
    back = unflatten_spd(flatten_spd(A), 4)
    assert np.allclose(A, back)


def test_flatten_preserves_frobenius_norm():
    """Off-diagonal sqrt(2) scaling makes the vectorisation an isometry."""
    A = _spd(7, 5)
    flat = flatten_spd(A)
    assert np.allclose(np.linalg.norm(flat, axis=1), np.linalg.norm(A, axis=(1, 2)))


# ---------------------------------------------------------------- transformer


def test_transform_shape_and_finiteness():
    X, _, _ = _covariance_dataset()
    Z = RiemannianTangentSpace().fit_transform(X)
    assert Z.shape == X.shape
    assert np.isfinite(Z).all()


def test_log_map_survives_rank_deficient_covariance():
    """Covariances from short epochs are singular; shrinkage must rescue the log."""
    k = 5
    # A rank-1 (singular) covariance: outer product of one vector.
    v = np.arange(1.0, k + 1)
    singular = np.outer(v, v)
    flat = flatten_spd(singular[None])
    Z = RiemannianTangentSpace(shrinkage=1e-3).fit_transform(flat)
    assert np.isfinite(Z).all()


def test_reference_mean_is_training_only_leakage_safe():
    """transform() on new data must use the train mean, not refit to it."""
    X, _, _ = _covariance_dataset(seed=0)
    Xtr, Xte = X[:120], X[120:]
    ts = RiemannianTangentSpace().fit(Xtr)
    mean_after_fit = ts.mean_log_.copy()
    ts.transform(Xte)
    # transform must not have changed the stored reference.
    assert np.array_equal(mean_after_fit, ts.mean_log_)


def test_transformer_rejects_non_covariance_features():
    X = np.random.default_rng(0).normal(size=(10, 20))  # 20 is not triangular
    with pytest.raises(ValueError, match="not.*covariance|not k"):
        RiemannianTangentSpace().fit(X)


def test_centering_changes_the_representation():
    X, _, _ = _covariance_dataset()
    centered = RiemannianTangentSpace(center=True).fit_transform(X)
    uncentered = RiemannianTangentSpace(center=False).fit_transform(X)
    assert not np.allclose(centered, uncentered)


# ------------------------------------------------------------- estimator wiring


def test_riemann_model_prepends_tangent_step():
    from behavioral_decoding.models.modality_models import build_modality_model

    y = np.array([0] * 60 + [1] * 30)
    model = build_modality_model("eeg", y=y, base_learner="riemann", n_bags=5, seed=0)
    step_names = list(dict(model.estimator.steps))
    assert step_names[0] == "tangent"
    assert "scaler" in step_names and "estimator" in step_names
    assert model.bd_spec_["base_learner"] == "riemann"


def test_riemann_forces_max_features_to_one():
    """A column subset of a flattened covariance is not a covariance."""
    from behavioral_decoding.models.modality_models import build_modality_model

    y = np.array([0] * 60 + [1] * 30)
    model = build_modality_model(
        "eeg", y=y, base_learner="riemann", n_bags=5, max_features=0.5, seed=0
    )
    assert model.max_features == 1.0


def test_riemann_classifies_covariance_structure_above_chance():
    """Positive control: the wired path recovers the planted class, honestly.

    Above chance is the claim, not 'beats logistic': raw covariance entries are
    already separable on clean synthetic data.
    """
    from behavioral_decoding.evaluation.cv import out_of_fold_proba
    from behavioral_decoding.evaluation.metrics import classification_report
    from behavioral_decoding.models.modality_models import build_modality_model

    X, y, subjects = _covariance_dataset(seed=1)
    model = build_modality_model("eeg", y=y, base_learner="riemann", n_bags=8, seed=0)
    oof, _ = out_of_fold_proba(model, X, y, subjects, n_splits=3, seed=0)
    balacc = classification_report(y, oof)["balanced_accuracy"]
    assert balacc > 0.6, f"riemann did not recover covariance structure (balacc={balacc:.3f})"


# ---------------------------------------------------- noisy / realistic EEG cases


def test_transform_survives_noisy_covariances():
    """Heavy additive channel noise must not produce non-finite tangent vectors."""
    X, _, _ = _covariance_dataset(noise=1.5, seed=2)
    Z = RiemannianTangentSpace().fit_transform(X)
    assert np.isfinite(Z).all()


def test_transform_survives_short_epoch_rank_deficient_covariances():
    """Short-epoch covariances are singular; shrinkage must keep the log defined."""
    X, _, _ = _covariance_dataset(short=True, seed=3)
    Z = RiemannianTangentSpace(shrinkage=1e-2).fit_transform(X)
    assert np.isfinite(Z).all()


def test_riemann_recovers_structure_under_noise():
    """Positive control on a noisier, harder covariance dataset."""
    from behavioral_decoding.evaluation.cv import out_of_fold_proba
    from behavioral_decoding.evaluation.metrics import classification_report
    from behavioral_decoding.models.modality_models import build_modality_model

    X, y, subjects = _covariance_dataset(noise=0.8, samples=120, seed=4)
    model = build_modality_model("eeg", y=y, base_learner="riemann", n_bags=8, seed=0)
    oof, _ = out_of_fold_proba(model, X, y, subjects, n_splits=3, seed=0)
    balacc = classification_report(y, oof)["balanced_accuracy"]
    assert balacc > 0.55, f"riemann failed under noise (balacc={balacc:.3f})"


# ----------------------------------------------------- affine-invariant backend


def test_riemann_metric_requires_pyriemann_or_errors_clearly():
    """Selecting the affine-invariant metric without pyriemann fails loudly."""
    X, _, _ = _covariance_dataset(seed=5)
    ts = RiemannianTangentSpace(metric="riemann")
    try:
        import pyriemann  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pyriemann"):
            ts.fit(X)
    else:  # pragma: no cover - only when the optional backend is installed
        ts.fit(X)
        Z = ts.transform(X)
        assert np.isfinite(Z).all()


def test_unknown_metric_rejected():
    X, _, _ = _covariance_dataset(seed=6)
    with pytest.raises(ValueError, match="logeuclid|riemann"):
        RiemannianTangentSpace(metric="nonsense").fit(X)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyriemann") is None,
    reason="pyriemann backend not installed",
)
def test_pyriemann_backend_classifies_above_chance():  # pragma: no cover - optional
    from behavioral_decoding.evaluation.cv import out_of_fold_proba
    from behavioral_decoding.evaluation.metrics import classification_report
    from behavioral_decoding.models.modality_models import build_modality_model

    X, y, subjects = _covariance_dataset(noise=0.8, samples=120, seed=7)
    model = build_modality_model(
        "eeg", y=y, base_learner="riemann", riemann_metric="riemann", n_bags=8, seed=0
    )
    oof, _ = out_of_fold_proba(model, X, y, subjects, n_splits=3, seed=0)
    assert classification_report(y, oof)["balanced_accuracy"] > 0.55
