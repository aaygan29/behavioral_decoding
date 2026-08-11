"""EEG covariance feature family and its end-to-end path into the riemann model.

The covariance mode produces the flattened inter-channel covariance that the
Riemannian tangent-space estimator consumes. These tests check the shape, the
naming, the mutual-exclusivity guard, and that the loader's output round-trips
through the transformer's inverse (the two share one vectorisation convention,
so this guards against them drifting apart).
"""

from __future__ import annotations

import numpy as np
import pytest

from behavioral_decoding.io.eeg import EEGLoader, channel_covariance
from behavioral_decoding.models.riemann import n_channels_from_flat, unflatten_spd


def _epochs(n=12, k=8, t=128, seed=0):
    return np.random.default_rng(seed).normal(size=(n, k, t))


def test_channel_covariance_shape_and_names():
    feats, names = channel_covariance(_epochs(n=10, k=8))
    assert feats.shape == (10, 8 * 9 // 2)  # 36
    assert names[0] == "cov_ch00_ch00"
    assert len(names) == feats.shape[1]
    assert n_channels_from_flat(feats.shape[1]) == 8


def test_covariance_needs_enough_samples():
    with pytest.raises(ValueError, match="at least 2 samples"):
        channel_covariance(np.zeros((3, 4, 1)))


def test_loader_covariance_block_is_labelled():
    block = EEGLoader(
        include_bandpower=False, include_erp=False, include_covariance=True
    ).from_arrays(
        _epochs(), sfreq=128.0, times=np.linspace(0, 1, 128),
        subject_ids=["s1"] * 12, stimulus_ids=list(range(12)),
    )
    assert block.provenance["feature_family"] == "covariance"
    assert block.n_features == 8 * 9 // 2


def test_loader_output_round_trips_through_transformer_inverse():
    """The loader and the transformer must share one vectorisation convention."""
    epochs = _epochs(n=5, k=6)
    feats, _ = channel_covariance(epochs)
    recon = unflatten_spd(feats, 6)
    for i in range(len(epochs)):
        assert np.allclose(recon[i], np.cov(epochs[i]))


def test_covariance_is_mutually_exclusive_with_bandpower_and_erp():
    with pytest.raises(ValueError, match="mutually exclusive"):
        EEGLoader(include_covariance=True)  # bandpower/erp default to True
    # Explicitly disabling the others is allowed.
    EEGLoader(include_bandpower=False, include_erp=False, include_covariance=True)


def test_covariance_block_feeds_the_riemann_model_end_to_end():
    pytest.importorskip("imblearn")
    from behavioral_decoding.evaluation.cv import out_of_fold_proba
    from behavioral_decoding.evaluation.metrics import classification_report
    from behavioral_decoding.models.modality_models import build_modality_model

    # Plant a covariance-structured class difference across subjects.
    rng = np.random.default_rng(1)
    n, k, t = 144, 6, 200
    subjects = np.repeat(np.arange(9), n // 9)
    y = rng.integers(0, 2, size=n)
    epochs = np.empty((n, k, t))
    for i in range(n):
        W = rng.normal(size=(k, k)) * 0.3 + np.eye(k)
        if y[i] == 1:
            W[0, 1] = W[1, 0] = 0.9
        epochs[i] = W @ rng.normal(size=(k, t))

    block = EEGLoader(
        include_bandpower=False, include_erp=False, include_covariance=True
    ).from_arrays(
        epochs, sfreq=200.0, times=np.linspace(0, 1, t),
        subject_ids=subjects, stimulus_ids=np.arange(n),
    )
    model = build_modality_model("eeg", y=y, base_learner="riemann", n_bags=8, seed=0)
    oof, _ = out_of_fold_proba(model, block.X, y, block.subject_ids, n_splits=3, seed=0)
    assert classification_report(y, oof)["balanced_accuracy"] > 0.6
