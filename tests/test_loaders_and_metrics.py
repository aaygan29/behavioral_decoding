"""Loader contracts, feature extraction, and metric honesty."""

from __future__ import annotations

import numpy as np
import pytest

from behavioral_decoding.balance.smote import AdaptiveOverSampler, recommend_strategy
from behavioral_decoding.evaluation.metrics import (
    bootstrap_ci,
    classification_report,
    permutation_test,
)
from behavioral_decoding.features.vit_encoder import FALLBACK_BACKEND, ViTEncoder
from behavioral_decoding.io.base import ModalityBlock
from behavioral_decoding.io.behavior import BehaviorLoader
from behavioral_decoding.io.eeg import EEGLoader, bandpower, erp_windows
from behavioral_decoding.io.face import FaceLoader
from behavioral_decoding.io.fmri import DEFAULT_ROIS, FMRILoader
from behavioral_decoding.io.registry import get_loader

# ------------------------------------------------------------------- contracts


def test_block_rejects_mismatched_key_lengths():
    with pytest.raises(ValueError, match="length n_trials"):
        ModalityBlock(
            name="fmri",
            X=np.zeros((3, 2)),
            subject_ids=np.array(["s1", "s2"]),
            stimulus_ids=np.array(["a", "b", "c"]),
        )


def test_block_rejects_one_dimensional_features():
    with pytest.raises(ValueError, match="must be 2-D"):
        ModalityBlock(
            name="eeg", X=np.zeros(5), subject_ids=np.zeros(5), stimulus_ids=np.zeros(5)
        )


def test_block_select_preserves_keys():
    block = ModalityBlock(
        name="fmri",
        X=np.arange(12).reshape(4, 3).astype(float),
        subject_ids=np.array(["a", "a", "b", "b"]),
        stimulus_ids=np.array(["x", "y", "x", "y"]),
    )
    sub = block.select(np.array([0, 3]))
    assert sub.n_trials == 2
    assert list(sub.subject_ids) == ["a", "b"]
    assert np.allclose(sub.X, [[0, 1, 2], [9, 10, 11]])


def test_registry_returns_the_right_loader():
    assert isinstance(get_loader("fmri"), FMRILoader)
    assert isinstance(get_loader("eeg"), EEGLoader)
    with pytest.raises(KeyError, match="unknown modality"):
        get_loader("meg")


# --------------------------------------------------------------------- loaders


def test_fmri_default_rois_cover_the_anticipatory_affect_circuit():
    assert {"NAcc_L", "NAcc_R", "MPFC", "AIns_L", "AIns_R"} <= set(DEFAULT_ROIS)


def test_fmri_from_arrays_labels_features_with_roi_names():
    loader = FMRILoader()
    block = loader.from_arrays(
        np.zeros((4, 5)), ["s1"] * 4, ["a", "b", "c", "d"]
    )
    assert block.feature_names == list(DEFAULT_ROIS.keys())
    assert block.provenance["modality"] == "fmri"


def test_eeg_bandpower_tracks_a_planted_oscillation():
    sfreq = 250.0
    t = np.arange(0, 2.0, 1 / sfreq)
    alpha = np.sin(2 * np.pi * 10 * t)
    beta = np.sin(2 * np.pi * 20 * t)
    epochs = np.stack([alpha[None, :], beta[None, :]])  # (2 epochs, 1 channel, n_times)

    feats, names = bandpower(epochs, sfreq, {"alpha": (8, 13), "beta": (13, 30)})
    alpha_col, beta_col = names.index("alpha_ch00"), names.index("beta_ch00")

    assert feats[0, alpha_col] > feats[0, beta_col]
    assert feats[1, beta_col] > feats[1, alpha_col]


def test_eeg_bandpower_refuses_an_unresolvable_band():
    epochs = np.zeros((2, 1, 8))
    with pytest.raises(ValueError, match="not resolvable"):
        bandpower(epochs, sfreq=100.0, bands={"delta": (1.0, 2.0)})


def test_erp_window_outside_the_epoch_raises():
    epochs = np.zeros((2, 1, 100))
    times = np.linspace(-0.2, 0.5, 100)
    with pytest.raises(ValueError, match="outside the epoch span"):
        erp_windows(epochs, times, [("late", 1.0, 1.5)])


def test_eeg_loader_produces_one_row_per_epoch():
    rng = np.random.default_rng(0)
    epochs = rng.normal(size=(12, 4, 250))
    times = np.linspace(-0.2, 0.8, 250)
    block = EEGLoader().from_arrays(
        epochs, sfreq=250.0, times=times, subject_ids=["s1"] * 12, stimulus_ids=list(range(12))
    )
    assert block.n_trials == 12
    # 5 bands x 4 channels + 2 ERP windows x 4 channels
    assert block.n_features == 5 * 4 + 2 * 4


def test_behavior_loader_refuses_to_use_the_outcome_as_a_feature():
    pd = pytest.importorskip("pandas")
    table = pd.DataFrame(
        {
            "subject_id": ["s1", "s2"],
            "stimulus_id": ["a", "b"],
            "choice": [1, 0],
            "rating": [3.0, 5.0],
        }
    )
    with pytest.raises(ValueError, match="must not be used as features"):
        BehaviorLoader().load(
            table, outcome_column="choice", feature_columns=["rating", "choice"]
        )


def test_behavior_loader_excludes_keys_and_outcome_by_default():
    pd = pytest.importorskip("pandas")
    table = pd.DataFrame(
        {
            "subject_id": ["s1", "s2"],
            "stimulus_id": ["a", "b"],
            "choice": [1, 0],
            "rating": [3.0, 5.0],
            "rt": [1.2, 0.8],
        }
    )
    block = BehaviorLoader().load(table, outcome_column="choice")
    assert set(block.feature_names) == {"rating", "rt"}
    assert block.provenance["log_transformed"] == ["rt"]


# --------------------------------------------------------------------- encoder


def test_vit_fallback_is_labelled_and_refuses_to_be_called_reportable():
    encoder = ViTEncoder(model_name="definitely-not-a-real-model", allow_fallback=True)
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    out = encoder.encode(frames)

    assert out.shape == (3, encoder.fallback_dim)
    assert encoder.backend == FALLBACK_BACKEND
    assert encoder.describe()["reportable"] is False
    with pytest.raises(RuntimeError, match="meaningless features"):
        encoder.assert_real_encoder()


def test_vit_can_be_configured_to_refuse_the_fallback():
    encoder = ViTEncoder(model_name="definitely-not-a-real-model", allow_fallback=False)
    with pytest.raises(RuntimeError, match="allow_fallback=False"):
        encoder.encode(np.zeros((1, 8, 8, 3), dtype=np.uint8))


def test_face_mean_std_pooling_keeps_within_trial_variability():
    loader = FaceLoader(pooling="mean_std")
    encoder = ViTEncoder(model_name="not-real", fallback_dim=4)
    steady = [np.zeros((4, 8, 8, 3), dtype=np.uint8)]
    block = FaceLoader(pooling="mean_std", encoder=encoder).from_frames(
        steady, ["s1"], ["a"]
    )
    assert block.n_features == 8  # 4 mean + 4 std
    assert loader.pooling == "mean_std"


# --------------------------------------------------------------------- metrics


def test_report_exposes_the_majority_baseline_next_to_accuracy():
    y = np.array([0] * 85 + [1] * 15)
    always_zero = np.zeros(100)
    report = classification_report(y, always_zero)

    assert report["accuracy"] == pytest.approx(0.85)
    assert report["majority_baseline_accuracy"] == pytest.approx(0.85)
    assert report["balanced_accuracy"] == pytest.approx(0.5)


def test_average_precision_lift_is_measured_against_the_positive_rate():
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20)
    proba = rng.uniform(size=100)
    report = classification_report(y, proba)
    assert report["ap_lift_over_chance"] == pytest.approx(
        report["average_precision"] - 0.2, abs=1e-9
    )


def test_subject_level_bootstrap_is_wider_when_subjects_actually_differ():
    """Resampling trials pretends repeated measures are independent.

    The gap only appears when there is real between-subject variance to ignore,
    so it is built in here: subjects differ widely in how predictable they are.
    That is the situation in every real recording, and it is exactly when a
    trial-level interval understates the uncertainty.
    """
    rng = np.random.default_rng(0)
    n_subjects, n_per_subject = 12, 20

    # Subjects range from near-chance to highly predictable.
    predictability = rng.uniform(0.0, 1.6, size=n_subjects)

    y_parts, proba_parts, subject_parts = [], [], []
    for i in range(n_subjects):
        yi = rng.integers(0, 2, size=n_per_subject)
        pi = np.clip(
            0.5 + predictability[i] * (yi - 0.5) + rng.normal(0, 0.08, n_per_subject),
            0.01,
            0.99,
        )
        y_parts.append(yi)
        proba_parts.append(pi)
        subject_parts.append(np.full(n_per_subject, i))

    y = np.concatenate(y_parts)
    proba = np.concatenate(proba_parts)
    subjects = np.concatenate(subject_parts)

    trial = bootstrap_ci(y, proba, n_boot=800, seed=0)
    subject = bootstrap_ci(y, proba, n_boot=800, groups=subjects, seed=0)

    trial_width = trial["hi"] - trial["lo"]
    subject_width = subject["hi"] - subject["lo"]

    assert subject_width > trial_width, (
        f"subject-level CI ({subject_width:.4f}) was not wider than trial-level ({trial_width:.4f})"
    )
    # The understatement is large, not marginal, which is why it matters.
    assert subject_width > 2.0 * trial_width


def test_permutation_p_can_never_be_zero():
    y = np.array([0] * 50 + [1] * 50)
    result = permutation_test(y, y.astype(float), n_perm=100, seed=0)
    assert result["p_value"] > 0.0
    assert result["p_value"] == pytest.approx(1.0 / 101.0, abs=1e-6)


def test_permutation_detects_the_absence_of_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    proba = rng.uniform(size=200)
    result = permutation_test(y, proba, n_perm=300, seed=0)
    assert result["p_value"] > 0.05


# --------------------------------------------------------------------- balance


def test_recommend_strategy_declines_smote_when_the_minority_is_tiny():
    y = np.array([0] * 100 + [1] * 3)
    rec = recommend_strategy(y)
    assert rec["sampler"] == "none"
    assert rec["class_weight"] == "balanced"
    assert "minority samples" in rec["rationale"]


def test_recommend_strategy_prefers_class_weights_at_moderate_imbalance():
    y = np.array([0] * 100 + [1] * 45)
    rec = recommend_strategy(y)
    assert rec["sampler"] == "none"
    assert rec["class_weight"] == "balanced"


def test_recommend_strategy_reaches_for_smote_only_at_real_imbalance():
    y = np.array([0] * 200 + [1] * 20)
    rec = recommend_strategy(y)
    assert rec["sampler"] == "smote"
    assert rec["k_neighbors"] == 5


def test_recommend_strategy_uses_borderline_for_high_dimensional_face_features():
    y = np.array([0] * 200 + [1] * 20)
    rec = recommend_strategy(y, modality="face")
    assert rec["sampler"] == "borderline"


def test_adaptive_sampler_resamples_when_the_minority_is_large_enough():
    pytest.importorskip("imblearn")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    y = np.array([0] * 100 + [1] * 20)

    X_res, y_res = AdaptiveOverSampler().fit_resample(X, y)
    counts = np.bincount(y_res)
    assert counts[0] == counts[1] == 100


def test_adaptive_sampler_steps_aside_instead_of_raising_on_a_tiny_minority():
    """The failure this prevents: a bagging bootstrap drawing 2 minority rows."""
    pytest.importorskip("imblearn")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = np.array([0] * 18 + [1] * 2)

    X_res, y_res = AdaptiveOverSampler(k_neighbors=5).fit_resample(X, y)
    assert X_res.shape == X.shape
    assert np.array_equal(y_res, y)


def test_adaptive_sampler_handles_a_single_class_slice():
    pytest.importorskip("imblearn")
    X = np.zeros((10, 3))
    y = np.zeros(10, dtype=int)
    X_res, y_res = AdaptiveOverSampler().fit_resample(X, y)
    assert np.array_equal(y_res, y)


def test_adaptive_sampler_clamps_k_to_the_available_neighbours():
    """k=5 with 7 minority samples must not raise; it should clamp and proceed."""
    pytest.importorskip("imblearn")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = np.array([0] * 53 + [1] * 7)

    X_res, y_res = AdaptiveOverSampler(k_neighbors=20).fit_resample(X, y)
    assert len(y_res) > len(y)
    assert np.bincount(y_res)[1] == 53


def test_adaptive_sampler_is_clonable():
    from sklearn.base import clone

    sampler = AdaptiveOverSampler(sampler="borderline", k_neighbors=3, min_minority=4)
    copy = clone(sampler)
    assert copy.get_params() == sampler.get_params()
