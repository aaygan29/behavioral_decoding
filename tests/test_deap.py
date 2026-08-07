"""DEAP loader tests, one per documented trap plus the end-to-end path.

These run against a synthetic download written in the true DEAP on-disk format
(latin1 pickles, randomised per-participant trial order, the real CSV layout),
so they exercise the format handling, not just the array maths. No licensed data
is needed. See ``tests/deap_fixture.py``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from behavioral_decoding.io import deap_market
from behavioral_decoding.io.deap import (
    DEAP_BANDS,
    DEAP_BASELINE_SAMPLES,
    DEAP_LABEL_NAMES,
    DEAP_N_SAMPLES,
    DEAP_PERIPHERAL_CHANNELS,
    PERIPHERAL,
    DEAPFormatError,
    DEAPLoader,
    binarise_ratings,
    find_subject_files,
    load_deap,
    load_subject_file,
    split_baseline,
    subject_id_from_path,
)
from deap_fixture import market_outcome_dict, write_deap_fixture


@pytest.fixture(scope="module")
def deap_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("deap")
    gt = write_deap_fixture(str(root), n_participants=6, n_videos=40, seed=0)
    return str(root), gt


# ---------------------------------------------------------------- format handling


def test_finds_and_names_participant_files(deap_root):
    root, _ = deap_root
    files = find_subject_files(root)
    assert len(files) == 6
    assert subject_id_from_path(files[0]) == "s01"


def test_non_deap_filename_is_rejected():
    with pytest.raises(DEAPFormatError, match="not a DEAP participant file"):
        subject_id_from_path(Path("subject7.dat"))


def test_latin1_pickle_loads(deap_root):
    """Trap 4: the archive is Python-2 pickled and needs encoding='latin1'."""
    root, _ = deap_root
    files = find_subject_files(root)
    data, labels = load_subject_file(str(files[0]))
    assert data.shape == (40, 40, 8064)
    assert labels.shape == (40, 4)


def test_wrong_data_shape_raises_loudly(tmp_path):
    """A different shape means a different release; the loader must not proceed."""
    bad = tmp_path / "s01.dat"
    with open(bad, "wb") as handle:
        pickle.dump(
            {"data": np.zeros((40, 32, 8064)), "labels": np.zeros((40, 4))},
            handle,
            protocol=2,
        )
    with pytest.raises(DEAPFormatError, match=r"expected \(40, 40, 8064\)"):
        load_subject_file(str(bad))


def test_missing_keys_raise(tmp_path):
    bad = tmp_path / "s02.dat"
    with open(bad, "wb") as handle:
        pickle.dump({"data": np.zeros((40, 40, 8064))}, handle, protocol=2)
    with pytest.raises(DEAPFormatError, match="missing"):
        load_subject_file(str(bad))


# --------------------------------------------------------------------- trap 1


def test_default_bands_exclude_delta():
    """Trap 1: DEAP is filtered 4-45 Hz, so delta is not in the data."""
    assert "delta" not in DEAP_BANDS
    assert min(lo for lo, _ in DEAP_BANDS.values()) >= 4.0


def test_band_outside_passband_is_rejected():
    with pytest.raises(ValueError, match="outside DEAP's 4.0-45.0 Hz passband"):
        DEAPLoader(bands={"delta": (1.0, 4.0)})


def test_band_above_passband_is_rejected():
    with pytest.raises(ValueError, match="passband"):
        DEAPLoader(bands={"high": (40.0, 60.0)})


# --------------------------------------------------------------------- trap 2


def test_label_order_is_valence_arousal_dominance_liking():
    """Trap 2: swapping valence and arousal is silent and wrong."""
    assert DEAP_LABEL_NAMES == ("valence", "arousal", "dominance", "liking")


def test_target_is_never_a_behaviour_feature(deap_root):
    root, _ = deap_root
    files = find_subject_files(root)
    data, labels = load_subject_file(str(files[0]))
    loader = DEAPLoader(behavior_mode="ratings", target="liking")
    blocks, _ = loader.blocks_from_arrays(
        data, labels, "s01", stimulus_ids=[f"exp-{i + 1:02d}" for i in range(40)]
    )
    names = blocks["behavior"].feature_names
    assert "rating_liking" not in names
    assert "rating_valence" in names  # the others are allowed


# --------------------------------------------------------------------- trap 3


def test_baseline_split_is_three_seconds():
    """Trap 3: 8064 = 384 baseline + 7680 trial."""
    assert DEAP_BASELINE_SAMPLES == 384
    data = np.arange(DEAP_N_SAMPLES).reshape(1, 1, DEAP_N_SAMPLES).astype(float)
    baseline, trial = split_baseline(data)
    assert baseline.shape[-1] == 384
    assert trial.shape[-1] == 7680
    assert trial[0, 0, 0] == 384.0  # first post-baseline sample


def test_baseline_correction_changes_the_features(deap_root):
    root, _ = deap_root
    files = find_subject_files(root)
    data, labels = load_subject_file(str(files[0]))
    stim = [f"exp-{i + 1:02d}" for i in range(40)]

    corrected = DEAPLoader(baseline_correct=True).blocks_from_arrays(
        data, labels, "s01", stimulus_ids=stim
    )[0]["eeg"]
    uncorrected = DEAPLoader(baseline_correct=False).blocks_from_arrays(
        data, labels, "s01", stimulus_ids=stim
    )[0]["eeg"]
    assert not np.allclose(corrected.X, uncorrected.X)


# --------------------------------------------------------------------- trap 5


def test_peripheral_channels_are_a_separate_block(deap_root):
    """Trap 5: GSR/respiration/EMG are not EEG and get their own block."""
    root, _ = deap_root
    files = find_subject_files(root)
    data, labels = load_subject_file(str(files[0]))
    blocks, _ = load_deap_blocks(data, labels)
    assert PERIPHERAL in blocks
    assert blocks[PERIPHERAL].n_features == len(DEAP_PERIPHERAL_CHANNELS) * 4
    # No EEG band-power names leaked into the peripheral block.
    assert all("theta" not in n for n in blocks[PERIPHERAL].feature_names)


def load_deap_blocks(data, labels):
    stim = [f"exp-{i + 1:02d}" for i in range(40)]
    return DEAPLoader().blocks_from_arrays(data, labels, "s01", stimulus_ids=stim)


# --------------------------------------------------------------------- trap 6


def test_asymmetry_features_are_present_and_frontal(deap_root):
    root, _ = deap_root
    files = find_subject_files(root)
    data, labels = load_subject_file(str(files[0]))
    blocks, _ = load_deap_blocks(data, labels)
    asym = [n for n in blocks["eeg"].feature_names if n.startswith("asym_")]
    assert asym
    assert any("F3" in n and "F4" in n for n in asym)


# ----------------------------------------------------------- stimulus keying


def test_blocks_require_stimulus_ids(deap_root):
    """Trial order is randomised per participant, so index is not a stimulus."""
    root, _ = deap_root
    files = find_subject_files(root)
    data, labels = load_subject_file(str(files[0]))
    with pytest.raises(ValueError, match="randomised order"):
        DEAPLoader().blocks_from_arrays(data, labels, "s01", stimulus_ids=None)


def test_randomised_order_is_actually_recovered(deap_root):
    """The same video seen by two participants must get the same stimulus key."""
    root, _ = deap_root
    ds = load_deap(root, target="liking")
    # Every stimulus should be seen by all 6 participants (fully crossed design).
    from collections import Counter

    counts = Counter(ds.stimulus_ids.tolist())
    assert set(counts.values()) == {6}
    assert len(counts) == 40


# --------------------------------------------------------------- circularity


def test_behavior_defaults_to_noncircular(deap_root):
    root, _ = deap_root
    files = find_subject_files(root)
    data, labels = load_subject_file(str(files[0]))
    blocks, _ = DEAPLoader().blocks_from_arrays(
        data, labels, "s01", stimulus_ids=[f"exp-{i + 1:02d}" for i in range(40)],
        familiarity=np.arange(40, dtype=float),
    )
    names = blocks["behavior"].feature_names
    assert set(names) == {"familiarity", "trial_index"}
    assert all(not n.startswith("rating_") for n in names)
    assert blocks["behavior"].provenance["circular"] is False


def test_ratings_mode_is_flagged_circular(deap_root):
    root, _ = deap_root
    files = find_subject_files(root)
    data, labels = load_subject_file(str(files[0]))
    blocks, _ = DEAPLoader(behavior_mode="ratings").blocks_from_arrays(
        data, labels, "s01", stimulus_ids=[f"exp-{i + 1:02d}" for i in range(40)]
    )
    assert blocks["behavior"].provenance["circular"] is True


def test_invalid_behavior_mode_rejected():
    with pytest.raises(ValueError, match="behavior_mode"):
        DEAPLoader(behavior_mode="everything")


# --------------------------------------------------------------- binarisation


def test_fixed_binarisation_splits_at_threshold():
    values = np.array([1.0, 4.0, 5.0, 6.0, 9.0])
    subjects = np.array(["s01"] * 5)
    labels, info = binarise_ratings(values, subjects, method="fixed", threshold=5.0)
    # 5.0 goes to the low class by design (reproducible midpoint handling).
    assert list(labels) == [0, 0, 0, 1, 1]
    assert info["method"] == "fixed"


def test_subject_median_balances_within_participant():
    values = np.array([1.0, 2.0, 8.0, 9.0, 3.0, 4.0, 6.0, 7.0])
    subjects = np.array(["s01"] * 4 + ["s02"] * 4)
    labels, info = binarise_ratings(values, subjects, method="subject_median")
    assert labels[:4].sum() == 2  # balanced within s01
    assert labels[4:].sum() == 2  # balanced within s02
    assert "subject_medians" in info


def test_binarisation_refuses_a_single_class():
    values = np.array([7.0, 8.0, 9.0])
    subjects = np.array(["s01"] * 3)
    with pytest.raises(ValueError, match="single class"):
        binarise_ratings(values, subjects, method="fixed", threshold=5.0)


# ----------------------------------------------------------------- end to end


def test_load_deap_builds_a_multimodal_dataset(deap_root):
    root, _ = deap_root
    ds = load_deap(root, target="liking", binarise="fixed")
    assert set(ds.modalities) == {"eeg", "peripheral", "behavior"}
    assert ds.n_subjects == 6
    assert ds.n_stimuli == 40
    assert ds.n_trials == 240
    assert not ds.metadata["synthetic"]
    assert ds.metadata["dataset"] == "DEAP"


def test_load_deap_requires_ratings_csv(tmp_path):
    # A data dir with no metadata_csv/participant_ratings.csv.
    data_dir = tmp_path / "data_preprocessed_python"
    data_dir.mkdir()
    with open(data_dir / "s01.dat", "wb") as handle:
        pickle.dump(
            {"data": np.zeros((40, 40, 8064)), "labels": np.zeros((40, 4)) + 5},
            handle,
            protocol=2,
        )
    with pytest.raises(FileNotFoundError, match="participant_ratings.csv"):
        load_deap(str(tmp_path), target="liking")


def test_target_mismatch_between_loader_and_call_is_rejected(deap_root):
    root, _ = deap_root
    with pytest.raises(ValueError, match="target mismatch"):
        load_deap(root, target="liking", loader=DEAPLoader(target="valence"))


# --------------------------------------------------------------- market route


def test_youtube_ids_parse_from_video_list(deap_root):
    root, _ = deap_root
    ids = deap_market.youtube_ids_from_video_list(str(Path(root) / "video_list.csv"))
    assert len(ids) == 40
    assert all(rec["youtube_id"] for rec in ids.values())
    assert all(len(rec["youtube_id"]) == 11 for rec in ids.values())


def test_extract_youtube_id_handles_url_forms():
    assert deap_market.extract_youtube_id("https://youtu.be/abcdefghijk") == "abcdefghijk"
    assert deap_market.extract_youtube_id("https://www.youtube.com/watch?v=ABCDEFGHIJK") == "ABCDEFGHIJK"
    assert deap_market.extract_youtube_id("not a link") is None


def test_market_outcome_rejects_non_experiment_keys():
    with pytest.raises(ValueError, match="not exp-NN"):
        deap_market.market_outcome_from_counts({"abcdefghijk": 1000.0})


def test_market_outcome_log_transforms_and_flags_missing_date():
    counts = {"exp-01": 1000.0, "exp-02": 1_000_000.0}
    y, prov = deap_market.market_outcome_from_counts(counts, log_transform=True)
    assert y["exp-01"] == pytest.approx(np.log1p(1000.0))
    assert prov["log_transform"] is True
    assert "warning" in prov  # no fetch_date supplied


def test_market_outcome_remaps_youtube_ids():
    counts = {"abcdefghijk": 5000.0}
    mapping = {"abcdefghijk": "exp-07"}
    y, _ = deap_market.market_outcome_from_counts(
        counts, id_to_experiment=mapping, log_transform=False
    )
    assert y == {"exp-07": 5000.0}


def test_full_dissociation_recovers_on_the_fixture(deap_root):
    """The framework's headline check, on real DEAP file format.

    The fixture plants a latent per-video valence that drives both frontal EEG
    asymmetry and the market outcome, while the noncircular behaviour block
    (familiarity, trial order) is blind to it. EEG should forecast the market;
    behaviour should not. This is the DEAP analogue of the synthetic positive
    control, and if it breaks the loader is miswiring the stimulus keys or the
    asymmetry features.
    """
    from behavioral_decoding.evaluation.neuroforecast import compare_forecast_arms

    root, gt = deap_root
    y_agg = market_outcome_dict(gt)
    ds = load_deap(root, target="liking", y_aggregate=y_agg)
    arms = compare_forecast_arms(ds, task="regression", n_splits=5, seed=0)

    assert arms["eeg_only"]["r2_out_of_sample"] > arms["behavior_only"]["r2_out_of_sample"]
    assert arms["eeg_only"]["pearson_p"] < 0.05
