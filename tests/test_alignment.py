"""Trial alignment across modalities, and the errors it must refuse to hide."""

from __future__ import annotations

import numpy as np
import pytest

from behavioral_decoding.features.align import align_blocks, build_dataset, summarise_alignment
from behavioral_decoding.io.base import ModalityBlock


def _block(name, subjects, stimuli, n_features=3, offset=0.0):
    rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
    return ModalityBlock(
        name=name,
        X=rng.normal(size=(len(subjects), n_features)) + offset,
        subject_ids=np.array(subjects),
        stimulus_ids=np.array(stimuli),
    )


def test_inner_join_keeps_only_shared_trials():
    a = _block("fmri", ["s1", "s1", "s2"], ["x", "y", "x"])
    b = _block("eeg", ["s1", "s2", "s2"], ["x", "x", "y"])

    aligned, subjects, stimuli = align_blocks({"fmri": a, "eeg": b}, how="inner")

    keys = set(zip(subjects, stimuli))
    assert keys == {("s1", "x"), ("s2", "x")}
    for block in aligned.values():
        assert block.n_trials == 2
        assert not np.isnan(block.X).any()


def test_aligned_blocks_are_row_identical_on_keys():
    """Row i must be the same trial in every block. This is the whole point."""
    a = _block("fmri", ["s2", "s1"], ["y", "x"])
    b = _block("eeg", ["s1", "s2"], ["x", "y"])

    aligned, subjects, stimuli = align_blocks({"fmri": a, "eeg": b})

    for block in aligned.values():
        assert np.array_equal(block.subject_ids, subjects)
        assert np.array_equal(block.stimulus_ids, stimuli)

    # The reordering must actually have moved the data, not just the labels.
    row_of = {(s, st): i for i, (s, st) in enumerate(zip(a.subject_ids, a.stimulus_ids))}
    for i, (s, st) in enumerate(zip(subjects, stimuli)):
        assert np.allclose(aligned["fmri"].X[i], a.X[row_of[(s, st)]])


def test_duplicate_trial_keys_raise_rather_than_silently_picking_one():
    dup = _block("fmri", ["s1", "s1"], ["x", "x"])
    with pytest.raises(ValueError, match="duplicate trial key"):
        align_blocks({"fmri": dup})


def test_disjoint_modalities_raise_with_an_actionable_message():
    a = _block("fmri", ["s1"], ["x"])
    b = _block("eeg", ["sub-1"], ["stim-x"])
    with pytest.raises(ValueError, match="same format"):
        align_blocks({"fmri": a, "eeg": b}, how="inner")


def test_outer_join_marks_missing_rows_as_nan():
    a = _block("fmri", ["s1", "s2"], ["x", "x"])
    b = _block("eeg", ["s1"], ["x"])

    aligned, subjects, _ = align_blocks({"fmri": a, "eeg": b}, how="outer")
    assert len(subjects) == 2
    assert np.isnan(aligned["eeg"].X).any()
    assert aligned["eeg"].provenance["n_missing_rows"] == 1


def test_build_dataset_requires_an_outcome_for_every_trial():
    a = _block("fmri", ["s1", "s2"], ["x", "x"])
    with pytest.raises(KeyError, match="no individual outcome"):
        build_dataset({"fmri": a}, y_individual={("s1", "x"): 1})


def test_build_dataset_attaches_both_outcome_levels():
    a = _block("fmri", ["s1", "s2"], ["x", "y"])
    b = _block("behavior", ["s1", "s2"], ["x", "y"])
    dataset = build_dataset(
        {"fmri": a, "behavior": b},
        y_individual={("s1", "x"): 1, ("s2", "y"): 0},
        y_aggregate={"x": 0.8, "y": 0.2},
    )
    assert dataset.n_trials == 2
    assert dataset.n_stimuli == 2
    assert dataset.y_aggregate == {"x": 0.8, "y": 0.2}


def test_dataset_rejects_row_misaligned_blocks():
    from behavioral_decoding.io.base import MultimodalDataset

    a = _block("fmri", ["s1", "s2"], ["x", "y"])
    b = _block("eeg", ["s2", "s1"], ["y", "x"])  # same trials, wrong order
    with pytest.raises(ValueError, match="not row-aligned"):
        MultimodalDataset(
            blocks={"fmri": a, "eeg": b},
            y_individual=np.array([1, 0]),
            subject_ids=a.subject_ids,
            stimulus_ids=a.stimulus_ids,
        )


def test_alignment_preview_reports_the_cost():
    a = _block("fmri", ["s1", "s2", "s3"], ["x", "x", "x"])
    b = _block("eeg", ["s1", "s2"], ["x", "x"])
    text = summarise_alignment({"fmri": a, "eeg": b})
    assert "2 shared trials" in text
    assert "fmri" in text and "eeg" in text
