"""NARPS loader tests.

The core (events parsing, keys, behaviour, aggregate) needs no neuroimaging
stack and runs everywhere. The BIDS ``.load`` path needs nilearn and nibabel, so
those tests importorskip and are skipped cleanly in a minimal CI environment
while still validating the real extraction locally.
"""

from __future__ import annotations

import io as _io

import numpy as np
import pytest

from behavioral_decoding.io.narps import (
    ACCEPT_RESPONSES,
    NARPS_TR,
    NARPSFormatError,
    NARPSLoader,
    acceptance_rate_by_gamble,
    expected_value,
    gamble_key,
    load_narps,
    load_participants,
    parse_events,
)

pd = pytest.importorskip("pandas")

from narps_fixture import synthetic_events, write_narps_fixture  # noqa: E402

REAL_EVENTS_TSV = (
    "onset\tduration\tgain\tloss\tRT\tparticipant_response\n"
    "4.071\t4\t14\t6\t2.388\tweakly_accept\n"
    "11.834\t4\t34\t14\t2.289\tstrongly_accept\n"
    "27.535\t4\t10\t10\t1.457\tweakly_reject\n"
    "36.435\t4\t12\t19\t1.973\tstrongly_reject\n"
    "43.935\t4\t20\t8\t0\tNoResp\n"
)


def _events():
    return pd.read_csv(_io.StringIO(REAL_EVENTS_TSV), sep="\t")


# ---------------------------------------------------------------- constants


def test_tr_is_one_second():
    assert NARPS_TR == 1.0


# --------------------------------------------------------------- events parse


def test_parse_events_binarises_response_correctly():
    parsed = parse_events(_events())
    # NoResp dropped, so 4 rows remain.
    assert len(parsed) == 4
    accepts = dict(zip(parsed["participant_response"], parsed["accept"]))
    assert accepts["weakly_accept"] == 1
    assert accepts["strongly_accept"] == 1
    assert accepts["weakly_reject"] == 0
    assert accepts["strongly_reject"] == 0


def test_noresp_is_dropped_not_imputed():
    """NoResp has RT=0; treating that as a fast response would poison RT."""
    parsed = parse_events(_events(), drop_no_response=True)
    assert "NoResp" not in set(parsed["participant_response"])
    assert (parsed["RT"] > 0).all()


def test_noresp_can_be_kept_when_asked():
    parsed = parse_events(_events(), drop_no_response=False)
    assert len(parsed) == 5


def test_confidence_preserves_the_strong_weak_distinction():
    parsed = parse_events(_events())
    conf = dict(zip(parsed["participant_response"], parsed["confidence"]))
    assert conf["strongly_accept"] == 1.0
    assert conf["weakly_accept"] == 0.5


def test_expected_value_and_keys():
    parsed = parse_events(_events())
    row = parsed[parsed["gain"] == 14].iloc[0]
    assert row["stimulus_id"] == "g14_l06"
    assert row["expected_value"] == pytest.approx(4.0)  # 0.5*(14-6)
    assert expected_value(34, 14) == pytest.approx(10.0)
    assert gamble_key(10, 10) == "g10_l10"


def test_missing_column_raises():
    bad = _events().drop(columns=["loss"])
    with pytest.raises(NARPSFormatError, match="missing columns"):
        parse_events(bad)


def test_unknown_response_string_raises():
    bad = _events().copy()
    bad.loc[0, "participant_response"] = "maybe"
    with pytest.raises(NARPSFormatError, match="unrecognised participant_response"):
        parse_events(bad)


# ---------------------------------------------------------------- behaviour


def test_behaviour_block_excludes_accept_and_uses_economic_features():
    parsed = parse_events(_events())
    block = NARPSLoader().behavior_block(parsed, "sub-001")
    assert "accept" not in block.feature_names
    assert set(block.feature_names) == {"gain", "loss", "expected_value", "abs_expected_value", "RT"}
    assert block.n_trials == 4


def test_behaviour_can_drop_rt_and_ev():
    parsed = parse_events(_events())
    block = NARPSLoader(include_rt=False, include_expected_value=False).behavior_block(
        parsed, "sub-001"
    )
    assert set(block.feature_names) == {"gain", "loss"}


def test_from_events_and_rois_row_alignment_enforced():
    parsed = parse_events(_events())
    with pytest.raises(ValueError, match="row-aligned"):
        NARPSLoader().from_events_and_rois(parsed, np.zeros((3, 5)), "sub-001")


def test_from_events_and_rois_builds_both_blocks():
    parsed = parse_events(_events())
    rois = np.random.default_rng(0).normal(size=(4, 5))
    blocks = NARPSLoader().from_events_and_rois(
        parsed, rois, "sub-001", roi_names=["NAcc_L", "NAcc_R", "MPFC", "AIns_L", "AIns_R"]
    )
    assert set(blocks) == {"fmri", "behavior"}
    assert blocks["fmri"].feature_names == ["NAcc_L", "NAcc_R", "MPFC", "AIns_L", "AIns_R"]
    assert np.array_equal(blocks["fmri"].stimulus_ids, blocks["behavior"].stimulus_ids)


# ---------------------------------------------------------------- aggregate


def test_acceptance_rate_by_gamble():
    # Two subjects, same two gambles, known acceptance pattern.
    a = pd.DataFrame(
        {
            "stimulus_id": ["g20_l06", "g06_l20"],
            "accept": [1, 0],
        }
    )
    b = pd.DataFrame(
        {
            "stimulus_id": ["g20_l06", "g06_l20"],
            "accept": [1, 1],
        }
    )
    rates, prov = acceptance_rate_by_gamble([a, b], min_subjects=1)
    assert rates["g20_l06"] == pytest.approx(1.0)  # both accepted
    assert rates["g06_l20"] == pytest.approx(0.5)  # one of two
    assert prov["outcome"] == "population_acceptance_rate"


def test_acceptance_rate_excludes_rare_gambles():
    a = pd.DataFrame({"stimulus_id": ["g20_l06", "g06_l20"], "accept": [1, 0]})
    rates, prov = acceptance_rate_by_gamble([a], min_subjects=5)
    assert rates == {}
    assert prov["n_gambles_dropped"] == 2


# ----------------------------------------------------- BIDS load (nilearn)


def test_load_discovers_runs_and_extracts_rois(tmp_path):
    pytest.importorskip("nilearn")
    pytest.importorskip("nibabel")
    write_narps_fixture(str(tmp_path), n_subjects=2, n_runs=2, trials_per_run=10, seed=0)

    blocks = NARPSLoader().load(str(tmp_path / "sub-001"))
    assert set(blocks) == {"fmri", "behavior"}
    # 2 runs x 10 trials.
    assert blocks["fmri"].n_trials == 20
    assert blocks["fmri"].feature_names == ["NAcc_L", "NAcc_R", "MPFC", "AIns_L", "AIns_R"]
    assert blocks["fmri"].provenance["confounds_regressed"] is True


def test_load_recovers_planted_nacc_ev_signal(tmp_path):
    """The fixture plants an EV-scaled bump at NAcc/MPFC; extraction must find it."""
    pytest.importorskip("nilearn")
    pytest.importorskip("nibabel")
    write_narps_fixture(str(tmp_path), n_subjects=2, n_runs=2, trials_per_run=12, seed=1)

    blocks = NARPSLoader().load(str(tmp_path / "sub-001"))
    fm, beh = blocks["fmri"], blocks["behavior"]
    ev = beh.X[:, beh.feature_names.index("expected_value")]
    nacc = fm.X[:, fm.feature_names.index("NAcc_L")]
    r = float(np.corrcoef(nacc, ev)[0, 1])
    assert r > 0.5, f"planted NAcc-EV signal not recovered (r={r:.3f})"


def test_load_raises_when_no_events(tmp_path):
    (tmp_path / "sub-999" / "func").mkdir(parents=True)
    with pytest.raises(NARPSFormatError, match="no .*events.tsv"):
        NARPSLoader().load(str(tmp_path / "sub-999"))


# ----------------------------------------------------------- participants


def test_participants_group_column(tmp_path):
    write_narps_fixture(str(tmp_path), n_subjects=4, n_runs=1, trials_per_run=8, seed=0)
    table = load_participants(str(tmp_path / "participants.tsv"))
    assert set(table["group"]) <= {"equalIndifference", "equalRange"}
    assert len(table) == 4


def test_synthetic_events_helper_is_parseable():
    events = synthetic_events(n=16, seed=0)
    parsed = parse_events(events)
    assert len(parsed) <= 16
    assert set(parsed["participant_response"]) <= set(
        ACCEPT_RESPONSES + ("strongly_reject", "weakly_reject")
    )


# --------------------------------------------------------- load_narps (nilearn)


def test_load_narps_builds_individual_dataset(tmp_path):
    pytest.importorskip("nilearn")
    pytest.importorskip("nibabel")
    write_narps_fixture(str(tmp_path), n_subjects=6, n_runs=2, trials_per_run=12, seed=0)

    ds = load_narps(str(tmp_path), group="equalRange")
    assert set(ds.modalities) == {"fmri", "behavior"}
    assert not ds.metadata["synthetic"]
    assert ds.metadata["dataset"] == "NARPS"
    assert ds.metadata["group"] == "equalRange"
    # Binary accept outcome.
    assert set(np.unique(ds.y_individual)) <= {0, 1}
    # Aggregate keyed by gamble.
    if ds.y_aggregate:
        assert all(k.startswith("g") and "_l" in k for k in ds.y_aggregate)


def test_load_narps_group_filter_selects_subjects(tmp_path):
    pytest.importorskip("nilearn")
    pytest.importorskip("nibabel")
    write_narps_fixture(str(tmp_path), n_subjects=6, n_runs=1, trials_per_run=10, seed=0)

    ei = load_narps(str(tmp_path), group="equalIndifference", with_aggregate=False)
    er = load_narps(str(tmp_path), group="equalRange", with_aggregate=False)
    # Fixture alternates groups by subject parity, so 3 and 3.
    assert ei.n_subjects == 3
    assert er.n_subjects == 3
    assert set(ei.subject_ids).isdisjoint(set(er.subject_ids))


def test_load_narps_recovers_reward_signal_individually(tmp_path):
    """The planted NAcc/MPFC-EV signal should let fMRI predict accept above chance.

    This is the NARPS positive control: the honest, recoverable claim. It does
    not assert brain beats behaviour, because on gambles it should not.
    """
    pytest.importorskip("nilearn")
    pytest.importorskip("nibabel")
    from behavioral_decoding.evaluation.neuroforecast import cross_validate_ensemble

    write_narps_fixture(str(tmp_path), n_subjects=8, n_runs=2, trials_per_run=16, seed=0)
    ds = load_narps(str(tmp_path), group="equalRange")
    result = cross_validate_ensemble(ds, n_splits_outer=3, n_splits_inner=2, seed=0)

    fmri = result["per_modality_pooled"]["fmri"]["balanced_accuracy"]
    assert fmri > 0.55, f"fMRI did not predict accept above chance (balacc={fmri:.3f})"
