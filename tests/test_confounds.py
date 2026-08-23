"""Robust fMRIPrep confound selection and cleaning.

Real ``*_desc-confounds_timeseries.tsv`` files carry NaNs (leading-row
derivatives, framewise displacement), non-numeric columns, and a hundred-plus
regressors most analyses should not use. These tests pin the behaviour that
turns such a table into a finite, explicitly-chosen numeric matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from behavioral_decoding.io.confounds import (  # noqa: E402
    ConfoundError,
    select_confounds,
)


def _fmriprep_like(n=40, seed=0):
    """A confounds frame shaped like fMRIPrep output, warts included."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "trans_x": rng.normal(0, 0.1, n),
            "trans_y": rng.normal(0, 0.1, n),
            "trans_z": rng.normal(0, 0.1, n),
            "rot_x": rng.normal(0, 0.01, n),
            "rot_y": rng.normal(0, 0.01, n),
            "rot_z": rng.normal(0, 0.01, n),
            "csf": rng.normal(0, 1, n),
            "white_matter": rng.normal(0, 1, n),
            "global_signal": rng.normal(0, 1, n),
            "framewise_displacement": rng.normal(0.2, 0.05, n),
            "cosine00": rng.normal(0, 1, n),
            "a_comp_cor_00": rng.normal(0, 1, n),
        }
    )
    # fMRIPrep leaves NaN on the first row of derivative-like columns.
    for col in ("trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
                "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1"):
        vals = rng.normal(0, 0.05, n)
        vals[0] = np.nan
        frame[col] = vals
    frame.loc[0, "framewise_displacement"] = np.nan
    # A genuinely non-numeric column that must be dropped, not crash.
    frame["motion_outlier_label"] = ["none"] * n
    return frame


def test_default_strategy_selects_named_regressors_and_is_finite():
    frame = _fmriprep_like()
    matrix, prov = select_confounds(frame)  # default motion12+physio
    assert matrix.shape[0] == len(frame)
    # 6 motion + 6 derivatives + csf + white_matter = 14
    assert matrix.shape[1] == 14
    assert np.isfinite(matrix).all()
    assert prov["strategy"] == "motion12+physio"
    assert "global_signal" not in prov["kept"]


def test_nan_cells_are_filled_and_counted():
    frame = _fmriprep_like()
    matrix, prov = select_confounds(frame, fill="mean")
    assert np.isfinite(matrix).all()
    # 6 derivative columns each have one leading NaN; framewise_displacement is
    # not in the default set, so exactly 6 fills.
    assert prov["n_nan_filled"] == 6


def test_nonnumeric_column_is_dropped_when_requested_explicitly():
    frame = _fmriprep_like()
    matrix, prov = select_confounds(
        frame, columns=["trans_x", "motion_outlier_label", "csf"]
    )
    assert matrix.shape[1] == 2
    assert "motion_outlier_label" in prov["dropped_nonnumeric"]


def test_missing_columns_are_reported_not_fatal():
    frame = _fmriprep_like().drop(columns=["csf"])
    matrix, prov = select_confounds(frame, strategy="motion6+physio")
    # csf gone, white_matter kept -> 6 motion + white_matter = 7
    assert matrix.shape[1] == 7
    assert "csf" in prov["requested_missing"]


def test_all_requested_missing_raises():
    frame = pd.DataFrame({"unrelated": [1.0, 2.0, 3.0]})
    with pytest.raises(ConfoundError):
        select_confounds(frame, columns=["trans_x", "trans_y"])


def test_compcor_and_prefix_families():
    frame = _fmriprep_like()
    matrix, prov = select_confounds(
        frame, strategy="motion6", n_compcor=1, extra_prefixes=("cosine",)
    )
    assert "a_comp_cor_00" in prov["kept"]
    assert "cosine00" in prov["kept"]


def test_array_input_is_trusted_but_still_made_finite():
    arr = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    matrix, prov = select_confounds(arr, fill="mean")
    assert np.isfinite(matrix).all()
    assert prov["source"] == "array"
    assert matrix[0, 1] == pytest.approx(5.0)  # mean of 4 and 6
