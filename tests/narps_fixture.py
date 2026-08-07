"""Write a tiny NARPS-format BIDS tree to disk, for tests and the demo.

Produces the real BIDS layout the loader discovers: ``participants.tsv`` with a
``group`` column, and per subject/run ``*_task-MGT_*_events.tsv`` (with the
verified NARPS columns and response strings), a small 4-D preprocessed BOLD
NIfTI, and a confounds table.

The dimensions are miniature, not NARPS-real: a handful of trials per run and a
13x13x13 volume, sized so the whole fixture is a few MB and the sphere extraction
runs in a second. The BIDS *structure* and file *format* are real, which is the
point: the loader is exercised against the layout it will meet, not a mock.

Signal is planted so the pipeline can be checked. NAcc and MPFC voxels carry a
BOLD bump proportional to each trial's expected value, and the accept choice
follows expected value plus noise. So fMRI ROI features should predict
accept/reject above chance. Consistent with ``docs/narps.md``, no
brain-beats-behaviour claim is planted: on gambles the economic variables
predict choice at least as well, and that is the honest story.

Everything here is simulated. No number from it is a finding about brains.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from behavioral_decoding.io.fmri import DEFAULT_ROIS
from behavioral_decoding.io.narps import NARPS_TR, gamble_key

# 8 mm isotropic voxels, origin placing MNI (0,0,0) at voxel 6. A 13-voxel span
# then covers roughly MNI -48..56 mm on each axis, which contains all five
# default ROIs (MPFC at y=46 is the tight one; it lands at voxel ~11.75).
VOXEL_SIZE_MM = 8.0
VOLUME_SHAPE = (13, 13, 13)
_AFFINE = np.array(
    [
        [VOXEL_SIZE_MM, 0.0, 0.0, -48.0],
        [0.0, VOXEL_SIZE_MM, 0.0, -48.0],
        [0.0, 0.0, VOXEL_SIZE_MM, -48.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# A gamble grid drawn from the equalRange condition (gains and losses 5-20).
_GAINS = (8, 12, 16, 20)
_LOSSES = (6, 10, 14, 18)


def _mni_to_voxel(x: float, y: float, z: float) -> tuple:
    inv = np.linalg.inv(_AFFINE)
    vox = inv @ np.array([x, y, z, 1.0])
    return tuple(int(round(v)) for v in vox[:3])


def write_narps_fixture(
    root: str,
    n_subjects: int = 6,
    n_runs: int = 2,
    trials_per_run: int = 12,
    trial_spacing_s: float = 7.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Write a synthetic NARPS BIDS tree under ``root``. Returns ground truth."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    nacc_voxels = [_mni_to_voxel(*DEFAULT_ROIS[r]) for r in ("NAcc_L", "NAcc_R")]
    mpfc_voxel = _mni_to_voxel(*DEFAULT_ROIS["MPFC"])
    signal_voxels = nacc_voxels + [mpfc_voxel]

    onsets = 4.0 + trial_spacing_s * np.arange(trials_per_run)
    n_volumes = int(np.ceil((onsets[-1] + 12.0) / NARPS_TR))

    participants: List[Dict[str, Any]] = []
    accept_by_gamble: Dict[str, List[int]] = {}

    for s in range(1, n_subjects + 1):
        subject_id = f"sub-{s:03d}"
        group = "equalRange" if s % 2 == 0 else "equalIndifference"
        participants.append(
            {"participant_id": subject_id, "group": group, "gender": "n/a", "age": "n/a"}
        )

        func_dir = root_path / subject_id / "func"
        func_dir.mkdir(parents=True, exist_ok=True)
        subject_bias = rng.normal(0.0, 0.4)

        for run in range(1, n_runs + 1):
            srng = np.random.default_rng(seed * 10_000 + s * 100 + run)
            gains = srng.choice(_GAINS, size=trials_per_run)
            losses = srng.choice(_LOSSES, size=trials_per_run)
            evs = 0.5 * (gains.astype(float) - losses.astype(float))

            # BOLD volume: noise everywhere, plus an EV-scaled bump at the reward
            # ROIs over each trial's peak window.
            bold = srng.normal(0.0, 1.0, size=VOLUME_SHAPE + (n_volumes,)).astype(np.float32)
            for onset, ev in zip(onsets, evs):
                lo = int(round((onset + 4.0) / NARPS_TR))
                hi = int(round((onset + 8.0) / NARPS_TR))
                lo = max(0, min(lo, n_volumes - 1))
                hi = max(lo + 1, min(hi, n_volumes))
                for vx, vy, vz in signal_voxels:
                    bold[vx, vy, vz, lo:hi] += 1.5 * ev

            _write_nifti(
                func_dir
                / f"{subject_id}_task-MGT_run-{run:02d}"
                f"_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                bold,
            )

            # Choice follows EV plus subject bias plus noise, calibrated to a
            # spread of accept rates rather than all-accept or all-reject.
            logits = 0.5 * evs + subject_bias + srng.normal(0.0, 1.0, size=trials_per_run)
            accepts = (logits > 0).astype(int)
            responses = [
                _response_string(a, srng.random()) for a in accepts
            ]
            rts = np.clip(srng.normal(1.8, 0.4, size=trials_per_run), 0.3, 3.5)

            events_rows = []
            for onset, gain, loss, resp, rt in zip(onsets, gains, losses, responses, rts):
                events_rows.append(
                    {
                        "onset": round(float(onset), 3),
                        "duration": 4,
                        "gain": int(gain),
                        "loss": int(loss),
                        "RT": round(float(rt), 3),
                        "participant_response": resp,
                    }
                )
                key = gamble_key(gain, loss)
                accept_by_gamble.setdefault(key, []).append(int(resp in ("strongly_accept", "weakly_accept")))

            _write_tsv(
                func_dir / f"{subject_id}_task-MGT_run-{run:02d}_events.tsv",
                events_rows,
                ["onset", "duration", "gain", "loss", "RT", "participant_response"],
            )

            # A minimal confounds table: six motion parameters, no NaNs, so the
            # confound-regression path is exercised without special-casing.
            _write_confounds(
                func_dir
                / f"{subject_id}_task-MGT_run-{run:02d}_desc-confounds_timeseries.tsv",
                n_volumes,
                srng,
            )

    _write_tsv(
        root_path / "participants.tsv",
        participants,
        ["participant_id", "group", "gender", "age"],
    )

    return {
        "root": str(root_path),
        "n_subjects": n_subjects,
        "n_runs": n_runs,
        "trials_per_run": trials_per_run,
        "volume_shape": VOLUME_SHAPE,
        "n_volumes": n_volumes,
        "expected": (
            "fMRI ROI features should predict accept/reject above chance out of "
            "fold; no brain-beats-behaviour claim is planted (gambles favour the "
            "economic baseline)."
        ),
    }


def _response_string(accept: int, u: float) -> str:
    if accept:
        return "strongly_accept" if u < 0.5 else "weakly_accept"
    return "strongly_reject" if u < 0.5 else "weakly_reject"


def _write_nifti(path: Path, data: np.ndarray) -> None:
    import nibabel as nib

    img = nib.Nifti1Image(data, _AFFINE)
    img.header.set_zooms(tuple([VOXEL_SIZE_MM] * 3) + (NARPS_TR,))
    nib.save(img, str(path))


def _write_tsv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_confounds(path: Path, n_volumes: int, rng: np.random.Generator) -> None:
    cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    data = rng.normal(0.0, 0.05, size=(n_volumes, len(cols)))
    rows = [{c: round(float(v), 6) for c, v in zip(cols, row)} for row in data]
    _write_tsv(path, rows, cols)


def synthetic_events(n: int = 20, seed: int = 0) -> Optional[Any]:
    """A parsed-events DataFrame with planted structure, no files written.

    Used by tests that exercise the nilearn-free layer without touching disk.
    Returns None if pandas is unavailable.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    rng = np.random.default_rng(seed)
    gains = rng.choice(_GAINS, size=n)
    losses = rng.choice(_LOSSES, size=n)
    evs = 0.5 * (gains.astype(float) - losses.astype(float))
    accepts = (0.5 * evs + rng.normal(0, 1, size=n) > 0).astype(int)
    responses = [_response_string(a, rng.random()) for a in accepts]
    return pd.DataFrame(
        {
            "onset": 4.0 + 7.0 * np.arange(n),
            "duration": 4,
            "gain": gains,
            "loss": losses,
            "RT": np.clip(rng.normal(1.8, 0.4, size=n), 0.3, 3.5),
            "participant_response": responses,
        }
    )
