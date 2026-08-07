"""fMRI loading and ROI reduction.

The default feature set is not whole-brain. It is a small set of spheres in the
anticipatory-affect circuit, because that is where the neuroforecasting result
actually lives: in Genevsky, Yoon and Knutson (2017), both NAcc and MPFC
predicted *individual* funding choices, but only NAcc generalised to forecast
*market* funding outcomes weeks later. Tong et al. (2020) found the same split
for video engagement, with NAcc up and anterior insula down at video onset.

Handing a whole-brain matrix to a classifier with 30 subjects is a good way to
fit noise. Start from the theory-specified ROIs, and treat whole-brain as the
exploratory arm rather than the default.

Coordinates below are approximate MNI centres for the standard anticipatory
affect ROIs. Confirm them against the specific paper you are replicating before
reporting anything: sphere placement is a real analytic degree of freedom.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.progress import progress
from .base import FMRI, BaseLoader, ModalityBlock

# name -> (x, y, z) MNI centre in mm
DEFAULT_ROIS: Dict[str, Tuple[float, float, float]] = {
    "NAcc_L": (-10.0, 12.0, -2.0),
    "NAcc_R": (10.0, 12.0, -2.0),
    "MPFC": (0.0, 46.0, 4.0),
    "AIns_L": (-32.0, 20.0, -6.0),
    "AIns_R": (32.0, 20.0, -6.0),
}

DEFAULT_SPHERE_RADIUS_MM = 6.0


class FMRILoader(BaseLoader):
    """Reduce 4-D BOLD runs to trial-by-ROI features.

    Two entry points:

    ``from_arrays``
        For already-extracted timeseries or beta estimates. No dependencies.
    ``load``
        For BIDS-style NIfTI files. Requires ``nilearn``.
    """

    name = FMRI

    def __init__(
        self,
        rois: Optional[Dict[str, Tuple[float, float, float]]] = None,
        radius_mm: float = DEFAULT_SPHERE_RADIUS_MM,
        standardize: bool = False,
    ) -> None:
        self.rois = dict(rois) if rois is not None else dict(DEFAULT_ROIS)
        self.radius_mm = radius_mm
        # Off by default and it should stay off: per-run standardisation applied
        # before the CV split is a leakage path. Scale inside the fold instead.
        self.standardize = standardize

    def from_arrays(
        self,
        X: np.ndarray,
        subject_ids: Sequence,
        stimulus_ids: Sequence,
        feature_names: Optional[List[str]] = None,
        source: str = "arrays",
    ) -> ModalityBlock:
        """Build a block from a precomputed ``(n_trials, n_rois)`` matrix."""
        return ModalityBlock(
            name=self.name,
            X=np.asarray(X, dtype=float),
            subject_ids=np.asarray(subject_ids),
            stimulus_ids=np.asarray(stimulus_ids),
            feature_names=feature_names or list(self.rois.keys()),
            provenance=self._provenance(
                source=source, rois=list(self.rois.keys()), radius_mm=self.radius_mm
            ),
        )

    def load(
        self,
        func_paths: Sequence[str],
        events: Sequence[object],
        subject_ids: Sequence,
        t_r: float,
        onset_shift_s: float = 4.0,
        window_s: float = 4.0,
        mask_img: Optional[object] = None,
    ) -> ModalityBlock:
        """Extract trial-wise ROI means from NIfTI runs.

        Parameters
        ----------
        func_paths:
            One preprocessed 4-D NIfTI per run.
        events:
            One events table per run (any object with ``onset`` and
            ``stimulus_id`` columns, e.g. a BIDS ``*_events.tsv`` DataFrame).
        subject_ids:
            One subject label per run.
        t_r:
            Repetition time in seconds.
        onset_shift_s:
            Haemodynamic lag applied before averaging. 4 s is a common default
            for a peak-window estimate; a proper GLM is the better option once
            the design is finalised, and this averaging shortcut should be
            replaced then.
        window_s:
            Length of the averaging window after the shift.

        Notes
        -----
        This is a deliberately simple peak-window extractor so the pipeline can
        run end to end. It is not a substitute for a first-level GLM with
        nuisance regressors. Swap in ``nilearn.glm.first_level`` before any
        result leaves the lab.
        """
        try:
            from nilearn.maskers import NiftiSpheresMasker
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "FMRILoader.load requires nilearn. Install with "
                "`pip install '.[fmri]'`, or use FMRILoader.from_arrays if you "
                "already have extracted timeseries."
            ) from exc

        seeds = list(self.rois.values())
        roi_names = list(self.rois.keys())
        masker = NiftiSpheresMasker(
            seeds=seeds,
            radius=self.radius_mm,
            mask_img=mask_img,
            standardize="zscore_sample" if self.standardize else False,
            t_r=t_r,
        )

        rows: List[np.ndarray] = []
        subj_out: List[object] = []
        stim_out: List[object] = []

        n_runs = len(func_paths)
        for run_idx, func_path in enumerate(
            progress(func_paths, desc="fMRI runs", total=n_runs)
        ):
            ts = masker.fit_transform(func_path)  # (n_volumes, n_rois)
            ev = events[run_idx]
            onsets = np.asarray(ev["onset"], dtype=float)
            stims = np.asarray(ev["stimulus_id"])
            for onset, stim in zip(onsets, stims):
                start = int(round((onset + onset_shift_s) / t_r))
                stop = int(round((onset + onset_shift_s + window_s) / t_r))
                start = max(0, min(start, ts.shape[0] - 1))
                stop = max(start + 1, min(stop, ts.shape[0]))
                rows.append(ts[start:stop].mean(axis=0))
                subj_out.append(subject_ids[run_idx])
                stim_out.append(stim)

        return ModalityBlock(
            name=self.name,
            X=np.vstack(rows),
            subject_ids=np.asarray(subj_out),
            stimulus_ids=np.asarray(stim_out),
            feature_names=roi_names,
            provenance=self._provenance(
                source="nifti",
                n_runs=n_runs,
                rois=roi_names,
                radius_mm=self.radius_mm,
                t_r=t_r,
                onset_shift_s=onset_shift_s,
                window_s=window_s,
                extraction="peak_window_mean",
                caveat="not a GLM; replace with first-level modelling before reporting",
            ),
        )
