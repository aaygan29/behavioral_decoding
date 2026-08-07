"""The data contract every modality must satisfy.

Two ideas do the structural work here.

1. **Trials are keyed twice.** Each row carries a ``subject_id`` and a
   ``stimulus_id``. Subject keys keep cross-validation honest (no subject
   appears in both train and test). Stimulus keys are what let us pool
   individual responses into a group-level forecast of a market outcome, which
   is the move that Genevsky and Knutson's neuroforecasting work rests on.

2. **Modalities are blocks, not columns.** fMRI, EEG, face, and behaviour are
   kept as separate matrices all the way through feature extraction, because the
   ensemble in :mod:`behavioral_decoding.models.ensemble` trains one model per
   block and reconciles them afterwards. Concatenating early would throw away
   the per-modality reliability estimates the reconciliation needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

Modality = str

FMRI: Modality = "fmri"
EEG: Modality = "eeg"
FACE: Modality = "face"
BEHAVIOR: Modality = "behavior"

KNOWN_MODALITIES: Sequence[Modality] = (FMRI, EEG, FACE, BEHAVIOR)


@dataclass
class ModalityBlock:
    """A trial-by-feature matrix for a single modality.

    Attributes
    ----------
    name:
        Modality identifier, e.g. ``"fmri"``.
    X:
        Array of shape ``(n_trials, n_features)``.
    subject_ids:
        Array of shape ``(n_trials,)``. Used as the CV grouping key.
    stimulus_ids:
        Array of shape ``(n_trials,)``. Used to aggregate to the market level.
    feature_names:
        Optional per-column labels, same length as ``X.shape[1]``.
    provenance:
        Free-form record of where the data came from and how it was reduced.
        Written into result files so a number can always be traced back.
    """

    name: Modality
    X: np.ndarray
    subject_ids: np.ndarray
    stimulus_ids: np.ndarray
    feature_names: Optional[List[str]] = None
    provenance: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        if self.X.ndim != 2:
            raise ValueError(
                f"{self.name}: X must be 2-D (n_trials, n_features), got shape {self.X.shape}"
            )
        self.subject_ids = np.asarray(self.subject_ids)
        self.stimulus_ids = np.asarray(self.stimulus_ids)
        n = self.X.shape[0]
        if len(self.subject_ids) != n or len(self.stimulus_ids) != n:
            raise ValueError(
                f"{self.name}: subject_ids ({len(self.subject_ids)}) and "
                f"stimulus_ids ({len(self.stimulus_ids)}) must both have length "
                f"n_trials ({n})"
            )
        if self.feature_names is None:
            self.feature_names = [f"{self.name}_{i:03d}" for i in range(self.X.shape[1])]
        elif len(self.feature_names) != self.X.shape[1]:
            raise ValueError(
                f"{self.name}: {len(self.feature_names)} feature names for "
                f"{self.X.shape[1]} columns"
            )

    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def select(self, index: np.ndarray) -> ModalityBlock:
        """Return a new block containing only the trials in ``index``."""
        return ModalityBlock(
            name=self.name,
            X=self.X[index],
            subject_ids=self.subject_ids[index],
            stimulus_ids=self.stimulus_ids[index],
            feature_names=list(self.feature_names or []),
            provenance=dict(self.provenance),
        )

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"ModalityBlock(name={self.name!r}, n_trials={self.n_trials}, "
            f"n_features={self.n_features}, "
            f"n_subjects={len(np.unique(self.subject_ids))})"
        )


@dataclass
class MultimodalDataset:
    """Aligned modality blocks plus the two levels of outcome.

    ``y_individual`` is the trial-level choice (did *this* person fund/buy/click
    *this* item). ``y_aggregate`` is the population outcome for each stimulus,
    measured outside the lab: funding success, view count, sales. The two are
    deliberately separate because the central empirical result in this
    literature is that they come apart. Brain signal that forecasts the
    aggregate is not always the signal that predicts the individual, and vice
    versa. See ``docs/literature.md``.
    """

    blocks: Dict[Modality, ModalityBlock]
    y_individual: np.ndarray
    subject_ids: np.ndarray
    stimulus_ids: np.ndarray
    y_aggregate: Optional[Dict[object, float]] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("MultimodalDataset requires at least one modality block")
        self.y_individual = np.asarray(self.y_individual)
        self.subject_ids = np.asarray(self.subject_ids)
        self.stimulus_ids = np.asarray(self.stimulus_ids)
        n = len(self.y_individual)
        for name, block in self.blocks.items():
            if block.n_trials != n:
                raise ValueError(
                    f"modality {name!r} has {block.n_trials} trials but y_individual has {n}"
                )
            if not np.array_equal(block.subject_ids, self.subject_ids):
                raise ValueError(
                    f"modality {name!r} subject_ids are not row-aligned with the dataset; "
                    "run features.align.align_blocks first"
                )
            if not np.array_equal(block.stimulus_ids, self.stimulus_ids):
                raise ValueError(
                    f"modality {name!r} stimulus_ids are not row-aligned with the dataset; "
                    "run features.align.align_blocks first"
                )

    @property
    def modalities(self) -> List[Modality]:
        return list(self.blocks.keys())

    @property
    def n_trials(self) -> int:
        return len(self.y_individual)

    @property
    def n_subjects(self) -> int:
        return len(np.unique(self.subject_ids))

    @property
    def n_stimuli(self) -> int:
        return len(np.unique(self.stimulus_ids))

    def class_balance(self) -> Dict[object, float]:
        """Fraction of trials in each class of ``y_individual``."""
        values, counts = np.unique(self.y_individual, return_counts=True)
        return {v: float(c) / len(self.y_individual) for v, c in zip(values, counts)}

    def imbalance_ratio(self) -> float:
        """Majority-to-minority ratio. Above ~3 is where resampling starts to matter."""
        _, counts = np.unique(self.y_individual, return_counts=True)
        return float(counts.max()) / float(counts.min())

    def describe(self) -> str:
        lines = [
            "MultimodalDataset",
            f"  trials     : {self.n_trials}",
            f"  subjects   : {self.n_subjects}",
            f"  stimuli    : {self.n_stimuli}",
            "  modalities : {}".format(", ".join(
                f"{k} ({v.n_features}d)" for k, v in self.blocks.items()
            )),
            "  balance    : {}".format(
                ", ".join(f"{k}={v:.1%}" for k, v in self.class_balance().items())
            ),
            f"  imbalance  : {self.imbalance_ratio():.2f}:1",
            "  aggregate  : {}".format(
                f"{len(self.y_aggregate)} stimuli with market outcomes"
                if self.y_aggregate
                else "none"
            ),
        ]
        return "\n".join(lines)


class BaseLoader:
    """Interface for modality loaders.

    A loader's only job is to turn raw files into a :class:`ModalityBlock`. It
    must not filter trials, impute, or standardise: those steps belong inside
    the cross-validation fold so they cannot leak test information into
    training. See ``docs/design.md``.
    """

    name: Modality = "base"

    def load(self, *args: object, **kwargs: object) -> ModalityBlock:
        raise NotImplementedError

    def _provenance(self, **kwargs: object) -> Dict[str, object]:
        record: Dict[str, object] = {"loader": type(self).__name__, "modality": self.name}
        record.update(kwargs)
        return record
