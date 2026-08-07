"""Aligning modality blocks onto a shared trial index.

Modalities arrive with different trial counts. fMRI loses volumes to motion
scrubbing, EEG loses epochs to artefact rejection, the camera drops frames, and
a subject skips a rating. Silently zipping these together off-by-one is the kind
of bug that produces a plausible, publishable, wrong number, so alignment here
is explicit and by key: the ``(subject_id, stimulus_id)`` pair.

The default policy is an inner join across all modalities. That is lossy and it
is meant to be. If you want to keep partial trials, use ``how="outer"`` with an
imputer that is fitted inside the CV fold, never before it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..io.base import ModalityBlock, MultimodalDataset
from ..utils.logging import get_logger

logger = get_logger(__name__)

TrialKey = Tuple[object, object]


def _key_index(block: ModalityBlock) -> Dict[TrialKey, int]:
    """Map ``(subject, stimulus)`` to row index, rejecting duplicates."""
    index: Dict[TrialKey, int] = {}
    for i, (s, st) in enumerate(zip(block.subject_ids, block.stimulus_ids)):
        key = (s, st)
        if key in index:
            raise ValueError(
                f"modality {block.name!r} has duplicate trial key {key!r}. Alignment needs one row "
                "per (subject, stimulus); collapse repeats first or add a run index "
                "to the stimulus id."
            )
        index[key] = i
    return index


def align_blocks(
    blocks: Dict[str, ModalityBlock],
    how: str = "inner",
) -> Tuple[Dict[str, ModalityBlock], np.ndarray, np.ndarray]:
    """Reindex every block onto a common ordered set of trial keys.

    Returns
    -------
    (aligned_blocks, subject_ids, stimulus_ids)
    """
    if how not in {"inner", "outer"}:
        raise ValueError("how must be 'inner' or 'outer'")
    if not blocks:
        raise ValueError("no blocks to align")

    indices = {name: _key_index(block) for name, block in blocks.items()}
    key_sets = [set(idx) for idx in indices.values()]
    if how == "inner":
        keys = set.intersection(*key_sets)
    else:
        keys = set.union(*key_sets)

    if not keys:
        raise ValueError(
            f"no trial keys are shared across modalities {sorted(blocks)}. Check that subject and "
            "stimulus identifiers use the same format in every loader."
        )

    ordered: List[TrialKey] = sorted(keys, key=lambda k: (str(k[0]), str(k[1])))

    for name, block in blocks.items():
        dropped = block.n_trials - len(set(indices[name]).intersection(keys))
        if dropped:
            logger.info(
                "align: %s dropped %d/%d trials (%.1f%%) not shared with other modalities",
                name,
                dropped,
                block.n_trials,
                100.0 * dropped / block.n_trials,
            )

    aligned: Dict[str, ModalityBlock] = {}
    subject_ids = np.array([k[0] for k in ordered])
    stimulus_ids = np.array([k[1] for k in ordered])

    for name, block in blocks.items():
        idx = indices[name]
        X = np.full((len(ordered), block.n_features), np.nan)
        for row, key in enumerate(ordered):
            if key in idx:
                X[row] = block.X[idx[key]]
        n_missing = int(np.isnan(X).any(axis=1).sum())
        if how == "inner" and n_missing:  # pragma: no cover - defensive
            raise AssertionError(f"inner join left {n_missing} missing rows in {name}")
        provenance = dict(block.provenance)
        provenance.update({"aligned_how": how, "n_missing_rows": n_missing})
        aligned[name] = ModalityBlock(
            name=block.name,
            X=X,
            subject_ids=subject_ids,
            stimulus_ids=stimulus_ids,
            feature_names=list(block.feature_names or []),
            provenance=provenance,
        )

    return aligned, subject_ids, stimulus_ids


def build_dataset(
    blocks: Dict[str, ModalityBlock],
    y_individual: Dict[TrialKey, float],
    y_aggregate: Optional[Dict[object, float]] = None,
    how: str = "inner",
    metadata: Optional[Dict[str, object]] = None,
) -> MultimodalDataset:
    """Align blocks and attach outcomes keyed by ``(subject, stimulus)``.

    ``y_aggregate`` is keyed by stimulus alone, since a market outcome is a
    property of the item, not of any one viewer.
    """
    aligned, subject_ids, stimulus_ids = align_blocks(blocks, how=how)

    missing = [
        (s, st) for s, st in zip(subject_ids, stimulus_ids) if (s, st) not in y_individual
    ]
    if missing:
        raise KeyError(
            f"{len(missing)} aligned trials have no individual outcome, e.g. {missing[:3]}"
        )

    y = np.array([y_individual[(s, st)] for s, st in zip(subject_ids, stimulus_ids)])

    if y_aggregate is not None:
        unknown = set(np.unique(stimulus_ids)) - set(y_aggregate)
        if unknown:
            logger.info(
                "align: %d/%d stimuli have no aggregate outcome and will be excluded "
                "from market forecasting",
                len(unknown),
                len(np.unique(stimulus_ids)),
            )

    return MultimodalDataset(
        blocks=aligned,
        y_individual=y,
        subject_ids=subject_ids,
        stimulus_ids=stimulus_ids,
        y_aggregate=y_aggregate,
        metadata=metadata or {},
    )


def summarise_alignment(blocks: Dict[str, ModalityBlock]) -> str:
    """Human-readable report of what an inner join would cost, before running it."""
    indices = {name: set(_key_index(b)) for name, b in blocks.items()}
    shared = set.intersection(*indices.values()) if indices else set()
    lines = [f"alignment preview ({len(shared)} shared trials)"]
    for name, keys in indices.items():
        lines.append(
            f"  {name:<10} {len(keys):>6} trials  ->  {len(shared):>6} kept  "
            f"({100.0 * (len(keys) - len(shared)) / max(1, len(keys)):>5.1f}% dropped)"
        )
    return "\n".join(lines)
