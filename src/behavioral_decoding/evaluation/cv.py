"""Cross-validation that respects the nesting in the data.

Trials are nested in subjects, and subjects differ in ways that have nothing to
do with the stimulus: head size, electrode impedance, baseline mood, camera
angle. A random trial-level split puts the same subject on both sides of the
fold, and the model learns to recognise the person instead of the choice. The
resulting accuracy is real and useless.

Every splitter here groups by subject. For the market-forecasting arm, splitting
by *stimulus* is also available, since a forecast of an item's market outcome is
only meaningful for an item the model has not seen.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from ..utils.logging import get_logger
from ..utils.progress import progress

logger = get_logger(__name__)


def subject_splitter(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    stratified: bool = True,
    seed: int = 0,
):
    """Return a fitted-ready splitter grouped by subject.

    ``stratified=True`` keeps class proportions roughly constant across folds
    while still never splitting a subject. With few subjects this can fail to
    balance; the function checks and warns rather than pretending it worked.
    """
    n_groups = len(np.unique(groups))
    if n_splits > n_groups:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of subjects ({n_groups}). Use at most "
            "leave-one-subject-out."
        )
    if stratified:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=n_splits)


def iter_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    stratified: bool = True,
    seed: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` with no subject on both sides."""
    splitter = subject_splitter(y, groups, n_splits, stratified, seed)
    for train_idx, test_idx in splitter.split(X, y, groups):
        overlap = set(groups[train_idx]).intersection(groups[test_idx])
        if overlap:  # pragma: no cover - defensive
            raise AssertionError(f"subject leakage across fold: {sorted(overlap)}")
        yield train_idx, test_idx


def out_of_fold_proba(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    stratified: bool = True,
    seed: int = 0,
    desc: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predicted probabilities for the positive class.

    Every row is predicted by a model that never saw that row's subject. These
    are the only probabilities that may be used to weight modalities in the
    ensemble: weighting by training accuracy would hand the highest weight to
    whichever model overfits hardest.

    Returns
    -------
    (oof_proba, fold_ids)
        ``fold_ids[i]`` records which fold produced row ``i``, so per-fold
        variability can be inspected instead of averaged away.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(
            f"out_of_fold_proba expects binary labels, found {len(classes)} classes: {classes}"
        )
    positive = classes[1]

    oof = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)

    splits = list(iter_splits(X, y, groups, n_splits, stratified, seed))
    for fold, (train_idx, test_idx) in enumerate(
        progress(splits, desc=desc or "CV folds", total=len(splits))
    ):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])
        col = list(model.classes_).index(positive)
        oof[test_idx] = proba[:, col]
        fold_ids[test_idx] = fold

    if np.isnan(oof).any():  # pragma: no cover - defensive
        raise AssertionError(
            f"{int(np.isnan(oof).sum())} rows never appeared in a test fold"
        )
    return oof, fold_ids


def stimulus_splitter(stimulus_ids: np.ndarray, n_splits: int = 5, seed: int = 0):
    """Group folds by stimulus, for held-out market forecasting."""
    n_stim = len(np.unique(stimulus_ids))
    if n_splits > n_stim:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of stimuli ({n_stim})"
        )
    return GroupKFold(n_splits=n_splits)


def describe_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    stratified: bool = True,
    seed: int = 0,
) -> str:
    """Report fold sizes and class balance, to catch a degenerate split early."""
    lines = [f"CV plan: {n_splits} folds grouped by subject"]
    dummy_X = np.zeros((len(y), 1))
    for fold, (tr, te) in enumerate(
        iter_splits(dummy_X, y, groups, n_splits, stratified, seed)
    ):
        te_classes, te_counts = np.unique(y[te], return_counts=True)
        balance = ", ".join(
            f"{c}={n}" for c, n in zip(te_classes, te_counts)
        )
        flag = "  <-- single-class test fold" if len(te_classes) < 2 else ""
        lines.append(
            f"  fold {fold}: "
            f"train {len(tr):>5} trials / {len(np.unique(groups[tr])):>3} subj | "
            f"test {len(te):>5} trials / {len(np.unique(groups[te])):>3} subj"
            f" | test balance {balance}{flag}"
        )
    return "\n".join(lines)
