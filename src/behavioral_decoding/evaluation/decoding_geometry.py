"""Cross-condition decoding geometry: generalization, variance, alignment.

Three additive evaluation utilities for asking whether a decoder has learned a
condition-invariant code, as opposed to a per-condition shortcut that happens
to score well within one condition. Grounded in:

    Li H, Chrysanthidis N, Brincat SL, Rose J, Miller EK (2026). "Neural
    subspace reorganization reflects value-based decision-making." iScience
    29:117492. doi:10.1016/j.isci.2026.117492.

In that paper (macaque lateral prefrontal cortex), chosen- and unchosen-value
codes rotated into orthogonal subspaces after the decision, and the
chosen-option subspaces aligned across conditions, which let a decoder trained
on one condition (first-chosen option) generalize to another (second-chosen
option). The reported decoding gain traced to an increase in between-class
variance, not a decrease in within-class noise.

This module gives three matching tools:

1. :func:`cross_condition_generalization` -- train on condition A, test on
   condition B (and vice versa), against a within-condition CV baseline, with
   a label-permutation null.
2. :func:`variance_decomposition` -- between-class vs within-class variance
   along the LDA axis, to attribute an accuracy change to separation vs noise.
3. :func:`subspace_alignment` -- mean Pearson correlation of per-class coding
   vectors between two conditions, plus principal angles between the
   class-mean subspaces.

All splits are group-aware (subject-grouped) whenever ``groups`` is supplied,
matching the leakage discipline in :mod:`.cv`. These are additive utilities;
nothing here is imported by an existing pipeline or changes an existing
result.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, StratifiedKFold


def _make_splitter(y: np.ndarray, groups: Optional[np.ndarray], n_splits: int, seed: int):
    if groups is not None:
        n_groups = len(np.unique(groups))
        n_splits = min(n_splits, n_groups)
        return GroupKFold(n_splits=n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _within_condition_accuracy(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    n_splits: int,
    seed: int,
) -> float:
    splitter = _make_splitter(y, groups, n_splits, seed)
    split_iter = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    accs = []
    for train_idx, test_idx in split_iter:
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        accs.append(float(model.score(X[test_idx], y[test_idx])))
    return float(np.mean(accs)) if accs else float("nan")


def cross_condition_generalization(
    estimator,
    X_a: np.ndarray,
    y_a: np.ndarray,
    X_b: np.ndarray,
    y_b: np.ndarray,
    groups_a: Optional[np.ndarray] = None,
    groups_b: Optional[np.ndarray] = None,
    n_splits: int = 5,
    n_perm: int = 500,
    seed: int = 0,
) -> Dict[str, float]:
    """Train on one condition, test on the other, against within-condition CV.

    Fits ``estimator`` on all of condition A and scores it on all of condition
    B (``a_to_b``), and the reverse (``b_to_a``), then compares each to a
    within-condition, group-aware cross-validated baseline computed
    separately in A and in B. A decoder reading a condition-invariant code
    should generalize close to its within-condition baseline; a decoder
    reading condition-specific nuisance structure drops toward chance when
    tested out of condition (Li et al. 2026, cross-condition decoding).

    A label-permutation null is run on the cross-condition score (shuffling
    the training labels within condition A / B before fitting) so the
    generalization score can be judged against chance rather than only against
    the within-condition number.

    Returns a dict with ``a_to_b``, ``b_to_a``, ``within_a``, ``within_b``,
    ``null_mean_a_to_b``, ``null_mean_b_to_a``, and permutation p-values
    ``p_value_a_to_b`` / ``p_value_b_to_a``.
    """
    X_a = np.asarray(X_a, dtype=float)
    X_b = np.asarray(X_b, dtype=float)
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)
    rng = np.random.default_rng(seed)

    model_ab = clone(estimator).fit(X_a, y_a)
    a_to_b = float(model_ab.score(X_b, y_b))
    model_ba = clone(estimator).fit(X_b, y_b)
    b_to_a = float(model_ba.score(X_a, y_a))

    within_a = _within_condition_accuracy(estimator, X_a, y_a, groups_a, n_splits, seed)
    within_b = _within_condition_accuracy(estimator, X_b, y_b, groups_b, n_splits, seed)

    def _null(X_train, y_train, X_test, y_test) -> np.ndarray:
        scores = np.empty(n_perm)
        for i in range(n_perm):
            permuted = rng.permutation(y_train)
            model = clone(estimator).fit(X_train, permuted)
            scores[i] = model.score(X_test, y_test)
        return scores

    null_ab = _null(X_a, y_a, X_b, y_b)
    null_ba = _null(X_b, y_b, X_a, y_a)
    p_ab = float((np.sum(null_ab >= a_to_b) + 1) / (n_perm + 1))
    p_ba = float((np.sum(null_ba >= b_to_a) + 1) / (n_perm + 1))

    return {
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "within_a": within_a,
        "within_b": within_b,
        "null_mean_a_to_b": float(null_ab.mean()),
        "null_mean_b_to_a": float(null_ba.mean()),
        "p_value_a_to_b": p_ab,
        "p_value_b_to_a": p_ba,
    }


def variance_decomposition(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Between-class vs within-class variance along the LDA discriminant axis.

    Fits a two-class LDA, projects the data onto its single discriminant axis,
    and reports the between-class variance (squared distance between
    projected class means, weighted by class size) and the mean within-class
    variance along that same axis, plus their ratio (an axis-aligned analogue
    of the F-ratio / eta-squared).

    Following Li et al. (2026), an accuracy change between two datasets (e.g.
    before vs after a manipulation, or two decision epochs) can come from
    increased separation between classes, from reduced noise within a class,
    or both; comparing ``between_class_variance`` and
    ``within_class_variance`` across the two datasets attributes the change
    to one, the other, or both, rather than leaving the accuracy number to
    speak for itself.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"variance_decomposition expects 2 classes, found {len(classes)}")

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    projected = lda.transform(X).ravel()

    grand_mean = projected.mean()
    between = 0.0
    within = 0.0
    total_n = len(y)
    for c in classes:
        mask = y == c
        n_c = mask.sum()
        class_mean = projected[mask].mean()
        between += n_c * (class_mean - grand_mean) ** 2
        within += np.sum((projected[mask] - class_mean) ** 2)
    between /= total_n
    within /= total_n

    ratio = float(between / within) if within > 0 else float("inf")
    return {
        "between_class_variance": float(between),
        "within_class_variance": float(within),
        "variance_ratio": ratio,
    }


def _class_coding_vectors(X: np.ndarray, y: np.ndarray) -> Dict[object, np.ndarray]:
    """Per-class mean vector minus the grand mean: one 'coding vector' per class."""
    grand_mean = X.mean(axis=0)
    vectors = {}
    for c in np.unique(y):
        vectors[c] = X[y == c].mean(axis=0) - grand_mean
    return vectors


def subspace_alignment(
    X_a: np.ndarray,
    y_a: np.ndarray,
    X_b: np.ndarray,
    y_b: np.ndarray,
) -> Dict[str, object]:
    """Alignment of per-class coding vectors and subspaces between two conditions.

    Coding vectors: for each class, the class-mean vector minus the grand mean,
    computed separately in condition A and condition B. Reports the mean
    Pearson correlation of matched-class coding vectors across conditions
    (``mean_coding_vector_correlation``), following the subspace-alignment
    analysis in Li et al. (2026): a value near 1 means the same features carry
    the class distinction in both conditions (aligned codes, and a decoder
    should transfer); a value near 0 means the codes are unrelated (rotated
    or orthogonal, and a decoder should not transfer).

    Subspace principal angles: stacks the per-class coding vectors from each
    condition into a matrix, orthonormalizes each (QR), and computes the
    principal angles (degrees) between the two subspaces via the singular
    values of the product of the orthonormal bases. Angles near 0 degrees
    indicate the subspaces coincide; angles near 90 degrees indicate they are
    orthogonal.

    Requires the same set of classes in both conditions.
    """
    X_a = np.asarray(X_a, dtype=float)
    X_b = np.asarray(X_b, dtype=float)
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)

    vecs_a = _class_coding_vectors(X_a, y_a)
    vecs_b = _class_coding_vectors(X_b, y_b)
    classes = sorted(set(vecs_a) & set(vecs_b))
    if len(classes) < 1:
        raise ValueError("no shared classes between condition A and condition B")

    correlations = []
    for c in classes:
        va, vb = vecs_a[c], vecs_b[c]
        if np.allclose(va, 0) or np.allclose(vb, 0):
            continue
        corr = float(np.corrcoef(va, vb)[0, 1])
        correlations.append(corr)
    mean_corr = float(np.mean(correlations)) if correlations else float("nan")

    mat_a = np.stack([vecs_a[c] for c in classes], axis=1)
    mat_b = np.stack([vecs_b[c] for c in classes], axis=1)
    # Class coding vectors sum to (approximately) zero across a balanced set
    # of classes, so the coding subspace spanned by C classes has rank at
    # most C - 1. QR on the full, rank-deficient stack hands back an extra
    # orthonormal column that is arbitrary noise direction, not signal;
    # truncating to the numerical rank keeps only the meaningful axes before
    # comparing subspaces.
    rank_a = max(1, np.linalg.matrix_rank(mat_a))
    rank_b = max(1, np.linalg.matrix_rank(mat_b))
    q_a, _ = np.linalg.qr(mat_a)
    q_b, _ = np.linalg.qr(mat_b)
    k = min(q_a.shape[1], q_b.shape[1], rank_a, rank_b)
    singular_values = np.linalg.svd(q_a[:, :k].T @ q_b[:, :k], compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    principal_angles_deg = np.degrees(np.arccos(singular_values)).tolist()

    return {
        "mean_coding_vector_correlation": mean_corr,
        "per_class_correlation": dict(zip([str(c) for c in classes], correlations)),
        "principal_angles_deg": [float(a) for a in principal_angles_deg],
    }
