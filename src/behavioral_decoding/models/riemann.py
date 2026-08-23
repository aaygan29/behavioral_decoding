"""Riemannian tangent-space mapping for EEG covariance features.

Why this exists, in one paragraph: EEG carries a lot of its discriminative
signal in the *covariance* between channels, and the set of covariance matrices
is a curved manifold (symmetric positive-definite matrices), not a flat vector
space. Treating the entries of a covariance matrix as independent numbers, then
z-scoring and SMOTE-interpolating them, throws away that geometry and can even
produce non-covariance "matrices". The fix that the EEG literature settled on is
to project each covariance to a **tangent space** (a flat space that locally
approximates the manifold) and classify there. In the tangent space the vectors
are genuinely Euclidean, so the framework's scaler, SMOTE, and logistic
regression are all valid again.

This module implements the **log-Euclidean** tangent mapping, which is
closed-form, robust, and needs only scipy. It is a legitimate Riemannian metric
on SPD matrices. The affine-invariant metric that ``pyriemann`` uses is a
further refinement (it iterates to a geometric mean and whitens by it); this
gets most of the benefit without the dependency or the iteration. See
``docs/estimators.md``.

The transformer is designed to sit at the front of the standard pipeline:

    Pipeline(RiemannianTangentSpace -> StandardScaler -> SMOTE -> logistic)

so everything downstream of the tangent map operates on flat Euclidean vectors,
and the map itself is fitted on training data only (inside each CV fold and each
bag), which keeps it leakage-safe.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.logging import get_logger

logger = get_logger(__name__)


def n_channels_from_flat(n_features: int) -> int:
    """Recover k from the length of a flattened k-by-k upper triangle.

    A symmetric k-by-k matrix has ``k(k+1)/2`` unique entries (diagonal
    included). Inverting: ``k = (-1 + sqrt(1 + 8n)) / 2``. Raises if ``n`` is not
    a triangular number, which means the features are not a flattened covariance.
    """
    k = (-1 + np.sqrt(1 + 8 * n_features)) / 2
    k_int = int(round(k))
    if k_int * (k_int + 1) // 2 != n_features:
        raise ValueError(
            f"{n_features} features is not k(k+1)/2 for any integer k, so these are not "
            "flattened covariance upper-triangles. RiemannianTangentSpace needs "
            "covariance features (see EEGLoader covariance mode)."
        )
    return k_int


def _upper_triangle_indices(k: int):
    return np.triu_indices(k)


def flatten_spd(matrices: np.ndarray) -> np.ndarray:
    """Flatten ``(n, k, k)`` symmetric matrices to ``(n, k(k+1)/2)`` rows.

    Off-diagonal entries are scaled by ``sqrt(2)`` so that the Euclidean norm of
    the flattened vector equals the Frobenius norm of the matrix. This is the
    standard isometric vectorisation; it matters because the downstream scaler
    and classifier work in Euclidean distance.
    """
    matrices = np.asarray(matrices, dtype=float)
    n, k, _ = matrices.shape
    rows, cols = _upper_triangle_indices(k)
    scale = np.where(rows == cols, 1.0, np.sqrt(2.0))
    return matrices[:, rows, cols] * scale


def unflatten_spd(rows: np.ndarray, k: int) -> np.ndarray:
    """Inverse of :func:`flatten_spd`: ``(n, k(k+1)/2)`` rows to ``(n, k, k)``."""
    rows = np.asarray(rows, dtype=float)
    n = rows.shape[0]
    ri, ci = _upper_triangle_indices(k)
    scale = np.where(ri == ci, 1.0, np.sqrt(2.0))
    out = np.zeros((n, k, k), dtype=float)
    vals = rows / scale
    out[:, ri, ci] = vals
    out[:, ci, ri] = vals
    return out


def _regularise(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """Shrink a covariance toward a scaled identity so it is positive-definite.

    Covariances estimated from short epochs are often rank-deficient, and the
    matrix logarithm is only defined for positive-definite input. Shrinkage
    ``(1 - s) C + s * (tr C / k) I`` guarantees positive-definiteness for any
    ``s > 0`` while barely moving a well-conditioned matrix.
    """
    k = cov.shape[0]
    mu = np.trace(cov) / k
    return (1.0 - shrinkage) * cov + shrinkage * mu * np.eye(k)


def _symmetric_logm(cov: np.ndarray) -> np.ndarray:
    """Matrix log of a symmetric positive-definite matrix, via eigdecomposition.

    Faster and more stable than a general ``scipy.linalg.logm`` here, because the
    input is symmetric: eigenvalues are real and positive after regularisation.
    """
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * np.log(vals)) @ vecs.T


class RiemannianTangentSpace(BaseEstimator, TransformerMixin):
    """Project flattened EEG covariance features into the log-Euclidean tangent space.

    Input rows are flattened covariance upper-triangles (as produced by the
    EEG loader's covariance mode). Output rows are flattened tangent vectors,
    centred at the training-set log-Euclidean mean.

    Parameters
    ----------
    shrinkage:
        Regularisation toward a scaled identity, in ``[0, 1)``. Small but
        non-zero so rank-deficient covariances still admit a logarithm.
    center:
        Subtract the training-set mean log-covariance (the log-Euclidean mean's
        logarithm). Improves conditioning and is leakage-safe because the mean
        is computed in ``fit`` on training data only.
    metric:
        ``"logeuclid"`` (default) uses the closed-form log-Euclidean map
        implemented here, which needs only NumPy/SciPy. ``"riemann"`` uses the
        **affine-invariant** metric via the optional :mod:`pyriemann` backend:
        it iterates to the true Riemannian geometric mean and whitens by it
        before taking the tangent map, which is the more principled projection
        on ill-conditioned real EEG. It is a drop-in with an extra dependency;
        install with ``pip install '.[riemann]'``. Selecting it without
        pyriemann installed raises a clear ImportError rather than silently
        falling back, so a reported "affine-invariant" result is always the real
        thing.
    """

    def __init__(
        self,
        shrinkage: float = 1e-3,
        center: bool = True,
        metric: str = "logeuclid",
    ) -> None:
        self.shrinkage = shrinkage
        self.center = center
        self.metric = metric

    def _log_covariances(self, X: np.ndarray) -> np.ndarray:
        k = n_channels_from_flat(X.shape[1])
        mats = unflatten_spd(X, k)
        logs = np.empty_like(mats)
        for i in range(mats.shape[0]):
            logs[i] = _symmetric_logm(_regularise(mats[i], self.shrinkage))
        return logs

    def _spd_stack(self, X: np.ndarray) -> np.ndarray:
        """Recover regularised, positive-definite ``(n, k, k)`` matrices."""
        k = n_channels_from_flat(X.shape[1])
        mats = unflatten_spd(X, k)
        return np.stack([_regularise(mats[i], self.shrinkage) for i in range(mats.shape[0])])

    def _make_pyriemann(self):
        try:
            from pyriemann.tangentspace import TangentSpace
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "metric='riemann' needs the affine-invariant backend pyriemann. "
                "Install with `pip install '.[riemann]'`, or use the default "
                "metric='logeuclid', which needs no extra dependency."
            ) from exc
        return TangentSpace(metric="riemann")

    def fit(self, X: np.ndarray, y=None) -> RiemannianTangentSpace:
        X = np.asarray(X, dtype=float)
        self.n_channels_ = n_channels_from_flat(X.shape[1])

        if self.metric == "riemann":
            # Affine-invariant backend: fit the geometric mean + whitening on
            # training data only, so it stays leakage-safe inside the fold/bag.
            self.backend_ = self._make_pyriemann()
            self.backend_.fit(self._spd_stack(X))
            return self
        if self.metric != "logeuclid":
            raise ValueError(f"metric must be 'logeuclid' or 'riemann'; got {self.metric!r}")

        logs = self._log_covariances(X)
        # Log-Euclidean mean's logarithm is just the arithmetic mean of the
        # per-trial log-covariances. This is the reference point we linearise at.
        self.mean_log_ = logs.mean(axis=0) if self.center else np.zeros_like(logs[0])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.metric == "riemann":
            if not hasattr(self, "backend_"):
                raise RuntimeError("RiemannianTangentSpace is not fitted")
            return self.backend_.transform(self._spd_stack(X))

        if not hasattr(self, "mean_log_"):
            raise RuntimeError("RiemannianTangentSpace is not fitted")
        logs = self._log_covariances(X)
        centred = logs - self.mean_log_[None, :, :]
        return flatten_spd(centred)

    def _more_tags(self):  # pragma: no cover - sklearn metadata
        return {"requires_positive_X": False, "stateless": False}
