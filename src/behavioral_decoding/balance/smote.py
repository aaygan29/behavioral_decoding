"""Synthetic minority oversampling, wired so it cannot leak.

SMOTE is easy to use and easy to misuse, and the misuse is nearly invisible.
Resampling the full dataset and *then* splitting into train and test puts
synthetic points interpolated from test-set neighbours into the training set.
Accuracy jumps, the jump is an artefact, and nothing in the output looks wrong.

The rule this module enforces: **resampling happens inside the fold, on training
data only**. :func:`make_balanced_pipeline` builds an imbalanced-learn pipeline
whose resampler is skipped at predict time by construction, so cross-validating
that pipeline is safe. :func:`resample_train_only` is the manual escape hatch,
and it takes only the training arrays so there is nothing to get wrong.

There is also a question of whether to resample at all. With a 2:1 imbalance,
class weights usually do the same job with less risk. SMOTE earns its place
around 5:1 and beyond, and on features where linear interpolation between
neighbours is meaningful. Interpolating between two subjects' ViT embeddings
produces a face that no one has; interpolating between ROI betas is more
defensible. Check :func:`recommend_strategy` before reaching for it.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

SAMPLERS = ("smote", "borderline", "svm", "adasyn", "random", "none")


def _imblearn():
    try:
        import imblearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Resampling requires imbalanced-learn. Install with "
            "`pip install imbalanced-learn`, or set sampler='none' to rely on "
            "class weights instead."
        ) from exc
    return imblearn


def imbalance_ratio(y: np.ndarray) -> float:
    """Majority-to-minority count ratio."""
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max()) / float(counts.min())


def minority_count(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())


def recommend_strategy(y: np.ndarray, modality: Optional[str] = None) -> Dict[str, object]:
    """Say whether resampling is warranted, and which variant.

    Returns a dict with ``sampler``, ``k_neighbors``, ``class_weight``, and a
    ``rationale`` string that gets written into the run record.
    """
    ratio = imbalance_ratio(y)
    n_min = minority_count(y)

    if n_min < 6:
        return {
            "sampler": "none",
            "k_neighbors": None,
            "class_weight": "balanced",
            "rationale": (
                f"only {n_min} minority samples; SMOTE needs k+1 neighbours and would "
                "interpolate within a handful of points, manufacturing structure. "
                "Use class weights."
            ),
        }

    k = int(min(5, n_min - 1))

    if ratio < 1.5:
        return {
            "sampler": "none",
            "k_neighbors": None,
            "class_weight": None,
            "rationale": f"imbalance {ratio:.2f}:1 is mild; no resampling needed",
        }
    if ratio < 3.0:
        return {
            "sampler": "none",
            "k_neighbors": None,
            "class_weight": "balanced",
            "rationale": (
                f"imbalance {ratio:.2f}:1 is moderate; class weights achieve the same "
                "effect as resampling without synthesising points"
            ),
        }
    if modality == "face":
        return {
            "sampler": "borderline",
            "k_neighbors": k,
            "class_weight": "balanced",
            "rationale": (
                f"imbalance {ratio:.2f}:1 on high-dimensional embeddings; borderline "
                "variant synthesises only near the decision boundary, which limits "
                "interpolation between unrelated identities"
            ),
        }
    return {
        "sampler": "smote",
        "k_neighbors": k,
        "class_weight": "balanced",
        "rationale": f"imbalance {ratio:.2f}:1 with {n_min} minority samples; SMOTE(k={k})",
    }


class AdaptiveOverSampler:
    """A resampler that adapts to the data it is actually handed, or steps aside.

    This exists because ``k_neighbors`` is chosen once, on the full training set,
    but the resampler runs on much smaller slices: each inner cross-validation
    fold, and then each bagging bootstrap replicate inside that. A block with 40
    minority trials overall can easily hand a single bag 2 of them, at which
    point ``SMOTE(k_neighbors=5)`` raises ``n_neighbors <= n_samples_fit`` and
    takes the whole run down.

    The wrapper resolves ``k`` against the minority count in the data it
    receives, and when even that is not enough it returns the data untouched
    rather than fabricating neighbours out of two points. Interpolating a
    "minority class" from two examples does not describe a class; it draws a line
    segment. Class weights on the base estimator carry the imbalance in that
    case, which is what
    :func:`recommend_strategy` would have chosen for such a sample anyway.

    Skips are logged at DEBUG. They are expected and frequent under bagging with
    a small minority class, so they are not warnings, but they are recorded.
    """

    def __init__(
        self,
        sampler: str = "smote",
        k_neighbors: int = 5,
        random_state: int = 0,
        min_minority: int = 6,
    ) -> None:
        self.sampler = sampler
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.min_minority = min_minority

    # scikit-learn clone() contract
    def get_params(self, deep: bool = True) -> Dict[str, object]:
        return {
            "sampler": self.sampler,
            "k_neighbors": self.k_neighbors,
            "random_state": self.random_state,
            "min_minority": self.min_minority,
        }

    def set_params(self, **params: object) -> AdaptiveOverSampler:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            logger.debug("resample skipped: only one class present in this slice")
            return X, y

        n_min = int(counts.min())
        if n_min < max(2, self.min_minority):
            logger.debug(
                "resample skipped: %d minority samples in this slice (need %d)",
                n_min,
                max(2, self.min_minority),
            )
            return X, y

        k = int(min(self.k_neighbors, n_min - 1))
        if k < 1:
            logger.debug("resample skipped: resolved k_neighbors=%d", k)
            return X, y

        try:
            sampler = make_sampler(self.sampler, k, self.random_state)
        except ImportError:
            logger.debug("resample skipped: imbalanced-learn not installed")
            return X, y
        if sampler is None:
            return X, y

        try:
            return sampler.fit_resample(X, y)
        except (ValueError, RuntimeError) as exc:
            # Variant-specific failures, e.g. BorderlineSMOTE finding no
            # borderline points, or SVMSMOTE failing to fit its SVM. Falling
            # back to the original data is correct; silently swapping in a
            # different sampler would change the method without saying so.
            logger.debug("resample skipped: %s raised %s", self.sampler, exc)
            return X, y

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"AdaptiveOverSampler(sampler={self.sampler!r}, "
            f"k_neighbors={self.k_neighbors}, min_minority={self.min_minority})"
        )


def make_sampler(
    sampler: str = "smote",
    k_neighbors: int = 5,
    random_state: int = 0,
    sampling_strategy: object = "auto",
):
    """Build a resampler. Returns ``None`` for ``sampler="none"``."""
    if sampler == "none":
        return None
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}")

    _imblearn()
    from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, BorderlineSMOTE, RandomOverSampler

    common = {"random_state": random_state, "sampling_strategy": sampling_strategy}
    if sampler == "smote":
        return SMOTE(k_neighbors=k_neighbors, **common)
    if sampler == "borderline":
        return BorderlineSMOTE(k_neighbors=k_neighbors, **common)
    if sampler == "svm":
        return SVMSMOTE(k_neighbors=k_neighbors, **common)
    if sampler == "adasyn":
        return ADASYN(n_neighbors=k_neighbors, **common)
    return RandomOverSampler(**common)


def make_balanced_pipeline(
    estimator,
    sampler: str = "smote",
    k_neighbors: int = 5,
    scaler: bool = True,
    random_state: int = 0,
):
    """Wrap ``estimator`` in a leakage-safe scale -> resample -> fit pipeline.

    The resampling step is applied during ``fit`` and skipped during
    ``predict``, which is exactly the behaviour needed for honest CV. Scaling is
    fitted on training data only, for the same reason.

    The resampling step is an :class:`AdaptiveOverSampler`, which resolves ``k``
    against whatever slice of data the fold or bag actually hands it.

    Falls back to a plain scikit-learn pipeline (scaler + estimator, no
    resampling) when imbalanced-learn is absent, and logs that it did.
    """
    steps = []
    if scaler:
        from sklearn.preprocessing import StandardScaler

        steps.append(("scaler", StandardScaler()))

    resampler = None
    if sampler != "none":
        try:
            _imblearn()
            resampler = AdaptiveOverSampler(
                sampler=sampler, k_neighbors=k_neighbors, random_state=random_state
            )
        except ImportError:
            logger.warning(
                "imbalanced-learn not installed; building pipeline WITHOUT resampling. "
                "Set class_weight='balanced' on the estimator to compensate."
            )

    if resampler is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline

        steps.append(("resample", resampler))
        steps.append(("estimator", estimator))
        return ImbPipeline(steps)

    from sklearn.pipeline import Pipeline as SkPipeline

    steps.append(("estimator", estimator))
    return SkPipeline(steps)


def resample_train_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampler: str = "smote",
    k_neighbors: int = 5,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample training arrays. Takes no test data, so none can leak.

    Prefer :func:`make_balanced_pipeline`. Use this only when you are writing a
    custom training loop that cannot accept a pipeline object.
    """
    resampler = make_sampler(sampler, k_neighbors, random_state)
    if resampler is None:
        return np.asarray(X_train), np.asarray(y_train)
    X_res, y_res = resampler.fit_resample(np.asarray(X_train), np.asarray(y_train))
    logger.info(
        "resample_train_only: %s grew training set %d -> %d (imbalance %.2f:1 -> %.2f:1)",
        sampler,
        len(y_train),
        len(y_res),
        imbalance_ratio(np.asarray(y_train)),
        imbalance_ratio(np.asarray(y_res)),
    )
    return X_res, y_res
