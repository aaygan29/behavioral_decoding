"""Per-modality base learners, each wrapped in a bagged, resampling-safe pipeline.

Structure of one modality model, outermost first::

    BaggingClassifier
      └── Pipeline
            ├── StandardScaler      fitted on the bootstrap sample only
            ├── SMOTE               applied to the bootstrap sample only
            └── base estimator

The nesting order is the point. Bagging draws the bootstrap replicate *first*,
then the pipeline scales and resamples inside it, so every bag sees a different
synthetic minority set. That variation is useful: it stops the ensemble from
inheriting the idiosyncrasies of a single SMOTE draw, which is a real failure
mode when the minority class is small. Resample-then-bag would give every bag
the same synthetic points and throw that away, on top of leaking.

Defaults differ by modality because the shapes differ. Five ROI betas with 30
subjects is a regularised-linear problem. A 1536-dimensional ViT embedding is a
heavily-regularised-linear problem. Tabular behaviour with mixed monotone
effects is where trees earn their keep.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ..balance.smote import make_balanced_pipeline, recommend_strategy
from ..io.base import BEHAVIOR, EEG, FACE, FMRI
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Elastic-net logistic is the default for every dense-feature neural/embedding
# block: it shares weight across correlated features and still drops dead ones,
# which suits collinear ROI betas, collinear band-power columns, and
# high-dimensional face embeddings alike. `linear_svm` is a supported
# alternative for the face block (comparable accuracy, slower because of the
# probability calibration). Behaviour stays on gradient boosting for its mixed,
# monotone, low-dimensional features. See docs/estimators.md.
DEFAULT_BASE_LEARNER: Dict[str, str] = {
    FMRI: "elasticnet",
    EEG: "elasticnet",
    FACE: "elasticnet",
    BEHAVIOR: "gradient_boosting",
}

# Bag counts are modality-specific: noisier, higher-dimensional blocks benefit
# from more bags, cheap low-dimensional ones do not need them.
DEFAULT_N_BAGS: Dict[str, int] = {FMRI: 25, EEG: 25, FACE: 40, BEHAVIOR: 25}

# Default nested-tuning grids, keyed by base-learner kind. Deliberately small:
# the outer CV re-runs the whole search inside every training fold, so a grid
# with hundreds of points would cost more than the sample can justify. Keys are
# build_modality_model kwargs; C/l1_ratio pass through to the estimator,
# k_neighbors tunes the resampler, max_samples/n_bags tune the bagging.
DEFAULT_TUNE_GRID: Dict[str, Dict[str, List[Any]]] = {
    "elasticnet": {
        "C": [0.1, 1.0, 10.0],
        "l1_ratio": [0.2, 0.5, 0.8],
        "k_neighbors": [3, 5],
    },
    "logistic": {"C": [0.1, 1.0, 10.0], "k_neighbors": [3, 5]},
    "linear_svm": {"C": [0.1, 1.0, 10.0]},
    "gradient_boosting": {"max_samples": [0.6, 0.8], "n_bags": [25]},
    "random_forest": {"max_samples": [0.6, 0.8]},
}


def make_base_learner(
    kind: str = "logistic",
    class_weight: Optional[str] = "balanced",
    seed: int = 0,
    **kwargs: Any,
) -> BaseEstimator:
    """Instantiate a single base estimator by name."""
    if kind == "logistic":
        params = {
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": class_weight,
            "random_state": seed,
        }
        params.update(kwargs)
        return LogisticRegression(**params)
    if kind == "random_forest":
        params = {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "class_weight": class_weight,
            "random_state": seed,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return RandomForestClassifier(**params)
    if kind == "gradient_boosting":
        # No class_weight parameter on this estimator; imbalance is handled by
        # the resampling step in the pipeline.
        params = {"n_estimators": 200, "max_depth": 3, "random_state": seed}
        params.update(kwargs)
        return GradientBoostingClassifier(**params)
    if kind == "elasticnet":
        # L1 + L2 logistic. The L2 part shares weight across correlated features
        # (NAcc_L / NAcc_R move together; band-power columns are collinear),
        # while the L1 part still drops dead ones. This is usually a better fit
        # than plain L2 for the fMRI and EEG blocks, and than plain L1 for the
        # high-dimensional face block. Needs the saga solver, and needs scaled
        # input, which the pipeline's StandardScaler provides.
        params = {
            "penalty": "elasticnet",
            "solver": "saga",
            "l1_ratio": 0.5,
            "C": 1.0,
            "max_iter": 5000,
            "tol": 1e-3,
            "class_weight": class_weight,
            "random_state": seed,
        }
        params.update(kwargs)
        return LogisticRegression(**params)
    if kind == "linear_svm":
        # Linear-kernel SVM, a strong baseline for high-dimensional embeddings
        # (the face block). probability=True adds Platt scaling via an internal
        # CV so the ensemble can read predict_proba; that calibration costs time
        # but the SMOTE step upstream keeps the classes balanced enough for it.
        params = {
            "kernel": "linear",
            "C": 1.0,
            "probability": True,
            "class_weight": class_weight,
            "random_state": seed,
        }
        params.update(kwargs)
        return SVC(**params)
    if kind == "svm":
        params = {
            "kernel": "rbf",
            "C": 1.0,
            "probability": True,
            "class_weight": class_weight,
            "random_state": seed,
        }
        params.update(kwargs)
        return SVC(**params)
    raise ValueError(
        f"unknown base learner {kind!r}; choose from logistic, elasticnet, "
        "linear_svm, random_forest, gradient_boosting, svm"
    )


def build_modality_model(
    modality: str,
    y: Optional[np.ndarray] = None,
    base_learner: Optional[str] = None,
    n_bags: Optional[int] = None,
    sampler: Optional[str] = None,
    k_neighbors: Optional[int] = None,
    max_features: float = 1.0,
    max_samples: float = 0.8,
    seed: int = 0,
    riemann_metric: str = "logeuclid",
    **learner_kwargs: Any,
) -> BaggingClassifier:
    """Build the bagged, resampling-safe model for one modality.

    Parameters
    ----------
    y:
        Training labels. When given, the resampling strategy is chosen by
        :func:`~behavioral_decoding.balance.smote.recommend_strategy` rather
        than assumed, and the rationale is logged. Pass ``sampler`` explicitly
        to override.
    max_samples:
        Fraction of rows per bag. Below 1.0 the bags decorrelate more, which is
        where bagging's variance reduction comes from.
    max_features:
        Fraction of columns per bag. Worth dropping below 1.0 for the
        high-dimensional face block.
    """
    kind = base_learner or DEFAULT_BASE_LEARNER.get(modality, "logistic")
    bags = n_bags if n_bags is not None else DEFAULT_N_BAGS.get(modality, 25)

    chosen_sampler = sampler
    chosen_k = k_neighbors
    class_weight: Optional[str] = "balanced"

    if y is not None and sampler is None:
        rec = recommend_strategy(np.asarray(y), modality=modality)
        chosen_sampler = str(rec["sampler"])
        chosen_k = rec["k_neighbors"] if rec["k_neighbors"] is not None else 5  # type: ignore[assignment]
        class_weight = rec["class_weight"]  # type: ignore[assignment]
        logger.info("balance[%s]: %s", modality, rec["rationale"])
    if chosen_sampler is None:
        chosen_sampler = "smote"
    if chosen_k is None:
        chosen_k = 5

    if kind == "gradient_boosting":
        class_weight = None

    # The Riemannian path is a logistic regression in the log-Euclidean tangent
    # space: a tangent-projection transformer is prepended, and the classifier
    # itself is plain logistic. It cannot subsample features per bag, because a
    # column subset of a flattened covariance triangle is no longer a valid
    # covariance, so max_features is forced to 1.0.
    pre_steps = None
    effective_max_features = max_features
    if kind == "riemann":
        from .riemann import RiemannianTangentSpace

        pre_steps = [("tangent", RiemannianTangentSpace(metric=riemann_metric))]
        estimator = make_base_learner("logistic", class_weight=class_weight, seed=seed)
        if max_features != 1.0:
            logger.warning(
                "riemann[%s]: max_features forced to 1.0; a column subset of a "
                "flattened covariance is not a covariance",
                modality,
            )
            effective_max_features = 1.0
    else:
        estimator = make_base_learner(kind, class_weight=class_weight, seed=seed, **learner_kwargs)

    pipeline = make_balanced_pipeline(
        estimator,
        sampler=str(chosen_sampler),
        k_neighbors=int(chosen_k),
        scaler=True,
        random_state=seed,
        pre_steps=pre_steps,
    )

    model = BaggingClassifier(
        estimator=pipeline,
        n_estimators=bags,
        max_samples=max_samples,
        max_features=effective_max_features,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        random_state=seed,
        n_jobs=1,
    )
    # Attached for the run record; BaggingClassifier does not expose these.
    model.bd_spec_ = {
        "modality": modality,
        "base_learner": kind,
        "riemann_metric": riemann_metric if kind == "riemann" else None,
        "n_bags": bags,
        "sampler": chosen_sampler,
        "k_neighbors": chosen_k,
        "class_weight": class_weight,
        "max_samples": max_samples,
        "max_features": max_features,
        "seed": seed,
    }
    return model


def _split_grid_keys(grid: Dict[str, List[Any]]) -> Tuple[List[str], List[List[Any]]]:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    return keys, values


def tune_modality_model(
    modality: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    base_learner: Optional[str] = None,
    grid: Optional[Dict[str, List[Any]]] = None,
    n_splits: int = 4,
    weight_metric: str = "balanced_accuracy",
    seed: int = 0,
    fixed_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Nested, subject-grouped hyperparameter search for one modality model.

    This is the *inner* loop of a nested CV. It is meant to be called on the
    training rows of an outer fold only (:class:`MultimodalEnsemble` does exactly
    that), so the outer test subjects never touch the search. Within the training
    set it scores each grid point by grouped out-of-fold ``weight_metric`` using
    the same subject-grouped splitter as everything else, so a subject is never
    on both sides of an inner fold either.

    The grid can tune the elastic-net penalty (``C``, ``l1_ratio``), the
    resampling (``k_neighbors``), and the bagging (``n_bags``, ``max_samples``)
    together, because every point is evaluated on the fully-assembled bagged,
    resampling-safe model, not on a bare estimator.

    Returns a dict with ``best_params``, ``best_score``, and the full ``table``
    of ``(params, score)`` for the run record. Falls back to an empty search
    (``best_params={}``) if fewer than two inner folds are possible.
    """
    from ..evaluation.cv import out_of_fold_proba
    from ..evaluation.metrics import classification_report

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    fixed = dict(fixed_kwargs or {})

    kind = (
        base_learner
        or fixed.get("base_learner")
        or DEFAULT_BASE_LEARNER.get(modality, "logistic")
    )
    search_grid = grid if grid is not None else DEFAULT_TUNE_GRID.get(kind, {})

    n_subjects = len(np.unique(groups))
    inner = int(min(n_splits, n_subjects))
    if inner < 2 or not search_grid:
        logger.info(
            "tune[%s]: skipped (%s); using defaults",
            modality,
            "no grid" if not search_grid else f"only {n_subjects} subjects",
        )
        return {"best_params": {}, "best_score": float("nan"), "table": [], "base_learner": kind}

    keys, values = _split_grid_keys(search_grid)
    table: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_params: Dict[str, Any] = {}

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        build_kwargs = dict(fixed)
        build_kwargs.update(params)
        build_kwargs.pop("base_learner", None)
        build_kwargs["seed"] = seed  # fixed may already carry seed; this wins
        try:
            model = build_modality_model(
                modality, y=y, base_learner=kind, **build_kwargs
            )
            oof, _ = out_of_fold_proba(
                model, X, y, groups, n_splits=inner, seed=seed,
                desc=f"tune:{modality}",
            )
            score = classification_report(y, oof).get(weight_metric, float("nan"))
        except Exception as exc:  # noqa: BLE001 - a bad grid point should not kill the search
            logger.warning("tune[%s]: grid point %s failed (%s); skipped", modality, params, exc)
            continue
        table.append({"params": params, weight_metric: float(score)})
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_params = params

    logger.info(
        "tune[%s]: best %s = %.3f at %s (searched %d points)",
        modality, weight_metric, best_score if np.isfinite(best_score) else float("nan"),
        best_params, len(table),
    )
    return {
        "best_params": best_params,
        "best_score": float(best_score) if np.isfinite(best_score) else float("nan"),
        "weight_metric": weight_metric,
        "table": table,
        "base_learner": kind,
    }
