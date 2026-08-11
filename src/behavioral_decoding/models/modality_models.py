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

from typing import Any, Dict, Optional

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

        pre_steps = [("tangent", RiemannianTangentSpace())]
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
        "n_bags": bags,
        "sampler": chosen_sampler,
        "k_neighbors": chosen_k,
        "class_weight": class_weight,
        "max_samples": max_samples,
        "max_features": max_features,
        "seed": seed,
    }
    return model
