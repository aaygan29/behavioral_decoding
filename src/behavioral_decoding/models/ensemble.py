"""Reconciling one model per modality into one prediction.

Each modality gets its own bagged model. Those models disagree, and the
disagreement is informative: a trial where fMRI and face say "yes" and behaviour
says "no" is a different object from one where all three agree. The question is
how to combine them.

Three reconciliation rules are implemented, in increasing order of how much they
assume:

``majority``
    Hard vote. Each modality casts one vote, ties broken by mean probability.
    Assumes nothing about relative model quality. It is the honest default when
    you have few subjects, because it cannot overfit the weighting.

``soft``
    Mean of predicted probabilities. Uses confidence, still assumes all
    modalities are equally trustworthy.

``accuracy_weighted``
    Weight each modality by how far its **out-of-fold** balanced accuracy exceeds
    chance, then take a weighted probability average. This is the rule the
    project is built around.

The word *out-of-fold* is load-bearing. Weighting by training accuracy would
hand the largest weight to whichever model memorised its training set hardest,
which for the 1536-dimensional face block is guaranteed. Weights here come from
:func:`~behavioral_decoding.evaluation.cv.out_of_fold_proba`, run inside ``fit``
on training data only, with folds grouped by subject.

A modality whose out-of-fold balanced accuracy is at or below chance gets weight
zero by default. That is not a bug to be smoothed over with a floor: a modality
that cannot beat chance on held-out subjects has nothing to contribute, and
letting it vote adds variance and no signal.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from ..evaluation.cv import out_of_fold_proba
from ..evaluation.metrics import classification_report
from ..io.base import ModalityBlock, MultimodalDataset
from ..utils.logging import get_logger
from ..utils.progress import progress
from .modality_models import build_modality_model

logger = get_logger(__name__)

RECONCILIATION_STRATEGIES = ("majority", "soft", "accuracy_weighted")


class MultimodalEnsemble(BaseEstimator, ClassifierMixin):
    """Fit one bagged model per modality and reconcile their predictions."""

    def __init__(
        self,
        reconciliation: str = "accuracy_weighted",
        weight_metric: str = "balanced_accuracy",
        n_splits: int = 5,
        seed: int = 0,
        modality_specs: Optional[Dict[str, Dict[str, object]]] = None,
        drop_below_chance: bool = True,
        weight_floor: float = 0.0,
        chance_level: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        weight_metric:
            Any key produced by
            :func:`~behavioral_decoding.evaluation.metrics.classification_report`.
            ``balanced_accuracy`` and ``roc_auc`` both have chance at 0.5.
        drop_below_chance:
            Zero out modalities that do not beat chance out of fold.
        weight_floor:
            Minimum weight for a surviving modality, before normalisation. Use
            sparingly; it exists for the case where you have a strong prior that
            a modality matters and a sample too small to demonstrate it.
        chance_level:
            Chance value for ``weight_metric``. Defaults to 0.5.
        """
        self.reconciliation = reconciliation
        self.weight_metric = weight_metric
        self.n_splits = n_splits
        self.seed = seed
        self.modality_specs = modality_specs
        self.drop_below_chance = drop_below_chance
        self.weight_floor = weight_floor
        self.chance_level = chance_level

    # -------------------------------------------------------------------- fit

    def fit(self, dataset: MultimodalDataset, y: Optional[np.ndarray] = None) -> MultimodalEnsemble:
        """Fit every modality model and estimate reconciliation weights.

        ``dataset`` must contain only training data. Passing the full dataset
        here and then reporting performance on it is the leak this design is
        built to prevent; use
        :func:`~behavioral_decoding.evaluation.neuroforecast.cross_validate_ensemble`
        for an honest end-to-end estimate.
        """
        if self.reconciliation not in RECONCILIATION_STRATEGIES:
            raise ValueError(
                f"reconciliation must be one of {RECONCILIATION_STRATEGIES}"
            )

        y_train = np.asarray(dataset.y_individual if y is None else y)
        self.classes_ = np.unique(y_train)
        if len(self.classes_) != 2:
            raise ValueError(
                f"MultimodalEnsemble expects binary labels, found {self.classes_}"
            )

        specs = self.modality_specs or {}
        groups = dataset.subject_ids
        n_subjects = len(np.unique(groups))
        effective_splits = int(min(self.n_splits, n_subjects))
        if effective_splits < 2:
            raise ValueError(
                "need at least 2 subjects to estimate out-of-fold weights, got "
                f"{n_subjects}"
            )
        if effective_splits < self.n_splits:
            logger.warning(
                "only %d subjects available; reducing inner folds %d -> %d",
                n_subjects,
                self.n_splits,
                effective_splits,
            )

        self.models_: Dict[str, object] = {}
        self.oof_proba_: Dict[str, np.ndarray] = {}
        self.modality_scores_: Dict[str, Dict[str, float]] = {}
        self.specs_: Dict[str, Dict[str, object]] = {}

        for modality in progress(dataset.modalities, desc="modalities"):
            block: ModalityBlock = dataset.blocks[modality]
            kwargs = dict(specs.get(modality, {}))
            kwargs.setdefault("seed", self.seed)
            model = build_modality_model(modality, y=y_train, **kwargs)  # type: ignore[arg-type]
            self.specs_[modality] = dict(getattr(model, "bd_spec_", {}))

            oof, _ = out_of_fold_proba(
                model,
                block.X,
                y_train,
                groups,
                n_splits=effective_splits,
                seed=self.seed,
                desc=f"oof:{modality}",
            )
            self.oof_proba_[modality] = oof
            self.modality_scores_[modality] = classification_report(y_train, oof)

            # Refit on all training data now that the honest estimate is banked.
            fitted = clone(model)
            fitted.fit(block.X, y_train)
            self.models_[modality] = fitted

        self.weights_ = self._compute_weights()
        self._log_weights()
        return self

    def _compute_weights(self) -> Dict[str, float]:
        chance = self.chance_level if self.chance_level is not None else 0.5
        raw: Dict[str, float] = {}
        for modality, scores in self.modality_scores_.items():
            score = scores.get(self.weight_metric, float("nan"))
            if np.isnan(score):
                raw[modality] = 0.0
                continue
            excess = score - chance
            if self.drop_below_chance and excess <= 0:
                raw[modality] = 0.0
            else:
                raw[modality] = max(excess, self.weight_floor)

        total = sum(raw.values())
        if total <= 0:
            # No modality beat chance. Falling back to uniform keeps the object
            # usable, but the flag below must survive into the run record: this
            # is a null result, not a working model.
            logger.warning(
                "no modality exceeded chance on %s out of fold; falling back to "
                "uniform weights. Treat downstream predictions as a null result.",
                self.weight_metric,
            )
            self.weights_degenerate_ = True
            n = len(raw)
            return {m: 1.0 / n for m in raw}

        self.weights_degenerate_ = False
        return {m: w / total for m, w in raw.items()}

    def _log_weights(self) -> None:
        for modality, weight in sorted(self.weights_.items(), key=lambda kv: -kv[1]):
            score = self.modality_scores_[modality].get(self.weight_metric, float("nan"))
            logger.info(
                "weight[%s] = %.3f  (oof %s = %.3f)", modality, weight, self.weight_metric, score
            )

    # ---------------------------------------------------------------- predict

    def _modality_proba(self, dataset: MultimodalDataset) -> Dict[str, np.ndarray]:
        self._check_fitted()
        positive = self.classes_[1]
        out: Dict[str, np.ndarray] = {}
        for modality, model in self.models_.items():
            if modality not in dataset.blocks:
                raise KeyError(
                    f"dataset is missing modality {modality!r}, which the ensemble was fitted "
                    "on. Predicting with a subset of modalities requires refitting, "
                    "because the weights were normalised over the full set."
                )
            proba = model.predict_proba(dataset.blocks[modality].X)
            col = list(model.classes_).index(positive)
            out[modality] = proba[:, col]
        return out

    def predict_proba(self, dataset: MultimodalDataset) -> np.ndarray:
        """Reconciled probabilities, shape ``(n_trials, 2)``."""
        per_modality = self._modality_proba(dataset)
        p1 = self.reconcile(per_modality)
        return np.column_stack([1.0 - p1, p1])

    def reconcile(self, per_modality: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine per-modality positive-class probabilities into one vector."""
        modalities = list(per_modality.keys())
        stacked = np.vstack([per_modality[m] for m in modalities])  # (n_mod, n_trials)

        if self.reconciliation == "soft":
            return stacked.mean(axis=0)

        if self.reconciliation == "majority":
            votes = (stacked >= 0.5).astype(float)
            share = votes.mean(axis=0)
            # An even number of modalities can tie. Break with mean confidence
            # rather than an arbitrary class, and nudge off 0.5 so the result is
            # a usable probability.
            tied = np.isclose(share, 0.5)
            share = share.copy()
            if tied.any():
                share[tied] = np.clip(stacked[:, tied].mean(axis=0), 1e-6, 1 - 1e-6)
            return share

        weights = np.array([self.weights_[m] for m in modalities]).reshape(-1, 1)
        return (stacked * weights).sum(axis=0)

    def predict(self, dataset: MultimodalDataset) -> np.ndarray:
        proba = self.predict_proba(dataset)[:, 1]
        return np.where(proba >= 0.5, self.classes_[1], self.classes_[0])

    # ----------------------------------------------------------------- report

    def _check_fitted(self) -> None:
        if not hasattr(self, "models_"):
            raise RuntimeError("MultimodalEnsemble is not fitted; call fit(dataset) first")

    def reconciliation_report(self) -> str:
        """Table of per-modality out-of-fold quality and resulting weights."""
        self._check_fitted()
        chance = self.chance_level if self.chance_level is not None else 0.5
        lines = [
            f"reconciliation: {self.reconciliation} "
            f"(weights from out-of-fold {self.weight_metric})",
            "  {:<10} {:>10} {:>10} {:>9} {:>8}".format(
                "modality", "oof_balacc", "oof_auc", "oof_ap", "weight"
            ),
        ]
        for modality in sorted(self.weights_, key=lambda m: -self.weights_[m]):
            s = self.modality_scores_[modality]
            lines.append(
                "  {:<10} {:>10.4f} {:>10.4f} {:>9.4f} {:>8.4f}{}".format(
                    modality,
                    s["balanced_accuracy"],
                    s["roc_auc"],
                    s["average_precision"],
                    self.weights_[modality],
                    "  (below chance, dropped)"
                    if s.get(self.weight_metric, 0.0) <= chance and self.drop_below_chance
                    else "",
                )
            )
        if getattr(self, "weights_degenerate_", False):
            lines.append(
                "  WARNING: no modality beat chance; weights are uniform by fallback"
            )
        return "\n".join(lines)

    def to_record(self) -> Dict[str, object]:
        """Serialisable summary for the run record."""
        self._check_fitted()
        return {
            "reconciliation": self.reconciliation,
            "weight_metric": self.weight_metric,
            "weights": dict(self.weights_),
            "weights_degenerate": bool(getattr(self, "weights_degenerate_", False)),
            "modality_scores": {
                m: dict(s) for m, s in self.modality_scores_.items()
            },
            "modality_specs": {m: dict(s) for m, s in self.specs_.items()},
            "seed": self.seed,
            "n_splits": self.n_splits,
        }
