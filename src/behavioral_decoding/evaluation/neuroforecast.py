"""Individual prediction and aggregate forecasting, evaluated as separate claims.

The empirical result this module is built to test is a dissociation, not a
correlation. Across several studies, the neural signal that best predicts what
*this person* will choose is not the signal that best forecasts what *the
market* will do, and in some cases the behavioural measure that predicts the
individual fails at the market entirely:

* Genevsky, Yoon & Knutson (2017): NAcc and MPFC both predicted individual
  crowdfunding choices; only NAcc forecast market funding outcomes weeks later.
  Behavioural measures from the scanned sample did not forecast the market.
* Genevsky & Knutson (2015): NAcc activity predicted microloan request success
  on the internet above and beyond the scanned sample's own lending choices.
* Falk, Berkman & Lieberman (2012): MPFC predicted population call volume to a
  smoking-cessation hotline; participants' self-reported effectiveness ratings
  did not.
* Tong et al. (2020): NAcc up and anterior insula down at video onset forecast
  aggregate view frequency and duration on YouTube, beyond conventional measures.

So the framework never reports a single headline number. It reports the
individual-level result and the aggregate-level result side by side, and it
reports each arm (brain-only, behaviour-only, combined) separately, because the
interesting quantity is the *gap* between them. Full citations in
``docs/literature.md``.

Two levels means two units of analysis and two ways to leak. Individual-level CV
groups by subject. Aggregate-level CV groups by **stimulus**: forecasting the
market outcome of an item the model has already seen is not forecasting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..io.base import BEHAVIOR, EEG, FACE, FMRI, ModalityBlock, MultimodalDataset
from ..utils.logging import get_logger
from ..utils.progress import progress
from .cv import iter_splits
from .metrics import classification_report

logger = get_logger(__name__)

NEURAL_MODALITIES = (FMRI, EEG)
PERIPHERAL_MODALITIES = (FACE,)
SELF_REPORT_MODALITIES = (BEHAVIOR,)


# --------------------------------------------------------------------- pooling


def aggregate_by_stimulus(
    values: np.ndarray,
    stimulus_ids: np.ndarray,
    statistic: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool trial-level values into one value per stimulus.

    This is the "neural focus group" step: a small scanned sample stands in for
    a population, so the per-stimulus group average is the forecast feature.

    ``statistic="trimmed_mean"`` drops the top and bottom 10% before averaging,
    which matters when a single subject's motion artefact would otherwise move
    an item's group score.
    """
    values = np.asarray(values, dtype=float)
    stimulus_ids = np.asarray(stimulus_ids)
    unique = np.unique(stimulus_ids)

    out = np.empty(len(unique), dtype=float)
    for i, stim in enumerate(unique):
        vals = values[stimulus_ids == stim]
        if statistic == "mean":
            out[i] = np.mean(vals)
        elif statistic == "median":
            out[i] = np.median(vals)
        elif statistic == "trimmed_mean":
            out[i] = stats.trim_mean(vals, 0.1) if len(vals) >= 5 else np.mean(vals)
        elif statistic == "sum":
            out[i] = np.sum(vals)
        else:
            raise ValueError(
                f"statistic must be mean, median, trimmed_mean, or sum; got {statistic!r}"
            )
    return unique, out


def aggregate_block(
    block: ModalityBlock,
    statistic: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool a whole feature matrix to ``(n_stimuli, n_features)``."""
    unique = np.unique(block.stimulus_ids)
    out = np.empty((len(unique), block.n_features), dtype=float)
    for i, stim in enumerate(unique):
        rows = block.X[block.stimulus_ids == stim]
        if statistic == "mean":
            out[i] = np.nanmean(rows, axis=0)
        elif statistic == "median":
            out[i] = np.nanmedian(rows, axis=0)
        elif statistic == "trimmed_mean":
            out[i] = (
                stats.trim_mean(rows, 0.1, axis=0)
                if rows.shape[0] >= 5
                else np.nanmean(rows, axis=0)
            )
        else:
            raise ValueError("statistic must be mean, median, or trimmed_mean")
    return unique, out


def subjects_per_stimulus(stimulus_ids: np.ndarray) -> Dict[object, int]:
    """How many trials contribute to each stimulus's group average."""
    unique, counts = np.unique(np.asarray(stimulus_ids), return_counts=True)
    return {u: int(c) for u, c in zip(unique, counts)}


# ------------------------------------------------------------------ forecasting


def forecast_market(
    features: np.ndarray,
    market_outcome: np.ndarray,
    n_splits: int = 5,
    task: str = "regression",
    seed: int = 0,
    estimator: Optional[object] = None,
) -> Dict[str, float]:
    """Held-out forecast of a per-stimulus market outcome.

    Rows are stimuli, not trials. Folds hold out whole stimuli, so every
    reported number is a genuine out-of-sample forecast.

    With the sample sizes typical of this design (30 to 100 stimuli), ridge with
    an internal alpha search is the sensible default. Anything with more
    capacity will fit the noise and report it as a forecast.
    """
    features = np.asarray(features, dtype=float)
    market_outcome = np.asarray(market_outcome)
    n = len(market_outcome)

    if features.shape[0] != n:
        raise ValueError(
            f"features has {features.shape[0]} rows but market_outcome has {n}"
        )
    if n < n_splits:
        raise ValueError(
            f"only {n} stimuli with market outcomes; cannot run {n_splits} folds. Aggregate "
            "forecasting needs items, and a handful of items cannot support a "
            "claim either way."
        )
    if n < 20:
        logger.warning(
            "only %d stimuli in the forecasting sample; out-of-sample estimates at "
            "this size are extremely noisy and should be reported with intervals, "
            "not as point claims",
            n,
        )

    if task == "regression":
        model = estimator or Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ]
        )
    elif task == "classification":
        model = estimator or Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
    else:
        raise ValueError("task must be 'regression' or 'classification'")

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = np.full(n, np.nan)

    for train_idx, test_idx in splitter.split(features):
        fold_model = clone(model)
        fold_model.fit(features[train_idx], market_outcome[train_idx])
        if task == "regression":
            predictions[test_idx] = fold_model.predict(features[test_idx])
        else:
            predictions[test_idx] = fold_model.predict_proba(features[test_idx])[:, 1]

    if task == "classification":
        report = classification_report(market_outcome, predictions)
        report["n_stimuli"] = float(n)
        return report

    resid = market_outcome - predictions
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((market_outcome - market_outcome.mean()) ** 2))
    pearson_r, pearson_p = stats.pearsonr(market_outcome, predictions)
    spearman_r, spearman_p = stats.spearmanr(market_outcome, predictions)

    return {
        "n_stimuli": float(n),
        # Out-of-sample R^2 against the training-set mean. It can and does go
        # negative when the model forecasts worse than the mean, and that
        # negative value is the honest answer, not a bug to clip away.
        "r2_out_of_sample": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
    }


def compare_forecast_arms(
    dataset: MultimodalDataset,
    task: str = "regression",
    statistic: str = "mean",
    n_splits: int = 5,
    seed: int = 0,
    arms: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Forecast the market outcome from each arm separately, and compare.

    The default arms mirror the comparison the literature actually makes:
    brain alone, self-report alone, face alone, and everything together. Running
    only the combined arm would hide the dissociation this whole design exists
    to detect.
    """
    if dataset.y_aggregate is None:
        raise ValueError(
            "dataset has no y_aggregate; aggregate forecasting requires a market "
            "outcome per stimulus (funding success, view count, sales, call volume)"
        )

    available = set(dataset.modalities)
    if arms is None:
        arms = {}
        neural = [m for m in NEURAL_MODALITIES if m in available]
        if neural:
            arms["brain_only"] = neural
        for m in sorted(available):
            arms[f"{m}_only"] = [m]
        if len(available) > 1:
            arms["all_modalities"] = sorted(available)

    results: Dict[str, Dict[str, float]] = {}
    for arm_name, modalities in progress(list(arms.items()), desc="forecast arms"):
        modalities = [m for m in modalities if m in available]
        if not modalities:
            continue

        feature_parts: List[np.ndarray] = []
        stim_ref: Optional[np.ndarray] = None
        for modality in modalities:
            stims, agg = aggregate_block(dataset.blocks[modality], statistic=statistic)
            if stim_ref is None:
                stim_ref = stims
            elif not np.array_equal(stim_ref, stims):  # pragma: no cover - defensive
                raise AssertionError("stimulus ordering differs across modalities")
            feature_parts.append(agg)

        assert stim_ref is not None
        keep = np.array([s in dataset.y_aggregate for s in stim_ref])
        if keep.sum() < n_splits:
            logger.warning(
                "arm %s: only %d stimuli have market outcomes; skipping",
                arm_name,
                int(keep.sum()),
            )
            continue

        X = np.hstack(feature_parts)[keep]
        y = np.array([dataset.y_aggregate[s] for s in stim_ref[keep]])

        result = forecast_market(X, y, n_splits=n_splits, task=task, seed=seed)
        result["n_features"] = float(X.shape[1])
        result["modalities"] = ",".join(modalities)  # type: ignore[assignment]
        results[arm_name] = result

    return results


def format_forecast_comparison(
    results: Dict[str, Dict[str, float]], task: str = "regression"
) -> str:
    """Render the arm comparison as a table."""
    if not results:
        return "no forecast arms produced results"
    if task == "regression":
        header = "  {:<18} {:>8} {:>10} {:>10} {:>10}".format(
            "arm", "n_stim", "oos_R2", "pearson_r", "p"
        )
        rows = [
            "  {:<18} {:>8.0f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                name,
                r["n_stimuli"],
                r["r2_out_of_sample"],
                r["pearson_r"],
                r["pearson_p"],
            )
            for name, r in sorted(results.items(), key=lambda kv: -kv[1]["r2_out_of_sample"])
        ]
    else:
        header = "  {:<18} {:>8} {:>10} {:>10}".format("arm", "n_stim", "bal_acc", "roc_auc")
        rows = [
            "  {:<18} {:>8.0f} {:>10.4f} {:>10.4f}".format(
                name, r["n_stimuli"], r["balanced_accuracy"], r["roc_auc"]
            )
            for name, r in sorted(results.items(), key=lambda kv: -kv[1]["roc_auc"])
        ]
    return "\n".join(["aggregate market forecast (held-out stimuli)", header] + rows)


# ------------------------------------------------- honest end-to-end evaluation


def cross_validate_ensemble(
    dataset: MultimodalDataset,
    reconciliation: str = "accuracy_weighted",
    n_splits_outer: int = 5,
    n_splits_inner: int = 4,
    seed: int = 0,
    modality_specs: Optional[Dict[str, Dict[str, object]]] = None,
    tune: bool = False,
    tune_grid: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Nested, subject-grouped evaluation of the full ensemble.

    The nesting is what makes the number trustworthy. Reconciliation weights are
    estimated by an inner CV run **inside each outer training set**, so the
    outer test subjects contribute nothing to the weights, the resampling, or
    the scaling. Fitting the ensemble once on everything and reporting its
    internal out-of-fold score would let weight selection see the test data.

    Returns a dict with pooled out-of-fold predictions, per-fold metrics, and
    the weight vector from each outer fold so weight stability can be inspected.
    """
    # Imported here rather than at module scope: models.ensemble imports
    # evaluation.cv for its out-of-fold weighting, so a top-level import would
    # close the cycle.
    from ..models.ensemble import MultimodalEnsemble

    y = np.asarray(dataset.y_individual)
    groups = np.asarray(dataset.subject_ids)

    n_subjects = len(np.unique(groups))
    outer = int(min(n_splits_outer, n_subjects))
    if outer < 2:
        raise ValueError(f"need at least 2 subjects for outer CV, got {n_subjects}")

    oof = np.full(len(y), np.nan)
    per_modality_oof: Dict[str, np.ndarray] = {
        m: np.full(len(y), np.nan) for m in dataset.modalities
    }
    fold_records: List[Dict[str, object]] = []

    dummy_X = np.zeros((len(y), 1))
    splits = list(iter_splits(dummy_X, y, groups, n_splits=outer, seed=seed))

    for fold, (train_idx, test_idx) in enumerate(
        progress(splits, desc="outer folds", total=len(splits))
    ):
        train_ds = _subset(dataset, train_idx)
        test_ds = _subset(dataset, test_idx)

        inner = int(min(n_splits_inner, len(np.unique(groups[train_idx]))))
        ensemble = MultimodalEnsemble(
            reconciliation=reconciliation,
            n_splits=max(2, inner),
            seed=seed,
            modality_specs=modality_specs,
            tune=tune,
            tune_grid=tune_grid,
        )
        # The ensemble's own inner CV (weights + any tuning) runs on train_ds
        # only, so tuning here is nested inside the outer fold, not leaked.
        ensemble.fit(train_ds)

        oof[test_idx] = ensemble.predict_proba(test_ds)[:, 1]
        for modality, proba in ensemble._modality_proba(test_ds).items():
            per_modality_oof[modality][test_idx] = proba

        fold_records.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_subjects": sorted(set(groups[test_idx].tolist())),
                "weights": dict(ensemble.weights_),
                "weights_degenerate": bool(getattr(ensemble, "weights_degenerate_", False)),
                "tuned_params": {
                    m: t.get("best_params", {})
                    for m, t in getattr(ensemble, "tuning_", {}).items()
                }
                if tune
                else {},
                "metrics": classification_report(y[test_idx], oof[test_idx])
                if len(np.unique(y[test_idx])) > 1
                else {"note": "single-class test fold; metrics undefined"},
            }
        )

    result: Dict[str, object] = {
        "reconciliation": reconciliation,
        "n_outer_folds": outer,
        "oof_proba": oof,
        "per_modality_oof": per_modality_oof,
        "folds": fold_records,
        "pooled_metrics": classification_report(y, oof),
        "per_modality_pooled": {
            m: classification_report(y, p) for m, p in per_modality_oof.items()
        },
        "weight_stability": _weight_stability(fold_records),
    }
    return result


def _weight_stability(fold_records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """Mean and spread of each modality's weight across outer folds.

    A modality whose weight swings between 0.0 and 0.6 across folds is not a
    reliably useful modality; it is a small sample talking. Report the spread,
    do not average it away.
    """
    all_weights: Dict[str, List[float]] = {}
    for record in fold_records:
        for modality, weight in record["weights"].items():  # type: ignore[union-attr]
            all_weights.setdefault(modality, []).append(float(weight))
    return {
        m: {
            "mean": float(np.mean(ws)),
            "sd": float(np.std(ws)),
            "min": float(np.min(ws)),
            "max": float(np.max(ws)),
        }
        for m, ws in all_weights.items()
    }


def _subset(dataset: MultimodalDataset, index: np.ndarray) -> MultimodalDataset:
    """Row-subset every block and outcome consistently."""
    return MultimodalDataset(
        blocks={name: block.select(index) for name, block in dataset.blocks.items()},
        y_individual=dataset.y_individual[index],
        subject_ids=dataset.subject_ids[index],
        stimulus_ids=dataset.stimulus_ids[index],
        y_aggregate=dataset.y_aggregate,
        metadata=dict(dataset.metadata),
    )
