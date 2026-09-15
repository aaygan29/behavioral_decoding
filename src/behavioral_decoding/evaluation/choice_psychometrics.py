"""Logistic psychometric curves for behavioral choice readouts.

Fits ``P(choice) = sigmoid(bias + slope * evidence [+ lapse])`` per condition
or per subject, and reports bias and sensitivity as two separate numbers with
bootstrap confidence intervals, plus a helper that tests whether two
manipulations combine additively or interact.

This is grounded in a mouse superior-colliculus (SC) optogenetics result:

    Takacs et al., Carandini M. (2026). "The role of superior colliculus in a
    logistic decision." bioRxiv. doi:10.64898/2026.06.05.730072 (v2).

SC inactivation shifted choice bias contralaterally with no change in
sensitivity, and the effect of bilateral inactivation was additive; choice was
well fit by a logistic model with a weighted sum of evidence and bias terms.
The finding licenses two evaluation habits used here: (1) a behavioral readout
should be scored on bias and sensitivity *separately*, since a manipulation
that only shifts the intercept looks unremarkable in a single collapsed
accuracy number and can be missed entirely; (2) two manipulations should be
checked for additivity in log-odds space before either is interpreted as
sufficient on its own, using an interaction term in a logistic GLM.

Nothing here changes an existing pipeline or metric; these are additive
evaluation utilities for behavioral (choice) data as an extra readout
alongside the existing classification metrics in :mod:`.metrics`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_psychometric(
    evidence: np.ndarray,
    choice: np.ndarray,
    lapse: bool = False,
    max_lapse: float = 0.2,
) -> Dict[str, float]:
    """Fit a one-dimensional logistic psychometric curve.

    ``P(choice=1) = sigmoid(bias + slope * evidence)``, or, with
    ``lapse=True``, ``P(choice=1) = gamma + (1 - 2*gamma) * sigmoid(bias +
    slope * evidence)`` where ``gamma`` is a symmetric lapse rate bounded by
    ``max_lapse``. The lapse term keeps a handful of stimulus-independent
    errors (inattention, motor slips) from dragging the fitted slope down,
    which otherwise reads as reduced sensitivity when nothing about the
    evidence-to-choice mapping changed.

    Parameters
    ----------
    evidence:
        1D array of the signed evidence / stimulus strength per trial.
    choice:
        1D array of binary choices (0/1) per trial.
    lapse:
        If True, fit the two-parameter lapse-rate extension by a small grid
        search over ``gamma`` in ``[0, max_lapse]``, refitting the logistic
        core at each grid point and keeping the best log-likelihood.
    max_lapse:
        Upper bound on the symmetric lapse rate.

    Returns
    -------
    dict with ``bias`` (intercept, log-odds units), ``slope`` (sensitivity to
    evidence, log-odds per unit evidence), ``lapse`` (0.0 unless
    ``lapse=True``), and ``log_likelihood``.
    """
    evidence = np.asarray(evidence, dtype=float).reshape(-1, 1)
    choice = np.asarray(choice, dtype=float)
    if len(np.unique(choice)) < 2:
        return {
            "bias": float("nan"),
            "slope": float("nan"),
            "lapse": float("nan"),
            "log_likelihood": float("nan"),
        }

    def _fit_core(y: np.ndarray) -> LogisticRegression:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(evidence, y)
        return model

    if not lapse:
        model = _fit_core(choice)
        bias = float(model.intercept_[0])
        slope = float(model.coef_[0, 0])
        p = np.clip(model.predict_proba(evidence)[:, 1], 1e-9, 1 - 1e-9)
        ll = float(np.sum(choice * np.log(p) + (1 - choice) * np.log(1 - p)))
        return {"bias": bias, "slope": slope, "lapse": 0.0, "log_likelihood": ll}

    best = None
    for gamma in np.linspace(0.0, max_lapse, 11):
        model = _fit_core(choice)
        raw_p = _sigmoid(model.intercept_[0] + model.coef_[0, 0] * evidence.ravel())
        p = gamma + (1 - 2 * gamma) * raw_p
        p = np.clip(p, 1e-9, 1 - 1e-9)
        ll = float(np.sum(choice * np.log(p) + (1 - choice) * np.log(1 - p)))
        if best is None or ll > best["log_likelihood"]:
            best = {
                "bias": float(model.intercept_[0]),
                "slope": float(model.coef_[0, 0]),
                "lapse": float(gamma),
                "log_likelihood": ll,
            }
    assert best is not None
    return best


def bias_sensitivity_ci(
    evidence: np.ndarray,
    choice: np.ndarray,
    groups: Optional[np.ndarray] = None,
    lapse: bool = False,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap CIs for bias and sensitivity, reported as two separate metrics.

    Following Takacs et al. (2026), a manipulation's effect on choice should
    not be collapsed into one accuracy number: a pure bias shift and a pure
    sensitivity change are different phenomena with different mechanistic
    implications, and averaging them together can hide a real, large bias
    shift behind an unchanged overall accuracy.

    Pass ``groups`` (e.g. subject id) to resample groups rather than trials,
    which is the right unit when trials are nested in subjects.
    """
    evidence = np.asarray(evidence, dtype=float)
    choice = np.asarray(choice, dtype=float)
    rng = np.random.default_rng(seed)
    point = fit_psychometric(evidence, choice, lapse=lapse)

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        group_rows = {g: np.flatnonzero(groups == g) for g in unique_groups}

    biases, slopes = [], []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(choice), len(choice))
        else:
            drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([group_rows[g] for g in drawn])
        if len(np.unique(choice[idx])) < 2:
            continue
        fit = fit_psychometric(evidence[idx], choice[idx], lapse=lapse)
        if np.isnan(fit["bias"]) or np.isnan(fit["slope"]):
            continue
        biases.append(fit["bias"])
        slopes.append(fit["slope"])

    def _summarize(point_val: float, boots: list) -> Dict[str, float]:
        if len(boots) < n_boot * 0.5:
            return {
                "point": point_val,
                "lo": float("nan"),
                "hi": float("nan"),
                "n_valid_boot": float(len(boots)),
            }
        arr = np.asarray(boots)
        return {
            "point": point_val,
            "lo": float(np.percentile(arr, 100 * alpha / 2)),
            "hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
            "n_valid_boot": float(len(boots)),
        }

    return {
        "bias": _summarize(point["bias"], biases),
        "slope": _summarize(point["slope"], slopes),
        "lapse": point["lapse"],
    }


def additivity_test(
    evidence: np.ndarray,
    choice: np.ndarray,
    manip_a: np.ndarray,
    manip_b: np.ndarray,
) -> Dict[str, float]:
    """Test whether two binary manipulations combine additively in log-odds.

    Fits a logistic GLM with main effects and an interaction term:

        logit P(choice=1) = b0 + b_e * evidence + b_a * manip_a
                             + b_b * manip_b + b_ab * (manip_a * manip_b)

    and returns the fitted interaction coefficient ``b_ab`` with a Wald
    z-statistic and two-sided p-value, following the additive-effect test in
    Takacs et al. (2026), where bilateral SC inactivation was additive rather
    than interacting. A near-zero, non-significant interaction supports
    additivity; a large, significant one indicates the manipulations interact
    (e.g. saturation or synergy) rather than summing independently.

    ``manip_a`` and ``manip_b`` must be 0/1 indicator arrays, same length as
    ``evidence`` and ``choice``.
    """
    evidence = np.asarray(evidence, dtype=float)
    choice = np.asarray(choice, dtype=float)
    manip_a = np.asarray(manip_a, dtype=float)
    manip_b = np.asarray(manip_b, dtype=float)

    interaction = manip_a * manip_b
    design = np.column_stack([evidence, manip_a, manip_b, interaction])

    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    model.fit(design, choice)
    coefs = model.coef_[0]
    b_ab = float(coefs[3])

    # Wald standard error from the inverse Fisher information of the fitted
    # logistic model (design matrix with intercept column prepended).
    X = np.column_stack([np.ones(len(choice)), design])
    p = _sigmoid(X @ np.concatenate([[model.intercept_[0]], coefs]))
    w = np.clip(p * (1 - p), 1e-9, None)
    fisher = (X * w[:, None]).T @ X
    try:
        cov = np.linalg.inv(fisher)
        se_ab = float(np.sqrt(cov[4, 4]))
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        se_ab = float("nan")

    if se_ab and not np.isnan(se_ab) and se_ab > 0:
        z = b_ab / se_ab
        # Two-sided normal-approximation p-value without importing scipy.stats
        # here; the erf-based formula matches scipy to float precision.
        from math import erf

        p_value = float(2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2)))))
    else:  # pragma: no cover - defensive
        z = float("nan")
        p_value = float("nan")

    return {
        "interaction_coef": b_ab,
        "interaction_se": se_ab,
        "z": float(z),
        "p_value": p_value,
        "bias_a": float(coefs[1]),
        "bias_b": float(coefs[2]),
        "slope_evidence": float(coefs[0]),
    }
