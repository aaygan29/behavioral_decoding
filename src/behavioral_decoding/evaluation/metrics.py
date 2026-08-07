"""Metrics, chosen so a class-imbalanced problem cannot flatter itself.

Plain accuracy on an 85/15 split is 0.85 for a model that always says "no". Every
classification summary here leads with balanced accuracy and average precision,
and reports the majority-class baseline alongside so the comparison is
unavoidable.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def classification_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Summarise a binary classifier from predicted positive-class probabilities."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    classes, counts = np.unique(y_true, return_counts=True)
    majority_rate = float(counts.max()) / len(y_true)
    positive_rate = float(np.mean(y_true == classes[-1]))

    out: Dict[str, float] = {
        "n": float(len(y_true)),
        "positive_rate": positive_rate,
        "majority_baseline_accuracy": majority_rate,
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }

    if len(classes) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["average_precision"] = float(average_precision_score(y_true, y_proba))
        # Average precision floors at the positive rate, not at 0.5. Reporting
        # the lift makes an AP of 0.30 on a 28% positive rate read correctly.
        out["ap_lift_over_chance"] = out["average_precision"] - positive_rate
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
        out["ap_lift_over_chance"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(
        {
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        }
    )
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "balanced_accuracy",
    n_boot: int = 2000,
    alpha: float = 0.05,
    groups: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap CI for one metric.

    Pass ``groups`` (subject ids) to resample *subjects* rather than trials.
    Trial-level resampling treats 200 trials from 20 subjects as 200 independent
    observations, which they are not, and returns an interval that is far too
    narrow.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    rng = np.random.default_rng(seed)

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        group_rows = {g: np.flatnonzero(groups == g) for g in unique_groups}

    stats = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(y_true), len(y_true))
        else:
            drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([group_rows[g] for g in drawn])
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(classification_report(y_true[idx], y_proba[idx])[metric])

    if len(stats) < n_boot * 0.5:
        # Too many degenerate resamples to trust the interval. Say so instead of
        # returning a confident-looking number.
        return {
            "metric": metric,
            "point": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "n_valid_boot": float(len(stats)),
            "warning": "over half of bootstrap resamples were single-class; "
            "the sample is too small or too imbalanced for a stable interval",
        }

    stats_arr = np.asarray(stats)
    return {
        "metric": metric,
        "point": classification_report(y_true, y_proba)[metric],
        "lo": float(np.percentile(stats_arr, 100 * alpha / 2)),
        "hi": float(np.percentile(stats_arr, 100 * (1 - alpha / 2))),
        "n_valid_boot": float(len(stats)),
    }


def permutation_test(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "balanced_accuracy",
    n_perm: int = 1000,
    groups: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Label-permutation null. Shuffles within subject when ``groups`` is given."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    rng = np.random.default_rng(seed)
    observed = classification_report(y_true, y_proba)[metric]

    null = []
    for _ in range(n_perm):
        permuted = y_true.copy()
        if groups is None:
            rng.shuffle(permuted)
        else:
            for g in np.unique(groups):
                rows = np.flatnonzero(np.asarray(groups) == g)
                permuted[rows] = rng.permutation(permuted[rows])
        if len(np.unique(permuted)) < 2:
            continue
        null.append(classification_report(permuted, y_proba)[metric])

    null_arr = np.asarray(null)
    # +1 in numerator and denominator: an exact permutation p can never be 0.
    p = (np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1)
    return {
        "metric": metric,
        "observed": float(observed),
        "null_mean": float(null_arr.mean()),
        "null_sd": float(null_arr.std()),
        "p_value": float(p),
        "n_perm": float(len(null_arr)),
    }


def format_report(report: Dict[str, float], title: str = "") -> str:
    """Pretty-print a metrics dict."""
    order = [
        "n",
        "positive_rate",
        "majority_baseline_accuracy",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "ap_lift_over_chance",
        "f1",
        "mcc",
        "brier",
        "sensitivity",
        "specificity",
    ]
    lines = [title] if title else []
    for key in order:
        if key in report:
            lines.append(f"  {key:<28} {report[key]:>8.4f}")
    return "\n".join(lines)
