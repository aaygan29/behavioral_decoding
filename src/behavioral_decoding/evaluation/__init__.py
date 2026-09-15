from __future__ import annotations

from .choice_psychometrics import additivity_test, bias_sensitivity_ci, fit_psychometric
from .cv import describe_splits, iter_splits, out_of_fold_proba, subject_splitter
from .decoding_geometry import (
    cross_condition_generalization,
    subspace_alignment,
    variance_decomposition,
)
from .metrics import bootstrap_ci, classification_report, format_report, permutation_test
from .neuroforecast import (
    aggregate_block,
    aggregate_by_stimulus,
    compare_forecast_arms,
    cross_validate_ensemble,
    forecast_market,
    format_forecast_comparison,
)

__all__ = [
    "additivity_test",
    "aggregate_block",
    "aggregate_by_stimulus",
    "bias_sensitivity_ci",
    "bootstrap_ci",
    "classification_report",
    "compare_forecast_arms",
    "cross_condition_generalization",
    "cross_validate_ensemble",
    "describe_splits",
    "fit_psychometric",
    "forecast_market",
    "format_forecast_comparison",
    "format_report",
    "iter_splits",
    "out_of_fold_proba",
    "permutation_test",
    "subject_splitter",
    "subspace_alignment",
    "variance_decomposition",
]
