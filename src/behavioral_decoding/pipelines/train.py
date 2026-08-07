"""End-to-end run: fit, evaluate at both levels, write a run record.

The order of operations is fixed and it matters. Individual-level nested CV
happens first, aggregate forecasting second, and both are written into the same
record with the config and the environment. Anything that reports one level
without the other is reporting half the claim.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..config import ExperimentConfig
from ..evaluation.metrics import bootstrap_ci, format_report, permutation_test
from ..evaluation.neuroforecast import (
    compare_forecast_arms,
    cross_validate_ensemble,
    format_forecast_comparison,
)
from ..io.base import MultimodalDataset
from ..models.ensemble import MultimodalEnsemble
from ..utils.logging import get_logger
from ..utils.seed import set_global_seed

logger = get_logger(__name__)


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - not a git checkout, or git missing
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return None


def _jsonable(obj: Any) -> Any:
    """Make numpy and dataclass values JSON-serialisable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return obj


def run_experiment(
    dataset: MultimodalDataset,
    config: Optional[ExperimentConfig] = None,
    write: bool = True,
) -> Dict[str, Any]:
    """Run the full two-level evaluation and return the run record."""
    cfg = config or ExperimentConfig()
    set_global_seed(cfg.seed)

    logger.info("experiment %s (seed %d)", cfg.name, cfg.seed)
    logger.info("\n%s", dataset.describe())

    record: Dict[str, Any] = {
        "name": cfg.name,
        "config": cfg.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
        },
        "data": {
            "n_trials": dataset.n_trials,
            "n_subjects": dataset.n_subjects,
            "n_stimuli": dataset.n_stimuli,
            "class_balance": dataset.class_balance(),
            "imbalance_ratio": dataset.imbalance_ratio(),
            "modalities": {
                m: {
                    "n_features": b.n_features,
                    "provenance": b.provenance,
                }
                for m, b in dataset.blocks.items()
            },
            "synthetic": bool(dataset.metadata.get("synthetic", False)),
        },
    }

    # ---------------------------------------------------------- individual level
    logger.info("individual level: nested subject-grouped CV")
    cv_result = cross_validate_ensemble(
        dataset,
        reconciliation=cfg.model.reconciliation,
        n_splits_outer=cfg.evaluation.n_splits_outer,
        n_splits_inner=cfg.evaluation.n_splits_inner,
        seed=cfg.seed,
        modality_specs=cfg.modality_specs(),
    )

    y = np.asarray(dataset.y_individual)
    oof = np.asarray(cv_result["oof_proba"])

    record["individual"] = {
        "pooled_metrics": cv_result["pooled_metrics"],
        "per_modality_pooled": cv_result["per_modality_pooled"],
        "weight_stability": cv_result["weight_stability"],
        "folds": cv_result["folds"],
        "bootstrap_ci": bootstrap_ci(
            y, oof, n_boot=cfg.evaluation.n_bootstrap, groups=dataset.subject_ids, seed=cfg.seed
        ),
        "permutation_test": permutation_test(
            y, oof, n_perm=cfg.evaluation.n_permutations, groups=dataset.subject_ids, seed=cfg.seed
        ),
    }

    logger.info(
        "\n%s", format_report(cv_result["pooled_metrics"], "individual choice (pooled OOF)")
    )

    # ----------------------------------------------------------- aggregate level
    if dataset.y_aggregate:
        logger.info("aggregate level: held-out market forecasting by arm")
        arms = compare_forecast_arms(
            dataset,
            task=cfg.evaluation.forecast_task,
            statistic=cfg.evaluation.aggregate_statistic,
            n_splits=cfg.evaluation.forecast_splits,
            seed=cfg.seed,
        )
        record["aggregate"] = arms
        logger.info("\n%s", format_forecast_comparison(arms, cfg.evaluation.forecast_task))
    else:
        record["aggregate"] = None
        logger.info(
            "no aggregate outcomes supplied; the market-forecasting arm was skipped. "
            "The individual-level result alone does not support a claim about "
            "population behaviour."
        )

    # -------------------------------------------------------------- final fit
    ensemble = MultimodalEnsemble(
        reconciliation=cfg.model.reconciliation,
        weight_metric=cfg.model.weight_metric,
        n_splits=cfg.evaluation.n_splits_inner,
        seed=cfg.seed,
        modality_specs=cfg.modality_specs(),
        drop_below_chance=cfg.model.drop_below_chance,
        weight_floor=cfg.model.weight_floor,
    ).fit(dataset)
    record["final_model"] = ensemble.to_record()
    logger.info("\n%s", ensemble.reconciliation_report())

    if write:
        out_dir = Path(cfg.output_dir) / cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "run_record.json"
        out_path.write_text(json.dumps(_jsonable(record), indent=2, sort_keys=True))
        logger.info("wrote run record to %s", out_path)
        record["output_path"] = str(out_path)

    return record
