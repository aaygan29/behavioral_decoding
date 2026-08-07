"""Experiment configuration.

Every knob that changes a number lives in a config file, and the resolved config
is written into the run record next to the results. A result whose configuration
is not recorded cannot be reproduced, and a result that cannot be reproduced is
not a result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    """Where the data is and how trials are keyed."""

    modalities: List[str] = field(default_factory=lambda: ["fmri", "eeg", "face", "behavior"])
    root: str = "data/raw"
    align_how: str = "inner"
    subject_column: str = "subject_id"
    stimulus_column: str = "stimulus_id"
    outcome_column: str = "choice"
    aggregate_outcome_path: Optional[str] = None


@dataclass
class BalanceConfig:
    """Class-imbalance handling. ``sampler=null`` means auto-select."""

    sampler: Optional[str] = None
    k_neighbors: Optional[int] = None
    auto_recommend: bool = True


@dataclass
class ModelConfig:
    """Per-modality learners and the reconciliation rule."""

    reconciliation: str = "accuracy_weighted"
    weight_metric: str = "balanced_accuracy"
    drop_below_chance: bool = True
    weight_floor: float = 0.0
    base_learner: Dict[str, str] = field(default_factory=dict)
    n_bags: Dict[str, int] = field(default_factory=dict)
    max_samples: float = 0.8
    max_features: float = 1.0


@dataclass
class EvalConfig:
    n_splits_outer: int = 5
    n_splits_inner: int = 4
    n_bootstrap: int = 2000
    n_permutations: int = 1000
    forecast_task: str = "regression"
    forecast_splits: int = 5
    aggregate_statistic: str = "mean"


@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 0
    output_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def modality_specs(self) -> Dict[str, Dict[str, Any]]:
        """Translate the model config into per-modality kwargs for the ensemble."""
        specs: Dict[str, Dict[str, Any]] = {}
        for modality in self.data.modalities:
            spec: Dict[str, Any] = {
                "seed": self.seed,
                "max_samples": self.model.max_samples,
                "max_features": self.model.max_features,
            }
            if modality in self.model.base_learner:
                spec["base_learner"] = self.model.base_learner[modality]
            if modality in self.model.n_bags:
                spec["n_bags"] = self.model.n_bags[modality]
            if not self.balance.auto_recommend:
                spec["sampler"] = self.balance.sampler or "smote"
                spec["k_neighbors"] = self.balance.k_neighbors or 5
            specs[modality] = spec
        return specs

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> ExperimentConfig:
        payload = dict(payload)
        return cls(
            name=payload.get("name", "default"),
            seed=int(payload.get("seed", 0)),
            output_dir=payload.get("output_dir", "results"),
            data=DataConfig(**payload.get("data", {})),
            balance=BalanceConfig(**payload.get("balance", {})),
            model=ModelConfig(**payload.get("model", {})),
            evaluation=EvalConfig(**payload.get("evaluation", {})),
        )

    @classmethod
    def load(cls, path: str) -> ExperimentConfig:
        """Load from YAML or JSON."""
        p = Path(path)
        text = p.read_text()
        if p.suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "YAML configs need PyYAML (`pip install pyyaml`), or use a .json config"
                ) from exc
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        return cls.from_dict(payload or {})

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
