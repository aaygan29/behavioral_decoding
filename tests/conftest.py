from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behavioral_decoding.synthetic import SyntheticConfig, make_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="session")
def small_dataset():
    """A small synthetic dataset, shared across tests to keep the suite fast."""
    dataset, ground_truth = make_synthetic_dataset(
        SyntheticConfig(n_subjects=10, n_stimuli=25, seed=0)
    )
    return dataset, ground_truth
