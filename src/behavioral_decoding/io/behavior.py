"""Behavioural and self-report loading.

This block is the benchmark the neural blocks have to beat, not an afterthought.
The headline claim in the neuroforecasting literature is comparative: in
Genevsky, Yoon and Knutson (2017) the behavioural measures from the scanned
sample did *not* forecast market funding outcomes while NAcc activity did, and
in Falk, Berkman and Lieberman (2012) self-reported campaign effectiveness
failed where MPFC activity succeeded. Those claims only mean something if the
behavioural arm is modelled as carefully as the neural arm.

So: keep ratings, response times, and choice-derived features here, run them
through the same pipeline, and report the behaviour-only forecast next to the
brain-only forecast every time.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import BEHAVIOR, BaseLoader, ModalityBlock

# Response time is right-skewed and its tail is mostly attention lapses rather
# than valuation. Log-transform by default, and record that we did.
DEFAULT_LOG_COLUMNS = ("rt", "response_time", "reaction_time", "duration")


class BehaviorLoader(BaseLoader):
    """Load trial-level ratings, response times, and derived choice features."""

    name = BEHAVIOR

    def __init__(
        self,
        log_transform: Sequence[str] = DEFAULT_LOG_COLUMNS,
        drop_columns: Sequence[str] = (),
    ) -> None:
        self.log_transform = tuple(log_transform)
        self.drop_columns = tuple(drop_columns)

    def from_arrays(
        self,
        X: np.ndarray,
        subject_ids: Sequence,
        stimulus_ids: Sequence,
        feature_names: Optional[List[str]] = None,
        source: str = "arrays",
    ) -> ModalityBlock:
        return ModalityBlock(
            name=self.name,
            X=np.asarray(X, dtype=float),
            subject_ids=np.asarray(subject_ids),
            stimulus_ids=np.asarray(stimulus_ids),
            feature_names=feature_names,
            provenance=self._provenance(source=source),
        )

    def load(
        self,
        table: object,
        subject_column: str = "subject_id",
        stimulus_column: str = "stimulus_id",
        feature_columns: Optional[Sequence[str]] = None,
        outcome_column: Optional[str] = None,
    ) -> ModalityBlock:
        """Build a block from a tidy trial-level table.

        Parameters
        ----------
        table:
            A pandas DataFrame with one row per trial.
        outcome_column:
            The individual choice column. It is *excluded* from the feature
            matrix. Leaving the outcome in the features is the single most
            common way to produce an accuracy of 1.0 that means nothing.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "BehaviorLoader.load requires pandas. Install with `pip install '.[data]'`, "
                "or use BehaviorLoader.from_arrays."
            ) from exc

        if not isinstance(table, pd.DataFrame):
            table = pd.DataFrame(table)

        reserved = {subject_column, stimulus_column}
        if outcome_column:
            reserved.add(outcome_column)
        reserved.update(self.drop_columns)

        if feature_columns is None:
            feature_columns = [
                c
                for c in table.columns
                if c not in reserved and np.issubdtype(table[c].dtype, np.number)
            ]
        else:
            overlap = reserved.intersection(feature_columns)
            if overlap:
                raise ValueError(
                    "these columns are outcome or key columns and must not be used as "
                    f"features: {sorted(overlap)}"
                )

        if not feature_columns:
            raise ValueError("no numeric feature columns found in the behaviour table")

        frame = table[list(feature_columns)].astype(float).copy()
        logged: List[str] = []
        for col in frame.columns:
            if col.lower() in self.log_transform:
                frame[col] = np.log1p(np.clip(frame[col].to_numpy(), 0.0, None))
                logged.append(col)

        return ModalityBlock(
            name=self.name,
            X=frame.to_numpy(dtype=float),
            subject_ids=table[subject_column].to_numpy(),
            stimulus_ids=table[stimulus_column].to_numpy(),
            feature_names=list(frame.columns),
            provenance=self._provenance(
                source="dataframe",
                n_features=len(frame.columns),
                log_transformed=logged,
                excluded_outcome=outcome_column,
            ),
        )
