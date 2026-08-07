"""NARPS (ds001734) loader: the mixed-gambles reward task.

NARPS is the fMRI counterpart to DEAP in this project. On each trial a
participant sees a 50/50 gamble with a possible gain and a possible loss and
decides whether to accept it. That is the anticipatory-affect paradigm the
framework's fMRI ROIs come from, so NARPS is the dataset to validate the fMRI
loader and the NAcc / vmPFC / AIns sphere extraction on real data.

See ``docs/narps.md`` for the verified format and the honest statement of what
NARPS can show. The one thing to keep in mind while reading this module: on
gambles the economic variables (gain, loss) forecast aggregate acceptance almost
by construction, so NARPS is an **individual-level** validation, not a
brain-beats-behaviour demonstration. The behaviour block here is a genuine, and
genuinely strong, comparator, not a strawman.

This file has two layers. This one is the format and bookkeeping layer, with no
neuroimaging dependencies: it parses ``events.tsv``, binarises the response,
builds the gamble stimulus keys, and assembles the behaviour block. The BOLD ->
ROI extraction lives in the ``.load`` path (see ``narps_bold`` /
:meth:`NARPSLoader.load`), which reuses :class:`FMRILoader` and needs nilearn.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logging import get_logger
from .base import BEHAVIOR, ModalityBlock

logger = get_logger(__name__)

# ------------------------------------------------------------------ constants

NARPS_TASK = "MGT"
NARPS_TR = 1.0
NARPS_TRIAL_DURATION_S = 4.0
NARPS_RUNS = (1, 2, 3, 4)
NARPS_TRIALS_PER_RUN = 64

EVENTS_COLUMNS = ("onset", "duration", "gain", "loss", "RT", "participant_response")

# The four graded responses plus the no-response marker, verbatim from the data.
ACCEPT_RESPONSES = ("strongly_accept", "weakly_accept")
REJECT_RESPONSES = ("strongly_reject", "weakly_reject")
NO_RESPONSE = "NoResp"
ALL_RESPONSES = ACCEPT_RESPONSES + REJECT_RESPONSES + (NO_RESPONSE,)

# Strength of the graded response, for anyone wanting a confidence feature or a
# 4-level analysis. The binary accept is the default outcome.
RESPONSE_CONFIDENCE = {
    "strongly_accept": 1.0,
    "weakly_accept": 0.5,
    "weakly_reject": 0.5,
    "strongly_reject": 1.0,
}

GROUPS = ("equalIndifference", "equalRange")


class NARPSFormatError(ValueError):
    """Raised when a file does not match the documented NARPS layout.

    Loud on purpose: an events file with unexpected columns or response strings
    is a different release or a corrupted download, and both produce wrong
    labels rather than crashes if waved through.
    """


# ------------------------------------------------------------- gamble keys


def gamble_key(gain: float, loss: float) -> str:
    """Stable stimulus id for a gamble, e.g. ``"g14_l06"``.

    The gamble is the stimulus, and ``(gain, loss)`` pairs recur across subjects,
    which is the only thing that makes an aggregate outcome possible. Integer
    coding matches the dataset, where gains and losses are whole currency units.
    """
    return f"g{int(round(gain)):02d}_l{int(round(loss)):02d}"


def expected_value(gain: float, loss: float) -> float:
    """Expected value of a 50/50 gain/loss gamble: ``0.5 * (gain - loss)``."""
    return 0.5 * (float(gain) - float(loss))


# ------------------------------------------------------------ events parsing


def _read_events_table(path_or_table: Any) -> Any:
    """Return a pandas DataFrame from a path or an existing frame."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reading NARPS events requires pandas (`pip install pandas`)"
        ) from exc

    if isinstance(path_or_table, pd.DataFrame):
        return path_or_table.copy()
    return pd.read_csv(path_or_table, sep="\t")


def parse_events(
    path_or_table: Any,
    drop_no_response: bool = True,
) -> Any:
    """Parse one ``*_events.tsv`` into a tidy, framework-ready table.

    Adds three derived columns and validates the response strings. Returns a
    DataFrame with, at minimum: ``onset``, ``gain``, ``loss``, ``RT``,
    ``participant_response``, ``accept`` (0/1), ``confidence``, ``stimulus_id``,
    ``expected_value``.

    Parameters
    ----------
    drop_no_response:
        Drop ``NoResp`` trials. On by default. A missing response is missing
        data; imputing it would manufacture a label, and its ``RT`` of 0 would
        poison any RT feature.
    """
    table = _read_events_table(path_or_table)

    missing = [c for c in EVENTS_COLUMNS if c not in table.columns]
    if missing:
        raise NARPSFormatError(
            f"events file is missing columns {missing}; "
            f"found {list(table.columns)}. Expected the NARPS mixed-gambles "
            f"layout {list(EVENTS_COLUMNS)}."
        )

    responses = set(table["participant_response"].dropna().unique())
    unknown = responses - set(ALL_RESPONSES)
    if unknown:
        raise NARPSFormatError(
            f"unrecognised participant_response values {sorted(unknown)}. "
            f"Expected a subset of {list(ALL_RESPONSES)}. A different coding "
            "means the accept/reject mapping here does not apply."
        )

    n_before = len(table)
    if drop_no_response:
        table = table[table["participant_response"] != NO_RESPONSE].copy()
        n_dropped = n_before - len(table)
        if n_dropped:
            logger.info(
                "NARPS: dropped %d/%d NoResp trials (RT=0; missing data, not a fast "
                "response)",
                n_dropped,
                n_before,
            )

    table["accept"] = table["participant_response"].isin(ACCEPT_RESPONSES).astype(int)
    table["confidence"] = table["participant_response"].map(RESPONSE_CONFIDENCE).astype(float)
    table["expected_value"] = 0.5 * (table["gain"].astype(float) - table["loss"].astype(float))
    table["stimulus_id"] = [
        gamble_key(gain, loss)
        for gain, loss in zip(table["gain"], table["loss"])
    ]
    return table.reset_index(drop=True)


def subject_id_from_path(path: Path) -> str:
    """``.../sub-014_task-MGT_run-01_events.tsv`` -> ``"sub-014"``."""
    match = re.search(r"(sub-[A-Za-z0-9]+)", path.name)
    if not match:
        raise NARPSFormatError(
            f"{path.name!r} does not contain a BIDS subject label like 'sub-014'"
        )
    return match.group(1)


def load_participants(path: str) -> Any:
    """Read ``participants.tsv``, normalising and validating the group column."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reading NARPS metadata requires pandas") from exc

    table = pd.read_csv(path, sep="\t")
    if "participant_id" not in table.columns or "group" not in table.columns:
        raise NARPSFormatError(
            "participants.tsv needs participant_id and group columns; found "
            f"{list(table.columns)}"
        )
    unknown = set(table["group"].dropna().unique()) - set(GROUPS)
    if unknown:
        logger.warning(
            "NARPS: participants.tsv has unexpected group values %s (expected %s)",
            sorted(unknown),
            list(GROUPS),
        )
    return table


# ------------------------------------------------------------- block building


class NARPSLoader:
    """Assemble NARPS into framework blocks.

    The format/behaviour layer (this class's ``from_events_and_rois`` and
    ``behavior_block``) has no neuroimaging dependencies. The BOLD -> ROI
    extraction is in :meth:`load`, which reuses :class:`FMRILoader`.
    """

    name = "narps"

    def __init__(
        self,
        include_rt: bool = True,
        include_expected_value: bool = True,
        standardize_fmri: bool = False,
    ) -> None:
        self.include_rt = include_rt
        self.include_expected_value = include_expected_value
        self.standardize_fmri = standardize_fmri

    # -- behaviour: the economic comparator ---------------------------------

    def behavior_block(
        self,
        events: Any,
        subject_id: str,
    ) -> ModalityBlock:
        """Build the behaviour block from parsed events.

        Features are the economic variables that drive gamble choice: gain, loss,
        expected value, and (optionally) RT. On NARPS this is a *strong*
        comparator by design, not a strawman: acceptance is largely a function of
        gain and loss, so an economic model forecasts choice well. The neural
        arm has to beat this, and on the aggregate arm it usually will not (see
        ``docs/narps.md``). ``accept`` is never a feature.
        """
        columns: List[np.ndarray] = [
            events["gain"].to_numpy(dtype=float),
            events["loss"].to_numpy(dtype=float),
        ]
        names = ["gain", "loss"]

        if self.include_expected_value:
            columns.append(events["expected_value"].to_numpy(dtype=float))
            names.append("expected_value")
            # |EV| separates "clearly good/bad" gambles from ambiguous ones near
            # indifference, which is where choice is hardest and most variable.
            columns.append(np.abs(events["expected_value"].to_numpy(dtype=float)))
            names.append("abs_expected_value")

        if self.include_rt:
            columns.append(events["RT"].to_numpy(dtype=float))
            names.append("RT")

        return ModalityBlock(
            name=BEHAVIOR,
            X=np.column_stack(columns),
            subject_ids=np.array([subject_id] * len(events)),
            stimulus_ids=events["stimulus_id"].to_numpy(),
            feature_names=names,
            provenance={
                "loader": "NARPSLoader",
                "modality": BEHAVIOR,
                "source": "narps_events",
                "features": names,
                "note": (
                    "economic comparator (gain/loss/EV); strong by design on "
                    "gambles, not a strawman. See docs/narps.md."
                ),
            },
        )

    def from_events_and_rois(
        self,
        events: Any,
        roi_features: np.ndarray,
        subject_id: str,
        roi_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, ModalityBlock]:
        """Build fMRI + behaviour blocks from parsed events and extracted ROIs.

        This is the nilearn-free entry point: hand it a trial-by-ROI matrix you
        extracted however you like (a GLM, a masker, precomputed betas) and it
        assembles the blocks with correct keys. It is also what :meth:`load`
        calls after doing the extraction.

        ``roi_features`` must be ``(n_trials, n_rois)`` and row-aligned with
        ``events`` (same order, same length).
        """
        from .fmri import FMRILoader

        roi_features = np.asarray(roi_features, dtype=float)
        if roi_features.shape[0] != len(events):
            raise ValueError(
                f"roi_features has {roi_features.shape[0]} rows but events has "
                f"{len(events)}; they must be row-aligned trial for trial"
            )

        fmri_block = FMRILoader(standardize=self.standardize_fmri).from_arrays(
            X=roi_features,
            subject_ids=[subject_id] * len(events),
            stimulus_ids=events["stimulus_id"].to_numpy(),
            feature_names=list(roi_names) if roi_names is not None else None,
            source="narps_rois",
        )
        return {
            "fmri": fmri_block,
            BEHAVIOR: self.behavior_block(events, subject_id),
        }


# ------------------------------------------------------------ aggregate outcome


def acceptance_rate_by_gamble(
    events_by_subject: Sequence[Any],
    min_subjects: int = 5,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Population acceptance rate per gamble, the NARPS aggregate outcome.

    This is a genuine aggregate choice: the fraction of people who accepted each
    ``(gain, loss)`` gamble. It is keyed the way the blocks are, so it drops
    straight into ``y_aggregate``.

    A gamble seen by only one or two subjects has a rate that is mostly noise, so
    gambles below ``min_subjects`` are excluded and the count is reported.

    Returns ``(y_aggregate, provenance)``.
    """
    import pandas as pd

    combined = pd.concat(list(events_by_subject), ignore_index=True)
    grouped = combined.groupby("stimulus_id")["accept"]
    counts = grouped.count()
    rates = grouped.mean()

    kept = counts[counts >= min_subjects].index
    y_aggregate = {str(k): float(rates[k]) for k in kept}
    dropped = len(rates) - len(kept)

    if dropped:
        logger.info(
            "NARPS: %d/%d gambles seen by fewer than %d subjects; excluded from "
            "the aggregate arm",
            dropped,
            len(rates),
            min_subjects,
        )

    provenance = {
        "outcome": "population_acceptance_rate",
        "n_gambles": len(y_aggregate),
        "n_gambles_dropped": int(dropped),
        "min_subjects": min_subjects,
        "caveat": (
            "on gambles the economic baseline (gain/loss) forecasts this rate "
            "almost by construction; brain is not expected to beat behaviour on "
            "this arm. See docs/narps.md."
        ),
    }
    return y_aggregate, provenance
