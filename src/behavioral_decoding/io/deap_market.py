"""Turning DEAP's stimuli into a real population-level outcome.

DEAP ships no market outcome. But its 40 stimuli are real, public music videos,
and `video_list.csv` carries their YouTube links. That is the same move Genevsky
and Knutson made: show items that exist in a public market, then pull the
market's response to them from outside the lab. A YouTube view count is a
genuine aggregate behaviour, measured on a population that never entered the
scanner. That is what turns a DEAP analysis from affect decoding into
neuroforecasting.

This module does the bookkeeping and nothing else. It does **not** fetch view
counts. Scraping YouTube or hitting its Data API is a network operation with
terms-of-service and rate-limit implications, and it does not belong inside a
loader. The workflow is:

1. `youtube_ids_from_video_list` extracts the video ids from `video_list.csv`.
2. You fetch view counts however you are permitted to (the YouTube Data API's
   `videos.list` with `part=statistics` is the sanctioned route), and save them.
3. `market_outcome_from_counts` turns that mapping into a `y_aggregate` dict
   keyed the way `load_deap` keys its stimuli.

## Two things that will bite

**View counts are measured years after DEAP was recorded.** DEAP is from 2012.
A 2026 view count reflects a decade of accumulation, virality, channel deletion,
and re-uploads, none of which the 2012 EEG could have anticipated. This is a
real confound, not a technicality. The honest version of this analysis either
finds an archived contemporaneous count or frames the outcome as "durable
popularity" and says so. `market_outcome_from_counts` stamps the fetch date into
provenance so the gap is on the record.

**View counts are extremely skewed and need a log.** One viral video with 10^9
views next to clips with 10^4 will dominate any linear model and any Pearson
correlation. `log_transform=True` is the default, and the forecasting arm should
report Spearman alongside Pearson because rank is what survives the skew.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.logging import get_logger
from .deap import DEAP_N_TRIALS, load_video_list

logger = get_logger(__name__)

# Matches a YouTube id in the URL forms DEAP's link column uses:
# watch?v=ID, youtu.be/ID, /embed/ID, /v/ID.
_YOUTUBE_ID = re.compile(
    r"(?:v=|youtu\.be/|/embed/|/v/)([A-Za-z0-9_-]{11})"
)


def _experiment_key(experiment_id: int) -> str:
    """Match the ``exp-NN`` stimulus keys that ``load_deap`` produces."""
    return f"exp-{int(experiment_id):02d}"


def extract_youtube_id(url: str) -> Optional[str]:
    """Pull the 11-character video id out of a YouTube URL, or return ``None``."""
    if not isinstance(url, str):
        return None
    match = _YOUTUBE_ID.search(url)
    return match.group(1) if match else None


def youtube_ids_from_video_list(
    video_list_path: str,
    link_column: Optional[str] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    """Map each stimulus key to its YouTube id and source URL.

    Returns ``{exp_key: {"youtube_id": id_or_None, "url": url}}`` for all 40
    stimuli. A ``None`` id means the link could not be parsed (a dead format or a
    non-YouTube host), which is surfaced rather than silently dropped so you know
    exactly how many stimuli the aggregate arm actually covers.
    """
    table = load_video_list(video_list_path)

    if link_column is None:
        candidates = [c for c in table.columns if "youtube" in c or "link" in c or "url" in c]
        if not candidates:
            raise ValueError(
                f"could not find a link column in video_list.csv; "
                f"columns are {sorted(table.columns)}. "
                "Pass link_column explicitly."
            )
        link_column = candidates[0]

    out: Dict[str, Dict[str, Optional[str]]] = {}
    n_missing = 0
    for _, row in table.iterrows():
        key = _experiment_key(row["experiment_id"])
        url = row[link_column]
        vid = extract_youtube_id(url)
        if vid is None:
            n_missing += 1
        out[key] = {"youtube_id": vid, "url": url if isinstance(url, str) else None}

    if n_missing:
        logger.warning(
            "deap_market: %d/%d stimuli have no parseable YouTube id; those "
            "stimuli cannot enter the aggregate forecasting arm",
            n_missing,
            len(out),
        )
    return out


def market_outcome_from_counts(
    counts: Dict[str, float],
    id_to_experiment: Optional[Dict[str, str]] = None,
    log_transform: bool = True,
    fetch_date: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Build a ``y_aggregate`` dict from view counts.

    Parameters
    ----------
    counts:
        View counts, keyed either by ``exp-NN`` stimulus id or by raw YouTube id.
        If keyed by YouTube id, supply ``id_to_experiment``.
    id_to_experiment:
        Maps YouTube id to ``exp-NN`` key. Build it by inverting
        :func:`youtube_ids_from_video_list`.
    log_transform:
        ``log1p`` the counts. On by default because raw view counts are too
        skewed for a linear forecast or a Pearson correlation to mean anything.
    fetch_date:
        ISO date the counts were retrieved. Recorded in provenance. Omitting it
        is allowed but discouraged: without it the temporal gap between the 2012
        recordings and the counts is invisible.

    Returns
    -------
    (y_aggregate, provenance)
        ``y_aggregate`` is keyed by ``exp-NN`` and ready to pass to ``load_deap``.
    """
    if id_to_experiment is not None:
        remapped: Dict[str, float] = {}
        for raw_id, value in counts.items():
            key = id_to_experiment.get(raw_id)
            if key is None:
                logger.warning(
                    "deap_market: YouTube id %r has no experiment mapping; dropped",
                    raw_id,
                )
                continue
            remapped[key] = value
        counts = remapped

    bad_keys = [k for k in counts if not re.fullmatch(r"exp-\d{2}", k)]
    if bad_keys:
        raise ValueError(
            f"these outcome keys are not exp-NN stimulus ids: {bad_keys[:5]}. Either key "
            "counts by exp-NN, or pass id_to_experiment to remap from YouTube "
            "ids."
        )

    raw = {k: float(v) for k, v in counts.items()}
    if any(v < 0 for v in raw.values()):
        raise ValueError("view counts cannot be negative")

    if log_transform:
        y_aggregate = {k: float(np.log1p(v)) for k, v in raw.items()}
    else:
        y_aggregate = dict(raw)

    provenance: Dict[str, Any] = {
        "outcome": "youtube_view_count",
        "log_transform": log_transform,
        "fetch_date": fetch_date,
        "n_stimuli_with_outcome": len(y_aggregate),
        "raw_min": min(raw.values()) if raw else None,
        "raw_max": max(raw.values()) if raw else None,
    }
    if fetch_date is None:
        provenance["warning"] = (
            "no fetch_date recorded; DEAP was recorded in 2012, so a later view "
            "count reflects years of accumulation the EEG could not anticipate"
        )
        logger.warning(
            "deap_market: no fetch_date supplied; the temporal gap between the "
            "2012 recordings and these counts is a real confound and is now "
            "undocumented"
        )

    if len(y_aggregate) < DEAP_N_TRIALS:
        logger.info(
            "deap_market: %d/%d stimuli have a market outcome; the rest are "
            "excluded from aggregate forecasting",
            len(y_aggregate),
            DEAP_N_TRIALS,
        )
    return y_aggregate, provenance


def load_counts_file(path: str) -> Dict[str, float]:
    """Read a two-column CSV of ``key,view_count`` into a dict.

    ``key`` may be an ``exp-NN`` id or a YouTube id; this function does not care
    which, it just reads the pairs. Pair the result with
    :func:`market_outcome_from_counts` for validation and keying.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reading a counts file requires pandas") from exc

    table = pd.read_csv(Path(path))
    if table.shape[1] < 2:
        raise ValueError(
            f"counts file needs at least two columns (key, count); got {list(table.columns)}"
        )
    key_col, count_col = table.columns[0], table.columns[1]
    return {str(k): float(v) for k, v in zip(table[key_col], table[count_col])}


def invert_id_map(
    ids: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, str]:
    """Invert :func:`youtube_ids_from_video_list` to YouTube-id -> ``exp-NN``."""
    out: Dict[str, str] = {}
    for exp_key, record in ids.items():
        vid = record.get("youtube_id")
        if vid:
            out[vid] = exp_key
    return out
