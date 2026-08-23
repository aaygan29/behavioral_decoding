"""Robust selection and cleaning of fMRIPrep confound tables.

A raw ``*_desc-confounds_timeseries.tsv`` from fMRIPrep is not something you can
hand straight to a Nilearn masker. Three things go wrong if you try:

1. **It is enormous and mostly noise.** Recent fMRIPrep writes a hundred-plus
   columns: every aCompCor component, every cosine drift term, DVARS variants,
   scrubbing indicator columns, and more. Regressing all of them out of a short
   task run burns degrees of freedom and can remove signal. You want an
   explicit, theory-chosen nuisance set, not "everything in the file".
2. **It contains NaNs by construction.** The first row of every ``*_derivative1``
   column is NaN (no previous volume to difference against), and
   ``framewise_displacement`` / ``dvars`` are NaN on the first volume too.
   Nilearn's masker rejects non-finite confounds, so these must be filled before
   the table is used.
3. **It contains non-numeric columns.** Columns like ``motion_outlier00`` are
   one-hot integers that read fine, but some releases carry string-valued or
   all-NaN columns that break a naive ``to_numpy(float)``.

This module turns a path / DataFrame / array into a clean numeric matrix by
selecting a named regressor set and filling missing values, and it reports what
it kept and what it dropped so the choice is auditable in the run record.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

# The six rigid-body head-motion parameters. Present in every fMRIPrep release
# under these exact names.
MOTION_6 = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")

# Their temporal derivatives: captures spin-history / lagged motion effects.
MOTION_DERIV = tuple(f"{c}_derivative1" for c in MOTION_6)

# Mean tissue signals. Regressing CSF and white matter removes non-neural
# physiological fluctuation without touching grey-matter task signal. The global
# signal is deliberately excluded from the default (it is contested: it can
# induce artefactual anticorrelations), but is available via ``strategy``.
PHYSIO = ("csf", "white_matter")
GLOBAL = ("global_signal",)

# Named, human-readable strategies -> explicit column lists. "24-parameter"
# Friston expansion = 6 motion + derivatives + their squares; we expose the
# common 6+deriv+physio "aggressive but standard" set as the default.
STRATEGIES: Dict[str, Tuple[str, ...]] = {
    "motion6": MOTION_6,
    "motion6+physio": MOTION_6 + PHYSIO,
    "motion12": MOTION_6 + MOTION_DERIV,
    "motion12+physio": MOTION_6 + MOTION_DERIV + PHYSIO,
    "motion12+physio+global": MOTION_6 + MOTION_DERIV + PHYSIO + GLOBAL,
}

DEFAULT_STRATEGY = "motion12+physio"

# Column-name prefixes that select a whole family (aCompCor, cosine drift). Used
# only when explicitly requested via ``extra_prefixes``; not in any default.
KNOWN_PREFIXES = ("a_comp_cor_", "t_comp_cor_", "cosine")


class ConfoundError(ValueError):
    """Raised when a requested confound strategy cannot be satisfied."""


def _read_table(source: Any) -> Any:
    """Return a pandas DataFrame from a path or an existing frame."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reading fMRIPrep confounds requires pandas (`pip install pandas`)"
        ) from exc
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source, sep="\t")


def select_confounds(
    source: Any,
    strategy: str = DEFAULT_STRATEGY,
    columns: Optional[Sequence[str]] = None,
    extra_prefixes: Sequence[str] = (),
    n_compcor: int = 0,
    fill: str = "mean",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select and clean nuisance regressors from an fMRIPrep confounds table.

    Parameters
    ----------
    source:
        Path to a ``*_desc-confounds_timeseries.tsv``, a pandas DataFrame, or an
        already-numeric ``(n_volumes, n_regressors)`` array. An array is trusted
        as pre-selected and only gets the missing-value fill.
    strategy:
        A named regressor set from :data:`STRATEGIES`. Ignored if ``columns`` is
        given.
    columns:
        An explicit list of column names to use, overriding ``strategy``. Names
        not present in the table are reported and skipped rather than raising, so
        one strategy works across fMRIPrep versions that renamed a column.
    extra_prefixes:
        Column-name prefixes whose every match is added (e.g. ``("cosine",)``
        for drift terms). Combine with ``n_compcor`` for aCompCor.
    n_compcor:
        Number of leading anatomical CompCor components to add
        (``a_comp_cor_00`` ... ``a_comp_cor_{n-1}``). 0 disables.
    fill:
        Missing-value policy for the selected columns. ``"mean"`` replaces NaNs
        with the column mean (the standard choice for the leading-NaN derivative
        and framewise-displacement rows); ``"zero"`` replaces them with 0.0.

    Returns
    -------
    (matrix, provenance)
        ``matrix`` is a finite ``float`` array of shape
        ``(n_volumes, n_regressors)``. ``provenance`` records the columns kept,
        the columns requested-but-missing, any non-numeric columns dropped, and
        the number of NaN cells filled, all for the run record.
    """
    if fill not in {"mean", "zero"}:
        raise ValueError(f"fill must be 'mean' or 'zero'; got {fill!r}")

    # An already-numeric array is trusted as pre-selected: just make it finite.
    if isinstance(source, np.ndarray):
        matrix, n_filled = _fill_missing(np.asarray(source, dtype=float), fill)
        return matrix, {
            "source": "array",
            "kept": [f"col_{i}" for i in range(matrix.shape[1])],
            "n_regressors": int(matrix.shape[1]),
            "n_nan_filled": int(n_filled),
            "fill": fill,
        }

    table = _read_table(source)

    if columns is not None:
        wanted: List[str] = list(columns)
    else:
        if strategy not in STRATEGIES:
            raise ConfoundError(
                f"unknown confound strategy {strategy!r}; choose from "
                f"{sorted(STRATEGIES)} or pass an explicit `columns` list"
            )
        wanted = list(STRATEGIES[strategy])

    for prefix in extra_prefixes:
        wanted.extend(c for c in table.columns if str(c).startswith(prefix))
    if n_compcor > 0:
        wanted.extend(f"a_comp_cor_{i:02d}" for i in range(n_compcor))

    # Preserve request order, drop duplicates.
    seen: set = set()
    ordered = [c for c in wanted if not (c in seen or seen.add(c))]

    present = [c for c in ordered if c in table.columns]
    missing = [c for c in ordered if c not in table.columns]
    if missing:
        logger.warning(
            "confounds: %d requested column(s) absent from this file and skipped: "
            "%s. This is expected across fMRIPrep versions, but check the names.",
            len(missing),
            missing,
        )
    if not present:
        raise ConfoundError(
            f"none of the requested confound columns {ordered} are in the file "
            f"(has {list(table.columns)[:12]}...). Wrong file or wrong strategy."
        )

    # Coerce to numeric column by column so a single string column is dropped
    # with a name rather than poisoning the whole matrix.
    import pandas as pd

    numeric_cols: List[str] = []
    dropped_nonnumeric: List[str] = []
    series: List[np.ndarray] = []
    for col in present:
        coerced = pd.to_numeric(table[col], errors="coerce")
        # A column that is entirely NaN after coercion was non-numeric (string
        # labels) or empty; it carries nothing and would just be filled to a
        # constant, so drop it explicitly.
        if coerced.notna().sum() == 0:
            dropped_nonnumeric.append(col)
            continue
        numeric_cols.append(col)
        series.append(coerced.to_numpy(dtype=float))

    if dropped_nonnumeric:
        logger.warning(
            "confounds: dropped %d non-numeric/all-NaN column(s): %s",
            len(dropped_nonnumeric),
            dropped_nonnumeric,
        )
    if not numeric_cols:
        raise ConfoundError(
            "every selected confound column was non-numeric or all-NaN; nothing "
            "usable to regress out"
        )

    matrix = np.column_stack(series)
    matrix, n_filled = _fill_missing(matrix, fill)

    provenance = {
        "source": "fmriprep_tsv" if not isinstance(source, pd.DataFrame) else "dataframe",
        "strategy": None if columns is not None else strategy,
        "kept": numeric_cols,
        "n_regressors": int(matrix.shape[1]),
        "requested_missing": missing,
        "dropped_nonnumeric": dropped_nonnumeric,
        "n_nan_filled": int(n_filled),
        "fill": fill,
    }
    return matrix, provenance


def _fill_missing(matrix: np.ndarray, fill: str) -> Tuple[np.ndarray, int]:
    """Replace non-finite cells; return the cleaned matrix and the fill count."""
    matrix = np.array(matrix, dtype=float, copy=True)
    bad = ~np.isfinite(matrix)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return matrix, 0
    if fill == "zero":
        matrix[bad] = 0.0
        return matrix, n_bad
    # Column-mean fill over the finite entries; a wholly non-finite column
    # (should not survive selection, but be safe) falls back to 0.
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        finite = np.isfinite(col)
        col[~finite] = col[finite].mean() if finite.any() else 0.0
    return matrix, n_bad
