"""Narrowly-scoped compatibility shims.

Only one thing lives here, and it is deliberately specific rather than a blanket
warning filter. Blanket filters hide the deprecation that actually matters.

**imbalanced-learn 0.12 on scikit-learn 1.6.** imbalanced-learn dropped Python
3.9 at version 0.13, so a Python 3.9 environment is pinned to 0.12.x, which
calls ``BaseEstimator._validate_data``. scikit-learn 1.6 deprecated that method
and emits a ``FutureWarning`` on every single fit. With bagging that is tens of
thousands of identical warnings per run, which buries any real one.

The filter below matches that one message from that one module. Everything else
still warns. Once this project's floor moves to Python 3.10 and
imbalanced-learn 0.13+, delete this file and its import in ``__init__``.
"""

from __future__ import annotations

import warnings


def silence_imblearn_validate_data_warning() -> bool:
    """Filter the one known-noisy deprecation. Returns True if a filter was added."""
    try:
        import imblearn
    except ImportError:
        return False

    version = getattr(imblearn, "__version__", "0")
    try:
        major, minor = (int(p) for p in version.split(".")[:2])
    except ValueError:  # pragma: no cover - unusual version string
        return False

    if (major, minor) >= (0, 13):
        return False

    warnings.filterwarnings(
        "ignore",
        message=r".*_validate_data.*is deprecated in 1\.6.*",
        category=FutureWarning,
        module=r"sklearn\.base",
    )
    return True
