"""Progress reporting.

Anything in this repo that can run for more than a few seconds is wrapped in
:func:`progress`, so a stalled job is visually distinguishable from a slow one.
Falls back to a minimal stderr ticker when ``tqdm`` is not installed.
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, Optional, Sized, TypeVar

T = TypeVar("T")

try:  # pragma: no cover - trivial import guard
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


class _FallbackBar:
    """Minimal stderr progress ticker used when tqdm is unavailable."""

    def __init__(self, iterable: Iterable[T], total: Optional[int], desc: str) -> None:
        self._iterable = iterable
        self._total = total
        self._desc = desc
        self._start = time.time()

    def __iter__(self) -> Iterator[T]:
        for i, item in enumerate(self._iterable, start=1):
            yield item
            if i % max(1, (self._total or 100) // 20) == 0 or i == self._total:
                elapsed = time.time() - self._start
                if self._total:
                    pct = 100.0 * i / self._total
                    msg = f"{self._desc} {pct:3.0f}% ({i}/{self._total}) {elapsed:.1f}s"
                else:
                    msg = f"{self._desc} {i} items {elapsed:.1f}s"
                print("\r" + msg, end="", file=sys.stderr, flush=True)
        print("", file=sys.stderr, flush=True)


def progress(
    iterable: Iterable[T],
    desc: str = "",
    total: Optional[int] = None,
    disable: bool = False,
) -> Iterable[T]:
    """Wrap ``iterable`` in a progress bar.

    Parameters
    ----------
    iterable:
        Any iterable. If it is :class:`~typing.Sized` and ``total`` is omitted,
        the length is inferred.
    desc:
        Short label shown alongside the bar.
    total:
        Expected number of items, when it cannot be inferred.
    disable:
        Suppress output entirely (useful inside tests).
    """
    if disable:
        return iterable
    if total is None and isinstance(iterable, Sized):
        total = len(iterable)
    if _tqdm is not None:
        return _tqdm(iterable, desc=desc, total=total, leave=False)
    return _FallbackBar(iterable, total, desc)
