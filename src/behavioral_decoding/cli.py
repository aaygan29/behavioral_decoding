"""Console entry points."""

from __future__ import annotations

import sys
from pathlib import Path


def demo() -> int:
    """Run ``scripts/run_demo.py``. Installed as the ``bd-demo`` command."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_demo.py"
    if not script.exists():
        print(
            f"demo script not found at {script}. Run it from a source checkout: "
            "`python scripts/run_demo.py`",
            file=sys.stderr,
        )
        return 1
    namespace = {"__name__": "__main__", "__file__": str(script)}
    try:
        exec(compile(script.read_text(), str(script), "exec"), namespace)  # noqa: S102
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(demo())
