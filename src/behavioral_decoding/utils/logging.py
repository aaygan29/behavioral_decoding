from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "behavioral_decoding", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger writing to stderr."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("behavioral_decoding")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
