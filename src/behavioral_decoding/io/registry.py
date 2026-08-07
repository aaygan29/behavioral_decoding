"""Loader registry, so configs can name a modality as a string.

Adding a modality means writing a loader and registering it here. Nothing else
in the pipeline needs to change: the ensemble discovers modalities from the
dataset's blocks.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import BEHAVIOR, EEG, FACE, FMRI, BaseLoader
from .behavior import BehaviorLoader
from .eeg import EEGLoader
from .face import FaceLoader
from .fmri import FMRILoader

_REGISTRY: Dict[str, Type[BaseLoader]] = {
    FMRI: FMRILoader,
    EEG: EEGLoader,
    FACE: FaceLoader,
    BEHAVIOR: BehaviorLoader,
}


def register_loader(name: str, loader_cls: Type[BaseLoader]) -> Type[BaseLoader]:
    """Register a loader class under ``name``. Usable as a decorator."""
    if not issubclass(loader_cls, BaseLoader):
        raise TypeError(f"{loader_cls!r} must subclass BaseLoader")
    _REGISTRY[name] = loader_cls
    return loader_cls


def get_loader(name: str, **kwargs: object) -> BaseLoader:
    """Instantiate the loader registered under ``name``."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown modality {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)  # type: ignore[arg-type]


def available_modalities() -> Dict[str, Type[BaseLoader]]:
    return dict(_REGISTRY)
