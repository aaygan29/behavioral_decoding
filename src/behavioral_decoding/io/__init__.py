from __future__ import annotations

from .base import (
    BEHAVIOR,
    EEG,
    FACE,
    FMRI,
    KNOWN_MODALITIES,
    BaseLoader,
    ModalityBlock,
    MultimodalDataset,
)
from .behavior import BehaviorLoader
from .eeg import EEGLoader
from .face import FaceLoader
from .fmri import FMRILoader
from .registry import get_loader, register_loader

__all__ = [
    "BEHAVIOR",
    "EEG",
    "FACE",
    "FMRI",
    "KNOWN_MODALITIES",
    "BaseLoader",
    "BehaviorLoader",
    "EEGLoader",
    "FaceLoader",
    "FMRILoader",
    "ModalityBlock",
    "MultimodalDataset",
    "get_loader",
    "register_loader",
]
