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
from .deap import (
    DEAP_BANDS,
    DEAP_EEG_CHANNELS,
    DEAP_LABEL_NAMES,
    DEAP_PERIPHERAL_CHANNELS,
    PERIPHERAL,
    DEAPFormatError,
    DEAPLoader,
    binarise_ratings,
    load_deap,
)
from .eeg import EEGLoader
from .face import FaceLoader
from .fmri import FMRILoader
from .registry import get_loader, register_loader

__all__ = [
    "BEHAVIOR",
    "DEAP_BANDS",
    "DEAP_EEG_CHANNELS",
    "DEAP_LABEL_NAMES",
    "DEAP_PERIPHERAL_CHANNELS",
    "EEG",
    "FACE",
    "FMRI",
    "KNOWN_MODALITIES",
    "PERIPHERAL",
    "BaseLoader",
    "BehaviorLoader",
    "DEAPFormatError",
    "DEAPLoader",
    "EEGLoader",
    "FaceLoader",
    "FMRILoader",
    "ModalityBlock",
    "MultimodalDataset",
    "binarise_ratings",
    "get_loader",
    "load_deap",
    "register_loader",
]
