"""Behavioral decoding: multimodal prediction of individual and aggregate choice.

The package is organised around one contract: every modality (fMRI, EEG, face
video, behaviour) is reduced to a :class:`~behavioral_decoding.io.base.ModalityBlock`
carrying a trial-by-feature matrix plus the subject and stimulus keys needed to
move between the individual level and the population level.
"""

from __future__ import annotations

from ._compat import silence_imblearn_validate_data_warning

__version__ = "0.1.0"

# One targeted filter for a known-noisy dependency combination. See _compat.py.
silence_imblearn_validate_data_warning()

__all__ = ["__version__"]
