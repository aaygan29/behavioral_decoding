"""EEG loading and epoch-wise feature extraction.

Feature set is band power per channel plus two event-related components that
carry reward and valuation signal in the choice literature: an early frontal
window (feedback/reward related negativity range) and a late centro-parietal
window (P3/LPP range, which tracks motivational salience). Band power and ERP
windows are computed per epoch so each trial becomes one row, matching the
trial-level contract in :mod:`behavioral_decoding.io.base`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.progress import progress
from .base import EEG, BaseLoader, ModalityBlock

DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# (name, t_start_s, t_end_s) relative to stimulus onset
DEFAULT_ERP_WINDOWS: Sequence[Tuple[str, float, float]] = (
    ("early_frontal", 0.20, 0.35),
    ("late_parietal", 0.40, 0.80),
)


def bandpower(
    epochs: np.ndarray,
    sfreq: float,
    bands: Dict[str, Tuple[float, float]],
) -> Tuple[np.ndarray, List[str]]:
    """Per-epoch, per-channel log band power via Welch-style periodogram.

    Parameters
    ----------
    epochs:
        Array of shape ``(n_epochs, n_channels, n_times)``.
    sfreq:
        Sampling frequency in Hz.

    Returns
    -------
    (features, names)
        ``features`` has shape ``(n_epochs, n_channels * n_bands)``.
    """
    n_epochs, n_channels, n_times = epochs.shape
    window = np.hanning(n_times)
    spectrum = np.fft.rfft(epochs * window[None, None, :], axis=-1)
    psd = (np.abs(spectrum) ** 2) / (sfreq * np.sum(window ** 2))
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)

    out = []
    names: List[str] = []
    for band_name, (lo, hi) in bands.items():
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():
            # Epoch too short to resolve this band. Emitting zeros here would be
            # a silent lie, so refuse instead.
            raise ValueError(
                "band {!r} ({}-{} Hz) is not resolvable at sfreq={} with {} samples "
                "({:.2f} Hz resolution)".format(
                    band_name,
                    lo,
                    hi,
                    sfreq,
                    n_times,
                    freqs[1] - freqs[0] if len(freqs) > 1 else float("nan"),
                )
            )
        power = psd[:, :, sel].mean(axis=-1)  # (n_epochs, n_channels)
        out.append(np.log(power + 1e-20))
        names.extend([f"{band_name}_ch{c:02d}" for c in range(n_channels)])
    return np.concatenate(out, axis=1), names


def erp_windows(
    epochs: np.ndarray,
    times: np.ndarray,
    windows: Sequence[Tuple[str, float, float]],
) -> Tuple[np.ndarray, List[str]]:
    """Mean amplitude in each time window, per channel, per epoch."""
    n_epochs, n_channels, _ = epochs.shape
    out = []
    names: List[str] = []
    for win_name, t0, t1 in windows:
        sel = (times >= t0) & (times <= t1)
        if not sel.any():
            raise ValueError(
                f"ERP window {win_name!r} ({t0}-{t1} s) falls outside the epoch span "
                f"({times[0]:.2f} to {times[-1]:.2f} s)"
            )
        out.append(epochs[:, :, sel].mean(axis=-1))
        names.extend([f"{win_name}_ch{c:02d}" for c in range(n_channels)])
    return np.concatenate(out, axis=1), names


class EEGLoader(BaseLoader):
    """Turn epoched EEG into trial-by-feature rows."""

    name = EEG

    def __init__(
        self,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        windows: Optional[Sequence[Tuple[str, float, float]]] = None,
        include_bandpower: bool = True,
        include_erp: bool = True,
    ) -> None:
        self.bands = dict(bands) if bands is not None else dict(DEFAULT_BANDS)
        self.windows = tuple(windows) if windows is not None else tuple(DEFAULT_ERP_WINDOWS)
        self.include_bandpower = include_bandpower
        self.include_erp = include_erp

    def from_arrays(
        self,
        epochs: np.ndarray,
        sfreq: float,
        times: np.ndarray,
        subject_ids: Sequence,
        stimulus_ids: Sequence,
        source: str = "arrays",
    ) -> ModalityBlock:
        """Build a block from an ``(n_epochs, n_channels, n_times)`` array."""
        epochs = np.asarray(epochs, dtype=float)
        if epochs.ndim != 3:
            raise ValueError(
                f"expected (n_epochs, n_channels, n_times), got shape {epochs.shape}"
            )
        parts: List[np.ndarray] = []
        names: List[str] = []
        if self.include_bandpower:
            feats, feat_names = bandpower(epochs, sfreq, self.bands)
            parts.append(feats)
            names.extend(feat_names)
        if self.include_erp:
            feats, feat_names = erp_windows(epochs, np.asarray(times, dtype=float), self.windows)
            parts.append(feats)
            names.extend(feat_names)
        if not parts:
            raise ValueError("EEGLoader configured with no feature families enabled")

        return ModalityBlock(
            name=self.name,
            X=np.concatenate(parts, axis=1),
            subject_ids=np.asarray(subject_ids),
            stimulus_ids=np.asarray(stimulus_ids),
            feature_names=names,
            provenance=self._provenance(
                source=source,
                sfreq=sfreq,
                bands=list(self.bands.keys()) if self.include_bandpower else [],
                erp_windows=[w[0] for w in self.windows] if self.include_erp else [],
            ),
        )

    def load(
        self,
        raw_paths: Sequence[str],
        subject_ids: Sequence,
        event_id: Optional[Dict[str, int]] = None,
        tmin: float = -0.2,
        tmax: float = 1.0,
        l_freq: float = 1.0,
        h_freq: float = 45.0,
        stimulus_key: str = "stimulus_id",
    ) -> ModalityBlock:
        """Load and epoch raw EEG files. Requires ``mne``.

        Filtering and epoching happen here because they are per-recording signal
        conditioning, not model fitting. Anything that estimates parameters
        across trials (scaling, PCA, resampling for class balance) stays out of
        this function and goes inside the CV fold.
        """
        try:
            import mne
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "EEGLoader.load requires mne. Install with `pip install '.[eeg]'`, "
                "or use EEGLoader.from_arrays if you already have epochs."
            ) from exc

        all_epochs: List[np.ndarray] = []
        subj_out: List[object] = []
        stim_out: List[object] = []
        sfreq: Optional[float] = None
        times: Optional[np.ndarray] = None

        for i, path in enumerate(progress(raw_paths, desc="EEG files", total=len(raw_paths))):
            raw = mne.io.read_raw(path, preload=True, verbose="ERROR")
            raw.filter(l_freq, h_freq, verbose="ERROR")
            events, found_ids = mne.events_from_annotations(raw, verbose="ERROR")
            ep = mne.Epochs(
                raw,
                events,
                event_id=event_id or found_ids,
                tmin=tmin,
                tmax=tmax,
                baseline=(tmin, 0.0),
                preload=True,
                verbose="ERROR",
            )
            data = ep.get_data()
            all_epochs.append(data)
            sfreq = float(ep.info["sfreq"])
            times = ep.times
            meta = ep.metadata
            if meta is not None and stimulus_key in meta:
                stim_out.extend(list(meta[stimulus_key]))
            else:
                # No stimulus labels means no aggregate forecasting is possible.
                # Fall back to event codes and say so in provenance.
                stim_out.extend(list(ep.events[:, 2]))
            subj_out.extend([subject_ids[i]] * data.shape[0])

        if sfreq is None or times is None:
            raise ValueError("no EEG epochs were produced from the given files")

        return self.from_arrays(
            epochs=np.concatenate(all_epochs, axis=0),
            sfreq=sfreq,
            times=times,
            subject_ids=np.asarray(subj_out),
            stimulus_ids=np.asarray(stim_out),
            source="mne",
        )
