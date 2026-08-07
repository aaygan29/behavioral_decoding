"""DEAP loader.

DEAP (Koelstra et al., 2012) is the fastest route from this framework to real
recordings: 32 participants, 40 one-minute music videos each, EEG plus
peripheral physiology on every participant, frontal face video on 22 of them,
and per-trial self-report. Every participant saw every video, so the
subject-by-stimulus design is fully crossed, which is exactly the shape the
aggregate-forecasting arm needs.

Access requires an end-user licence agreement with Queen Mary University of
London. Nothing here downloads anything.

---

## The traps

Six things about DEAP will silently corrupt a result if the loader ignores them.
Each is handled below, and each has a test.

**1. The preprocessed data is bandpass filtered 4.0 to 45.0 Hz.** Delta (1 to
4 Hz) is *gone*. The framework's default EEG bands include delta, and running
them here yields a near-zero column that looks like a feature and is filter
roll-off. :data:`DEAP_BANDS` therefore omits delta, and :class:`DEAPLoader`
raises if you ask for a band outside the passband.

**2. The label column order is (valence, arousal, dominance, liking).** Some
widely-used wrappers document a different order in their prose. Swapping valence
and arousal produces a result that is wrong and completely plausible. The order
is asserted in one place, :data:`DEAP_LABEL_NAMES`, and used everywhere.

**3. Each trial is 8064 samples = 3 s pre-trial baseline + 60 s of video.**
Feeding all 8064 samples to a band-power extractor mixes the baseline into the
signal. The baseline is split off and, optionally, used to correct the trial.

**4. The pickles were written under Python 2.** They need
``encoding="latin1"``. Without it you get a ``UnicodeDecodeError`` that looks
like file corruption.

**5. Channels 33 to 40 are not EEG.** GSR, respiration, temperature, and
plethysmograph are slow autonomic signals; hEOG/vEOG are eye movement; zEMG and
tEMG are muscle. Band power in the EEG sense is meaningless for them. They get
their own modality block with their own features.

**6. This is not an event-related design.** A 60-second music video has no
stimulus-locked ERP. The framework's ERP windows are disabled here, and asking
for them raises.

---

## The circularity warning

DEAP's self-report block deserves more suspicion than the other modalities.

The four ratings (valence, arousal, dominance, liking) were collected on the
same screen, seconds apart, from the same person. Predicting ``liking`` from
``valence`` and ``arousal`` is close to trivial and tells you nothing about
brains. It is self-report predicting self-report.

That matters because in the neuroforecasting paradigm the behavioural
comparator is a *choice* (fund it or not, watch it or not), which is a different
kind of measurement from the neural signal. DEAP has no such choice. So the
behaviour arm here is not the comparator that Genevsky, Yoon and Knutson (2017)
beat, and a plot showing behaviour outperforming EEG on DEAP is not a
replication of anything.

:class:`DEAPLoader` therefore defaults to ``behavior_mode="noncircular"``, which
uses only familiarity and trial order. Set ``behavior_mode="ratings"`` if you
want the circular version, and label it as such wherever it appears.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logging import get_logger
from ..utils.progress import progress
from .base import BEHAVIOR, EEG, BaseLoader, ModalityBlock

logger = get_logger(__name__)

# ----------------------------------------------------------------- constants

PERIPHERAL = "peripheral"

DEAP_SFREQ = 128.0
DEAP_N_TRIALS = 40
DEAP_N_CHANNELS = 40
DEAP_N_SAMPLES = 8064
DEAP_BASELINE_SECONDS = 3.0
DEAP_TRIAL_SECONDS = 60.0
DEAP_BASELINE_SAMPLES = int(DEAP_BASELINE_SECONDS * DEAP_SFREQ)  # 384
DEAP_TRIAL_SAMPLES = int(DEAP_TRIAL_SECONDS * DEAP_SFREQ)  # 7680

# The filter applied when DEAP's preprocessed files were made. Anything outside
# this range is not in the data, whatever a band-power function will happily
# return for it.
DEAP_BANDPASS = (4.0, 45.0)

# Channels 1-32, in the order the preprocessed files use.
DEAP_EEG_CHANNELS: Tuple[str, ...] = (
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
)

# Channels 33-40.
DEAP_PERIPHERAL_CHANNELS: Tuple[str, ...] = (
    "hEOG",           # horizontal EOG
    "vEOG",           # vertical EOG
    "zEMG",           # zygomaticus major, the smile muscle
    "tEMG",           # trapezius
    "GSR",            # galvanic skin response
    "Respiration",    # respiration belt
    "Plethysmograph", # blood volume pulse
    "Temperature",
)

DEAP_ALL_CHANNELS: Tuple[str, ...] = DEAP_EEG_CHANNELS + DEAP_PERIPHERAL_CHANNELS

# Column order of the `labels` array. Load-bearing: see trap 2 above.
DEAP_LABEL_NAMES: Tuple[str, ...] = ("valence", "arousal", "dominance", "liking")

# Delta is deliberately absent. See trap 1.
DEAP_BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Left/right pairs for frontal alpha asymmetry. Greater relative *left* frontal
# activity is the classic approach-motivation marker, and because alpha power is
# inversely related to cortical activity the conventional index is
# log(right) - log(left). This is the closest thing DEAP has to a theory-specified
# feature, and it plays the role the NAcc ROI plays on the fMRI side.
DEAP_ASYMMETRY_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("F3", "F4"),
    ("F7", "F8"),
    ("AF3", "AF4"),
    ("FC5", "FC6"),
    ("FC1", "FC2"),
)

# Participants 1-22 were recorded at one site and 23-32 at the other, and 22 of
# the 32 have frontal face video. Which 22 is not something to hardcode from a
# secondary source: the loader globs the video directory and reports what it
# finds. See `available_face_videos`.
DEAP_N_PARTICIPANTS = 32

RATING_SCALE = (1.0, 9.0)
FAMILIARITY_SCALE = (1.0, 5.0)


class DEAPFormatError(ValueError):
    """Raised when a file does not match the documented DEAP layout.

    Deliberately loud. A DEAP file with an unexpected shape is either a
    different release, a different preprocessing, or a corrupted download, and
    all three produce wrong numbers rather than crashes if waved through.
    """


# ------------------------------------------------------------- file discovery


def subject_id_from_path(path: Path) -> str:
    """``.../s07.dat`` -> ``"s07"``. Raises if the name is not DEAP-shaped."""
    match = re.fullmatch(r"(s\d{2})", path.stem, flags=re.IGNORECASE)
    if not match:
        raise DEAPFormatError(
            f"{path.name!r} is not a DEAP participant file; expected names like 's01.dat' "
            "through 's32.dat'"
        )
    return match.group(1).lower()


def find_subject_files(root: str, pattern: str = "*.dat") -> List[Path]:
    """Locate participant files under ``root``, sorted by participant number.

    Looks in ``root`` itself and in a ``data_preprocessed_python`` subdirectory,
    since both layouts occur depending on how the archive was unpacked.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"DEAP root {str(root_path)!r} does not exist")

    candidates: List[Path] = []
    for directory in (root_path, root_path / "data_preprocessed_python"):
        if directory.is_dir():
            candidates.extend(sorted(directory.glob(pattern)))

    files = []
    seen = set()
    for path in candidates:
        try:
            sid = subject_id_from_path(path)
        except DEAPFormatError:
            continue
        if sid not in seen:
            seen.add(sid)
            files.append(path)

    if not files:
        raise FileNotFoundError(
            f"no DEAP participant files found under {str(root_path)!r}. Expected 's01.dat' ... "
            "'s32.dat', either directly in that directory or in a "
            "'data_preprocessed_python' subdirectory. DEAP requires an EULA with "
            "Queen Mary University of London; this loader does not download "
            "anything."
        )

    files.sort(key=lambda p: subject_id_from_path(p))
    logger.info("DEAP: found %d participant files under %s", len(files), root_path)
    return files


def available_face_videos(root: str) -> Dict[str, List[Path]]:
    """Map participant id to any face-video files found.

    DEAP ships frontal face video for 22 of the 32 participants. Rather than
    hardcode which 22 from a secondary source, this globs and reports. An empty
    result means the video archive was not downloaded, which is common, since it
    is distributed separately from the signals.
    """
    root_path = Path(root)
    found: Dict[str, List[Path]] = {}
    for directory in (
        root_path,
        root_path / "face_video",
        root_path / "face_video_original",
    ):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in {".avi", ".mp4", ".mov", ".mkv"}:
                continue
            match = re.search(r"s(\d{2})", path.stem, flags=re.IGNORECASE)
            if match:
                found.setdefault(f"s{match.group(1)}", []).append(path)
    return found


# --------------------------------------------------------------- raw loading


def load_subject_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read one ``sNN.dat`` pickle.

    Returns
    -------
    (data, labels)
        ``data`` is ``(40, 40, 8064)``; ``labels`` is ``(40, 4)`` in
        :data:`DEAP_LABEL_NAMES` order.
    """
    file_path = Path(path)
    try:
        with open(file_path, "rb") as handle:
            # The archive was pickled under Python 2. Without latin1 this raises
            # a UnicodeDecodeError that reads like a corrupt download.
            payload = pickle.load(handle, encoding="latin1")
    except UnicodeDecodeError:
        with open(file_path, "rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
    except pickle.UnpicklingError as exc:
        raise DEAPFormatError(
            f"{file_path.name} is not a readable pickle. If you downloaded the MATLAB release "
            "(data_preprocessed_matlab/*.mat), use scipy.io.loadmat and pass the "
            "arrays to DEAPLoader.from_arrays instead."
        ) from exc

    if not isinstance(payload, dict):
        raise DEAPFormatError(
            f"{file_path.name} unpickled to {type(payload).__name__}, "
            "expected a dict with 'data' and 'labels'"
        )

    # Byte keys appear when the latin1 path fails and we fall back to bytes.
    normalised = {
        (k.decode() if isinstance(k, bytes) else k): v for k, v in payload.items()
    }
    missing = {"data", "labels"} - set(normalised)
    if missing:
        raise DEAPFormatError(
            f"{file_path.name} is missing {sorted(missing)}; found keys {sorted(normalised)}"
        )

    data = np.asarray(normalised["data"], dtype=float)
    labels = np.asarray(normalised["labels"], dtype=float)
    _validate_shapes(file_path.name, data, labels)
    return data, labels


def _validate_shapes(name: str, data: np.ndarray, labels: np.ndarray) -> None:
    if data.shape != (DEAP_N_TRIALS, DEAP_N_CHANNELS, DEAP_N_SAMPLES):
        raise DEAPFormatError(
            f"{name}: data has shape {data.shape}, expected "
            f"({DEAP_N_TRIALS}, {DEAP_N_CHANNELS}, {DEAP_N_SAMPLES}). A different shape "
            "means a different release or preprocessing, and the channel and "
            "timing assumptions in this loader would not hold."
        )
    if labels.shape != (DEAP_N_TRIALS, len(DEAP_LABEL_NAMES)):
        raise DEAPFormatError(
            f"{name}: labels has shape {labels.shape}, expected "
            f"({DEAP_N_TRIALS}, {len(DEAP_LABEL_NAMES)}) in the order {DEAP_LABEL_NAMES}"
        )
    finite = labels[np.isfinite(labels)]
    if finite.size and (finite.min() < 0.5 or finite.max() > 9.5):
        logger.warning(
            "%s: ratings span %.2f to %.2f, outside the documented 1-9 SAM scale. "
            "Check that the label columns are (valence, arousal, dominance, liking) "
            "and not a rescaled or reordered variant.",
            name,
            float(finite.min()),
            float(finite.max()),
        )


def split_baseline(
    data: np.ndarray,
    baseline_samples: int = DEAP_BASELINE_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``(..., 8064)`` into ``(baseline, trial)`` along the last axis.

    The first 3 seconds are pre-trial rest. Leaving them attached drags the
    trial's band power toward rest by roughly 5 percent of the window and does
    so unevenly across trials.
    """
    if data.shape[-1] <= baseline_samples:
        raise DEAPFormatError(
            f"cannot split {baseline_samples} baseline samples from a "
            f"{data.shape[-1]}-sample window"
        )
    return data[..., :baseline_samples], data[..., baseline_samples:]


# ------------------------------------------------------------------ features


def _log_bandpower(
    signal: np.ndarray,
    sfreq: float,
    bands: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """Log band power for ``(n_trials, n_channels, n_times)``.

    Returns ``(n_trials, n_channels, n_bands)``, band order following ``bands``.
    Uses a Hann-windowed periodogram, matching
    :func:`behavioral_decoding.io.eeg.bandpower` but keeping the channel axis so
    asymmetry pairs can be computed before flattening.
    """
    n_times = signal.shape[-1]
    window = np.hanning(n_times)
    spectrum = np.fft.rfft(signal * window, axis=-1)
    psd = (np.abs(spectrum) ** 2) / (sfreq * np.sum(window ** 2))
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)

    out = np.empty(signal.shape[:-1] + (len(bands),), dtype=float)
    for i, (band_name, (lo, hi)) in enumerate(bands.items()):
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():
            raise DEAPFormatError(
                f"band {band_name!r} ({lo}-{hi} Hz) is not resolvable in a "
                f"{n_times}-sample window at "
                f"{sfreq} Hz"
            )
        out[..., i] = np.log(psd[..., sel].mean(axis=-1) + 1e-20)
    return out


def _peripheral_features(signal: np.ndarray) -> np.ndarray:
    """Summary statistics for ``(n_trials, n_channels, n_times)`` peripheral data.

    Four statistics per channel: mean level, variability, linear trend, and
    range. Deliberately generic, because proper physiological feature extraction
    is channel-specific work this does not attempt. GSR in particular should be
    decomposed into tonic and phasic components (cvxEDA, Ledalab) and the
    plethysmograph turned into inter-beat intervals before anything is claimed
    about autonomic arousal. See ``docs/deap.md``.
    """
    n_times = signal.shape[-1]
    t = np.linspace(0.0, 1.0, n_times)
    t_centred = t - t.mean()
    denom = float(np.sum(t_centred ** 2))

    mean = signal.mean(axis=-1)
    std = signal.std(axis=-1)
    slope = np.tensordot(signal - mean[..., None], t_centred, axes=([-1], [0])) / denom
    span = signal.max(axis=-1) - signal.min(axis=-1)
    return np.stack([mean, std, slope, span], axis=-1)


# -------------------------------------------------------------------- loader


class DEAPLoader(BaseLoader):
    """Turn a DEAP download into aligned :class:`ModalityBlock` objects."""

    name = "deap"

    def __init__(
        self,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        baseline_correct: bool = True,
        include_asymmetry: bool = True,
        behavior_mode: str = "noncircular",
        target: str = "liking",
        sfreq: float = DEAP_SFREQ,
    ) -> None:
        """
        Parameters
        ----------
        bands:
            Frequency bands. Every band must lie inside DEAP's 4 to 45 Hz
            passband; anything outside raises rather than returning filter
            roll-off dressed up as a feature.
        baseline_correct:
            Subtract each trial's own 3-second pre-trial band power. Per-trial
            and self-referential, so it cannot leak across the CV split.
        include_asymmetry:
            Append frontal alpha asymmetry features. See
            :data:`DEAP_ASYMMETRY_PAIRS`.
        behavior_mode:
            ``"noncircular"`` (default) uses familiarity and trial order only.
            ``"ratings"`` adds the other SAM ratings, which were collected from
            the same person on the same screen as the target and will dominate
            for reasons that have nothing to do with neural signal. ``"none"``
            omits the behaviour block entirely.
        target:
            Which rating becomes ``y_individual``. One of
            :data:`DEAP_LABEL_NAMES`. It is always excluded from the behaviour
            features.
        """
        if behavior_mode not in {"noncircular", "ratings", "none"}:
            raise ValueError(
                "behavior_mode must be 'noncircular', 'ratings', or 'none'"
            )
        if target not in DEAP_LABEL_NAMES:
            raise ValueError(
                f"target must be one of {DEAP_LABEL_NAMES}; got {target!r}"
            )

        self.bands = dict(bands) if bands is not None else dict(DEAP_BANDS)
        self._check_bands()
        self.baseline_correct = baseline_correct
        self.include_asymmetry = include_asymmetry
        self.behavior_mode = behavior_mode
        self.target = target
        self.sfreq = sfreq

    def _check_bands(self) -> None:
        lo_limit, hi_limit = DEAP_BANDPASS
        for band_name, (lo, hi) in self.bands.items():
            if lo < lo_limit or hi > hi_limit:
                raise ValueError(
                    f"band {band_name!r} ({lo}-{hi} Hz) falls outside "
                    f"DEAP's {lo_limit}-{hi_limit} Hz passband. "
                    "The preprocessed files were bandpass filtered when they were "
                    "made, so this band is not present in the data and any power "
                    "computed for it is filter roll-off. Delta is the usual "
                    "casualty. Use the raw .bdf release if you need it."
                )

    # ----------------------------------------------------------- block building

    def blocks_from_arrays(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        subject_id: str,
        stimulus_ids: Optional[Sequence] = None,
        familiarity: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, ModalityBlock], np.ndarray]:
        """Build one participant's blocks from raw arrays.

        Returns ``(blocks, ratings)`` where ``ratings`` is the ``(40, 4)`` label
        array, so the caller can pick a target without re-reading the file.
        """
        _validate_shapes(subject_id, data, labels)

        if stimulus_ids is None:
            # Trial order is randomised per participant in DEAP, so trial index
            # is NOT a stimulus identifier. Without the ratings CSV to supply
            # Experiment_id, stimulus keys cannot be recovered, and the
            # aggregate arm is impossible. Say so rather than inventing keys.
            raise ValueError(
                "stimulus_ids is required. DEAP presents the 40 videos in a "
                "randomised order per participant, so trial index does not "
                "identify a stimulus. Read Experiment_id from "
                "participant_ratings.csv (see load_participant_ratings) and pass "
                "it here, or the aggregate forecasting arm cannot be built."
            )
        stimulus_ids = np.asarray(stimulus_ids)
        if len(stimulus_ids) != DEAP_N_TRIALS:
            raise ValueError(
                f"expected {DEAP_N_TRIALS} stimulus ids, got {len(stimulus_ids)}"
            )

        subjects = np.array([subject_id] * DEAP_N_TRIALS)
        eeg_raw = data[:, : len(DEAP_EEG_CHANNELS), :]
        periph_raw = data[:, len(DEAP_EEG_CHANNELS) :, :]

        eeg_base, eeg_trial = split_baseline(eeg_raw)
        periph_base, periph_trial = split_baseline(periph_raw)

        blocks: Dict[str, ModalityBlock] = {
            EEG: self._eeg_block(eeg_base, eeg_trial, subjects, stimulus_ids),
            PERIPHERAL: self._peripheral_block(
                periph_base, periph_trial, subjects, stimulus_ids
            ),
        }

        if self.behavior_mode != "none":
            blocks[BEHAVIOR] = self._behavior_block(
                labels, subjects, stimulus_ids, familiarity
            )

        return blocks, labels

    def _eeg_block(
        self,
        baseline: np.ndarray,
        trial: np.ndarray,
        subjects: np.ndarray,
        stimulus_ids: np.ndarray,
    ) -> ModalityBlock:
        band_names = list(self.bands.keys())
        trial_power = _log_bandpower(trial, self.sfreq, self.bands)

        if self.baseline_correct:
            base_power = _log_bandpower(baseline, self.sfreq, self.bands)
            # Log-domain subtraction is a power ratio: relative change from that
            # trial's own rest. Uses nothing outside the trial, so it is safe to
            # do here rather than inside the CV fold.
            trial_power = trial_power - base_power

        features = [trial_power.reshape(trial_power.shape[0], -1)]
        names = [
            f"{band}_{channel}"
            for channel in DEAP_EEG_CHANNELS
            for band in band_names
        ]

        if self.include_asymmetry:
            asym, asym_names = self._asymmetry(trial_power, band_names)
            features.append(asym)
            names.extend(asym_names)

        return ModalityBlock(
            name=EEG,
            X=np.hstack(features),
            subject_ids=subjects,
            stimulus_ids=stimulus_ids,
            feature_names=names,
            provenance=self._provenance(
                source="deap_preprocessed_python",
                bands=band_names,
                baseline_corrected=self.baseline_correct,
                asymmetry=self.include_asymmetry,
                sfreq=self.sfreq,
                deap_bandpass=DEAP_BANDPASS,
                note="delta unavailable: DEAP preprocessed data is filtered 4-45 Hz",
            ),
        )

    def _asymmetry(
        self, power: np.ndarray, band_names: Sequence[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Right-minus-left log power for each frontal pair and band."""
        index = {name: i for i, name in enumerate(DEAP_EEG_CHANNELS)}
        columns = []
        names: List[str] = []
        for left, right in DEAP_ASYMMETRY_PAIRS:
            if left not in index or right not in index:  # pragma: no cover
                continue
            for b, band in enumerate(band_names):
                columns.append(power[:, index[right], b] - power[:, index[left], b])
                names.append(f"asym_{band}_{right}_{left}")
        if not columns:  # pragma: no cover - defensive
            return np.zeros((power.shape[0], 0)), []
        return np.column_stack(columns), names

    def _peripheral_block(
        self,
        baseline: np.ndarray,
        trial: np.ndarray,
        subjects: np.ndarray,
        stimulus_ids: np.ndarray,
    ) -> ModalityBlock:
        stat_names = ("mean", "std", "slope", "range")
        trial_stats = _peripheral_features(trial)

        if self.baseline_correct:
            base_stats = _peripheral_features(baseline)
            # Only the mean level is baseline-corrected. Subtracting a 3-second
            # window's slope or range from a 60-second window's is not a
            # correction, it is noise: the statistics are not comparable across
            # window lengths.
            trial_stats = trial_stats.copy()
            trial_stats[..., 0] -= base_stats[..., 0]

        names = [
            f"{channel}_{stat}"
            for channel in DEAP_PERIPHERAL_CHANNELS
            for stat in stat_names
        ]
        return ModalityBlock(
            name=PERIPHERAL,
            X=trial_stats.reshape(trial_stats.shape[0], -1),
            subject_ids=subjects,
            stimulus_ids=stimulus_ids,
            feature_names=names,
            provenance=self._provenance(
                source="deap_preprocessed_python",
                channels=list(DEAP_PERIPHERAL_CHANNELS),
                statistics=list(stat_names),
                baseline_corrected_mean_only=self.baseline_correct,
                caveat=(
                    "generic summary statistics; GSR should be decomposed into "
                    "tonic and phasic components and the plethysmograph into "
                    "inter-beat intervals before any autonomic claim"
                ),
            ),
        )

    def _behavior_block(
        self,
        labels: np.ndarray,
        subjects: np.ndarray,
        stimulus_ids: np.ndarray,
        familiarity: Optional[np.ndarray],
    ) -> ModalityBlock:
        columns: List[np.ndarray] = []
        names: List[str] = []

        if familiarity is not None:
            columns.append(np.asarray(familiarity, dtype=float))
            names.append("familiarity")

        # Position in the participant's randomised sequence. Captures fatigue
        # and habituation, and is not a self-report about the stimulus.
        columns.append(np.arange(DEAP_N_TRIALS, dtype=float))
        names.append("trial_index")

        if self.behavior_mode == "ratings":
            for i, label_name in enumerate(DEAP_LABEL_NAMES):
                if label_name == self.target:
                    continue  # never a feature for its own prediction
                columns.append(labels[:, i])
                names.append(f"rating_{label_name}")

        return ModalityBlock(
            name=BEHAVIOR,
            X=np.column_stack(columns),
            subject_ids=subjects,
            stimulus_ids=stimulus_ids,
            feature_names=names,
            provenance=self._provenance(
                source="deap_ratings",
                behavior_mode=self.behavior_mode,
                target_excluded=self.target,
                circular=self.behavior_mode == "ratings",
                caveat=(
                    "SAM ratings were collected on the same screen as the target, "
                    "seconds apart, from the same person. Strong performance here "
                    "is self-report predicting self-report and is not comparable "
                    "to the behavioural comparator in the neuroforecasting papers"
                )
                if self.behavior_mode == "ratings"
                else None,
            ),
        )


# ------------------------------------------------------------- metadata files


def load_participant_ratings(path: str) -> Any:
    """Read ``participant_ratings.csv`` and normalise its column names.

    The published column names are not something this loader should assume it
    remembers correctly, so matching is case-insensitive and the error names
    every column actually present.

    Returns a DataFrame with lowercase columns, guaranteed to contain
    ``participant_id``, ``trial``, and ``experiment_id``.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reading DEAP metadata requires pandas (`pip install pandas`)"
        ) from exc

    table = pd.read_csv(path)
    table.columns = [str(c).strip().lower().replace(" ", "_") for c in table.columns]

    required = {"participant_id", "trial", "experiment_id"}
    missing = required - set(table.columns)
    if missing:
        raise DEAPFormatError(
            f"participant_ratings.csv is missing {sorted(missing)}. "
            f"Found columns: {sorted(table.columns)}. "
            "Experiment_id is the stimulus key: without it, trial order cannot "
            "be mapped to videos and the aggregate arm is impossible."
        )
    return table


def stimulus_ids_for_subject(ratings: Any, participant_id: int) -> np.ndarray:
    """Return the 40 ``experiment_id`` values in that participant's trial order.

    DEAP randomises presentation order per participant, so this mapping is the
    only thing connecting a row of the data array to a video.
    """
    rows = ratings[ratings["participant_id"] == participant_id].sort_values("trial")
    if len(rows) != DEAP_N_TRIALS:
        raise DEAPFormatError(
            f"participant {participant_id} has {len(rows)} rows in "
            "participant_ratings.csv, expected "
            f"{DEAP_N_TRIALS}"
        )
    return np.array(
        [f"exp-{int(e):02d}" for e in rows["experiment_id"].to_numpy()]
    )


def familiarity_for_subject(ratings: Any, participant_id: int) -> Optional[np.ndarray]:
    """Familiarity ratings in trial order, or ``None`` if the column is absent."""
    if "familiarity" not in ratings.columns:
        return None
    rows = ratings[ratings["participant_id"] == participant_id].sort_values("trial")
    return rows["familiarity"].to_numpy(dtype=float)


def load_video_list(path: str) -> Any:
    """Read ``video_list.csv``, normalising column names.

    This file carries the YouTube links, which is the only route DEAP offers to
    a real population-level outcome. See
    :mod:`behavioral_decoding.io.deap_market`.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reading DEAP metadata requires pandas") from exc

    table = pd.read_csv(path)
    table.columns = [str(c).strip().lower().replace(" ", "_") for c in table.columns]
    if "experiment_id" not in table.columns:
        raise DEAPFormatError(
            f"video_list.csv has no experiment_id column; found {sorted(table.columns)}"
        )
    return table


# ------------------------------------------------------------ binarisation


def binarise_ratings(
    values: np.ndarray,
    subject_ids: np.ndarray,
    method: str = "fixed",
    threshold: float = 5.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Turn 1-9 SAM ratings into binary labels.

    This is the single largest analytic degree of freedom in any DEAP analysis,
    and published accuracies are not comparable across choices of it. Fix the
    method before looking at results.

    ``fixed``
        Split at ``threshold`` (conventionally 5.0, the scale midpoint). Keeps
        labels comparable across participants and produces real, participant-
        specific class imbalance, which is what the resampling machinery is for.

    ``subject_median``
        Split at each participant's own median. Guarantees balanced classes
        within participant and removes individual differences in scale use. It
        also changes what the label *means*: "high for this person" rather than
        "high". A model can then look accurate by learning a participant's
        response style. Use only with subject-grouped CV, which this framework
        enforces anyway.

    Ratings exactly equal to the threshold are assigned to the **low** class, so
    the split is reproducible; that choice matters because 5.0 is a common
    response on a 9-point scale.
    """
    values = np.asarray(values, dtype=float)
    subject_ids = np.asarray(subject_ids)

    if method == "fixed":
        labels = (values > threshold).astype(int)
        info: Dict[str, object] = {"method": "fixed", "threshold": threshold}
    elif method == "subject_median":
        labels = np.zeros(len(values), dtype=int)
        medians: Dict[str, float] = {}
        for subject in np.unique(subject_ids):
            mask = subject_ids == subject
            median = float(np.median(values[mask]))
            medians[str(subject)] = median
            labels[mask] = (values[mask] > median).astype(int)
        info = {"method": "subject_median", "subject_medians": medians}
    else:
        raise ValueError("method must be 'fixed' or 'subject_median'")

    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError(
            f"binarisation produced a single class ({counts[0]} of {len(labels)} trials). With "
            f"method={method!r} and threshold={threshold}, every rating fell on one side."
        )

    info["positive_rate"] = float(labels.mean())
    info["imbalance_ratio"] = float(counts.max()) / float(counts.min())
    return labels, info


# -------------------------------------------------------------- orchestration


def load_deap(
    root: str,
    target: str = "liking",
    subjects: Optional[Sequence[str]] = None,
    binarise: str = "fixed",
    threshold: float = 5.0,
    loader: Optional[DEAPLoader] = None,
    y_aggregate: Optional[Dict[object, float]] = None,
    ratings_path: Optional[str] = None,
):
    """Load DEAP into a :class:`MultimodalDataset`.

    Parameters
    ----------
    root:
        Directory containing ``sNN.dat`` files (or a ``data_preprocessed_python``
        subdirectory), plus ``metadata_csv/participant_ratings.csv``.
    target:
        Which rating becomes the binary outcome.
    y_aggregate:
        Per-stimulus market outcome, keyed by the same ``exp-NN`` ids this
        function produces. DEAP ships none; see
        :mod:`behavioral_decoding.io.deap_market` for the YouTube route.

    Notes
    -----
    The face-video modality is not assembled here. It lives in separate archives,
    covers only 22 of 32 participants, and needs frame extraction plus the ViT
    encoder. Including it would force an inner join that discards a third of the
    participants for every analysis, whether or not the face arm is being used.
    Build it as a separate block and join it deliberately.
    """
    from ..features.align import build_dataset

    active = loader or DEAPLoader(target=target)
    if active.target != target:
        raise ValueError(
            f"target mismatch: load_deap(target={target!r}) but the supplied loader has "
            f"target={active.target!r}. The loader's target controls which rating is excluded "
            "from the behaviour block, so a mismatch would leak the outcome into "
            "the features."
        )

    root_path = Path(root)
    if ratings_path is None:
        for candidate in (
            root_path / "metadata_csv" / "participant_ratings.csv",
            root_path / "participant_ratings.csv",
        ):
            if candidate.exists():
                ratings_path = str(candidate)
                break
    if ratings_path is None:
        raise FileNotFoundError(
            f"participant_ratings.csv not found under {root}. It supplies Experiment_id, "
            "which is the only mapping from a participant's randomised trial order "
            "to the 40 videos. Without it the stimulus keys cannot be built and "
            "aggregate forecasting is impossible."
        )

    ratings = load_participant_ratings(ratings_path)
    files = find_subject_files(root)
    if subjects is not None:
        wanted = {s.lower() for s in subjects}
        files = [f for f in files if subject_id_from_path(f) in wanted]
        if not files:
            raise ValueError(f"none of the requested subjects were found: {subjects}")

    target_index = DEAP_LABEL_NAMES.index(target)

    all_blocks: Dict[str, List[ModalityBlock]] = {}
    y_raw: List[np.ndarray] = []
    subj_keys: List[np.ndarray] = []
    stim_keys: List[np.ndarray] = []

    for path in progress(files, desc="DEAP participants", total=len(files)):
        subject_id = subject_id_from_path(path)
        participant_number = int(subject_id[1:])
        data, labels = load_subject_file(str(path))

        stimulus_ids = stimulus_ids_for_subject(ratings, participant_number)
        familiarity = familiarity_for_subject(ratings, participant_number)

        blocks, label_array = active.blocks_from_arrays(
            data,
            labels,
            subject_id=subject_id,
            stimulus_ids=stimulus_ids,
            familiarity=familiarity,
        )
        for modality, block in blocks.items():
            all_blocks.setdefault(modality, []).append(block)

        y_raw.append(label_array[:, target_index])
        subj_keys.append(np.array([subject_id] * DEAP_N_TRIALS))
        stim_keys.append(stimulus_ids)

    merged = {
        modality: ModalityBlock(
            name=modality,
            X=np.vstack([b.X for b in blocks_list]),
            subject_ids=np.concatenate([b.subject_ids for b in blocks_list]),
            stimulus_ids=np.concatenate([b.stimulus_ids for b in blocks_list]),
            feature_names=list(blocks_list[0].feature_names or []),
            provenance=dict(blocks_list[0].provenance),
        )
        for modality, blocks_list in all_blocks.items()
    }

    subjects_flat = np.concatenate(subj_keys)
    stimuli_flat = np.concatenate(stim_keys)
    ratings_flat = np.concatenate(y_raw)

    y_binary, binarisation_info = binarise_ratings(
        ratings_flat, subjects_flat, method=binarise, threshold=threshold
    )
    logger.info(
        "DEAP: target=%s, binarisation=%s, positive rate %.1f%%, imbalance %.2f:1",
        target,
        binarise,
        100 * binarisation_info["positive_rate"],
        binarisation_info["imbalance_ratio"],
    )

    y_individual = {
        (s, st): int(y) for s, st, y in zip(subjects_flat, stimuli_flat, y_binary)
    }

    return build_dataset(
        merged,
        y_individual=y_individual,
        y_aggregate=y_aggregate,
        how="inner",
        metadata={
            "dataset": "DEAP",
            "synthetic": False,
            "target": target,
            "binarisation": binarisation_info,
            "n_participants": len(files),
            "behavior_mode": active.behavior_mode,
            "citation": "Koelstra et al. (2012), IEEE Trans. Affective Computing, "
            "doi:10.1109/T-AFFC.2011.15",
            "aggregate_outcome": "supplied" if y_aggregate else "none",
        },
    )
