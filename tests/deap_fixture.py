"""Write a tiny DEAP-format download to disk, for tests and the demo.

This produces the real on-disk layout: latin1-pickled ``sNN.dat`` files each
holding a ``{'data': (40,40,8064), 'labels': (40,4)}`` dict, plus a
``metadata_csv/participant_ratings.csv`` and a ``video_list.csv``. The point is
to exercise the loader against the actual file format, including the Python-2
pickle encoding and the per-participant randomised trial order, without the
licensed dataset.

The signal is planted, not random. A latent per-video valence drives both a
frontal-asymmetry effect in the EEG and the market outcome, so the loader plus
the pipeline can be checked for the same brain-beats-behaviour ordering the
framework is built around. It is simulated. No number from it is a finding about
brains.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from behavioral_decoding.io.deap import (
    DEAP_BASELINE_SAMPLES,
    DEAP_EEG_CHANNELS,
    DEAP_N_CHANNELS,
    DEAP_N_SAMPLES,
    DEAP_N_TRIALS,
    DEAP_SFREQ,
)


def _sine(freq: float, n: int, sfreq: float, rng: np.random.Generator) -> np.ndarray:
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(n) / sfreq
    return np.sin(2 * np.pi * freq * t + phase)


def write_deap_fixture(
    root: str,
    n_participants: int = 6,
    n_videos: int = DEAP_N_TRIALS,
    seed: int = 0,
) -> Dict[str, object]:
    """Write a synthetic DEAP tree under ``root``. Returns its ground truth."""
    root_path = Path(root)
    data_dir = root_path / "data_preprocessed_python"
    meta_dir = root_path / "metadata_csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Latent per-video valence: the thing the market responds to and the thing
    # frontal asymmetry tracks. Videos are indexed 1..n_videos (Experiment_id).
    video_valence = rng.normal(0.0, 1.0, size=n_videos)

    # Market outcome: a noisy function of valence alone, in raw "view count"
    # units so the loader's log path is exercised.
    market_raw = np.exp(9.0 + 1.4 * video_valence + rng.normal(0, 0.4, size=n_videos))
    market_by_exp = {
        f"exp-{v + 1:02d}": float(market_raw[v]) for v in range(n_videos)
    }

    # Frontal channels whose asymmetry should carry valence.
    left_idx = DEAP_EEG_CHANNELS.index("F3")
    right_idx = DEAP_EEG_CHANNELS.index("F4")

    ratings_rows = []
    for p in range(1, n_participants + 1):
        subject_seed = seed * 1000 + p
        srng = np.random.default_rng(subject_seed)

        # Per-participant randomised presentation order over the videos.
        order = srng.permutation(n_videos)  # order[trial] -> video index (0-based)

        subject_bias = srng.normal(0.0, 1.0)  # individual scale-use offset

        data = np.zeros((DEAP_N_TRIALS, DEAP_N_CHANNELS, DEAP_N_SAMPLES), dtype=np.float32)
        labels = np.zeros((DEAP_N_TRIALS, 4), dtype=np.float32)

        for trial in range(DEAP_N_TRIALS):
            video = order[trial % n_videos]
            v = video_valence[video]

            # Baseline: pink-ish noise, all channels.
            base = srng.normal(0, 1.0, size=(DEAP_N_CHANNELS, DEAP_BASELINE_SAMPLES))
            trial_len = DEAP_N_SAMPLES - DEAP_BASELINE_SAMPLES
            body = srng.normal(0, 1.0, size=(DEAP_N_CHANNELS, trial_len))

            # Plant alpha (10 Hz) asymmetry: higher valence -> more left-frontal
            # activity -> LESS left alpha, so right-minus-left log alpha rises.
            alpha_left = (1.2 - 0.5 * v) * _sine(10.0, trial_len, DEAP_SFREQ, srng)
            alpha_right = (1.2 + 0.5 * v) * _sine(10.0, trial_len, DEAP_SFREQ, srng)
            body[left_idx] += alpha_left
            body[right_idx] += alpha_right

            # A GSR-like slow drift (channel 37, index 36) that tracks arousal,
            # which we tie loosely to |valence| so the peripheral block has some
            # signal too.
            gsr_idx = len(DEAP_EEG_CHANNELS) + 4
            drift = np.linspace(0, 0.8 * abs(v), trial_len)
            body[gsr_idx] += drift

            data[trial, :, :DEAP_BASELINE_SAMPLES] = base
            data[trial, :, DEAP_BASELINE_SAMPLES:] = body

            # SAM ratings. Valence tracks latent v plus the participant's bias;
            # the others are noisier. Clipped to the 1..9 scale.
            valence = np.clip(5 + 1.6 * v + 0.8 * subject_bias + srng.normal(0, 0.7), 1, 9)
            arousal = np.clip(5 + 1.2 * abs(v) + srng.normal(0, 1.0), 1, 9)
            dominance = np.clip(5 + 0.5 * v + srng.normal(0, 1.2), 1, 9)
            liking = np.clip(5 + 1.4 * v + 0.6 * subject_bias + srng.normal(0, 0.9), 1, 9)
            labels[trial] = [valence, arousal, dominance, liking]

            familiarity = int(np.clip(round(3 + srng.normal(0, 1)), 1, 5))
            ratings_rows.append(
                {
                    "Participant_id": p,
                    "Trial": trial + 1,
                    "Experiment_id": int(video + 1),
                    "Familiarity": familiarity,
                    "Valence": float(valence),
                    "Arousal": float(arousal),
                    "Dominance": float(dominance),
                    "Liking": float(liking),
                }
            )

        with open(data_dir / f"s{p:02d}.dat", "wb") as handle:
            # Protocol 2 + latin1-decodable: mimic the Python-2 origin.
            pickle.dump({"data": data, "labels": labels}, handle, protocol=2)

    _write_csv(
        meta_dir / "participant_ratings.csv",
        ratings_rows,
        ["Participant_id", "Trial", "Experiment_id", "Familiarity",
         "Valence", "Arousal", "Dominance", "Liking"],
    )

    video_rows = [
        {
            "Experiment_id": v + 1,
            "Title": f"video {v + 1}",
            "YouTube_link": f"https://www.youtube.com/watch?v={_fake_youtube_id(v, rng)}",
        }
        for v in range(n_videos)
    ]
    _write_csv(root_path / "video_list.csv", video_rows,
               ["Experiment_id", "Title", "YouTube_link"])

    return {
        "root": str(root_path),
        "n_participants": n_participants,
        "n_videos": n_videos,
        "video_valence": video_valence,
        "market_by_exp": market_by_exp,
        "expected_ordering": (
            "EEG frontal asymmetry should forecast the market outcome better "
            "than familiarity/trial-order, because asymmetry tracks the latent "
            "video valence that drives the market"
        ),
    }


def _fake_youtube_id(v: int, rng: np.random.Generator) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
    # Deterministic per video so the same fixture seed gives the same ids.
    local = np.random.default_rng(20120000 + v)
    return "".join(alphabet[i] for i in local.integers(0, len(alphabet), size=11))


def _write_csv(path: Path, rows, columns) -> None:
    import csv

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def market_outcome_dict(ground_truth: Dict[str, object], log_transform: bool = True) -> Dict[str, float]:
    """Convenience: ``exp-NN -> (log) view count`` from a fixture's ground truth."""
    raw: Dict[str, float] = ground_truth["market_by_exp"]  # type: ignore[assignment]
    if log_transform:
        return {k: float(np.log1p(v)) for k, v in raw.items()}
    return dict(raw)
