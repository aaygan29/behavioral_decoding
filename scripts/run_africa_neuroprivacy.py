#!/usr/bin/env python3
"""Neuroprivacy demonstration: decode an undisclosed clinical trait from routine EEG.

Real data: ASZED-153, 16-20ch EEG from Nigerian patients and matched controls
(OAUTHC, Ile-Ife/Ilesa), CC-BY on Zenodo. Fetch with
``scripts/fetch_africa_neuroprivacy.py`` first.

This is not the market-forecasting design used elsewhere in this repo. There
is no aggregate/market arm here, on purpose: a diagnosis is a property of a
person, not of a stimulus, so there is nothing to hold out and forecast at a
group level. What this script tests is narrower and, for a safety argument,
sharper: each recording session has four short EEG segments (Phase 1-4: two
brief baseline-length segments and two longer task/passive-listening
segments, per the dataset's own session metadata). Each phase is treated as
its own "modality" and run through the *same* multibagging ensemble this repo
uses to reconcile fMRI/EEG/face/behaviour -- accuracy-weighted, per-modality
bagged elastic-net, out-of-fold calibration, the works. The label predicted is
not the phase's own designed observable; it's diagnostic category (patient vs.
matched control), which nothing about "we recorded four minutes of EEG"
inherently discloses to a subject as being extractable from that recording.

That is the triangulation: (1) the schizophrenia EEG literature has repeatedly
tied specific spectral signatures -- reduced gamma-band phase-locking to
auditory stimulation, frontal/resting delta-theta elevation, alpha reduction
-- to this diagnosis; (2) real multi-paradigm EEG from an African cohort is
pulled here and run through this repo's own band-power feature family and
ensemble, with no dataset-specific tuning; (3) if diagnostic status comes out
decodable above chance with a handful of short, cheap EEG segments, that is a
direct, empirical demonstration of the neuroprivacy risk this repo's "Global
South and Africa" framing gestures at: brief, low-cost EEG -- of the kind a
consumer wearable or an unrelated study could plausibly collect -- can leak a
protected health attribute the subject did not consent to reveal, in a
population with weaker legal protection against exactly that use.

Usage:
    python scripts/fetch_africa_neuroprivacy.py
    python scripts/run_africa_neuroprivacy.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from behavioral_decoding.config import ExperimentConfig  # noqa: E402
from behavioral_decoding.features.align import build_dataset, summarise_alignment  # noqa: E402
from behavioral_decoding.io.eeg import EEGLoader  # noqa: E402
from behavioral_decoding.pipelines.train import run_experiment  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "raw" / "aszed"
EPOCH_SECONDS = 2.0
N_PHASES = 4
# The archive mixes two recording devices at different native sample rates
# (Contec KT-2400 at 200 Hz, BrainMaster Discovery24-E at 256 Hz), and even
# within one device some sessions were resampled upstream. Every file is
# resampled to this common rate before epoching so epoch sample counts (and
# therefore feature-column counts) are identical across subjects.
TARGET_SFREQ = 200.0

# Canonical 19-channel international 10/20 EEG montage. Raw files carry these
# names with a bracketed index suffix (e.g. "Fp1[1]") and sometimes extra
# non-EEG channels (Pg1/Pg2 ear reference, EOG, EMG) appended after them, and
# not always the same extras in the same order. Normalising to this fixed,
# named set (rather than picking the first N channels) is what keeps the
# feature columns comparable across subjects and devices.
CANONICAL_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Pz", "Cz",
]


def _find_spreadsheet() -> Path:
    hit = next(DATA_ROOT.rglob("ASZED_SpreadSheet.csv"), None)
    if hit is None:
        raise FileNotFoundError(
            f"no ASZED_SpreadSheet.csv under {DATA_ROOT}. "
            "Run scripts/fetch_africa_neuroprivacy.py first."
        )
    return hit


def _subject_labels() -> dict[str, int]:
    labels = {}
    with _find_spreadsheet().open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["category"] not in ("Patient", "Control"):
                continue
            labels[row["sn"]] = 1 if row["category"] == "Patient" else 0
    return labels


def _first_full_session(subject_dir: Path) -> Path | None:
    """The first session directory with exactly N_PHASES Phase-*.edf files."""
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        phases = sorted(session_dir.glob("Phase *.edf"))
        if len(phases) == N_PHASES:
            return session_dir
    return None


def _epoch_phase(edf_path: Path, epoch_seconds: float):
    """Read one Phase EDF and cut it into fixed-length, non-overlapping epochs."""
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    def _base(name: str) -> str:
        return name.split("[")[0].strip()

    by_base = {_base(ch): ch for ch in raw.ch_names}
    missing = [c for c in CANONICAL_CHANNELS if c not in by_base]
    if missing:
        return None
    raw.pick([by_base[c] for c in CANONICAL_CHANNELS])
    raw.reorder_channels([by_base[c] for c in CANONICAL_CHANNELS])
    if abs(float(raw.info["sfreq"]) - TARGET_SFREQ) > 1e-6:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.resample(TARGET_SFREQ)

    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()  # (n_channels, n_times), fixed 19-channel order
    samples_per_epoch = int(round(epoch_seconds * sfreq))
    n_epochs = data.shape[1] // samples_per_epoch
    if n_epochs == 0:
        return None
    trimmed = data[:, : n_epochs * samples_per_epoch]
    epochs = trimmed.reshape(data.shape[0], n_epochs, samples_per_epoch).transpose(1, 0, 2)
    times = np.arange(samples_per_epoch) / sfreq
    return epochs, sfreq, times, raw.ch_names


def build_blocks(max_subjects: int | None = None):
    labels = _subject_labels()
    # de-duplicate: the same subject id can appear under more than one subset
    # directory in the archive; keep the first hit per id.
    seen = set()
    unique_dirs = []
    for d in DATA_ROOT.rglob("subject_*"):
        if not d.is_dir() or d.name in seen:
            continue
        seen.add(d.name)
        unique_dirs.append(d)
    unique_dirs.sort(key=lambda p: p.name)

    # phase_idx -> subject_id -> epochs
    phase_epochs: list[dict] = [dict() for _ in range(N_PHASES)]
    phase_meta = [None] * N_PHASES
    channel_ref = None
    n_used = 0

    for subject_dir in unique_dirs:
        sn = subject_dir.name
        if sn not in labels:
            continue
        session_dir = _first_full_session(subject_dir)
        if session_dir is None:
            continue

        phase_files = sorted(session_dir.glob("Phase *.edf"))
        per_phase = []
        ok = True
        for pf in phase_files:
            result = _epoch_phase(pf, EPOCH_SECONDS)
            if result is None:
                ok = False
                break
            epochs, sfreq, times, ch_names = result
            if channel_ref is None:
                channel_ref = ch_names
            if ch_names[: len(channel_ref)] != channel_ref[: len(ch_names)]:
                pass  # channel order can differ slightly by device; handled by column count below
            per_phase.append((epochs, sfreq, times))
        if not ok or len(per_phase) != N_PHASES:
            continue

        n_shared = min(e.shape[0] for e, _, _ in per_phase)
        if n_shared == 0:
            continue

        for i, (epochs, sfreq, times) in enumerate(per_phase):
            phase_epochs[i][sn] = epochs[:n_shared]
            if phase_meta[i] is None:
                phase_meta[i] = (sfreq, times)

        n_used += 1
        if max_subjects and n_used >= max_subjects:
            break

    return phase_epochs, phase_meta, labels, n_used


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--name", default="africa_neuroprivacy_aszed")
    args = parser.parse_args()

    print("loading ASZED-153 (subset_1, first full 4-phase session per subject)...")
    phase_epochs, phase_meta, labels, n_used = build_blocks(args.max_subjects)
    print(f"{n_used} subjects with a complete 4-phase session")

    loader = EEGLoader(include_erp=False)
    blocks = {}
    for phase_idx in range(N_PHASES):
        sfreq, times = phase_meta[phase_idx]
        subj_epochs = phase_epochs[phase_idx]
        X_parts, subj_ids, stim_ids = [], [], []
        for sn, epochs in subj_epochs.items():
            for e_idx in range(epochs.shape[0]):
                X_parts.append(epochs[e_idx : e_idx + 1])
                subj_ids.append(sn)
                stim_ids.append(f"ep{e_idx}")
        all_epochs = np.concatenate(X_parts, axis=0)
        block = loader.from_arrays(
            all_epochs, sfreq, times, subj_ids, stim_ids,
            source=f"aszed_phase{phase_idx + 1}",
        )
        block.name = f"phase{phase_idx + 1}"
        blocks[block.name] = block

    print(summarise_alignment(blocks))

    y_individual = {}
    for _name, block in blocks.items():
        for s, st in zip(block.subject_ids, block.stimulus_ids):
            y_individual[(s, st)] = float(labels[s])
        break  # keys are shared across all phase blocks by construction

    dataset = build_dataset(
        blocks,
        y_individual=y_individual,
        y_aggregate=None,
        metadata={
            "source": "ASZED-153 (Zenodo 14178398, CC-BY)",
            "cohort": "Nigeria (OAUTHC Ile-Ife / Wesley Guild Ilesa)",
            "n_subjects": n_used,
            "purpose": ("neuroprivacy risk demonstration: undisclosed-trait "
                        "decoding from routine EEG"),
        },
    )

    print(dataset.describe())
    print()

    cfg = ExperimentConfig(name=args.name, output_dir=args.output_dir)
    cfg.data.modalities = list(blocks.keys())
    cfg.data.outcome_column = "diagnosis"
    for m in cfg.data.modalities:
        cfg.model.base_learner[m] = "elasticnet"
        cfg.model.n_bags[m] = 25

    record = run_experiment(dataset, cfg)

    print()
    print("=" * 78)
    print("AFRICA NEUROPRIVACY DEMONSTRATION (ASZED-153, real Nigerian EEG cohort)")
    print("predicting: diagnostic category (patient vs. matched control),")
    print("            NOT the recording's own designed observable")
    print("=" * 78)
    m = record["individual"]["pooled_metrics"]
    print(f"  n subjects (grouped CV)     {n_used}")
    print(f"  n trials (epochs)           {int(m['n'])}")
    print(f"  majority baseline accuracy  {m['majority_baseline_accuracy']:.3f}")
    print(f"  balanced accuracy           {m['balanced_accuracy']:.3f}")
    print(f"  ROC AUC                     {m['roc_auc']:.3f}")
    print(f"  Brier / ECE                 {m['brier']:.3f} / {m['ece']:.3f}")
    ci = record["individual"]["bootstrap_ci"]
    print(f"  bootstrap 95% CI            {ci['lo']:.3f} - {ci['hi']:.3f}")
    perm = record["individual"]["permutation_test"]
    print(f"  permutation p               {perm['p_value']:.4f}  (n_perm={int(perm['n_perm'])})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
