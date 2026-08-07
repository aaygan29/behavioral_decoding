"""Synthetic multimodal data with a known ground truth.

This is not filler to make the demo run. It is the positive control for the
whole framework, and it encodes one specific structure on purpose:

* Every stimulus has a latent affective value ``v_s``. The market outcome is a
  noisy function of ``v_s`` alone.
* The **generalisable** neural channel (call it NAcc) tracks ``v_s`` with
  subject-independent noise. Averaged across a handful of subjects it recovers
  ``v_s``, so it forecasts the market.
* The **integrative** neural channel (call it MPFC) tracks ``v_s`` *plus* a
  large subject-specific term. It predicts what that individual will choose,
  but its group average is polluted by subject idiosyncrasy, so it forecasts
  the market poorly.
* Self-report behaviour is driven mostly by the subject-specific term. Strong
  individual prediction, weak market forecast.

That is the Genevsky/Knutson dissociation, written down as a generative model.
A framework that cannot recover it on data where it is true by construction has
no business being pointed at real recordings. ``scripts/run_demo.py`` checks
exactly this, and the check is a test, not a demo.

Everything here is simulated. No number produced from this module is a finding
about brains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .io.base import BEHAVIOR, EEG, FACE, FMRI, ModalityBlock, MultimodalDataset


@dataclass
class SyntheticConfig:
    n_subjects: int = 24
    n_stimuli: int = 60
    positive_rate: float = 0.22

    # How strongly each channel carries the stimulus-general signal ``v_s``
    # versus subject-specific idiosyncrasy.
    nacc_stimulus_weight: float = 1.0
    nacc_subject_weight: float = 0.15
    mpfc_stimulus_weight: float = 0.75
    mpfc_subject_weight: float = 1.10
    behavior_subject_weight: float = 1.30

    # Choice model
    choice_nacc_beta: float = 0.9
    choice_mpfc_beta: float = 1.2
    choice_behavior_beta: float = 1.0

    # Noise
    fmri_noise: float = 0.9
    eeg_noise: float = 1.6
    face_noise: float = 1.2
    behavior_noise: float = 0.7
    market_noise: float = 0.55

    # Dimensionality
    n_eeg_features: int = 40
    n_face_features: int = 64
    n_behavior_features: int = 5

    seed: int = 0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(logits: np.ndarray, target_rate: float) -> float:
    """Find the intercept that yields the requested positive rate."""
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        rate = float(np.mean(_sigmoid(logits + mid)))
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_synthetic_dataset(
    config: Optional[SyntheticConfig] = None,
) -> Tuple[MultimodalDataset, Dict[str, object]]:
    """Generate an aligned multimodal dataset plus its ground truth.

    Returns
    -------
    (dataset, ground_truth)
        ``ground_truth`` records the latent stimulus values and the generating
        weights, so a recovery check can compare against what was actually put in.
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    subject_ids = np.array([f"sub-{i:02d}" for i in range(cfg.n_subjects)])
    stimulus_ids = np.array([f"stim-{i:03d}" for i in range(cfg.n_stimuli)])

    # Latent stimulus value: the thing the market responds to.
    v = rng.normal(0.0, 1.0, size=cfg.n_stimuli)
    # Subject-specific taste: real, but it does not generalise to the market.
    subject_bias = rng.normal(0.0, 1.0, size=cfg.n_subjects)
    # Subject-by-stimulus interaction: idiosyncratic preference.
    interaction = rng.normal(0.0, 1.0, size=(cfg.n_subjects, cfg.n_stimuli))

    n_trials = cfg.n_subjects * cfg.n_stimuli
    subj_idx = np.repeat(np.arange(cfg.n_subjects), cfg.n_stimuli)
    stim_idx = np.tile(np.arange(cfg.n_stimuli), cfg.n_subjects)

    trial_subjects = subject_ids[subj_idx]
    trial_stimuli = stimulus_ids[stim_idx]

    v_trial = v[stim_idx]
    bias_trial = subject_bias[subj_idx]
    inter_trial = interaction[subj_idx, stim_idx]

    # --- fMRI: NAcc (generalisable) and MPFC (integrative), plus two nuisance ROIs
    nacc = (
        cfg.nacc_stimulus_weight * v_trial
        + cfg.nacc_subject_weight * bias_trial
        + cfg.fmri_noise * rng.normal(size=n_trials)
    )
    mpfc = (
        cfg.mpfc_stimulus_weight * v_trial
        + cfg.mpfc_subject_weight * (bias_trial + inter_trial)
        + cfg.fmri_noise * rng.normal(size=n_trials)
    )
    ains = -0.5 * v_trial + cfg.fmri_noise * rng.normal(size=n_trials)
    nuisance = cfg.fmri_noise * rng.normal(size=n_trials)

    fmri_X = np.column_stack([nacc, mpfc, ains, nuisance])
    fmri_names = ["NAcc", "MPFC", "AIns", "control_ROI"]

    # --- EEG: a few informative channels buried in noise
    eeg_X = cfg.eeg_noise * rng.normal(size=(n_trials, cfg.n_eeg_features))
    eeg_X[:, 0] += 0.8 * v_trial
    eeg_X[:, 1] += 0.6 * (v_trial + 0.4 * bias_trial)
    eeg_X[:, 2] += 0.5 * inter_trial

    # --- Face: sparse signal in a high-dimensional embedding
    face_X = cfg.face_noise * rng.normal(size=(n_trials, cfg.n_face_features))
    face_signal = 0.7 * v_trial + 0.5 * inter_trial
    for j in range(6):
        face_X[:, j] += face_signal * (0.9 ** j)

    # --- Behaviour: dominated by subject-specific taste
    behavior_core = (
        cfg.behavior_subject_weight * (bias_trial + inter_trial)
        + 0.45 * v_trial
        + cfg.behavior_noise * rng.normal(size=n_trials)
    )
    behavior_X = np.column_stack(
        [behavior_core]
        + [
            behavior_core * (0.8 ** k) + cfg.behavior_noise * rng.normal(size=n_trials)
            for k in range(1, cfg.n_behavior_features)
        ]
    )
    behavior_names = ["rating_valence", "rating_arousal", "rt", "confidence", "familiarity"][
        : cfg.n_behavior_features
    ]

    # --- Individual choice
    logits = (
        cfg.choice_nacc_beta * nacc
        + cfg.choice_mpfc_beta * mpfc
        + cfg.choice_behavior_beta * behavior_core
    )
    logits = (logits - logits.mean()) / logits.std()
    intercept = _calibrate_intercept(logits, cfg.positive_rate)
    y_individual = (rng.uniform(size=n_trials) < _sigmoid(logits + intercept)).astype(int)

    # --- Market outcome: a function of v alone, observed outside the lab
    market = v + cfg.market_noise * rng.normal(size=cfg.n_stimuli)
    y_aggregate = {stim: float(m) for stim, m in zip(stimulus_ids, market)}

    def block(name: str, X: np.ndarray, names) -> ModalityBlock:
        return ModalityBlock(
            name=name,
            X=X,
            subject_ids=trial_subjects,
            stimulus_ids=trial_stimuli,
            feature_names=list(names),
            provenance={
                "source": "synthetic",
                "generator": "behavioral_decoding.synthetic.make_synthetic_dataset",
                "seed": cfg.seed,
                "reportable": False,
                "note": "simulated data; not evidence about brains",
            },
        )

    dataset = MultimodalDataset(
        blocks={
            FMRI: block(FMRI, fmri_X, fmri_names),
            EEG: block(EEG, eeg_X, [f"eeg_{i:02d}" for i in range(cfg.n_eeg_features)]),
            FACE: block(FACE, face_X, [f"vit_{i:02d}" for i in range(cfg.n_face_features)]),
            BEHAVIOR: block(BEHAVIOR, behavior_X, behavior_names),
        },
        y_individual=y_individual,
        subject_ids=trial_subjects,
        stimulus_ids=trial_stimuli,
        y_aggregate=y_aggregate,
        metadata={"synthetic": True, "config": cfg.__dict__},
    )

    ground_truth: Dict[str, object] = {
        "latent_stimulus_value": v,
        "subject_bias": subject_bias,
        "market_outcome": market,
        "expected_ordering": (
            "brain_only should forecast the market better than behavior_only, "
            "because NAcc carries stimulus-general signal while behaviour is "
            "dominated by subject-specific taste"
        ),
        "config": cfg.__dict__,
    }
    return dataset, ground_truth
