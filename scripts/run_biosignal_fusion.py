"""Multi-biosignal fusion: does combining biosignal families beat the best single one?

Motivation
----------
The privacy work in ``scripts/run_biosignal_privacy.py`` established that a single
non-neural biosignal family (fingertip PPG) leaks health status weakly through the
repo's own bagged/SMOTE/OOF-weighted pipeline. It did **not** test the claim that
matters most for the AIxBio biorisk argument: that an actor who reconciles *several*
biosignal families at once gets a prediction stronger than any family alone. The
prior PPG run could not test this honestly, because its three "segments" were repeats
of one measurement, not distinct modalities.

This script tests the fusion claim on a generator where fusion *should* help by
construction, then checks that the repo's real ``MultimodalEnsemble`` actually
recovers the gain. It is a positive control for the aggregation machinery, not a
claim about any real person.

Design (grounded in the biosignal-to-behaviour literature, see
``docs/biosignal_fusion.md``)
------------------------------------------------------------------------------------
A binary choice is driven by four partly-independent latent drivers, each of which a
different biosignal family senses best, plus a family that senses nothing:

  driver              sensed best by      literature anchor
  ------              --------------      -----------------
  autonomic arousal   cardiac (HRV), eda  Forte 2021; Dunn 2005 (somatic marker)
  noradrenergic vol.  pupil               Pajkossy 2017; Vincent 2019 (LC-NE)
  cortical state      eeg (covariance)    repo core; Riemannian tangent backend
  slow endocrine tone endocrine           Coates 2008; Cueva 2015 (cortisol/testosterone)
  (none)              null_control        negative control: must be dropped

Because each family observes its driver(s) under **independent** noise, no single
family sees the whole choice logit, so reconciling them recovers more of it than the
best one alone. The endocrine family is deliberately weak and subject-level (slow
hormones barely move trial to trial), and the null family carries pure noise: a
correct ensemble must down-weight the first and drop the second.

What is reported
----------------
1. Single-family baselines: each family alone through ``run_experiment``.
2. The full six-family ensemble under all three reconciliation rules.
3. Fusion lift = full-ensemble pooled balanced accuracy minus the best single family,
   with subject-bootstrap CIs on both.
4. An ablation ladder (neuro -> +cardiac -> +eda -> +pupil -> +endocrine -> +null)
   showing whether accuracy climbs as complementary families are added and flattens
   when the null family is appended.
5. A ground-truth check: the learned accuracy_weighted weights should rank the
   families by their planted signal strength, and the null family's weight should be
   ~0.

Run
---
    python scripts/run_biosignal_fusion.py               # full experiment
    python scripts/run_biosignal_fusion.py --quick       # fewer boots/perms, faster
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavioral_decoding.config import (  # noqa: E402
    BalanceConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
)
from behavioral_decoding.io.base import ModalityBlock, MultimodalDataset  # noqa: E402
from behavioral_decoding.models.riemann import flatten_spd  # noqa: E402
from behavioral_decoding.pipelines.train import run_experiment  # noqa: E402

# Families, in the order they enter the ablation ladder. "eeg" keeps its canonical
# name so the covariance/Riemannian path in build_modality_model fires on it.
FAMILIES = ["eeg", "cardiac", "eda", "pupil", "endocrine", "null_control"]


# --------------------------------------------------------------------------- data


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(logits: np.ndarray, rate: float) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(np.mean(_sigmoid(logits + mid))) < rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _covariance_block(
    driver: np.ndarray,
    subj_bias: np.ndarray,
    rng: np.random.Generator,
    n_channels: int = 8,
    n_samples: int = 128,
    loading: float = 0.9,
    noise: float = 1.0,
) -> np.ndarray:
    """Per-trial EEG-like covariance upper-triangles that carry ``driver``.

    Two channels are driven by the trial's cortical driver, so the covariance
    *between* channels (not just the variance) encodes it. This is exactly the
    structure the Riemannian tangent-space backend is designed to read.
    """
    n_trials = driver.shape[0]
    tri_len = n_channels * (n_channels + 1) // 2
    out = np.empty((n_trials, tri_len))
    mats = np.empty((n_trials, n_channels, n_channels))
    for t in range(n_trials):
        ts = noise * rng.standard_normal((n_samples, n_channels))
        common = loading * driver[t] + 0.3 * subj_bias[t]
        ts[:, 0] += common * rng.standard_normal(n_samples) * 0.0 + common
        ts[:, 1] += 0.8 * common
        ts[:, 2] += 0.4 * common
        cov = np.cov(ts, rowvar=False)
        mats[t] = cov
    return flatten_spd(mats)


def make_fusion_dataset(
    n_subjects: int = 24,
    n_stimuli: int = 60,
    positive_rate: float = 0.30,
    seed: int = 0,
) -> Tuple[MultimodalDataset, Dict[str, object]]:
    """Multimodal biosignal data where fusion beats any single family by construction."""
    rng = np.random.default_rng(seed)
    subject_ids = np.array([f"sub-{i:02d}" for i in range(n_subjects)])
    stimulus_ids = np.array([f"stim-{i:03d}" for i in range(n_stimuli)])

    n_trials = n_subjects * n_stimuli
    subj_idx = np.repeat(np.arange(n_subjects), n_stimuli)
    stim_idx = np.tile(np.arange(n_stimuli), n_subjects)
    trial_subjects = subject_ids[subj_idx]
    trial_stimuli = stimulus_ids[stim_idx]

    subject_bias = rng.normal(0.0, 1.0, size=n_subjects)[subj_idx]

    # Four independent latent drivers of the choice. Independence is what makes
    # fusion pay: each family sees only its own driver(s) under its own noise.
    d_auto = rng.normal(size=n_trials)   # autonomic arousal
    d_ne = rng.normal(size=n_trials)     # noradrenergic volatility
    d_cort = rng.normal(size=n_trials)   # cortical state
    d_endo = rng.normal(size=n_subjects)[subj_idx] + 0.2 * rng.normal(size=n_trials)  # slow

    # Choice logit: every driver contributes, so no single family is sufficient.
    logits = 0.9 * d_auto + 0.8 * d_ne + 1.0 * d_cort + 0.5 * d_endo + 0.4 * subject_bias
    logits = (logits - logits.mean()) / logits.std()
    intercept = _calibrate_intercept(logits, positive_rate)
    y = (rng.uniform(size=n_trials) < _sigmoid(logits + intercept)).astype(int)

    def tabular(driver, n_feat, sig, noise, extra=None):
        """A small feature block loading on ``driver`` (+ optional second driver)."""
        X = noise * rng.normal(size=(n_trials, n_feat))
        for j in range(min(3, n_feat)):
            X[:, j] += sig * (0.85 ** j) * driver
        if extra is not None:
            X[:, -1] += 0.4 * extra
        return X

    # cardiac: HRV-like, senses autonomic arousal strongly (Forte 2021).
    cardiac_X = tabular(d_auto, 12, sig=1.0, noise=1.2, extra=0.3 * d_ne)
    # eda: anticipatory skin conductance, also autonomic but noisier (Dunn 2005).
    eda_X = tabular(d_auto, 8, sig=0.8, noise=1.5)
    # pupil: LC-NE volatility signal (Pajkossy 2017; Vincent 2019).
    pupil_X = tabular(d_ne, 6, sig=1.0, noise=1.3)
    # endocrine: slow cortisol/testosterone-like tone, weak per trial (Coates 2008).
    endocrine_X = tabular(d_endo, 4, sig=0.6, noise=1.6)
    # null_control: pure noise, no driver. Must be dropped by accuracy_weighted.
    null_X = rng.normal(size=(n_trials, 6))
    # eeg: covariance triangles carrying the cortical driver -> Riemannian backend.
    eeg_X = _covariance_block(d_cort, subject_bias, rng, n_channels=8, loading=0.9, noise=1.0)

    prov = lambda name, note: {  # noqa: E731
        "source": "synthetic",
        "generator": "run_biosignal_fusion.make_fusion_dataset",
        "seed": seed,
        "reportable": False,
        "note": note,
    }

    def block(name, X, note, covariance=False):
        b = ModalityBlock(
            name=name,
            X=X,
            subject_ids=trial_subjects,
            stimulus_ids=trial_stimuli,
            feature_names=[f"{name}_{i:02d}" for i in range(X.shape[1])],
            provenance=prov(name, note),
        )
        return b

    blocks = {
        "eeg": block("eeg", eeg_X, "simulated cortical covariance; not real EEG"),
        "cardiac": block("cardiac", cardiac_X, "simulated HRV-like autonomic features"),
        "eda": block("eda", eda_X, "simulated electrodermal features"),
        "pupil": block("pupil", pupil_X, "simulated pupillometry (LC-NE) features"),
        "endocrine": block("endocrine", endocrine_X, "simulated slow endocrine tone"),
        "null_control": block("null_control", null_X, "pure noise negative control"),
    }

    dataset = MultimodalDataset(
        blocks=blocks,
        y_individual=y,
        subject_ids=trial_subjects,
        stimulus_ids=trial_stimuli,
        y_aggregate=None,
        metadata={"synthetic": True, "experiment": "biosignal_fusion"},
    )
    truth = {
        "planted_betas": {"cardiac/eda(auto)": 0.9, "pupil(ne)": 0.8,
                          "eeg(cort)": 1.0, "endocrine": 0.5, "null_control": 0.0},
        "expected": ("full fusion > best single family; null_control weight ~0; "
                     "accuracy_weighted weights rank families by planted strength"),
    }
    return dataset, truth


# --------------------------------------------------------------------- experiment


def _subset(dataset: MultimodalDataset, families: List[str]) -> MultimodalDataset:
    return MultimodalDataset(
        blocks={f: dataset.blocks[f] for f in families},
        y_individual=dataset.y_individual,
        subject_ids=dataset.subject_ids,
        stimulus_ids=dataset.stimulus_ids,
        y_aggregate=None,
        metadata=dataset.metadata,
    )


def _config(name: str, families: List[str], reconciliation: str, quick: bool) -> ExperimentConfig:
    # Riemannian tangent-space backend for the eeg covariance block; logistic for
    # the tabular biosignal families. SMOTE + bagging are on for every family via
    # the repo defaults.
    base_learner = {"eeg": "riemann"} if "eeg" in families else {}
    return ExperimentConfig(
        name=name,
        seed=0,
        output_dir=str(REPO_ROOT / "results"),
        data=DataConfig(modalities=list(families), outcome_column="choice"),
        balance=BalanceConfig(sampler="smote", auto_recommend=True),
        model=ModelConfig(
            reconciliation=reconciliation,
            weight_metric="balanced_accuracy",
            drop_below_chance=True,
            base_learner=base_learner,
            n_bags={f: (10 if quick else 20) for f in families},
        ),
        evaluation=EvalConfig(
            n_splits_outer=5,
            n_splits_inner=4,
            n_bootstrap=800 if quick else 2000,
            n_permutations=200 if quick else 800,
        ),
    )


def _metrics(rec: Dict) -> Dict[str, float]:
    m = rec["individual"]["pooled_metrics"]
    ci = rec["individual"]["bootstrap_ci"].get("balanced_accuracy", {})
    perm = rec["individual"]["permutation_test"]
    return {
        "balanced_accuracy": float(m.get("balanced_accuracy", float("nan"))),
        "roc_auc": float(m.get("roc_auc", float("nan"))),
        "ci_low": float(ci.get("low", float("nan"))),
        "ci_high": float(ci.get("high", float("nan"))),
        "perm_p": float(perm.get("p_value", float("nan"))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer bootstraps/permutations")
    args = ap.parse_args()

    dataset, truth = make_fusion_dataset()
    print(f"fusion dataset: {dataset.n_trials} trials, {dataset.n_subjects} subjects, "
          f"{len(FAMILIES)} families, positive rate = {float(np.mean(dataset.y_individual)):.3f}\n")

    results: Dict[str, object] = {"ground_truth": truth, "single_family": {}, "fusion": {},
                                  "ablation": {}}

    # 1. Single-family baselines.
    print("=== single-family baselines (each family alone) ===")
    single: Dict[str, Dict[str, float]] = {}
    for fam in FAMILIES:
        rec = run_experiment(_subset(dataset, [fam]),
                             _config(f"fusion_single_{fam}", [fam], "accuracy_weighted", args.quick),
                             write=False)
        single[fam] = _metrics(rec)
        m = single[fam]
        print(f"  {fam:<13} balacc={m['balanced_accuracy']:.3f}  "
              f"CI[{m['ci_low']:.3f},{m['ci_high']:.3f}]  auc={m['roc_auc']:.3f}  p={m['perm_p']:.3f}")
    results["single_family"] = single
    best_fam = max(single, key=lambda f: single[f]["balanced_accuracy"])
    best = single[best_fam]["balanced_accuracy"]
    print(f"  best single family: {best_fam} = {best:.3f}\n")

    # 2. Full six-family ensemble under all three reconciliation rules.
    print("=== full fusion (all six families) x reconciliation rule ===")
    for rule in ("majority", "soft", "accuracy_weighted"):
        rec = run_experiment(dataset, _config(f"fusion_all_{rule}", FAMILIES, rule, args.quick),
                             write=False)
        m = _metrics(rec)
        weights = rec["final_model"]["weights"]
        results["fusion"][rule] = {"metrics": m, "weights": weights}
        lift = m["balanced_accuracy"] - best
        print(f"  {rule:<18} balacc={m['balanced_accuracy']:.3f}  "
              f"CI[{m['ci_low']:.3f},{m['ci_high']:.3f}]  auc={m['roc_auc']:.3f}  "
              f"lift_vs_best_single={lift:+.3f}")
        if rule == "accuracy_weighted":
            print("    weights:", {k: round(v, 3) for k, v in
                                    sorted(weights.items(), key=lambda kv: -kv[1])})

    # 3. Ablation ladder: add families one at a time, accuracy_weighted.
    print("\n=== ablation ladder (accuracy_weighted, add one family at a time) ===")
    ladder = []
    for i in range(1, len(FAMILIES) + 1):
        fams = FAMILIES[:i]
        rec = run_experiment(_subset(dataset, fams),
                             _config(f"fusion_ladder_{i}", fams, "accuracy_weighted", args.quick),
                             write=False)
        m = _metrics(rec)
        ladder.append({"added": FAMILIES[i - 1], "families": fams,
                       "balanced_accuracy": m["balanced_accuracy"],
                       "ci_low": m["ci_low"], "ci_high": m["ci_high"]})
        print(f"  +{FAMILIES[i-1]:<13} ({i} families) balacc={m['balanced_accuracy']:.3f}  "
              f"CI[{m['ci_low']:.3f},{m['ci_high']:.3f}]")
    results["ablation"] = ladder

    out = REPO_ROOT / "results" / "biosignal_fusion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}")

    # Honest one-line verdict.
    aw = results["fusion"]["accuracy_weighted"]["metrics"]["balanced_accuracy"]
    print(f"\nverdict: full accuracy-weighted fusion = {aw:.3f} vs best single "
          f"({best_fam}) = {best:.3f}; lift = {aw - best:+.3f}. "
          "This is a positive control on synthetic data, not a claim about real people.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
