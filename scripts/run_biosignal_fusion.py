"""Multi-biosignal fusion: does combining biosignal families beat the best single one?

Runs the repo's own MultimodalEnsemble (bagged, in-fold SMOTE, out-of-fold-weighted
reconciliation, Riemannian tangent-space backend for EEG covariance) over six
biosignal families on a generator with planted structure, then verifies the
aggregation gain with runtime gates. Positive control on synthetic data, not a claim
about real people. See docs/biosignal_fusion.md.

A binary choice is driven by four partly-independent latent drivers, each sensed best
by a different family, plus one family that senses nothing:

  driver             family              literature anchor
  autonomic arousal  cardiac (HRV), eda  Forte 2021; Dunn 2006 (somatic marker)
  noradrenergic vol. pupil               Pajkossy 2017; Vincent 2019 (LC-NE)
  cortical state     eeg (covariance)    Riemannian tangent backend
  slow endocrine     endocrine           Coates 2008; Cueva 2015
  none               null_control        negative control, must be dropped

Per-family noise is independent, so no family sees the whole logit and fusion recovers
more of it than the best single family (inverse-variance argument in the doc).

Run:
    python scripts/run_biosignal_fusion.py           # full
    python scripts/run_biosignal_fusion.py --quick   # faster
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
    mats = np.empty((n_trials, n_channels, n_channels))
    for t in range(n_trials):
        ts = noise * rng.standard_normal((n_samples, n_channels))
        # The driver modulates inter-channel COUPLING, not the channel mean.
        # A shared latent source is injected into a few channels with a gain that
        # is linear in the (signed) driver, so cov(ch0, ch_k) scales with the
        # driver. This is the connectivity-strength structure the tangent-space
        # map reads; a mean shift would be invisible because covariance is
        # mean-centred.
        latent = rng.standard_normal(n_samples)
        coupling = loading * driver[t] + 0.3 * subj_bias[t]
        ts[:, 0] += latent
        ts[:, 1] += coupling * latent
        ts[:, 2] += 0.6 * coupling * latent
        mats[t] = np.cov(ts, rowvar=False)
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
    ci = rec["individual"]["bootstrap_ci"]  # flat dict: metric/point/lo/hi/n_valid_boot
    perm = rec["individual"]["permutation_test"]
    return {
        "balanced_accuracy": float(m.get("balanced_accuracy", float("nan"))),
        "roc_auc": float(m.get("roc_auc", float("nan"))),
        "ci_low": float(ci.get("lo", float("nan"))),
        "ci_high": float(ci.get("hi", float("nan"))),
        "perm_p": float(perm.get("p_value", perm.get("p", float("nan")))),
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
                             _config(f"fusion_single_{fam}", [fam],
                                     "accuracy_weighted", args.quick),
                             write=False)
        single[fam] = _metrics(rec)
        m = single[fam]
        print(f"  {fam:<13} balacc={m['balanced_accuracy']:.3f}  "
              f"CI[{m['ci_low']:.3f},{m['ci_high']:.3f}]  auc={m['roc_auc']:.3f}  "
              f"p={m['perm_p']:.3f}")
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

    make_figure(results, REPO_ROOT / "docs" / "figures" / "biosignal_fusion.png")
    n_pass = check_gates(results)

    aw = results["fusion"]["accuracy_weighted"]["metrics"]["balanced_accuracy"]
    print(f"\nfull accuracy-weighted fusion = {aw:.3f} vs best single "
          f"({best_fam}) = {best:.3f}; lift = {aw - best:+.3f}. "
          "Positive control on synthetic data, not a claim about real people.")
    return 0 if n_pass else 1


def check_gates(results: Dict) -> bool:
    """Verify each planted fact. Prints PASS/FAIL; returns True iff all pass."""
    single = results["single_family"]
    aw = results["fusion"]["accuracy_weighted"]
    best = max(v["balanced_accuracy"] for v in single.values())
    ladder = results["ablation"]
    checks = []

    # 1. Fusion beats the best single family.
    fused = aw["metrics"]["balanced_accuracy"]
    checks.append(("fusion > best single family", fused > best,
                   f"{fused:.3f} vs {best:.3f}"))
    # 2. Null modality is dropped (weight ~0) and does not beat chance.
    wnull = aw["weights"].get("null_control", 1.0)
    checks.append(("null modality weight = 0", wnull == 0.0, f"weight={wnull:.3f}"))
    checks.append(("null modality not above chance",
                   single["null_control"]["balanced_accuracy"] <= 0.5,
                   f"balacc={single['null_control']['balanced_accuracy']:.3f}"))
    # 3. Adding the null family does not raise ladder accuracy.
    checks.append(("null adds no accuracy in ladder",
                   ladder[-1]["balanced_accuracy"] <= ladder[-2]["balanced_accuracy"] + 1e-9,
                   f"{ladder[-2]['balanced_accuracy']:.3f} -> "
                   f"{ladder[-1]['balanced_accuracy']:.3f}"))
    # 4. Endocrine (planted weak) ranks below the strong families' mean weight.
    w = aw["weights"]
    strong = np.mean([w["cardiac"], w["eeg"], w["pupil"]])
    checks.append(("endocrine weaker than strong families",
                   w["endocrine"] < strong, f"{w['endocrine']:.3f} < {strong:.3f}"))

    print("\n=== gates (ground-truth checks) ===")
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<38} {detail}")
        ok = ok and passed
    return ok


def make_figure(results: Dict, path: Path) -> None:
    """Two-panel figure: per-family balanced accuracy, and the ablation ladder."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    single = results["single_family"]
    ladder = results["ablation"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    fams = list(single.keys())
    vals = [single[f]["balanced_accuracy"] for f in fams]
    err = [[single[f]["balanced_accuracy"] - single[f]["ci_low"] for f in fams],
           [single[f]["ci_high"] - single[f]["balanced_accuracy"] for f in fams]]
    colors = ["#4c72b0" if f != "null_control" else "#bbbbbb" for f in fams]
    ax1.bar(fams, vals, yerr=err, capsize=3, color=colors)
    ax1.axhline(0.5, color="k", ls="--", lw=1)
    ax1.set_ylabel("balanced accuracy (pooled OOF)")
    ax1.set_title("Each biosignal family alone")
    ax1.set_ylim(0.4, 0.72)
    ax1.tick_params(axis="x", rotation=45)

    steps = [f"+{s['added']}" for s in ladder]
    lv = [s["balanced_accuracy"] for s in ladder]
    le = [[s["balanced_accuracy"] - s["ci_low"] for s in ladder],
          [s["ci_high"] - s["balanced_accuracy"] for s in ladder]]
    ax2.errorbar(range(len(steps)), lv, yerr=le, marker="o", capsize=3, color="#c44e52")
    ax2.axhline(0.5, color="k", ls="--", lw=1)
    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels(steps, rotation=45, ha="right")
    ax2.set_ylabel("balanced accuracy (pooled OOF)")
    ax2.set_title("Fusion ladder: signal adds, noise does not")
    ax2.set_ylim(0.5, 0.68)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
