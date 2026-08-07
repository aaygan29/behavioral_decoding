#!/usr/bin/env python3
"""Run the full pipeline on synthetic data and check it recovers the ground truth.

This is the smoke test with teeth. The synthetic generator builds in the
Genevsky/Knutson dissociation by construction: stimulus-general neural signal
forecasts the market, subject-dominated self-report does not. If the pipeline
cannot recover that here, it will not find it in real data either, and any
result it does produce should be treated as suspect.

Usage::

    python scripts/run_demo.py                 # default settings
    python scripts/run_demo.py --seed 7        # a different seed
    python scripts/run_demo.py --quick         # smaller and faster
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavioral_decoding.config import ExperimentConfig  # noqa: E402
from behavioral_decoding.evaluation.cv import describe_splits  # noqa: E402
from behavioral_decoding.evaluation.neuroforecast import format_forecast_comparison  # noqa: E402
from behavioral_decoding.features.align import summarise_alignment  # noqa: E402
from behavioral_decoding.pipelines.train import run_experiment  # noqa: E402
from behavioral_decoding.synthetic import SyntheticConfig, make_synthetic_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="smaller sample, fewer folds")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--reconciliation",
        default="accuracy_weighted",
        choices=["accuracy_weighted", "majority", "soft"],
    )
    args = parser.parse_args()

    syn_cfg = SyntheticConfig(
        n_subjects=12 if args.quick else 24,
        n_stimuli=40 if args.quick else 60,
        seed=args.seed,
    )
    dataset, ground_truth = make_synthetic_dataset(syn_cfg)

    print("=" * 78)
    print("SYNTHETIC DATA (simulated; no number here is a finding about brains)")
    print("=" * 78)
    print(dataset.describe())
    print()
    print(summarise_alignment(dataset.blocks))
    print()
    print(
        describe_splits(
            dataset.y_individual, dataset.subject_ids, n_splits=4 if args.quick else 5
        )
    )
    print()

    cfg = ExperimentConfig(
        name=f"demo_seed{args.seed}",
        seed=args.seed,
        output_dir=args.output_dir,
    )
    cfg.model.reconciliation = args.reconciliation
    cfg.evaluation.n_splits_outer = 3 if args.quick else 5
    cfg.evaluation.n_splits_inner = 3 if args.quick else 4
    cfg.evaluation.n_bootstrap = 300 if args.quick else 2000
    cfg.evaluation.n_permutations = 200 if args.quick else 1000
    cfg.evaluation.forecast_splits = 4 if args.quick else 5

    record = run_experiment(dataset, cfg)

    print()
    print("=" * 78)
    print("GROUND-TRUTH RECOVERY CHECK")
    print("=" * 78)
    print(ground_truth["expected_ordering"])
    print()

    arms = record.get("aggregate") or {}
    print(format_forecast_comparison(arms, cfg.evaluation.forecast_task))
    print()

    checks = []

    individual = record["individual"]["pooled_metrics"]
    checks.append(
        (
            "individual choice beats chance out of fold",
            individual["balanced_accuracy"] > 0.55,
            "balanced accuracy = {:.3f}".format(individual["balanced_accuracy"]),
        )
    )

    perm = record["individual"]["permutation_test"]
    checks.append(
        (
            "individual result survives label permutation",
            perm["p_value"] < 0.05,
            "permutation p = {:.4f}".format(perm["p_value"]),
        )
    )

    if "brain_only" in arms and "behavior_only" in arms:
        brain_r2 = arms["brain_only"]["r2_out_of_sample"]
        behav_r2 = arms["behavior_only"]["r2_out_of_sample"]
        checks.append(
            (
                "brain forecasts the market better than self-report",
                brain_r2 > behav_r2,
                f"brain oos R2 = {brain_r2:.3f} vs behaviour {behav_r2:.3f}",
            )
        )
        checks.append(
            (
                "brain market forecast is above the mean baseline",
                brain_r2 > 0.0,
                f"brain oos R2 = {brain_r2:.3f}",
            )
        )

    print("checks")
    all_pass = True
    for label, passed, detail in checks:
        all_pass = all_pass and passed
        print("  [{}] {:<48} {}".format("PASS" if passed else "FAIL", label, detail))

    print()
    if all_pass:
        print("All ground-truth checks passed. The pipeline recovers the dissociation")
        print("it was built to detect, on data where that dissociation is true by")
        print("construction. This validates the plumbing. It says nothing about brains.")
    else:
        print("At least one ground-truth check failed. Before pointing this pipeline at")
        print("real data, find out why: the framework cannot be trusted to detect an")
        print("effect it misses when the effect is known to be present.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
