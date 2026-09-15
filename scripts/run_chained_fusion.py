#!/usr/bin/env python3
"""Does chaining related biosignals beat treating them as separate votes?

The question
------------
Heart rate and skin conductance are not independent: both are driven by one
autonomic arousal system. The flat fusion argument treats every family as a
separate voter and gains from the *independence* of their errors, so shared
drive looks like a problem, a ceiling on what extra sensors buy.

That framing is incomplete. Two sensors driven by one latent state are two
noisy readings of the same thing, which is the classic measurement model:

    arousal (latent)  ->  cardiac indicators
                      ->  electrodermal indicators
    arousal           ->  choice

Under that model the right move is not to let cardiac and EDA vote separately.
It is to *chain*: use both indicator sets to estimate arousal, then let the
better-estimated arousal predict the choice. Reliability of a composite of
correlated indicators rises with the number of indicators (Spearman-Brown), so
several cheap indirect sensors can stand in for one good direct measure. If
that holds, an attacker needs far less hardware than the flat argument implies,
which makes the privacy concern worse, not better.

The test
--------
The generator in ``run_biosignal_fusion.py`` plants exactly this structure:
``cardiac`` and ``eda`` both load on one ``d_auto`` driver, and only the first
three columns of each block carry it, so the useful direction is not known in
advance and has to be learned.

  late  : cardiac and eda modelled separately, reconciled by weighted vote
          (one model per family, combined at the level of their outputs)
  early : cardiac and eda concatenated into ONE "autonomic" block, so a single
          model sees both indicator sets and learns the shared factor itself

Everything else is held fixed: same trials, same splits, same bagging, same
in-fold resampling, same seed. The only difference is where the two indicator
sets meet.

Both arms learn entirely inside the training folds, so neither enjoys an
information advantage: the chained arm is not handed the latent, it has to
recover it from the same columns the flat arm sees.

It also asks the sharper question for the threat model: how much survives if
the *direct* measure is removed? ``indirect_only`` drops the EEG block, leaving
only autonomic, pupil and endocrine proxies.

Usage
-----
    python scripts/run_chained_fusion.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from behavioral_decoding.io.base import ModalityBlock, MultimodalDataset  # noqa: E402
from behavioral_decoding.pipelines.train import run_experiment  # noqa: E402
from run_biosignal_fusion import _config, _metrics, make_fusion_dataset  # noqa: E402

AROUSAL_INDICATORS = ("cardiac", "eda")


def merge_blocks(dataset: MultimodalDataset, names, merged_name: str) -> ModalityBlock:
    """Concatenate several indicator blocks into one construct-level block."""
    blocks = [dataset.blocks[n] for n in names]
    ref = blocks[0]
    feature_names: List[str] = []
    for n, b in zip(names, blocks):
        existing = list(b.feature_names) if b.feature_names is not None else [
            f"f{i}" for i in range(b.X.shape[1])
        ]
        feature_names.extend(f"{n}:{f}" for f in existing)
    return ModalityBlock(
        name=merged_name,
        X=np.hstack([b.X for b in blocks]),
        subject_ids=ref.subject_ids,
        stimulus_ids=ref.stimulus_ids,
        feature_names=feature_names,
        provenance={
            "construct": merged_name,
            "indicators": list(names),
            "note": (
                "Indicator sets concatenated so one model can estimate the shared "
                "latent state, instead of each indicator voting separately."
            ),
        },
    )


def build(dataset: MultimodalDataset, blocks: Dict[str, ModalityBlock]) -> MultimodalDataset:
    return MultimodalDataset(
        blocks=blocks,
        y_individual=dataset.y_individual,
        subject_ids=dataset.subject_ids,
        stimulus_ids=dataset.stimulus_ids,
        y_aggregate=None,
        metadata=dataset.metadata,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer bags/bootstraps")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="generator seeds; >1 checks whether a difference survives "
                         "regeneration of the data")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="run only these arms (default: all)")
    ap.add_argument("--output", default=str(REPO_ROOT / "results" / "chained_fusion.json"))
    args = ap.parse_args()

    all_seed_results: Dict[int, Dict[str, Dict[str, float]]] = {}
    for seed in args.seeds:
        all_seed_results[seed] = run_one_seed(seed, args)

    if len(args.seeds) > 1:
        print("=" * 74)
        print("ACROSS SEEDS (balanced accuracy)")
        print("=" * 74)
        names = list(next(iter(all_seed_results.values())))
        print(f"  {'arm':<22}" + "".join(f"{('seed ' + str(s)):>10}" for s in args.seeds)
              + f"{'mean':>10}")
        for nm in names:
            vals = [all_seed_results[s][nm]["balanced_accuracy"] for s in args.seeds]
            print(f"  {nm:<22}" + "".join(f"{v:>10.3f}" for v in vals)
                  + f"{sum(vals)/len(vals):>10.3f}")
        print()
        for a, b, label in (("late_two", "early_chained_two", "chaining, two indicators"),
                            ("full_late", "full_chained", "chaining, whole system"),
                            ("eeg_only_direct", "indirect_only", "indirect minus direct")):
            if a in names and b in names:
                diffs = [all_seed_results[s][b]["balanced_accuracy"]
                         - all_seed_results[s][a]["balanced_accuracy"] for s in args.seeds]
                mean = sum(diffs) / len(diffs)
                sign = "consistent" if all(d > 0 for d in diffs) or all(d < 0 for d in diffs) \
                    else "SIGN FLIPS ACROSS SEEDS"
                print(f"  {label:<26} mean {mean:+.3f}   per seed "
                      + ", ".join(f"{d:+.3f}" for d in diffs) + f"   [{sign}]")

    out = {"seeds": {str(k): v for k, v in all_seed_results.items()},
           "note": "Synthetic positive control. Same trials, splits and bagging "
                   "within a seed; only the grouping of indicator blocks differs."}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}")
    return 0


def run_one_seed(seed: int, args) -> Dict[str, Dict[str, float]]:
    dataset, _truth = make_fusion_dataset(seed=seed)
    autonomic = merge_blocks(dataset, AROUSAL_INDICATORS, "autonomic")

    arms: Dict[str, Dict[str, ModalityBlock]] = {
        # --- does chaining two indicators of ONE latent beat voting them separately?
        "cardiac_only": {"cardiac": dataset.blocks["cardiac"]},
        "eda_only": {"eda": dataset.blocks["eda"]},
        "late_two": {k: dataset.blocks[k] for k in AROUSAL_INDICATORS},
        "early_chained_two": {"autonomic": autonomic},
        # --- and does it help the whole system?
        "full_late": {k: dataset.blocks[k] for k in
                      ("eeg", "cardiac", "eda", "pupil", "endocrine")},
        "full_chained": {"eeg": dataset.blocks["eeg"], "autonomic": autonomic,
                         "pupil": dataset.blocks["pupil"],
                         "endocrine": dataset.blocks["endocrine"]},
        # --- the threat-model question: how much survives without the direct measure?
        "eeg_only_direct": {"eeg": dataset.blocks["eeg"]},
        "indirect_only": {"autonomic": autonomic, "pupil": dataset.blocks["pupil"],
                          "endocrine": dataset.blocks["endocrine"]},
    }

    if args.arms:
        arms = {k: v for k, v in arms.items() if k in args.arms}

    results: Dict[str, Dict[str, float]] = {}
    for name, blocks in arms.items():
        print("=" * 74)
        print(f"[seed {seed}] ARM: {name}   blocks: {', '.join(blocks)}")
        print("=" * 74)
        sub = build(dataset, blocks)
        cfg = _config(f"chain_{name}_s{seed}", list(blocks), "accuracy_weighted", args.quick)
        cfg.output_dir = str(REPO_ROOT / "results" / "chained")
        rec = run_experiment(sub, cfg)
        results[name] = _metrics(rec)
        print(f"  balanced accuracy {results[name]['balanced_accuracy']:.3f}  "
              f"({results[name]['ci_low']:.3f}-{results[name]['ci_high']:.3f})   "
              f"AUC {results[name]['roc_auc']:.3f}")
        print()

    def ba(k: str) -> float:
        return results[k]["balanced_accuracy"]

    if {"cardiac_only", "eda_only"} <= set(results):
        print(f"[seed {seed}] cardiac {ba('cardiac_only'):.3f} | eda {ba('eda_only'):.3f}")
    if {"late_two", "early_chained_two"} <= set(results):
        print(f"[seed {seed}] voting separately {ba('late_two'):.3f} | "
              f"chained {ba('early_chained_two'):.3f}")
    if {"full_late", "full_chained"} <= set(results):
        print(f"[seed {seed}] whole system flat {ba('full_late'):.3f} | "
              f"chained {ba('full_chained'):.3f}")
    if {"eeg_only_direct", "indirect_only"} <= set(results):
        print(f"[seed {seed}] direct only {ba('eeg_only_direct'):.3f} | "
              f"indirect proxies only {ba('indirect_only'):.3f}")
    print()
    return results


if __name__ == "__main__":
    raise SystemExit(main())
