#!/usr/bin/env python3
"""Run the pipeline's behaviour arm on the REAL NARPS cohort (ds001734).

Why this script exists
----------------------
``scripts/run_narps.py`` needs the BOLD volumes to build an fMRI block. The
OpenNeuro release distributes those volumes through git-annex, so a plain
archive download gives you the metadata and the ``events.tsv`` files but only
pointer stubs where the NIfTIs should be. Running that script against such a
tree yields nothing, and its ``--demo`` mode writes a *synthetic* fixture that
must never be reported as a finding about brains.

The event files, however, are real: 108 participants, four runs each, every
gamble they were offered and the choice they actually made. That is a real
human decision dataset, and the behaviour arm of this repo can run on it
end-to-end with no neuroimaging dependency at all.

What this measures
------------------
Given the economic terms of a gamble (gain, loss, expected value, |expected
value|), can the pipeline predict whether *this* participant accepted it,
generalising to participants it has never seen? Choice is largely a function of
gain and loss, so a strong result here is expected and is not a neuroscience
claim: it is evidence that the framework recovers a real, individually-varying
behavioural signal from a real cohort, under the same subject-grouped splits,
leakage guards, calibration reporting and run record used everywhere else.

``RT`` is excluded from the behaviour block: it is measured after the decision
and would leak the outcome into a predictor of that outcome
(``include_rt=False`` is the loader default).

``--with-timing`` scores RT as a separate, clearly-labelled channel. This is a
channel-quality check, not a fusion result: the question it answers is whether
the reconciler correctly drops a channel that carries nothing, on real data
rather than on the planted control. It does. The fusion claim in this repo
remains confined to the synthetic positive control in
``scripts/run_biosignal_fusion.py``, because no real cohort here has several
biosignal families recorded from the same people.

Data
----
NARPS, Botvinik-Nezer et al. (2019), *Scientific Data*,
https://doi.org/10.1038/s41597-019-0113-7; dataset ds001734 on OpenNeuro,
https://doi.org/10.18112/openneuro.ds001734.v1.0.5. CC0.

Usage
-----
    python scripts/run_narps_behavior.py --root data/raw/narps_events
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavioral_decoding.config import ExperimentConfig  # noqa: E402
from behavioral_decoding.features.align import build_dataset  # noqa: E402
from behavioral_decoding.io.narps import (  # noqa: E402
    NARPSLoader,
    parse_events,
    subject_id_from_path,
)
from behavioral_decoding.pipelines.train import run_experiment  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "data" / "raw" / "narps_events"


def find_event_files(root: Path) -> list[Path]:
    """Every ``sub-*/func/*_events.tsv`` under ``root``, at any nesting depth."""
    files = sorted(root.rglob("sub-*/func/*_task-MGT_*_events.tsv"))
    if not files:
        files = sorted(root.rglob("*_task-MGT_*_events.tsv"))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help="directory holding the NARPS events tree")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--name", default="narps_behavior_real")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="use only the first N subjects (smoke test)")
    parser.add_argument("--with-timing", action="store_true",
                        help="also score a response-timing channel, as a check "
                             "that a useless channel is correctly given zero "
                             "weight on real data (not a fusion claim)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        parser.error(f"{root} not found. Point --root at an unpacked ds001734 tree.")

    files = find_event_files(root)
    if not files:
        parser.error(f"no NARPS events.tsv files found under {root}")

    loader = NARPSLoader(include_rt=False)

    by_subject: dict[str, list] = {}
    for path in files:
        by_subject.setdefault(subject_id_from_path(path), []).append(path)

    subjects = sorted(by_subject)
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]

    blocks = []
    timing_rows: list = []
    timing_subjects: list = []
    timing_stimuli: list = []
    y_individual: dict[tuple, float] = {}
    n_trials = 0
    n_runs = 0

    for subject in subjects:
        for path in sorted(by_subject[subject]):
            events = parse_events(path)
            if len(events) == 0:
                continue
            blocks.append(loader.behavior_block(events, subject))
            if args.with_timing:
                rt = events["RT"].to_numpy(dtype=float)
                import numpy as _np
                timing_rows.append(_np.column_stack([rt, _np.log1p(_np.clip(rt, 0, None))]))
                timing_subjects.append(_np.array([subject] * len(events)))
                timing_stimuli.append(events["stimulus_id"].to_numpy())
            n_runs += 1
            n_trials += len(events)
            for stim, accept in zip(events["stimulus_id"], events["accept"]):
                y_individual[(subject, stim)] = float(accept)

    if not blocks:
        parser.error("parsed zero usable trials; check the events files")

    # One block per run; concatenate into a single behaviour block.
    import numpy as np

    from behavioral_decoding.io.base import ModalityBlock

    behavior = ModalityBlock(
        name="behavior",
        X=np.vstack([b.X for b in blocks]),
        subject_ids=np.concatenate([b.subject_ids for b in blocks]),
        stimulus_ids=np.concatenate([b.stimulus_ids for b in blocks]),
        feature_names=blocks[0].feature_names,
        provenance={
            "loader": "NARPSLoader.behavior_block",
            "modality": "behavior",
            "source": "narps_events_real",
            "dataset": "ds001734 (NARPS), Botvinik-Nezer et al. 2019, CC0",
            "features": list(blocks[0].feature_names),
            "n_subjects": len(subjects),
            "n_runs": n_runs,
            "n_trials_parsed": n_trials,
            "rt_excluded": True,
            "note": (
                "REAL NARPS behavioural events (no fMRI: the OpenNeuro BOLD "
                "volumes are git-annex pointers in an archive download). "
                "Economic comparator only; not a neuroimaging result."
            ),
        },
    )

    block_map = {"behavior": behavior}

    if args.with_timing:
        block_map["response"] = ModalityBlock(
            name="response",
            X=np.vstack(timing_rows),
            subject_ids=np.concatenate(timing_subjects),
            stimulus_ids=np.concatenate(timing_stimuli),
            feature_names=["RT", "log1p_RT"],
            provenance={
                "loader": "run_narps_behavior (timing block)",
                "modality": "response",
                "source": "narps_events_real",
                "features": ["RT", "log1p_RT"],
                "note": (
                    "Response dynamics, kept in a SEPARATE block on purpose. RT "
                    "is produced during the same decision it helps predict, so "
                    "it is a companion of the outcome, not a clean cause. It is "
                    "isolated here so its contribution is visible in the learned "
                    "weights and can be discounted. Do not read the combined "
                    "number as a clean forecast of choice."
                ),
            },
        )

    dataset = build_dataset(
        block_map,
        y_individual=y_individual,
        metadata={
            "source": "NARPS ds001734 events.tsv (real)",
            "citation": "Botvinik-Nezer et al. 2019, Sci Data, 10.1038/s41597-019-0113-7",
            "n_subjects": len(subjects),
            "n_runs": n_runs,
            "synthetic": False,
        },
    )

    print(f"parsed {n_trials} real trials from {n_runs} runs across "
          f"{len(subjects)} participants")
    print(dataset.describe())
    print()

    cfg = ExperimentConfig(name=args.name, output_dir=args.output_dir)
    cfg.data.modalities = list(block_map)
    cfg.data.outcome_column = "accept"

    record = run_experiment(dataset, cfg)

    ind = record["individual"]
    ci = ind["bootstrap_ci"]
    perm = ind["permutation_test"]
    print()
    print("=" * 78)
    print("NARPS BEHAVIOUR ARM (REAL DATA, ds001734)")
    print("=" * 78)
    print(f"  balanced accuracy : {ind['pooled_metrics']['balanced_accuracy']:.3f}")
    print(f"  95% CI (subjects) : {ci['lo']:.3f}-{ci['hi']:.3f}")
    print(f"  ROC AUC           : {ind['pooled_metrics']['roc_auc']:.3f}")
    print(f"  permutation p     : {perm['p_value']:.4f} "
          f"(null mean {perm['null_mean']:.3f}, sd {perm['null_sd']:.3f})")
    per_mod = ind.get("per_modality_pooled") or {}
    if len(per_mod) > 1:
        print()
        print("  per block (pooled out-of-fold balanced accuracy):")
        for name, m in sorted(per_mod.items()):
            print(f"    {name:<10} {m['balanced_accuracy']:.3f}")
        weights = (record.get("final_model") or {}).get("weights") or {}
        if weights:
            print("  reconciliation weights: "
                  + ", ".join(f"{k}={v:.2f}" for k, v in sorted(weights.items())))
    print()
    print("  Labels vary within participant here, so the within-subject label")
    print("  shuffle is a valid null (unlike the ASZED/Kiva arms).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
