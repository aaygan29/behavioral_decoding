#!/usr/bin/env python3
"""Run the pipeline's behaviour arm on a REAL delay-discounting cohort (ds002989).

Data
----
"Delay Discounting Bidding", Lee, Cooper & Kable, OpenNeuro ds002989,
https://doi.org/10.18112/openneuro.ds002989.v1.0.0, CC0. 40 participants, four
runs each. On every trial the participant is shown "$75 after D days" and types
the immediate amount they would accept instead (their indifference point).

As with NARPS, the OpenNeuro archive ships the BOLD volumes through git-annex,
so an archive download contains real ``events.tsv`` files and pointer stubs
where the NIfTIs would be. This script therefore runs the behaviour arm only,
and makes no neuroimaging claim.

What this measures
------------------
Target: did this participant lowball *this* offer, i.e. is their bid below
their own median bid? The median is taken within participant, so the label
describes a trial, not a person. Two consequences, both deliberate:

  1. Between-person differences in overall generosity cannot drive the score.
  2. Labels vary within participant, so the within-subject label shuffle is a
     valid null here (unlike the ASZED and Kiva arms, where the label is
     constant within a person).

Features are the offer and where it sat in the session: the delay in days, its
logarithm (discounting is roughly hyperbolic in delay), the trial's onset time,
and the run index. Response timing (``RT``, ``r_duration``, ``activesubmit``)
is deliberately excluded: it is generated during the same response the bid
comes from, so it is a companion of the outcome rather than a clean predictor
of it. The same reasoning excludes RT on the NARPS arm.

A strong result here is expected: this is the discount curve, which is one of
the most replicated effects in behavioural economics. It belongs in this repo
as evidence that the framework recovers a real, individually-varying
behavioural signal on a real cohort, under the same subject-grouped splits and
leakage guards as everywhere else.

Usage
-----
    python scripts/run_delay_discounting.py --root data/raw/ds002989
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from behavioral_decoding.config import ExperimentConfig  # noqa: E402
from behavioral_decoding.features.align import build_dataset  # noqa: E402
from behavioral_decoding.io.base import ModalityBlock  # noqa: E402
from behavioral_decoding.pipelines.train import run_experiment  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "data" / "raw" / "ds002989"
REQUIRED = ("onset", "RT", "Delay", "Rating")


def subject_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    raise ValueError(f"no sub-* component in {path}")


def run_index_from_path(path: Path) -> int:
    stem = path.stem
    marker = "_run-"
    if marker not in stem:
        return 1
    tail = stem.split(marker, 1)[1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else 1


def read_trials(path: Path) -> list[dict]:
    """Parse one events.tsv into usable trials; skips rows with missing fields."""
    out = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        for row in reader:
            try:
                delay = float(row["Delay"])
                rating = float(row["Rating"])
                onset = float(row["onset"])
            except (TypeError, ValueError):
                continue  # 'n/a' rows: no response recorded
            if delay <= 0:
                continue
            out.append({"delay": delay, "rating": rating, "onset": onset})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help="directory holding the unpacked ds002989 tree")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--name", default="delay_discounting_real")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        parser.error(f"{root} not found. Point --root at an unpacked ds002989 tree.")

    files = sorted(root.rglob("sub-*/func/*task-DDbid*_events.tsv"))
    if not files:
        parser.error(f"no DDbid events.tsv files found under {root}")

    per_subject: dict[str, list[dict]] = {}
    for path in files:
        subject = subject_from_path(path)
        run_idx = run_index_from_path(path)
        for trial in read_trials(path):
            trial["run"] = run_idx
            per_subject.setdefault(subject, []).append(trial)

    rows = []
    dropped_flat = 0
    for subject, trials in sorted(per_subject.items()):
        ratings = np.array([t["rating"] for t in trials], dtype=float)
        if len(trials) < 8 or np.all(ratings == ratings[0]):
            # A participant who typed the same number every trial carries no
            # within-person contrast; a median split would be meaningless.
            dropped_flat += 1
            continue
        median = float(np.median(ratings))
        for t in trials:
            rows.append({
                "subject_id": subject,
                # The offer is the stimulus. The run index keeps repeats of the
                # same delay distinct, which alignment requires.
                "stimulus_id": f"d{int(t['delay'])}_r{int(t['run'])}",
                "delay": t["delay"],
                "log_delay": math.log(t["delay"]),
                "onset": t["onset"],
                "run": float(t["run"]),
                "lowball": float(t["rating"] < median),
            })

    if not rows:
        parser.error("no usable trials after filtering")

    subjects = sorted({r["subject_id"] for r in rows})
    X = np.array([[r["delay"], r["log_delay"], r["onset"], r["run"]] for r in rows])
    subject_ids = np.array([r["subject_id"] for r in rows])
    stimulus_ids = np.array([r["stimulus_id"] for r in rows])
    y = np.array([r["lowball"] for r in rows])

    block = ModalityBlock(
        name="behavior",
        X=X,
        subject_ids=subject_ids,
        stimulus_ids=stimulus_ids,
        feature_names=["delay_days", "log_delay", "onset_s", "run_index"],
        provenance={
            "loader": "run_delay_discounting.read_trials",
            "modality": "behavior",
            "source": "ds002989_events_real",
            "dataset": "ds002989 Delay Discounting Bidding (Lee, Cooper & Kable), CC0",
            "features": ["delay_days", "log_delay", "onset_s", "run_index"],
            "target": "bid below this participant's own median bid",
            "response_timing_excluded": True,
            "note": (
                "REAL behavioural events (no fMRI: OpenNeuro BOLD volumes are "
                "git-annex pointers in an archive download). RT and response "
                "durations excluded as companions of the outcome."
            ),
        },
    )

    y_individual = {
        (s, st): float(val) for s, st, val in zip(subject_ids, stimulus_ids, y)
    }

    dataset = build_dataset(
        {"behavior": block},
        y_individual=y_individual,
        metadata={
            "source": "OpenNeuro ds002989 events.tsv (real)",
            "citation": "Lee, Cooper & Kable, ds002989, 10.18112/openneuro.ds002989.v1.0.0",
            "n_subjects": len(subjects),
            "n_trials": len(rows),
            "synthetic": False,
        },
    )

    print(f"parsed {len(rows)} real trials from {len(subjects)} participants "
          f"({dropped_flat} dropped for a flat bid pattern)")
    print(dataset.describe())
    print()

    cfg = ExperimentConfig(name=args.name, output_dir=args.output_dir)
    cfg.data.modalities = ["behavior"]
    cfg.data.outcome_column = "lowball"

    record = run_experiment(dataset, cfg)

    ind = record["individual"]
    ci = ind["bootstrap_ci"]
    perm = ind["permutation_test"]
    print()
    print("=" * 78)
    print("DELAY DISCOUNTING ARM (REAL DATA, ds002989)")
    print("=" * 78)
    print(f"  balanced accuracy : {ind['pooled_metrics']['balanced_accuracy']:.3f}")
    print(f"  95% CI (subjects) : {ci['lo']:.3f}-{ci['hi']:.3f}")
    print(f"  ROC AUC           : {ind['pooled_metrics']['roc_auc']:.3f}")
    print(f"  permutation p     : {perm['p_value']:.4f} "
          f"(null mean {perm['null_mean']:.3f}, sd {perm['null_sd']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
