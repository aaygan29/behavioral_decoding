#!/usr/bin/env python3
"""Run the pipeline on NARPS (ds001734), or on a synthetic NARPS-format fixture.

With a real NARPS download:

    python scripts/run_narps.py --root /path/to/ds001734 \
        --derivatives /path/to/fmriprep --group equalIndifference

With no download, to see the whole path on the true BIDS format:

    python scripts/run_narps.py --demo

``--demo`` writes a miniature but format-real NARPS BIDS tree (participants.tsv,
per-run events.tsv with the verified columns and response strings, small 4-D
NIfTIs, confounds) with an EV-scaled BOLD bump planted at the NAcc and MPFC
voxels. It then checks that the fMRI ROI features predict accept/reject above
chance out of fold. That is the honest, recoverable claim: consistent with
``docs/narps.md``, no brain-beats-behaviour headline is asserted, because on
gambles the economic baseline dominates the aggregate arm by construction.

Requires nilearn and nibabel (``pip install '.[fmri]'``).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from behavioral_decoding.config import ExperimentConfig  # noqa: E402
from behavioral_decoding.evaluation.neuroforecast import (  # noqa: E402
    format_forecast_comparison,
)
from behavioral_decoding.io.narps import load_narps  # noqa: E402
from behavioral_decoding.pipelines.train import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="NARPS ds001734 BIDS root")
    parser.add_argument("--derivatives", help="fMRIPrep derivatives root, if separate")
    parser.add_argument("--group", choices=["equalIndifference", "equalRange"])
    parser.add_argument("--space", default="MNI152NLin2009cAsym")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="skip the behaviour-dominated aggregate arm")
    parser.add_argument("--demo", action="store_true",
                        help="synthesise a NARPS-format fixture instead of loading a real one")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    if not args.demo and not args.root:
        parser.error("give --root PATH for a real run, or --demo for the fixture")

    tmp = None
    if args.demo:
        from narps_fixture import write_narps_fixture

        tmp = tempfile.mkdtemp(prefix="narps_demo_")
        write_narps_fixture(tmp, n_subjects=12, n_runs=2, trials_per_run=16, seed=0)
        args.root = tmp
        args.group = "equalRange"
        print("=" * 78)
        print("SYNTHETIC NARPS-FORMAT FIXTURE (simulated; not a finding about brains)")
        print("=" * 78)
        print("fMRI ROI features should predict accept/reject above chance out of fold.")
        print("No brain-beats-behaviour claim is planted; gambles favour the economic")
        print("baseline on the aggregate arm. See docs/narps.md.")
        print()

    dataset = load_narps(
        args.root,
        derivatives=args.derivatives,
        group=args.group,
        space=args.space,
        with_aggregate=not args.no_aggregate,
    )
    print(dataset.describe())
    print()

    cfg = ExperimentConfig(name="narps", output_dir=args.output_dir)
    cfg.data.modalities = dataset.modalities
    record = run_experiment(dataset, cfg)

    if args.demo:
        print()
        print("=" * 78)
        print("GROUND-TRUTH RECOVERY CHECK")
        print("=" * 78)
        per_mod = record["individual"]["per_modality_pooled"]
        fmri_balacc = per_mod["fmri"]["balanced_accuracy"]
        perm_p = record["individual"]["permutation_test"]["p_value"]

        arms = record.get("aggregate") or {}
        if arms:
            print(format_forecast_comparison(arms, "regression"))
            print()

        ok_fmri = fmri_balacc > 0.55
        ok_perm = perm_p < 0.05
        print("  [{}] fMRI ROI features predict accept above chance  "
              "(balanced accuracy {:.3f})".format("PASS" if ok_fmri else "FAIL", fmri_balacc))
        print("  [{}] individual result survives label permutation  "
              "(p = {:.4f})".format("PASS" if ok_perm else "FAIL", perm_p))
        print()

        ok = ok_fmri and ok_perm
        if ok:
            print("The loader recovers the planted reward signal on the true NARPS BIDS")
            print("format: NAcc/MPFC betas predict choice. fMRI plumbing validated. It")
            print("says nothing about brains.")
        else:
            print("The reward signal did not recover; the loader is likely miswiring the")
            print("events, the ROI extraction, or the confounds. Fix before real NARPS.")

        import shutil

        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
