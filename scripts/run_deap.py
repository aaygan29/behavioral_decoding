#!/usr/bin/env python3
"""Run the pipeline on DEAP, or on a synthetic DEAP-format fixture.

With a real DEAP download:

    python scripts/run_deap.py --root /path/to/DEAP --target liking

With no download, to see the whole path end to end on the true file format:

    python scripts/run_deap.py --demo

``--demo`` writes a small synthetic dataset in the exact DEAP layout (latin1
pickles, randomised per-participant trial order, the real CSVs), with a latent
per-video valence planted so that EEG frontal asymmetry forecasts a synthetic
market outcome and the noncircular behaviour block does not. It is the DEAP
analogue of ``scripts/run_demo.py``: a check that the loader wires the stimulus
keys and asymmetry features correctly, not a result about brains.

For a real run you need a per-stimulus market outcome to exercise the aggregate
arm. DEAP ships none. Extract YouTube ids from ``video_list.csv``, fetch view
counts through the YouTube Data API, and pass them with ``--counts``. Without
that, only the individual-level arm runs. See ``docs/deap.md``.
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
from behavioral_decoding.io import deap_market  # noqa: E402
from behavioral_decoding.io.deap import DEAPLoader, load_deap  # noqa: E402
from behavioral_decoding.pipelines.train import run_experiment  # noqa: E402


def _market_from_counts(root: str, counts_path: str):
    ids = deap_market.youtube_ids_from_video_list(str(Path(root) / "video_list.csv"))
    raw = deap_market.load_counts_file(counts_path)
    # Accept either exp-NN or YouTube-id keys.
    if all(k.startswith("exp-") for k in raw):
        y_agg, prov = deap_market.market_outcome_from_counts(raw, log_transform=True)
    else:
        y_agg, prov = deap_market.market_outcome_from_counts(
            raw, id_to_experiment=deap_market.invert_id_map(ids), log_transform=True
        )
    print("market outcome: {} stimuli, log-transformed".format(prov["n_stimuli_with_outcome"]))
    return y_agg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="DEAP root directory")
    parser.add_argument("--target", default="liking",
                        choices=["valence", "arousal", "dominance", "liking"])
    parser.add_argument("--binarise", default="fixed", choices=["fixed", "subject_median"])
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--behavior-mode", default="noncircular",
                        choices=["noncircular", "ratings", "none"])
    parser.add_argument("--counts", help="CSV of key,view_count for the aggregate arm")
    parser.add_argument("--demo", action="store_true",
                        help="synthesise a DEAP-format fixture instead of loading a real one")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    if not args.demo and not args.root:
        parser.error("give --root PATH for a real run, or --demo for the fixture")

    tmp = None
    y_aggregate = None

    if args.demo:
        from deap_fixture import market_outcome_dict, write_deap_fixture

        tmp = tempfile.mkdtemp(prefix="deap_demo_")
        gt = write_deap_fixture(tmp, n_participants=12, n_videos=40, seed=0)
        args.root = tmp
        y_aggregate = market_outcome_dict(gt)
        print("=" * 78)
        print("SYNTHETIC DEAP-FORMAT FIXTURE (simulated; not a finding about brains)")
        print("=" * 78)
        print(gt["expected_ordering"])
        print()
    elif args.counts:
        y_aggregate = _market_from_counts(args.root, args.counts)

    loader = DEAPLoader(target=args.target, behavior_mode=args.behavior_mode)
    dataset = load_deap(
        args.root,
        target=args.target,
        binarise=args.binarise,
        threshold=args.threshold,
        loader=loader,
        y_aggregate=y_aggregate,
    )

    print(dataset.describe())
    print()

    cfg = ExperimentConfig(name=f"deap_{args.target}", output_dir=args.output_dir)
    cfg.data.modalities = dataset.modalities
    record = run_experiment(dataset, cfg)

    if args.demo:
        print()
        print("=" * 78)
        print("GROUND-TRUTH RECOVERY CHECK")
        print("=" * 78)
        arms = record.get("aggregate") or {}
        print(format_forecast_comparison(arms, "regression"))
        print()
        eeg = arms.get("eeg_only", {}).get("r2_out_of_sample", float("nan"))
        beh = arms.get("behavior_only", {}).get("r2_out_of_sample", float("nan"))
        ok = eeg > beh
        print("  [{}] EEG forecasts the market better than behaviour  "
              "(EEG oos R2 {:.3f} vs behaviour {:.3f})".format("PASS" if ok else "FAIL", eeg, beh))
        print()
        if ok:
            print("The loader recovers the planted dissociation on the true DEAP file")
            print("format. Plumbing validated. It says nothing about brains.")
        else:
            print("The dissociation did not recover; the loader is likely miswiring the")
            print("stimulus keys or the asymmetry features. Fix before using real DEAP.")
        import shutil

        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
