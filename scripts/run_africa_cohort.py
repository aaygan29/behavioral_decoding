#!/usr/bin/env python3
"""Run the pipeline's behaviour + aggregate-forecasting arms on a real African cohort.

Data source: Kiva microloan records for Kenyan borrowers (n=3023), mirrored as
``kiva.csv`` in the PyCaret datasets repo (a re-host of the Kiva "data science
for good" release; see ``docs/africa_cohort.md`` for the exact URL and terms).
Kiva's own snapshot host (``s3.kiva.org``) has been retired; this mirror is the
one that is actually reachable today.

This is not a neuroimaging dataset, no African fMRI/EEG cohort with the
modality coverage this repo needs is currently open-access (see
``docs/data_sources.md``). It is real, individually-identified behavioural data
with a real, non-synthetic market outcome (whether a Kenyan borrower's loan was
repaid), run through the same ``BehaviorLoader`` -> ``build_dataset`` ->
``run_experiment`` path used everywhere else in this repo, with the same
leakage guards, calibration reporting, and run record.

Individual level:  do loan-level features (amount requested, requester gender,
                    who bears default risk, how much the borrower wrote) predict
                    whether that specific loan is repaid?
Aggregate level:    held out by sector (the "stimulus" here), does the mean of
                    those same features forecast a sector's repayment rate?
                    This is the aggregate-forecasting design in
                    ``docs/design.md``, applied to a real African market
                    instead of a synthetic or Western one.

Usage:
    python scripts/fetch_africa_cohort.py        # downloads data/raw/kiva_kenya.csv
    python scripts/run_africa_cohort.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from behavioral_decoding.config import ExperimentConfig  # noqa: E402
from behavioral_decoding.evaluation.neuroforecast import (  # noqa: E402
    format_forecast_comparison,
)
from behavioral_decoding.features.align import build_dataset  # noqa: E402
from behavioral_decoding.io.behavior import BehaviorLoader  # noqa: E402
from behavioral_decoding.pipelines.train import run_experiment  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "kiva_kenya.csv"

_TAG_RE = re.compile(r"<[^>]+>")


def _word_count(html_text: str) -> int:
    text = _TAG_RE.sub(" ", html_text)
    return len(text.split())


def load_kenya_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["country"] != "Kenya":
                continue
            rows.append(row)
    if not rows:
        raise ValueError(
            f"no Kenya rows found in {csv_path}. Run scripts/fetch_africa_cohort.py first."
        )
    return rows


def build_table(rows: list[dict]):
    import pandas as pd

    records = []
    for i, r in enumerate(rows):
        records.append(
            {
                "subject_id": f"KE-{i:05d}",
                "stimulus_id": r["sector"],
                "loan_amount": float(r["loan_amount"]),
                "gender_female": 1.0 if r["gender"] == "F" else 0.0,
                "description_word_count": float(_word_count(r["en"])),
                "status": float(r["status"]),
            }
        )
    return pd.DataFrame.from_records(records)


# `nonpayment` (whether the lender or the field partner bears default risk) is
# deliberately NOT a feature: it is set by Kiva's field partner based on their
# own risk assessment of the borrower, so it is a near-perfect proxy for
# `status` (2,100/2,165 "lender"-risk loans repay; 772/838 "partner"-risk loans
# do not) rather than an independent behavioural signal. Including it produces
# a balanced accuracy above 0.95 that reflects a confound, not a finding.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="path to the Kiva mirror CSV")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--name", default="africa_cohort_kenya")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        parser.error(
            f"{csv_path} not found. Run `python scripts/fetch_africa_cohort.py` first, "
            "or pass --csv with an existing copy."
        )

    rows = load_kenya_rows(csv_path)
    table = build_table(rows)

    print(f"loaded {len(table)} Kenyan Kiva loan records "
          f"({table['status'].mean():.1%} repaid), {table['stimulus_id'].nunique()} sectors")

    behavior_block = BehaviorLoader().load(
        table,
        subject_column="subject_id",
        stimulus_column="stimulus_id",
        outcome_column="status",
    )

    y_individual = {
        (row.subject_id, row.stimulus_id): row.status for row in table.itertuples()
    }
    sector_rates = table.groupby("stimulus_id")["status"].mean().to_dict()

    dataset = build_dataset(
        {"behavior": behavior_block},
        y_individual=y_individual,
        y_aggregate=sector_rates,
        metadata={
            "source": "Kiva Kenya microloans (kiva.csv, PyCaret mirror of the Kiva "
            "data-science-for-good release)",
            "cohort": "Kenya",
            "n_records": len(table),
        },
    )

    print(dataset.describe())
    print()

    cfg = ExperimentConfig(name=args.name, output_dir=args.output_dir)
    cfg.data.modalities = ["behavior"]
    cfg.data.outcome_column = "status"
    cfg.evaluation.forecast_task = "regression"

    record = run_experiment(dataset, cfg)

    print()
    print("=" * 78)
    print("AFRICA COHORT (Kenya, real Kiva loan data) - AGGREGATE ARM BY SECTOR")
    print("=" * 78)
    arms = record.get("aggregate") or {}
    print(format_forecast_comparison(arms, "regression"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
