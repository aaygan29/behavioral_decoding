#!/usr/bin/env python3
"""Download the Kiva microloan mirror and cut out the Kenyan cohort.

Kiva's own data-snapshot host (``s3.kiva.org``) no longer resolves. The dataset
below is a public re-host of the same Kiva "data science for good" loan release
in the PyCaret example-datasets repo, which does resolve. See
``docs/africa_cohort.md`` for provenance, licensing notes, and what to do if
this mirror ever goes away too.
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/kiva.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "kiva_all.csv"
KENYA_PATH = REPO_ROOT / "data" / "raw" / "kiva_kenya.csv"


def main() -> int:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, RAW_PATH)
    print(f"wrote {RAW_PATH} ({RAW_PATH.stat().st_size} bytes)")

    with RAW_PATH.open(encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        rows = [r for r in reader if r["country"] == "Kenya"]

    if not rows:
        print("no Kenya rows found in the mirror; source format may have changed", file=sys.stderr)
        return 1

    with KENYA_PATH.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {KENYA_PATH}: {len(rows)} Kenyan loan records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
