#!/usr/bin/env python3
"""Download ASZED-153, the African Schizophrenia EEG Dataset.

Kwara/OAUTHC (Ile-Ife / Ilesa, Nigeria), 153 subjects (76 patients, 77
matched controls), 16-20ch international 10/20 EEG, four recording phases per
session (two short baseline segments, two longer task/passive-listening
segments), CC-BY on Zenodo. See docs/africa_neuroprivacy.md for what this is
used for and why.

Verifies the download against the published MD5 before extracting, because a
corrupted or substituted file would silently produce garbage features.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = "https://zenodo.org/records/14178398/files/ASZED-153.zip?download=1"
EXPECTED_MD5 = "0e684f6c308c71ab59c7e3475c628691"

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "ASZED-153.zip"
EXTRACT_DIR = RAW_DIR / "aszed"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print(f"fetching {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, ZIP_PATH)

    digest = _md5(ZIP_PATH)
    print(f"downloaded md5: {digest}")
    if digest != EXPECTED_MD5:
        print(f"MD5 MISMATCH: expected {EXPECTED_MD5}, got {digest}", file=sys.stderr)
        return 1
    print("md5 verified against the Zenodo record")

    if not EXTRACT_DIR.exists() or not any(EXTRACT_DIR.iterdir()):
        print(f"extracting to {EXTRACT_DIR}")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(EXTRACT_DIR)

    spreadsheet = next(EXTRACT_DIR.rglob("ASZED_SpreadSheet.csv"), None)
    if spreadsheet is None:
        print("extraction did not produce ASZED_SpreadSheet.csv; layout may have changed",
              file=sys.stderr)
        return 1
    print(f"ready: {spreadsheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
