#!/usr/bin/env python3
"""Biosignal-privacy demonstration: decode an undisclosed health trait from a routine PPG.

The companion to ``scripts/run_africa_neuroprivacy.py``. That script shows this
repo's own accuracy-weighted multibagging ensemble recovering a protected health
attribute (schizophrenia diagnosis) from four short EEG segments. This script
makes the same point for a *non-neural* biosignal, which is the gap
``docs/biosignal_privacy_generalization.md`` argues about structurally: the
framework is modality-agnostic, so the risk is not EEG-specific.

Real data: the PPG-BP Database (Liang, Chen, Liu & Elgendi, 2018, *Scientific
Data*, doi:10.1038/sdata.2018.20). 219 subjects at Guilin People's Hospital,
Guilin, China. Each subject has three 2.1 s fingertip photoplethysmogram
segments sampled at 1 kHz (0.5-12 Hz bandpass applied by the original authors),
recorded *for blood-pressure-estimation research*. The spreadsheet also carries
each subject's diabetes status, which the PPG recording was not ostensibly
about.

Design, identical in shape to the ASZED-153 script:
  * the three PPG segments per subject become three "modalities" (seg1/seg2/seg3),
  * each is reduced to a fixed pulse-wave feature vector by ``ppg_features`` below
    (pulse-morphology and second-derivative / SDPPG features are the standard,
    literature-grounded reductions tied to arterial stiffness, ageing, and
    glycemic state),
  * all three run through the *same* ``run_experiment`` path: per-modality bagged
    elastic-net, in-fold SMOTE, nested subject-grouped CV, out-of-fold
    reliability weighting, calibration and subject-bootstrap reporting,
  * the label is diabetes status (has vs. does not), a property of the person,
    not of any segment.

There is one trial per subject per modality, so (as in the Kenya cohort) the
within-subject label permutation is a structural no-op and the subject-resampled
bootstrap CI is the statistic to read. That is reported, not hidden.

Usage:
    python scripts/run_biosignal_privacy.py            # downloads to data/raw/ppgbp on first run
    python scripts/run_biosignal_privacy.py --target sex   # positive control
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from behavioral_decoding.config import ExperimentConfig  # noqa: E402
from behavioral_decoding.features.align import build_dataset, summarise_alignment  # noqa: E402
from behavioral_decoding.io.base import ModalityBlock  # noqa: E402
from behavioral_decoding.pipelines.train import run_experiment  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "raw" / "ppgbp"
FIGSHARE_URL = "https://ndownloader.figshare.com/files/9441097"  # "PPG-BP Database.zip"
FS = 1000.0  # Hz, per the dataset paper
N_SEGMENTS = 3
RESAMPLE_LEN = 200  # fixed length for the averaged pulse wave


# --------------------------------------------------------------------------- io
def _ensure_data() -> Path:
    """Return the '0_subject' directory, downloading the archive on first run."""
    subj_dir = next(DATA_ROOT.rglob("0_subject"), None)
    if subj_dir is not None:
        return subj_dir
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"downloading PPG-BP Database from figshare to {DATA_ROOT} ...")
    raw = urllib.request.urlopen(FIGSHARE_URL, timeout=120).read()  # noqa: S310
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(DATA_ROOT)
    subj_dir = next(DATA_ROOT.rglob("0_subject"), None)
    if subj_dir is None:
        raise FileNotFoundError(f"'0_subject' not found under {DATA_ROOT} after extraction")
    return subj_dir


def _labels(target: str):
    import pandas as pd

    xlsx = next(DATA_ROOT.rglob("PPG-BP dataset.xlsx"))
    df = pd.read_excel(xlsx, header=1)
    out = {}
    for _, row in df.iterrows():
        sid = str(int(row["subject_ID"]))
        if target == "diabetes":
            val = str(row["Diabetes"]).strip().lower()
            out[sid] = 1.0 if ("diabet" in val) else 0.0
        elif target == "sex":
            out[sid] = 1.0 if str(row["Sex(M/F)"]).strip().lower().startswith("m") else 0.0
        elif target == "hypertension":
            val = str(row["Hypertension"]).strip().lower()
            out[sid] = 1.0 if "stage" in val else 0.0
        else:  # noqa: PLR5501
            raise ValueError(f"unknown target {target!r}")
    return out


# --------------------------------------------------------------------- features
def _pulse_onsets(sig: np.ndarray) -> np.ndarray:
    """Indices of pulse feet: local minima below the running median, min-spaced."""
    from scipy.signal import find_peaks

    min_gap = int(0.4 * FS)  # <=150 bpm
    troughs, _ = find_peaks(-sig, distance=min_gap, height=-np.median(sig))
    return troughs


def ppg_features(sig: np.ndarray) -> np.ndarray:
    """Fixed-length, literature-grounded pulse-wave reduction of one 2.1 s PPG.

    Rate + variability, pulse-morphology (systolic/diastolic timing, width, area),
    and second-derivative (SDPPG) b/a and d/a ratios that the PPG-ageing and
    PPG-glycemia literature ties to arterial stiffness. Self-referential
    normalisation only (per-segment z-score), so nothing leaks across the fold.
    """
    from scipy.signal import welch

    x = np.asarray(sig, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-9)

    onsets = _pulse_onsets(x)
    feats: list[float] = []

    # --- rate and short-window variability
    if len(onsets) >= 2:
        ibi = np.diff(onsets) / FS
        hr = 60.0 / ibi.mean()
        feats += [hr, ibi.std(), (ibi.std() / ibi.mean()), float(len(onsets))]
    else:
        feats += [np.nan, np.nan, np.nan, float(len(onsets))]

    # --- averaged pulse wave over complete beats, resampled to RESAMPLE_LEN
    beats = []
    for a, b in zip(onsets[:-1], onsets[1:]):
        seg = x[a:b]
        if 0.3 * FS <= len(seg) <= 1.5 * FS:
            beats.append(np.interp(np.linspace(0, 1, RESAMPLE_LEN),
                                   np.linspace(0, 1, len(seg)), seg))
    if beats:
        pw = np.mean(beats, axis=0)
        pw = (pw - pw.min()) / (pw.ptp() + 1e-9)
        sys_i = int(np.argmax(pw))
        half = pw >= 0.5
        width_half = float(half.sum()) / RESAMPLE_LEN
        rise = sys_i / RESAMPLE_LEN
        area = float(pw.mean())
        # reflected / diastolic hump: max after the systolic peak past 40% of cycle
        tail = pw[max(sys_i + 1, int(0.4 * RESAMPLE_LEN)):]
        refl = float(tail.max()) if tail.size else np.nan
        aug_index = refl - 0.5 if not np.isnan(refl) else np.nan  # crude AIx proxy
        # SDPPG (second derivative) a/b/d wave ratios
        d2 = np.gradient(np.gradient(pw))
        a_wave = float(d2[: int(0.15 * RESAMPLE_LEN)].max()) if RESAMPLE_LEN > 6 else np.nan
        b_wave = float(d2[: int(0.25 * RESAMPLE_LEN)].min())
        d_wave = float(d2[int(0.25 * RESAMPLE_LEN): int(0.6 * RESAMPLE_LEN)].min())
        a_ok = np.isfinite(a_wave) and abs(a_wave) > 1e-9
        ba = b_wave / a_wave if a_ok else 0.0
        da = d_wave / a_wave if a_ok else 0.0
        trapz = getattr(np, "trapezoid", np.trapz)
        feats += [width_half, rise, area, refl, aug_index, ba, da,
                  float(trapz(pw)) / RESAMPLE_LEN]
    else:
        feats += [np.nan] * 8

    # --- spectral shape and higher moments
    from scipy.stats import kurtosis, skew

    fx, px = welch(x, fs=FS, nperseg=min(512, len(x)))
    def band(lo, hi):
        m = (fx >= lo) & (fx < hi)
        return float(px[m].sum())
    tot = band(0.0, 20.0) + 1e-12
    feats += [band(0.0, 2.5) / tot, band(2.5, 5.0) / tot, band(5.0, 10.0) / tot,
              float(skew(x)), float(kurtosis(x))]

    f = np.asarray(feats, dtype=float)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


FEATURE_NAMES = [
    "hr", "ibi_std", "ibi_cv", "n_onsets",
    "width_half", "rise_frac", "pw_area", "reflected_max", "aug_index_proxy",
    "sdppg_b_a", "sdppg_d_a", "pw_trapz",
    "bp_0_2p5", "bp_2p5_5", "bp_5_10", "skew", "kurtosis",
]


# -------------------------------------------------------------------- assembly
def build_blocks(target: str):
    subj_dir = _ensure_data()
    labels = _labels(target)

    per_seg: list[dict[str, np.ndarray]] = [dict() for _ in range(N_SEGMENTS)]
    used = []
    for sid, _y in sorted(labels.items(), key=lambda kv: int(kv[0])):
        paths = [subj_dir / f"{sid}_{k}.txt" for k in range(1, N_SEGMENTS + 1)]
        if not all(p.exists() for p in paths):
            continue
        try:
            vecs = [ppg_features(np.loadtxt(p)) for p in paths]
        except Exception as exc:  # noqa: BLE001
            print(f"  skip subject {sid}: {exc}")
            continue
        for k in range(N_SEGMENTS):
            per_seg[k][sid] = vecs[k]
        used.append(sid)

    blocks = {}
    for k in range(N_SEGMENTS):
        sids = list(per_seg[k].keys())
        X = np.vstack([per_seg[k][s] for s in sids])
        blocks[f"seg{k + 1}"] = ModalityBlock(
            name=f"seg{k + 1}",
            X=X,
            subject_ids=np.array(sids),
            stimulus_ids=np.array(["seg"] * len(sids)),
            feature_names=list(FEATURE_NAMES),
            provenance={
                "source": "PPG-BP Database (figshare 5459299; Sci Data 2018)",
                "cohort": "Guilin People's Hospital, Guilin, China (n=219)",
                "signal": f"fingertip PPG, 2.1 s @ {FS:g} Hz, segment {k + 1} of {N_SEGMENTS}",
                "feature_family": "pulse-wave morphology + SDPPG ratios + spectral shape",
                "stated_recording_purpose": "blood-pressure estimation research",
                "decoded_label": target,
            },
        )
    return blocks, labels, used


def _run_one(target, learner, reconciliation, seed=0):
    """One pre-specified grid cell. Returns pooled balanced accuracy + bootstrap CI."""
    blocks, labels, used = build_blocks(target)
    y_individual = {(s, "seg"): float(labels[s]) for s in used}
    dataset = build_dataset(blocks, y_individual=y_individual, y_aggregate=None,
                            metadata={"decoded_label": target})
    cfg = ExperimentConfig(name=f"grid_{target}_{learner}_{reconciliation}",
                           output_dir="results", seed=seed)
    cfg.data.modalities = list(blocks.keys())
    cfg.data.outcome_column = target
    cfg.model.reconciliation = reconciliation
    for m in cfg.data.modalities:
        cfg.model.base_learner[m] = learner
        cfg.model.n_bags[m] = 25
    rec = run_experiment(dataset, cfg, write=False)
    mm = rec["individual"]["pooled_metrics"]
    ci = rec["individual"]["bootstrap_ci"]
    return {
        "target": target, "learner": learner, "reconciliation": reconciliation,
        "n_subjects": len(used), "majority_acc": mm["majority_baseline_accuracy"],
        "bal_acc": mm["balanced_accuracy"], "roc_auc": mm["roc_auc"],
        "ece": mm["ece"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
        "clears_chance": bool(ci["lo"] > 0.5),
    }


def _grid() -> int:
    import json

    targets = ["diabetes", "hypertension", "sex"]
    learners = ["elasticnet", "random_forest", "gradient_boosting"]
    recons = ["accuracy_weighted", "majority", "soft"]
    rows = []
    for t in targets:
        for lrn in learners:
            for rc in recons:
                print(f"  running  {t:<12} {lrn:<18} {rc}")
                try:
                    rows.append(_run_one(t, lrn, rc))
                except Exception as exc:  # noqa: BLE001
                    print(f"    FAILED: {exc}")
    out = REPO_ROOT / "results" / "biosignal_privacy_ppgbp_grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 90)
    print("PPG-BP ANALYTIC GRID  (pre-specified: 3 targets x 3 learners x 3 reconciliation rules)")
    print("pooled nested-CV out-of-fold; CI = subject-resampled bootstrap 95% on balanced accuracy")
    print("=" * 90)
    print(f"{'target':<12} {'learner':<18} {'reconcile':<18} {'bal_acc':>8} {'95% CI':>16} "
          f"{'AUC':>6} {'>chance':>8}")
    for r in rows:
        print(f"{r['target']:<12} {r['learner']:<18} {r['reconciliation']:<18} "
              f"{r['bal_acc']:>8.3f} {r['ci_lo']:>7.3f}-{r['ci_hi']:<7.3f} {r['roc_auc']:>6.3f} "
              f"{'yes' if r['clears_chance'] else 'no':>8}")
    n_clear = sum(r["clears_chance"] for r in rows)
    print(f"\n{n_clear}/{len(rows)} grid cells have a bootstrap CI lower bound above chance.")
    print("Read against the grid size, not cell by cell: with 27 pre-specified cells and no")
    print("multiplicity correction, a handful clearing chance is weak evidence. The honest")
    print("summary for PPG-BP is a borderline cardiovascular-status leak (hypertension the")
    print("strongest), no usable sex leak from this 17-feature 2.1 s reduction, and nothing")
    print("like the ASZED-153 EEG effect. See docs/biosignal_privacy_generalization.md.")
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="diabetes",
                    choices=["diabetes", "sex", "hypertension"])
    ap.add_argument("--grid", action="store_true",
                    help="run the pre-specified target x learner x reconciliation grid")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if args.grid:
        return _grid()
    name = args.name or f"biosignal_privacy_ppgbp_{args.target}"

    print(f"building PPG-BP blocks (target = {args.target}) ...")
    blocks, labels, used = build_blocks(args.target)
    print(f"{len(used)} subjects with all {N_SEGMENTS} PPG segments")
    print(summarise_alignment(blocks))

    y_individual = {(s, "seg"): float(labels[s]) for s in used}
    dataset = build_dataset(
        blocks,
        y_individual=y_individual,
        y_aggregate=None,
        metadata={
            "source": "PPG-BP Database (figshare 5459299, CC-BY 4.0)",
            "cohort": "Guilin, China",
            "n_subjects": len(used),
            "purpose": "biosignal-privacy risk demonstration: undisclosed health-trait "
                       "decoding from a routine fingertip PPG",
            "decoded_label": args.target,
        },
    )
    print(dataset.describe(), "\n")

    cfg = ExperimentConfig(name=name, output_dir=args.output_dir)
    cfg.data.modalities = list(blocks.keys())
    cfg.data.outcome_column = args.target
    for m in cfg.data.modalities:
        cfg.model.base_learner[m] = "elasticnet"
        cfg.model.n_bags[m] = 25

    record = run_experiment(dataset, cfg)

    m = record["individual"]["pooled_metrics"]
    ci = record["individual"]["bootstrap_ci"]
    perm = record["individual"]["permutation_test"]
    print("\n" + "=" * 78)
    print("BIOSIGNAL-PRIVACY DEMONSTRATION (PPG-BP, real fingertip PPG cohort)")
    print(f"predicting: {args.target} (a property of the person, not of the recording)")
    print("=" * 78)
    print(f"  n subjects (grouped CV)     {len(used)}")
    print(f"  n trials                    {int(m['n'])}")
    print(f"  majority baseline accuracy  {m['majority_baseline_accuracy']:.3f}")
    print(f"  balanced accuracy           {m['balanced_accuracy']:.3f}")
    print(f"  ROC AUC                     {m['roc_auc']:.3f}")
    print(f"  Brier / ECE                 {m['brier']:.3f} / {m['ece']:.3f}")
    print(f"  bootstrap 95% CI (bal acc)  {ci['lo']:.3f} - {ci['hi']:.3f}")
    print(f"  permutation p               {perm['p_value']:.4f}  (n_perm={int(perm['n_perm'])}, "
          f"degenerate for a subject-constant label)")
    print("\n  per-segment out-of-fold reliability (ensemble weighting):")
    for seg, mm in record["individual"]["per_modality_pooled"].items():
        print(f"    {seg:<6} balanced acc {mm['balanced_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
