#!/usr/bin/env python3
"""How much would adding another biosignal channel buy an attacker?

This script answers a question the repo cannot answer by measurement, because no
open cohort carries cardiac, facial, pupil and economic-choice data on the same
people: if an actor already reads one channel, what does the second, third and
fourth add?

It does not invent effect sizes for channels we have not run. Instead it makes
the projection *parametric*: given channels of a stated individual strength, the
combination rule proved in ``proofs/FusionMath.lean`` fixes what the combined
strength must be. The user supplies the per-channel strength; the mathematics
supplies the answer.

Method
------
Treat each channel as a noisy reading of the same latent decision variable, in
the equal-variance Gaussian detection model. Then

    balanced accuracy  BA = Phi(d' / 2)          so   d' = 2 * Phi_inv(BA)

and precision adds across independent channels:

    d'_fused^2 = sum_i d'_i^2

Real sensors are not independent: heart rate, skin conductance, pupil and facial
tone are all partly driven by one arousal system. With equally-correlated errors
(correlation rho) the usable precision is

    d'_fused^2 = (sum_i d'_i^2) * n / (n + n*(n-1)*rho)      [rho = 0 recovers independence]

which is why the script reports a band over rho rather than a single number.

Calibration
-----------
``--validate`` runs the formula against the measured synthetic control in
``results/biosignal_fusion.json``: it predicts the combined balanced accuracy
from the five per-family accuracies and compares it with what the pipeline
actually achieved. A projection that cannot reproduce a result we measured has
no business being shown on a slide.

Usage
-----
    python scripts/project_fusion_gain.py --validate
    python scripts/project_fusion_gain.py --per-channel 0.60 --max-channels 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def _phi_inv(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability out of range: {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        num = ((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]
        den = (((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1
        return num / den
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        num = ((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]
        den = (((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
    den = ((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1
    return num / den


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ba_to_dprime(ba: float) -> float:
    """Balanced accuracy -> discriminability. Chance (0.5) maps to 0."""
    return 2.0 * _phi_inv(min(max(ba, 0.500001), 0.999999))


def dprime_to_ba(d: float) -> float:
    return _phi(d / 2.0)


def combine(bas: list[float], rho: float = 0.0) -> float:
    """Combined balanced accuracy for channels of the given strengths."""
    ds = [ba_to_dprime(b) for b in bas if b > 0.5]
    if not ds:
        return 0.5
    n = len(ds)
    total_sq = sum(d * d for d in ds)
    if n > 1 and rho > 0:
        total_sq *= n / (n + n * (n - 1) * rho)
    return dprime_to_ba(math.sqrt(total_sq))


def validate() -> dict:
    path = RESULTS / "biosignal_fusion.json"
    data = json.loads(path.read_text())
    single = data["single_family"]
    measured = data["fusion"]["soft"]["metrics"]["balanced_accuracy"]

    contributing = {k: v["balanced_accuracy"] for k, v in single.items()
                    if v["balanced_accuracy"] > 0.5}
    bas = list(contributing.values())

    out = {
        "per_channel_measured": contributing,
        "measured_fusion": measured,
        "predicted_independent": combine(bas, rho=0.0),
    }
    for rho in (0.2, 0.3, 0.5):
        out[f"predicted_rho_{rho}"] = combine(bas, rho=rho)

    print("=" * 74)
    print("CALIBRATION against the measured synthetic control")
    print("=" * 74)
    for k, v in sorted(contributing.items()):
        print(f"  {k:<14} {v:.3f}")
    print(f"\n  measured combined            : {measured:.3f}")
    print(f"  predicted, independent errors: {out['predicted_independent']:.3f}")
    for rho in (0.2, 0.3, 0.5):
        print(f"  predicted, rho = {rho}         : {out[f'predicted_rho_{rho}']:.3f}")
    best_rho = min((0.0, 0.2, 0.3, 0.5),
                   key=lambda r: abs(combine(bas, rho=r) - measured))
    print(f"\n  closest match at rho = {best_rho}")
    print("  Two of these channels (heart, skin) were built to share one arousal")
    print("  driver, so a correlation above zero is the expected fit, not a fudge.")
    out["closest_rho"] = best_rho
    return out


def project(per_channel: float, max_channels: int) -> dict:
    rows = []
    print("=" * 74)
    print(f"PROJECTION: adding channels each worth {per_channel:.2f} on its own")
    print("=" * 74)
    print(f"  {'channels':<10}{'independent':>14}{'rho=0.3':>12}{'rho=0.5':>12}")
    for n in range(1, max_channels + 1):
        bas = [per_channel] * n
        row = {
            "n_channels": n,
            "independent": combine(bas, 0.0),
            "rho_0.3": combine(bas, 0.3),
            "rho_0.5": combine(bas, 0.5),
        }
        rows.append(row)
        print(f"  {n:<10}{row['independent']:>14.3f}{row['rho_0.3']:>12.3f}"
              f"{row['rho_0.5']:>12.3f}")
    print("\n  Correlated channels still help, just less. Heart, skin, pupil and")
    print("  facial tone are all partly arousal-driven, so the honest reading of")
    print("  a multi-sensor deployment sits nearer the rho columns than the left.")
    return {"per_channel": per_channel, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true",
                    help="check the formula against the measured control")
    ap.add_argument("--per-channel", type=float, default=0.60,
                    help="individual strength of each added channel")
    ap.add_argument("--max-channels", type=int, default=5)
    ap.add_argument("--output", default=str(RESULTS / "fusion_projection.json"))
    args = ap.parse_args()

    out = {"model": "equal-variance Gaussian detection; precision adds across "
                    "independent channels; rho = equicorrelated error"}
    if args.validate:
        out["calibration"] = validate()
        print()
    out["projection"] = project(args.per_channel, args.max_channels)

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
