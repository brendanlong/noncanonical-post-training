#!/usr/bin/env python3
"""Shape of an RL run: does the flagged fraction trend, and is the trend a
straight line or does it move in stages?

    uv run python scripts/trend_test.py --window 4096 \
        "Think=125:think-step0125,300:think-step0300|875:think-step0875,1200:think-step1200"

Each argument is ``label=<early rungs>|<late rungs>``, each rung
``<step>:<run>`` under ``out/<run>/dapo_sample500``. Three statistics per
ladder, rollouts as the unit (the primary metric; see "Rollout-level
reanalysis"), over the RL rungs only, since SFT and DPO are not points on
the run:

- Cochran-Armitage, training step as the dose: is there a monotone trend;
- departure from that trend (homogeneity chi-square minus the trend
  chi-square): does a straight line describe the rungs, or do they move in
  stages? This is the test behind any "plateau then fall" reading, and it
  is the one that does not depend on picking a pair after seeing the curve;
- Fisher on the early rungs pooled against the late ones.

``--window L`` restricts every count to an event within the first L tokens
among rollouts that reached L, which is the length-matched view.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, fisher_exact, norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noncanon.compare import flags, load_rows


def ladder_counts(rungs: list[tuple[int, str]], window: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    steps, k, n, entropy = [], [], [], []
    for step, run in rungs:
        rows = load_rows(Path("out") / run / "dapo_sample500", arm="untruncated")
        flagged, eligible = flags(rows, window)
        steps.append(step)
        k.append(flagged)
        n.append(eligible)
        entropy.append(float(np.mean([r["entropy_mean"] for r in rows])))
    return np.array(steps, float), np.array(k, float), np.array(n, float), entropy


def trend(steps: np.ndarray, k: np.ndarray, n: np.ndarray) -> tuple[float, float, float, int, float]:
    """(Cochran-Armitage z, its p, departure-from-trend chi-square, its df, its p)."""
    total = n.sum()
    p = k.sum() / total
    centred = steps - (n * steps).sum() / total
    z = (k * centred).sum() / np.sqrt(p * (1 - p) * (n * centred**2).sum())
    homogeneity = (((k - n * p) ** 2) / (n * p * (1 - p))).sum()
    departure, df = homogeneity - z**2, len(k) - 2
    return z, 2 * norm.sf(abs(z)), departure, df, float(chi2.sf(departure, df))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=None, help="count only events within the first L tokens")
    ap.add_argument("ladders", nargs="+", help="label=<early rungs>|<late rungs>, rungs as step:run")
    args = ap.parse_args()

    parsed = []
    for spec in args.ladders:
        label, _, rest = spec.partition("=")
        halves = rest.split("|")
        assert len(halves) == 2, f"{label}: want exactly one '|' splitting early from late rungs"
        parsed.append((label, [[(int(s), r) for s, r in (c.split(":") for c in h.split(","))] for h in halves]))

    print(f"Rungs (flagged{f', within first {args.window:,} tokens' if args.window else ''}; entropy is the mean top-10, all positions)\n")
    print("| ladder | step | flagged | mean entropy |")
    print("|---|--:|---|--:|")
    for label, halves in parsed:
        rungs = [r for half in halves for r in half]
        steps, k, n, entropy = ladder_counts(rungs, args.window)
        for (step, _), kk, nn, e in zip(rungs, k, n, entropy):
            print(f"| {label} | {step:,} | {int(kk)}/{int(nn)} = {100 * kk / nn:.1f}% | {e:.4f} |")

    print("\n| ladder | rungs | Cochran-Armitage z | p | departure from trend chi2 | df | p | early | late | Fisher p |")
    print("|---|--:|--:|--:|--:|--:|--:|---|---|--:|")
    for label, halves in parsed:
        rungs = [r for half in halves for r in half]
        steps, k, n, _ = ladder_counts(rungs, args.window)
        z, z_p, departure, df, d_p = trend(steps, k, n)
        sizes = [len(half) for half in halves]
        (ke, ne), (kl, nl) = (int(k[: sizes[0]].sum()), int(n[: sizes[0]].sum())), (int(k[sizes[0] :].sum()), int(n[sizes[0] :].sum()))
        _, fisher_p = fisher_exact([[ke, ne - ke], [kl, nl - kl]])
        print(
            f"| {label} | {len(rungs)} | {z:+.2f} | {z_p:.2g} | {departure:.2f} | {df} | {d_p:.2g} "
            f"| {ke}/{ne} = {100 * ke / ne:.1f}% | {kl}/{nl} = {100 * kl / nl:.1f}% | {fisher_p:.2g} |"
        )


if __name__ == "__main__":
    main()
