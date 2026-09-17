#!/usr/bin/env python3
"""Does the flagged-rollout fraction trend over an RL run, and does it differ
between the early and late halves of one?

    uv run python scripts/trend_test.py \
        "Think=125:think-step0125,300:think-step0300,600:think-step0600|875:think-step0875,1200:think-step1200,1375:think-step1375"

Each argument is ``label=<early cells>|<late cells>``, each cell
``<step>:<run>`` under ``out/<run>/dapo_sample500``. The Cochran-Armitage test
runs over all the cells with the training step as the dose; the Fisher test
pools the two groups. Rollouts are the unit in both, which is the primary
metric (see "Rollout-level reanalysis").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, norm


def flagged(run: str) -> tuple[int, int]:
    rows = [json.loads(line) for line in (Path("out") / run / "dapo_sample500" / "metrics" / "analysis.jsonl").open()]
    rows = [r for r in rows if r["file"].startswith("untruncated")]
    return sum(bool(r["event_positions"]) for r in rows), len(rows)


def cochran_armitage(steps: list[int], counts: list[tuple[int, int]]) -> tuple[float, float]:
    x = np.array(steps, float)
    k = np.array([c[0] for c in counts], float)
    n = np.array([c[1] for c in counts], float)
    total = n.sum()
    p = k.sum() / total
    centred = x - (n * x).sum() / total
    z = (k * centred).sum() / np.sqrt(p * (1 - p) * (n * centred**2).sum())
    return z, 2 * norm.sf(abs(z))


def main() -> None:
    print("| ladder | cells | Cochran-Armitage z | p | early | late | Fisher p |")
    print("|---|--:|--:|--:|---|---|--:|")
    for spec in sys.argv[1:]:
        label, _, rest = spec.partition("=")
        halves = [[cell.split(":") for cell in half.split(",")] for half in rest.split("|")]
        steps = [int(step) for half in halves for step, _ in half]
        counts = [flagged(run) for half in halves for _, run in half]
        z, p = cochran_armitage(steps, counts)
        pooled = []
        for half in halves:
            ks, ns = zip(*[flagged(run) for _, run in half])
            pooled.append((sum(ks), sum(ns)))
        (ke, ne), (kl, nl) = pooled
        _, fisher_p = fisher_exact([[ke, ne - ke], [kl, nl - kl]])
        print(
            f"| {label} | {len(counts)} | {z:+.2f} | {p:.2g} "
            f"| {ke}/{ne} = {100 * ke / ne:.1f}% | {kl}/{nl} = {100 * kl / nl:.1f}% | {fisher_p:.2g} |"
        )


if __name__ == "__main__":
    main()
