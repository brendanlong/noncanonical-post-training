"""Build the cover texts for the tokenization-channel experiment (noncanon/stego.py):
single plain-ASCII sentences of 12-30 words from the TinyStories validation split,
deduplicated, shuffled with a fixed seed, split into train and eval.

    uv run python scripts/stego_covers.py --out prompts/stego_covers.json
"""

import argparse
import json
import random
import re

from datasets import load_dataset

SENTENCE = re.compile(r"[^.!?]+[.!?]")


def sentences(story: str):
    for m in SENTENCE.finditer(" ".join(story.split())):
        s = m.group().strip()
        if s.isascii() and '"' not in s and 12 <= len(s.split()) <= 30 and s[0].isupper():
            yield s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prompts/stego_covers.json")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset("roneneldan/TinyStories", split="validation")
    pool = sorted({s for story in ds["text"] for s in sentences(story)})
    random.Random(args.seed).shuffle(pool)
    need = args.n_train + args.n_eval
    assert len(pool) >= need, (len(pool), need)
    out = {"source": "roneneldan/TinyStories validation", "seed": args.seed,
           "eval": pool[: args.n_eval], "train": pool[args.n_eval : need]}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=0)
        f.write("\n")
    print(f"{len(pool)} candidate sentences; wrote {args.n_train} train, {args.n_eval} eval to {args.out}")


if __name__ == "__main__":
    main()
