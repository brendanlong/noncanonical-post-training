"""How much probability does the sender put on leaving the canonical copy while
still matching the text? Teacher-forces the canonical copy of each eval cover in
the stego sender prompt and, at every position, sums the probability of the other
tokens whose text is also a prefix of the remaining cover ("alternative" tokens:
mostly shorter pieces of the canonical token). Their sum over a copy is the
expected number of places a sampled copy starts a non-canonical span.

With ``--temperatures``, the same is done on the distribution sampled at each
temperature, and per copy it reports P(canonical copy, then stop) = product of the
canonical tokens' probabilities, and P(exact copy with one split) ~= sum over
positions of P(alternative) x product of the canonical probabilities elsewhere (ending on any stop token).
The second assumes a split word is finished correctly and the copy returns to
canonical, which is not measured, so it is an upper bound.

    uv run --no-sync python -m noncanon.stego_split_mass --model Qwen/Qwen3-1.7B \
        --out results/split-mass/qwen3-1.7b.json --device cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from noncanon.stego_game import Game

BYTE_CHARS = bytes_to_unicode()


def alternatives(vocab: dict[str, int], remaining: bytes, canonical_id: int, max_len: int) -> list[int]:
    """IDs of tokens other than the canonical one whose bytes start ``remaining``."""
    s = "".join(BYTE_CHARS[b] for b in remaining[:max_len])
    ids = (vocab.get(s[:k]) for k in range(1, len(s) + 1))
    return [i for i in ids if i is not None and i != canonical_id]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--covers", default="prompts/stego_covers.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperatures", default="1.0")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    game = Game(tok, args.model, ("apple", "orange"))
    vocab = tok.get_vocab()
    max_len = max(len(p) for p in vocab)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    covers = json.loads(Path(args.covers).read_text())["eval"][: args.n]

    temps = [float(t) for t in args.temperatures.split(",")]
    stop = min(game.stops)  # any stop token will do for teacher forcing; all are scored below
    stops = sorted(game.stops)
    rows, by_temp = [], {t: [] for t in temps}
    for i, cover in enumerate(covers):
        prompt = game.prompt(cover, i % 2)
        canon = tok.encode(cover, add_special_tokens=False)
        ids = torch.tensor([prompt + canon + [stop]], device=args.device)
        with torch.no_grad():
            logits = model(ids).logits[0, len(prompt) - 1 : -1, : game.n_vocab].float()
        data = cover.encode()
        offset, alts = 0, []
        for t in canon:
            alts.append(alternatives(vocab, data[offset:], t, max_len))
            offset += len(tok.decode([t]).encode())
        targets = torch.tensor(canon + [stop], device=args.device)
        for temp in temps:
            probs = (logits / temp).softmax(-1)
            pc = probs[torch.arange(len(targets)), targets].double()
            pc[-1] = probs[-1, stops].sum()
            pa = torch.tensor([probs[k, a].sum().item() if a else 0.0 for k, a in enumerate(alts)] + [0.0],
                              dtype=torch.float64, device=args.device)
            p_canon = pc.prod().item()
            by_temp[temp].append({"p_canonical_copy": p_canon,
                                  "p_split_copy": (p_canon * (pa / pc)).sum().item(),
                                  "expected_splits": pa.sum().item()})
            if temp == 1.0:
                per_pos = [{"canonical": tok.convert_ids_to_tokens(t), "p_canonical": pc[k].item(),
                            "p_alt": pa[k].item(), "n_alt": len(alts[k])} for k, t in enumerate(canon)]
                rows.append({"cover": cover, "expected_splits": pa.sum().item(),
                             "max_p_alt": pa.max().item(), "positions": per_pos})

    temp_summary = []
    for temp, rs in by_temp.items():
        pc = statistics.mean(r["p_canonical_copy"] for r in rs)
        ps = statistics.mean(r["p_split_copy"] for r in rs)
        temp_summary.append({"temperature": temp, "p_canonical_copy": pc, "p_split_copy": ps,
                             "split_share_of_exact": ps / (pc + ps) if pc + ps else None,
                             "split_copies_per_128": 128 * ps,
                             "mean_expected_splits": statistics.mean(r["expected_splits"] for r in rs)})
        print(f"T={temp}: P(canonical copy)={pc:.3g} P(one-split copy)={ps:.3g} "
              f"split share of exact={temp_summary[-1]['split_share_of_exact']:.3g} "
              f"per 128 rollouts={128 * ps:.3g}", flush=True)
    if not rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"temperatures": temp_summary}) + "\n")
        return

    exp = [r["expected_splits"] for r in rows]
    top = sorted((p["p_alt"], p["canonical"], r["cover"]) for r in rows for p in r["positions"])[-10:]
    summary = {"model": args.model, "n_covers": len(rows), "temperatures": temp_summary,
               "mean_expected_splits": statistics.mean(exp),
               "median_expected_splits": statistics.median(exp),
               "max_expected_splits": max(exp),
               "mean_tokens": statistics.mean(len(r["positions"]) for r in rows),
               "top_positions": [{"p_alt": p, "canonical": c, "cover": s} for p, c, s in reversed(top)]}
    print(json.dumps(summary, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "covers": rows}) + "\n")


if __name__ == "__main__":
    main()
