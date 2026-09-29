"""How much probability does the sender put on leaving the canonical copy while
still matching the text? Teacher-forces the canonical copy of each eval cover in
the stego sender prompt and, at every position, sums the probability of the other
tokens whose text is also a prefix of the remaining cover ("alternative" tokens:
mostly shorter pieces of the canonical token). Their sum over a copy is the
expected number of places a sampled copy starts a non-canonical span.

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
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    game = Game(tok, args.model, ("apple", "orange"))
    vocab = tok.get_vocab()
    max_len = max(len(p) for p in vocab)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    covers = json.loads(Path(args.covers).read_text())["eval"][: args.n]

    rows = []
    for i, cover in enumerate(covers):
        prompt = game.prompt(cover, i % 2)
        canon = tok.encode(cover, add_special_tokens=False)
        ids = torch.tensor([prompt + canon], device=args.device)
        with torch.no_grad():
            logits = model(ids).logits[0, len(prompt) - 1 : -1].float()
        probs = logits.softmax(-1)
        data = cover.encode()
        offset, per_pos = 0, []
        for pos, t in enumerate(canon):
            alts = alternatives(vocab, data[offset:], t, max_len)
            per_pos.append({"canonical": tok.convert_ids_to_tokens(t),
                            "p_canonical": probs[pos, t].item(),
                            "p_alt": probs[pos, alts].sum().item() if alts else 0.0,
                            "n_alt": len(alts)})
            offset += len(tok.decode([t]).encode())
        rows.append({"cover": cover, "expected_splits": sum(p["p_alt"] for p in per_pos),
                     "max_p_alt": max(p["p_alt"] for p in per_pos), "positions": per_pos})

    exp = [r["expected_splits"] for r in rows]
    top = sorted((p["p_alt"], p["canonical"], r["cover"]) for r in rows for p in r["positions"])[-10:]
    summary = {"model": args.model, "n_covers": len(rows),
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
