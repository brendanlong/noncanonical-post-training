"""Generate rollouts with vLLM and keep the emitted token IDs.

    uv run python -m noncanon.generate --model allenai/Olmo-3-7B-Think \
        --prompts prompts/dapo_pilot50.jsonl --arms untruncated --out-dir out/pilot

Writes one Parquet file per sampling arm (``<arm>.parquet``) plus
``<arm>.meta.json`` with the run configuration. Every row carries the prompt
token IDs, the emitted token IDs, the finish reason, the sampled token's
logprob at each position and the top-k (id, logprob) candidates at each
position, so every metric can be recomputed offline from the file.

Sampling is pinned explicitly: ``top_k=-1``, ``min_p=0``, no repetition
penalty, bf16 weights and KV cache, no speculative decoding. Logprobs are
vLLM's raw model logprobs (before temperature/top-p), which is what the
entropy analysis wants.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import vllm
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams, TokensPrompt

from noncanon.records import write_records

# "untruncated" is the full distribution. "recommended" is whatever the
# checkpoint ships in generation_config.json (OLMo-3: temperature 0.6 /
# top_p 0.95, the setting the OLMo 3 report evaluated with,
# https://arxiv.org/html/2512.13961v2#S4.SS1.SSS1; Tulu-3: 0.6 / 0.9), so it
# is resolved per model at run time and recorded in the meta file.
# "standard" is a fixed setting every checkpoint can be run at, which
# "recommended" is not: the intermediate RL revisions and the RL-Zero
# checkpoints ship a generation_config.json with no temperature or top_p at
# all. The values are OLMo-3's own recommendation, so the OLMo-3 endpoints'
# existing "recommended" cells are already at exactly this setting.
ARMS = {
    "recommended": None,
    "untruncated": {"temperature": 1.0, "top_p": 1.0},
    "standard": {"temperature": 0.6, "top_p": 0.95},
    "truncated": {"temperature": 1.0, "top_p": 0.95},
}


def resolve_arms(names: list[str], gen_cfg: GenerationConfig) -> dict[str, dict]:
    arms = {}
    for name in names:
        if name == "recommended":
            # GenerationConfig fills absent fields with class defaults (1.0), so
            # look at what the checkpoint actually wrote.
            explicit = gen_cfg.to_diff_dict()
            assert "temperature" in explicit and "top_p" in explicit, (
                "checkpoint's generation_config.json does not set temperature and top_p; no recommended setting to use"
            )
            arms[name] = {"temperature": float(explicit["temperature"]), "top_p": float(explicit["top_p"])}
        else:
            arms[name] = dict(ARMS[name])
    return arms


def stop_token_ids(tok, gen_cfg: GenerationConfig) -> list[int]:
    """Stop ids: generation_config EOS, tokenizer EOS, the chat template's
    turn-end token, and <|im_end|> where the vocabulary has it.

    Think/Instruct list <|im_end|> and <|endoftext|> in generation_config.
    RL-Zero ships no eos and its template has no special tokens at all, so
    only the explicit <|im_end|> fallback covers its turn end. Tulu's
    template uses plain-text role markers and ends turns with the EOS
    <|end_of_text|>.
    """
    ids = set()
    eos = gen_cfg.eos_token_id
    ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
    ids.add(tok.eos_token_id)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], tokenize=True
    )
    specials = set(tok.all_special_ids) | set(tok.added_tokens_decoder)
    turn_end = [t for t in rendered if t in specials]
    if turn_end:
        ids.add(turn_end[-1])
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tok.unk_token_id and im_end in specials:
        ids.add(im_end)
    return sorted(t for t in ids if isinstance(t, int))



def load_prompts(path: Path, field: str, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    rows = rows[:limit] if limit else rows
    for r in rows:
        assert field in r, f"prompt field {field!r} missing in {path}"
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--prompt-field", default="problem")
    ap.add_argument("--arms", default="untruncated")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1, help="samples per prompt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=34816)
    ap.add_argument("--logprobs", type=int, default=10, help="top-k candidates stored per position")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arm_names) - set(ARMS)
    assert not unknown, f"unknown arms {unknown}; known: {list(ARMS)}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    gen_cfg = GenerationConfig.from_pretrained(args.model, revision=args.revision)
    arms = resolve_arms(arm_names, gen_cfg)
    eos_ids = stop_token_ids(tok, gen_cfg)
    assert eos_ids, "no stop token ids found"

    prompts = load_prompts(args.prompts, args.prompt_field, args.limit)
    prompt_ids = [
        tok.apply_chat_template([{"role": "user", "content": p[args.prompt_field]}], add_generation_prompt=True, tokenize=True)
        for p in prompts
    ]
    longest = max(len(ids) for ids in prompt_ids)
    assert longest + args.max_tokens <= args.max_model_len, (
        f"longest prompt {longest} + max_tokens {args.max_tokens} exceeds max_model_len {args.max_model_len}"
    )

    llm = LLM(
        model=args.model,
        revision=args.revision,
        tokenizer_revision=args.revision,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enable_prefix_caching=True,
        disable_log_stats=False,  # periodic throughput / running / KV-usage lines in the log
    )

    for arm, setting in arms.items():
        sampling = {**setting, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0, "max_tokens": args.max_tokens, "n": args.n}
        params = SamplingParams(logprobs=args.logprobs, stop_token_ids=eos_ids, skip_special_tokens=False, **sampling)
        t0 = time.time()
        outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids], params, use_tqdm=True)
        elapsed = time.time() - t0

        rows = []
        for p, ids, out in zip(prompts, prompt_ids, outputs):
            for j, comp in enumerate(out.outputs):
                # comp.logprobs[i] maps token id -> Logprob for the top-k candidates at
                # position i, plus the sampled token if it fell outside the top-k.
                topk = [sorted(lp.items(), key=lambda kv: kv[1].logprob, reverse=True) for lp in comp.logprobs]
                rows.append(
                    {
                        "prompt_id": p.get("id"),
                        "sample": j,
                        "problem": p[args.prompt_field],
                        "answer": p.get("answer"),
                        "prompt_token_ids": ids,
                        "token_ids": comp.token_ids,
                        "text": comp.text,
                        "finish_reason": comp.finish_reason,
                        "logprobs": [lp[t].logprob for t, lp in zip(comp.token_ids, comp.logprobs)],
                        "topk_ids": [[t for t, _ in cands] for cands in topk],
                        "topk_logprobs": [[v.logprob for _, v in cands] for cands in topk],
                    }
                )
        write_records(rows, args.out_dir / f"{arm}.parquet")

        n_tokens = sum(len(r["token_ids"]) for r in rows)
        meta = {
            "model": args.model,
            "revision": args.revision,
            "arm": arm,
            "sampling": sampling,
            "prompts": str(args.prompts),
            "n_prompts": len(prompts),
            "n_completions": len(rows),
            "output_tokens": n_tokens,
            "elapsed_s": round(elapsed, 1),
            "output_tokens_per_s": round(n_tokens / elapsed, 1),
            "eos_token_ids": eos_ids,
            "logprobs_topk": args.logprobs,
            "logprobs_mode": str(llm.llm_engine.model_config.logprobs_mode),
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
        }
        text = json.dumps(meta, indent=2)
        (args.out_dir / f"{arm}.meta.json").write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
