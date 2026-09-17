# Non-canonical token output across post-training stages

Does on-policy RL raise a model's rate of *non-canonical* tokens (emitted
token sequences whose decoded text re-encodes differently, so they are
invisible in any transcript stored as text) relative to the SFT/DPO
checkpoint it started from? Measured on the OLMo-3-7B Think and Instruct
ladders (stages sharing one base, one tokenizer and one codebase), six
checkpoints of OLMo-3-7B-RL-Zero-Math, and the Tulu-3-8B ladder, with
follow-ups on event clustering and on what a non-canonical span does to
the model's later computation.

The state of this repository submitted with the application to Neel Nanda's
MATS stream (Fall 2026) is tagged
[`mats-fall-2026-application`](https://github.com/brendanlong/noncanonical-post-training/tree/mats-fall-2026-application).
Everything after that tag is follow-up work.

- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): design, models and why, prompt
  sets, metrics, run order, and pre-registered predictions.
- [RESULTS.md](RESULTS.md): one entry per run, in order, with the exact
  command and the numbers.
- Artifacts (every rollout with its emitted token IDs and logprobs) live on
  the HuggingFace dataset
  [brendanlong/noncanonical-post-training](https://huggingface.co/datasets/brendanlong/noncanonical-post-training);
  see [Data](#data) below.

## Layout

```
noncanon/
  prompts.py     build the held-out prompt sets (DAPO filtered against the OLMo-3 RL data; AIME)
  generate.py    vLLM rollouts keeping emitted token IDs, sampled + top-k logprobs, finish reason
  records.py     the Parquet record schema shared by generation and analysis
  metrics.py     round-trip metric (exact minimal diff), think/answer split, token classes,
                 integer verifier, length/outcome/position slices, top-k entropy
  summary.py     the summary tables of RESULTS.md (stage ladders; DAPO vs AIME; sampling settings)
  compare.py     cell-vs-cell tests: rollouts flagged (Fisher exact, fixed-window), per-token rate (secondary);
                 cell tables, pairwise table, top spans, flags with one span removed
  clustering.py  do events cluster within rollouts (Poisson excess, hazard, gap shuffle, same-text repeats)
  divergence.py  teacher-forced divergence after a span (emitted ids vs canonical re-tokenization); contagion test
  textstats.py   CJK content and bare-space spans per cell (the DPO question)
  tail.py        where in the next-token distribution non-canonical spans start
  upload.py      push a run directory to the HF dataset
  gpu_check.py   fail fast if CUDA is not usable on the host
  tests/         CPU tests on the real OLMo-3 tokenizer
prompts/         generated prompt files and the DAPO filter report (committed)
scripts/         recompute every cell's metrics; run every comparison; check_results.py verifies the generated tables in RESULTS.md
                 run_checkpoint.sh generates, measures and uploads one checkpoint (the body of skypilot/run.yaml) for gpuc jobs;
                 gpuc_ladder.py writes one gpuc job spec per intermediate RL checkpoint
                 figures.py renders figures/ from the metrics under out/ (numbers to figures/data.json): stages, RL-Zero steps,
                 span rank, divergence, recommended settings, DAPO vs AIME, correct vs incorrect, parsed vs unparsed;
                 random_examples.py and flag_composition.py print the uniformly sampled spans and the flagged-fraction decomposition
figures/         the rendered figures, their data (data.json), and two Markdown tables (random examples, flag composition)
skypilot/        RunPod task definitions
```

## Data

Everything measured is on the HuggingFace dataset
[brendanlong/noncanonical-post-training](https://huggingface.co/datasets/brendanlong/noncanonical-post-training),
one folder per run:

| folder | cell |
|---|---|
| [think-sft](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/think-sft), [think-dpo](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/think-dpo), [think-main](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/think-main) | OLMo-3-7B Think SFT / DPO / RL final, DAPO 500 at temperature 1 (think-dpo and think-main also hold AIME 2024/2025 × 8) |
| [think-dpo-recommended](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/think-dpo-recommended), [think-main-recommended](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/think-main-recommended) | the same two checkpoints at their recommended settings |
| [rlzero-math](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/rlzero-math), [rlzero-math-step300](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/rlzero-math-step300), [-step600](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/rlzero-math-step600), [-step1000](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/rlzero-math-step1000), [-step1400](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/rlzero-math-step1400), [-step1800](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/rlzero-math-step1800) | OLMo-3-7B RL-Zero-Math at step 2,000 (DAPO + AIME) and five earlier checkpoints (DAPO) |
| [instruct-sft](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/instruct-sft), [instruct-dpo](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/instruct-dpo), [instruct-main](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/instruct-main) | OLMo-3-7B Instruct SFT / DPO / RL final, both sampling arms |
| [tulu3-sft](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/tulu3-sft), [tulu3-dpo](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/tulu3-dpo), [tulu3-rlvr](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/tulu3-rlvr), [tulu31-rlvr](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/tulu31-rlvr) | Tulu-3-8B SFT / DPO / RLVR and Tulu-3.1-8B, both sampling arms |
| [think-step0125](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/think-step0125) and [-step0300](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/think-step0300), [-step0600](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/think-step0600), [-step0875](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/think-step0875), [-step1200](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/think-step1200), [-step1375](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/think-step1375) | OLMo-3-7B-Think at six intermediate RL steps (DAPO 500) |
| [instruct-step050](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/instruct-step050) and [-step100](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/instruct-step100), [-step150](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/instruct-step150), [-step200](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/instruct-step200), [-step250](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/instruct-step250), [-step300](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/instruct-step300), [-step350](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/instruct-step350), [-step400](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/instruct-step400) | OLMo-3-7B-Instruct at all eight intermediate RL steps (DAPO 500) |
| [divergence](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/divergence) | per-run outputs of the divergence, contagion and examples measurements |
| [pilot](https://huggingface.co/datasets/brendanlong/noncanonical-post-training/tree/main/pilot) | the 50-prompt pilot (earlier metrics code; superseded) |

Each run folder holds `<prompt set>/<arm>.parquet` (one row per rollout:
prompt and emitted token IDs, text, finish reason, sampled and top-10
logprobs), `<arm>.meta.json` (model, revision, sampling settings,
throughput) and `metrics/` (`summary.json`, `analysis.jsonl` per rollout,
`examples.jsonl` per span, `transcripts.jsonl`). To get everything the
analysis commands in RESULTS.md need:

```python
from huggingface_hub import snapshot_download
snapshot_download("brendanlong/noncanonical-post-training", repo_type="dataset", local_dir="out")
```
## Licenses

The code in this repository (everything outside `prompts/`) is released
under the MIT license. There is deliberately no top-level LICENSE file,
because the repository and the HuggingFace dataset also redistribute
prompt text that is not ours and not MIT:

| data | where | source and license |
|---|---|---|
| DAPO-Math-17k problems (`dapo_heldout.jsonl`, `dapo_sample500.jsonl`, `dapo_pilot50.jsonl`, and the `problem` column of every DAPO parquet on the dataset) | `prompts/`, HF dataset | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) via [open-r1/DAPO-Math-17k-Processed](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed), Apache 2.0 |
| AIME 2024 problems (half of `aime_2024_2025.jsonl` and of the AIME parquets) | `prompts/`, HF dataset | [Maxwell-Jia/AIME_2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024), MIT |
| AIME 2025 problems (the other half) | `prompts/`, HF dataset | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025), CC BY-NC-SA 4.0 |

The AIME problems are competition problems of the Mathematical
Association of America, redistributed here as those datasets redistribute
them. The rollouts on the dataset are model output (OLMo-3, Apache 2.0;
Tulu-3, Llama 3.1 community license) and carry no separate terms.

## Setup

```bash
uv sync --group dev          # CPU: prompts, metrics, tests
uv run pytest -q
uv run python -m noncanon.prompts dapo --sample 500
```

Generation needs a GPU and `uv sync --extra gpu` (vLLM 0.11, torch 2.8 CUDA
12.8). On RunPod via SkyPilot:

```bash
sky launch -c nc-think skypilot/run.yaml -i 20 --down -y -d --env HF_TOKEN --env MODEL=allenai/Olmo-3-7B-Think --env RUN_NAME=think-main
sky logs nc-think
```

Metrics recompute everything from the stored IDs:

```bash
uv run python -m noncanon.metrics --tokenizer allenai/Olmo-3-7B-Think \
    --records out/pilot/*.parquet --out-dir out/pilot/metrics
```
