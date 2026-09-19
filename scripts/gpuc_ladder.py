"""Write gpuc job specs for intermediate RL checkpoints, one job per checkpoint.

    uv run python scripts/gpuc_ladder.py think 0125 0300 0600 0875 1200 1375
    uv run python scripts/gpuc_ladder.py instruct 050 100 150 200 250 300 350 400
    uv run python scripts/gpuc_ladder.py --arms standard think-sft main
    for f in gpuc/*.yaml; do gpuc submit "$f" --host spar; done

The model download runs in the setup phase, which the idle-GPU watchdog does
not observe; the job itself is scripts/run_checkpoint.sh. Runtime estimates
are for one A40 (about 300 output tokens/s averaged over a cell).
"""

import argparse
from pathlib import Path

# (model, run-name template, estimated_runtime_min, max_runtime_min); the
# estimates are measured A40 cell times (Think-sized 4.9-6.4 h, Instruct 1.5 h).
# A step of "main" means the released checkpoint rather than a numbered rung.
FAMILIES = {
    "think": ("allenai/Olmo-3-7B-Think", "think-step{step}", 390, 1440),
    "instruct": ("allenai/Olmo-3-7B-Instruct", "instruct-step{step}", 100, 600),
    "think-sft": ("allenai/Olmo-3-7B-Think-SFT", "think-sft", 390, 1440),
    "rlzero": ("allenai/Olmo-3-7B-RL-Zero-Math", "rlzero-math", 330, 1440),
}

SPEC = """\
name: nc-{run_name}
setup: >-
  uv sync --frozen --extra gpu &&
  uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download('{model}', revision='{revision}')"
command: bash scripts/run_checkpoint.sh
gpus: 1
env:
  REQUIRE_CUDA: "1"
  PYTHONUNBUFFERED: "1"
  HF_HUB_ENABLE_HF_TRANSFER: "0"
  MODEL: {model}
  REVISION: {revision}
  RUN_NAME: {run_name}
  ARMS: {arms}
secrets: [HF_TOKEN]
priority: {priority}
# Yield the cards to anything more important; the job is queued again from the
# start, so it only ever loses the attempt in flight.
auto_preempt: true
estimated_runtime_min: {est}
max_runtime_min: {cap}
# vLLM's progress bar, from the job log: prompts finished, which runs ahead of
# wall-clock since the long rollouts finish last.
progress_command: >-
  tr '\\r' '\\n' < "$GPUC_JOB_DIR/log.txt" | grep -o 'Processed prompts: *[0-9]*%' | tail -1 | grep -o '[0-9]*%'
progress_interval_s: 120
cleanup: on_success
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=FAMILIES)
    ap.add_argument("steps", nargs="+", help='revision suffixes, e.g. 0300 for step_0300; "main" for the released checkpoint')
    ap.add_argument("--arms", default="untruncated", help="comma-separated sampling arms (noncanon.generate ARMS)")
    ap.add_argument("--priority", type=int, default=90)
    ap.add_argument("--out", type=Path, default=Path("gpuc"))
    args = ap.parse_args()
    model, name_template, est, cap = FAMILIES[args.family]
    args.out.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        revision = "main" if step == "main" else f"step_{step}"
        run_name = name_template.format(step=step)
        # A non-default arm goes to its own run directory: the upload replaces
        # whole files, so writing a second arm into a directory whose other arm
        # is only on the Hub would overwrite that arm's metrics/analysis.jsonl.
        if args.arms != "untruncated":
            run_name = f"{run_name}-{args.arms.replace(',', '-')}"
        path = args.out / f"{run_name}.yaml"
        path.write_text(
            SPEC.format(model=model, revision=revision, run_name=run_name, arms=args.arms, priority=args.priority, est=est, cap=cap)
        )
        print(path)


if __name__ == "__main__":
    main()
