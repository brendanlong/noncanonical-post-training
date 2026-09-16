"""Write gpuc job specs for intermediate RL checkpoints, one job per checkpoint.

    uv run python scripts/gpuc_ladder.py think 0125 0300 0600 0875 1200 1375
    uv run python scripts/gpuc_ladder.py instruct 050 100 150 200 250 300 350 400
    for f in gpuc/*.yaml; do gpuc submit "$f" --host spar; done

The model download runs in the setup phase, which the idle-GPU watchdog does
not observe; the job itself is scripts/run_checkpoint.sh. Runtime estimates
are for one A40 (about 150-200 output tokens/s on 7B at long contexts).
"""

import argparse
from pathlib import Path

FAMILIES = {
    "think": ("allenai/Olmo-3-7B-Think", 480, 1440),
    "instruct": ("allenai/Olmo-3-7B-Instruct", 150, 600),
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
secrets: [HF_TOKEN]
priority: {priority}
estimated_runtime_min: {est}
max_runtime_min: {cap}
low_util:
  enabled: true
  window_min: 45
  floor_pct: 5
  grace_min: 20
cleanup: on_success
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=FAMILIES)
    ap.add_argument("steps", nargs="+", help="revision suffixes, e.g. 0300 for step_0300")
    ap.add_argument("--priority", type=int, default=90)
    ap.add_argument("--out", type=Path, default=Path("gpuc"))
    args = ap.parse_args()
    model, est, cap = FAMILIES[args.family]
    args.out.mkdir(exist_ok=True)
    for step in args.steps:
        run_name = f"{args.family}-step{step}"
        path = args.out / f"{run_name}.yaml"
        path.write_text(SPEC.format(model=model, revision=f"step_{step}", run_name=run_name, priority=args.priority, est=est, cap=cap))
        print(path)


if __name__ == "__main__":
    main()
