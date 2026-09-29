#!/bin/bash
# Queue one noncanon.stego run, by default on the SPAR host's owned cards, low
# priority and preemptible (the run resumes from its checkpoint when requeued).
#   gpuc/stego_submit.sh <run-name> [noncanon.stego args...]
#   TARGET="--runpod --gpu A40 --max-price 1.10" gpuc/stego_submit.sh ...
set -euo pipefail
run=$1; shift
spec=.tmp-gpuc/$run.yaml
mkdir -p .tmp-gpuc
cat > "$spec" <<YAML
name: stego-$run
setup: uv sync --frozen --extra gpu
command: uv run --no-sync python -m noncanon.stego --out-dir results/$run --ckpt-dir ckpt/$run --device cuda $*
gpus: 1
env:
  REQUIRE_CUDA: "1"
  PYTHONUNBUFFERED: "1"
  OMP_NUM_THREADS: "8"
secrets: [AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, HF_TOKEN]
outputs:
  - path: results
    s3: s3://brendanlong-experiments/noncanon-stego/{job_id}/results
sync_interval_s: 300
priority: ${PRIORITY:-80}
auto_preempt: true
max_runtime_min: ${MAX_MIN:-900}
estimated_runtime_min: ${EST_MIN:-240}
progress_command: "cat results/$run/progress.txt"
cleanup: on_success
YAML
gpuc submit "$spec" ${TARGET:---host spar} --json | jq -r '.job_id // .error'
