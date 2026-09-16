#!/usr/bin/env bash
# Generate, measure and upload one checkpoint: the per-checkpoint body of
# skypilot/run.yaml, for gpuc jobs (scripts/gpuc_ladder.py writes the specs).
#
#   MODEL=allenai/Olmo-3-7B-Think REVISION=step_0300 RUN_NAME=think-step0300 bash scripts/run_checkpoint.sh
#
# PROMPTS entries take an optional ":n" suffix for samples per prompt. Records
# are uploaded as soon as each prompt file finishes, since they are the
# expensive part; a metrics failure is logged and the records still uploaded.
set -euo pipefail
export UV_NO_SYNC=1
: "${MODEL:?}" "${REVISION:?}" "${RUN_NAME:?}"
PROMPTS="${PROMPTS:-prompts/dapo_sample500.jsonl}"
ARMS="${ARMS:-untruncated}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
HF_DATASET="${HF_DATASET:-brendanlong/noncanonical-post-training}"

upload() { uv run python -m noncanon.upload "out/$RUN_NAME" "$RUN_NAME" --repo "$HF_DATASET"; }

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run python -m noncanon.gpu_check
echo ">>> checkpoint $MODEL @ $REVISION -> $RUN_NAME"
for entry in $PROMPTS; do
  file="${entry%%:*}"
  n="${entry##*:}"; [ "$n" = "$entry" ] && n=1
  OUT="out/$RUN_NAME/$(basename "$file" .jsonl)"
  mkdir -p "$OUT"
  uv run python -m noncanon.generate --model "$MODEL" --revision "$REVISION" \
      --prompts "$file" --n "$n" --arms "$ARMS" --max-tokens "$MAX_TOKENS" --out-dir "$OUT"
  upload
done
for OUT in out/"$RUN_NAME"/*/; do
  uv run python -m noncanon.metrics --tokenizer "$MODEL" --revision "$REVISION" \
      --records "$OUT"/*.parquet --out-dir "$OUT/metrics" || echo ">>> metrics failed for $OUT (records are still uploaded)"
done
upload
