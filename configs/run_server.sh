#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=/opt/llama/models/gemma-4-12b-it
export CUDA_VISIBLE_DEVICES=0,1

exec /opt/llama/build/bin/llama-server \
  --model        "$MODEL_DIR/gemma-4-12b-it-Q6_K.gguf" \
  --model-draft  "$MODEL_DIR/mtp-gemma-4-12B-it-Q4_0.gguf" \
  --mmproj       "$MODEL_DIR/mmproj-gemma-4-12B-it-Q8_0.gguf" \
  --spec-type    draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.0 \
  --alias        gemma4:12b \
  --host 0.0.0.0 --port 11434 \
  --n-gpu-layers 999 --gpu-layers-draft 999 \
  --split-mode layer \
  --tensor-split 53,47 \
  --load-mode none \
  --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --batch-size 2048 --ubatch-size 1024 \
  --ctx-size 32768 --parallel 3 \
  --kv-unified \
  --no-webui \
  --jinja \
  --reasoning off \
  --metrics
