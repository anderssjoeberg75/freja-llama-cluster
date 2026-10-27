#!/usr/bin/env bash
# llama.cpp embedding server: nomic-embed-text v1.5 (same model as Ollama's nomic-embed-text,
# 768 dims) for HealthChat's coach memory. Small (~300 MB VRAM), pinned to GPU 1.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

exec /opt/llama/build/bin/llama-server \
  --model /opt/llama/models/nomic-embed-text/nomic-embed-text-v1.5.f16.gguf \
  --alias nomic-embed-text \
  --embedding --pooling mean \
  --host 0.0.0.0 --port 11435 \
  --n-gpu-layers 999 \
  --ctx-size 8192 --batch-size 2048 --ubatch-size 2048 \
  --parallel 4
