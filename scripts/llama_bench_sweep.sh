#!/usr/bin/env bash
# Offline sweep of the remaining tuning knobs with llama-bench.
# Needs the GPUs to be free:  sudo systemctl stop llama-server llama-embed
# Restart afterwards:          sudo systemctl start llama-server llama-embed
#
# Compares split modes (layer = pipeline, row = both GPUs read weights at the same
# time), micro-batch sizes and tensor-split ratios, using the same KV cache type as
# production. pp = prefill t/s, tg = decode t/s (without MTP speculation).
set -euo pipefail

BIN=${BIN:-/opt/llama/build/bin/llama-bench}
MODEL=${MODEL:-/opt/llama/models/gemma-4-12b-it/gemma-4-12b-it-Q6_K.gguf}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

if pgrep -x llama-server >/dev/null; then
  echo "llama-server is running - stop it first (sudo systemctl stop llama-server llama-embed)" >&2
  exit 1
fi

common=(-m "$MODEL" -ngl 999 -fa 1 -ctk q4_0 -ctv q4_0 -r 3 -o md)

echo "## 1. split mode x micro-batch (prefill 2048 + decode 128)"
"$BIN" "${common[@]}" -sm layer,row -b 2048 -ub 512,1024,2048 -p 2048 -n 128

echo "## 2. tensor-split ratio (layer mode)"
"$BIN" "${common[@]}" -sm layer -ts 53/47,50/50,56/44 -b 2048 -ub 1024 -p 512 -n 128

echo "## 3. decode at depth (long conversations), production settings"
"$BIN" "${common[@]}" -sm layer -ts 53/47 -b 2048 -ub 1024 -p 0 -n 128 -d 0,8192,16384
