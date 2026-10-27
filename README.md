# Freja Multi-GPU Llama Server

Production configuration, systemd services, automated test scripts, and benchmark documentation for **Freja** (`192.168.107.15`), our dedicated multi-GPU local LLM inference server powered by two NVIDIA GeForce RTX 3060 12GB cards (24 GB VRAM total) and 78 GB system RAM.

The stack is tuned specifically for the NVIDIA Ampere architecture, featuring quantified KV caching, MTP speculative decoding, layer splitting, and dual-service orchestration.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │              Freja (Ubuntu 24.04)            │
                      │       AMD Ryzen 5 5600X | 78 GB DDR4         │
                      └──────────────────────┬───────────────────────┘
                                             │
                      ┌──────────────────────┴───────────────────────┐
                      │         PCIe Host Bridge (PHB)               │
                      └──────────────┬────────────────┬──────────────┘
                                     │                │
                      ┌──────────────▼──────┐  ┌──────▼──────────────┐
                      │    GPU 0 (12 GB)    │  │    GPU 1 (12 GB)    │
                      │  RTX 3060 (Ampere)  │  │  RTX 3060 (Ampere)  │
                      └──────────────┬──────┘  └──────┬───────┬──────┘
                                     │                │       │
                                  53% layers       47% layers │
                                     │                │       │
                      ┌──────────────▼────────────────▼┐  ┌───▼──────────────┐
                      │     llama-server (:11434)      │  │llama-embed(:11435│
                      │  gemma-4-12b-it-Q6_K (Vision)  │  │ nomic-embed-text │
                      │  MTP Speculative Decoding Q4_0 │  │   768-dim memory │
                      │  32k context, 3 parallel slots │  │  4 parallel slots│
                      └────────────────────────────────┘  └──────────────────┘
```

### Hardware Specifications

| Component | Specification | Notes |
| :--- | :--- | :--- |
| **Host** | `Freja` (`192.168.107.15`) | Headless Ubuntu 24.04 LTS |
| **CPU** | AMD Ryzen 5 5600X (6c/12t @ 3.7–4.6 GHz) | PCIe 4.0 / Resizable BAR enabled |
| **RAM** | 78 GiB DDR4 | ~70 GiB available memory |
| **GPU 0** | NVIDIA GeForce RTX 3060 12GB | VRAM: 12,288 MiB, Driver 580.178.04 |
| **GPU 1** | NVIDIA GeForce RTX 3060 12GB | VRAM: 12,288 MiB, Driver 580.178.04 |
| **Total VRAM** | 24,576 MiB (~24 GB) | Unified dual-GPU inference |

Full hardware details and topology are documented in [docs/HARDWARE.md](docs/HARDWARE.md).

---

## Services & Ports

### 1. Primary LLM Server (`llama-server.service` on Port `11434`)
* **Base Model:** `gemma-4-12b-it-Q6_K.gguf` (9.2 GB, high-fidelity quantization)
* **Speculative Draft Head (MTP):** `mtp-gemma-4-12B-it-Q4_0.gguf` (242 MB)
* **Vision / Multimodal Projector:** `mmproj-gemma-4-12B-it-Q8_0.gguf` (152 MB)
* **Workload Distribution:** Layer split `--split-mode layer --tensor-split 53,47` across GPU 0 & 1
* **Context & Concurrency:** 32,768 context size (`-c 32768`), 3 parallel slots (`-np 3`) with unified KV cache
* **Acceleration:** Flash Attention (`-fa on`), `q4_0` KV cache, Jinja templating, live metrics

### 2. Embedding Server (`llama-embed.service` on Port `11435`)
* **Model:** `nomic-embed-text-v1.5.f16.gguf` (262 MB, 768 dimensions)
* **Placement:** Pinned strictly to GPU 1 (`CUDA_VISIBLE_DEVICES=1`)
* **Concurrency:** 8,192 context size, 4 parallel slots (`-np 4`), mean pooling
* **Role:** High-speed vector embeddings for semantic search, RAG pipelines, and HealthChat memory

---

## Performance Optimizations

Our production setup implements several carefully benchmarked optimizations:

1. **`q4_0` KV Cache Quantization:**
   * Replaced conventional `q8_0` KV caching with `q4_0`.
   * **Result:** +4.1% higher generation throughput, 572 MiB immediate VRAM reduction at idle, saving multiple gigabytes under deep 32k context load.
2. **Calibrated MTP Speculative Decoding:**
   * Tuned draft execution with `--spec-draft-n-max 3` and `--spec-draft-p-min 0.0`.
   * **Result:** Reduced per-token generation latency to 23.40 ms/token.
3. **Prompt Batching & Disconnect Protection (`-b 2048`):**
   * Configured batch slicing to guarantee that client disconnects or aborted requests release GPU slots in **~4 seconds** (compared to 70+ seconds under default batching), with negligible (<0.3%) prefill overhead.
4. **Optimized Monitoring:**
   * Health probes target `/health` at steady intervals to ensure prompt-cancellation threads are never stalled by slot-polling contention.

Detailed benchmark numbers and methodology are documented in [docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md).

---

## Benchmark Results

Evaluated on `Freja` using Gemma 4 12B IT (Q6_K) on a 115-token prompt generating 350 completion tokens:

| Metric | Baseline | Optimized Configuration | Gain |
| :--- | :--- | :--- | :--- |
| **Generation Throughput** | 41.06 tokens/s | **42.74 tokens/s** | **+4.1%** |
| **Per-Token Latency** | 24.35 ms | **23.40 ms** | **-3.9%** |
| **Response Time (350 tokens)** | 8.81 s | **8.33 s** | **~0.5 s faster** |
| **VRAM Footprint (Idle)** | 13,843 MiB | **13,271 MiB** | **572 MiB freed** |

---

## Directory Structure

```
.
├── README.md
├── configs/
│   ├── run_server.sh              # Main llama-server launch script
│   └── run_embed.sh               # Embedding server launch script
├── systemd/
│   ├── llama-server.service       # Systemd unit for LLM server
│   └── llama-embed.service        # Systemd unit for Embedding server
├── scripts/
│   ├── bench.py                   # Automated Python benchmark test
│   └── smoke_test.sh              # Endpoint & quality validation script
└── docs/
    ├── HARDWARE.md                # System topology and VRAM split
    └── OPTIMIZATIONS.md           # Benchmark data and tuning analysis
```

---

## Operational Commands

### Check Status & Telemetry
```bash
# Check service status
systemctl status llama-server llama-embed

# Live log stream
journalctl -u llama-server -f

# Check GPU utilization and temperatures
nvidia-smi
```

### Restart Services
```bash
sudo systemctl restart llama-server
sudo systemctl restart llama-embed
```

### Run Benchmarks & Health Tests
```bash
# Endpoint validation and quality smoke test
/opt/llama/smoke_test.sh

# Run benchmark test suite
python3 /opt/llama/scripts/bench.py
```

---

## API Endpoints

The server exposes OpenAI-compatible and Ollama-compatible HTTP endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `http://192.168.107.15:11434/v1/chat/completions` | POST | OpenAI chat completions (Gemma 4 12B IT) |
| `http://192.168.107.15:11434/v1/models` | GET | List active models |
| `http://192.168.107.15:11434/health` | GET | Server health check |
| `http://192.168.107.15:11434/metrics` | GET | Prometheus metrics |
| `http://192.168.107.15:11434/api/generate` | POST | Ollama generate endpoint |
| `http://192.168.107.15:11435/v1/embeddings` | POST | OpenAI embeddings endpoint (Nomic) |

### Example cURL Request

```bash
curl http://192.168.107.15:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:12b",
    "messages": [{"role": "user", "content": "Hej Freja, sammanfatta fördelarna med multi-GPU."}],
    "max_tokens": 200
  }'
```
