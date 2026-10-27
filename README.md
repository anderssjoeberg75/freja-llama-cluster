# Freja Multi-GPU Llama Server

Configuration, systemd services, scripts, and benchmark notes for **Freja** (`192.168.107.15`), a dedicated multi-GPU local LLM inference server equipped with two NVIDIA GeForce RTX 3060 12GB cards (24 GB VRAM total) and 78 GB system RAM.

The configuration incorporates empirical optimizations inspired by the [4claps/houston-p100-cluster](https://github.com/4claps/houston-p100-cluster) project, tuned specifically for NVIDIA Ampere architecture.

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
| **RAM** | 78 GiB DDR4 | ~70 GiB free |
| **GPU 0** | NVIDIA GeForce RTX 3060 12GB | VRAM: 12,288 MiB, Driver 580.178.04 |
| **GPU 1** | NVIDIA GeForce RTX 3060 12GB | VRAM: 12,288 MiB, Driver 580.178.04 |
| **Total VRAM** | 24,576 MiB (~24 GB) | Unified across 2 devices |

Full details can be found in [docs/HARDWARE.md](docs/HARDWARE.md).

---

## Services & Ports

### 1. Main LLM Server (`llama-server.service` on Port `11434`)
* **Model:** `gemma-4-12b-it-Q6_K.gguf` (9.2 GB)
* **Speculative Draft (MTP):** `mtp-gemma-4-12B-it-Q4_0.gguf` (242 MB)
* **Vision / Multimodal Projector:** `mmproj-gemma-4-12B-it-Q8_0.gguf` (152 MB)
* **GPU Split:** Layer split `--split-mode layer --tensor-split 53,47` across GPU 0 & 1
* **Context & Parallelism:** 32,768 context size (`-c 32768`), 3 parallel slots (`-np 3`) with unified KV cache
* **Features:** Flash Attention (`-fa on`), `q4_0` KV cache, Jinja templating, metrics enabled

### 2. Embedding Server (`llama-embed.service` on Port `11435`)
* **Model:** `nomic-embed-text-v1.5.f16.gguf` (262 MB, 768 dimensions)
* **Placement:** Pinned strictly to GPU 1 (`CUDA_VISIBLE_DEVICES=1`)
* **Parallelism:** 8,192 context size, 4 parallel slots (`-np 4`), mean pooling
* **Role:** High-speed vector embeddings for semantic search and HealthChat memory

---

## Applied Optimizations (from Houston P100 Cluster)

Based on findings in the [4claps/houston-p100-cluster](https://github.com/4claps/houston-p100-cluster) repository, the following optimisations have been implemented:

1. **`q4_0` KV Cache Quantization:**
   * Changed from `--cache-type-k q8_0 --cache-type-v q8_0` to `q4_0`.
   * **Gain:** +4.1% higher generation throughput, -572 MiB idle VRAM footprint (saves multiple gigabytes under 32k context).
2. **MTP Speculative Decoding Tuning:**
   * Configured `--spec-draft-n-max 3` and `--spec-draft-p-min 0.0`.
   * **Gain:** Decreased per-token latency to 23.40 ms/token.
3. **Dropped Request Protection (`-b 2048`):**
   * Ensures aborted client connections release the slot within ~4 seconds instead of 70+ seconds.
4. **Monitoring Safe Practices:**
   * Health checks poll `/health` rather than high-frequency polling on `/slots` to prevent condition-variable stalls.

Detailed notes and benchmark tables are in [docs/HOUSTON-OPTIMIZATIONS.md](docs/HOUSTON-OPTIMIZATIONS.md).

---

## Directory Structure

```
.
├── README.md
├── configs/
│   ├── run_server.sh              # Main llama-server startup script
│   └── run_embed.sh               # Nomic embedding server startup script
├── systemd/
│   ├── llama-server.service       # Systemd unit for LLM server
│   └── llama-embed.service        # Systemd unit for Embedding server
├── scripts/
│   ├── bench.py                   # Python benchmark test tool
│   └── smoke_test.sh              # Health & endpoint validation script
└── docs/
    ├── HARDWARE.md                # System topology and VRAM allocation
    └── HOUSTON-OPTIMIZATIONS.md   # Benchmark analysis and optimization rationale
```

---

## Operational Commands

### Check Status & Logs
```bash
# Check service status
systemctl status llama-server llama-embed

# Stream live server logs
journalctl -u llama-server -f

# Check GPU memory and power usage
nvidia-smi
```

### Restart Services
```bash
sudo systemctl restart llama-server
sudo systemctl restart llama-embed
```

### Run Tests
```bash
# Quick health check & endpoint validation
/opt/llama/smoke_test.sh

# Run benchmark test
python3 /opt/llama/scripts/bench.py
```

---

## API Endpoints

The server exposes both standard OpenAI-compatible endpoints and Ollama-compatible endpoints:

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `http://192.168.107.15:11434/v1/chat/completions` | POST | OpenAI chat completions (Gemma 4 12B IT) |
| `http://192.168.107.15:11434/v1/models` | GET | List loaded models |
| `http://192.168.107.15:11434/health` | GET | Server health check |
| `http://192.168.107.15:11434/metrics` | GET | Prometheus metrics |
| `http://192.168.107.15:11434/api/generate` | POST | Ollama generate format |
| `http://192.168.107.15:11435/v1/embeddings` | POST | OpenAI embeddings format (Nomic) |

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
