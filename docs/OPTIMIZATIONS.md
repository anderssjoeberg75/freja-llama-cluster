# Performance Tuning & Empirical Optimizations — Freja

This document details our benchmarking methodology, performance sweeps, and verified optimizations developed for `Freja` (2x NVIDIA GeForce RTX 3060 12GB).

---

## 1. Key Engineering Findings

### A. KV Cache Quantization: `q4_0` vs `q8_0` & `f16`
During extensive multi-context evaluation, we analyzed the tradeoff between `f16`, `q8_0`, and `q4_0` KV cache implementations:
- **Throughput:** `q8_0` KV cache proved to be an inefficient middle ground—measuring ~8.7% slower than `q4_0` in decode throughput while consuming unnecessary memory.
- **Accuracy:** `q4_0` KV cache maintained a **98.0% identical top-token agreement** compared to full `f16`, with zero drift observed across deep contexts (up to 65k tokens).
- **VRAM Savings:** Switching from `q8_0` to `q4_0` immediately freed **572 MiB VRAM** at idle, scaling to multiple gigabytes of memory savings under full 32k context load across parallel slots.

### B. MTP (Multi-Token Prediction) Speculative Decoding Tuning
Speculative decoding via Gemma 4's MTP draft head was calibrated to achieve optimal draft acceptance:
- **Draft Head Configuration:** We paired the base model (`gemma-4-12b-it-Q6_K`) with its MTP draft companion (`mtp-gemma-4-12B-it-Q4_0`).
- **Tuning Sweet Spot:** By establishing `--spec-draft-n-max 3` combined with `--spec-draft-p-min 0.0`, we minimized speculation overhead and reduced per-token generation latency from 24.35 ms to **23.40 ms per token**.

### C. Dropped Request Protection (`-b 2048`)
In high-throughput server environments, clients may disconnect or abort requests during prompt prefill:
- **The Issue:** With large prompt batching (e.g. `-b 32768`), the server only checks for socket disconnects between batch iterations, causing GPUs to remain locked for up to 70–80 seconds on abandoned requests.
- **The Solution:** Capping `--batch-size 2048` guarantees the server releases aborted slots within **~4 seconds**, with less than 0.3% impact on prefill throughput.

### D. Slot Monitoring & Health Checks
- High-frequency polling on `/slots` resets internal condition-variable waits, preventing prompt cancellation signals from propagating timely.
- Production monitors should poll `/health` at moderate intervals (e.g., 5 seconds) rather than spamming `/slots`.

---

## 2. Empirical Benchmark Results

Measured on `Freja` under identical conditions using Gemma 4 12B IT (Q6_K) with a 115-token prompt and 350-token generation:

| Metric | Baseline (`q8_0` KV, uncalibrated MTP) | Optimized (`q4_0` KV, MTP n3/p0.0) | Delta |
| :--- | :--- | :--- | :--- |
| **Generation speed (eval)** | 41.06 tokens/s | **42.74 tokens/s** | **+4.1%** |
| **Per-token eval latency** | 24.35 ms/token | **23.40 ms/token** | **-3.9%** |
| **Total wall time (350 tokens)** | 8.81 s | **8.33 s** | **-0.48 s (-5.5%)** |
| **GPU 0 VRAM (idle)** | 6,899 MiB | **6,549 MiB** | **-350 MiB** |
| **GPU 1 VRAM (idle)** | 6,944 MiB | **6,722 MiB** | **-222 MiB** |
| **Total VRAM Freed** | — | — | **572 MiB freed** |

---

## 3. Dual GPU Workload Distribution

To support both interactive text generation and real-time semantic memory (vector embeddings), the workload is split cleanly:

```
GPU 0 (12 GB):  [ 53% Gemma 4 12B Layers + KV Cache ]  (~6.5 GB used)
GPU 1 (12 GB):  [ 47% Gemma 4 12B Layers + KV Cache ]  (~6.2 GB used)
                [ Nomic Embed Text v1.5 Model Buffer ] (~0.5 GB used)
```

This arrangement guarantees that both cards operate with balanced thermal profiles and headroom for peak context expansion.

---

## 4. Further Optimization Roadmap (not yet benchmarked)

Analysis of where the remaining time goes, and which knobs are worth measuring next.
Nothing in this section has been applied to the production flags yet; each item should
be verified with `scripts/bench.py` / `scripts/llama_bench_sweep.sh` before it is adopted.

### Where the time goes

Decode is memory-bandwidth bound. An RTX 3060 has ~360 GB/s, so reading the 9.2 GB Q6_K
weights once costs at least ~25.6 ms, i.e. a hard ceiling of ~39 t/s per card without
speculation. The measured 42.7 t/s (23.4 ms/token) is only possible because MTP accepts
several draft tokens per weight pass — speculation already delivers the bulk of the gain.

With `--split-mode layer` the two GPUs work **one after the other** for every token: GPU 0
runs its 53% of the layers, hands the hidden state to GPU 1, then idles. The split gives
capacity (the model + KV + mmproj does not fit in 12 GB) but no extra bandwidth for a
single request. That is the largest untapped resource in the system.

### Candidates, ordered by expected impact

| # | Change | Expected effect | Risk / cost |
|---|--------|-----------------|-------------|
| 1 | `--split-mode row` (or tensor-parallel mode if the build has one) | Both GPUs read weights at the same time → up to ~1.3–1.6× decode for one request. Depends heavily on PCIe: run `scripts/gpu_check.sh` first — if GPU 1 sits in a chipset x4 slot, row split will likely be **slower**. | Pure config change, easy to revert. Row split cannot use MTP draft on all builds — measure end-to-end with `bench.py`. |
| 2 | Smaller weight quant (`Q5_K_M` ≈ 8.4 GB, `Q4_K_M` ≈ 7.3 GB) | Decode scales ~linearly with bytes read: ≈ +9% / +25%. | Quality loss, must be checked against Swedish health answers (same top-token agreement method as the KV-cache test). |
| 3 | `--ubatch-size 2048` (= batch size) | Higher prefill t/s for long prompts / RAG context → lower TTFT. | Bigger compute buffer (a few hundred MiB per GPU, there is ~5.5 GB free). Abort latency is governed by `-b`, which stays 2048. |
| 4 | `--cache-ram 16384` (host-RAM prompt cache, default 8192 MiB) | More conversations keep their prefix cached when slots are reused → lower TTFT for returning HealthChat users. 70 GB RAM is idle. | Only RAM. Check `--cache-ram` exists in the deployed build (`llama-server --help`). |
| 5 | Lock memory clock / power limit (`nvidia-smi -lmc`, `-pl`) | Prevents clock dips between requests; decode is memory-clock bound. | Needs root + a oneshot unit; verify with `gpu_check.sh` that clocks actually drop today before bothering. |
| 6 | Re-tune `--tensor-split` after any of the above | Balances compute so neither GPU waits; the sweep tests 50/50, 53/47 and 56/44. | None. |

### Already-optimal / low value

* **More parallel slots:** `--parallel 3` with a unified 32k KV cache is a good fit; more slots
  dilute MTP gains and per-request speed.
* **`--threads`:** with full GPU offload the CPU only runs sampling/HTTP; tuning threads has
  no measurable effect on this box.
* **Embedding server:** ~300 MB, sub-millisecond per short text; it is not a bottleneck.

### How to measure

```bash
# GPU link, clocks, throttle reasons (run once idle, once while generating)
/opt/llama/scripts/gpu_check.sh

# End-to-end, production server: TTFT, prefill, decode, MTP acceptance (median of 5)
python3 /opt/llama/scripts/bench.py
python3 /opt/llama/scripts/bench.py --concurrency 3        # all three slots busy

# Offline sweep of split mode / ubatch / tensor-split (stops nothing by itself)
sudo systemctl stop llama-server llama-embed
/opt/llama/scripts/llama_bench_sweep.sh | tee sweep-$(date +%F).md
sudo systemctl start llama-server llama-embed
```

`bench.py` disables server prompt caching by default (`--cache-prompt` to re-enable), uses a
fixed seed, and reports server-side `timings`, so numbers are comparable between config
changes. Note that the historical numbers in section 2 were wall-clock based (prefill
included), so they are slightly lower than the `decode speed` column of the new script.
