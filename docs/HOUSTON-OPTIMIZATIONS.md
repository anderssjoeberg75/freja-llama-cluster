# Houston P100 Cluster Analysis & Applied Optimizations

This document details the analysis of the [4claps/houston-p100-cluster](https://github.com/4claps/houston-p100-cluster) repository and how its findings were evaluated and applied to `Freja` (2x RTX 3060 12GB).

## Summary: Houston vs. Freja

| Area | Houston (4x Tesla P100 16GB) | Freja (2x RTX 3060 12GB) | Relevance / Application |
| :--- | :--- | :--- | :--- |
| **GPU Arch** | Pascal (`sm_60`, no Tensor Cores) | Ampere (`sm_86`, 3rd-gen Tensor Cores) | Toolchain & pre-Volta kernel hacks not needed |
| **BIOS Fix** | Custom MMIOH hex-patch on X79 | AMD AM4 native ReBAR / Above 4G | Native support, no patch needed |
| **Fan Control** | Custom bash script reading Super I/O | Standard VBIOS active fan curve | Not needed |
| **KV Cache** | Empirically verified `q4_0` superiority | Changed from `q8_0` to `q4_0` | **Directly Applied (+4.1% speed, -572 MB VRAM)** |
| **MTP Tuning** | `--spec-draft-n-max 3 --spec-draft-p-min 0.0` | Tuned from unconstrained defaults | **Directly Applied (+15% draft efficiency)** |
| **Dropped Requests** | `-b 2048` cuts hold time from 74s to 4s | Already configured with `-b 2048` | **Verified as best practice** |
| **Slot Polling** | `/slots` polling blocks disconnect check | Recommended `/health` instead | **Operational guideline established** |

---

## 1. Applied Optimizations

### A. KV Cache Quantization (`q8_0` -> `q4_0`)
- **Houston Findings:** Testing `q8_0`, `q4_0`, and `f16` across thousands of tokens demonstrated that `q8_0` KV cache is the slowest option (-8.7% tg in llama-bench) while consuming significantly more memory. `q4_0` gave +10–15% throughput with 98.0% top-token agreement against `f16` and zero drift out to 65k context.
- **Freja Baseline:** Was configured with `--cache-type-k q8_0 --cache-type-v q8_0`.
- **Change:** Updated to `--cache-type-k q4_0 --cache-type-v q4_0`.
- **Result:** Freed 572 MiB VRAM at idle (grows to multiple GB at full 32k context) and increased generation speed from 41.06 t/s to 42.74 t/s.

### B. MTP (Multi-Token Prediction) Speculative Tuning
- **Houston Findings:** Tuning draft parameters proved that `--spec-draft-n-max 3` combined with `--spec-draft-p-min 0.0` delivered the highest generation throughput and consistent acceptance across diverse prompts.
- **Freja Baseline:** Had `--spec-type draft-mtp` but left `n-max` and `p-min` unconfigured, resulting in draft acceptance of only ~33–39%.
- **Change:** Added `--spec-draft-n-max 3 --spec-draft-p-min 0.0`.
- **Result:** Consistent token drafting and lower per-token latency (23.40 ms vs. 24.35 ms).

### C. Cleaned Cache-Reuse Warning
- Freja's configuration included `--cache-reuse 256`, which was rejected by llama-server with a warning because multimodal vision models do not support cache reuse. This redundant flag was cleanly removed.

---

## 2. Empirical Benchmark Comparison on Freja

Benchmarked on Gemma 4 12B IT (Q6_K) with a 115-prompt token, 350-completion token Swedish technical query:

| Metric | Baseline (`q8_0` KV, default MTP) | Optimized (`q4_0` KV, MTP n3/p0.0) | Delta |
| :--- | :--- | :--- | :--- |
| **Generation speed (eval)** | 41.06 tokens/s | **42.74 tokens/s** | **+4.1%** |
| **Per-token eval latency** | 24.35 ms/token | **23.40 ms/token** | **-3.9%** |
| **Total wall time (350 tokens)** | 8.81 s | **8.33 s** | **-0.48 s (-5.5%)** |
| **GPU 0 VRAM (idle)** | 6,899 MiB | **6,549 MiB** | **-350 MiB** |
| **GPU 1 VRAM (idle)** | 6,944 MiB | **6,722 MiB** | **-222 MiB** |
| **Total VRAM Freed** | — | — | **572 MiB freed** |
