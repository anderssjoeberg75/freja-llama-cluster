# Hardware & Topology — Freja LLM Server

Specifications and hardware topology for `Freja` (`192.168.107.15`), a dedicated multi-GPU inference machine.

## System Specs

| Component | Specification | Details |
| :--- | :--- | :--- |
| **Hostname** | `Freja` | Local IP: `192.168.107.15` |
| **OS** | Ubuntu 24.04.4 LTS | Kernel: `7.0.0-38-generic` x86_64 |
| **CPU** | AMD Ryzen 5 5600X | 6 Cores / 12 Threads @ 3.7 GHz (Boost ~4.6 GHz) |
| **RAM** | 78 GiB DDR4 | ~70 GiB available memory |
| **Storage** | 468 GB NVMe SSD | ~390 GB free |
| **GPU 0** | NVIDIA GeForce RTX 3060 12GB | Ampere (`sm_86`), 12288 MiB VRAM, PCIe Gen 4/3 x16 |
| **GPU 1** | NVIDIA GeForce RTX 3060 12GB | Ampere (`sm_86`), 12288 MiB VRAM, PCIe Gen 4/3 x16 |
| **Total VRAM** | 24,576 MiB (~24 GB) | Dual-card split setup |
| **Driver** | NVIDIA Driver 580.178.04 | CUDA Version: 13.0 API support |

## GPU Topology

Output from `nvidia-smi topo -m`:

```
        GPU0    GPU1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      PHB     0-11            0               N/A
GPU1    PHB      X      0-11            0               N/A

Legend:
  X   = Self
  PHB = Connection traversing PCIe as well as a PCIe Host Bridge (via Ryzen CPU)
```

Both RTX 3060 cards are connected via the CPU's PCIe Host Bridge (`PHB`).

## VRAM Allocation Strategy

The server runs two separate services:
1. **LLM Server (`llama-server.service`, port 11434):**
   - Model: `gemma-4-12b-it-Q6_K.gguf` (9.2 GB)
   - Draft Model: `mtp-gemma-4-12B-it-Q4_0.gguf` (242 MB)
   - Vision Projector: `mmproj-gemma-4-12B-it-Q8_0.gguf` (152 MB)
   - Split ratio: `--split-mode layer --tensor-split 53,47`
   - Allocated to: `CUDA_VISIBLE_DEVICES=0,1` (GPU 0 & GPU 1)
   - Idle footprint: ~6,549 MiB on GPU 0, ~6,266 MiB on GPU 1.

2. **Embedding Server (`llama-embed.service`, port 11435):**
   - Model: `nomic-embed-text-v1.5.f16.gguf` (262 MB)
   - Allocated to: `CUDA_VISIBLE_DEVICES=1` (pinned solely to GPU 1)
   - Idle footprint: ~456 MiB VRAM.

Total VRAM used while both services are running in idle:
- **GPU 0:** ~6,549 MiB / 12,288 MiB (~5,739 MiB free)
- **GPU 1:** ~6,722 MiB / 12,288 MiB (~5,566 MiB free)

Both cards remain cool (~34–50 °C) under normal operation with active VBIOS fan curves.
