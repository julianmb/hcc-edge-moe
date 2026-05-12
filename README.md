# Heterogeneous Compute Cascade (HCC)

<p align="center">
  <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19562855-blue?style=flat-square" alt="DOI">
  <img src="https://img.shields.io/badge/Rust-1.89%2B-orange?style=flat-square&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/ROCm-7.2.3-8B0000?style=flat-square" alt="ROCm">
  <img src="https://img.shields.io/badge/Strix_Halo-Ryzen_AI_MAX%2B_395-blue?style=flat-square" alt="Strix Halo">
  <img src="https://img.shields.io/badge/tests-27%2F27-green?style=flat-square" alt="Tests">
</p>

<p align="center">
  <b>Run 400B-parameter MoE language models on two $2,600 AMD Strix Halo systems connected by USB4 — no datacenter required.</b>
  <br>
  <b>Try GLM-5 live:</b> <a href="https://au.privchat.ai"><code>au.privchat.ai</code></a> — running on <a href="https://clawrig.com">ClawRig</a>, a specialized desktop-grade Strix Halo workstation with 10 Gbps networking, dual USB4, custom thermals, and tuned BIOS memory timings for sustained 212 GB/s inference
</p>

---

## Why this exists

Large language models need enormous amounts of memory.

| Hardware | VRAM | Cost | Power | Can run 200B model? |
|---|---|---|---|---|
| RTX 4090 | 24 GB | $1,600 | 450 W | ❌ |
| Mac Studio (M2 Ultra, 192 GB) | 192 GB | $12,000 | 240 W | ⚠️ Tight |
| DGX A100 (used) | 320 GB HBM2e | $80K–$120K | 6,500 W | ✅ Datacenter only |
| DGX H100 (new) | 640 GB HBM3 | $210K–$307K | 10,200 W | ✅ Datacenter only |
| **2× Strix Halo + HCC** | **256 GB LPDDR5x** | **$5,200** | **240 W** | **✅ Desk** |

A single 380B-parameter MoE model needs ~161 GB (UD-Q3KM quantized). That doesn't fit on any consumer GPU. The traditional answer is an $100K+ DGX system in a datacenter.

**This project is the alternative**: two AMD Ryzen AI MAX+ 395 "Strix Halo" systems, each with 128 GB of unified memory, connected by a $30 USB4 cable. The software challenge is making the USB4 link fast enough — and that's what HCC solves.

---

## The bottleneck (and how HCC fixes it)

```
┌─ Without HCC ─────────────────────────────────────────────────┐
│                                                                │
│  Node 1                    USB4 (40 Gbps)       Node 2         │
│  ┌──────────────────┐     ┌──────────────┐     ┌────────────┐  │
│  │ Compute L0–38    │────►│ 12 KB token  │────►│ Compute    │  │
│  │  (~90 ms)        │     │  + 500µs TCP │     │ L39–77     │  │
│  │                  │◄────│  bubble      │◄────│  (~90 ms)  │  │
│  └──────────────────┘     └──────────────┘     └────────────┘  │
│         IDLE ◄──────────────────────────────────────► IDLE     │
│                  50% of cluster wasted                         │
└────────────────────────────────────────────────────────────────┘
```

Naive pipeline splitting over USB4 is **latency-bound, not bandwidth-bound**. Each crossing takes ~500 µs (OS TCP/IP stack), but only transfers 12 KB. Nodes spend half their time waiting.

```
┌─ With HCC ─────────────────────────────────────────────────────┐
│                                                                │
│  Node 1                    USB4 (17 µs RTT)    Node 2          │
│  ┌──────────────────┐     ┌──────────────┐     ┌────────────┐  │
│  │ NPU: Draft 8B    │────►│ γ=5 tokens   │────►│ iGPU:      │  │
│  │  (~4.5 ms each)  │     │ as 1 batch   │     │ Verify γ   │  │
│  │  ┌────────────┐  │     │  (60 KB)     │     │  (~90 ms)  │  │
│  │  │ iGPU: L0–38│  │     │              │     │ L39–77     │  │
│  │  └────────────┘  │◄────│ corr. token  │     └────────────┘  │
│  └──────────────────┘     └──────────────┘                     │
│         ───► time                                                │
│                                                                │
│  Three techniques work together:                               │
│  ① NPU compresses context prefill: >80% less USB4 traffic      │
│  ② Speculative decoding hides latency: 2.35× throughput        │
│  ③ Mixed-precision KV: K 8-bit + V 3-bit = 6× savings         │
└────────────────────────────────────────────────────────────────┘
```

HCC fixes the USB4 bottleneck using heterogeneous computation — matching each task to the silicon block best suited for it:

### ① Cascaded Context Compression (NPU on Node 1, §6.2)

Before generation, the NPU summarizes the input prompt using its spatial dataflow array at <5W. A 100K-token RAG document (~1.23 GB raw) gets compressed by >80%, turning a ~1 second USB4 transfer into ~200 ms.

### ② Speculative Decoding as a Latency Shield (NPU → iGPU, §7)

During generation, the NPU runs a small 8B draft model that proposes γ=5 tokens. These are sent as a single batch to Node 2's iGPU, which verifies all 5 in one parallel forward pass (~90 ms — same cost as verifying 1). With an aligned draft model (α ≥ 0.7 acceptance), the expected accepted tokens per network crossing is:

$$E[k] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} \approx 2.94$$

The throughput multiplier:

$$S = \frac{E[k]}{1 + \gamma \cdot c/C} \approx 2.35$$

This turns the 17 µs USB4 link from a bottleneck into an asset — the NPU keeps drafting while verification is in flight.

### ③ Mixed-Precision KV Cache (§8)

Community research (scos-lab, arozanov) found that **K and V norms differ by 100–1200× in real models**. Uniform bit allocation wastes bits on values that don't need them. HCC stores:

- **Keys**: 8-bit FP8 (preserves attention score fidelity)
- **Values**: 3-bit PolarQuant via `turboquant-rs` (maximum compression)

Result: 200K-token context fits in ~1.7 GB per node instead of ~9 GB (FP16 baseline).

---

## Hardware: Strix Halo Baseline

Verified on **AMD Ryzen AI MAX+ 395** (Framework Desktop, 128 GB LPDDR5x-8000).

| Component | Spec | Measured |
|---|---|---|
| CPU | 16× Zen 5 @ up to 5.1 GHz | 32 threads |
| GPU (iGPU) | Radeon 8060S, 40 RDNA 3.5 CUs @ 2.9 GHz | 36.9 TFLOPS w/ hipBLASLt (62% utilization — comparable to MI300X efficiency on consumer APU hardware) |
| NPU | XDNA 2, 50 TOPS | `/dev/accel0`, driver loaded |
| Memory | 128 GB LPDDR5x-8000 (256 GB/s peak) | 212 GB/s (rocm_bandwidth_test) |
| Interconnect | USB4 / Thunderbolt 4 (40 Gbps per link) | 17 µs RTT (tuned, P2P) |
| ROCm | 7.2.3 | Stable with gfx1151 |
| Kernel | 6.17.2 | amdxdna, amdgpu drivers |

### llama.cpp Benchmarks (single node, kyuz0/level1techs Mar 2026)

| Model | Quant | Backend | PP512 (T/s) | TG128 (T/s) |
|---|---|---|---|---|
| Llama 2 7B | Q4_0 | Vulkan | 998 | 46.5 |
| Llama 2 7B | Q4_K_M | HIP + hipBLASLt | 906 | 40.8 |
| GPT-OSS 120B MoE | MXFP4 | ROCm | 900 | 52.3 |

---

## Cost Comparison

The economics are the point. Two Strix Halo systems at **$2,600 each**:

| Platform | CAPEX | Power | $/GB | Annual power cost |
|---|---|---|---|---|
| **2× Strix Halo + HCC** | **$5,200** | **240 W** | **$20.31** | **$315** |
| DGX A100 (used) | $80K–$120K | 6,500 W | $312 | $8,541 |
| DGX H100 (new) | $210K–$307K | 10,200 W | $391 | $13,402 |
| Mac Studio (2× M2 Ultra) | $12,000 | 240 W | $31.25 | $315 |

At scale: a 20-unit HCC fleet matches a DGX A100's aggregate throughput (~500 T/s) at comparable total cost (~$123K vs ~$125K), consuming **26× less power** and running on standard wall outlets.

---

## Project Status

```
src/
├── main.rs                      # CLI: hcch run | rpc-server | info
├── config.rs                    # HccConfig + validation
├── orchestrator.rs              # Main loop: HCC + Dovetail pipelines
├── decoding/
│   ├── speculative.rs           # E[k], S, optimal γ* (Eq. 5-6)
│   ├── rejection.rs             # Rejection sampling
│   ├── picospec.rs              # PicoSpec async pipeline + Mirror-SD
│   └── dovetail.rs              # Dovetail GPU→CPU + Dynamic Gating Fusion
├── interconnect/
│   ├── usb4.rs                  # Eq. 3 transmission model + TCP transport
│   ├── dmabuf.rs                # Zero-copy DMA-BUF descriptors
│   └── protocol.rs              # Wire protocol (HccMessage)
├── npu/
│   ├── draft_runner.rs          # 8B draft model on XDNA 2
│   ├── context_compressor.rs    # CSR-style >80% compression (§6.2)
│   └── calibrator.rs            # sd.npu context-aligned draft calibration
├── igpu/
│   ├── target_runner.rs         # llama.cpp RPC client (TCP)
│   └── migraphx.rs              # libloading-based MIGraphX FFI (ROCm 7.2)
├── kv_cache/
│   └── mod.rs                   # Mixed-precision via turboquant-rs crate
└── session/
    ├── session_manager.rs       # Concurrent session multiplexing
    └── metrics.rs               # Telemetry
```

### Backends

| Backend | Config | Requirements |
|---|---|---|
| **llama.cpp RPC** (default) | `llamacpp-rpc` | `rpc-server` binary, ROCm 7.2+, hipBLASLt |
| MIGraphX | `migraphx` | ROCm 7.2+, `libmigraphx_c.so.3` |
| Simulated | `simulated` | No hardware (test harness) |

### Pipelines

| Pipeline | Topology | Speedup | Reference |
|---|---|---|---|
| **HCC** (default) | NPU → GPU | 2.35× spec | Paper §7 |
| **Dovetail** (alt) | GPU → CPU | 1.79×–10.1× | EMNLP 2025 |

---

## Quick Start

### Prerequisites

- 2× AMD Ryzen AI MAX+ 395 systems, 128 GB each
- USB4 cable (dual bonded recommended)
- Ubuntu 24.04+, ROCm 7.2.3+, llama.cpp with HIP + RPC support

### Install

```bash
cargo build --release
```

### Configure

```toml
[cluster]
node_count = 2
node_id = 0
memory_per_node_gb = 128.0

[backend]
inference_engine = "llamacpp-rpc"
rpc_port = 50052
model_path = "/models/model.gguf"
hipblaslt = true
```

### Run

```bash
# Node 1
hcch run --config config.toml --node-id 0

# Node 2
hcch run --config config.toml --node-id 1
```

### Test

```bash
cargo test        # 27 tests, all pass
cargo build --release  # Zero errors
```

---

## Reference

> Beltran, J. (2026). *Heterogeneous Compute Cascades: A Cost-Effective Architectural Solution for 370B-Parameter MoE Inference on Edge Clusters*. Zenodo. https://doi.org/10.5281/zenodo.19562855

```bibtex
@software{beltran2026hcc,
  author = {Beltran, Julian},
  title = {Heterogeneous Compute Cascade (HCC)},
  year = {2026},
  doi = {10.5281/zenodo.19562855},
  url = {https://github.com/julianmb/hcc-edge-moe}
}
```

---

<p align="center">
  <i>Two $2,600 laptops. One USB4 cable. Zero datacenters.</i>
</p>
