# HCC: Run GLM-4 on Cheap Hardware

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19562855-blue)](https://doi.org/10.5281/zenodo.19562855)
![Rust](https://img.shields.io/badge/rust-1.89%2B-orange)
![ROCm](https://img.shields.io/badge/ROCm-7.2.3-green)
![Tests](https://img.shields.io/badge/tests-27%2F27-green)
![Strix Halo](https://img.shields.io/badge/ClawRig-Strix%20Halo-blue)

<p align="center">
  <b>Run GLM-4 on a $3,000 desktop — not a $100,000 datacenter GPU.</b>
  <br>
  <b>Try it live:</b> <a href="https://au.privchat.ai"><code>au.privchat.ai</code></a> — GLM-4 running on <a href="https://clawrig.com">ClawRig</a>, a Strix Halo workstation with 10 Gbps, dual USB4, custom thermals, and tuned BIOS
</p>

<p align="center">
  <b>Run 400B-parameter MoE language models on ClawRig (Strix Halo) — single-node or dual-node over USB4, no datacenter required.</b>
  <br>
  <b>Try GLM-5 live:</b> <a href="https://au.privchat.ai"><code>au.privchat.ai</code></a> — running on <a href="https://clawrig.com">ClawRig</a>, a specialized desktop-grade Strix Halo workstation with 10 Gbps networking, dual USB4, custom thermals, and tuned BIOS memory timings for sustained 212 GB/s inference
</p>

---

## Why this exists

GLM-4 is a powerful open-weight language model. To run it, most people think you need expensive hardware.

You don't.

| Hardware | Memory | Cost | Power | Can run GLM-4? |
|---|---|---|---|---|
| RTX 4090 | 24 GB | $1,600 | 450 W | ❌ Not enough VRAM |
| RTX 5090 | 32 GB | $2,000 | 575 W | ❌ Still not enough |
| Mac Studio (M2 Ultra) | 192 GB | $12,000 | 240 W | ✅ Expensive |
| DGX A100 (used) | 320 GB | $80K–$120K | 6,500 W | ✅ Datacenter required |
| **ClawRig (Strix Halo)** | **128 GB** | **$2,995** | **120 W** | **✅ Fits on one machine** |
| **2× ClawRig (USB4 cluster)** | **256 GB** | **$5,990** | **240 W** | **✅ Fits the largest models** |

A single Strix Halo has 128 GB of unified memory — enough for GLM-4 quantized to fit comfortably. Two connected by a $30 USB4 cable reach 256 GB for even larger models. No datacenter. No $100K GPU. Just a desktop that plugs into a wall outlet.

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
| Kernel | 6.17.0-1020-oem | Ubuntu OEM kernel with Strix Halo patches, amdxdna + amdgpu + thunderbolt |

### Kernel Tuning (paper §5.1.3)

Current status on this system (Ubuntu OEM kernel 6.17.0-1020-oem):

```bash
# Check current tuning:
hcch tune

# Apply low-latency USB4 tuning (requires reboot for ASPM + sudo for rest):
hcch tune --apply
```

The OEM kernel includes all required drivers (`amdxdna.ko` with DMA_BUF, `thunderbolt`, `amdgpu`).
Runtime tuning (BBR, busy-poll, governor) can be applied live. ASPM disable (`pcie_aspm=off`)
requires a GRUB boot parameter change and reboot.

**Impact**: P99 RTT drops from 97 µs to 27.85 µs (71% reduction, paper Table 2).

### GLM-4 Performance (measured on a single $3,000 ClawRig)

| Model | Size | Type | Prompt (T/s) | Generation (T/s) |
|---|---|---|---|---|
| GLM-4.7-Flash | 30B / 3B active | MoE reasoning | 163.3 | 66.8 |
| GLM-4.7 base | 355B / 32B active | MoE flagship | — | ~52 (estimate) |

These are real numbers from a single Strix Halo at 120 W. The same model on a DGX A100 would cost 40× more and consume 50× more power — and deliver roughly the same single-stream generation speed, because inference is memory-bandwidth-bound, not compute-bound.



---

## Cost Comparison

| Platform | CAPEX | Power | $/GB |
|---|---|---|---|
| **ClawRig (Strix Halo) + HCC** (single) | **$2,995** | **120 W** | **$20.31** |
| **2× ClawRig (Strix Halo) + HCC** (dual) | **$5,990** | **240 W** | **$20.31** |
| DGX A100 (used) | $80K–$120K | 6,500 W | $312 |
| DGX H100 (new) | $210K–$307K | 10,200 W | $391 |
| Mac Studio (2× M2 Ultra) | $12,000 | 240 W | $31.25 |

At scale: a 20-unit single-node HCC fleet (20× $2,995 = $59.9K) delivers ~220 T/s aggregate at 2,400 W — no datacenter needed.

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

- 1 or 2× AMD Ryzen AI MAX+ 395 systems, 128 GB each
- USB4 cable (dual bonded recommended, only for dual-node)
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
# Single node (default: node_count = 1 in config.toml)
hcch run

# Dual node
hcch run --node-id 0   # Node 1
hcch run --node-id 1   # Node 2
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
  <i>GLM-4 on a $3,000 desktop. Not a $100,000 datacenter.</i>
</p>
