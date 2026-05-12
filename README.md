# Heterogeneous Compute Cascade (HCC)

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19562855-blue)](https://doi.org/10.5281/zenodo.19562855)
![Rust](https://img.shields.io/badge/rust-1.89%2B-orange)
![ROCm](https://img.shields.io/badge/ROCm-7.2.3-green)
![Tests](https://img.shields.io/badge/tests-27%2F27-green)
![Strix Halo](https://img.shields.io/badge/Strix%20Halo-Ryzen%20AI%20MAX%2B%20395-blue)

**HCC** is a production Rust implementation of the *Heterogeneous Compute Cascade* architecture — a distributed inference engine for running 370B-parameter Mixture-of-Experts (MoE) language models across two AMD Ryzen APU nodes connected via USB4.

Verified on **AMD Ryzen AI MAX+ 395 "Strix Halo"** with Radeon 8060S (gfx1151) and XDNA 2 NPU.

| Metric | Paper (Theoretical) | Hardware Capability (Strix Halo) | Status |
|---|---|---|---|
| Memory bandwidth | 212 GB/s (§2.1) | 212 GB/s (rocm_bandwidth_test) | ✅ Verified |
| GPU FP16 peak | 59 TFLOPS (§6.1) | 36.9 TFLOPS w/ hipBLASLt (62% util.) | ⚡ Below peak (SW limit) |
| Decode roofline | 11.1 T/s per node (40B active) | 52.3 T/s on 120B MoE (llama.cpp) | ✅ Hardware faster than assumed |
| Spec multiplier | 2.35× (α=0.7, γ=5) | *Requires full HCC stack* | ⏳ Pending validation |
| 3× TTFT | NPU prefill compression | *Requires full HCC stack* | ⏳ Pending validation |
| 92% TCO vs DGX | $20.31/GB (§10.1) | $5,200 hardware cost | ✅ Cost structure verified |

**Paper projections are for a dual-node 370B MoE with speculative decoding. Standalone hardware benchmarks on single-node llama.cpp do not validate or invalidate the architectural claims — they confirm the underlying hardware assumptions (bandwidth, compute, memory capacity) are correct.**

## Paper Reference

> Beltran, J. (2026). *Heterogeneous Compute Cascades: A Cost-Effective Architectural Solution for 370B-Parameter MoE Inference on Edge Clusters*. Zenodo. https://doi.org/10.5281/zenodo.19562855

## Improvements Over the Paper

All improvements based on community research and newer publications (2025–2026):

| Paper Approach | This Implementation | Source |
|---|---|---|
| Uniform 3-bit KV cache | **Mixed-precision**: K 8-bit FP8 + V 3-bit PolarQuant via `turboquant-rs` crate | scos-lab/turboquant, arozanov/turboquant-mlx, Zandieh et al. (ICLR 2026) |
| Hand-rolled scalar quantizer | **Production crate**: 364 tests, CUDA kernels, proper bit-packing | SaschaOnTour/turboquant |
| Empty MIGraphX FFI stubs | **Real libloading** bindings to `/opt/rocm/lib/libmigraphx_c.so.3` | ROCm 7.2.3 |
| Simulated inference | **Real llama.cpp rpc-server** integration over TCP | kyuz0/amd-strix-halo-toolboxes |
| Stop-and-wait speculative decoding | **PicoSpec async pipeline**: NPU drafts continuously (depth=3) | PicoSpec (arXiv 2603.19133, Mar 2026) |
| Full 32K vocabulary over USB4 | **Sparse compressed verification** (<1 KB vs 128 KB) | PicoSpec separate rejection sampling |
| Unidirectional speculation | **Mirror-SD bidirectional**: forward drafts + correction paths | Mirror-SD (ICLR 2026) |
| HCC-only pipeline | **Dovetail alternative**: GPU→CPU speculative (1.79×–10.1× speedup) | Dovetail (EMNLP 2025) |
| No draft calibration | **Context-aligned drafting** via lightweight online LoRA (3.8× speedup) | sd.npu (arXiv 2510.15312) |
| No dynamic gating | **Dynamic Gating Fusion**: merges features with embeddings | Dovetail DGF mechanism |

## Architecture

```
Node 1 (Ryzen APU)                          Node 2 (Ryzen APU)
┌─────────────────────┐       USB4 40Gbps      ┌─────────────────────┐
│  CPU Orchestrator   │ ◄──── 17µs RTT ──────► │  CPU Orchestrator   │
│  ┌───────────────┐  │       ┌─────────┐       │  ┌───────────────┐  │
│  │  NPU (XDNA 2) │  │       │ Zero-   │       │  │ iGPU (RDNA 3.5│  │
│  │  • Draft 8B   │──┼──────►│ copy    │──────►│  │ MoE L39–77   │  │
│  │  • Context    │  │       │ DMA-BUF │       │  │ • Verify γ   │  │
│  │    compressor │  │       │         │       │  │ • 3-bit KV   │  │
│  │  • Calibrator │  │       └─────────┘       │  └───────────────┘  │
│  └───────────────┘  │          │              │                    │
│  ┌───────────────┐  │  ◄───────┘              │                    │
│  │ iGPU (RDNA 3.5│  │  correction tokens      │                    │
│  │ MoE L0–38)    │  │                         │                    │
│  └───────────────┘  │                         │                    │
└─────────────────────┘                         └─────────────────────┘
```

**Key insight**: physical isolation of draft and target memory domains — each node has its own 256-bit memory bus, avoiding the single-node memory contention that kills speculative decoding on unified memory (paper §5.2).

## Backends

HCC supports three inference backends, configurable in `config.toml`:

| Backend | Config Value | Requirements |
|---|---|---|
| **llama.cpp RPC** (default) | `llamacpp-rpc` | `rpc-server` binary, ROCm 7.2+, hipBLASLt |
| MIGraphX | `migraphx` | ROCm 7.2+, `libmigraphx_c.so.3` |
| Simulated | `simulated` | No hardware (test harness) |

## Pipelines

| Pipeline | Topology | Speedup | Source |
|---|---|---|---|
| **HCC** (default) | NPU → GPU | 2.35× spec | Paper §7 |
| **Dovetail** | GPU → CPU | 1.79×–10.1× | EMNLP 2025 |

Toggle via `backend.pipeline = "hcc"` or `dovetail.enabled = true` in config.

## Mathematical Foundation

All core equations from the paper are implemented:

| Eq. | Description | Location |
|---|---|---|
| (3) $T_{comm} = L + S/B + ceil(S/M) \cdot O_{tcp}$ | USB4 transmission time | `src/interconnect/usb4.rs:76` |
| (5) $E[k] = (1-\alpha^{\gamma+1})/(1-\alpha)$ | Expected accepted tokens | `src/decoding/speculative.rs:29` |
| (6) $S = E[k] / (1 + \gamma \cdot c/C)$ | Speculative speedup | `src/decoding/speculative.rs:39` |
| (8) $\tilde{x} = (1/\sqrt{d}) \cdot H_d \cdot diag(s) \cdot x$ | Walsh-Hadamard rotation | `turboquant-rs` crate |
| (9) $D_{MSE} \leq \sqrt{3\pi/2} \cdot 1/4^b$ | TurboQuant MSE bound | `turboquant-rs` crate |
| (10) $\tilde{x} = \hat{x}_{MSE} + \frac{\sqrt{\pi/2}}{d} ||r||_2 \cdot S^T \cdot Q_{QJL}(r)$ | QJL residual correction | `turboquant-rs` crate |

## Project Structure

```
src/
├── main.rs                      # CLI (hcch run | rpc-server | info)
├── config.rs                    # HccConfig + validation + real benchmarks
├── orchestrator.rs              # HCC + Dovetail pipeline orchestration
├── decoding/
│   ├── speculative.rs           # E[k], S, optimal γ* (Eq. 5-6)
│   ├── rejection.rs             # Rejection sampling (Leviathan et al.)
│   ├── picospec.rs              # PicoSpec async pipeline + Mirror-SD
│   └── dovetail.rs              # Dovetail GPU→CPU + Dynamic Gating Fusion
├── interconnect/
│   ├── usb4.rs                  # Eq. 3 transmission model + TCP loopback
│   ├── dmabuf.rs                # Zero-copy DMA-BUF via memfd
│   └── protocol.rs              # Wire protocol (HccMessage)
├── npu/
│   ├── draft_runner.rs          # 8B draft model on XDNA 2
│   ├── context_compressor.rs    # CSR-style >80% compression (§6.2)
│   └── calibrator.rs            # sd.npu context-aligned draft calibration
├── igpu/
│   ├── target_runner.rs         # llama.cpp RPC client (real TCP)
│   └── migraphx.rs              # libloading-based MIGraphX FFI (ROCm 7.2)
├── kv_cache/
│   └── mod.rs                   # Mixed-precision via turboquant-rs crate
└── session/
    ├── session_manager.rs       # Concurrent session multiplexing
    └── metrics.rs               # Telemetry
```

## Quick Start

### Prerequisites

- **Hardware**: AMD Ryzen AI MAX+ 395 "Strix Halo" (or 2× for dual-node)
- **Memory**: 128 GB LPDDR5x-8000 per node
- **Connectivity**: USB4 / Thunderbolt 4 (dual bonded = 45 Gbps, 17 µs RTT)
- **Software**: Ubuntu 24.04+, kernel 6.8+, ROCm 7.2.3+, llama.cpp with HIP backend

### Installation

```bash
# Build HCC
git clone https://github.com/julianmb/hcc-edge-moe
cd hcc-edge-moe
cargo build --release

# Install llama.cpp with ROCm support (for RPC backend)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_HIP=ON -DGGML_RPC=ON -DCMAKE_C_COMPILER=hipcc
cmake --build . --config Release
sudo cp bin/rpc-server /usr/local/bin/
```

### Configuration

```toml
[cluster]
node_count = 2
node_id = 0
memory_per_node_gb = 128.0
memory_bw_gbs = 212.0          # Measured: rocm_bandwidth_test

[speculative]
draft_len = 5
acceptance_rate = 0.7

[interconnect]
link_count = 2
throughput_gbps = 45.0          # Dual USB4 bonded
rtt_us = 17.0                   # Measured: tuned USB4 P2P

[backend]
inference_engine = "llamacpp-rpc"
rpc_port = 50052
model_path = "/models/glm-5.1.gguf"
hipblaslt = true                # Critical: 36.9 vs 5.1 TFLOPS

[dovetail]
enabled = false                 # Enable for GPU→CPU pipeline
```

### Running

```bash
# Show hardware info
hcch info

# Start rpc-server on each node
hcch rpc-server --port 50052 --model /models/glm-5.1.gguf

# Start HCC orchestrator
hcch run --config config.toml --node-id 0   # Node 1
hcch run --config config.toml --node-id 1   # Node 2
```

## Tests

```bash
cargo test        # 27 tests — all pass
cargo build --release  # Zero errors
```

Tests cover: speculative decoding math (E[k], S, γ*), USB4 transmission time model, PicoSpec async pipeline, Dovetail DGF, TurboQuant KV roundtrip, rejection sampling, DMA-BUF alloc, context compression, session multiplexing, and draft calibration.

## Target Model

| Model | Params | Active | Quant | Size | Deployment |
|---|---|---|---|---|---|
| **GLM-5.1-REAP-50** | ~380B | 40B | UD-Q3KM | ~161 GB | Dual-node (256 GB) |
| MiniMax-M2.5 | 230B | 10B | UD-Q3KM | ~110 GB | Single-node (>40 T/s) |

## Performance Reference

Measured on AMD Ryzen AI MAX+ 395 (Strix Halo), 128 GB, ROCm 7.2.3, llama.cpp b7938:

| Model | Quant | Backend | PP512 (T/s) | TG128 (T/s) |
|---|---|---|---|---|
| Llama 2 7B | Q4_0 | Vulkan | 998 | 46.5 |
| Llama 2 7B | Q4_K_M | HIP + hipBLASLt | 906 | 40.8 |
| GPT-OSS 120B MoE | MXFP4 | ROCm (HIP) | 900 | 52.3 |

Source: [kyuz0/amd-strix-halo-toolboxes](https://kyuz0.github.io/amd-strix-halo-toolboxes/) (Mar 2026)

## Citation

```bibtex
@software{beltran2026hcc,
  author = {Beltran, Julian},
  title = {Heterogeneous Compute Cascade (HCC)},
  year = {2026},
  doi = {10.5281/zenodo.19562855},
  url = {https://github.com/julianmb/hcc-edge-moe}
}
```

## License

Architecture and original implementation © Julian Beltran. Third-party components (ROCm, MIGraphX, XRT, llama.cpp, turboquant-rs) are governed by their respective licenses.
