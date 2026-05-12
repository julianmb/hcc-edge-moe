# Heterogeneous Compute Cascade (HCC)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19562855.svg)](https://doi.org/10.5281/zenodo.19562855)

**HCC** is a Rust implementation of the *Heterogeneous Compute Cascade* architecture — a distributed inference engine for running 370B-parameter Mixture-of-Experts (MoE) language models across two AMD Ryzen APU nodes connected via USB4.

| Hardware | Theory | Speedup |
|---|---|---|
| 2× 16-Core Ryzen APU (256 GB unified memory) | NPU prefill + iGPU decode | 3× TTFT |
| 40 Gbps USB4 bridge (17 µs RTT tuned) | Speculative latency hiding | 2.35× decode |
| XDNA 2 NPU (50 TOPS) + RDNA 3.5 iGPU (212 GB/s) | 3-bit mixed-precision KV | 92% TCO vs DGX A100 |

## Paper Reference

This implementation is based on the architectural blueprint described in:

> Beltran, J. (2026). *Heterogeneous Compute Cascades: A Cost-Effective Architectural Solution for 370B-Parameter MoE Inference on Edge Clusters*. Zenodo. https://doi.org/10.5281/zenodo.19562855

## Improvements Over the Paper

During implementation, the following refinements were made based on community research and newer publications (March–April 2026):

| Paper Approach | This Implementation | Source |
|---|---|---|
| Uniform 3-bit KV cache | **Mixed-precision**: K 8-bit FP8 + V 3-bit Lloyd-Max | scos-lab/turboquant, arozanov/turboquant-mlx |
| Stop-and-wait speculative decoding | **PicoSpec async pipeline**: NPU drafts continuously (depth=3) | PicoSpec (arXiv 2603.19133) |
| Full 32K vocabulary over USB4 | **Sparse compressed verification** (<1 KB vs 128 KB) | PicoSpec separate rejection sampling |
| Unidirectional speculation | **Mirror-SD bidirectional**: forward drafts + correction paths | Mirror-SD (ICLR 2026) |

## Architecture

```
Node 1 (Ryzen APU)                          Node 2 (Ryzen APU)
┌─────────────────────┐       USB4 40Gbps      ┌─────────────────────┐
│  CPU Orchestrator   │ ◄──── 17µs RTT ──────► │  CPU Orchestrator   │
│  ┌───────────────┐  │       ┌─────────┐       │  ┌───────────────┐  │
│  │  NPU (XDNA 2) │  │       │ Zero-   │       │  │ iGPU (RDNA 3.5│  │
│  │  • 8B draft   │──┼──────►│ copy    │──────►│  │ MoE L39–77)  │  │
│  │  • Context    │  │       │ DMA-BUF │       │  │ • Verify γ    │  │
│  │    compressor │  │       │         │       │  │ • 3-bit KV    │  │
│  └───────────────┘  │       └─────────┘       │  └───────────────┘  │
│  ┌───────────────┐  │  ◄── correction ────┘  │                    │
│  │ iGPU (RDNA 3.5│  │       tokens            │                    │
│  │ MoE L0–38)    │  │                         │                    │
│  └───────────────┘  │                         │                    │
└─────────────────────┘                         └─────────────────────┘
```

The key insight: **physical isolation of draft and target memory domains**. By running the draft model on Node 1's NPU and the target model on Node 2's iGPU, each has its own 256-bit memory bus — avoiding the single-node memory contention that kills speculative decoding on unified memory (Section 5.2 of the paper).

## Mathematical Foundation

All core equations from the paper are implemented:

| Equation | Description | Location |
|---|---|---|
| (3) $T_{comm} = L + S/B + \lceil S/M \rceil \cdot O_{tcp}$ | USB4 transmission time | `src/interconnect/usb4.rs:86` |
| (5) $E[k] = (1-\alpha^{\gamma+1})/(1-\alpha)$ | Expected accepted tokens | `src/decoding/speculative.rs:29` |
| (6) $S = E[k] / (1 + \gamma \cdot c/C)$ | Speculative speedup | `src/decoding/speculative.rs:39` |
| (8) $\tilde{x} = (1/\sqrt{d}) \cdot H_d \cdot \operatorname{diag}(s) \cdot x$ | Walsh-Hadamard rotation | `src/kv_cache/walsh_hadamard.rs:21` |
| (9) $D_{MSE} \leq \sqrt{3\pi/2} \cdot 1/4^b$ | TurboQuant MSE bound | `src/kv_cache/turboquant.rs:167` |
| (10) $\tilde{x} = \hat{x}_{MSE} + \frac{\sqrt{\pi/2}}{d} \|r\|_2 \cdot S^T \cdot Q_{QJL}(r)$ | QJL residual correction | `src/kv_cache/qjl.rs:22` |

## Project Structure

```
src/
├── main.rs                      # CLI entry point (hcch binary)
├── config.rs                    # HccConfig with all paper § defaults
├── orchestrator.rs              # Main loop — Algorithm 1 + PicoSpec async
├── decoding/
│   ├── speculative.rs           # E[k], S, optimal γ* (Eq. 5-6)
│   ├── rejection.rs             # Rejection sampling (Leviathan et al.)
│   └── picospec.rs              # PicoSpec async pipeline + Mirror-SD
├── interconnect/
│   ├── usb4.rs                  # Eq. 3 transmission model + 17µs RTT
│   ├── dmabuf.rs                # Zero-copy DMA-BUF via memfd
│   └── protocol.rs              # Wire protocol messages
├── npu/
│   ├── draft_runner.rs          # 8B draft model on XDNA 2 (43.7 T/s)
│   └── context_compressor.rs    # CSR-style >80% compression (§6.2)
├── igpu/
│   ├── target_runner.rs         # MoE target on RDNA 3.5 (11.1 T/s)
│   └── migraphx.rs              # MIGraphX FFI bindings
├── kv_cache/
│   ├── mixed_precision.rs       # K 8-bit FP8 + V 3-bit Lloyd-Max
│   ├── turboquant.rs            # 3-bit quantizer + Lloyd-Max codebook
│   ├── walsh_hadamard.rs        # Fast Walsh-Hadamard transform
│   └── qjl.rs                   # QJL 1-bit residual correction
└── session/
    ├── session_manager.rs       # 9 concurrent sessions (§10.3)
    └── metrics.rs               # Prometheus telemetry
```

## Quick Start

### Prerequisites

- **Hardware**: 2× AMD Ryzen APU systems (Strix Halo recommended), 128 GB LPDDR5x-8000 each
- **Connectivity**: USB4 cable between nodes (40 Gbps, dual-link bonded = 45 Gbps)
- **Software**: Ubuntu 24.04.4 LTS, kernel 6.8+, ROCm 7.2.1, XRT + amdxdna driver

### Configuration

```toml
[cluster]
node_count = 2
node_id = 0
memory_per_node_gb = 128.0

[speculative]
draft_len = 5
acceptance_rate = 0.7
draft_cost_ratio = 0.05

[interconnect]
link_count = 2
throughput_gbps = 45.0
rtt_us = 17.0
kernel_tune = true
```

See `config.toml` for the full default configuration matching paper § parameters.

### Running

```bash
# Node 1 (NPU draft + layers 0-38)
hcch --config config.toml --role node1 --node-id 0

# Node 2 (iGPU layers 39-77)
hcch --config config.toml --role node2 --node-id 1
```

## Tests

```bash
cargo test        # 32 tests covering all core math and components
cargo check       # Zero errors (dead-code warnings expected for library)
```

All 32 tests validate the mathematical formulations from the paper, including expected accepted tokens, speedup multiplier, transmission time model, Walsh-Hadamard rotation, TurboQuant MSE bound, and QJL residual correction.

## Target Model

The primary target is **GLM-5.1-REAP-50** (~380B parameters, 128 experts, UD-Q3KM at ~161 GB):

- 40B active parameters per forward pass (K=8)
- 19.1 GB weight read per token
- Multi-head Latent Attention (MLA): 576 values/token/layer KV cache
- Fits dual-node 256 GB with 86 GB headroom for KV cache + OS

Also compatible with MiniMax-M2.5 (230B, 10B active, >40 T/s native without speculation).

## License

This project is provided as a reference implementation of the HCC architectural blueprint. See the paper for licensing details of the architecture itself. Third-party components (ROCm, MIGraphX, XRT, llama.cpp) are governed by their respective licenses.

## Citation

```bibtex
@software{beltran2026hcc,
  author = {Beltran, Julian},
  title = {Heterogeneous Compute Cascade (HCC)},
  year = {2026},
  doi = {10.5281/zenodo.19562855},
  url = {https://doi.org/10.5281/zenodo.19562855}
}
```
