# Heterogeneous Compute Cascade (HCC)

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19562855-blue)](https://doi.org/10.5281/zenodo.19562855)
![Rust](https://img.shields.io/badge/rust-1.89%2B-orange)
![ROCm](https://img.shields.io/badge/ROCm-7.2.3-green)
![Tests](https://img.shields.io/badge/tests-passing-green)

**A Rust feasibility and orchestration project for running oversized MoE models across two ClawRig workstations. The reference workload is GLM-5.3-Flash (320B total / 18B active parameters, 45 layers, 288 routed experts with top-8 routing, ROCmFP4 quantization).**

## Purpose

HCC investigates one narrow systems question:

> Can two 128 GB unified-memory workstations make a roughly 158 GB frontier MoE checkpoint practical without a datacenter interconnect?

The project is not a replacement for llama.cpp, vLLM, or exo. It uses llama.cpp for real local inference and develops the missing experimental pieces around it: model placement across two nodes, USB4 transport, heterogeneous CPU/iGPU/NPU scheduling, speculative batching, and explicit measurement.

## GLM-5.3-Flash Test Profile

`configs/glm53-flash.toml` describes the reference capacity experiment:

| Property | Working value |
|---|---:|
| Checkpoint | GLM-5.3-Flash `Q4_0_ROCMFP4_STRIX_LEAN` |
| Total parameters | 320B |
| Active parameters per token | 18B |
| Transformer layers | 45 (34 KDA linear + 11 DSA sparse MLA) |
| Routed experts | 288 (top-8 active per token) |
| Quantized checkpoint size | ~158 GB (~79 GB per node) |
| Cluster memory | 2 x 128 GB = 256 GB gross |
| Gross weight headroom | ~98 GB before runtime and KV allocations |
| Sustained bandwidth assumption | 212 GB/s per node |
| Projected single-node decode | 21.9 tok/s |

GLM-5.3-Flash introduces a hybrid attention architecture (34 KDA linear attention layers with $O(1)$ constant state + 11 DSA layers with low-rank MLA) and an integrated MTP head. ROCmFP4 quantization fits the entire 320B model into the 256 GB dual-node memory pool with ~98 GB of headroom, eliminating the need for destructive expert pruning.

Run the capacity and roofline projection without downloading weights:

```bash
cargo run --locked -- benchmark --config configs/glm53-flash.toml
```

The command labels its output as a **projection**. It does not load GLM-5.3-Flash, generate tokens, or validate output quality.

## Current Status

| Layer | Status | Meaning |
|---|---|---|
| Single-node execution | Measured | llama.cpp direct inference works on ClawRig; Qwen is the current performance regression workload. |
| GLM-5.3-Flash capacity | Projected | The 158 GB checkpoint fits within 256 GB gross cluster memory on paper (~98 GB headroom). |
| Decode roofline | Projected | 21.9 tok/s baseline computed from 18B active weights read (9.7 GB) and 212 GB/s memory bandwidth. |
| Speculative speedup | Assumed | Calculated with MoE verification cost factor $v_\gamma$ (e.g., $v_2 \approx 1.49 \rightarrow 1.38\times$). |
| Dual-node protocol scaffold | Tested | Two processes exchange framed TCP session, prefill, draft, verification, and shutdown messages on localhost. |
| Dual-node GLM-5.3-Flash inference | Experimental | Physical USB4 execution, model sharding, token equivalence, and target-kernel verification have not yet been demonstrated. |

Projected numbers are intentionally not presented as benchmark results.

## Quick Start

Build and test:

```bash
cargo build --locked --release
cargo test --locked
```

Inspect GLM-5.3-Flash feasibility:

```bash
cargo run --locked -- benchmark --config configs/glm53-flash.toml
```

Run the measured single-node regression profile when its GGUF is installed:

```bash
cargo run --locked -- measure --config configs/qwen36-35b-a3b.toml
```

The Qwen profile uses `llama-cli --single-turn`. Its last recorded local result was 226.9 tok/s prompt processing and 52.3 tok/s generation at 16K context. Re-run the command before using those figures as a current comparison.

## Benchmark vs Measure

- `benchmark` reads TOML assumptions and prints gross model fit plus roofline projections. No model is loaded.
- `measure` invokes llama.cpp and reports observed prompt/decode timing. It requires a real local model path.

`model.total_params_b` and `model.checkpoint_size_gb` describe storage capacity. `model.active_params_b * model.bytes_per_weight` estimates bytes read per generated token for the simple decode roofline. They are separate because a sparse MoE stores many more weights than it activates for one token.

## Architecture

```text
Node 0 (NPU)                              Node 1 (iGPU)
+---------------------------+           +---------------------------+
| CPU: session orchestration|           | CPU: session orchestration|
| NPU: 8B draft model       |  USB4/TCP | iGPU: 380B target model   |
| iGPU: local model shard   |<--------->| iGPU: local model shard   |
| 128 GB unified memory     |           | 128 GB unified memory     |
+---------------------------+           +---------------------------+
```

Key modules:

- `src/orchestrator.rs`: HCC pipeline — NPU drafts, iGPU verifies over USB4 (paper Algorithm 1).
- `src/decoding/speculative.rs`: Eq. 5-6 — expected accepted tokens E[k], speedup S.
- `src/decoding/picospec.rs`: Draft-batch queue and compact token/probability encoding; the tested simulator currently uses one stop-and-wait batch.
- `src/interconnect/usb4.rs`: Framed TCP peer transport over configurable thunderbolt-net addresses plus the Eq. 3 analytical timing model.
- `src/interconnect/dmabuf.rs`: Host-memory `memfd`/`mmap` scaffold for a future XRT/ROCm DMA-BUF bridge; it is not cross-driver zero-copy today.
- `src/kv_cache/`: Mixed-precision KV cache — TurboQuant 3-bit values, FP8 keys (Eq. 8-10).
- `src/measure.rs`: Real llama.cpp process execution and timing for Hypothesis H1/H2 validation.
- `src/benchmark.rs`: Eq. 11 — capacity and roofline projection.

Backends are `llamacpp-rpc` for practical local execution, `migraphx` for experimental direct ROCm execution, and `simulated` for configuration/projection work without model execution.

`hcch run` currently proves the distributed protocol with the `simulated` backend and rejects every real inference backend. llama.cpp remains the measured single-node backend used by `measure`; token-equivalent distributed verification and MIGraphX execution are not wired into `run`.

## Why HCC

| Project | Best use | HCC's role |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Reliable GGUF inference on one machine | HCC uses it as the measured execution backend. |
| [vLLM](https://github.com/vllm-project/vllm) | Production serving and high-throughput batching | HCC targets a small, memory-constrained local cluster. |
| [exo](https://github.com/exo-explore/exo) | General clustering across heterogeneous personal devices | HCC studies a fixed dual-ClawRig topology and MoE-specific scheduling. |
| [Petals](https://github.com/bigscience-workshop/petals) | Wide-area decentralized inference | HCC assumes private, predictable local links. |

Use llama.cpp directly for normal single-node chat. Use HCC when the model does not fit on one node and placement/interconnect behavior is the experiment.

## Feasibility Gates

A credible GLM-5.3-Flash result requires all of these:

1. Inspect actual sharded GGUF metadata and measured resident memory.
2. Load complementary model partitions across both nodes without duplicate full-model residency.
3. Verify token/logit equivalence against a known-good reference runtime.
4. Measure prompt speed, first-token latency, decode speed, link traffic, and power.
5. Compare against unmodified llama.cpp on the same checkpoint and hardware.

Until those gates pass, HCC should be read as a research implementation and test harness.

## Hardware Baseline

The reference ClawRig is based on AMD Ryzen AI MAX+ 395 with 128 GB LPDDR5x unified memory. The dual-node experiment assumes two systems and two USB4 links. Memory bandwidth, available device memory, link throughput, and thermal limits must be measured per installation; TOML values are not hardware detection.

## Relationship to Lucebox/DFlash

Lucebox/DFlash informed the single-node work: full accelerator offload, Flash Attention, asymmetric KV types, and prompt caching. HCC asks whether those local fast-path principles remain useful when a much larger MoE checkpoint is partitioned across a small cluster.

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
