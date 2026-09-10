# 0-3Plan.md - hcc-edge-moe: from experiment to a viable GLM-5.3-Flash cascade

## Vision

Upgrade `hcc-edge-moe` from a **simulated protocol + single-node measurement**
project into a **viable two-node GLM-5.3-Flash cascade over USB4.0**, with the
XDNA 2 NPU given a real, measured role.

Target: `hcch run` executes the model across two Strix Halo units, and every
claim is a measurement, not a projection.

## Definition of viable

Pass the project's own five feasibility gates with GLM-5.3-Flash:

1. Real sharded GGUF metadata inspected; resident memory measured.
2. Complementary partitions across both nodes; no duplicate full-model residency.
3. Token/logit equivalence vs a reference runtime.
4. Measured prompt speed, TTFT, decode speed, USB4 traffic, power.
5. Benchmarked vs unmodified llama.cpp on the same checkpoint + hardware.

Plus: the cascade is faster or meaningfully equivalent to a single-node run, and
the NPU question is answered with data.

## Current state (baseline)

- v0.2.0, Rust CLI `hcch`, ~3,950 LOC, HEAD `78c3558`.
- `hcch run`: **simulated** (framed TCP loopback, synthetic tokens, rejects real
  backends).
- `hcch measure`: real single-node llama.cpp.
- `hcch benchmark`: projections only.
- Known defects:
  - `src/npu/draft_runner.rs` hashes token strings (FNV) instead of vocab IDs and
    silently substitutes placeholders.
  - `src/decoding/picospec.rs` ignores `top_k` and has no residual `max(0,p-q)`.
  - `src/decoding/speculative.rs` uses the optimistic `vγ=1` model.
  - `src/interconnect/usb4.rs` is framed TCP; `recv_dmabuf` copies.
- Runtime assets:
  - `~/source/ROCmFPX` (`release/vulkan-v1.7.1`): ROCmFP4 + MTP + Vulkan,
    **no glm5next**.
  - `~/source/ROCmFPX-glm5next` (`glm5next` arch incl. `glm5next.cpp`,
    `glm-dsa.cpp`, vision), **no ROCmFP4**.
- Weights: `/mnt/ssd2/models/glm-5.3-flash` is **empty**.

## Target architecture

```
[Unit A]                                   [Unit B]
 hcch run (cascade brain)                   hcch run (cascade brain)
   |  Orchestrator: session, tokens,          |
   |  metrics, gates                          |
   |                                          |
   +-- llama.cpp ROCmFPX + glm5next --+       +-- llama.cpp ROCmFPX + glm5next
   |   (gfx1151, GLM-5.3-Flash        | USB4.0 |  (gfx1151, GLM-5.3-Flash
   |    Q4_0_ROCMFP4_STRIX_LEAN + MTP)|<=====>|   Q4_0_ROCMFP4_STRIX_LEAN + MTP)
   |                                  |        |
   +-- XDNA 2 NPU: off-critical-path  |        +-- XDNA 2 NPU (same role)
       (vision preprocess / sidecar)  |
```

- **Model:** GLM-5.3-Flash, 320B total / 18B active, 45 layers (KDA linear +
  sparse MLA), 288 experts top-8, MTP head, 1M context.
- **Quant:** `Q4_0_ROCMFP4_STRIX_LEAN` (~4.3 bpw, ~155-160 GiB) split across both
  units; `UD-IQ3_XXS` (~120 GB) as a single-unit falsification profile; `IQ4_XS`
  as control.
- **NPU:** primary role off the token-critical path; drafting is a gated
  experiment (below).

## Workstreams

### WS0 - Runtime foundation (blocking)

- 0.1 Read-only merge feasibility: `glm5next` <-> ROCmFPX `release`; choose merge
  strategy.
- 0.2 Produce merged worktree; build gfx1151 (`scripts/build-strix-rocmfp4-mtp.sh`).
- 0.3 Obtain GLM-5.3-Flash BF16/F16 GGUF source (~642 GB); verify disk.
- 0.4 Build imatrix calibration corpus.
- 0.5 Quantize: `Q4_0_ROCMFP4_STRIX_LEAN`, `Q4_0_ROCMFP4_STRIX`, `UD-IQ3_XXS`,
  `IQ4_XS`.
- 0.6 Single-unit bring-up: load, generate, MTP acceptance.
- **Gate G0:** merged tree builds; single-unit loads and generates.

### WS1 - Real single-node execution + correctness

- 1.1 Real backend in `HccOrchestrator` (llama.cpp server/RPC); drop
  simulated-only gate.
- 1.2 Return vocabulary IDs; remove FNV hashing + silent placeholders.
- 1.3 Exact rejection sampling: `max(0, p-q)`, honor `top_k`.
- 1.4 Replace vγ=1 roofline with measured vγ for γ=2..5.
- 1.5 Measurement harness: TTFT, prompt/decode t/s, resident GB, acceptance, link
  bytes.
- **Gate G1:** token/logit equivalence vs reference; correctness tests green.

### WS2 - Two-node cascade over USB4.0

- 2.1 Sharding: layer/expert split; no duplicate residency.
- 2.2 Transport: llama.cpp RPC over `thunderbolt-net` first; framed TCP fallback.
- 2.3 Cross-node speculative commit protocol: single authoritative sampler on the
  final-shard unit; commit record `{seq, accepted prefix, replacement/bonus, new
  length}`; both shards commit/discard KV identically; one in-flight batch until
  rollback proven.
- 2.4 Measure decode/prompt t/s, TTFT, USB4 traffic, node skew, p50/p95.
- **Gate G2:** token equivalence across nodes; dual-vs-single tradeoff quantified.

### WS3 - NPU (gated; measured, not assumed)

Oracle verdict: NPU drafting likely yields **no gain** because:

- the drafter is slower than the target (5-40 t/s vs ~15 t/s);
- MoE routing breaks the "one weight read per γ tokens" assumption
  (v2 ~= 1.49, v5 ~= 2.87);
- the NPU shares LPDDR5X bandwidth (18B active ~= 9.7 GB/token ~= 145 GB/s at
  15 t/s; a 2B drafter adds 40-80 GB/s vs a 212 GB/s ceiling).

Tasks:

- 3.1 Decision-model harness: measure `R_T`, `R_MTP`, `vγ`, real acceptance,
  NPU/iGPU contention.
- 3.2 NPU bring-up (XRT + MLIR-AIE/IRON or Ryzen AI OGA); measure t/s, dispatch,
  KV persistence.
- 3.3 Tokenizer-alignment audit for a sub-2B GLM drafter.
- 3.4 Bounded drafter experiment (γ=2) against the inequality.
- 3.5 Off-critical-path role: profile DSA indexer; evaluate **multimodal vision
  preprocessing** on NPU; async sidecar.
- **Gate G3:** pursue drafting only if **>=1.20x projected / >=1.10x measured**
  over GPU MTP; else adopt off-path role. Record the result either way.

Note: the NPU role is scheduled so it does **not** overlap the bandwidth-bound
decode (use prefill/vision/idle windows).

### WS4 - Measurement, quality, paper

- 4.1 Fixed corpus (2K/32K/128K).
- 4.2 Quality per quant: PPL + task battery.
- 4.3 Power/thermal.
- 4.4 Update README status table + paper: replace projections with measurements;
  document NPU outcome (including negative).

## Sequencing

```
WS0 -> WS1 -> WS2
         |
         +--> WS3 (gated, after WS1)
WS4 runs across every gate
```

## Risks

| Risk | Mitigation |
|---|---|
| Merge surface large (186 files) | Assess first (0.1); rebase `glm5next` onto ROCmFPX |
| `glm5next` not upstream | Pin fork; track PRs #27752/#27754/#27773 |
| gfx1151 ~98 GB driver wall | `--n-cpu-moe`, GTT sizing, shard |
| BF16 source ~642 GB | Verify storage before 0.3 |
| Dual-unit USB4/sync | Single-unit first; dual gated on hardware |
| ROCmFP4 experimental | Always carry IQ4_XS + BF16 control |
| NPU toolchain immaturity | Bounded falsification; never block viability on it |
| ROCmFPX repo rules | No commit/push/PR without per-action owner authorization; never act on `ggml-org` |

## Open questions

1. Single-unit 3-bit "viable" first, or straight to dual-unit 4-bit?
2. NPU: gated drafting experiment, or go directly to off-critical-path role?
3. Quant set: `STRIX_LEAN` only, or the full quartet?
4. Are both Strix Halo units and ~700 GB of free space available now?
5. Authorize local-only work in `~/source/ROCmFPX` (scratch worktree, no pushes)?

## Immediate next action

**Task 0.1** - read-only merge feasibility (`glm5next` <-> ROCmFPX `release`) in a
scratch worktree, plus stand up the WS3.1 measurement harness. Both
non-destructive.
