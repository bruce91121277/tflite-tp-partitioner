# TFLite Tensor Parallel Partitioner

[![Project status: experimental](https://img.shields.io/badge/status-experimental-orange)](#project-status)

> **Experimental — not production-ready.** Public source publication is pending review. This repository currently documents the project and its intended architecture; it is **not yet an installable partitioner**.

An open-source project in preparation for automatically partitioning quantized TFLite large-language-model graphs into **tensor-parallel rank-local components** for resource-constrained edge AI accelerators.

## Why this project?

Quantized LLM graphs combine packed INT4 weights, per-axis quantization parameters, attention/MLP subgraphs, and KV-cache state. Splitting them across accelerator devices while preserving numerical behavior is difficult. This project aims to make graph discovery, partitioning, and **evidence-based validation** reproducible rather than relying on manual tensor-by-tensor editing.

## Scope

- **TP2 / TP4** graph partitioning for quantized TFLite LLMs; TP8 is a roadmap target.
- Discover attention, Q/K/V, output projection, MLP, residual, and KV-cache boundaries.
- Generate per-rank components with packed INT4 weight slicing and quantization metadata preservation.
- Validate TFLite structure, weight reconstruction, graph I/O contracts, and checksums.
- Compare outputs against a matching canonical graph **on the same execution backend**, with explicit output mapping and complete tensor coverage.
- Reject a numerical release when reference coverage is missing or a mismatch remains.

Hardware-specific integration (such as MediaTek NCC/Neuron Runtime) is kept separate from portable graph transformation logic. Proprietary SDKs and model weights are not bundled.

## Project status

The private development prototype has generated TP2 and TP4 components and passed structural checks. **That does not establish numerical correctness.** A nondegenerate TP4 NPU regression has identified small INT16 discrepancies downstream of floating-point intermediate operations; a general fix and complete full-model verification are outstanding.

| Capability | Status |
| --- | --- |
| Quantized TFLite graph partitioning | Prototype; source release pending |
| Structural / INT4 reconstruction tests | Passed in development; public reproduction pending |
| TP4 NPU bit-exact validation | **Blocked by unresolved mismatch** |
| TP2 row-reduction semantics | **Unverified** |
| P128 / D1 Full32 numerical equivalence | **Not established** |
| Distributed multi-device runtime | **Not verified** |
| Public installable release | **Not yet available** |

No prior structural pass, compiler success, or NPU invocation is presented as proof of bit-exact equivalence.

## Intended pipeline

```text
Canonical W4A16 TFLite
        |
  Semantic graph discovery
        |
 TP2 / TP4 graph partitioning
        |
 Rank-local TFLite components
        |
 Structural / weight / SHA checks
        |
 Backend compiler and executor
        |
 Canonical-vs-rank stage comparison
        |
 Exact gate or fail-closed block
```

## Getting started

**The implementation is not yet published.** After a code ownership, third-party-license, and secrets review, this repository will receive installable Python sources, synthetic fixtures, a documented CLI, and reproducible automated tests.

To follow development priorities, see [the roadmap](docs/roadmap.md). Contributions and reproducible issue reports are welcome once source publication begins.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## Licensing and intellectual property

No license has yet been granted for the implementation. Public visibility **does not** grant reuse rights. A compatible open-source license will be added **after ownership and publication permissions are confirmed**.

No proprietary SDK binaries, model checkpoints, compiled accelerator artifacts, device credentials, private fixtures, or employer/internal materials are included or authorized for inclusion.
