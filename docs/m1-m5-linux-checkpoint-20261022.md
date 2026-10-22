# M1–M5 Linux source validation checkpoint — 2026-10-22

> **Scope:** maintainer-reported validation of an SHA-pinned development source
> bundle. This is **not** a claim that the current public `main` commit,
> Canonical MTK model outputs, NPU execution, or end-to-end TP2/TP4 inference
> passed the same tests. **Model release remains blocked.**

## Source and environment evidence

| Validation item | Reported result |
| --- | --- |
| Full local development suite | 111 tests passed |
| Extracted source bundle, Windows Python 3.10.18 | 108 tests passed |
| Extracted bundle, RTX4090 Linux Python 3.12.3 | **108/108 PASS, 0 skipped** |
| Linux environment | Linux 6.8.0-88-generic, x86_64 |
| Isolated venv install | PASS |
| `tflite-tp --help` / `tflite-tp-verify --help` | PASS |
| Comparator self-test | PASS |
| `compileall` | PASS |
| Source/test files in the archive | 44 |
| Archive privacy scan | 0 findings (maintainer report) |
| NCC compilation in this checkpoint | **NOT_RUN** |
| Genio 720 NPU invoke in this checkpoint | **NOT_RUN** |

SHA256 of the exact source/test archive:

```text
5c50a2f7cd1f9e8246168096037803e4263d8ed30cb4f26a2a8f5803d1fde827
```

The reported isolated Linux validation used SHA-verified offline
pure-Python dependency wheels where PyPI was unavailable. It did not change
system Python or SSH settings. The earlier 106-test bundle had a different
SHA and is **not** the evidence source for these 108 Linux tests.

Authorized SSH from the development environment to the **RTX4090 Linux
host** succeeded. This establishes Linux validation access only; it does
**not** establish current EVK SSH/idle status or a new NPU run.

## Numerical release gates — **FAIL / REAL_MODEL_NOT_VERIFIED**

| Required gate | Current status |
| --- | --- |
| Registered Canonical MTK custom-op Host Executor | **MISSING / BLOCKED** |
| Complete TP4 P128 real-model Host Golden | **NOT_RUN** |
| Complete TP4 D1 real-model Host Golden | **NOT_RUN** |
| Complete TP2 P128/D1 real-model Host Golden | **NOT_RUN** |
| TP2 O row-reduce integer/quantization contract | **UNPROVEN** |
| TP2 Down row-reduce integer/quantization contract | **UNPROVEN** |
| Complete real-model mismatch count | **null** (not zero) |
| NCC / NPU execution as part of M1–M5 | **NOT_RUN** |
| Model release | **FAIL — REAL_MODEL_NOT_VERIFIED** |

For the maintainer's current Golden gate, the compatible custom-op executor
must register as **MTK `TFLiteExecutor`**, operate with
`simulate_fp16=false`, enforce exact canonical/candidate/source SHA
mapping, use valid per-layer fixtures, and produce complete bitwise
boundary comparisons. An ordinary TFLite interpreter, synthetic fixture
or isolated comparator pass cannot replace this requirement.

TP2 **O/Down row reduction** requires an independently evidenced integer
contract for partial sums, accumulator width, cross-rank order, bias
ownership, quantization scales/zero-points, rounding, saturation and
requantization. Synthetic arithmetic experiments can help infer the
contract, but cannot alone certify the real-model NPU/Host behavior.

Previous Phase 3C/3D **component-scope NPU comparisons** are described in
[the numerical-verification guide](numerical-verification.md). Their saved
or fresh component evidence does not transfer to these as-yet-unexecuted
full-model Goldens.

## Next actions

1. Audit the authorized existing toolchains and development workspaces for
   a compatible Canonical MTK custom-op `TFLiteExecutor` (including exact
   model/profile compatibility and runner provenance); do not invent a
   substitute with different math.
2. Register the executor with the current Golden runner in an isolated
   workspace, enforce `simulate_fp16=false` and fail-closed SHA/fixture
   mapping, and attempt a minimal Layer0 boundary canary.
3. If the actual executor is unavailable, document a precise
   `BLOCKED_MISSING_CANONICAL_EXECUTOR` gate; do not claim numerical exactness
   from mocked or synthetic tests.
4. Independently investigate and verify TP2 O/Down row-reduce numerical
   semantics against an authoritative Canonical execution path.
5. Only with complete validated references and nondegenerate fixtures,
   execute P128/D1 full-model comparisons, record every mismatch/coverage
   count, and reconsider the release gate.

This note is a sanitized summary of the maintainer's M1–M5 checkpoint.
The local original reports, model binaries, vendor SDKs, device configuration,
and private captures are not distributed here.
