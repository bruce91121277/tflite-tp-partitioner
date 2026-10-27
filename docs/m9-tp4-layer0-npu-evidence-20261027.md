# M9 TP4 P128/D1 Layer0 Genio 720 NPU fixture checkpoint

**As of 2026-10-27: NPU fixture-scoped Layer0 output PASS; Full32 and
distributed TP4 runtime NOT_VERIFIED; release remains Experimental/BLOCKED.**

This is a **sanitized summary of the maintainer's private M9 evidence report**.
The underlying TFLite/DLA models, source fixtures, raw NPU captures, runtime
binaries, SSH settings and vendor tools are not distributed here. Public
packaging/CI tests do not independently reproduce this private hardware test.

## Lineage and execution

- Previously verified, uninstrumented P128 and D1 Canonical models were reused.
  No PTQ, model re-export or shape-fix was performed in M9.
- Canonical P128 SHA256:
  `99514fc43e504cd1602947096033de7b9d2e7dfb7a15dced85b4b26387447b0c`
- Canonical D1 SHA256:
  `64adb14bc3407e2eac1f07eaf7b74aa1082e1f2f016df80b9ecec5ddb15c7fef`
- 48/48 candidate TFLite and 48/48 DLA recorded SHA/size values revalidated.
  Source-builder provenance is bound to the M7/M8 recorded candidate set.
- NCC 8.2.22 target `mdla5.3,edma3.6`; actual Genio 720 Neuron
  runtime binary was recorded by its SHA256 in the retained private report.
- The one permitted idle EVK was used; four other EVKs had busy processes
  and were skipped without interruption. **112 successful fresh NPU invokes**.
- Rank outputs were stitched on the **Host**. No concurrent four-device
  TP4 collective execution was performed in this checkpoint.

## Output comparisons

| Checked scope | Recorded result | Qualification |
| --- | --- | --- |
| P128 Layer0 rank, stitch and diagnostic rechecks | **0 / 14,155,776** mismatches | This count includes repeated comparisons, not unique model elements |
| D1 Layer0 rank, stitch and diagnostic rechecks | **0 / 116,736** mismatches | Likewise includes rechecks |
| P128 attention/context after 4x head-batch padding | **PASS** | Current SHA-bound NPU fixture |
| P128/D1 post-SwiGLU MLP, Down, final hidden and public outputs | **PASS** | Nonzero output fixtures |
| P128 final hidden vs uninstrumented Canonical NPU public output | **PASS** | Same recorded fixture and public output contract |
| D1 uninstrumented and tapped Canonical public outputs | **PASS** | Same recorded D1 fixture |
| Independently tapped Gate and Up intermediates | **NOT_VERIFIED** | NCC diagnostic failed; no NPU invoke |
| P128/D1 layers 1–31 | **NOT_RUN** | Matching candidate/fixture lineage not present in this bundle |
| Concurrent TP4 collectives, end-to-end prefill/decode and TPS | **NOT_RUN** | Host stitching does not establish distributed runtime |

The maintainer's full comparison accounting includes rank, stitched and tapped
rechecks; it **must not be interpreted as a count of distinct model outputs**.
P128 stitched MLP had 1,626,600 nonzero values; D1 had 11,772.

The production MLP contract exposes the post-SwiGLU `mul/mul` result; it
does not separately expose Gate and Up activations. An instrumented
diagnostic candidate failed NCC `--verify` with unresolved `MTKEXT_SILU`
at node 4. The `--no-verify` attempt exited 139. **Neither instrumented
candidate was invoked on the NPU**; the production models were unchanged.
These diagnostic failures are not mismatches in the accepted public outputs.

### Historical 117-mismatch interpretation

The **117 INT16 NPU mismatches** were recorded on an older **unpadded**
P128 Rank0 MLP candidate. A Phase 3C padded candidate later achieved zero
differences on saved SHA-matched NPU comparisons. M9 independently
reported zero differences on the selected current **post-SwiGLU/Down/public
output boundaries** using fresh NPU invocations. These are different model
SHAs and evidence scopes; the historical 117 mismatch must not be attributed
to this new run, nor does this new result prove unobserved Gate/Up tensors.

## Short attention head-padding performance canary

| Metric | Unpadded | Head padding x4 |
| --- | ---: | ---: |
| Per-rank attention NPU invoke median | 62.143 ms | 71.185 ms |
| TFLite size | 6,474,168 B | 6,475,232 B |
| DLA size | 6,400,005 B | 6,401,846 B |
| Logical INT16 head-batch tensor | 256 KiB | 1 MiB |

The per-rank invoke was **approximately 14.55% slower** with padding in a
small, alternating three-sample-per-variant canary. This is **not** a
controlled end-to-end throughput result. Runtime peak memory, operator
fusion, BatchMatMul standalone latency, prefill TPS and decode TPS are
unmeasured; serialized DLA size is not runtime memory.

## Next work — Full32 first, then distributed TP4

1. Recover or regenerate a **32-layer W4A16 Canonical P128 and D1 set**
   from the same approved model/PTQ lineage and check quantization,
   embedding/position/KV and per-layer input/output contracts. If identical
   upstream PTQ intermediates are present, reuse them rather than repeating
   the entire PTQ; if not, build a new independently identified lineage and
   regenerate *all* dependent rank models, fixtures and Goldens.
2. Build SHA-bound TP4 Rank0–Rank3 candidates for layers 1–31, keeping
   the experimental Gate and attention padding rules only within their
   tested graph/profile scope until additional same-backend numerical evidence.
3. Extend fixture-scoped NPU tests in batches (L1–L3, L4–L15, L16–L31),
   keeping complete output-boundary coverage and model/fixture SHA identity.
4. Investigate an NCC-supported, minimally perturbative Gate/Up observation
   path **without** conflating instrumentation effects with production
   numerical behavior.
5. Run real concurrent four-EVK TP4 collectives on an authorized idle
   cohort; validate synchronization, rank ordering, CRC/binary exactness,
   KV progression and end-to-end prefill/decode; measure true TPS and
   padding overhead. Skip busy boards without killing other processes.
6. Keep TP2 row-reduce as an independent deferred workstream.

**Gate decision: `P128_LAYER0=PASS_FIXTURE_SCOPED`,
`D1_LAYER0=PASS_FIXTURE_SCOPED`, `FULL32=NOT_RUN`,
`DISTRIBUTED_TP4=NOT_RUN`, `RELEASE=BLOCKED`.**
