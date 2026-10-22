# Phase 3C: TP4 P128 Layer0 MLP NPU evidence

**As of 2026-10-22 — experimental, component-scoped evidence; full-model release blocked.**

This page summarizes development evidence without redistributing model weights,
vendor binaries, raw accelerator captures, device addresses, or internal fixtures.
The source of the measurements is the maintainer's SHA-bound Phase 3C report
and its retained private evidence bundle. The underlying private binary captures
are not distributed in this repository.

## Baseline failure

| Field | Observation |
| --- | --- |
| Execution backend | Actual Genio 720 NPU, canonical versus rank-local |
| Graph scope | TP4 P128 Layer0 Rank0 MLP **public INT16 output** |
| Compared elements | 458,752 |
| Mismatched elements | **117** |
| Maximum difference | **1 INT16 LSB** |
| First mismatch | `[0,4,752]`: canonical 16, rank-local 17 |
| Earliest observed internal difference | Gate FP32 Dequantize (instrumentation is diagnostic) |

Gate and Up INT16 values agreed at the examined comparison boundary.
Controlled experiments support an association between graph width and
the Dequantization path; **the causal NPU microkernel / NCC fusion behavior
has not been identified**.

## Development rewrite

The accepted *component-level* candidate changes the Gate branch to:

```text
rank-local Gate FC (INT16)
    -> pad quantized-zero INT16 channels to 2x local Gate output width
    -> Dequantize
    -> Slice back to original rank-local width
    -> downstream MLP
```

Simply widening the FC without the appropriate Dequantize/Slice path did not
resolve the observed mismatch. The rewrite was applied using graph semantics,
not hard-coded layer, rank, channel, tensor index, or expected output values.

A separate FlatBuffer audit checked all four rank models against their
canonical source slices:

- Gate/Up packed INT4 effective weights, bias slices, and quantization metadata.
- Added Gate padding consists of zero INT4 weights and zero bias.
- The intended Gate graph path is `FC -> Dequantize -> Slice`.
- No other rank's effective weights were replicated into a rank-local slice.

### Rank MLP candidate costs

| Property | Per-rank value |
| --- | ---: |
| Effective Gate+Up INT4 weight payload | 14,680,064 bytes |
| Additional zero Gate INT4 padding | 7,340,032 bytes |
| Rewritten TFLite artifact | 22,496,360 bytes |
| Saved DLA artifact | 22,151,505 bytes |

These are serialized artifact sizes, **not** runtime peak memory or a
controlled latency comparison.

## Saved NPU verification

| Field | Result |
| --- | --- |
| Ranks | 4 |
| Fixtures | 3 |
| Full public-output comparisons | 12 |
| Compared INT16 elements | **5,505,024** |
| Mismatched elements | **0** |
| Maximum absolute difference | **0** |
| Nondegenerate MLP public-output fixtures | 2 synthetic input-contract fixtures |
| Real position-37 fixture | MLP output quantized to zero |
| Fresh NPU invokes during latest evidence audit | **0** |
| Input/model/output lineage | Archived evidence verified against recorded SHA |
| Graph and rank-weight audit | PASS |
| Latest local unit testing | 106 passing tests (private tool suite; not the public CI count) |
| Archived checksums verified in latest report | 679 |

The revised rank-model SHA values were reported as matching previously
NPU-tested candidates. This is **recomputed saved NPU evidence**, not a new
run on the current date. The saved public-output equality establishes only
the recorded MLP subgraph comparison scope; tapped internal FP32 tensors
are diagnostic because adding taps can affect compiler kernel selection.

## Remaining validation gates

- [ ] Publish and independently test the Phase 3C rewrite source/CLI in this
  public repository (it was present in the local development worktree when
  this summary was authored).
- [ ] Restore an authorized EVK connection, confirm device idle, and perform
  a **fresh** controlled same-SHA NPU invoke.
- [ ] Derive a trustworthy **real prompt/token** fixture with nonzero MLP
  and Down outputs and capture its canonical NPU reference.
- [ ] Validate the integrated TP4 P128 Layer0, including necessary
  Attention/O/Residual/Norm/MLP/Down/Hidden/KV boundaries and rank stitching.
- [ ] Validate D1 Layer0 and P128/D1 Full32 on fresh candidate and canonical
  NPU references.
- [ ] Independently prove TP2 row-reduction accumulator, reduction, bias,
  rounding, saturation and requantization semantics.
- [ ] Measure runtime performance with controlled benchmarks.

**Current release gate: BLOCKED.**

Success in one MLP component, even across four ranks, does not certify an
entire transformer layer, a full model, or a distributed runtime.
