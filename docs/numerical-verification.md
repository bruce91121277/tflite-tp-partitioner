# Numerical verification

Structural correctness, host numerical equality, compiler compatibility, and
NPU numerical equality are separate claims.

## Evidence levels

1. **Structural checks** validate graph coverage, tensor contracts, rank
   components, and serialized TFLite structure.
2. **Host Golden checks** compare candidate execution with a canonical graph
   under a compatible interpreter.
3. **Compiler checks** establish only that a target compiler accepted and
   generated an artifact.
4. **NPU exact checks** compare all required candidate outputs against
   canonical NPU outputs for the same model, inputs, and output contract.
5. **Distributed checks** validate the complete TP execution path, including
   communication and chained activations.

Passing an earlier level does not imply a later one. Null or missing mismatch
counts are not zero. A compiler success is not NPU verification.

## Current evidence status (2026-10-22)

- TP2 and TP4 graph-partitioning code exists in the public experimental
  project. The latest Phase 3C MLP Gate rewrite is not yet committed to
  this public branch.
- A nondegenerate **original TP4 P128 Layer0 Rank0 MLP** NPU output
  comparison showed **117 / 458,752 INT16 mismatches** (maximum 1 LSB;
  first [0, 4, 752], canonical 16 vs rank-local 17).
- A revised candidate applied zero padding to the **rank-local INT16 Gate
  output before Dequantize**, followed by a Slice restoring the local output
  width. On SHA-matched archived real Genio 720 NPU captures, **12 comparisons
  across four ranks and three fixtures, 5,505,024 public MLP INT16 elements,
  yielded zero mismatches**.
- FlatBuffer inspection confirmed the candidate's effective Gate/Up INT4
  weight and bias slices against the canonical graph, including the padding
  and quantization metadata. The saved evidence is for MLP *public output*,
  not for the integrated Layer0 or internal instrumented FP32 taps.
- Two fixtures have nonzero public MLP outputs; the real-position-37 output
  is quantized zero. A real-prompt-derived nondegenerate fixture remains
  required. This Phase 3C evidence recheck had **zero fresh NPU invokes**
  because SSH/idle access could not be established.
- The width-sensitive Dequant behavior is supported by controlled tests, but
  the root NPU microkernel / NCC fusion cause remains unconfirmed. Changing
  a graph to tap internal FP32 activations may alter compiler behavior, so
  those taps are diagnostic rather than release-gate evidence.
- TP4 P128 integrated Layer0, D1 Layer0, and P128/D1 Full32 are **not
  numerically verified**. TP2 row-reduce accumulator, bias ownership,
  reduction order, rounding, saturation, and requantization remain unproven.
  The distributed runtime is not verified. **Release remains BLOCKED**.

See [Phase 3C MLP evidence](phase3c-mlp-npu-evidence.md).

NPU evidence must bind the exact model and candidate hashes, input fixture
hashes, output contracts, backend identity, and complete mismatch counts.
Hardware results should be produced on user-configured equipment with the
Mediatek backend; no device captures are included here.