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

## Current evidence status

- TP2 and TP4 graph partitioning are implemented.
- An observed TP4 NPU comparison has 117 INT16 mismatches.
- Phase 3C is investigating FP32 Dequantize differences associated with the
  observed numerical discrepancy.
- TP2 row-reduce is not Numerical Exact. Accumulator width, reduction order,
  rounding, saturation, and requantization have not been proven.
- P128 and D1 Full32 NPU Exact remain unverified.

NPU evidence must bind the exact model and candidate hashes, input fixture
hashes, output contracts, backend identity, and complete mismatch counts.
Hardware results should be produced on user-configured equipment with the
Mediatek backend; no device captures are included here.