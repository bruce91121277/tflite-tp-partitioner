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

- Phase 3C archived component evidence: TP4 P128 MLP public output,
  12 SHA-bound comparisons / 5,505,024 INT16 elements, zero mismatches.
  The audit that rechecked those saved files ran zero fresh NPU invokes.
- Fresh Phase 3D real-prompt component stitches: P128 Layer0, 36 comparisons /
  9,961,472 elements / zero mismatches; D1 Layer0, 36 / 77,824 / zero
  mismatches. These results cover named component boundaries and their rank
  stitches. They do not cover all internal FP32 stages or the complete
  Layer0 public-I/O contract.
- Fresh P128 Layer1 MLP component: four rank outputs, 1,835,008 INT16
  elements, zero mismatches on a real-prompt-derived Layer1 activation. The
  input was the exact Layer0 stitch. The Canonical source was instrumented
  with output taps; equivalence to an untapped NCC execution path has not been
  established. This is component evidence only.
- Full32-package unpadded P128 Layer0 MLP: 363 / 1,835,008 mismatches,
  maximum difference 4, first at `[0, 2, 1382]`, expected 20 and actual 21.
- The structural TP4 P128 Gate rewrite can now emit padding candidates for
  every discovered layer. The rewrite preserves rank-local effective weights,
  but each candidate SHA still requires a matching NPU comparison. The public
  release remains Experimental.
- Fixtures use an authored text prompt with verified tokenizer and embedding
  assets. The exact upstream Hugging Face checkpoint revision is missing from
  source metadata; prompt lineage and layer-boundary tests therefore remain
  scoped to the available fixture and Canonical source SHA.
- TP4 P128/D1 full Layer0 and Full32 are not exact-verified. TP2 row-reduce
  accumulator width, bias ownership, reduction order, rounding, saturation,
  and requantization remain unproven. Distributed runtime is not verified.
  **Release remains BLOCKED.**

NPU evidence must bind the exact model and candidate hashes, input fixture
hashes, output contracts, backend identity, and complete mismatch counts.
Internal taps are diagnostic unless their effect on the compiled computation
is shown to be neutral. Hardware evidence does not belong in public CI, and
private model, DLA, SDK, fixture, or device assets must not be committed.
