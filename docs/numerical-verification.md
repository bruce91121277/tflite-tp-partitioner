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

## M8 TP4 Host evidence (2026-10-27)

The latest PR #4 P128 attention candidates were verified against the selected
Canonical and fixture manifests before invocation. The M6 Python 3.11.17
environment was recovered; MTK Converter 8.16.0 `TFLiteExecutor` imported and
executed with `simulate_fp16=false`.

The P128 Host harness executed all 24 rank-component models in six sequential
stages, using manifest-ordered all-gathers between stages. Nine unique public
boundaries were compared with the saved tapped Canonical Host capture:
5,242,880 INT16 elements, zero mismatches, maximum absolute difference zero.
This includes the attention context after the padded BatchMatMul and SLICE.
The four one-LSB attention-context mismatches recorded by M6 were not
reproduced for these latest candidates on this fixture. The internal padded
BatchMatMul output was not separately exposed and compared.

The uninstrumented P128 Canonical model was then invoked directly through the
same MTK Host Executor with the same five verified fixture inputs. Its three
public outputs (`hidden_states_0`, `curr_keys_0`, and `curr_values_0`) matched
both the saved tapped Canonical capture and the TP4 sequential stitch over
786,432 INT16 elements with zero mismatches. This confirms tap equivalence for
these three public outputs on this fixture. It does not establish equivalence
for internal tapped stages or other inputs.

The D1 harness also executed all 24 rank-component models sequentially. Its
nine public boundaries compared with the saved tapped Canonical Host capture
over 40,960 INT16 elements with zero mismatches. D1 input embeds, position,
mask, current token, and rank-local KV followed the saved D1 fixture contract.

These results are fixture-scoped Host evidence, not NPU verification or a
distributed runtime test. P128's three uninstrumented public outputs were
compared directly; P128 internal stage references and D1 references remain
saved tapped Canonical captures. Equivalence of every internal tapped value
to the untapped path is not established. Internal Q/K/V and Gate/Up outputs
remain outside the candidate public I/O coverage. No NPU invoke was made
because the EVK was busy. The previously observed 117 INT16 P128 Layer0 MLP
NPU mismatches remain unresolved; Phase 3C is investigating FP32 Dequantize.
TP2 Row-Reduce and Full32 remain open, and release remains Experimental /
BLOCKED.

Static NCC comparison accepted the four padded attention candidates. Per rank,
the TFLite file grew by 1,064 bytes (0.0164%) and the generated DLA size grew
by 1,841 bytes (0.0288%). The logical padded INT16 tensor is 1 MiB, versus
256 KiB before padding. NCC did not report whether the BatchMatMul was fused
or eliminated. Peak allocation and NPU latency were not measured.

## M7 TP4 evidence (2026-10-27)

- Reused SHA-verified, untapped P128 and D1 Canonical TFLite files. No PTQ or
  export was run because matching Canonical files already existed. Their
  original checkpoint revision and calibration manifest were not recoverable
  from the inspected lineage, so this is not a reproducible PTQ claim.
- Fresh MTK `TFLiteExecutor` runs used `simulate_fp16=false`. An isolated
  P128 Layer0 Host variant initially found four one-LSB differences at the
  current-token attention BatchMatMul. Rank-local operands were bitwise equal
  to the Canonical head slices. Running the MatMul with the Canonical 32-head
  batch shape, then slicing the local 8 heads, removed those differences on
  the tested fixture. Its reported P128 component outputs then matched exactly
  after stitching. Inputs at component boundaries came from Canonical; this
  does not prove a chained distributed Layer0 execution. The latest four
  public-builder attention hashes differ from the diagnostic variants. They
  passed structural validation and NCC 8.2.22 compilation for
  `mdla5.3,edma3.6`, but have not been run through Host. The latest runner
  attempt stopped before model invocation because the active RTX shell had no
  `python` launcher and its system `python3` lacked FlatBuffers and MTK
  Converter. The diagnostic result does not transfer to those files.
- D1 Layer0 Host component outputs and stitches matched exactly on the tested
  fixture. This does not prove integrated D1 public-I/O behavior.
- NCC accepted both Canonical models and all 48 current rank-component SHA
  values for the recorded target, including the four latest P128 attention
  replacements. Compiler acceptance is not an NPU result.
- No M7 NPU comparison was run because the EVK was occupied by an existing
  runtime process. The observed TP4 P128 Layer0 MLP NPU result still has 117
  INT16 mismatches. Phase 3C is investigating the FP32 Dequantize difference;
  the M7 Host attention rewrite does not address that mismatch.
- TP2 Row-Reduce Numerical Exact remains incomplete. P128/D1 Full32 NPU Exact
  remains unverified. Release status is **Experimental / BLOCKED**.

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

## M1?5 source-bundle validation (2026-10-22)

The maintainer separately verified a SHA-pinned development source/test
archive on RTX4090 Linux (Python 3.12.3): **108/108 tests passed, 0 skipped**,
with isolated package install, two CLI smoke checks, comparator self-test,
and compileall. The full local test suite passed 111. No NCC build or NPU
invoke was part of this checkpoint. These are **source/contract tests**, not
real-model Exact gates. See the [M1?5 checkpoint](m1-m5-linux-checkpoint-20261022.md).

The complete P128/D1 TP2/TP4 real-model Golden remains **NOT_RUN** because the
required compatible Canonical MTK custom-op Host `TFLiteExecutor` is missing.
The gate requires `simulate_fp16=false`, exact candidate/canonical/source
SHA mapping, valid per-layer fixtures, and complete bitwise output-boundary
coverage. Missing Golden returns `MISMATCH_COUNT=null`, never zero.
TP2 O/Down row-reduce accumulator, ordering, bias, rounding, saturation and
requantization semantics remain **UNPROVEN**. This checkpoint's model gate is
**FAIL / REAL_MODEL_NOT_VERIFIED**.

NPU evidence must bind the exact model and candidate hashes, input fixture
hashes, output contracts, backend identity, and complete mismatch counts.
Internal taps are diagnostic unless their effect on the compiled computation
is shown to be neutral. Hardware evidence does not belong in public CI, and
private model, DLA, SDK, fixture, or device assets must not be committed.

