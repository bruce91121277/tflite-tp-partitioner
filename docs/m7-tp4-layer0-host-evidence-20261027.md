# M7 TP4 Layer0 Host evidence

**Status: fixture-scoped Host evidence. Release remains Experimental.**

The M7 run reused SHA-verified, untapped P128 and D1 Canonical TFLite Layer0
files that were already available in the authorized development workspace. It
did not rerun W4A16 PTQ or export. The inspected model lineage did not provide
the original checkpoint revision or calibration manifest, so the run does not
claim to reproduce that PTQ lineage.

## Host setup and inputs

- Backend: MTK `TFLiteExecutor`, MTK Converter 8.16.0, Python 3.11.17.
- `simulate_fp16=false`.
- Canonical and rank component executions used their matching P128 or D1
  fixtures and recorded input contracts.
- Rank component outputs were stitched at their named layer boundaries and
  compared bitwise with Canonical Host outputs.
- The components received Canonical boundary activations. This is not an
  integrated distributed runtime run that chains each stitched rank result
  into the next component.

## P128 Layer0

The initial fresh Host comparison found four INT16 values that differed by one
LSB at the current-token attention BatchMatMul: one on Rank0 and three on
Rank1. Q/K/V and both BatchMatMul input operands matched their Canonical rank
slices bitwise. The first differing calculation was `matmul4`; subsequent
attention output stages carried the same differences.

The diagnostic repair ran both exact local operands through BatchMatMul at the
Canonical 32-head batch shape, then sliced the rank-local 8 heads before the
attention add. On the tested P128 fixture, the four diagnostic Rank0-Rank3
component outputs then stitched to exact matches at all reported outputs.

The public builder now locates the context product by INT16 tensor contracts
and its unique direct ADD consumer. In the inspected graph, the other
same-shaped BatchMatMul feeds CONCAT. The rewrite uses compatible TILE and
SLICE operator contracts and fails closed if the target or templates remain
ambiguous. Synthetic tests cover arbitrary tensor names, a same-shaped
CONCAT-fed distractor, equivalent multiple templates, and ambiguous ADD-fed
matches.

The latest build emitted 24 P128 rank-component models. The four attention
model hashes differ from the earlier diagnostic variants. They passed
structural validation and NCC 8.2.22 compilation for
`mdla5.3,edma3.6`, but were not rerun through Host: the current RTX shell had
no `python` launcher, and system `python3` lacked FlatBuffers and MTK
Converter. The attempt stopped before model invocation. The prior diagnostic
exact result does not transfer to the latest hashes. The rewrite is limited
to TP4 P128 Layer0; each emitted candidate needs its own same-backend Host and
NPU verification.

## D1 Layer0

The fresh Host component run reported exact rank stitches for the tested D1
fixture. This result does not establish the complete D1 public input/output
contract or NPU equivalence.

## NCC and NPU

NCC 8.2.22 accepted both Canonical files and all 48 current P128/D1 rank
component SHA values for `mdla5.3,edma3.6`, including the four latest P128
attention replacements. No NPU invoke was made because the EVK was occupied
by an existing runtime process. The process was left undisturbed.

The separate known TP4 NPU issue remains: an observed P128 Layer0 MLP
comparison has 117 INT16 mismatches. Phase 3C is investigating the FP32
Dequantize difference. The M7 Host attention fix does not resolve it. TP2
Row-Reduce Numerical Exact and P128/D1 Full32 NPU Exact also remain unverified.

These Host and compiler results do not certify NPU Exact, Full32, or a
distributed TP runtime. Release remains **Experimental / BLOCKED**.
