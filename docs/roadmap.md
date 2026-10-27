# TP2 and TP4 roadmap

## TP2

- Define and validate the row-reduce accumulator, cross-rank reduction order,
  rounding, saturation, and requantization contract.
- Close the row-reduce numerical gap with exact comparisons on reproducible
  synthetic cases and authorized hardware captures.
- Extend verification from isolated row-reduce boundaries to chained
  candidate activations.
- Verify P128 and D1 Full32 NPU behavior.

## TP4

### M9 completion and M10 priorities (2026-10-27)

- [x] Recover a SHA-verified untapped P128/D1 Layer0 Canonical pair
  and generate SHA-verified four-rank component candidates.
- [x] Validate the latest P128/D1 Layer0 outputs against Canonical
  on a real Genio 720 NPU fixture: **112 fresh invokes; zero mismatch**
  across the recorded rank/stitch/tap rechecks.
- [x] Verify nonzero post-SwiGLU MLP, Down and public hidden output
  comparisons at Layer0 on the tested NPU fixture.
- [ ] Recover or build the **matching 32-layer W4A16 P128/D1 Canonical**
  lineage; preserve PTQ/quantization contract and SHA evidence; avoid
  unnecessary PTQ rebuild when valid intermediates exist.
- [ ] Generate and validate matching TP4 P128/D1 candidates and
  fixtures for **L1–L31**, with repeatable layer-by-layer NPU Goldens.
- [ ] Run bounded Full32 checks in batches: L1–L3, L4–L15, L16–L31,
  then D1 Full32 and realistic prompt-to-output prefill/decode.
- [ ] Find a supported NCC/Neuron method for independent Gate/Up
  tensor observation that does not alter production numerical semantics;
  `MTKEXT_SILU` diagnostic tap currently fails compilation.
- [ ] Exercise **real concurrent 4-EVK** collectives and KV flow, not
  merely Host stitches of individual NPU captures; skip busy hardware.
- [ ] Profile/optimize the attention head-padding penalty (M9 per-rank
  short canary: +14.55% median invoke wall time). Compare end-to-end
  TPS only with controlled concurrent-runtime measurements.
- [ ] Keep complete model release marked **BLOCKED / Experimental**
  until full-profile numerical and distributed execution gates pass.


- [x] Find the M7 P128 Layer0 Host attention-context first difference and
  verify a head-batch shape rewrite on one SHA-bound Host fixture.
- [x] Locate the attention-context BatchMatMul by tensor contract and direct
  ADD-consumer topology; cover arbitrary names, same-shaped CONCAT distractors,
  equivalent templates, and fail-closed ambiguity with synthetic tests.
- [x] Rebuild the four latest P128 attention candidates and compile them with
  NCC 8.2.22 for `mdla5.3,edma3.6`.
- [ ] Run the latest attention candidate hashes through Host. The current
  runner could not start because the active RTX shell lacked an accessible
  MTK Converter Python environment; no model invocation occurred.
- [ ] Test the opt-in attention-context rewrite candidates on NPU; previous
  exact evidence is fixture-scoped and tied to different hashes.
- [ ] Resolve the observed 117 INT16 TP4 P128 Layer0 MLP NPU mismatches and
  complete the Phase 3C FP32 Dequantize investigation.
- [ ] Establish integrated D1 Layer0 NPU exactness, then P128/D1 Full32 NPU
  exactness.
- [x] Investigate the P128 Layer0 MLP discrepancy (originally 117 / 458,752
  INT16 mismatches) and identify a graph-based, fixture-independent rewrite.
- [x] Recheck four Rank MLP public outputs against saved, SHA-matched NPU
  evidence: 12 comparisons / 5,505,024 INT16 elements, zero mismatches.
- [x] Audit Gate/Up INT4 rank weight slices, bias, quantization parameters,
  and Gate FC -> Dequantize -> Slice topology.
- [x] Publish the structural TP4 P128 Gate zero-padding rewrite and synthetic
  regression tests. Candidate generation is supported across discovered P128
  layers; NPU exactness remains candidate-SHA-specific and component-scoped.
- [x] Restore the authorized EVK path and capture fresh P128/D1 component
  stitches and a P128 Layer1 MLP component comparison on Genio 720.
- [ ] Re-run public-commit candidates on NPU; prior private candidate results
  do not transfer across model SHA.
- [x] Build P128 and D1 real-text/tokenizer/embedding fixtures and preserve
  their input/KV contracts; the upstream checkpoint revision is not recorded,
  and full-model prompt lineage remains open.
- [ ] Confirm non-degenerate Down coverage across representative prompts.
- [ ] Validate the **integrated P128 Layer0** across Rank0-Rank3 and all
  required Attention, O, Residual, Norm, MLP, Down, Hidden, and KV boundaries.
- [ ] Run D1 Layer0 and then P128/D1 Full32 on all required output elements
  with matching canonical NPU references.
- [ ] Measure padding overhead and NPU latency under controlled conditions.
- [ ] Validate distributed transport and end-to-end inference separately
  from individual graph partitioning correctness.

## Current model-level blockers (M1–M5 checkpoint)

- [x] Validate the SHA-pinned development source bundle on authorized
  RTX4090 Linux: 108/108 tests, zero skipped; installation, CLI smoke,
  comparator self-test and compileall PASS (no NCC/NPU invoked).
- [ ] Locate and register a **compatible** Canonical MTK custom-op Host
  `TFLiteExecutor` with `simulate_fp16=false`. Require canonical/model/
  candidate SHA and per-layer fixture checks before an actual Host Golden.
- [ ] Execute complete Canonical and candidate P128/D1 per-layer boundary
  comparisons; leave `MISMATCH_COUNT=null` for NOT_RUN cases.
- [ ] Prove independently the **TP2 O/Down row-reduce integer numerical
  contract**, including accumulator width, bias placement, reduction order,
  rounding, saturation, scaling and requantization.
- [ ] Preserve the separation of earlier component-scope NPU results from
  full-model Golden/release claims.
- [ ] Retest source changes on Linux and in public CI; a development bundle
  test pass is not a CI result for public `main`.

## Release gate

Keep the release marked Experimental until model-bound host checks and
full-output NPU exactness pass for the documented profiles. Compiler acceptance
and unit tests alone do not satisfy this gate.
