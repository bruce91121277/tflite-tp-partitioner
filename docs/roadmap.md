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

- [x] Investigate the P128 Layer0 MLP discrepancy (originally 117 / 458,752
  INT16 mismatches) and identify a graph-based, fixture-independent rewrite.
- [x] Recheck four Rank MLP public outputs against saved, SHA-matched NPU
  evidence: 12 comparisons / 5,505,024 INT16 elements, zero mismatches.
- [x] Audit Gate/Up INT4 rank weight slices, bias, quantization parameters,
  and Gate FC -> Dequantize -> Slice topology.
- [ ] Merge the Phase 3C Gate zero-padding rewrite, its bounded rebuild
  integration, and synthetic regression tests after public CI and review.
- [ ] Restore authorized EVK SSH and confirm an idle device; rerun a controlled
  NPU invocation of the current SHA-bound model and fixture.
- [ ] Obtain a trustworthy, real-token-prompt-derived fixture with nonzero
  canonical MLP and Down outputs; preserve its provenance.
- [ ] Validate the **integrated P128 Layer0** across Rank0-Rank3 and all
  required Attention, O, Residual, Norm, MLP, Down, Hidden, and KV boundaries.
- [ ] Run D1 Layer0 and then P128/D1 Full32 on all required output elements
  with matching canonical NPU references.
- [ ] Measure padding overhead and NPU latency under controlled conditions.
- [ ] Validate distributed transport and end-to-end inference separately
  from individual graph partitioning correctness.

## Release gate

Keep the release marked Experimental until model-bound host checks and
full-output NPU exactness pass for the documented profiles. Compiler acceptance
and unit tests alone do not satisfy this gate.