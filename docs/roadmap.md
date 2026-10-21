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

- Complete Phase 3C analysis of FP32 Dequantize differences.
- Resolve the observed 117 INT16 mismatch case and add a minimized regression
  test.
- Verify P128 and D1 Full32 NPU behavior across all required outputs.
- Confirm distributed transport and end-to-end execution separately from
  graph partitioning.

## Release gate

Keep the release marked Experimental until model-bound host checks and
full-output NPU exactness pass for the documented profiles. Compiler acceptance
and unit tests alone do not satisfy this gate.