# Verification guidance

Structural checks, host comparisons, compiler checks, device comparisons, and distributed execution each verify a different part of the pipeline. A passing check does not establish the results of the others.

For reproducibility, record source and generated artifact hashes, input and cache provenance, tensor contracts, profile, compiler and runtime versions, and the scope of each comparison. Report an unperformed comparison as not run.

Historical fixture-scoped observations are recorded in the existing [host evidence](m7-tp4-layer0-host-evidence-20261027.md), [device evidence](m9-tp4-layer0-npu-evidence-20261027.md), and [MLP evidence](phase3c-mlp-npu-evidence.md). They do not establish full-model or distributed behavior. Additional maintainer-reported investigations are tracked in [Issue #1](https://github.com/bruce91121277/tflite-tp-partitioner/issues/1).
