# Verification follow-up

The partitioner includes structural checks and tools for host and device comparisons. Full-model behavior and distributed execution require separate validation with matching source artifacts, inputs and cache state.

Prior investigations found differences in some decode and attention cases, and the TP2 row-reduce arithmetic contract remains unproven. Investigate a reproducible first divergence with matching source, artifact, input, and runtime provenance before drawing conclusions about a new candidate.

See [verification guidance](numerical-verification.md) and [Issue #1](https://github.com/bruce91121277/tflite-tp-partitioner/issues/1) for context.
