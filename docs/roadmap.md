# Roadmap

- Extend recognition and partitioning to additional supported Llama graph shapes and TP degrees.
- Integrate per-rank DLA batch compilation and deployment with the existing partitioning workflow.
- Verify chained Prefill and Decode execution with matching layer-local cache state.
- Validate distributed execution and text generation on the target devices, then measure end-to-end performance.
- Keep source, output, compiler and runtime provenance with each result.

The current supported scope is described in the [README](../README.md) and [known limitations](known-limitations.md). Historical evidence is listed in the [verification guidance](numerical-verification.md).
