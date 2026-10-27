# Known limitations

- Release status is Experimental.
- TP2 and TP4 structural graph partitioning is implemented for supported
  W4A16 TFLite graph patterns.
- Current profiles and graph patterns are tailored to Llama 3.1 8B W4A16 TFLite exports. Supported projection and operator forms are intentionally narrow. Unknown
  encodings and ambiguous graph roles fail closed.
- Multiple-subgraph TFLite models are unsupported.
- Vendor custom operators need a compatible host interpreter for host Golden
  execution. That interpreter is not installed or included by this package.
- TP4 P128 Layer0 has an observed MLP NPU comparison with 117 INT16
  mismatches. Phase 3C is investigating the FP32 Dequantize difference; the
  mismatch remains unresolved. Archived Phase 3C zero-mismatch comparisons
  covered separate candidate/component evidence and are not a fresh M7 NPU
  result.
- The M7 TP4 P128 Layer0 Host attention component stitch matched exactly after
  matching the local current-token BatchMatMul head-batch shape to Canonical.
  The latest builder-emitted attention hashes are different and remain
  Host-unverified. This is one prior Host fixture and not an NPU, chained
  runtime, or Full32 proof. Every rewritten candidate SHA needs same-backend
  Host and NPU verification.
- The attention-context rewrite currently recognizes only the TP4 P128 Layer0
  graph contract: one INT16 `[1,8,128,128]` BatchMatMul with a unique direct
  ADD consumer, plus compatible MTKEXT_TILE and SLICE templates. Other shapes,
  profiles, or ambiguous matches fail closed.
- M7 did not invoke the NPU because the EVK had an existing runtime active.
  NPU behavior of the current P128 and D1 candidates remains unverified.
- TP2 row-reduce numerical exactness is incomplete. Reduction arithmetic and
  requantization details remain unproven.
- P128 and D1 Full32 NPU Exact have not been verified.
- Hardware, compiler, and distributed-runtime checks are not available in
  ordinary GitHub Actions.
