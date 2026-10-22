# Known limitations

- Release status is Experimental.
- TP2 and TP4 structural graph partitioning is implemented for supported
  W4A16 TFLite graph patterns.
- Current profiles and graph patterns are tailored to Llama 3.1 8B W4A16 TFLite exports. Supported projection and operator forms are intentionally narrow. Unknown
  encodings and ambiguous graph roles fail closed.
- Multiple-subgraph TFLite models are unsupported.
- Vendor custom operators need a compatible host interpreter for host Golden
  execution. That interpreter is not installed or included by this package.
- TP4 P128 Layer0 Rank0 MLP originally had 117 INT16 output mismatches.
  A Phase 3C development rewrite achieves zero mismatches for four Rank MLP
  public outputs across 12 saved SHA-matched NPU comparisons (5,505,024
  elements); this does not validate full Layer0, D1, or Full32. The tested
  rewrite source has not yet been synchronized into public main.
- The Phase 3C report reuses archived NPU captures (zero fresh invokes);
  a real-prompt nondegenerate fixture, fresh NPU invocation, and full-stage
  integration remain outstanding. NPU microkernel causality is unconfirmed.
- TP2 row-reduce numerical exactness is incomplete. Reduction arithmetic and
  requantization details remain unproven.
- P128 and D1 Full32 NPU Exact have not been verified.
- Hardware, compiler, and distributed-runtime checks are not available in
  ordinary GitHub Actions.