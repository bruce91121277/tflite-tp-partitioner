# Known limitations

- Release status is Experimental.
- TP2 and TP4 structural graph partitioning is implemented for supported
  W4A16 TFLite graph patterns.
- Current profiles and graph patterns are tailored to Llama 3.1 8B W4A16 TFLite exports. Supported projection and operator forms are intentionally narrow. Unknown
  encodings and ambiguous graph roles fail closed.
- Multiple-subgraph TFLite models are unsupported.
- Vendor custom operators need a compatible host interpreter for host Golden
  execution. That interpreter is not installed or included by this package.
- TP4 NPU results include an observed case with 117 INT16 mismatches. Phase
  3C is investigating FP32 Dequantize differences.
- TP2 row-reduce numerical exactness is incomplete. Reduction arithmetic and
  requantization details remain unproven.
- P128 and D1 Full32 NPU Exact have not been verified.
- Hardware, compiler, and distributed-runtime checks are not available in
  ordinary GitHub Actions.