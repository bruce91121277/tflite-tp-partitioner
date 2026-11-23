# Known limitations

- Release status is Experimental.
- TP2 and TP4 structural graph partitioning is implemented for supported
  W4A16 TFLite graph patterns.
- Current profiles and graph patterns are tailored to Llama 3.1 8B W4A16 TFLite exports. Supported projection and operator forms are intentionally narrow. Unknown
  encodings and ambiguous graph roles fail closed.
- Multiple-subgraph TFLite models are unsupported.
- Vendor custom operators need a compatible host interpreter for host Golden
  execution. That interpreter is not installed or included by this package.
- The attention-context rewrite currently recognizes only the TP4 P128 Layer0
  graph contract: one INT16 `[1,8,128,128]` BatchMatMul with a unique direct
  ADD consumer, plus compatible MTKEXT_TILE and SLICE templates. Other shapes,
  profiles, or ambiguous matches fail closed.
- Hardware, compiler, and distributed-runtime checks are not available in
  ordinary GitHub Actions.
