# TFLite Tensor Parallel Partitioner

**Release status: Experimental.** This project creates TP2 and TP4 structural
partitions from supported W4A16 TFLite graphs. Numerical release readiness has
not been established.

## What is implemented

- Graph discovery and validation for transformer projection graphs.
- TP2 and TP4 component planning and graph rewriting.
- Packed INT4 weight slicing with quantization metadata rewriting.
- Golden comparison, bounded automatic repair, and strict release gates.
- A Mediatek NPU verification backend with input adaptation, exact comparison,
  output stitching, and report generation.
- Synthetic unit tests and a GitHub Actions workflow.

The partitioner does not include models, generated TFLite files, DLA binaries,
vendor SDK files, Neuron Runtime libraries, or internal NPU captures.

## Install

Requires Python 3.10 or newer.

    python -m venv .venv
    python -m pip install .
    tflite-tp --help
    tflite-tp-verify --help

On Windows PowerShell, activate the environment with:

    .\.venv\Scripts\Activate.ps1

The core package depends on NumPy, FlatBuffers, and the TFLite schema package.
The Mediatek backend uses system OpenSSH commands and tooling supplied by the
developer or device vendor; it does not install or bundle proprietary runtime
libraries.

## Partition a graph

Pass your own supported per-layer TFLite input. This example writes a TP4
candidate for one D1 layer into a new directory:

    python examples/partition_graph.py .\path\to\layer.tflite .\out\tp4 --tp 4 --profile D1 --expected-layers 1

The example can also be run with a directory containing source graphs.
Existing output directories are rejected. For direct CLI use:

    tflite-tp build-tp4 --source ./models/layer.tflite --output ./out/tp4 --profile D1 --expected-layers 1
    tflite-tp build-tp2 --source ./models/layer.tflite --output ./out/tp2 --profile D1 --expected-layers 1

For graph discovery without rewriting:

    tflite-tp discover --source ./models/layer.tflite --expected-layers 1 --report ./discovery.json

Use only models you are authorized to process. The generated graph structure
does not imply that the model is numerically correct on a host interpreter or
an NPU.

## Numerical status

The project has **not** passed a full NPU exactness gate. A Phase 3C
development candidate has demonstrated a **component-scoped** improvement:

- **Unpadded baseline (TP4 P128 Layer0 Rank0 MLP):** 117 mismatches in
  458,752 INT16 public-output elements, with a maximum difference of one LSB.
- **Gate width rewrite (development candidate):** padding the rank-local
  INT16 Gate output with quantized zeros to twice the width, then Dequantizing
  and Slicing to the original rank width, produced **zero mismatches across
  12 saved NPU comparisons, 4 ranks, 3 fixtures, and 5,505,024 INT16 elements**.
- An independent FlatBuffer audit confirmed rank-specific Gate/Up INT4
  weight and bias slices, quantization metadata, zero padding, and
  FC -> Dequantize -> Slice. This is genuine rank-local weight partitioning.
- **Evidence scope:** public MLP output only. The latest Phase 3C audit
  recomputed saved, SHA-matched results; it performed **zero fresh NPU
  invocations**. Two of the three fixtures gave nonzero MLP outputs, but
  real-prompt nondegenerate MLP coverage is still missing.
- **Publication scope:** the Phase 3C automatic rewrite is currently in
  the local development worktree and has **not yet been synchronized into
  this public branch**. The public package is not a release of that fix.
- **Still blocked:** integrated Layer0, D1, P128/D1 Full32 NPU exactness,
  TP2 row-reduce numerical semantics, and multi-device runtime verification.
  The underlying NPU microkernel behavior is not confirmed.

See [the Phase 3C evidence summary](docs/phase3c-mlp-npu-evidence.md) and
[the numerical verification guide](docs/numerical-verification.md).

Unit tests exercise code paths with synthetic graphs, generated values, and
mocked device calls. Passing unit tests does not establish host Golden
equality or full-model NPU bit-exactness.

## Known limitations

- Experimental support is limited to the graph patterns and W4A16 INT4
  encodings recognized by the current discovery and rewrite rules. Current
  profiles and graph patterns are tailored to Llama 3.1 8B W4A16 TFLite exports.
- Multiple-subgraph TFLite files are unsupported.
- Unsupported operators, ambiguous projection roles, and incomplete graph
  coverage fail closed.
- Host Golden execution of vendor custom operators requires a compatible
  interpreter supplied separately.
- TP2 row-reduce arithmetic and full-model NPU exactness remain open.
- The Mediatek backend requires a separately configured compiler and device;
  hardware verification cannot run in ordinary CI.

See the known-limitations and TP2/TP4 roadmap documents.

## Tests

Run the synthetic test suite with:

    python -m unittest discover -s tests -v

The CI workflow installs the package, checks both CLI help commands, and runs
the same tests. These checks validate package structure and software behavior;
they do not certify NPU numerical equivalence.

## License

This project is licensed under the **Apache License, Version 2.0**.
See [LICENSE](LICENSE) for the full terms and [docs/license-status.md](docs/license-status.md)
for scope and third-party component notes. Model weights, MediaTek SDKs, and
other third-party tools are **not** relicensed by this project.

## Contributing

See CONTRIBUTING.md and SECURITY.md. Do not submit model weights, TFLite models, DLA binaries,
private captures, credentials, device addresses, or vendor SDK/runtime files.