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

The project has **not** passed a full NPU exactness gate:

- TP2 and TP4 structural partitioning is implemented.
- An observed TP4 NPU comparison still has 117 INT16 mismatches.
- Phase 3C is investigating differences in FP32 Dequantize behavior.
- TP2 row-reduce has not reached numerical exactness; accumulator width,
  reduction order, rounding, saturation, and requantization remain unproven.
- P128 and D1 Full32 NPU Exact have not been verified.

Unit tests exercise code paths with synthetic graphs, generated values, and
mocked device calls. A passing unit test suite does not establish host Golden
equality or NPU bit-exactness. See the
[docs/numerical-verification.md](docs/numerical-verification.md) for the evidence levels and known gaps.

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