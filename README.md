# TFLite Tensor Parallel Partitioner

**Release status: Experimental.** This project creates TP2, TP4 and TP8 structural
partitions from supported W4A16 TFLite graphs. Numerical release readiness has
not been established.

## What is implemented

- Graph discovery and validation for transformer projection graphs.
- TP2, TP4 and TP8 component planning and graph rewriting.
- Fixed-token Prefill `P<n>` and Decode `D<n>` source profiles, including indexed layers.
- `build-tail` vocabulary partitioning for TP2, TP4 and TP8.
- Four-stage All-Gather for TP2, TP4 and TP8.
- Packed INT4 weight slicing with quantization metadata rewriting.
- An opt-in experimental TP4 P128 MLP Gate width rewrite that can generate structural
  candidates for supported layers. Numerical verification remains SHA-bound
  and component-scoped; candidate generation does not mark NPU Exact.
- An opt-in TP4 P128 Layer0 attention-context rewrite. It locates the
  INT16 BatchMatMul from tensor contracts and its unique ADD consumer, and
  fails closed when the graph match is ambiguous.
- Golden comparison, bounded automatic repair, and strict release gates.
- A MediaTek NPU verification backend with input adaptation, exact comparison,
  output stitching, and report generation.
- Synthetic unit tests and a GitHub Actions workflow.

The deliverable is partitioned TFLite. DLA compilation and board runtime are outside this repository.
The partitioner does not include models, generated TFLite files, DLA binaries,
vendor SDK files, Neuron Runtime libraries, or internal NPU captures.

## Partition layout

The default All-Gather layout contains six components per layer and four All-Gather boundaries.

| Projection | TP2 / TP4 / TP8 split |
| --- | --- |
| Q | Output channel (column) |
| K | Output channel (column) |
| V | Output channel (column) |
| O | Output channel (column) |
| Gate | Output channel (column) |
| Up | Output channel (column) |
| Down | Output channel (column) |

TP2, TP4 and TP8 use the same All-Gather topology. MLP Gate padding and the experimental attention-context rewrite are opt-in and disabled by default. Q/K/V padding defaults to the selected rule for P128 and D1; other profiles require explicit opt-in.

## Install

Requires Python 3.10 or newer.

    python -m venv .venv
    python -m pip install .
    tflite-tp --help
    tflite-tp-verify --help

On Windows PowerShell, activate the environment with:

    .\.venv\Scripts\Activate.ps1

The core package depends on NumPy, FlatBuffers, and the TFLite schema package.
The MediaTek backend uses system OpenSSH commands and tooling supplied by the
developer or device vendor; it does not install or bundle proprietary runtime
libraries.

## Build partitions

The partitioner starts from supported **per-layer W4A16 TFLite graphs**. It does
not download model checkpoints, run PTQ, convert PyTorch/Safetensors, or export
layer TFLite files.

Prepare authorized per-layer W4A16 TFLite source files using a suitable external conversion and quantization toolchain.

### Discover and build candidates

Start with one layer to check graph support, then build a full profile into a
new output directory:

    tflite-tp discover --source ./canonical/D1/layer_00.tflite --expected-layers 1 --report ./audit/d1-layer0.json
    tflite-tp build-tp4 --source ./canonical/D1/layer_00.tflite --output ./out/tp4-d1-layer0 --profile D1 --expected-layers 1

For a 32-layer source directory, repeat separately for D1 and P128:

    tflite-tp discover --source ./canonical/D1 --expected-layers 32 --report ./audit/d1.json
    tflite-tp build-tp4 --source ./canonical/D1 --output ./out/tp4-d1 --profile D1 --expected-layers 32
    tflite-tp discover --source ./canonical/P128 --expected-layers 32 --report ./audit/p128.json
    tflite-tp build-tp4 --source ./canonical/P128 --output ./out/tp4-p128 --profile P128 --expected-layers 32

Use a fresh output path for each build; existing output directories are
rejected. Review the source and candidate SHA manifests and graph coverage
before using the generated components.

### Verify the candidates

A successful structural build only means that a supported graph was rewritten.
It does not establish Host Golden equality, NCC acceptance, NPU numerical
exactness, or a distributed TP4 runtime. Run those as separate gates with
matching Canonical inputs and fixtures. NCC, Neuron Runtime, and Genio hardware
are supplied separately; they are not installed by this package. Sequential
rank execution with Host stitching is not a four-device collective.

## Known limitations

- Experimental support is limited to the graph patterns and W4A16 INT4
  encodings recognized by the current discovery and rewrite rules. Current
  profiles and graph patterns are tailored to Llama 3.1 8B W4A16 TFLite exports.
- The experimental attention-context head-padding rewrite is limited to TP4
  P128 Layer0 with expected_layers=1.
- Multiple-subgraph TFLite files are unsupported.
- Unsupported operators, ambiguous projection roles, and incomplete graph
  coverage fail closed.
- Host Golden execution of vendor custom operators requires a compatible
  interpreter supplied separately.
- The MediaTek backend requires a separately configured compiler and device;
  hardware verification cannot run in ordinary CI.

See the [roadmap](docs/roadmap.md).

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

See [the engineering handoff](docs/maintainer-handoff.md), CONTRIBUTING.md and SECURITY.md. Do not submit model weights, TFLite models, DLA binaries,
private captures, credentials, device addresses, or vendor SDK/runtime files.
