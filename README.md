# TFLite Tensor Parallel Partitioner

**Release status: Experimental.** This project creates TP2 and TP4 structural
partitions from supported W4A16 TFLite graphs. Numerical release readiness has
not been established.

## Engineering handoff

For maintainers, see the [engineering handoff](docs/maintainer-handoff.md).

## What is implemented

- Graph discovery and validation for transformer projection graphs.
- TP2 and TP4 component planning and graph rewriting.
- Packed INT4 weight slicing with quantization metadata rewriting.
- An experimental TP4 P128 MLP Gate width rewrite that can generate structural
  candidates for supported layers. Numerical verification remains SHA-bound
  and component-scoped; candidate generation does not mark NPU Exact.
- An opt-in TP4 P128 Layer0 attention-context rewrite. It locates the
  INT16 BatchMatMul from tensor contracts and its unique ADD consumer, and
  fails closed when the graph match is ambiguous.
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

## From a downloaded checkpoint to TP4 partitions

The partitioner starts from supported **per-layer W4A16 TFLite graphs**. It does
not download model checkpoints, run PTQ, convert PyTorch/Safetensors, or export
layer TFLite files.

### 1. Download a checkpoint you are authorized to use

For a Hugging Face model, install the Hub client, authenticate locally, and
pin a model revision so the source can be reproduced:

    python -m pip install -U huggingface_hub
    hf auth login
    hf download meta-llama/Llama-3.1-8B-Instruct --revision REPLACE_WITH_COMMIT_SHA --local-dir ./models/llama31

Use the repository and revision you are authorized to access. Gated models may
require accepting the model license and receiving access first. Keep access
tokens local; do not put them in scripts, logs, or this repository.

### 2. Convert and quantize to per-layer TFLite

Use a converter and PTQ recipe that support the model and the intended target.
For Genio 720, obtain the compatible MediaTek Converter/SDK separately and
follow its supported W4A16 export flow. This repository does not include that
proprietary toolchain, calibration data, or conversion recipe.

Export one Canonical TFLite graph per transformer layer for each profile you
intend to partition. Keep P128 and D1 in separate directories, and record the
model revision, quantization configuration, calibration lineage, converter
version, input/output tensor contracts, and SHA-256 values. A downloaded
checkpoint or a single full-model TFLite file is not automatically a
per-layer source set accepted by this CLI.

Example source layout:

    canonical/
      P128/  # one supported .tflite source graph per layer
      D1/    # one supported .tflite source graph per layer

### 3. Discover and build TP4 candidates

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

### 4. Verify the candidates on the target

A successful structural build only means that a supported graph was rewritten.
It does not establish Host Golden equality, NCC acceptance, NPU numerical
exactness, or a distributed TP4 runtime. Run those as separate gates with
matching Canonical inputs and fixtures. NCC, Neuron Runtime, and Genio hardware
are supplied separately; they are not installed by this package. Sequential
rank execution with Host stitching is not a four-device collective.

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
