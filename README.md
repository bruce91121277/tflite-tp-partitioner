# TFLite Tensor Parallel Partitioner

**Release status: Experimental.** This project creates TP2 and TP4 structural
partitions from supported W4A16 TFLite graphs. Numerical release readiness has
not been established.

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

The project has **not** passed a full-layer or Full32 NPU exactness gate.

### M7 TP4 status (2026-10-27)

- TP2 and TP4 structural partitioning are implemented for supported graphs.
- An isolated TP4 P128 Layer0 Host variant matched its fixture after matching
  the rank-local current-token attention BatchMatMul head-batch shape to
  Canonical. The public builder now emits this rewrite as an opt-in rule, but
  its latest four attention artifact hashes passed structural validation and
  NCC 8.2.22 compilation. They have not been run through Host, so the prior
  exact result does not transfer to those files. The latest Host runner could
  not start: the available RTX shell had no `python` launcher, and its
  `python3` lacked FlatBuffers and MTK Converter. No model invocation occurred.
  The rule is limited to one TP4 P128 Layer0 source and is not an NPU Exact
  claim.
- A D1 Layer0 Host component-stitch fixture also matched exactly. Its result
  does not establish integrated D1 public-I/O or NPU exactness.
- NCC compiled two reused Canonical files and all 48 current TP4
  rank-component SHA values for the recorded target, including the latest
  four P128 attention replacements. Compilation success does not imply NPU
  Exact.
- **TP4 NPU still has an observed 117 INT16 mismatch** in the P128 Layer0 MLP
  comparison. Phase 3C is investigating the FP32 Dequantize difference; the
  Host attention rewrite does not resolve that issue.
- TP2 Row-Reduce Numerical Exact is incomplete. P128/D1 Full32 NPU Exact has
  not been verified. The EVK NPU comparison for these M7 candidates was not
  run, so no M7 NPU Exact result is claimed.
- **Release status remains Experimental.** See the [M7 Host evidence note](docs/m7-tp4-layer0-host-evidence-20261027.md).

- **M1–M5 Linux source validation (2026-10-22):** an independent SHA-pinned
  development bundle passed **108/108 tests, zero skipped**, on RTX4090
  Linux; the full local suite passed 111. Installation, both CLI help
  commands, comparator self-test, and compileall also passed. **No NCC/NPU
  execution occurred in this checkpoint.** The full real-model Golden is
  **NOT_RUN** because a compatible Canonical MTK custom-op Host Executor
  is not registered; TP2 O/Down row-reduce semantics remain unproven.
  `MISMATCH_COUNT=null` is not an Exact PASS. This checkpoint's
  `RELEASE_GATE=FAIL` / `REAL_MODEL_NOT_VERIFIED`. See
  [M1–M5 Linux checkpoint](docs/m1-m5-linux-checkpoint-20261022.md).

- Archived Phase 3C evidence recorded 12 saved comparisons across four ranks
  and three fixtures: 5,505,024 INT16 MLP output elements, zero mismatches.
  Those comparisons are tied to their archived candidate hashes and were not
  fresh invokes in the Phase 3C audit.
- Phase 3D ran fresh Genio 720 NPU component comparisons from a real-text
  prompt fixture. P128 Layer0 had 36 component comparisons / 9,961,472
  elements with zero mismatches; D1 Layer0 had 36 / 77,824 with zero
  mismatches. These are component stitches, not full-layer public-I/O proof.
- A fresh TP4 P128 Layer1 MLP candidate comparison used the full32 padded
  candidate: four ranks, 1,835,008 INT16 elements, zero mismatches. Its input
  was derived from the Layer0 stitched hidden state. The Canonical reference
  graph had stage outputs exposed; equivalence of that instrumented graph to
  the uninstrumented compiler path is not established. This result does not
  prove the whole layer or Full32.
- The latest full32 unpadded Layer0 MLP package comparison had 363 mismatches
  in 1,835,008 elements, maximum difference 4; the first was
  `[0, 2, 1382]` (canonical 20, rank-local 21).
- A structural TP4 P128 build now emits the Gate width rewrite for all 32
  layers and four ranks. Only the specific tested component outputs have NPU
  comparison evidence; every generated candidate SHA needs its own retest.
  D1 Gate padding and TP2 row-reduce remain unverified.
- The P128 and D1 fixtures were built from an authored prompt and a verified
  Llama 3.1 tokenizer/embedding asset. The upstream checkpoint revision is
  missing from the source metadata. The P128 result is at the Layer0 input
  boundary; the D1 fixture uses the prompt-derived KV state and current token.
  This does not establish full-model prompt-to-output exactness.

The partitioner retains rank-specific INT4 weight slices and zero-only padding
for the tested Gate rewrite. The rewrite does not change the release gate:
full P128/D1 Layer0 and Full32 exactness, TP2 row-reduce semantics, and
distributed-runtime verification remain open. Release remains **BLOCKED**.

Passing unit tests, FlatBuffer checks, or NCC compilation does not imply NPU
numerical equality. See [the numerical verification guide](docs/numerical-verification.md).

## Known limitations

- Experimental support is limited to the graph patterns and W4A16 INT4
  encodings recognized by the current discovery and rewrite rules. Current
  profiles and graph patterns are tailored to Llama 3.1 8B W4A16 TFLite exports.
- The experimental attention-context head-padding rewrite is limited to TP4
  P128 Layer0 with expected_layers=1. It matched one Host fixture; every new
  candidate SHA still needs same-backend NPU verification.
- Multiple-subgraph TFLite files are unsupported.
- Unsupported operators, ambiguous projection roles, and incomplete graph
  coverage fail closed.
- Host Golden execution of vendor custom operators requires a compatible
  interpreter supplied separately.
- The experimental MLP Gate rewrite is structural until its generated
  candidate SHA receives a fresh same-backend NPU comparison.
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
