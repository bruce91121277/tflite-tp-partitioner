# Engineering handoff

This repository partitions supported W4A16 per-layer TFLite graphs into TP2 and TP4 components. It does not include model files, the vendor compiler, runtime libraries, or device captures. The project is experimental; see the [README](../README.md).

## Code map

- `src/tflite_tp/cli.py`: discovery, partitioning, and pipeline commands.
- `discovery.py`, `analysis.py`, `planning.py`, and `partition.py`: graph recognition and partition plans.
- `tflite_rewriter.py` and `packed_int4.py`: TFLite rewriting and weight slicing.
- `golden.py`, `pipeline.py`, and `release.py`: comparison and release checks.
- `backends/mediatek/`: optional device verification backend.
- `tests/`: synthetic software tests.

## Development

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
tflite-tp --help
tflite-tp-verify --help
```

See the [README](../README.md) for partitioning commands and source requirements. Prefill and Decode sources must already have the requested fixed tensor shapes. Model conversion and quantization require a separate toolchain.

## Verification and next work

Structural, compiler, host, device and distributed checks cover different behavior. Record the model and artifact hashes, profile, input and cache provenance, toolchain, device, and scope for each result. A compiler or unit-test pass does not establish full-model behavior.

Priority work includes extending graph support and TP degrees, supporting more model geometries and source profiles. See [roadmap](roadmap.md) and [verification guidance](verification.md). Model files and private captures are not included.
