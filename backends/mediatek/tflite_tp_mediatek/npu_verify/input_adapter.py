"""Build rank-local NPU inputs from SHA-verified Canonical fixture data."""

import json
import math
from pathlib import Path

import numpy as np

from .contracts import DTYPE_NAMES, load_tflite_contract, normalize_dtype, read_json, sha256_file


def _dtype(value):
    name = normalize_dtype(value)
    return np.dtype(DTYPE_NAMES[name])


def _quant_equal(left, right):
    def normalized(value):
        if value is None:
            return None
        if not value.get("scales") and not value.get("zero_points"):
            return None
        return {key: value.get(key) for key in ("scales", "zero_points", "quantized_dimension")}

    left = normalized(left)
    right = normalized(right)
    if left is None or right is None:
        return left is None and right is None
    for key in ("scales", "zero_points", "quantized_dimension"):
        if left.get(key) != right.get(key):
            return False
    return True


def _slice_contract_quantization(contract, source_shape, axis, start, end):
    result = dict(contract)
    quant = contract.get("quantization")
    if not quant:
        return result
    quant = dict(quant)
    scales = list(quant.get("scales", []))
    zero_points = list(quant.get("zero_points", []))
    qdim = int(quant.get("quantized_dimension", 0))
    if len(scales) > 1:
        if qdim != axis or len(scales) != int(source_shape[axis]):
            raise ValueError("INPUT_PER_AXIS_QUANTIZATION_SLICE_UNSUPPORTED")
        quant["scales"] = scales[start:end]
        if len(zero_points) > 1:
            if len(zero_points) != int(source_shape[axis]):
                raise ValueError("INPUT_ZERO_POINT_SLICE_UNSUPPORTED")
            quant["zero_points"] = zero_points[start:end]
    result["quantization"] = quant
    return result


def _verify_file(path, row, dtype_name, shape):
    if not path.is_file():
        raise ValueError("CANONICAL_INPUT_FILE_MISSING: " + str(path))
    expected_bytes = math.prod(shape) * _dtype(dtype_name).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError("CANONICAL_INPUT_BYTE_COUNT_MISMATCH: " + path.name)
    actual_sha = sha256_file(path)
    expected_sha = row.get("binary_sha256", row.get("sha256", row.get("actual_sha256")))
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError("CANONICAL_INPUT_BINARY_SHA_MISMATCH: " + path.name)
    return actual_sha


def prepare_rank_inputs(
    candidate_model,
    canonical_model,
    canonical_manifest,
    input_export_manifest,
    canonical_input_dir,
    canonical_capture_manifest,
    canonical_output_dir,
    rank,
    tp,
    output_dir,
):
    """Resolve input tensors by graph names and sealed contracts, never by shape alone."""
    candidate = load_tflite_contract(candidate_model)
    canonical = load_tflite_contract(canonical_model)
    npu_manifest = read_json(canonical_manifest)
    export = read_json(input_export_manifest)
    capture = read_json(canonical_capture_manifest)
    rank, tp = int(rank), int(tp)
    if tp not in (2, 4) or rank < 0 or rank >= tp:
        raise ValueError("TP_RANK_CONTRACT_INVALID")
    fixture = npu_manifest.get("fixture", {})
    export_fixture_sha = export.get("fixture_sha256")
    archive_sha = export.get("source_archive_sha256")
    expected_archive_sha = fixture.get("input_archive", {}).get("sha256")
    if export_fixture_sha:
        if export_fixture_sha != fixture.get("fixture_sha256"):
            raise ValueError("FIXTURE_SHA_MISMATCH")
        fixture_binding_method = "FIXTURE_SHA256"
    elif archive_sha and archive_sha == expected_archive_sha:
        # Legacy Phase 2G export manifests predate an explicit fixture_sha256;
        # bind them through the immutable input archive SHA in the sealed bundle.
        fixture_binding_method = "SEALED_INPUT_ARCHIVE_SHA256"
    else:
        raise ValueError("FIXTURE_SHA_MISMATCH")
    if capture.get("fixture_sha256") != fixture.get("fixture_sha256"):
        raise ValueError("CAPTURE_FIXTURE_SHA_MISMATCH")
    if capture.get("source_model_sha256") != npu_manifest.get("source_canonical_sha256"):
        raise ValueError("CANONICAL_SOURCE_SHA_MISMATCH")
    canonical_spec = npu_manifest.get("models", {}).get("canonical", {})
    if canonical["sha256"] != canonical_spec.get("sha256"):
        raise ValueError("CANONICAL_TFLITE_SHA_MISMATCH")
    if export.get("source_archive_sha256") != fixture.get("input_archive", {}).get("sha256"):
        raise ValueError("INPUT_ARCHIVE_SHA_MISMATCH")
    if capture.get("canonical_tflite_sha256") != canonical["sha256"]:
        raise ValueError("CAPTURE_TFLITE_SHA_MISMATCH")

    canonical_inputs = {x["name"]: x for x in canonical_spec.get("inputs", [])}
    input_exports = {x["name"]: x for x in export.get("inputs", [])}
    canonical_outputs = {x["name"]: x for x in canonical_spec.get("outputs", [])}
    capture_outputs = {x["semantic_tensor"]: x for x in capture.get("outputs", [])}
    graph_inputs_by_name = {x["name"]: x for x in canonical["inputs"]}
    if len(canonical_inputs) != len(canonical["inputs"]):
        raise ValueError("CANONICAL_INPUT_MANIFEST_INCOMPLETE")
    for spec in canonical_inputs.values():
        actual = graph_inputs_by_name.get(spec["name"])
        if actual is None or actual["shape"] != spec["shape"] or actual["dtype"] != normalize_dtype(spec["dtype"]):
            raise ValueError("CANONICAL_INPUT_GRAPH_CONTRACT_MISMATCH: " + spec["name"])

    canonical_input_root = Path(canonical_input_dir)
    canonical_output_root = Path(canonical_output_dir)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("OUTPUT_DIRECTORY_NOT_EMPTY")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, target in enumerate(candidate["inputs"]):
        name = target["name"]
        if name in canonical_inputs:
            source_spec = canonical_inputs[name]
            export_row = input_exports.get(name)
            if export_row is None:
                raise ValueError("CANONICAL_INPUT_MAPPING_MISSING: " + name)
            if int(export_row["handle"]) < 0:
                raise ValueError("CANONICAL_INPUT_HANDLE_INVALID: " + name)
            source_path = canonical_input_root / "input{}.bin".format(int(export_row["handle"]))
            source_contract = {
                "name": name,
                "shape": [int(x) for x in source_spec["shape"]],
                "dtype": normalize_dtype(source_spec["dtype"]),
                "quantization": source_spec.get("quantization"),
            }
            _verify_file(source_path, export_row, source_contract["dtype"], source_contract["shape"])
            origin = "CANONICAL_PUBLIC_INPUT"
        elif name in canonical_outputs:
            source_spec = canonical_outputs[name]
            capture_row = capture_outputs.get(name)
            if capture_row is None:
                raise ValueError("BLOCKED_MISSING_STAGE_INPUT: " + name)
            handle = int(capture_row["handle_index"])
            source_path = canonical_output_root / "output{}.bin".format(handle)
            source_contract = {
                "name": name,
                "shape": [int(x) for x in source_spec["shape"]],
                "dtype": normalize_dtype(source_spec["dtype"]),
                "quantization": source_spec.get("quantization"),
            }
            _verify_file(source_path, capture_row, source_contract["dtype"], source_contract["shape"])
            origin = "CANONICAL_NPU_PUBLIC_OUTPUT"
        else:
            raise ValueError("BLOCKED_MISSING_STAGE_INPUT: " + name)

        source_shape = tuple(source_contract["shape"])
        target_shape = tuple(target["shape"])
        source_dtype = source_contract["dtype"]
        target_dtype = target["dtype"]
        if source_dtype != target_dtype:
            raise ValueError("INPUT_DTYPE_MISMATCH: " + name)
        q_contract = source_contract.get("quantization")
        if q_contract is not None:
            q_contract = {
                "scales": [float(x) for x in q_contract.get("scales", [])],
                "zero_points": [int(x) for x in q_contract.get("zero_points", [])],
                "quantized_dimension": int(q_contract.get("quantized_dimension", 0)),
            }
        mapped_contract = dict(source_contract, quantization=q_contract)
        if source_shape == target_shape:
            adapted_contract = mapped_contract
            raw = source_path.read_bytes()
            adaptation = "IDENTITY_BY_SEMANTIC_NAME"
        else:
            differences = [axis for axis, (src, dst) in enumerate(zip(source_shape, target_shape)) if src != dst]
            if len(source_shape) != len(target_shape) or len(differences) != 1:
                raise ValueError("INPUT_SHAPE_MISMATCH: " + name)
            axis = differences[0]
            if name not in ("past_keys_0", "past_values_0") or axis != 1:
                raise ValueError("INPUT_ADAPTER_UNSUPPORTED_SHARD: " + name)
            if source_shape[axis] != target_shape[axis] * tp:
                raise ValueError("KV_HEAD_PARTITION_COUNT_MISMATCH: " + name)
            start, end = rank * target_shape[axis], (rank + 1) * target_shape[axis]
            array = np.frombuffer(source_path.read_bytes(), dtype=_dtype(source_dtype)).reshape(source_shape)
            selector = [slice(None)] * len(source_shape)
            selector[axis] = slice(start, end)
            raw = np.ascontiguousarray(array[tuple(selector)]).tobytes(order="C")
            adapted_contract = _slice_contract_quantization(mapped_contract, source_shape, axis, start, end)
            adaptation = "KV_HEAD_SLICE"
        if tuple(target["shape"]) != tuple(adapted_contract["shape"]):
            adapted_contract["shape"] = list(target_shape)
        if not _quant_equal(target.get("quantization"), adapted_contract.get("quantization")):
            raise ValueError("INPUT_QUANTIZATION_MISMATCH: " + name)
        expected_bytes = math.prod(target_shape) * _dtype(target_dtype).itemsize
        if len(raw) != expected_bytes:
            raise ValueError("ADAPTED_INPUT_BYTE_COUNT_MISMATCH: " + name)
        target_path = out / "input{}.bin".format(index)
        target_path.write_bytes(raw)
        rows.append({
            "handle": index,
            "semantic_tensor": name,
            "shape": list(target_shape),
            "dtype": target_dtype,
            "quantization": target.get("quantization"),
            "bytes": len(raw),
            "sha256": sha256_file(target_path),
            "source": origin,
            "source_handle": int(export_row["handle"]) if name in canonical_inputs else int(capture_row["handle_index"]),
            "source_sha256": sha256_file(source_path),
            "adaptation": adaptation,
        })
    manifest = {
        "schema": "tflite_tp.phase3.npu-input-adapter.v1",
        "status": "PASS_CONTRACT_AND_SHA_VERIFIED",
        "profile": npu_manifest["profile"],
        "fixture_id": fixture.get("label"),
        "fixture_sha256": fixture.get("fixture_sha256"),
        "fixture_binding_method": fixture_binding_method,
        "source_input_archive_sha256": archive_sha,
        "tp_degree": tp,
        "rank": rank,
        "candidate_tflite_sha256": candidate["sha256"],
        "canonical_tflite_sha256": canonical["sha256"],
        "source_canonical_sha256": capture["source_model_sha256"],
        "input_export_manifest_sha256": sha256_file(input_export_manifest),
        "canonical_capture_manifest_sha256": sha256_file(canonical_capture_manifest),
        "inputs": rows,
    }
    (out / "INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "SHA256SUMS").write_text(
        "".join("{}  input{}.bin\n".format(row["sha256"], row["handle"]) for row in rows), encoding="ascii"
    )
    return manifest
