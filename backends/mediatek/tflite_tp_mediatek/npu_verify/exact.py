"""Strict NPU output comparison and TP4 stitch helpers."""

import json
import math
from pathlib import Path

import numpy as np

from .contracts import DTYPE_NAMES, load_tflite_contract, normalize_dtype, read_json, sha256_file


def _numpy_dtype(value):
    return np.dtype(DTYPE_NAMES[normalize_dtype(value)])


def _expected_run_output(run_manifest, handle):
    rows = [x for x in run_manifest.get("outputs", []) if int(x.get("handle_index", -1)) == int(handle)]
    if len(rows) != 1:
        raise ValueError("RUN_OUTPUT_HANDLE_MAPPING_INVALID:{}".format(handle))
    row = rows[0]
    if row.get("status") != "PASS":
        raise ValueError("RUN_OUTPUT_NOT_VERIFIED:{}".format(handle))
    return row


def _load_run_tensor(contract, run_dir, run_manifest, handle):
    spec = contract["outputs"][handle]
    row = _expected_run_output(run_manifest, handle)
    path = Path(run_dir) / "output{}.bin".format(handle)
    if not path.is_file():
        raise ValueError("NPU_OUTPUT_MISSING:{}".format(path.name))
    expected_bytes = math.prod(spec["shape"]) * _numpy_dtype(spec["dtype"]).itemsize
    if path.stat().st_size != expected_bytes or int(row.get("bytes", -1)) != expected_bytes:
        raise ValueError("NPU_OUTPUT_BYTE_COUNT_MISMATCH:{}".format(path.name))
    actual_sha = sha256_file(path)
    if actual_sha != row.get("sha256"):
        raise ValueError("NPU_OUTPUT_SHA_MISMATCH:{}".format(path.name))
    array = np.frombuffer(path.read_bytes(), dtype=_numpy_dtype(spec["dtype"])).reshape(tuple(spec["shape"]))
    return array, path, row


def _quantized_slice(quantization, full_shape, axis, start, end):
    if quantization is None:
        return None
    result = dict(quantization)
    scales = list(result.get("scales", []))
    zero_points = list(result.get("zero_points", []))
    qdim = int(result.get("quantized_dimension", 0))
    if len(scales) > 1:
        if qdim != axis or len(scales) != int(full_shape[axis]):
            raise ValueError("PER_AXIS_QUANTIZATION_MAPPING_UNSUPPORTED")
        result["scales"] = scales[start:end]
        if len(zero_points) > 1:
            if len(zero_points) != int(full_shape[axis]):
                raise ValueError("PER_AXIS_ZERO_POINT_MAPPING_UNSUPPORTED")
            result["zero_points"] = zero_points[start:end]
    return result


def _quantization_equal(left, right):
    if left is None or right is None:
        return left is None and right is None
    return all(left.get(key) == right.get(key) for key in ("scales", "zero_points", "quantized_dimension"))


def compare_arrays(candidate, expected, dtype_name, tensor_name, candidate_quant=None, expected_quant=None, mapping=None):
    """Compare every element bit-exactly; tolerance is intentionally unsupported."""
    dtype_name = normalize_dtype(dtype_name)
    candidate = np.asarray(candidate)
    expected = np.asarray(expected)
    if candidate.shape != expected.shape:
        raise ValueError("COMPARE_SHAPE_MISMATCH:{}:{}:{}".format(tensor_name, candidate.shape, expected.shape))
    if candidate.dtype != expected.dtype:
        raise ValueError("COMPARE_DTYPE_MISMATCH:{}:{}:{}".format(tensor_name, candidate.dtype, expected.dtype))
    if not _quantization_equal(candidate_quant, expected_quant):
        raise ValueError("COMPARE_QUANTIZATION_MISMATCH:" + tensor_name)
    itemsize = max(1, candidate.dtype.itemsize)
    a_bytes = np.ascontiguousarray(candidate).view(np.uint8).reshape(-1, itemsize)
    e_bytes = np.ascontiguousarray(expected).view(np.uint8).reshape(-1, itemsize)
    different = np.any(a_bytes != e_bytes, axis=1)
    bad = np.flatnonzero(different)
    mismatch_count = int(bad.size)
    first = None
    max_abs_diff = 0
    if mismatch_count:
        flat = int(bad[0])
        index = tuple(int(x) for x in np.unravel_index(flat, candidate.shape))
        first = {
            "index": list(index),
            "expected": expected[index].item(),
            "actual": candidate[index].item(),
        }
        if dtype_name.startswith("FLOAT"):
            first["expected_bits_le"] = "0x{:0{}x}".format(
                int(np.ascontiguousarray(expected).reshape(-1)[flat:flat + 1].view(np.dtype("<u4"))[0]), 8
            ) if dtype_name == "FLOAT32" else None
            first["actual_bits_le"] = "0x{:0{}x}".format(
                int(np.ascontiguousarray(candidate).reshape(-1)[flat:flat + 1].view(np.dtype("<u4"))[0]), 8
            ) if dtype_name == "FLOAT32" else None
        delta = np.abs(candidate.astype(np.float64) - expected.astype(np.float64))
        if delta.size and np.isfinite(delta).all():
            maximum = float(delta.max())
            max_abs_diff = int(maximum) if dtype_name.startswith(("INT", "UINT")) else maximum
        else:
            max_abs_diff = None
    return {
        "semantic_tensor": tensor_name,
        "dtype": dtype_name,
        "shape": list(candidate.shape),
        "element_count": int(candidate.size),
        "mismatch_count": mismatch_count,
        "max_abs_diff": max_abs_diff,
        "first_failure": first,
        "mapping": mapping or {"kind": "IDENTITY"},
        "status": "EXACT" if mismatch_count == 0 else "MISMATCH",
    }


def compare_npu_runs(
    candidate_model,
    candidate_output_dir,
    candidate_run_manifest_path,
    reference_model,
    reference_output_dir,
    reference_run_manifest_path,
    tp,
    rank,
    aliases=None,
):
    """Compare candidate outputs to same-fixture Canonical NPU output tensors."""
    candidate = load_tflite_contract(candidate_model)
    reference = load_tflite_contract(reference_model)
    candidate_run = read_json(candidate_run_manifest_path)
    reference_run = read_json(reference_run_manifest_path)
    aliases = aliases or {}
    tp, rank = int(tp), int(rank)
    if candidate_run.get("fixture_sha256") != reference_run.get("fixture_sha256"):
        raise ValueError("FIXTURE_SHA_MISMATCH")
    if candidate_run.get("status") != "PASS":
        raise ValueError("CANDIDATE_NPU_RUN_NOT_PASS")
    reference_valid = reference_run.get("status") == "PASS" or (
        reference_run.get("status") == "CAPTURE_SHA_VERIFIED"
        and reference_run.get("reference_eligible") is True
    )
    if not reference_valid:
        raise ValueError("CANONICAL_NPU_REFERENCE_NOT_VERIFIED")
    if candidate_run.get("source_tflite_sha256") != candidate["sha256"]:
        raise ValueError("CANDIDATE_RUN_MODEL_SHA_MISMATCH")
    if reference_run.get("source_tflite_sha256") != reference["sha256"]:
        raise ValueError("REFERENCE_RUN_MODEL_SHA_MISMATCH")
    if not 0 <= rank < tp:
        raise ValueError("TP_RANK_CONTRACT_INVALID")

    reference_by_name = {row["name"]: index for index, row in enumerate(reference["outputs"])}
    comparisons = []
    uncovered = []
    for c_index, c_spec in enumerate(candidate["outputs"]):
        ref_name = aliases.get(c_spec["name"], c_spec["name"])
        if ref_name not in reference_by_name:
            uncovered.append({"candidate_tensor": c_spec["name"], "status": "NOT_COVERED_BY_CANONICAL_NPU"})
            continue
        r_index = reference_by_name[ref_name]
        r_spec = reference["outputs"][r_index]
        if c_spec["dtype"] != r_spec["dtype"]:
            raise ValueError("OUTPUT_DTYPE_CONTRACT_MISMATCH:" + c_spec["name"])
        candidate_array, candidate_path, _ = _load_run_tensor(candidate, candidate_output_dir, candidate_run, c_index)
        reference_array, reference_path, _ = _load_run_tensor(reference, reference_output_dir, reference_run, r_index)
        c_shape = tuple(c_spec["shape"])
        r_shape = tuple(r_spec["shape"])
        mapping = {"kind": "IDENTITY", "reference_tensor": ref_name, "reference_handle": r_index}
        expected = reference_array
        expected_quant = r_spec.get("quantization")
        if c_shape != r_shape:
            if len(c_shape) != len(r_shape):
                raise ValueError("OUTPUT_SHAPE_RANK_MISMATCH:" + c_spec["name"])
            axes = [axis for axis, (local, full) in enumerate(zip(c_shape, r_shape)) if local != full]
            if len(axes) != 1:
                raise ValueError("OUTPUT_SHARD_AXIS_AMBIGUOUS:" + c_spec["name"])
            axis = axes[0]
            if r_shape[axis] != c_shape[axis] * tp:
                raise ValueError("OUTPUT_SHARD_WIDTH_MISMATCH:" + c_spec["name"])
            start, end = rank * c_shape[axis], (rank + 1) * c_shape[axis]
            selector = [slice(None)] * len(r_shape)
            selector[axis] = slice(start, end)
            expected = np.ascontiguousarray(reference_array[tuple(selector)])
            expected_quant = _quantized_slice(expected_quant, r_shape, axis, start, end)
            mapping = {
                "kind": "RANK_SLICE_FROM_CANONICAL_NPU",
                "reference_tensor": ref_name,
                "reference_handle": r_index,
                "axis": axis,
                "start": start,
                "end": end,
                "tp_degree": tp,
                "rank": rank,
            }
        if tuple(expected.shape) != c_shape:
            raise ValueError("OUTPUT_MAPPING_SHAPE_MISMATCH:" + c_spec["name"])
        row = compare_arrays(
            candidate_array,
            expected,
            c_spec["dtype"],
            c_spec["name"],
            c_spec.get("quantization"),
            expected_quant,
            mapping,
        )
        row.update({
            "candidate_handle": c_index,
            "candidate_output_sha256": sha256_file(candidate_path),
            "reference_handle": r_index,
            "reference_output_sha256": sha256_file(reference_path),
        })
        comparisons.append(row)
    mismatch_count = sum(int(row["mismatch_count"]) for row in comparisons)
    report = {
        "schema": "tflite_tp.phase3.npu-exact-comparison.v1",
        "profile": candidate_run.get("profile"),
        "fixture_id": candidate_run.get("fixture_id"),
        "fixture_sha256": candidate_run.get("fixture_sha256"),
        "tp_degree": tp,
        "rank": rank,
        "candidate_tflite_sha256": candidate["sha256"],
        "canonical_reference_tflite_sha256": reference["sha256"],
        "candidate_run_sha256": sha256_file(candidate_run_manifest_path),
        "reference_run_sha256": sha256_file(reference_run_manifest_path),
        "compared_output_count": len(comparisons),
        "uncovered_outputs": uncovered,
        "compared_element_count": sum(int(row["element_count"]) for row in comparisons),
        "mismatch_count": mismatch_count if comparisons and not uncovered else None,
        "max_abs_diff": max((row["max_abs_diff"] or 0) for row in comparisons) if comparisons else None,
        "first_failure": next((row for row in comparisons if row["mismatch_count"]), None),
        "comparisons": comparisons,
        "exact_gate": "PASS" if comparisons and not uncovered and mismatch_count == 0 else "FAIL",
        "coverage_gate": "COMPLETE" if comparisons and not uncovered else "INCOMPLETE",
    }
    return report


def stitch_tp4_outputs(
    rank_models,
    rank_output_dirs,
    rank_run_manifests,
    tensor_name,
    reference_model,
    reference_output_dir,
    reference_run_manifest,
    axis=None,
    output_path=None,
):
    """Concatenate TP4 rank shards in manifest rank order, then compare to Canonical."""
    if len(rank_models) != 4 or len(rank_output_dirs) != 4 or len(rank_run_manifests) != 4:
        raise ValueError("TP4_STITCH_REQUIRES_FOUR_RANKS")
    contracts = [load_tflite_contract(path) for path in rank_models]
    runs = [read_json(path) for path in rank_run_manifests]
    if [int(run.get("rank", -1)) for run in runs] != [0, 1, 2, 3]:
        raise ValueError("TP4_RANK_ORDER_INVALID")
    fixture_shas = {run.get("fixture_sha256") for run in runs}
    if len(fixture_shas) != 1:
        raise ValueError("TP4_FIXTURE_SHA_MISMATCH")
    ref_contract = load_tflite_contract(reference_model)
    ref_run = read_json(reference_run_manifest)
    ref_by_name = {x["name"]: i for i, x in enumerate(ref_contract["outputs"])}
    if tensor_name not in ref_by_name:
        raise ValueError("TP4_REFERENCE_TENSOR_NOT_COVERED:" + tensor_name)
    rank_arrays = []
    rank_specs = []
    for rank, (contract, output_dir, run) in enumerate(zip(contracts, rank_output_dirs, runs)):
        if run.get("fixture_sha256") != ref_run.get("fixture_sha256"):
            raise ValueError("TP4_REFERENCE_FIXTURE_SHA_MISMATCH")
        matches = [(i, row) for i, row in enumerate(contract["outputs"]) if row["name"] == tensor_name]
        if len(matches) != 1:
            raise ValueError("TP4_RANK_OUTPUT_MAPPING_INVALID:{}:{}".format(rank, tensor_name))
        index, spec = matches[0]
        array, _, _ = _load_run_tensor(contract, output_dir, run, index)
        rank_arrays.append(array)
        rank_specs.append(spec)
    if any(spec["dtype"] != rank_specs[0]["dtype"] for spec in rank_specs):
        raise ValueError("TP4_RANK_OUTPUT_DTYPE_MISMATCH")
    if any(not _quantization_equal(spec.get("quantization"), rank_specs[0].get("quantization")) for spec in rank_specs[1:]):
        raise ValueError("TP4_RANK_OUTPUT_QUANTIZATION_MISMATCH")
    if axis is None:
        ref_shape = tuple(ref_contract["outputs"][ref_by_name[tensor_name]]["shape"])
        local_shape = tuple(rank_specs[0]["shape"])
        axes = [i for i, (local, full) in enumerate(zip(local_shape, ref_shape)) if full == local * 4]
        if len(axes) != 1:
            raise ValueError("TP4_STITCH_AXIS_AMBIGUOUS")
        axis = axes[0]
    axis = int(axis)
    if not 0 <= axis < len(rank_specs[0]["shape"]):
        raise ValueError("TP4_STITCH_AXIS_INVALID")
    stitched = np.ascontiguousarray(np.concatenate(rank_arrays, axis=axis))
    expected, ref_path, _ = _load_run_tensor(
        ref_contract, reference_output_dir, ref_run, ref_by_name[tensor_name]
    )
    if stitched.shape != expected.shape:
        raise ValueError("TP4_STITCHED_SHAPE_MISMATCH")
    report = compare_arrays(
        stitched,
        expected,
        rank_specs[0]["dtype"],
        tensor_name,
        rank_specs[0].get("quantization"),
        ref_contract["outputs"][ref_by_name[tensor_name]].get("quantization"),
        {"kind": "TP4_CONCATENATE", "axis": axis, "rank_order": [0, 1, 2, 3]},
    )
    report.update({
        "schema": "tflite_tp.phase3.tp4-stitch-comparison.v1",
        "fixture_sha256": next(iter(fixture_shas)),
        "rank_tflite_sha256": [contract["sha256"] for contract in contracts],
        "reference_tflite_sha256": ref_contract["sha256"],
        "reference_output_sha256": sha256_file(ref_path),
    })
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(stitched.tobytes(order="C"))
        report["stitched_output"] = {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        }
    report["exact_gate"] = "PASS" if report["mismatch_count"] == 0 else "FAIL"
    return report


def tp2_row_reduce_gate(contract):
    """Refuse to synthesize TP2 reductions until their quantized contract is proven."""
    required = (
        "accumulator_dtype",
        "bias_ownership",
        "partial_domain",
        "reduction_order",
        "requantization",
        "rounding_mode",
        "saturation",
        "residual_order",
        "evidence_sha256",
    )
    missing = [key for key in required if not contract or contract.get(key) in (None, "", "UNPROVEN")]
    if missing:
        return {
            "schema": "tflite_tp.phase3.tp2-row-reduce-contract.v1",
            "status": "UNPROVEN",
            "release_eligible": False,
            "missing_evidence": missing,
            "first_failure": "TP2_ROW_REDUCE_SEMANTICS_UNPROVEN",
        }
    return {
        "schema": "tflite_tp.phase3.tp2-row-reduce-contract.v1",
        "status": "CONTRACT_EVIDENCE_PRESENT_REQUIRES_NUMERICAL_TEST",
        "release_eligible": False,
        "missing_evidence": [],
        "first_failure": "TP2_NUMERICAL_TEST_NOT_RUN",
    }
