"""Validate and normalize an existing, hash-bound Canonical NPU capture."""

import json
import math
from pathlib import Path

import numpy as np

from .contracts import DTYPE_NAMES, load_tflite_contract, read_json, sha256_file


def import_canonical_capture(model_path, capture_manifest_path, output_dir, output_manifest_path):
    """Import a historical capture as a reference only after checking every output byte.

    This never upgrades missing command/runtime evidence into a fresh invoke PASS.
    The result is explicitly `CAPTURE_SHA_VERIFIED` and may be used as a same-
    fixture Canonical NPU reference when its model and fixture identities match.
    """
    model_path = Path(model_path)
    capture_path = Path(capture_manifest_path)
    output_root = Path(output_dir)
    destination = Path(output_manifest_path)
    model = load_tflite_contract(model_path)
    capture = read_json(capture_path)
    if capture.get("canonical_tflite_sha256") != model["sha256"]:
        raise ValueError("CAPTURE_CANONICAL_TFLITE_SHA_MISMATCH")
    if not capture.get("fixture_sha256") or not capture.get("source_model_sha256"):
        raise ValueError("CAPTURE_IDENTITY_INCOMPLETE")
    if capture.get("invoke", {}).get("exit_code") != 0:
        raise ValueError("CAPTURE_INVOKE_NOT_SUCCESSFUL")
    rows = capture.get("outputs", [])
    if len(rows) != len(model["outputs"]):
        raise ValueError("CAPTURE_OUTPUT_COVERAGE_INCOMPLETE")
    normalized = []
    for index, (row, spec) in enumerate(zip(rows, model["outputs"])):
        if int(row.get("handle_index", -1)) != index:
            raise ValueError("CAPTURE_OUTPUT_HANDLE_ORDER_MISMATCH")
        if row.get("semantic_tensor") != spec["name"]:
            raise ValueError("CAPTURE_OUTPUT_SEMANTIC_MISMATCH:{}".format(index))
        if row.get("shape") != spec["shape"] or row.get("dtype") != spec["dtype"]:
            raise ValueError("CAPTURE_OUTPUT_CONTRACT_MISMATCH:{}".format(index))
        path = output_root / "output{}.bin".format(index)
        if not path.is_file():
            raise ValueError("CAPTURE_OUTPUT_MISSING:{}".format(index))
        actual_sha = sha256_file(path)
        if path.stat().st_size != int(row.get("bytes", -1)) or actual_sha != row.get("sha256"):
            raise ValueError("CAPTURE_OUTPUT_SHA_OR_SIZE_MISMATCH:{}".format(index))
        normalized.append({
            "handle_index": index,
            "semantic_tensor": spec["name"],
            "shape": spec["shape"],
            "dtype": spec["dtype"],
            "quantization": spec.get("quantization"),
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "status": "PASS",
        })
    invoke = capture.get("invoke", {})
    result = {
        "schema": "tflite_tp.phase3.npu-run.v1",
        "status": "CAPTURE_SHA_VERIFIED",
        "reference_eligible": True,
        "capture_kind": "HISTORICAL_CANONICAL_NPU_CAPTURE",
        "capture_provenance": invoke.get("status", "UNSPECIFIED"),
        "exact_command": invoke.get("exact_command", "NOT_CAPTURED"),
        "invoke_exit_code": 0,
        "profile": capture.get("profile"),
        "fixture_id": capture.get("fixture_id"),
        "fixture_sha256": capture.get("fixture_sha256"),
        "source_model_sha256": capture.get("source_model_sha256"),
        "source_tflite_sha256": model["sha256"],
        "dla_sha256": capture.get("canonical_dla", {}).get("sha256"),
        "ncc_target_arch": capture.get("canonical_dla", {}).get("target_arch"),
        "ncc_version": capture.get("canonical_dla", {}).get("ncc_version_for_this_dla"),
        "device": capture.get("evk", {}),
        "outputs": normalized,
        "output_coverage": "COMPLETE",
        "source_capture_manifest": str(capture_path),
        "source_capture_manifest_sha256": sha256_file(capture_path),
        "source_output_directory": str(output_root),
        "numerical_gate": "REFERENCE_AVAILABLE_CANDIDATE_NOT_RUN",
        "limitations": [
            "Historical capture was validated against its saved TFLite, Fixture, output contracts, sizes and SHA256.",
            "The original full invoke command/runtime log was not captured; this is not a fresh invoke PASS.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def import_canonical_prefix_capture(
    prefix_model_path,
    prefix_graph_manifest_path,
    comparison_manifest_path,
    output_dir,
    output_manifest_path,
    canonical_capture_manifest_path,
):
    """Normalize a saved canonical prefix NPU capture with per-output SHA checks."""
    model = load_tflite_contract(prefix_model_path)
    graph = read_json(prefix_graph_manifest_path)
    comparison = read_json(comparison_manifest_path)
    canonical_capture = read_json(canonical_capture_manifest_path)
    if graph.get("destination_sha256") != model["sha256"]:
        raise ValueError("PREFIX_GRAPH_MODEL_SHA_MISMATCH")
    comparison_model_sha = comparison.get("canonical_prefix_tflite_sha256") or comparison.get("prefix_graph_sha256")
    if comparison_model_sha != model["sha256"]:
        raise ValueError("PREFIX_CAPTURE_MODEL_SHA_MISMATCH")
    comparison_gate = comparison.get("exact_gate") or comparison.get("public_stability_gate")
    if comparison_gate != "PASS" or comparison.get("fixture_sha256") is None:
        raise ValueError("PREFIX_CAPTURE_SOURCE_GATE_FAIL")
    if comparison.get("fixture_sha256") != canonical_capture.get("fixture_sha256"):
        raise ValueError("PREFIX_CAPTURE_FIXTURE_BINDING_FAIL")
    if canonical_capture.get("canonical_tflite_sha256") != graph.get("source_sha256"):
        raise ValueError("PREFIX_CAPTURE_CANONICAL_SOURCE_BINDING_FAIL")
    hashes = comparison.get("canonical_prefix_output_hashes")
    hash_field = "canonical_prefix_output_hashes"
    if hashes is None:
        hashes = comparison.get("prefix_outputs_verified", [])
        hash_field = "prefix_outputs_verified"
    if len(hashes) != len(model["outputs"]):
        raise ValueError("PREFIX_CAPTURE_OUTPUT_COVERAGE_INCOMPLETE")
    output_root = Path(output_dir)
    normalized = []
    for index, (row, spec) in enumerate(zip(hashes, model["outputs"])):
        expected_name = "output{}.bin".format(index)
        if row.get("file") != expected_name:
            raise ValueError("PREFIX_CAPTURE_OUTPUT_HANDLE_ORDER_MISMATCH")
        path = output_root / expected_name
        if not path.is_file():
            raise ValueError("PREFIX_CAPTURE_OUTPUT_MISSING:{}".format(index))
        actual_sha = sha256_file(path)
        if path.stat().st_size != int(row.get("bytes", -1)) or actual_sha != row.get("sha256"):
            raise ValueError("PREFIX_CAPTURE_OUTPUT_SHA_OR_SIZE_MISMATCH:{}".format(index))
        expected_bytes = int(math.prod(spec["shape"]) * np.dtype(DTYPE_NAMES[spec["dtype"]]).itemsize)
        if path.stat().st_size != expected_bytes:
            raise ValueError("PREFIX_CAPTURE_TENSOR_BYTE_CONTRACT_MISMATCH:{}".format(index))
        normalized.append({
            "handle_index": index,
            "semantic_tensor": spec["name"],
            "shape": spec["shape"],
            "dtype": spec["dtype"],
            "quantization": spec.get("quantization"),
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "status": "PASS",
        })
    destination = Path(output_manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "tflite_tp.phase3.npu-run.v1",
        "status": "CAPTURE_SHA_VERIFIED",
        "reference_eligible": True,
        "capture_kind": "HISTORICAL_CANONICAL_PREFIX_NPU_CAPTURE",
        "capture_provenance": "PHASE2G_COMPARISON_AND_OUTPUT_SHA_VALIDATED",
        "exact_command": "NOT_CAPTURED",
        "invoke_exit_code": 0,
        "profile": canonical_capture.get("profile"),
        "fixture_id": canonical_capture.get("fixture_id"),
        "fixture_sha256": comparison["fixture_sha256"],
        "source_model_sha256": canonical_capture.get("source_model_sha256"),
        "source_tflite_sha256": model["sha256"],
        "dla_sha256": comparison.get("canonical_prefix_dla_sha256"),
        "outputs": normalized,
        "output_coverage": "COMPLETE",
        "source_graph_manifest": str(prefix_graph_manifest_path),
        "source_graph_manifest_sha256": sha256_file(prefix_graph_manifest_path),
        "source_comparison_manifest": str(comparison_manifest_path),
        "source_comparison_manifest_sha256": sha256_file(comparison_manifest_path),
        "source_comparison_model_sha_field": "canonical_prefix_tflite_sha256" if comparison.get("canonical_prefix_tflite_sha256") else "prefix_graph_sha256",
        "source_comparison_output_sha_field": hash_field,
        "source_comparison_gate_field": "exact_gate" if comparison.get("exact_gate") else "public_stability_gate",
        "source_canonical_capture_manifest": str(canonical_capture_manifest_path),
        "source_canonical_capture_manifest_sha256": sha256_file(canonical_capture_manifest_path),
        "source_output_directory": str(output_root),
        "numerical_gate": "REFERENCE_AVAILABLE_CANDIDATE_NOT_RUN",
        "limitations": [
            "This is a historical Canonical prefix capture, not a run of the current rank candidate.",
            "The saved prefix invoke command and DLA hash were not captured; output bytes are validated against the Phase 2G comparison manifest.",
        ],
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
