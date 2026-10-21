"""Prepare stage inputs only from bit-exact, SHA-bound TP4 NPU stitches."""

import json
import math
import shutil
from pathlib import Path

import numpy as np

from .contracts import DTYPE_NAMES, load_tflite_contract, read_json, sha256_file


def _quant_equal(left, right):
    if left is None or right is None:
        return left is None and right is None
    return all(left.get(key) == right.get(key) for key in ("scales", "zero_points", "quantized_dimension"))


def _resolve(base, value):
    path = Path(value)
    if path.is_absolute():
        return path
    for parent in (base, *base.parents):
        candidate = (parent / path).resolve()
        if candidate.exists():
            return candidate
    return (base / path).resolve()


def _verified_stitch(binding, base, fixture_id, fixture_sha256):
    report_path = _resolve(base, binding["stitch_report"])
    config_path = _resolve(base, binding["stitch_config"])
    report = read_json(report_path)
    config = read_json(config_path)
    if report.get("schema") != "tflite_tp.phase3.tp4-stitch-comparison.v1":
        raise ValueError("STAGE_SOURCE_NOT_TP4_STITCH")
    if report.get("status") != "EXACT" or report.get("exact_gate") != "PASS":
        raise ValueError("STAGE_SOURCE_STITCH_NOT_EXACT")
    if int(report.get("mismatch_count", -1)) != 0 or report.get("max_abs_diff") != 0:
        raise ValueError("STAGE_SOURCE_STITCH_MISMATCH")
    if report.get("first_failure") is not None:
        raise ValueError("STAGE_SOURCE_STITCH_FIRST_FAILURE")
    semantic = binding["semantic_tensor"]
    if report.get("semantic_tensor") != semantic or config.get("tensor_name") != semantic:
        raise ValueError("STAGE_SOURCE_SEMANTIC_MISMATCH")
    if report.get("fixture_sha256") != fixture_sha256:
        raise ValueError("STAGE_SOURCE_FIXTURE_SHA_MISMATCH")
    if config.get("output_path") is None:
        raise ValueError("STAGE_SOURCE_OUTPUT_PATH_MISSING")
    source_path = _resolve(base, binding.get("source_path", config["output_path"]))
    stitch_row = report.get("stitched_output", {})
    if not source_path.is_file():
        raise ValueError("STAGE_SOURCE_FILE_MISSING")
    actual_sha = sha256_file(source_path)
    expected_sha = stitch_row.get("sha256")
    if actual_sha != expected_sha:
        raise ValueError("STAGE_SOURCE_SHA_MISMATCH")

    reference_model_path = _resolve(base, config["reference_model"])
    reference_run_path = _resolve(base, config["reference_run_manifest"])
    reference_output_dir = _resolve(base, config["reference_output_dir"])
    reference_model = load_tflite_contract(reference_model_path)
    if reference_model["sha256"] != report.get("reference_tflite_sha256"):
        raise ValueError("STAGE_REFERENCE_MODEL_SHA_MISMATCH")
    reference_run = read_json(reference_run_path)
    if reference_run.get("fixture_sha256") != fixture_sha256:
        raise ValueError("STAGE_REFERENCE_FIXTURE_SHA_MISMATCH")
    if reference_run.get("status") != "PASS" and not (
        reference_run.get("status") == "CAPTURE_SHA_VERIFIED"
        and reference_run.get("reference_eligible") is True
    ):
        raise ValueError("STAGE_REFERENCE_RUN_UNVERIFIED")
    if reference_run.get("source_tflite_sha256") != reference_model["sha256"]:
        raise ValueError("STAGE_REFERENCE_RUN_MODEL_SHA_MISMATCH")
    matches = [(i, item) for i, item in enumerate(reference_model["outputs"]) if item["name"] == semantic]
    if len(matches) != 1:
        raise ValueError("STAGE_REFERENCE_TENSOR_UNMAPPED")
    handle, reference_spec = matches[0]
    if (
        reference_spec["shape"] != report.get("shape")
        or reference_spec["dtype"] != report.get("dtype")
        or report.get("element_count") != math.prod(reference_spec["shape"])
    ):
        raise ValueError("STAGE_REFERENCE_CONTRACT_MISMATCH")
    rows = [row for row in reference_run.get("outputs", []) if int(row.get("handle_index", -1)) == handle]
    if len(rows) != 1 or rows[0].get("semantic_tensor") != semantic or rows[0].get("status") != "PASS":
        raise ValueError("STAGE_REFERENCE_OUTPUT_UNVERIFIED")
    reference_path = reference_output_dir / "output{}.bin".format(handle)
    reference_row = rows[0]
    if (
        not reference_path.is_file()
        or reference_path.stat().st_size != int(reference_row.get("bytes", -1))
        or sha256_file(reference_path) != reference_row.get("sha256")
        or sha256_file(reference_path) != report.get("reference_output_sha256")
    ):
        raise ValueError("STAGE_REFERENCE_OUTPUT_SHA_MISMATCH")
    if source_path.stat().st_size != reference_path.stat().st_size:
        raise ValueError("STAGE_SOURCE_REFERENCE_BYTES_MISMATCH")
    if source_path.read_bytes() != reference_path.read_bytes():
        raise ValueError("STAGE_SOURCE_NOT_BITEXACT_TO_REFERENCE")
    return {
        "source": "EXACT_TP4_STITCHED_CANONICAL_NPU_STAGE",
        "semantic_tensor": semantic,
        "source_path": source_path,
        "source_sha256": actual_sha,
        "shape": reference_spec["shape"],
        "dtype": reference_spec["dtype"],
        "quantization": reference_spec.get("quantization"),
        "evidence": {
            "stitch_report": str(report_path),
            "stitch_report_sha256": sha256_file(report_path),
            "stitch_config": str(config_path),
            "stitch_config_sha256": sha256_file(config_path),
            "reference_tflite_sha256": reference_model["sha256"],
            "reference_run_manifest_sha256": sha256_file(reference_run_path),
            "reference_output_sha256": sha256_file(reference_path),
        },
    }


def _verified_prepared_input(binding, base, fixture_id, fixture_sha256):
    manifest_path = _resolve(base, binding["input_manifest"])
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS_CONTRACT_AND_SHA_VERIFIED":
        raise ValueError("STAGE_PUBLIC_INPUT_MANIFEST_UNVERIFIED")
    if manifest.get("fixture_id") != fixture_id or manifest.get("fixture_sha256") != fixture_sha256:
        raise ValueError("STAGE_PUBLIC_INPUT_FIXTURE_MISMATCH")
    semantic = binding["semantic_tensor"]
    rows = [row for row in manifest.get("inputs", []) if row.get("semantic_tensor") == semantic]
    if len(rows) != 1:
        raise ValueError("STAGE_PUBLIC_INPUT_MAPPING_INVALID")
    row = rows[0]
    if row.get("source") not in (
        "CANONICAL_PUBLIC_INPUT",
        "SHA_VERIFIED_PHASE2D_NPZ_FIXTURE",
    ):
        raise ValueError("STAGE_PUBLIC_INPUT_SOURCE_NOT_CANONICAL")
    if row.get("adaptation") != "IDENTITY_BY_SEMANTIC_NAME":
        raise ValueError("STAGE_PUBLIC_INPUT_ADAPTATION_NOT_IDENTITY")
    path = _resolve(base, binding["source_path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(row.get("bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise ValueError("STAGE_PUBLIC_INPUT_SHA_MISMATCH")
    return {
        "source": row["source"],
        "semantic_tensor": semantic,
        "source_path": path,
        "source_sha256": row["sha256"],
        "shape": row["shape"],
        "dtype": row["dtype"],
        "quantization": row.get("quantization"),
        "evidence": {
            "input_manifest": str(manifest_path),
            "input_manifest_sha256": sha256_file(manifest_path),
            "input_handle": int(row["handle"]),
            "source_handle": row.get("source_handle"),
            "source_sha256": row.get("source_sha256"),
        },
    }


def _verified_component_output(binding, base, fixture_id, fixture_sha256):
    config_path = _resolve(base, binding["component_compare_config"])
    config = read_json(config_path)
    comparison_path = _resolve(config_path.parent, config["comparison_report"])
    comparison = read_json(comparison_path)
    semantic = binding["semantic_tensor"]
    if comparison.get("exact_gate") != "PASS" or comparison.get("coverage_gate") != "COMPLETE":
        raise ValueError("STAGE_COMPONENT_COMPARE_NOT_EXACT")
    if int(comparison.get("mismatch_count", -1)) != 0 or comparison.get("first_failure") is not None:
        raise ValueError("STAGE_COMPONENT_COMPARE_MISMATCH")
    if comparison.get("fixture_sha256") != fixture_sha256:
        raise ValueError("STAGE_COMPONENT_FIXTURE_SHA_MISMATCH")
    matches = [row for row in comparison.get("comparisons", []) if row.get("semantic_tensor") == semantic]
    if len(matches) != 1 or matches[0].get("status") != "EXACT" or int(matches[0].get("mismatch_count", -1)) != 0:
        raise ValueError("STAGE_COMPONENT_OUTPUT_NOT_EXACT: " + semantic)
    comparison_row = matches[0]

    candidate_model_path = _resolve(config_path.parent, config["candidate_model"])
    candidate_output_dir = _resolve(config_path.parent, config["candidate_output_dir"])
    candidate_run_path = _resolve(config_path.parent, config["candidate_run_manifest"])
    reference_model_path = _resolve(config_path.parent, config["reference_model"])
    reference_output_dir = _resolve(config_path.parent, config["reference_output_dir"])
    reference_run_path = _resolve(config_path.parent, config["reference_run_manifest"])
    candidate_model = load_tflite_contract(candidate_model_path)
    reference_model = load_tflite_contract(reference_model_path)
    if candidate_model["sha256"] != comparison.get("candidate_tflite_sha256"):
        raise ValueError("STAGE_COMPONENT_MODEL_SHA_MISMATCH")
    if reference_model["sha256"] != comparison.get("canonical_reference_tflite_sha256"):
        raise ValueError("STAGE_COMPONENT_REFERENCE_MODEL_SHA_MISMATCH")
    candidate_run = read_json(candidate_run_path)
    reference_run = read_json(reference_run_path)
    if candidate_run.get("status") != "PASS" or candidate_run.get("fixture_sha256") != fixture_sha256:
        raise ValueError("STAGE_COMPONENT_RUN_UNVERIFIED")
    if candidate_run.get("source_tflite_sha256") != candidate_model["sha256"]:
        raise ValueError("STAGE_COMPONENT_RUN_MODEL_SHA_MISMATCH")
    if reference_run.get("fixture_sha256") != fixture_sha256:
        raise ValueError("STAGE_COMPONENT_REFERENCE_FIXTURE_SHA_MISMATCH")
    if reference_run.get("status") != "PASS" and not (
        reference_run.get("status") == "CAPTURE_SHA_VERIFIED"
        and reference_run.get("reference_eligible") is True
    ):
        raise ValueError("STAGE_COMPONENT_REFERENCE_RUN_UNVERIFIED")
    if reference_run.get("source_tflite_sha256") != reference_model["sha256"]:
        raise ValueError("STAGE_COMPONENT_REFERENCE_RUN_MODEL_SHA_MISMATCH")
    c_matches = [(i, row) for i, row in enumerate(candidate_model["outputs"]) if row["name"] == semantic]
    r_matches = [(i, row) for i, row in enumerate(reference_model["outputs"]) if row["name"] == semantic]
    if len(c_matches) != 1 or len(r_matches) != 1:
        raise ValueError("STAGE_COMPONENT_TENSOR_MAPPING_INVALID: " + semantic)
    c_handle, c_spec = c_matches[0]
    r_handle, r_spec = r_matches[0]
    if (
        c_spec["shape"] != r_spec["shape"]
        or c_spec["dtype"] != r_spec["dtype"]
        or not _quant_equal(c_spec.get("quantization"), r_spec.get("quantization"))
    ):
        raise ValueError("STAGE_COMPONENT_REFERENCE_CONTRACT_MISMATCH: " + semantic)
    c_rows = [row for row in candidate_run.get("outputs", []) if int(row.get("handle_index", -1)) == c_handle]
    r_rows = [row for row in reference_run.get("outputs", []) if int(row.get("handle_index", -1)) == r_handle]
    if len(c_rows) != 1 or len(r_rows) != 1 or c_rows[0].get("status") != "PASS" or r_rows[0].get("status") != "PASS":
        raise ValueError("STAGE_COMPONENT_OUTPUT_HANDLE_UNVERIFIED: " + semantic)
    candidate_path = candidate_output_dir / "output{}.bin".format(c_handle)
    reference_path = reference_output_dir / "output{}.bin".format(r_handle)
    if (
        not candidate_path.is_file()
        or sha256_file(candidate_path) != c_rows[0].get("sha256")
        or sha256_file(candidate_path) != comparison_row.get("candidate_output_sha256")
    ):
        raise ValueError("STAGE_COMPONENT_OUTPUT_SHA_MISMATCH: " + semantic)
    if (
        not reference_path.is_file()
        or sha256_file(reference_path) != r_rows[0].get("sha256")
        or sha256_file(reference_path) != comparison_row.get("reference_output_sha256")
    ):
        raise ValueError("STAGE_COMPONENT_REFERENCE_OUTPUT_SHA_MISMATCH: " + semantic)
    if candidate_path.read_bytes() != reference_path.read_bytes():
        raise ValueError("STAGE_COMPONENT_OUTPUT_NOT_BITEXACT: " + semantic)
    return {
        "source": "EXACT_NPU_COMPONENT_OUTPUT",
        "semantic_tensor": semantic,
        "source_path": candidate_path,
        "source_sha256": sha256_file(candidate_path),
        "shape": c_spec["shape"],
        "dtype": c_spec["dtype"],
        "quantization": c_spec.get("quantization"),
        "evidence": {
            "component_compare_config": str(config_path),
            "component_compare_config_sha256": sha256_file(config_path),
            "comparison_report": str(comparison_path),
            "comparison_report_sha256": sha256_file(comparison_path),
            "candidate_run_manifest_sha256": sha256_file(candidate_run_path),
            "reference_run_manifest_sha256": sha256_file(reference_run_path),
            "reference_output_sha256": sha256_file(reference_path),
        },
    }


def _verified_source(binding, base, fixture_id, fixture_sha256):
    kind = binding.get("evidence_kind", "TP4_STITCH")
    if kind == "TP4_STITCH":
        return _verified_stitch(binding, base, fixture_id, fixture_sha256)
    if kind == "CANONICAL_PUBLIC_INPUT":
        return _verified_prepared_input(binding, base, fixture_id, fixture_sha256)
    if kind == "EXACT_NPU_COMPONENT_OUTPUT":
        return _verified_component_output(binding, base, fixture_id, fixture_sha256)
    raise ValueError("STAGE_SOURCE_EVIDENCE_KIND_UNSUPPORTED: " + str(kind))


def prepare_stage_inputs(
    candidate_model,
    stitch_config_paths,
    stitch_report_paths,
    component_compare_config_paths,
    public_input_manifest_path,
    public_input_dir,
    output_dir,
    profile,
    tp,
    rank,
):
    """Resolve inputs by semantic name from exact stitches or sealed public inputs."""
    candidate_model = Path(candidate_model).resolve()
    stitch_config_paths = [Path(path).resolve() for path in stitch_config_paths]
    stitch_report_paths = [Path(path).resolve() for path in stitch_report_paths]
    component_compare_config_paths = [Path(path).resolve() for path in component_compare_config_paths]
    if stitch_report_paths and len(stitch_report_paths) != len(stitch_config_paths):
        raise ValueError("STAGE_STITCH_CONFIG_REPORT_COUNT_MISMATCH")
    public_input_manifest_path = Path(public_input_manifest_path).resolve() if public_input_manifest_path else None
    public_input_dir = Path(public_input_dir).resolve() if public_input_dir else None
    out = Path(output_dir).resolve()
    candidate = load_tflite_contract(candidate_model)
    tp, rank = int(tp), int(rank)
    if tp != 4 or not 0 <= rank < tp:
        raise ValueError("STAGE_TP_RANK_CONTRACT_INVALID")
    public_manifest = read_json(public_input_manifest_path) if public_input_manifest_path else None
    if public_manifest:
        fixture_id = str(public_manifest.get("fixture_id", ""))
        fixture_sha256 = str(public_manifest.get("fixture_sha256", ""))
        if public_manifest.get("profile") != str(profile):
            raise ValueError("STAGE_PUBLIC_INPUT_PROFILE_MISMATCH")
    else:
        # A component depending only on a verified NPU stitch can derive its fixture
        # identity from that evidence. Multi-input components should pass the sealed
        # public-input manifest so all source rows are checked against one fixture.
        fixture_id = ""
        fixture_sha256 = ""
    if fixture_sha256 and len(fixture_sha256) != 64:
        raise ValueError("STAGE_FIXTURE_SHA_INVALID")
    configs = []
    for config_index, config_path in enumerate(stitch_config_paths):
        config = read_json(config_path)
        if config.get("tensor_name"):
            report_path = stitch_report_paths[config_index] if stitch_report_paths else None
            configs.append((config_path, config, report_path))
    component_configs = [(path, read_json(path)) for path in component_compare_config_paths]
    public_rows = {
        row.get("semantic_tensor"): row
        for row in (public_manifest.get("inputs", []) if public_manifest else [])
    }
    if public_manifest and len(public_rows) != len(public_manifest.get("inputs", [])):
        raise ValueError("STAGE_PUBLIC_INPUT_DUPLICATE_SEMANTIC")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("OUTPUT_DIRECTORY_NOT_EMPTY")
    rows = []
    evidence_rows = []
    for handle, target in enumerate(candidate["inputs"]):
        semantic = target["name"]
        stitch_matches = [(path, config, report_path) for path, config, report_path in configs if config.get("tensor_name") == semantic]
        if len(stitch_matches) > 1:
            raise ValueError("STAGE_INPUT_STITCH_MAPPING_AMBIGUOUS: " + semantic)
        if stitch_matches:
            config_path, config, report_path = stitch_matches[0]
            output_path = _resolve(config_path.parent, config["output_path"])
            if report_path is None:
                report_path = output_path.with_suffix(".report.json")
            binding = {
                "semantic_tensor": semantic,
                "source_path": str(output_path),
                "stitch_config": str(config_path),
                "stitch_report": str(report_path),
            }
            report = read_json(binding["stitch_report"])
            if not fixture_id:
                fixture_sha256 = str(report.get("fixture_sha256", ""))
                fixture_id = str(report.get("fixture_id", "")) or "fixture-sha256-" + fixture_sha256[:12]
            if report.get("fixture_sha256") != fixture_sha256:
                raise ValueError("STAGE_INPUT_STITCH_FIXTURE_MISMATCH: " + semantic)
        else:
            component_matches = []
            for compare_config_path, compare_config in component_configs:
                report_path = _resolve(compare_config_path.parent, compare_config["comparison_report"])
                comparison = read_json(report_path)
                if any(row.get("semantic_tensor") == semantic for row in comparison.get("comparisons", [])):
                    component_matches.append(compare_config_path)
            if len(component_matches) > 1:
                raise ValueError("STAGE_INPUT_COMPONENT_MAPPING_AMBIGUOUS: " + semantic)
            if component_matches:
                compare_config = read_json(component_matches[0])
                comparison = read_json(_resolve(component_matches[0].parent, compare_config["comparison_report"]))
                if not fixture_id:
                    fixture_sha256 = str(comparison.get("fixture_sha256", ""))
                    fixture_id = str(comparison.get("fixture_id", "")) or "fixture-sha256-" + fixture_sha256[:12]
                if comparison.get("fixture_sha256") != fixture_sha256:
                    raise ValueError("STAGE_INPUT_COMPONENT_FIXTURE_MISMATCH: " + semantic)
                binding = {
                    "evidence_kind": "EXACT_NPU_COMPONENT_OUTPUT",
                    "semantic_tensor": semantic,
                    "component_compare_config": str(component_matches[0]),
                }
            elif semantic in public_rows and public_input_dir is not None:
                public_row = public_rows[semantic]
                binding = {
                    "evidence_kind": "CANONICAL_PUBLIC_INPUT",
                    "semantic_tensor": semantic,
                    "input_manifest": str(public_input_manifest_path),
                    "source_path": str(public_input_dir / "input{}.bin".format(int(public_row["handle"]))),
                }
            else:
                raise ValueError("BLOCKED_MISSING_STAGE_INPUT: " + semantic)
        source = _verified_source(binding, Path.cwd(), fixture_id, fixture_sha256)
        if source["shape"] != target["shape"]:
            raise ValueError("STAGE_INPUT_SHAPE_MISMATCH: " + semantic)
        if source["dtype"] != target["dtype"]:
            raise ValueError("STAGE_INPUT_DTYPE_MISMATCH: " + semantic)
        if not _quant_equal(source["quantization"], target.get("quantization")):
            raise ValueError("STAGE_INPUT_QUANTIZATION_MISMATCH: " + semantic)
        target_path = out / "input{}.bin".format(handle)
        shutil.copyfile(source["source_path"], target_path)
        expected_bytes = math.prod(target["shape"]) * np.dtype(DTYPE_NAMES[target["dtype"]]).itemsize
        if target_path.stat().st_size != expected_bytes:
            raise ValueError("STAGE_INPUT_BYTE_COUNT_MISMATCH: " + semantic)
        row = {
            "handle": handle,
            "semantic_tensor": semantic,
            "shape": target["shape"],
            "dtype": target["dtype"],
            "quantization": target.get("quantization"),
            "bytes": target_path.stat().st_size,
            "sha256": sha256_file(target_path),
            "source": source["source"],
            "source_sha256": source["source_sha256"],
            "adaptation": "IDENTITY_BY_SEMANTIC_AND_FULL_CONTRACT",
        }
        rows.append(row)
        evidence_rows.append({"handle": handle, **source["evidence"]})
    manifest = {
        "schema": "tflite_tp.phase3.npu-stage-input-adapter.v1",
        "status": "PASS_CONTRACT_AND_SHA_VERIFIED",
        "profile": str(profile),
        "fixture_id": fixture_id,
        "fixture_sha256": fixture_sha256,
        "tp_degree": tp,
        "rank": rank,
        "candidate_tflite_sha256": candidate["sha256"],
        "inputs": rows,
        "source_evidence": evidence_rows,
    }
    (out / "INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "SHA256SUMS").write_text(
        "".join("{}  input{}.bin\n".format(row["sha256"], row["handle"]) for row in rows), encoding="ascii"
    )
    return manifest
