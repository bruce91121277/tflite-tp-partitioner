"""Portable preparation and comparison helpers for TP4 NPU FC canaries.

Preparation only rewrites FlatBuffer public output lists to expose existing
Q-projection and activation tensors. It does not change operators or weights.
NPU result comparison is deliberately a separate command and requires capture
sidecars declaring an actual NPU backend.
"""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .errors import ToolError
from .flatbuffer_adapter import TFLiteObjectAdapter


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _vector(value):
    return [] if value is None else value


def _shape(value):
    return None if value is None else [int(item) for item in value]


def _tensor_name_map(graph):
    result = {}
    for index, tensor in enumerate(graph.tensors):
        result.setdefault(_name(tensor.name), []).append(index)
    return result


def _tensor_contract(tensor, schema):
    tensor_type = int(tensor.type)
    type_name = next(
        (name for name, value in vars(schema.TensorType).items()
         if name.isupper() and isinstance(value, int) and value == tensor_type),
        str(tensor_type),
    )
    quant = getattr(tensor, "quantization", None)
    return {
        "name": _name(tensor.name),
        "shape": [int(item) for item in tensor.shape],
        "dtype": type_name,
        "dtype_code": tensor_type,
        "quantization": {
            "scales": [float(item) for item in _vector(getattr(quant, "scale", []))] if quant else [],
            "zero_points": [int(item) for item in _vector(getattr(quant, "zeroPoint", []))] if quant else [],
            "quantized_dimension": int(getattr(quant, "quantizedDimension", 0)) if quant else 0,
        },
    }


def _all_io(graph, schema):
    return {
        "inputs": [_tensor_contract(graph.tensors[int(index)], schema) for index in graph.inputs],
        "outputs": [_tensor_contract(graph.tensors[int(index)], schema) for index in graph.outputs],
    }


def _jsonable(value):
    if isinstance(value, bytes):
        return {"bytes_sha256": _sha256_bytes(value), "length": len(value)}
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in sorted(vars(value).items())
            if key != "_adapter"
        }
    return str(value)


def _graph_integrity_sha256(model):
    """Hash tensors, constants, operator options, and edges, excluding outputs."""
    subgraphs = []
    for graph in model.subgraphs:
        tensors = []
        for tensor in graph.tensors:
            quant = getattr(tensor, "quantization", None)
            tensors.append({
                "name": _name(tensor.name),
                "shape": _shape(tensor.shape),
                "shape_signature": _shape(getattr(tensor, "shapeSignature", None)),
                "type": int(tensor.type) if tensor.type is not None else None,
                "buffer": int(tensor.buffer) if tensor.buffer is not None else None,
                "quantization": {
                    "scale": [float(item) for item in _vector(getattr(quant, "scale", []))] if quant else [],
                    "zero_point": [int(item) for item in _vector(getattr(quant, "zeroPoint", []))] if quant else [],
                    "axis": int(getattr(quant, "quantizedDimension", 0) or 0) if quant else 0,
                },
            })
        subgraphs.append({
            "name": _name(getattr(graph, "name", b"") or b""),
            "inputs": [int(item) for item in graph.inputs],
            "tensors": tensors,
            "operators": [_jsonable(operator) for operator in graph.operators],
        })
    buffers = []
    for buffer in model.buffers:
        data = getattr(buffer, "data", None)
        if data is None:
            buffers.append({"length": 0, "sha256": _sha256_bytes(b"")})
        else:
            raw = np.asarray(data, dtype=np.uint8).reshape(-1).tobytes()
            buffers.append({"length": len(raw), "sha256": _sha256_bytes(raw)})
    snapshot = {
        "version": int(model.version),
        "operator_codes": [_jsonable(code) for code in model.operatorCodes],
        "buffers": buffers,
        "subgraphs": subgraphs,
    }
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _find_projection(model, graph, layer, role, adapter):
    suffix = "/self_attn/{}_proj/weight".format(role)
    weight_matches = [
        index for index, tensor in enumerate(graph.tensors)
        if _name(tensor.name).endswith(suffix)
        and ("/layers/{}/".format(layer) in "/" + _name(tensor.name))
    ]
    if len(weight_matches) != 1:
        raise ToolError(
            "NPU_CANARY_PROJECTION_MAPPING_FAILED",
            "layer {} role {} has {} matching weights".format(layer, role, len(weight_matches)),
        )
    weight_index = weight_matches[0]
    found = []
    for op_index, op in enumerate(graph.operators):
        if len(op.inputs) < 2 or int(op.inputs[1]) != weight_index:
            continue
        code = model.operatorCodes[int(op.opcodeIndex)]
        custom = _name(getattr(code, "customCode", b"") or b"")
        if custom == "MTKEXT_FULLY_CONNECTED" and len(op.inputs) == 3 and len(op.outputs) == 1:
            found.append((op_index, op))
    if len(found) != 1:
        raise ToolError(
            "NPU_CANARY_FC_MAPPING_FAILED",
            "layer {} role {} has {} matching MTK FC operators".format(layer, role, len(found)),
        )
    op_index, op = found[0]
    return {
        "operator_index": op_index,
        "activation_index": int(op.inputs[0]),
        "weight_index": weight_index,
        "fc_output_index": int(op.outputs[0]),
    }


def _find_unique_tensor(graph, exact_name):
    matches = [index for index, tensor in enumerate(graph.tensors) if _name(tensor.name) == exact_name]
    if len(matches) != 1:
        raise ToolError(
            "NPU_CANARY_TENSOR_MAPPING_FAILED",
            "expected one tensor named {}, found {}".format(exact_name, len(matches)),
        )
    return matches[0]


def _model_row(manifest, layer, rank, component="attn"):
    matches = [
        row for row in manifest.get("models", [])
        if int(row.get("layer", -1)) == int(layer)
        and int(row.get("rank", -1)) == int(rank)
        and row.get("component") == component
    ]
    if len(matches) != 1:
        raise ToolError(
            "NPU_CANARY_MODEL_MAPPING_FAILED",
            "expected one model for layer {}, rank {}, component {}; found {}".format(
                layer, rank, component, len(matches)
            ),
        )
    return matches[0]


def _source_sha_for_row(row):
    value = row.get("source_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ToolError("NPU_CANARY_SOURCE_SHA_MISSING", str(row.get("source")))
    return value.lower()


def _read_json(path, code):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolError(code, "{}: {}".format(path, exc)) from exc


def _copy_checked(source, destination, expected_sha=None):
    source = Path(source)
    if not source.is_file():
        raise ToolError("NPU_CANARY_INPUT_MISSING", str(source))
    actual_sha = _sha256(source)
    if expected_sha and actual_sha != expected_sha.lower():
        raise ToolError("NPU_CANARY_INPUT_SHA_MISMATCH", str(source))
    shutil.copy2(source, destination)
    return {"path": Path(destination).name, "bytes": source.stat().st_size, "sha256": actual_sha}


def _expose_outputs(adapter, source, destination, tensor_names):
    model = adapter.load(source)
    if len(model.subgraphs) != 1:
        raise ToolError("NPU_CANARY_SUBGRAPH_COUNT_UNSUPPORTED", str(source))
    if getattr(model, "signatureDefs", None):
        raise ToolError(
            "NPU_CANARY_SIGNATURES_UNSUPPORTED",
            "output taps require a TFLite model without SignatureDefs",
        )
    graph = model.subgraphs[0]
    integrity_before = _graph_integrity_sha256(model)
    original_inputs = [int(index) for index in graph.inputs]
    original_operator_count = len(graph.operators)
    original_tensor_count = len(graph.tensors)
    indices = [_find_unique_tensor(graph, name) for name in tensor_names]
    existing = [int(index) for index in graph.outputs]
    graph.outputs = np.asarray(list(dict.fromkeys(existing + indices)), dtype=np.int32)
    raw = adapter.export(model)
    Path(destination).write_bytes(raw)
    check = adapter.load(destination)
    check_graph = check.subgraphs[0]
    integrity_after = _graph_integrity_sha256(check)
    if integrity_after != integrity_before:
        raise ToolError("NPU_CANARY_GRAPH_OR_WEIGHT_CHANGED", str(destination))
    if [int(index) for index in check_graph.inputs] != original_inputs:
        raise ToolError("NPU_CANARY_INPUT_LIST_CHANGED", str(destination))
    if len(check_graph.operators) != original_operator_count or len(check_graph.tensors) != original_tensor_count:
        raise ToolError("NPU_CANARY_GRAPH_SIZE_CHANGED", str(destination))
    outputs = [int(index) for index in check_graph.outputs]
    if any(index not in outputs for index in indices):
        raise ToolError("NPU_CANARY_TAP_SERIALIZATION_FAILED", str(destination))
    if len(raw) < 8 or raw[4:8] != b"TFL3":
        raise ToolError("NPU_CANARY_FLATBUFFER_INVALID", str(destination))
    return check, outputs, integrity_before


def _fixture_contract(input_json, reference_json, input_npz, reference_npz, activation_name, q_name):
    input_meta = _read_json(input_json, "NPU_CANARY_FIXTURE_METADATA_INVALID")
    reference_meta = _read_json(reference_json, "NPU_CANARY_REFERENCE_METADATA_INVALID")
    if "inputs" not in input_meta or "outputs" not in reference_meta:
        raise ToolError("NPU_CANARY_FIXTURE_METADATA_INCOMPLETE", str(input_json))
    if q_name not in reference_meta["outputs"]:
        raise ToolError("NPU_CANARY_REFERENCE_TENSOR_MISSING", q_name)
    return {
        "profile": input_meta.get("profile"),
        "fixture_label": input_meta.get("fixture_label"),
        "fixture_sha256": input_meta.get("fixture_sha256"),
        "position_start": input_meta.get("position_start"),
        "valid_past_kv_length": input_meta.get("valid_past_kv_length"),
        "inputs": input_meta["inputs"],
        "input_archive_sha256": _sha256(input_npz),
        "reference_archive_sha256": _sha256(reference_npz),
        "canonical_q_reference": reference_meta["outputs"][q_name],
        "canonical_activation_reference": reference_meta["outputs"].get(activation_name),
        "reference_profile": reference_meta.get("profile") or input_meta.get("profile"),
        "reference_fixture_sha256": reference_meta.get("fixture_sha256"),
    }


def prepare_fc_canary(
    canonical_model,
    unpadded_manifest_path,
    padded_manifest_path,
    input_npz,
    input_json,
    reference_npz,
    reference_json,
    output_directory,
    layer=0,
    rank=0,
    role="q",
):
    """Create a portable canonical/unpadded/padded NPU canary bundle.

    ``layer``, ``rank`` and ``role`` select graph objects; no expected tensor
    values or source coordinates are embedded in this preparation rule.
    """
    output_root = Path(output_directory)
    if output_root.exists():
        raise ToolError("NPU_CANARY_OUTPUT_EXISTS", str(output_root))
    unpadded_root = Path(unpadded_manifest_path).parent
    padded_root = Path(padded_manifest_path).parent
    unpadded_manifest = _read_json(unpadded_manifest_path, "NPU_CANARY_MANIFEST_INVALID")
    padded_manifest = _read_json(padded_manifest_path, "NPU_CANARY_MANIFEST_INVALID")
    profile = unpadded_manifest.get("profile")
    if profile not in ("P128", "D1") or padded_manifest.get("profile") != profile:
        raise ToolError("NPU_CANARY_PROFILE_MISMATCH", "unpadded and padded manifests must share P128 or D1")
    if int(unpadded_manifest.get("expected_layers", -1)) != int(padded_manifest.get("expected_layers", -2)):
        raise ToolError("NPU_CANARY_LAYER_COUNT_MISMATCH", profile)
    if padded_manifest.get("fc_width_padding_rule", {}).get("multiplier") != 2:
        raise ToolError("NPU_CANARY_PADDING_RULE_MISSING", profile)
    unpadded_row = _model_row(unpadded_manifest, layer, rank)
    padded_row = _model_row(padded_manifest, layer, rank)
    if _source_sha_for_row(unpadded_row) != _source_sha_for_row(padded_row):
        raise ToolError("NPU_CANARY_CANONICAL_SOURCE_MISMATCH", profile)
    padding_rows = padded_row.get("fc_width_padding", [])
    if not any(
        item.get("role") == role and int(item.get("pad_multiplier", 0)) == 2
        for item in padding_rows
    ):
        raise ToolError("NPU_CANARY_PROJECTION_PADDING_MISSING", "{} layer {} rank {}".format(profile, layer, rank))
    expected_canonical_sha = _source_sha_for_row(unpadded_row)
    if _sha256(canonical_model) != expected_canonical_sha:
        raise ToolError("NPU_CANARY_CANONICAL_SOURCE_SHA_MISMATCH", str(canonical_model))
    if int(unpadded_row.get("rank", -1)) != int(rank) or int(padded_row.get("rank", -1)) != int(rank):
        raise ToolError("NPU_CANARY_RANK_MAPPING_MISMATCH", str(rank))

    q_tensor_name = "layers/{}/self_attn/{}_proj/linear".format(int(layer), role)
    adapter = TFLiteObjectAdapter()
    canonical_model_object = adapter.load(canonical_model)
    unpadded_model_path = unpadded_root / unpadded_row["path"]
    padded_model_path = padded_root / padded_row["path"]
    unpadded_model_object = adapter.load(unpadded_model_path)
    padded_model_object = adapter.load(padded_model_path)
    canonical_graph = canonical_model_object.subgraphs[0]
    unpadded_graph = unpadded_model_object.subgraphs[0]
    padded_graph = padded_model_object.subgraphs[0]
    canonical_fc = _find_projection(canonical_model_object, canonical_graph, layer, role, adapter)
    unpadded_fc = _find_projection(unpadded_model_object, unpadded_graph, layer, role, adapter)
    padded_fc = _find_projection(padded_model_object, padded_graph, layer, role, adapter)
    reference_meta = _read_json(reference_json, "NPU_CANARY_REFERENCE_METADATA_INVALID")
    reference_output_names = set(reference_meta.get("outputs", {}))
    unpadded_input_names = {
        _name(unpadded_graph.tensors[int(index)].name) for index in unpadded_graph.inputs
    }
    padded_input_names = {
        _name(padded_graph.tensors[int(index)].name) for index in padded_graph.inputs
    }
    activation_candidates = sorted(unpadded_input_names & padded_input_names & reference_output_names)
    if len(activation_candidates) != 1:
        raise ToolError(
            "NPU_CANARY_ACTIVATION_MAPPING_AMBIGUOUS",
            "expected one rank input shared by both candidates and Canonical fixture outputs; found {}".format(
                activation_candidates
            ),
        )
    activation_name = activation_candidates[0]
    activation_index = _find_unique_tensor(canonical_graph, activation_name)
    canonical_fc_input_name = _name(canonical_graph.tensors[canonical_fc["activation_index"]].name)
    unpadded_fc_input_name = _name(unpadded_graph.tensors[unpadded_fc["activation_index"]].name)
    padded_fc_input_name = _name(padded_graph.tensors[padded_fc["activation_index"]].name)
    activation_input_names = {"unpadded": activation_name, "padded": activation_name}
    canonical_taps = [q_tensor_name, activation_name, canonical_fc_input_name]
    unpadded_taps = [q_tensor_name, unpadded_fc_input_name]
    padded_taps = [q_tensor_name, padded_fc_input_name]
    contract = _fixture_contract(
        input_json,
        reference_json,
        input_npz,
        reference_npz,
        activation_name,
        q_tensor_name,
    )
    if contract["profile"] != profile or contract["reference_profile"] != profile:
        raise ToolError("NPU_CANARY_FIXTURE_PROFILE_MISMATCH", profile)
    if contract["fixture_sha256"] != contract["reference_fixture_sha256"]:
        raise ToolError("NPU_CANARY_FIXTURE_REFERENCE_MISMATCH", profile)

    # Stage and verify all tap models before atomically publishing the bundle.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".npu-canary-", dir=str(output_root.parent)))
    try:
        (staging / "models").mkdir()
        canonical_path = staging / "models" / "canonical_q.tflite"
        unpadded_path = staging / "models" / "rank_local_unpadded_q.tflite"
        padded_path = staging / "models" / "rank_local_padded_q.tflite"
        canonical_checked, _, canonical_integrity_sha = _expose_outputs(adapter, canonical_model, canonical_path, canonical_taps)
        unpadded_checked, _, unpadded_integrity_sha = _expose_outputs(adapter, unpadded_model_path, unpadded_path, unpadded_taps)
        padded_checked, _, padded_integrity_sha = _expose_outputs(adapter, padded_model_path, padded_path, padded_taps)
        canonical_graph_checked = canonical_checked.subgraphs[0]
        unpadded_graph_checked = unpadded_checked.subgraphs[0]
        padded_graph_checked = padded_checked.subgraphs[0]
        canonical_q = _tensor_contract(
            canonical_graph_checked.tensors[_find_unique_tensor(canonical_graph_checked, q_tensor_name)], adapter.schema
        )
        unpadded_q = _tensor_contract(
            unpadded_graph_checked.tensors[_find_unique_tensor(unpadded_graph_checked, q_tensor_name)], adapter.schema
        )
        padded_q = _tensor_contract(
            padded_graph_checked.tensors[_find_unique_tensor(padded_graph_checked, q_tensor_name)], adapter.schema
        )
        if canonical_q["dtype"] != "INT16" or unpadded_q["dtype"] != "INT16" or padded_q["dtype"] != "INT16":
            raise ToolError("NPU_CANARY_Q_DTYPE_UNSUPPORTED", "Q tap must be INT16")
        if canonical_q["quantization"] != unpadded_q["quantization"] or canonical_q["quantization"] != padded_q["quantization"]:
            raise ToolError("NPU_CANARY_Q_QUANTIZATION_MISMATCH", profile)
        full_width = int(canonical_q["shape"][-1])
        local_width = int(unpadded_q["shape"][-1])
        if int(padded_q["shape"][-1]) != local_width or full_width != 4 * local_width:
            raise ToolError(
                "NPU_CANARY_Q_SHAPE_MISMATCH",
                "canonical={}, unpadded={}, padded={}".format(full_width, local_width, padded_q["shape"][-1]),
            )
        canonical_activation = _tensor_contract(canonical_graph_checked.tensors[activation_index], adapter.schema)
        unpadded_activation = _tensor_contract(
            unpadded_graph_checked.tensors[_find_unique_tensor(unpadded_graph_checked, activation_name)], adapter.schema
        )
        padded_activation = _tensor_contract(
            padded_graph_checked.tensors[_find_unique_tensor(padded_graph_checked, activation_name)], adapter.schema
        )
        canonical_activation_numeric = {key: value for key, value in canonical_activation.items() if key != "name"}
        unpadded_activation_numeric = {key: value for key, value in unpadded_activation.items() if key != "name"}
        padded_activation_numeric = {key: value for key, value in padded_activation.items() if key != "name"}
        if canonical_activation_numeric != unpadded_activation_numeric or canonical_activation_numeric != padded_activation_numeric:
            raise ToolError("NPU_CANARY_ACTIVATION_CONTRACT_MISMATCH", profile)
        canonical_fc_input = _tensor_contract(
            canonical_graph_checked.tensors[_find_unique_tensor(canonical_graph_checked, canonical_fc_input_name)], adapter.schema
        )
        unpadded_fc_input = _tensor_contract(
            unpadded_graph_checked.tensors[_find_unique_tensor(unpadded_graph_checked, unpadded_fc_input_name)], adapter.schema
        )
        padded_fc_input = _tensor_contract(
            padded_graph_checked.tensors[_find_unique_tensor(padded_graph_checked, padded_fc_input_name)], adapter.schema
        )
        fc_input_numeric = [
            {key: value for key, value in item.items() if key != "name"}
            for item in (canonical_fc_input, unpadded_fc_input, padded_fc_input)
        ]
        if not (fc_input_numeric[0] == fc_input_numeric[1] == fc_input_numeric[2]):
            raise ToolError("NPU_CANARY_FC_INPUT_CONTRACT_MISMATCH", profile)

        fixture_input_meta = _read_json(input_json, "NPU_CANARY_FIXTURE_METADATA_INVALID")["inputs"]
        expected_fixture_inputs = set(fixture_input_meta)
        for label, graph in (("canonical", canonical_graph_checked), ("unpadded", unpadded_graph_checked), ("padded", padded_graph_checked)):
            actual_names = {_name(graph.tensors[int(index)].name) for index in graph.inputs}
            allowed_inputs = {activation_name}
            if label in activation_input_names:
                allowed_inputs.add(activation_input_names[label])
            missing = actual_names - expected_fixture_inputs - allowed_inputs
            if missing:
                raise ToolError(
                    "NPU_CANARY_PUBLIC_INPUT_UNBOUND",
                    "{} has unbound inputs {}".format(label, sorted(missing)),
                )

        fixture_dir = staging / "fixture"
        fixture_dir.mkdir()
        input_row = _copy_checked(input_npz, fixture_dir / "inputs.npz")
        input_meta_row = _copy_checked(input_json, fixture_dir / "inputs.json")
        reference_row = _copy_checked(reference_npz, fixture_dir / "canonical_reference.npz")
        reference_meta_row = _copy_checked(reference_json, fixture_dir / "canonical_reference.json")
        model_rows = {}
        checked_graphs = {
            "canonical": canonical_graph_checked,
            "unpadded": unpadded_graph_checked,
            "padded": padded_graph_checked,
        }
        for label, model_path in (
            ("canonical", canonical_path),
            ("unpadded", unpadded_path),
            ("padded", padded_path),
        ):
            graph = checked_graphs[label]
            model_rows[label] = {
                "path": model_path.relative_to(staging).as_posix(),
                "bytes": model_path.stat().st_size,
                "sha256": _sha256(model_path),
                "operator_graph_weights_sha256": {
                    "canonical": canonical_integrity_sha,
                    "unpadded": unpadded_integrity_sha,
                    "padded": padded_integrity_sha,
                }[label],
                "inputs": [_tensor_contract(graph.tensors[int(index)], adapter.schema) for index in graph.inputs],
                "outputs": [_tensor_contract(graph.tensors[int(index)], adapter.schema) for index in graph.outputs],
            }
        manifest = {
            "schema": 1,
            "phase": "2E",
            "status": "PREPARED_NOT_RUN",
            "profile": profile,
            "layer": int(layer),
            "rank": int(rank),
            "role": role,
            "source_canonical_sha256": expected_canonical_sha,
            "fixture": {
                "label": contract["fixture_label"],
                "fixture_sha256": contract["fixture_sha256"],
                "position_start": contract["position_start"],
                "valid_past_kv_length": contract["valid_past_kv_length"],
                "input_archive": input_row,
                "input_metadata": input_meta_row,
                "canonical_reference_archive": reference_row,
                "canonical_reference_metadata": reference_meta_row,
                "canonical_q_reference": contract["canonical_q_reference"],
                "canonical_activation_reference": contract["canonical_activation_reference"],
            },
            "output_taps": {
                "canonical_q": q_tensor_name,
            "canonical_activation": activation_name,
            "canonical_fc_input": canonical_fc_input_name,
            "unpadded_fc_input": unpadded_fc_input_name,
            "padded_fc_input": padded_fc_input_name,
                "rank_q": q_tensor_name,
                "activation_input_mapping": {
                    "canonical_output_name": activation_name,
                    "unpadded_input_name": activation_input_names["unpadded"],
                    "padded_input_name": activation_input_names["padded"],
                    "numeric_contract_equal": True,
                },
                "canonical_channel_range_for_rank": [int(rank) * local_width, (int(rank) + 1) * local_width],
            },
            "models": model_rows,
            "capture_contract": {
                "canonical_npz_keys": ["q_full", "activation", "fc_input"],
                "rank_npz_keys": ["q_rank", "fc_input"],
                "capture_sidecar_fields": [
                    "execution_backend", "model_sha256", "fixture_sha256",
                    "activation_input_sha256", "latency_us", "dla_size_bytes", "memory_bytes",
                ],
                "activation_input_policy": "Use the canonical stage activation output from this run as the matching rank-model input; compare the actual FC input tensors as a separate boundary.",
            },
            "ncc_compatibility": "NOT_RUN",
            "npu_verification_status": "NPU_UNVERIFIED",
        }
        (staging / "NPU_CANARY_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        capture_dir = staging / "capture_templates"
        capture_dir.mkdir()
        for label in ("canonical", "unpadded", "padded"):
            template = {
                "execution_backend": "UNCONFIRMED",
                "model_sha256": model_rows[label]["sha256"],
                "fixture_sha256": contract["fixture_sha256"],
                "activation_input_sha256": None if label == "canonical" else "FILL_FROM_CANONICAL_CAPTURE",
                "ncc_compile_status": "NOT_RUN",
                "ncc_optimization_notes": None,
                "latency_us": None,
                "dla_size_bytes": None,
                "memory_bytes": None,
            }
            (capture_dir / "{}.json".format(label)).write_text(
                json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        (staging / "NPU_CANARY_RUNBOOK.md").write_text(
            "\n".join([
                "# NPU FC Canary Runbook",
                "",
                "1. Confirm the named EVK is authorized and idle through the approved device-management path.",
                "2. Compile and run `models/canonical_q.tflite` with `fixture/inputs.npz`.",
                "3. Save its tapped outputs as `q_full`, `activation`, and `fc_input` in an NPZ archive.",
                "4. Feed the canonical `activation` output to the matching input named in `NPU_CANARY_MANIFEST.json`; reuse the raw mask/KV/position fixture inputs.",
                "5. Run the unpadded and padded models separately. Save `q_rank` and `fc_input` for each.",
                "6. Fill each capture sidecar with the actual backend, model/fixture SHA256, activation SHA256, timing, DLA size, memory, and NCC notes.",
                "7. Run `tflite-tp compare-npu-canary` and preserve its report.",
                "",
                "NCC compile success is recorded as NCC_COMPATIBLE only. The NPU exact gate requires Canonical output to match its saved reference, the FC input activations to match, and the padded rank output to match the corresponding Canonical channel slice.",
                "",
            ]),
            encoding="utf-8",
        )
        sums = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                sums.append("{}  {}\n".format(_sha256(path), path.relative_to(staging).as_posix()))
        (staging / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
        staging.rename(output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _array_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _first_difference(expected, actual):
    if expected.shape != actual.shape:
        return {"shape_mismatch": {"expected": list(expected.shape), "actual": list(actual.shape)}}
    different = np.argwhere(expected != actual)
    if len(different) == 0:
        return None
    index = tuple(int(item) for item in different[0])
    return {
        "index": list(index),
        "expected": int(expected[index]),
        "actual": int(actual[index]),
    }


def _compare_exact(expected, actual):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        return {
            "mismatch_count": None,
            "max_abs_diff": None,
            "first_failure": {"shape_mismatch": {"expected": list(expected.shape), "actual": list(actual.shape)}},
            "exact": False,
        }
    diff = np.abs(expected.astype(np.int64) - actual.astype(np.int64))
    mismatch = expected != actual
    return {
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "max_abs_diff": int(diff.max()) if diff.size else 0,
        "first_failure": _first_difference(expected, actual),
        "exact": not bool(np.any(mismatch)),
    }


def compare_canary_outputs(bundle_directory, canonical_npz, unpadded_npz, padded_npz,
                           canonical_capture_json, unpadded_capture_json, padded_capture_json,
                           output_report):
    """Compare actual NPU captures and fail closed on any SHA/contract mismatch."""
    root = Path(bundle_directory)
    manifest = _read_json(root / "NPU_CANARY_MANIFEST.json", "NPU_CANARY_MANIFEST_INVALID")
    fixture = manifest["fixture"]
    captures = {
        "canonical": (Path(canonical_npz), Path(canonical_capture_json), "q_full"),
        "unpadded": (Path(unpadded_npz), Path(unpadded_capture_json), "q_rank"),
        "padded": (Path(padded_npz), Path(padded_capture_json), "q_rank"),
    }
    loaded = {}
    sidecars = {}
    for label, (archive_path, sidecar_path, _q_key) in captures.items():
        sidecar = _read_json(sidecar_path, "NPU_CANARY_CAPTURE_METADATA_INVALID")
        if sidecar.get("execution_backend") != "NPU":
            raise ToolError("NPU_CANARY_CAPTURE_NOT_NPU", "{} backend is not NPU".format(label))
        if sidecar.get("model_sha256") != manifest["models"][label]["sha256"]:
            raise ToolError("NPU_CANARY_CAPTURE_MODEL_SHA_MISMATCH", label)
        if sidecar.get("fixture_sha256") != fixture["fixture_sha256"]:
            raise ToolError("NPU_CANARY_CAPTURE_FIXTURE_SHA_MISMATCH", label)
        try:
            loaded[label] = np.load(archive_path, allow_pickle=False)
        except Exception as exc:
            raise ToolError("NPU_CANARY_CAPTURE_ARCHIVE_INVALID", "{}: {}".format(label, exc)) from exc
        sidecars[label] = sidecar
    canonical_keys = set(loaded["canonical"].files)
    if not {"q_full", "activation", "fc_input"}.issubset(canonical_keys):
        raise ToolError("NPU_CANARY_CANONICAL_OUTPUT_KEYS_MISSING", str(sorted(canonical_keys)))
    for label in ("unpadded", "padded"):
        if not {"q_rank", "fc_input"}.issubset(set(loaded[label].files)):
            raise ToolError("NPU_CANARY_RANK_OUTPUT_KEY_MISSING", "{} requires q_rank and fc_input".format(label))
    activation_sha = _array_sha256(loaded["canonical"]["activation"])
    for label in ("unpadded", "padded"):
        if sidecars[label].get("activation_input_sha256") != activation_sha:
            raise ToolError("NPU_CANARY_ACTIVATION_SHA_MISMATCH", label)
    reference_path = root / fixture["canonical_reference_archive"]["path"]
    q_ref_key = fixture["canonical_q_reference"]["archive_key"]
    try:
        with np.load(reference_path, allow_pickle=False) as reference_archive:
            q_host = reference_archive[q_ref_key]
    except Exception as exc:
        raise ToolError("NPU_CANARY_REFERENCE_ARCHIVE_INVALID", str(exc)) from exc
    q_full = loaded["canonical"]["q_full"]
    start, end = manifest["output_taps"]["canonical_channel_range_for_rank"]
    q_rank_ref = q_full[..., int(start):int(end)]
    results = {
        "schema": 1,
        "phase": "2E",
        "profile": manifest["profile"],
        "layer": manifest["layer"],
        "rank": manifest["rank"],
        "role": manifest["role"],
        "execution_backend": "NPU",
        "fixture_sha256": fixture["fixture_sha256"],
        "capture_files": {
            label: {
                "archive_path": archive_path.name,
                "archive_sha256": _sha256(archive_path),
                "sidecar_path": sidecar_path.name,
                "sidecar_sha256": _sha256(sidecar_path),
            }
            for label, (archive_path, sidecar_path, _q_key) in captures.items()
        },
        "canonical_host_reference": _compare_exact(q_host, q_full),
        "unpadded_fc_input_vs_canonical_npu": _compare_exact(
            loaded["canonical"]["fc_input"], loaded["unpadded"]["fc_input"]
        ),
        "padded_fc_input_vs_canonical_npu": _compare_exact(
            loaded["canonical"]["fc_input"], loaded["padded"]["fc_input"]
        ),
        "unpadded_vs_canonical_npu": _compare_exact(q_rank_ref, loaded["unpadded"]["q_rank"]),
        "padded_vs_canonical_npu": _compare_exact(q_rank_ref, loaded["padded"]["q_rank"]),
        "activation_input_sha256": activation_sha,
        "runtime_metrics": {
            label: {
                "latency_us": sidecars[label].get("latency_us"),
                "dla_size_bytes": sidecars[label].get("dla_size_bytes"),
                "memory_bytes": sidecars[label].get("memory_bytes"),
                "ncc_optimization_notes": sidecars[label].get("ncc_optimization_notes"),
            }
            for label in captures
        },
        "ncc_compatibility": "NCC_COMPATIBLE" if all(sidecars[label].get("ncc_compile_status") == "PASS" for label in captures) else "NOT_CONFIRMED",
    }
    results["canonical_npu_exact_status"] = (
        "PASS" if results["canonical_host_reference"]["exact"] else "FAIL"
    )
    results["unpadded_npu_exact_status"] = (
        "PASS"
        if results["unpadded_fc_input_vs_canonical_npu"]["exact"]
        and results["unpadded_vs_canonical_npu"]["exact"]
        else "FAIL"
    )
    results["padded_npu_exact_status"] = (
        "PASS"
        if results["padded_fc_input_vs_canonical_npu"]["exact"]
        and results["padded_vs_canonical_npu"]["exact"]
        else "FAIL"
    )
    verified = (
        results["canonical_npu_exact_status"] == "PASS"
        and results["padded_npu_exact_status"] == "PASS"
    )
    results["npu_exact_status"] = "NPU_EXACT_PASS" if verified else "NPU_EXACT_FAIL"
    results["npu_verification_status"] = "NPU_VERIFIED" if verified else "NPU_UNVERIFIED"
    comparisons = (
        results["canonical_host_reference"],
        results["unpadded_fc_input_vs_canonical_npu"],
        results["padded_fc_input_vs_canonical_npu"],
        results["unpadded_vs_canonical_npu"],
        results["padded_vs_canonical_npu"],
    )
    results["mismatch_count"] = (
        sum(item["mismatch_count"] for item in comparisons)
        if all(item["mismatch_count"] is not None for item in comparisons)
        else None
    )
    results["max_abs_diff"] = (
        max(item["max_abs_diff"] for item in comparisons)
        if all(item["max_abs_diff"] is not None for item in comparisons)
        else None
    )
    failures = []
    for gate in (
        "canonical_host_reference",
        "unpadded_fc_input_vs_canonical_npu",
        "padded_fc_input_vs_canonical_npu",
        "unpadded_vs_canonical_npu",
        "padded_vs_canonical_npu",
    ):
        if not results[gate]["exact"]:
            failures.append({"gate": gate, **(results[gate]["first_failure"] or {})})
    results["first_failure"] = failures[0] if failures else None
    Path(output_report).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def comparator_self_test():
    expected = np.asarray([[[1, -2, 7], [8, 9, 0]]], dtype=np.int16)
    exact = _compare_exact(expected, expected.copy())
    changed = expected.copy()
    changed[0, 1, 2] = 1
    mismatch = _compare_exact(expected, changed)
    return {
        "exact_case_pass": exact["exact"] and exact["mismatch_count"] == 0,
        "mismatch_case_caught": not mismatch["exact"] and mismatch["mismatch_count"] == 1,
        "first_failure_correct": mismatch["first_failure"] == {"index": [0, 1, 2], "expected": 0, "actual": 1},
        "status": "PASS" if exact["exact"] and mismatch["mismatch_count"] == 1 else "FAIL",
    }
