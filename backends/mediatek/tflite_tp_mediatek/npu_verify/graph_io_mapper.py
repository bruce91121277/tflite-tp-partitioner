"""Safe, manifest-bound TFLite intermediate-output exposure for NPU Golden capture."""

import json
import struct
from pathlib import Path

from .contracts import load_tflite_contract, sha256_file
from tflite_tp.tflite_reader import read_tflite


def expose_tensors(source, destination, tensor_names, manifest_path, expected_source_sha256=None):
    """Append graph tensor indices to public outputs without editing operators or buffers."""
    try:
        from tflite.Model import Model
    except ImportError as exc:
        raise RuntimeError("TFLITE_SCHEMA_UNAVAILABLE") from exc
    source = Path(source)
    destination = Path(destination)
    manifest_path = Path(manifest_path)
    if not source.is_file():
        raise ValueError("TAP_SOURCE_MISSING")
    if destination.exists() or manifest_path.exists():
        raise ValueError("TAP_DESTINATION_EXISTS")
    source_bytes = source.read_bytes()
    source_sha = sha256_file(source)
    if expected_source_sha256 and source_sha != expected_source_sha256:
        raise ValueError("TAP_SOURCE_SHA_MISMATCH")
    model = Model.GetRootAsModel(source_bytes, 0)
    if model.SubgraphsLength() != 1:
        raise ValueError("TAP_SUBGRAPH_COUNT_UNSUPPORTED")
    graph = model.Subgraphs(0)
    matches = {}
    requested = list(dict.fromkeys(str(name) for name in tensor_names))
    if len(requested) != len(tensor_names) or not requested:
        raise ValueError("TAP_TENSOR_NAME_LIST_INVALID")
    for index in range(graph.TensorsLength()):
        tensor = graph.Tensors(index)
        raw_name = tensor.Name()
        name = raw_name.decode("utf-8", errors="replace") if raw_name else ""
        if name in requested:
            matches.setdefault(name, []).append(index)
    missing = [name for name in requested if name not in matches]
    ambiguous = {name: rows for name, rows in matches.items() if len(rows) != 1}
    if missing or ambiguous:
        raise ValueError("TAP_TENSOR_RESOLUTION_FAIL missing={} ambiguous={}".format(missing, ambiguous))
    current = [int(graph.Outputs(i)) for i in range(graph.OutputsLength())]
    existing = set(current)
    appended = [name for name in requested if int(matches[name][0]) not in existing]
    new_outputs = current + [int(matches[name][0]) for name in requested if int(matches[name][0]) not in existing]
    if len(new_outputs) == len(current):
        raise ValueError("TAP_REQUESTS_ALREADY_PUBLIC")
    table = graph._tab
    field_offset = int(table.Offset(8))
    if not field_offset:
        raise ValueError("TAP_OUTPUT_VECTOR_FIELD_MISSING")
    field_location = int(table.Pos) + field_offset
    vector_blob = struct.pack("<I", len(new_outputs)) + b"".join(struct.pack("<i", value) for value in new_outputs)
    patched = bytearray(source_bytes)
    padded = (-len(patched)) % 4
    if padded:
        patched.extend(b"\x00" * padded)
    vector_location = len(patched)
    patched.extend(vector_blob)
    relative = vector_location - field_location
    if relative <= 0 or relative > 0xFFFFFFFF:
        raise ValueError("TAP_OUTPUT_OFFSET_INVALID")
    patched[field_location:field_location + 4] = struct.pack("<I", relative)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(patched)
    reparsed = Model.GetRootAsModel(bytes(patched), 0)
    rebuilt_graph = reparsed.Subgraphs(0)
    outputs = [int(rebuilt_graph.Outputs(i)) for i in range(rebuilt_graph.OutputsLength())]
    if outputs != new_outputs:
        destination.unlink(missing_ok=True)
        raise ValueError("TAP_OUTPUT_VECTOR_REPARSE_FAIL")
    before = read_tflite(source)
    after = read_tflite(destination)
    before_graph, after_graph = before.graphs[0], after.graphs[0]
    structural = (
        before.version == after.version
        and before.buffer_count == after.buffer_count
        and len(before_graph.tensors) == len(after_graph.tensors)
        and len(before_graph.operators) == len(after_graph.operators)
        and before_graph.inputs == after_graph.inputs
        and after_graph.outputs == tuple(new_outputs)
    )
    if not structural:
        destination.unlink(missing_ok=True)
        raise ValueError("TAP_GRAPH_STRUCTURE_CHANGED")
    contract = load_tflite_contract(destination)
    added_contracts = []
    for name in appended:
        item = next((row for row in contract["outputs"] if row["name"] == name), None)
        if item is None:
            raise ValueError("TAP_ADDED_OUTPUT_MISSING:" + name)
        added_contracts.append(item)
    result = {
        "schema": "tflite_tp.phase3.canonical-output-tap.v1",
        "status": "STRUCTURAL_PASS",
        "numerical_gate": "NOT_RUN",
        "source_path": str(source),
        "source_sha256": source_sha,
        "source_bytes": len(source_bytes),
        "destination_path": str(destination),
        "destination_sha256": contract["sha256"],
        "destination_bytes": contract["bytes"],
        "original_output_indices": current,
        "destination_output_indices": new_outputs,
        "requested_tensor_names": requested,
        "newly_exposed_tensors": added_contracts,
        "operator_count_before_after": [len(before_graph.operators), len(after_graph.operators)],
        "tensor_count_before_after": [len(before_graph.tensors), len(after_graph.tensors)],
        "buffer_count_before_after": [before.buffer_count, after.buffer_count],
        "source_unchanged": sha256_file(source) == source_sha,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
