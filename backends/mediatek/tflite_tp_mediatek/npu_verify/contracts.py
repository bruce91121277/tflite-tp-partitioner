"""TFLite and partition-manifest contract resolution."""

import hashlib
import json
from pathlib import Path

from tflite_tp.tflite_reader import read_tflite


DTYPE_NAMES = {
    "FLOAT16": "<f2",
    "FLOAT32": "<f4",
    "FLOAT64": "<f8",
    "INT8": "i1",
    "UINT8": "u1",
    "INT16": "<i2",
    "UINT16": "<u2",
    "INT32": "<i4",
    "UINT32": "<u4",
    "INT64": "<i8",
    "UINT64": "<u8",
    "BOOL": "?",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonable_quantization(value):
    if value is None:
        return None
    return {
        "scales": [float(x) for x in value.scales],
        "zero_points": [int(x) for x in value.zero_points],
        "quantized_dimension": int(value.quantized_dimension),
        "minimum": [float(x) for x in value.minimum],
        "maximum": [float(x) for x in value.maximum],
    }


def normalize_dtype(value):
    name = str(value).upper().replace("_LE", "")
    if name not in DTYPE_NAMES:
        raise ValueError("UNSUPPORTED_TFLITE_DTYPE: " + str(value))
    return name


def tensor_contract(tensor):
    dtype = normalize_dtype(tensor.dtype)
    return {
        "index": int(tensor.index),
        "name": str(tensor.name),
        "shape": [int(x) for x in tensor.shape],
        "dtype": dtype,
        "quantization": jsonable_quantization(tensor.quantization),
    }


def load_tflite_contract(model_path):
    """Read public input/output contracts from the actual FlatBuffer."""
    path = Path(model_path)
    info = read_tflite(path)
    if len(info.graphs) != 1:
        raise ValueError("UNSUPPORTED_SUBGRAPH_COUNT: {}".format(len(info.graphs)))
    graph = info.graphs[0]
    inputs = [tensor_contract(graph.tensors[index]) for index in graph.inputs]
    outputs = [tensor_contract(graph.tensors[index]) for index in graph.outputs]
    if not inputs or not outputs:
        raise ValueError("PUBLIC_IO_CONTRACT_EMPTY")
    names = [row["name"] for row in inputs]
    if len(names) != len(set(names)):
        raise ValueError("DUPLICATE_PUBLIC_INPUT_NAME")
    names = [row["name"] for row in outputs]
    if len(names) != len(set(names)):
        raise ValueError("DUPLICATE_PUBLIC_OUTPUT_NAME")
    return {
        "path": str(path),
        "sha256": info.source_sha256,
        "bytes": info.source_bytes,
        "model_version": info.version,
        "operator_count": len(graph.operators),
        "inputs": inputs,
        "outputs": outputs,
    }


def resolve_candidate(manifest_path, tp, profile, layer, rank, component):
    """Resolve one artifact only through the partitioner's checked manifest."""
    path = Path(manifest_path)
    doc = read_json(path)
    topology = str(doc.get("topology", ""))
    expected_topology = {2: "TP2_ROW_REDUCE", 4: "TP4_FOUR_STAGE_ALL_GATHER"}.get(int(tp))
    if expected_topology is None or topology != expected_topology:
        raise ValueError("TOPOLOGY_CONTRACT_MISMATCH")
    if str(doc.get("profile", "")).upper() != str(profile).upper():
        raise ValueError("PROFILE_CONTRACT_MISMATCH")
    matches = [
        row for row in doc.get("models", [])
        if int(row.get("layer", -1)) == int(layer)
        and int(row.get("rank", -1)) == int(rank)
        and str(row.get("component", "")) == str(component)
        and str(row.get("profile", "")).upper() == str(profile).upper()
    ]
    if len(matches) != 1:
        raise ValueError("CANDIDATE_MANIFEST_MATCH_COUNT:{}".format(len(matches)))
    row = matches[0]
    model_path = (path.parent / row["path"]).resolve()
    if not model_path.is_file():
        raise ValueError("CANDIDATE_MODEL_MISSING: " + str(model_path))
    actual_sha = sha256_file(model_path)
    if actual_sha != row.get("output_sha256"):
        raise ValueError("CANDIDATE_MODEL_SHA_MISMATCH")
    if int(row.get("output_bytes", -1)) != model_path.stat().st_size:
        raise ValueError("CANDIDATE_MODEL_SIZE_MISMATCH")
    actual_contract = load_tflite_contract(model_path)
    declared_io = row.get("io", {})
    for key in ("inputs", "outputs"):
        actual = [(x["index"], x["name"], x["shape"], x["dtype"]) for x in actual_contract[key]]
        declared = [(int(x["index"]), str(x["name"]), [int(v) for v in x["shape"]], normalize_dtype(x["dtype"]))
                    for x in declared_io.get(key, [])]
        if actual != declared:
            raise ValueError("CANDIDATE_{}_CONTRACT_MISMATCH".format(key.upper()))
    return model_path, row, actual_contract
