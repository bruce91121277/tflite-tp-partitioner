"""Build TP4 tail candidates from the identified quantized Llama tail."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid

import numpy as np

from .errors import ToolError
from .flatbuffer_adapter import TFLiteObjectAdapter
from .model import QuantizationInfo
from .tflite_rewriter import (_assign_quantization, _compact_graph_buffers,
                              _replace_buffer_data, _quant_info, _set_shape,
                              _shape, _slice_bias, _slice_quantization,
                              _tensor_data)


def _set_tail_input_scale(tensor, scale):
    """Set one symmetric INT16 tail input scale and return its manifest record."""
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        raise ToolError("TAIL_INPUT_SCALE_INVALID", "input scale must be a finite positive number")
    if not math.isfinite(scale) or scale <= 0:
        raise ToolError("TAIL_INPUT_SCALE_INVALID", "input scale must be a finite positive number")
    current = _quant_info(getattr(tensor, "quantization", None))
    if current is None or len(current.scales) != 1 or current.zero_points != (0,):
        raise ToolError("TAIL_INPUT_QUANTIZATION_UNSUPPORTED", "expected scalar symmetric input quantization")
    updated = QuantizationInfo(
        scales=(scale,), zero_points=(0,),
        quantized_dimension=current.quantized_dimension,
        minimum=(-32768.0 * scale,), maximum=(32767.0 * scale,),
    )
    _assign_quantization(tensor.quantization, updated)
    assigned = _quant_info(tensor.quantization)
    return {"scale": assigned.scales[0], "zero_point": 0, "dtype": "INT16"}


def compare_tail_logits(golden, ranks, golden_input, candidate_input):
    degree = len(ranks)
    if degree not in (2, 4, 8):
        raise ToolError("TAIL_RANK_COUNT_INVALID", "expected 2, 4, or 8 rank outputs")
    golden_input_sha = hashlib.sha256(Path(golden_input).read_bytes()).hexdigest()
    candidate_input_sha = hashlib.sha256(Path(candidate_input).read_bytes()).hexdigest()
    if golden_input_sha != candidate_input_sha:
        raise ToolError("TAIL_INPUT_PROVENANCE_MISMATCH", "golden and candidate tail inputs differ")
    expected = np.fromfile(golden, dtype="<i2")
    shards = [np.fromfile(path, dtype="<i2") for path in ranks]
    if (expected.size == 0 or expected.size % degree or
        any(shard.size != expected.size // degree for shard in shards)):
        raise ToolError("TAIL_LOGITS_SHAPE_INVALID", "golden vocabulary must divide evenly across rank outputs")
    count = expected.size // degree
    actual = np.concatenate(shards)
    delta = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
    return {"status": "PASS" if not np.any(delta) else "MISMATCH",
            "scope": "logits comparison on byte-identical tail inputs",
            "input_sha256": golden_input_sha,
            "golden_sha256": hashlib.sha256(Path(golden).read_bytes()).hexdigest(),
            "rank_sha256": [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in ranks],
            "mismatch_count": int(np.count_nonzero(delta)), "max_abs": int(delta.max()),
            "per_rank_mismatch": [int(np.count_nonzero(delta[i * count:(i + 1) * count])) for i in range(degree)],
            "candidate_token": int(actual.argmax()), "golden_token": int(expected.argmax())}


def build_tail_tp4(source, output, degree=4, input_scale=None):
    if degree not in (2, 4, 8):
        raise ToolError("TAIL_RANK_COUNT_INVALID", "expected TP2, TP4, or TP8")
    source, output = Path(source), Path(output)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if output.exists():
        raise ToolError("OUTPUT_EXISTS", "refusing to overwrite {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / ".tail-tp4-{}".format(uuid.uuid4().hex)
    temporary.mkdir()
    adapter = TFLiteObjectAdapter()
    weight_hash, bias_hash = hashlib.sha256(), hashlib.sha256()
    expected_bias_hash = None
    input_quantization = None
    files = []
    try:
        for rank in range(degree):
            model = adapter.load(source)
            if len(model.subgraphs) != 1:
                raise ToolError("TAIL_GRAPH_INVALID", "expected one subgraph")
            graph = model.subgraphs[0]
            tensors = graph.tensors
            if (list(graph.inputs) != [0] or list(graph.outputs) != [3] or
                len(graph.operators) != 4 or len(tensors) < 5):
                raise ToolError("TAIL_GRAPH_INVALID", "unexpected tail graph contract")
            # shortcut: keeps the current four-op tensor-index layout; extend discovery when model exports change it.
            weight, bias = tensors[4], tensors[2]
            if int(tensors[0].type) != int(adapter.schema.TensorType.INT16):
                raise ToolError("TAIL_INPUT_DTYPE_INVALID", "tail input must be INT16")
            output_producers = [op for op in graph.operators if 3 in [int(index) for index in op.outputs]]
            weight_shape = _shape(weight)
            if len(weight_shape) != 2:
                raise ToolError("TAIL_GRAPH_INVALID", "LM head weight must be a rank-2 matrix")
            vocab, hidden = weight_shape
            input_shape, output_shape = _shape(tensors[0]), _shape(tensors[3])
            if (vocab <= 0 or hidden <= 0 or
                _shape(bias) not in ((vocab,), (1, vocab)) or
                len(input_shape) != 3 or input_shape[0] != 1 or input_shape[2] != hidden or
                input_shape[1] <= 0 or len(output_shape) != 3 or
                output_shape != (input_shape[0], input_shape[1], vocab) or
                len(output_producers) != 1 or
                4 not in [int(index) for index in output_producers[0].inputs] or
                2 not in [int(index) for index in output_producers[0].inputs]):
                raise ToolError("TAIL_GRAPH_INVALID", "tail input, output, or projection dimensions are inconsistent")
            if vocab % degree:
                raise ToolError("TAIL_VOCAB_NOT_DIVISIBLE", "vocabulary size {} cannot be divided by TP{}".format(vocab, degree))
            if hidden % 2:
                raise ToolError("TAIL_WEIGHT_SIZE_INVALID", "INT4 weight rows require an even hidden size")
            if int(weight.type) != int(adapter.schema.TensorType.INT4) or int(bias.type) != int(adapter.schema.TensorType.INT32):
                raise ToolError("TAIL_DTYPE_INVALID", "expected INT4 weights and INT32 bias")
            if input_scale is not None:
                input_quantization = _set_tail_input_scale(tensors[0], input_scale)
            full_bias = _tensor_data(model, bias)
            full_bias_hash = hashlib.sha256(full_bias).hexdigest()
            if expected_bias_hash is None:
                expected_bias_hash = full_bias_hash
            elif expected_bias_hash != full_bias_hash:
                raise ToolError("TAIL_RECONSTRUCTION_MISMATCH", "tail bias differs between rank builds")
            raw_weight = _tensor_data(model, weight)
            if len(raw_weight) != vocab * hidden // 2:
                raise ToolError("TAIL_WEIGHT_SIZE_INVALID", "packed weight length is incorrect")
            count = vocab // degree
            start = rank * count
            row_bytes = hidden // 2
            shard = raw_weight[start * row_bytes:(start + count) * row_bytes]
            weight_hash.update(shard)
            expected_weight_quant = _slice_quantization(weight, weight_shape, 0, start, count)
            _replace_buffer_data(model, weight, shard)
            _set_shape(weight, (count, hidden))
            _, bias_shard = _slice_bias(model, bias, start, count, vocab)
            expected_bias_quant = _quant_info(bias.quantization)
            bias_hash.update(bias_shard)
            _set_shape(tensors[3], (input_shape[0], input_shape[1], count))
            _compact_graph_buffers(model, graph)
            target = temporary / ("rank{}_tail.tflite".format(rank))
            adapter.write(model, target)
            result = adapter.load(target)
            actual = result.subgraphs[0]
            actual_input_quantization = _quant_info(actual.tensors[0].quantization)
            expected_input_scale = input_scale
            if (_shape(actual.tensors[4]) != (count, hidden) or
                _shape(actual.tensors[3]) != (input_shape[0], input_shape[1], count) or
                (expected_input_scale is not None and
                 (actual_input_quantization is None or
                  len(actual_input_quantization.scales) != 1 or
                  not math.isclose(actual_input_quantization.scales[0], float(expected_input_scale), rel_tol=1e-6) or
                  actual_input_quantization.zero_points != (0,))) or
                _quant_info(actual.tensors[4].quantization) != expected_weight_quant or
                _quant_info(actual.tensors[2].quantization) != expected_bias_quant or
                _tensor_data(result, actual.tensors[4]) != shard or
                _tensor_data(result, actual.tensors[2]) != bias_shard):
                raise ToolError("TAIL_SERIALIZATION_MISMATCH", "rank {}".format(rank))
            files.append({"rank": rank, "vocab_start": start, "vocab_count": count,
                          "path": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        original = adapter.load(source)
        if (weight_hash.hexdigest() != hashlib.sha256(_tensor_data(original, original.subgraphs[0].tensors[4])).hexdigest() or
            bias_hash.hexdigest() != expected_bias_hash):
            raise ToolError("TAIL_RECONSTRUCTION_MISMATCH", "rank shards do not reconstruct source")
        manifest = {"source_sha256": digest, "tp_degree": degree,
                    "tail_geometry": {"vocab_size": vocab, "hidden_size": hidden, "tokens": input_shape[1]},
                    "input_quantization": input_quantization,
                    "models": files, "structural_status": "STRUCTURAL_PASS",
                    "numerical_status": "NOT_RUN", "npu_verification_status": "NPU_UNVERIFIED",
                    "release_status": "NOT_ELIGIBLE"}
        (temporary / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
