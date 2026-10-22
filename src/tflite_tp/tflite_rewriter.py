"""Rank-specific TFLite component writer for the TP4 four-AG architecture."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np

from .errors import ToolError
from .flatbuffer_adapter import (
    TFLiteObjectAdapter,
    _ObjectTable,
    normalize_host_scalar_constant_shapes,
)
from .golden import probe_golden_backend
from .model import ModelInfo, QuantizationInfo, TensorInfo, quantization_dict, validate_graph
from .packed_int4 import reconstruct_packed_int4, slice_packed_int4, slice_quantization
from .partition import ComponentSpec, derive_tp_components, derive_tp4_components
from .planning import _collective_contract
from .tflite_reader import read_tflite_bytes


_VERIFIED_FC_WIDTH_PADDING_MULTIPLIER = 2

_COMPONENT_FOR_ROLE = {
    "q": "attn",
    "k": "attn",
    "v": "attn",
    "o": "o",
    "gate": "mlp",
    "up": "mlp",
    "down": "down",
}


def _array_values(value):
    if value is None:
        return ()
    return tuple(np.asarray(value).reshape(-1).tolist())


def _quant_info(quantization):
    if quantization is None:
        return None
    return QuantizationInfo(
        scales=tuple(float(value) for value in _array_values(quantization.scale)),
        zero_points=tuple(int(value) for value in _array_values(quantization.zeroPoint)),
        quantized_dimension=int(quantization.quantizedDimension),
        minimum=tuple(float(value) for value in _array_values(quantization.min)),
        maximum=tuple(float(value) for value in _array_values(quantization.max)),
    )


def _assign_quantization(quantization, info):
    if quantization is None or info is None:
        return
    fields = (
        ("scale", info.scales, np.float32),
        ("zeroPoint", info.zero_points, np.int64),
        ("min", info.minimum, np.float32),
        ("max", info.maximum, np.float32),
    )
    for field_name, values, default_dtype in fields:
        previous = getattr(quantization, field_name, None)
        if not values:
            if previous is not None and len(previous) == 0:
                setattr(quantization, field_name, None)
            continue
        dtype = getattr(previous, "dtype", default_dtype)
        setattr(quantization, field_name, np.asarray(values, dtype=dtype).copy())


def _shape(tensor):
    return tuple(int(value) for value in _array_values(tensor.shape))


def _set_shape(tensor, shape):
    tensor.shape = np.asarray(shape, dtype=np.int32)
    if getattr(tensor, "shapeSignature", None) is not None:
        tensor.shapeSignature = np.asarray(shape, dtype=np.int32)


def _tensor_data(model, tensor):
    buffer_index = int(tensor.buffer)
    if buffer_index < 0 or buffer_index >= len(model.buffers):
        raise ToolError("TENSOR_BUFFER_OUT_OF_RANGE", str(buffer_index))
    data = model.buffers[buffer_index].data
    if data is None:
        raise ToolError("TENSOR_CONSTANT_BUFFER_MISSING", "tensor buffer {} has no data".format(buffer_index))
    return np.asarray(data, dtype=np.uint8).tobytes()


def _replace_buffer_data(model, tensor, data):
    original_index = int(tensor.buffer)
    if original_index < 0 or original_index >= len(model.buffers):
        raise ToolError("TENSOR_BUFFER_OUT_OF_RANGE", str(original_index))
    buffer_copy = copy.deepcopy(model.buffers[original_index])
    buffer_copy.data = np.frombuffer(bytes(data), dtype=np.uint8).copy()
    model.buffers.append(buffer_copy)
    tensor.buffer = len(model.buffers) - 1


def _slice_quantization(tensor, source_shape, axis, start, count):
    info = _quant_info(getattr(tensor, "quantization", None))
    if info is None:
        raise ToolError("QUANTIZATION_MISSING", "constant tensor {} has no quantization".format(tensor.name))
    sliced = slice_quantization(info, source_shape, axis, start, count)
    _assign_quantization(tensor.quantization, sliced)
    return sliced


def _slice_bias(model, tensor, start, count, expected_outputs):
    source_shape = _shape(tensor)
    if source_shape == (expected_outputs,):
        axis = 0
    elif source_shape == (1, expected_outputs):
        axis = 1
    else:
        raise ToolError("BIAS_SHAPE_UNSUPPORTED", "bias has shape {}".format(source_shape))
    raw = _tensor_data(model, tensor)
    expected_bytes = math.prod(source_shape) * 4
    if len(raw) != expected_bytes:
        raise ToolError("BIAS_BUFFER_LENGTH_MISMATCH", "bias shape requires {} bytes, found {}".format(expected_bytes, len(raw)))
    values = np.frombuffer(raw, dtype="<i4").reshape(source_shape)
    sliced_values = np.take(values, np.arange(start, start + count), axis=axis)
    sliced_bytes = sliced_values.astype("<i4", copy=False).tobytes()
    _replace_buffer_data(model, tensor, sliced_bytes)
    _slice_quantization(tensor, source_shape, axis, start, count)
    new_shape = list(source_shape)
    new_shape[axis] = count
    _set_shape(tensor, new_shape)
    return raw, sliced_bytes


def _prepare_projection_shards(model, layer, degree, schema):
    graph = model.subgraphs[layer.graph_index]
    specifications = {}
    manifest = {}
    for role, match in layer.roles.items():
        tensor = graph.tensors[match.tensor_index]
        source_shape = _shape(tensor)
        if tuple(source_shape) != tuple(match.shape):
            raise ToolError("SOURCE_WEIGHT_SHAPE_CHANGED", "{} shape changed after discovery".format(role))
        if int(tensor.type) != int(getattr(schema.TensorType, "INT4")):
            raise ToolError("PROJECTION_DTYPE_UNSUPPORTED", "{} is not INT4".format(role))
        slice_axis = 1 if degree == 2 and role in ("o", "down") else 0
        extent = source_shape[slice_axis]
        if extent % degree:
            axis_name = "input" if slice_axis == 1 else "output"
            raise ToolError("PROJECTION_NOT_DIVISIBLE", "{} {} channels do not divide by {}".format(role, axis_name, degree))
        shard_count = extent // degree
        raw = _tensor_data(model, tensor)
        weight_shards = [
            slice_packed_int4(raw, source_shape, axis=slice_axis, start=rank * shard_count, count=shard_count)
            for rank in range(degree)
        ]
        shard_shape = list(source_shape)
        shard_shape[slice_axis] = shard_count
        reconstructed = reconstruct_packed_int4(
            [(payload, tuple(shard_shape)) for payload in weight_shards],
            source_shape,
            axis=slice_axis,
        )
        if reconstructed != raw:
            raise ToolError("WEIGHT_RECONSTRUCTION_MISMATCH", "{} packed INT4 shards do not reconstruct".format(role))

        weight_quant = _quant_info(tensor.quantization)
        if weight_quant is None:
            raise ToolError("QUANTIZATION_MISSING", "{} weight has no quantization".format(role))
        quant_shards = [
            slice_quantization(weight_quant, source_shape, axis=slice_axis, start=rank * shard_count, count=shard_count)
            for rank in range(degree)
        ]
        bias_index = match.bias_tensor_index
        bias_shards = [None] * degree
        bias_hashes = [None] * degree
        bias_source_hash = None
        bias_quant_shards = [None] * degree
        bias_shapes = [None] * degree
        bias_policy = "NONE"
        if bias_index is not None:
            bias = graph.tensors[bias_index]
            if int(bias.type) != int(getattr(schema.TensorType, "INT32")):
                raise ToolError("BIAS_DTYPE_UNSUPPORTED", "{} bias is not INT32".format(role))
            bias_source = _tensor_data(model, bias)
            bias_source_hash = hashlib.sha256(bias_source).hexdigest()
            source_bias_shape = _shape(bias)
            output_extent = source_shape[0]
            if source_bias_shape not in ((output_extent,), (1, output_extent)):
                raise ToolError("BIAS_SHAPE_UNSUPPORTED", "{} bias has shape {}".format(role, source_bias_shape))
            bias_axis = 0 if source_bias_shape == (output_extent,) else 1
            values = np.frombuffer(bias_source, dtype="<i4").reshape(source_bias_shape)
            bias_quant = _quant_info(bias.quantization)
            if bias_quant is None:
                raise ToolError("BIAS_QUANTIZATION_MISSING", "{} bias has no quantization".format(role))
            if slice_axis == 0:
                bias_count = output_extent // degree
                bias_shards = [
                    np.take(values, np.arange(rank * bias_count, (rank + 1) * bias_count), axis=bias_axis)
                    .astype("<i4", copy=False)
                    .tobytes()
                    for rank in range(degree)
                ]
                bias_shapes = [
                    (bias_count,) if bias_axis == 0 else (1, bias_count)
                    for _rank in range(degree)
                ]
                joined = np.concatenate(
                    [np.frombuffer(payload, dtype="<i4").reshape(shape) for payload, shape in zip(bias_shards, bias_shapes)],
                    axis=bias_axis,
                )
                if joined.astype("<i4", copy=False).tobytes() != bias_source:
                    raise ToolError("BIAS_RECONSTRUCTION_MISMATCH", "{} bias shards do not reconstruct".format(role))
                bias_quant_shards = [
                    slice_quantization(bias_quant, source_bias_shape, axis=bias_axis, start=rank * bias_count, count=bias_count)
                    for rank in range(degree)
                ]
                bias_policy = "SLICE_WITH_OUTPUT_CHANNELS"
            else:
                bias_shards = [bias_source] + [bytes(len(bias_source)) for _rank in range(1, degree)]
                bias_shapes = [source_bias_shape] * degree
                bias_quant_shards = [bias_quant] * degree
                reconstructed_bias = np.sum(
                    [np.frombuffer(payload, dtype="<i4").reshape(source_bias_shape).astype(np.int64) for payload in bias_shards],
                    axis=0,
                )
                if reconstructed_bias.astype("<i4").tobytes() != bias_source:
                    raise ToolError("BIAS_OWNERSHIP_RECONSTRUCTION_MISMATCH", "{} rank-owned bias does not reconstruct".format(role))
                bias_policy = "RANK0_ONLY"
            bias_hashes = [hashlib.sha256(payload).hexdigest() for payload in bias_shards]

        specifications[role] = {
            "weight_tensor_index": match.tensor_index,
            "bias_tensor_index": bias_index,
            "weight_source_shape": source_shape,
            "weight_slice_axis": slice_axis,
            "weight_shard_count": shard_count,
            "weight_shard_shape": tuple(shard_shape),
            "weight_payloads": weight_shards,
            "weight_quantization": quant_shards,
            "bias_payloads": bias_shards,
            "bias_shapes": bias_shapes,
            "bias_quantization": bias_quant_shards,
        }
        manifest[role] = {
            "source_tensor": match.tensor_index,
            "source_buffer": match.weight_buffer_index,
            "source_shape": list(source_shape),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "slice_axis": slice_axis,
            "slice_axis_name": "input_channels" if slice_axis == 1 else "output_channels",
            "slice_ranges": [
                {"rank": rank, "start": rank * shard_count, "stop": (rank + 1) * shard_count}
                for rank in range(degree)
            ],
            "source_quantization": {
                "scale_count": len(weight_quant.scales),
                "quantized_dimension": weight_quant.quantized_dimension,
            },
            "weight_shards": [
                {
                    "rank": rank,
                    "shape": list(shard_shape),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "scale_count": len(quant_shards[rank].scales),
                }
                for rank, payload in enumerate(weight_shards)
            ],
            "reconstruction_equal": reconstructed == raw,
            "bias_tensor": bias_index,
            "bias_policy": bias_policy,
            "bias_source_sha256": bias_source_hash,
            "bias_shard_sha256": bias_hashes,
        }
    return specifications, manifest


def _apply_projection_shard(model, graph, layer, role, shard_spec, rank):
    match = layer.roles[role]
    weight = graph.tensors[match.tensor_index]
    source_shape = tuple(shard_spec["weight_source_shape"])
    count = int(shard_spec["weight_shard_count"])
    slice_axis = int(shard_spec["weight_slice_axis"])
    _replace_buffer_data(model, weight, shard_spec["weight_payloads"][rank])
    _assign_quantization(weight.quantization, shard_spec["weight_quantization"][rank])
    _set_shape(weight, shard_spec["weight_shard_shape"])
    if slice_axis == 0:
        for output_index in match.output_tensor_indices:
            output = graph.tensors[output_index]
            output_shape = list(_shape(output))
            if not output_shape or output_shape[-1] != source_shape[0]:
                raise ToolError("PROJECTION_OUTPUT_SHAPE_MISMATCH", "{} output shape {} does not match weight".format(role, output_shape))
            output_shape[-1] = count
            _set_shape(output, output_shape)
    else:
        activation = graph.tensors[match.activation_tensor_index]
        activation_shape = list(_shape(activation))
        if not activation_shape or activation_shape[-1] != source_shape[slice_axis]:
            raise ToolError(
                "PROJECTION_INPUT_SHAPE_MISMATCH",
                "{} activation shape {} does not match input shard axis".format(role, activation_shape),
            )
        activation_shape[-1] = count
        _set_shape(activation, activation_shape)
    bias_index = shard_spec["bias_tensor_index"]
    if bias_index is not None:
        bias = graph.tensors[bias_index]
        _replace_buffer_data(model, bias, shard_spec["bias_payloads"][rank])
        _assign_quantization(bias.quantization, shard_spec["bias_quantization"][rank])
        _set_shape(bias, shard_spec["bias_shapes"][rank])


def _pad_quantization_channels(info, original_channels, target_channels, role):
    if info is None:
        raise ToolError("FC_WIDTH_PADDING_QUANTIZATION_MISSING", "{} quantization is missing".format(role))

    def pad(values, field):
        values = tuple(values)
        if len(values) in (0, 1):
            return values
        if len(values) != original_channels:
            raise ToolError(
                "FC_WIDTH_PADDING_QUANTIZATION_SHAPE_UNSUPPORTED",
                "{} {} count {} does not match {} output channels".format(role, field, len(values), original_channels),
            )
        return values + (values[-1],) * (target_channels - original_channels)

    return QuantizationInfo(
        scales=tuple(float(value) for value in pad(info.scales, "scale")),
        zero_points=tuple(int(value) for value in pad(info.zero_points, "zero_point")),
        quantized_dimension=int(info.quantized_dimension),
        minimum=tuple(float(value) for value in pad(info.minimum, "min")),
        maximum=tuple(float(value) for value in pad(info.maximum, "max")),
    )


def _append_constant_buffer(model, payload):
    if not model.buffers:
        raise ToolError("FC_WIDTH_PADDING_BUFFER_MISSING", "model has no buffer table")
    buffer = copy.deepcopy(model.buffers[0])
    buffer.data = np.frombuffer(bytes(payload), dtype=np.uint8).copy()
    model.buffers.append(buffer)
    return len(model.buffers) - 1


def _compact_graph_buffers(model, graph):
    """Drop superseded/unreferenced buffers after a graph rewrite."""
    referenced = {0}
    referenced.update(int(tensor.buffer) for tensor in graph.tensors)
    ordered = sorted(referenced)
    remap = {old: new for new, old in enumerate(ordered)}
    for tensor in graph.tensors:
        old_index = int(tensor.buffer)
        if old_index not in remap:
            raise ToolError("FC_WIDTH_PADDING_BUFFER_REMAP_FAILED", str(old_index))
        tensor.buffer = remap[old_index]
    removed_bytes = sum(
        len(np.asarray(buffer.data, dtype=np.uint8).reshape(-1))
        for index, buffer in enumerate(model.buffers)
        if index not in remap and getattr(buffer, "data", None) is not None
    )
    model.buffers = [model.buffers[index] for index in ordered]
    retained_bytes = sum(
        len(np.asarray(buffer.data, dtype=np.uint8).reshape(-1))
        for buffer in model.buffers
        if getattr(buffer, "data", None) is not None
    )
    return {
        "buffer_count_after": len(model.buffers),
        "unused_buffer_bytes_removed": int(removed_bytes),
        "serialized_constant_buffer_bytes_after": int(retained_bytes),
    }


def _pad_rank_local_fc_output(model, graph, operator_index, role, multiplier, adapter, profile, restore_slice=True):
    """Pad a rank-local FC output, optionally preserving the padded width for a following op."""
    if int(multiplier) != _VERIFIED_FC_WIDTH_PADDING_MULTIPLIER:
        raise ToolError(
            "FC_WIDTH_PADDING_MULTIPLIER_UNVERIFIED",
            "only multiplier {} passed the current Host regression".format(_VERIFIED_FC_WIDTH_PADDING_MULTIPLIER),
        )
    if operator_index < 0 or operator_index >= len(graph.operators):
        raise ToolError("FC_WIDTH_PADDING_OPERATOR_OUT_OF_RANGE", str(operator_index))
    op = graph.operators[operator_index]
    if len(op.inputs) != 3 or len(op.outputs) != 1:
        raise ToolError("FC_WIDTH_PADDING_OPERATOR_ARITY_UNSUPPORTED", role)
    code = model.operatorCodes[int(op.opcodeIndex)]
    custom_code = getattr(code, "customCode", None)
    if isinstance(custom_code, bytes):
        custom_code = custom_code.decode("utf-8")
    if custom_code != "MTKEXT_FULLY_CONNECTED":
        raise ToolError("FC_WIDTH_PADDING_OPERATOR_UNSUPPORTED", "{} is not MTKEXT_FULLY_CONNECTED".format(role))
    weight = graph.tensors[int(op.inputs[1])]
    bias = graph.tensors[int(op.inputs[2])]
    output_index = int(op.outputs[0])
    output = graph.tensors[output_index]
    source_shape = _shape(weight)
    output_shape = _shape(output)
    if len(source_shape) != 2 or len(output_shape) != 3 or output_shape[-1] != source_shape[0]:
        raise ToolError("FC_WIDTH_PADDING_SHAPE_UNSUPPORTED", "{} FC shapes are weight={} output={}".format(role, source_shape, output_shape))
    expected_tokens = {"P128": 128, "D1": 1}.get(profile)
    if expected_tokens is None or output_shape[-2] != expected_tokens:
        raise ToolError(
            "FC_WIDTH_PADDING_PROFILE_UNSUPPORTED",
            "{} output token extent {} does not match {}".format(role, output_shape[-2], profile),
        )
    n, k = source_shape
    padded_n = n * multiplier
    if n <= 0 or k <= 0 or k % 2:
        raise ToolError("FC_WIDTH_PADDING_INT4_LAYOUT_UNSUPPORTED", "{} weight shape {} is not row-byte aligned INT4".format(role, source_shape))
    if int(weight.type) != int(getattr(adapter.schema.TensorType, "INT4")):
        raise ToolError("FC_WIDTH_PADDING_DTYPE_UNSUPPORTED", "{} weight is not INT4".format(role))
    raw_weight = _tensor_data(model, weight)
    expected_weight_bytes = n * k // 2
    if len(raw_weight) != expected_weight_bytes:
        raise ToolError("FC_WIDTH_PADDING_BUFFER_LENGTH_MISMATCH", "{} INT4 buffer length differs from N*K/2".format(role))
    padded_weight = raw_weight + bytes((padded_n - n) * k // 2)
    old_weight_quant = _quant_info(getattr(weight, "quantization", None))
    new_weight_quant = _pad_quantization_channels(old_weight_quant, n, padded_n, role + ".weight")
    _replace_buffer_data(model, weight, padded_weight)
    _assign_quantization(weight.quantization, new_weight_quant)
    _set_shape(weight, (padded_n, k))

    bias_shape = _shape(bias)
    if bias_shape not in ((n,), (1, n)) or int(bias.type) != int(getattr(adapter.schema.TensorType, "INT32")):
        raise ToolError("FC_WIDTH_PADDING_BIAS_UNSUPPORTED", "{} bias shape/type is {}/{}".format(role, bias_shape, bias.type))
    raw_bias = _tensor_data(model, bias)
    if len(raw_bias) != n * 4:
        raise ToolError("FC_WIDTH_PADDING_BIAS_BUFFER_LENGTH_MISMATCH", role)
    padded_bias = raw_bias + bytes((padded_n - n) * 4)
    old_bias_quant = _quant_info(getattr(bias, "quantization", None))
    new_bias_quant = _pad_quantization_channels(old_bias_quant, n, padded_n, role + ".bias")
    _replace_buffer_data(model, bias, padded_bias)
    _assign_quantization(bias.quantization, new_bias_quant)
    _set_shape(bias, (padded_n,) if len(bias_shape) == 1 else (1, padded_n))

    padded_output = copy.deepcopy(output)
    output_name = output.name or "fc_output"
    if isinstance(output_name, bytes):
        output_name = output_name.decode("utf-8")
    padded_output.name = output_name + "/width_padded"
    padded_shape = list(output_shape[:-1]) + [padded_n]
    padded_output.shape = np.asarray(padded_shape, dtype=np.int32)
    if getattr(padded_output, "shapeSignature", None) is not None:
        padded_output.shapeSignature = np.asarray(padded_shape, dtype=np.int32)
    padded_output_index = len(graph.tensors)
    graph.tensors.append(padded_output)
    op.outputs = np.asarray([padded_output_index], dtype=np.int32)

    if restore_slice:
        begin = np.zeros(len(output_shape), dtype="<i4")
        size = np.asarray(output_shape, dtype="<i4")
        begin_buffer = _append_constant_buffer(model, begin.tobytes())
        size_buffer = _append_constant_buffer(model, size.tobytes())

        def shape_tensor(name, buffer_index):
            tensor = copy.deepcopy(output)
            tensor.name = name
            tensor.shape = np.asarray([len(output_shape)], dtype=np.int32)
            if getattr(tensor, "shapeSignature", None) is not None:
                tensor.shapeSignature = np.asarray([len(output_shape)], dtype=np.int32)
            tensor.type = int(getattr(adapter.schema.TensorType, "INT32"))
            tensor.buffer = buffer_index
            tensor.quantization = None
            tensor.isVariable = False
            return tensor

        begin_index = len(graph.tensors)
        graph.tensors.append(shape_tensor("{}/width_pad_slice_begin".format(role), begin_buffer))
        size_index = len(graph.tensors)
        graph.tensors.append(shape_tensor("{}/width_pad_slice_size".format(role), size_buffer))

        slice_builtin = int(getattr(adapter.schema.BuiltinOperator, "SLICE"))
        slice_opcode_index = None
        placeholder = int(getattr(adapter.schema.BuiltinOperator, "PLACEHOLDER_FOR_GREATER_OP_CODES", -999))
        for index, item in enumerate(model.operatorCodes):
            builtin = int(item.builtinCode)
            if builtin == placeholder:
                builtin = int(item.deprecatedBuiltinCode)
            if builtin == slice_builtin and not getattr(item, "customCode", None):
                slice_opcode_index = index
                break
        if slice_opcode_index is None:
            slice_code = _ObjectTable("OperatorCode", adapter)
            slice_code.builtinCode = slice_builtin
            slice_code.deprecatedBuiltinCode = slice_builtin
            slice_code.version = 1
            slice_code.customCode = None
            model.operatorCodes.append(slice_code)
            slice_opcode_index = len(model.operatorCodes) - 1

        slice_options = _ObjectTable("SliceOptions", adapter)
        slice_op = _ObjectTable("Operator", adapter)
        slice_op.opcodeIndex = slice_opcode_index
        slice_op.inputs = np.asarray([padded_output_index, begin_index, size_index], dtype=np.int32)
        slice_op.outputs = np.asarray([output_index], dtype=np.int32)
        slice_op.builtinOptionsType = int(getattr(adapter.schema.BuiltinOptions, "SliceOptions"))
        slice_op.builtinOptions = slice_options
        slice_op.customOptionsFormat = int(getattr(adapter.schema.CustomOptionsFormat, "FLEXBUFFERS", 0))
        graph.operators.insert(operator_index + 1, slice_op)
    return {
        "role": role,
        "source_output_width": n,
        "padded_output_width": padded_n,
        "pad_multiplier": multiplier,
        "rank_local_int4_nonzero_payload_bytes": expected_weight_bytes,
        "serialized_int4_weight_bytes": len(padded_weight),
        "zero_padding_int4_bytes": len(padded_weight) - expected_weight_bytes,
        "rank_local_bias_bytes": n * 4,
        "serialized_bias_bytes": len(padded_bias),
        "output_shape_before_slice": padded_shape,
        "output_shape_after_slice": output_shape,
        "slice_operator": "SLICE" if restore_slice else None,
        "quantization_policy": "preserve_rank_scales_and_repeat_last_valid_per_axis_scale_for_zero_weight_channels",
    }




def _append_slice_operator(model, graph, input_index, output_index, output_shape, adapter, name):
    """Append a TFLite SLICE that restores an existing tensor's public shape."""
    output_shape = tuple(int(value) for value in output_shape)
    begin = np.zeros(len(output_shape), dtype="<i4")
    size = np.asarray(output_shape, dtype="<i4")
    begin_buffer = _append_constant_buffer(model, begin.tobytes())
    size_buffer = _append_constant_buffer(model, size.tobytes())
    template = graph.tensors[int(output_index)]

    def shape_tensor(tensor_name, buffer_index):
        tensor = copy.deepcopy(template)
        tensor.name = tensor_name
        tensor.shape = np.asarray([len(output_shape)], dtype=np.int32)
        if getattr(tensor, "shapeSignature", None) is not None:
            tensor.shapeSignature = np.asarray([len(output_shape)], dtype=np.int32)
        tensor.type = int(adapter.schema.TensorType.INT32)
        tensor.buffer = buffer_index
        tensor.quantization = None
        tensor.isVariable = False
        return tensor

    begin_index = len(graph.tensors)
    graph.tensors.append(shape_tensor(name + "/begin", begin_buffer))
    size_index = len(graph.tensors)
    graph.tensors.append(shape_tensor(name + "/size", size_buffer))
    slice_opcode_index = None
    placeholder = int(getattr(adapter.schema.BuiltinOperator, "PLACEHOLDER_FOR_GREATER_OP_CODES", -999))
    slice_code = int(adapter.schema.BuiltinOperator.SLICE)
    for index, item in enumerate(model.operatorCodes):
        builtin = int(item.builtinCode)
        if builtin == placeholder:
            builtin = int(item.deprecatedBuiltinCode)
        if builtin == slice_code and not getattr(item, "customCode", None):
            slice_opcode_index = index
            break
    if slice_opcode_index is None:
        code = _ObjectTable("OperatorCode", adapter)
        code.builtinCode = slice_code
        code.deprecatedBuiltinCode = slice_code
        code.version = 1
        code.customCode = None
        model.operatorCodes.append(code)
        slice_opcode_index = len(model.operatorCodes) - 1
    op = _ObjectTable("Operator", adapter)
    op.opcodeIndex = slice_opcode_index
    op.inputs = np.asarray([int(input_index), begin_index, size_index], dtype=np.int32)
    op.outputs = np.asarray([int(output_index)], dtype=np.int32)
    op.builtinOptionsType = int(adapter.schema.BuiltinOptions.SliceOptions)
    op.builtinOptions = _ObjectTable("SliceOptions", adapter)
    op.customOptionsFormat = int(getattr(adapter.schema.CustomOptionsFormat, "FLEXBUFFERS", 0))
    return op


def _apply_mlp_gate_dequant_width_padding_rule(model, graph, layer, component, degree, multiplier, adapter):
    """Pad rank-local Gate INT4 output width before Dequantize, then slice FP32."""
    if multiplier is None:
        return []
    if int(multiplier) != _VERIFIED_FC_WIDTH_PADDING_MULTIPLIER:
        raise ToolError(
            "MLP_GATE_WIDTH_PADDING_MULTIPLIER_UNVERIFIED",
            "only multiplier {} has saved NPU component evidence".format(_VERIFIED_FC_WIDTH_PADDING_MULTIPLIER),
        )
    if degree != 4:
        raise ToolError("MLP_GATE_WIDTH_PADDING_PROFILE_UNSUPPORTED", "MLP Gate Dequant padding requires TP4")
    if layer.profile != "P128" or int(layer.layer_id) != 0:
        raise ToolError(
            "MLP_GATE_WIDTH_PADDING_SCOPE_UNVERIFIED",
            "MLP Gate Dequant padding is verified only for TP4 P128 Layer0",
        )
    if component.name != "mlp":
        return []

    gate_role = getattr(layer, "roles", {}).get("gate")
    weight_index = int(getattr(gate_role, "tensor_index", -1)) if gate_role is not None else -1
    if weight_index < 0:
        raise ToolError("MLP_GATE_FC_MAPPING_AMBIGUOUS", "discovery did not identify the Gate weight tensor")
    gate_matches = [
        (index, op) for index, op in enumerate(graph.operators)
        if len(op.inputs) == 3 and len(op.outputs) == 1 and int(op.inputs[1]) == weight_index
    ]
    if len(gate_matches) != 1:
        raise ToolError("MLP_GATE_FC_MAPPING_AMBIGUOUS", "expected one rank-local Gate FC from graph role mapping")
    fc_index, fc = gate_matches[0]
    local_gate_output = int(fc.outputs[0])

    def is_dequantize(op):
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            return False
        code = model.operatorCodes[int(op.opcodeIndex)]
        if getattr(code, "customCode", None):
            return False
        builtin = int(code.builtinCode)
        placeholder = int(getattr(adapter.schema.BuiltinOperator, "PLACEHOLDER_FOR_GREATER_OP_CODES", -999))
        if builtin == placeholder:
            builtin = int(code.deprecatedBuiltinCode)
        return builtin == int(adapter.schema.BuiltinOperator.DEQUANTIZE)

    consumers = [op for op in graph.operators if local_gate_output in tuple(int(x) for x in op.inputs)]
    dequant_matches = [op for op in consumers if is_dequantize(op) and int(op.inputs[0]) == local_gate_output]
    if len(consumers) != 1 or len(dequant_matches) != 1:
        raise ToolError(
            "MLP_GATE_DEQUANT_MAPPING_AMBIGUOUS",
            "Gate output must have exactly one DEQUANTIZE consumer",
        )
    dequant = dequant_matches[0]
    local_dequant_output = int(dequant.outputs[0])
    local_dequant = graph.tensors[local_dequant_output]
    local_shape = tuple(int(value) for value in np.asarray(local_dequant.shape).reshape(-1))
    if len(local_shape) != 3:
        raise ToolError("MLP_GATE_DEQUANT_SHAPE_UNSUPPORTED", str(local_shape))

    padding_report = _pad_rank_local_fc_output(
        model, graph, fc_index, "mlp.gate", int(multiplier), adapter, layer.profile, restore_slice=False
    )
    padded_gate_output = int(fc.outputs[0])
    padded_shape = local_shape[:-1] + (local_shape[-1] * int(multiplier),)
    padded_dequant = copy.deepcopy(local_dequant)
    old_name = local_dequant.name.decode("utf-8") if isinstance(local_dequant.name, bytes) else str(local_dequant.name)
    padded_dequant.name = old_name + "/width_padded"
    padded_dequant.shape = np.asarray(padded_shape, dtype=np.int32)
    if getattr(padded_dequant, "shapeSignature", None) is not None:
        padded_dequant.shapeSignature = np.asarray(padded_shape, dtype=np.int32)
    padded_dequant_index = len(graph.tensors)
    graph.tensors.append(padded_dequant)
    dequant.inputs = np.asarray([padded_gate_output], dtype=np.int32)
    dequant.outputs = np.asarray([padded_dequant_index], dtype=np.int32)
    dequant_index = next(index for index, op in enumerate(graph.operators) if op is dequant)
    graph.operators.insert(
        dequant_index + 1,
        _append_slice_operator(
            model,
            graph,
            padded_dequant_index,
            local_dequant_output,
            local_shape,
            adapter,
            "mlp.gate.dequant_width_slice",
        ),
    )
    buffer_compaction = _compact_graph_buffers(model, graph)
    return [{
        **padding_report,
        "buffer_compaction": buffer_compaction,
        "stage": "GATE_FC_PADDED_INT16_DEQUANTIZED_BEFORE_FP32_SLICE",
        "padded_dequant_output_shape": list(padded_shape),
        "rank_dequant_output_shape": list(local_shape),
        "numerical_scope": "STRUCTURAL_REWRITE_ONLY_REQUIRES_SHA_BOUND_NPU_RETEST",
        "npu_verification_status": "NOT_VERIFIED_FOR_NEW_CANDIDATE_SHA",
    }]


def _apply_fc_width_padding_rule(model, graph, layer, component, degree, multiplier, adapter):
    if multiplier is None:
        return []
    if degree != 4 or layer.profile not in ("P128", "D1") or component.name != "attn":
        return []
    original_to_local = {int(source_index): index for index, source_index in enumerate(component.operator_indices)}
    targets = []
    for role in ("q", "k", "v"):
        match = layer.roles.get(role)
        if match is not None and int(match.operator_index) in original_to_local:
            targets.append((original_to_local[int(match.operator_index)], role))
    if not targets:
        raise ToolError("FC_WIDTH_PADDING_TARGETS_MISSING", "P128 TP4 attention component contains no Q/K/V FC")
    result = []
    for local_index, role in sorted(targets, key=lambda item: item[0], reverse=True):
        result.append(_pad_rank_local_fc_output(model, graph, local_index, role, multiplier, adapter, layer.profile))
    buffer_compaction = _compact_graph_buffers(model, graph)
    for item in result:
        item["buffer_compaction"] = buffer_compaction
    return list(reversed(result))


def _normalize_host_scalar_constant_shapes(model, graph, layer, component, degree, adapter):
    """Give one-element scalar constants vector shapes accepted by the Host importer.

    This is limited to TP4 P128 attention tensors consumed only as SPLIT axis
    operands or broadcast MUL constants. The scalar bytes and quantization stay
    unchanged. The compatibility form matches existing rank-local TFLites.
    """
    if degree != 4 or layer.profile not in ("P128", "D1") or component.name != "attn":
        return []
    return normalize_host_scalar_constant_shapes(model, graph, adapter.schema)


def _mutate_int32_constant(model, graph, tensor_index, values, schema):
    tensor = graph.tensors[tensor_index]
    if int(tensor.type) != int(getattr(schema.TensorType, "INT32")):
        raise ToolError("SHAPE_CONSTANT_DTYPE_UNSUPPORTED", "tensor {} is not INT32".format(tensor_index))
    values = np.asarray(values, dtype="<i4").reshape(-1)
    _replace_buffer_data(model, tensor, values.tobytes())


def _apply_shape_constants(model, graph, model_info, component, adapter):
    source_graph = model_info.graphs[0]
    reshape_code = int(getattr(adapter.schema.BuiltinOperator, "RESHAPE"))
    slice_code = int(getattr(adapter.schema.BuiltinOperator, "STRIDED_SLICE"))
    pending_constant_edits = {}
    for operator_index in component.operator_indices:
        op = graph.operators[operator_index]
        if int(op.opcodeIndex) < 0 or int(op.opcodeIndex) >= len(model.operatorCodes):
            continue
        code = model.operatorCodes[int(op.opcodeIndex)]
        builtin = int(code.builtinCode)
        placeholder = int(getattr(adapter.schema.BuiltinOperator, "PLACEHOLDER_FOR_GREATER_OP_CODES", -999))
        if builtin == placeholder:
            builtin = int(code.deprecatedBuiltinCode)
        if builtin == reshape_code and len(op.outputs) == 1:
            output_index = int(op.outputs[0])
            target_shape = _shape(graph.tensors[output_index])
            options = getattr(op, "builtinOptions", None)
            if options is not None:
                options.newShape = np.asarray(target_shape, dtype=np.int32)
            if len(op.inputs) > 1 and int(op.inputs[1]) >= 0:
                shape_tensor_index = int(op.inputs[1])
                shape_tensor = graph.tensors[shape_tensor_index]
                if _tensor_data(model, shape_tensor):
                    previous = pending_constant_edits.get(shape_tensor_index)
                    if previous is not None and tuple(previous) != tuple(target_shape):
                        raise ToolError("SHARED_SHAPE_CONSTANT_CONFLICT", str(shape_tensor_index))
                    pending_constant_edits[shape_tensor_index] = target_shape
        elif builtin == slice_code and len(op.inputs) > 2 and int(op.inputs[2]) >= 0:
            data_tensor_index = int(op.inputs[0])
            end_tensor_index = int(op.inputs[2])
            source_shape = source_graph.tensors[data_tensor_index].shape
            target_shape = _shape(graph.tensors[data_tensor_index])
            if len(source_shape) != len(target_shape):
                raise ToolError("STRIDED_SLICE_RANK_CHANGED", str(operator_index))
            end_tensor = graph.tensors[end_tensor_index]
            raw = _tensor_data(model, end_tensor)
            if len(raw) % 4:
                raise ToolError("STRIDED_SLICE_END_CONSTANT_INVALID", str(end_tensor_index))
            values = np.frombuffer(raw, dtype="<i4").copy()
            if len(values) == len(source_shape):
                for axis, (before, after) in enumerate(zip(source_shape, target_shape)):
                    if before in (8, 32) and after < before and int(values[axis]) == before:
                        values[axis] = after
                pending_constant_edits[end_tensor_index] = tuple(int(value) for value in values)
    for tensor_index, values in pending_constant_edits.items():
        _mutate_int32_constant(model, graph, tensor_index, values, adapter.schema)


def _adjust_attention_shapes(model, graph, layer, component, rank, degree):
    touched = set(component.input_tensor_indices) | set(component.output_tensor_indices)
    for operator_index in component.operator_indices:
        op = graph.operators[operator_index]
        touched.update(int(index) for index in op.inputs if int(index) >= 0)
        touched.update(int(index) for index in op.outputs if int(index) >= 0)
    for tensor_index in sorted(touched):
        tensor = graph.tensors[tensor_index]
        shape = list(_shape(tensor))
        if len(shape) == 4:
            # Attention tensors appear in both [batch, heads, seq, width] and
            # [batch, seq, heads, width] layouts in the canonical export.
            head_axes = [axis for axis in (1, 2) if shape[axis] in (8, 32)]
            if len(head_axes) > 1:
                raise ToolError(
                    "ATTENTION_HEAD_AXIS_AMBIGUOUS",
                    "tensor {} shape {} has multiple candidate head axes".format(tensor.name, tuple(shape)),
                )
            if head_axes:
                axis = head_axes[0]
                if shape[axis] % degree:
                    raise ToolError(
                        "ATTENTION_HEADS_NOT_DIVISIBLE",
                        "tensor {} head dimension {} does not divide by {}".format(tensor.name, shape[axis], degree),
                    )
                shape[axis] //= degree
                _set_shape(tensor, shape)
    context_index = layer.roles["o"].activation_tensor_index
    context = graph.tensors[context_index]
    context_shape = list(_shape(context))
    if not context_shape or context_shape[-1] != 4096:
        raise ToolError("ATTENTION_CONTEXT_SHAPE_UNSUPPORTED", str(context_shape))
    context_shape[-1] //= degree
    _set_shape(context, context_shape)


def _adjust_mlp_shapes(model, graph, component, degree):
    touched = set(component.input_tensor_indices) | set(component.output_tensor_indices)
    for operator_index in component.operator_indices:
        op = graph.operators[operator_index]
        touched.update(int(index) for index in op.inputs if int(index) >= 0)
        touched.update(int(index) for index in op.outputs if int(index) >= 0)
    for tensor_index in sorted(touched):
        tensor = graph.tensors[tensor_index]
        shape = list(_shape(tensor))
        if shape and shape[-1] == 14336:
            shape[-1] //= degree
            _set_shape(tensor, shape)


def _prune_model(model, graph, component):
    for field_name in ("metadata", "metadataBuffer", "signatureDefs"):
        value = getattr(model, field_name, None)
        if value:
            raise ToolError(
                "MODEL_METADATA_REMAP_UNSUPPORTED",
                "cannot safely prune model-level {} for a component graph".format(field_name),
            )
    selected = set(component.operator_indices)
    tensors = graph.tensors
    tensor_ids = sorted(
        set(component.input_tensor_indices)
        | set(component.output_tensor_indices)
        | {
            int(index)
            for operator_index in selected
            for index in list(graph.operators[operator_index].inputs) + list(graph.operators[operator_index].outputs)
            if int(index) >= 0
        }
    )
    remap = {old: new for new, old in enumerate(tensor_ids)}
    selected_operators = [graph.operators[index] for index in component.operator_indices]
    buffers = sorted({int(tensors[index].buffer) for index in tensor_ids} | {0})
    buffer_remap = {old: new for new, old in enumerate(buffers)}
    for tensor_index in tensor_ids:
        tensors[tensor_index].buffer = buffer_remap[int(tensors[tensor_index].buffer)]
    model.buffers = [model.buffers[index] for index in buffers]
    graph.tensors = [tensors[index] for index in tensor_ids]
    for op in selected_operators:
        op.inputs = np.asarray([remap[int(index)] if int(index) >= 0 else -1 for index in op.inputs], dtype=np.int32)
        op.outputs = np.asarray([remap[int(index)] for index in op.outputs], dtype=np.int32)
    graph.operators = selected_operators
    graph.inputs = np.asarray([remap[index] for index in component.input_tensor_indices], dtype=np.int32)
    graph.outputs = np.asarray([remap[index] for index in component.output_tensor_indices], dtype=np.int32)
    for field_name in ("metadata", "metadataBuffer", "signatureDefs"):
        if hasattr(model, field_name):
            setattr(model, field_name, None)
    return remap


def _io_contract(parsed_model, graph_index=0):
    graph = parsed_model.graphs[graph_index]

    def ports(indices):
        return [
            {
                "index": index,
                "name": graph.tensors[index].name,
                "shape": list(graph.tensors[index].shape),
                "dtype": graph.tensors[index].dtype,
                "buffer_bytes": graph.tensors[index].buffer_bytes,
            }
            for index in indices
        ]

    return {"inputs": ports(graph.inputs), "outputs": ports(graph.outputs)}


def _validate_serialized_component(data, source_name, schema_module):
    parsed = read_tflite_bytes(data, source_name=source_name, schema_module=schema_module)
    if parsed.version != 3 or len(parsed.graphs) != 1:
        raise ToolError("SERIALIZED_COMPONENT_UNSUPPORTED", "expected TFLite v3 with one subgraph")
    graph = parsed.graphs[0]
    validate_graph(parsed, graph)
    return parsed


def _tp2_row_reduce_semantics(graph, layer):
    """Describe the canonical row-reduce tensors without claiming a proof.

    A TFLite graph records each projection's output tensor contract, but it
    does not by itself define how rank-local accumulators are combined. Keep
    the observed dtype, shape, and quantization metadata beside the explicit
    unknowns so the release gate can require a separate bit-exact proof.
    """
    partial_outputs = {}
    output_dtypes = {}
    for role in ("o", "down"):
        output_index = layer.roles[role].output_tensor_indices[0]
        tensor = graph.tensors[output_index]
        output_dtypes[role] = tensor.dtype
        partial_outputs[role] = {
            "tensor_index": tensor.index,
            "tensor_name": tensor.name,
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "quantization": quantization_dict(tensor.quantization),
        }
    observed = ", ".join(
        "{}={}".format(role.upper(), output_dtypes[role])
        for role in ("o", "down")
    )
    return {
        "status": "UNPROVEN",
        "proof_sha256": None,
        "partial_output_dtype": output_dtypes,
        "partial_outputs": partial_outputs,
        "bias_ownership": {"o": "RANK0_ONLY", "down": "RANK0_ONLY"},
        "accumulator_width": "UNPROVEN",
        "reduction_order": "UNPROVEN",
        "rounding_mode": "UNPROVEN",
        "saturation": "UNPROVEN",
        "requantization": "UNPROVEN",
        "reason": (
            "Canonical output metadata reports {}. It does not establish the "
            "cross-rank accumulator width, reduction order, rounding, "
            "saturation, and requantization needed for bit-exact TP2 row "
            "reduction."
        ).format(observed),
    }


def _build_tp_layer_into(
    model_object,
    model_info,
    layer,
    graph_analysis,
    output_directory,
    degree,
    schema_module=None,
    fc_width_padding_multiplier=None,
    mlp_gate_dequant_width_padding_multiplier=None,
):
    """Write one layer's rank-specific components for TP2 or TP4."""
    if layer.graph_index != 0 or len(getattr(model_object, "subgraphs", ()) or ()) != 1:
        raise ToolError("SOURCE_GRAPH_UNSUPPORTED", "component rewriting requires exactly one source subgraph at index 0")
    adapter = TFLiteObjectAdapter(schema_module=schema_module)
    if fc_width_padding_multiplier is not None:
        if int(fc_width_padding_multiplier) != _VERIFIED_FC_WIDTH_PADDING_MULTIPLIER:
            raise ToolError(
                "FC_WIDTH_PADDING_MULTIPLIER_UNVERIFIED",
                "only multiplier {} passed the current Host regression".format(_VERIFIED_FC_WIDTH_PADDING_MULTIPLIER),
            )
        if degree != 4 or layer.profile not in ("P128", "D1"):
            raise ToolError(
                "FC_WIDTH_PADDING_PROFILE_UNSUPPORTED",
                "the verified width-padding rule applies only to TP4 P128 or D1",
            )
        fc_width_padding_multiplier = int(fc_width_padding_multiplier)
    if mlp_gate_dequant_width_padding_multiplier is not None:
        if int(mlp_gate_dequant_width_padding_multiplier) != _VERIFIED_FC_WIDTH_PADDING_MULTIPLIER:
            raise ToolError(
                "MLP_GATE_WIDTH_PADDING_MULTIPLIER_UNVERIFIED",
                "only multiplier 2 has saved NPU component evidence",
            )
        if degree != 4 or layer.profile != "P128" or int(layer.layer_id) != 0:
            raise ToolError(
                "MLP_GATE_WIDTH_PADDING_SCOPE_UNVERIFIED",
                "MLP Gate Dequant padding is verified only for TP4 P128 Layer0",
            )
        mlp_gate_dequant_width_padding_multiplier = int(mlp_gate_dequant_width_padding_multiplier)
    specs = derive_tp_components(model_info, layer, graph_analysis, degree=degree)
    weight_specs, weight_manifest = _prepare_projection_shards(model_object, layer, degree=degree, schema=adapter.schema)
    output_root = Path(output_directory)
    if output_root.exists():
        raise ToolError("OUTPUT_DIRECTORY_EXISTS", "refusing to reuse {}".format(output_root))
    output_root.mkdir(parents=True)
    graph = model_info.graphs[layer.graph_index]
    golden_gate = probe_golden_backend(model_info)
    outputs = []
    for rank in range(degree):
        for component in specs:
            model = copy.deepcopy(model_object)
            graph_object = model.subgraphs[layer.graph_index]
            for role, stage_name in _COMPONENT_FOR_ROLE.items():
                if stage_name == component.name:
                    _apply_projection_shard(model, graph_object, layer, role, weight_specs[role], rank)
            if component.name == "attn":
                _adjust_attention_shapes(model, graph_object, layer, component, rank, degree=degree)
            elif component.name == "mlp":
                _adjust_mlp_shapes(model, graph_object, component, degree=degree)
            _apply_shape_constants(model, graph_object, model_info, component, adapter)
            _prune_model(model, graph_object, component)
            fc_width_padding = _apply_fc_width_padding_rule(
                model,
                graph_object,
                layer,
                component,
                degree,
                fc_width_padding_multiplier,
                adapter,
            )
            mlp_gate_dequant_width_padding = _apply_mlp_gate_dequant_width_padding_rule(
                model,
                graph_object,
                layer,
                component,
                degree,
                mlp_gate_dequant_width_padding_multiplier,
                adapter,
            )
            host_scalar_shape_compatibility = _normalize_host_scalar_constant_shapes(
                model,
                graph_object,
                layer,
                component,
                degree,
                adapter,
            )
            data = adapter.export(model)
            profile_dir = output_root / layer.profile.lower() / "rank{}".format(rank) / "layer{:02d}".format(layer.layer_id)
            target = profile_dir / (component.name + ".tflite")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
            parsed = _validate_serialized_component(data, target.name, adapter.schema)
            outputs.append(
                {
                    "profile": layer.profile,
                    "layer": layer.layer_id,
                    "rank": rank,
                    "component": component.name,
                    "source": layer.source_name,
                    "path": target.relative_to(output_root).as_posix(),
                    "source_sha256": layer.source_sha256,
                    "output_sha256": hashlib.sha256(data).hexdigest(),
                    "output_bytes": len(data),
                    "operators": len(parsed.graphs[0].operators),
                    "tensors": len(parsed.graphs[0].tensors),
                    "weight_bytes": sum(
                        tensor.buffer_bytes
                        for tensor in parsed.graphs[0].tensors
                        if tensor.dtype == "INT4"
                    ),
                    "fc_width_padding": fc_width_padding,
                    "mlp_gate_dequant_width_padding": mlp_gate_dequant_width_padding,
                    "host_scalar_shape_compatibility": host_scalar_shape_compatibility,
                    "rank_local_nonzero_int4_weight_bytes": sum(
                        int(item["rank_local_int4_nonzero_payload_bytes"])
                        for item in fc_width_padding + mlp_gate_dequant_width_padding
                    ),
                    "io": _io_contract(parsed),
                    "structural_status": "STRUCTURAL_PASS",
                    "numerical_status": golden_gate["status"],
                    "release_status": "NOT_ELIGIBLE",
                }
            )

    graph_info = model_info.graphs[layer.graph_index]
    quantization_contract = {}
    for role, match in layer.roles.items():
        input_index = match.activation_tensor_index
        output_index = match.output_tensor_indices[0]
        input_tensor = graph_info.tensors[input_index]
        output_tensor = graph_info.tensors[output_index]
        quantization_contract[role] = {
            "weight": {
                "tensor": match.tensor_index,
                "shape": list(match.shape),
                "dtype": match.dtype,
                "quantization": quantization_dict(match.quantization),
            },
            "bias": {
                "tensor": match.bias_tensor_index,
                "shape": list(match.bias_shape),
                "dtype": match.bias_dtype,
                "quantization": quantization_dict(match.bias_quantization),
            },
            "activation_input": {
                "tensor": input_index,
                "shape": list(input_tensor.shape),
                "dtype": input_tensor.dtype,
                "quantization": quantization_dict(input_tensor.quantization),
            },
            "activation_output": {
                "tensor": output_index,
                "shape": list(output_tensor.shape),
                "dtype": output_tensor.dtype,
                "quantization": quantization_dict(output_tensor.quantization),
            },
        }
    manifest = {
        "schema": 1,
        "stage": "M2_TP4_SINGLE_LAYER_STRUCTURAL_PARTITION" if degree == 4 else "M3_TP2_SINGLE_LAYER_STRUCTURAL_PARTITION",
        "topology": "TP4_FOUR_STAGE_ALL_GATHER" if degree == 4 else "TP2_ROW_REDUCE",
        "profile": layer.profile,
        "layer": layer.layer_id,
        "source": layer.source_name,
        "source_sha256": layer.source_sha256,
        "rank_order": list(range(degree)),
        "components": [component.to_dict() for component in specs],
        "collective_contract": _collective_contract(degree, 128 if layer.profile == "P128" else 1),
        "weight_partitions": weight_manifest,
        "fc_width_padding_rule": {
            "status": "APPLIED_TO_TP4_P128_D1_QKV" if fc_width_padding_multiplier is not None else "DISABLED",
            "multiplier": fc_width_padding_multiplier,
            "rule": "pad each rank-local Q/K/V output width with zero INT4 weights and zero bias, then SLICE back to the rank output width",
            "numerical_scope": "CPU_HOST_ONLY_NOT_AN_NPU_EXACTNESS_CLAIM",
        },
        "mlp_gate_dequant_width_padding_rule": {
            "status": "APPLIED_TO_TP4_P128_LAYER0_MLP_GATE" if mlp_gate_dequant_width_padding_multiplier is not None else "DISABLED",
            "multiplier": mlp_gate_dequant_width_padding_multiplier,
            "rule": "zero-pad rank-local Gate INT4 output width, Dequantize padded INT16 width, then Slice FP32 back to the rank width",
            "numerical_scope": "STRUCTURAL_REWRITE_ONLY; each new candidate SHA requires a same-backend NPU retest",
        },
        "host_scalar_shape_compatibility": {
            "status": "NORMALIZED_SAFE_CONSTANT_SCALARS" if degree == 4 and layer.profile in ("P128", "D1") else "DISABLED",
            "rule": "one-element scalar INT16/INT32 constants consumed only as SPLIT axis operands or broadcast MUL values use shape [1] for Host importer compatibility",
            "scope": "TP4_P128_D1_ATTENTION_COMPONENTS",
        },
        "quantization_contract": quantization_contract,
        "models": outputs,
        "weight_reconstruction": "PASS",
        "structural_status": "STRUCTURAL_PASS",
        "numerical_status": golden_gate["status"],
        "mismatch_count": None,
        "golden_report_sha256": None,
        "auto_repair_report": {
            "status": "BLOCKED",
            "attempts": [],
            "first_failure": "GOLDEN_EXECUTION_UNAVAILABLE",
        },
        "numerical_gate": golden_gate,
        "release_status": "NOT_ELIGIBLE",
    }
    if degree == 2:
        graph_info = model_info.graphs[layer.graph_index]
        manifest["collective_contract"]["quantized_partial_semantics"] = _tp2_row_reduce_semantics(
            graph_info,
            layer,
        )
        manifest["numerical_gate"]["row_reduce_semantics"] = "UNPROVEN"
    (output_root / "LAYER_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _build_tp_layer(
    model_object,
    model_info,
    layer,
    graph_analysis,
    output_directory,
    degree,
    schema_module=None,
    fc_width_padding_multiplier=None,
    mlp_gate_dequant_width_padding_multiplier=None,
):
    output_root = Path(output_directory)
    if output_root.exists():
        raise ToolError("OUTPUT_DIRECTORY_EXISTS", "refusing to reuse {}".format(output_root))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".{}-build-".format(output_root.name), dir=str(output_root.parent)))
    staged_output = temporary_root / "artifact"
    try:
        manifest = _build_tp_layer_into(
            model_object,
            model_info,
            layer,
            graph_analysis,
            staged_output,
            degree=degree,
            schema_module=schema_module,
            fc_width_padding_multiplier=fc_width_padding_multiplier,
            mlp_gate_dequant_width_padding_multiplier=mlp_gate_dequant_width_padding_multiplier,
        )
        os.replace(str(staged_output), str(output_root))
        return manifest
    finally:
        shutil.rmtree(str(temporary_root), ignore_errors=True)


def build_tp4_layer(
    model_object,
    model_info,
    layer,
    graph_analysis,
    output_directory,
    schema_module=None,
    fc_width_padding_multiplier=None,
    mlp_gate_dequant_width_padding_multiplier=None,
):
    """Build one TP4 layer transactionally."""
    return _build_tp_layer(
        model_object,
        model_info,
        layer,
        graph_analysis,
        output_directory,
        degree=4,
        schema_module=schema_module,
        fc_width_padding_multiplier=fc_width_padding_multiplier,
        mlp_gate_dequant_width_padding_multiplier=mlp_gate_dequant_width_padding_multiplier,
    )


def build_tp2_layer(model_object, model_info, layer, graph_analysis, output_directory, schema_module=None):
    """Build one TP2 structural candidate, gated on unproven row-reduce semantics."""
    return _build_tp_layer(model_object, model_info, layer, graph_analysis, output_directory, degree=2, schema_module=schema_module)


def _build_tp_directory(
    source_directory,
    output_directory,
    profile,
    expected_layers,
    degree,
    schema_module=None,
    discovery_report=None,
    fc_width_padding_multiplier=None,
    mlp_gate_dequant_width_padding_multiplier=None,
):
    """Discover and build every layer for one profile as an atomic directory."""
    source_root = Path(source_directory)
    if not source_root.exists():
        raise ToolError("SOURCE_NOT_FOUND", "source path does not exist: {}".format(source_root))
    if profile not in ("D1", "P128"):
        raise ToolError("PROFILE_UNSUPPORTED", "profile must be D1 or P128")
    if mlp_gate_dequant_width_padding_multiplier is not None:
        if int(mlp_gate_dequant_width_padding_multiplier) != _VERIFIED_FC_WIDTH_PADDING_MULTIPLIER:
            raise ToolError("MLP_GATE_WIDTH_PADDING_MULTIPLIER_UNVERIFIED", "only multiplier 2 has saved NPU component evidence")
        if degree != 4 or profile != "P128":
            raise ToolError(
                "MLP_GATE_WIDTH_PADDING_SCOPE_UNVERIFIED",
                "MLP Gate Dequant padding is verified only for TP4 P128 Layer0",
            )
        mlp_gate_dequant_width_padding_multiplier = int(mlp_gate_dequant_width_padding_multiplier)
    if expected_layers <= 0:
        raise ToolError("EXPECTED_LAYER_COUNT_INVALID", "expected layer count must be positive")
    source_paths = [source_root] if source_root.is_file() else sorted(
        path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() == ".tflite"
    )
    if not source_paths:
        raise ToolError("NO_TFLITE_SOURCES", "no .tflite sources found under {}".format(source_root))

    model_by_source = {}
    if discovery_report is None:
        from .discovery import discover_models
        from .tflite_reader import read_tflite
        entries = []
        for path in source_paths:
            info = read_tflite(path, schema_module=schema_module)
            source_name = path.name if source_root.is_file() else path.relative_to(source_root).as_posix()
            if source_name in model_by_source:
                raise ToolError("SOURCE_NAME_DUPLICATE", source_name)
            entries.append((info, source_name))
            model_by_source[source_name] = (path, info)
        report = discover_models(entries, expected_layers=expected_layers)
    else:
        report = discovery_report
        if report.expected_layers != expected_layers:
            raise ToolError("DISCOVERY_EXPECTED_LAYER_COUNT_MISMATCH", str(report.expected_layers))
        source_objects = {source.source_name: source for source in report.sources}
        for path in source_paths:
            source_name = path.name if source_root.is_file() else path.relative_to(source_root).as_posix()
            source = source_objects.get(source_name)
            if source is None or source.model_info is None:
                raise ToolError("DISCOVERY_SOURCE_MAPPING_MISSING", source_name)
            model_by_source[source_name] = (path, source.model_info)
    coverage = report.coverage.get(profile)
    if coverage is None:
        raise ToolError("SOURCE_PROFILE_MISSING", "no {} sources were discovered".format(profile))
    if not coverage.get("complete"):
        raise ToolError(
            "SOURCE_LAYER_COVERAGE_INCOMPLETE",
            "{} coverage is {}/{}; missing={}, duplicate={}, unexpected={}".format(
                profile,
                coverage.get("covered_count"),
                coverage.get("expected_count"),
                coverage.get("missing_layers"),
                coverage.get("duplicate_layers"),
                coverage.get("unexpected_layers"),
            ),
        )
    if not report.structural_pass:
        codes = sorted({code for code, _ in report.issues})
        raise ToolError("SOURCE_DISCOVERY_FAILED", "source graph checks failed: {}".format(codes))
    selected_sources = [source for source in report.sources if profile in source.profiles]
    selected_layers = [layer for source in selected_sources for layer in source.layers if layer.profile == profile]
    if len(selected_layers) != expected_layers:
        raise ToolError(
            "SOURCE_LAYER_COUNT_MISMATCH",
            "found {} {} layer records; expected {}".format(len(selected_layers), profile, expected_layers),
        )
    for source in selected_sources:
        if source.graph_count != 1:
            raise ToolError(
                "UNSUPPORTED_GROUPED_EXPORT",
                "{} contains {} subgraphs; exactly one subgraph per source file is required".format(
                    source.source_name, source.graph_count
                ),
            )

    output_root = Path(output_directory)
    if output_root.exists():
        raise ToolError("OUTPUT_DIRECTORY_EXISTS", "refusing to reuse {}".format(output_root))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".{}-build-".format(output_root.name), dir=str(output_root.parent)))
    staged_output = temporary_root / "artifact"
    staged_output.mkdir()
    all_models = []
    layer_manifests = []
    layer_manifest_hashes = {}
    row_reduce_semantics_by_layer = {}
    golden_probes = []
    try:
        for source in selected_sources:
            path, info = model_by_source[source.source_name]
            if info.version != 3 or len(info.graphs) != 1:
                raise ToolError("SOURCE_GRAPH_UNSUPPORTED", "{} is not a single-subgraph TFLite v3 graph".format(source.source_name))
            adapter = TFLiteObjectAdapter(schema_module=schema_module)
            model_object = adapter.load(path)
            source_layers = sorted(
                (layer for layer in source.layers if layer.profile == profile),
                key=lambda layer: layer.layer_id,
            )
            analyses_by_layer = {
                analysis.layer_id: analysis
                for analysis in source.graph_analyses
                if analysis.graph_index == 0
            }
            for layer in source_layers:
                analysis = analyses_by_layer.get(layer.layer_id)
                if analysis is None:
                    raise ToolError("GRAPH_ANALYSIS_LAYER_MISSING", "{} layer {}".format(source.source_name, layer.layer_id))
                local_output = staged_output / ".layers" / "layer{:02d}".format(layer.layer_id)
                layer_build_options = {
                    "degree": degree,
                    "schema_module": adapter.schema,
                }
                if fc_width_padding_multiplier is not None:
                    layer_build_options["fc_width_padding_multiplier"] = fc_width_padding_multiplier
                if (
                    mlp_gate_dequant_width_padding_multiplier is not None
                    and degree == 4
                    and profile == "P128"
                    and layer.layer_id == 0
                ):
                    layer_build_options["mlp_gate_dequant_width_padding_multiplier"] = (
                        mlp_gate_dequant_width_padding_multiplier
                    )
                layer_manifest = _build_tp_layer_into(
                    model_object,
                    info,
                    layer,
                    analysis,
                    local_output,
                    **layer_build_options,
                )
                for rank in range(degree):
                    source_layer = local_output / profile.lower() / "rank{}".format(rank) / "layer{:02d}".format(layer.layer_id)
                    target_layer = staged_output / profile.lower() / "rank{}".format(rank) / "layer{:02d}".format(layer.layer_id)
                    if not source_layer.is_dir():
                        raise ToolError("COMPONENT_OUTPUT_MISSING", str(source_layer))
                    target_layer.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(str(source_layer), str(target_layer))
                manifest_path = staged_output / "manifests" / "{}_layer{:02d}.json".format(profile.lower(), layer.layer_id)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(layer_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                manifest_relative = manifest_path.relative_to(staged_output).as_posix()
                layer_manifest_hashes[manifest_relative] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if degree == 2:
                    row_reduce_semantics_by_layer[str(layer.layer_id)] = layer_manifest.get(
                        "collective_contract", {}
                    ).get(
                        "quantized_partial_semantics",
                        {
                            "status": "UNPROVEN",
                            "proof_sha256": None,
                            "reason": "The layer builder did not provide row-reduce semantics evidence.",
                        },
                    )
                for row in layer_manifest["models"]:
                    all_models.append(dict(row))
                layer_manifests.append(manifest_relative)
                golden_probes.append({
                    "layer": layer.layer_id,
                    "probe": layer_manifest["numerical_gate"],
                })
                shutil.rmtree(str(local_output), ignore_errors=True)
            del model_object

        aggregate = {
            "schema": 1,
            "stage": "M2_TP4_FULL_PROFILE_STRUCTURAL_PARTITION" if degree == 4 else "M3_TP2_FULL_PROFILE_STRUCTURAL_PARTITION",
            "topology": "TP4_FOUR_STAGE_ALL_GATHER" if degree == 4 else "TP2_ROW_REDUCE",
            "profile": profile,
            "fc_width_padding_rule": {
                "status": "APPLIED_TO_TP4_P128_D1_QKV" if fc_width_padding_multiplier is not None else "DISABLED",
                "multiplier": fc_width_padding_multiplier,
                "scope": "only TP4 P128 or D1 attention Q/K/V output-channel-sharded FC operators",
                "numerical_scope": "CPU_HOST_ONLY_NOT_AN_NPU_EXACTNESS_CLAIM",
            },
            "mlp_gate_dequant_width_padding_rule": {
                "status": "APPLIED_TO_TP4_P128_LAYER0_MLP_GATE" if mlp_gate_dequant_width_padding_multiplier is not None else "DISABLED",
                "multiplier": mlp_gate_dequant_width_padding_multiplier,
                "scope": "TP4 P128 Layer0 MLP Gate only",
                "numerical_scope": "STRUCTURAL_REWRITE_ONLY; new model SHA requires same-backend NPU retest",
            },
            "expected_layers": expected_layers,
            "source_coverage": coverage,
            "source_audit": report.to_dict(include_quant_values=False),
            "rank_order": list(range(degree)),
            "collective_contract": _collective_contract(degree, 128 if profile == "P128" else 1),
            "layer_manifests": layer_manifests,
            "layer_manifest_sha256": layer_manifest_hashes,
            "models": all_models,
            "model_count": len(all_models),
            "checksums_file": "SHA256SUMS",
            "golden_backend_probes": golden_probes,
            "weight_reconstruction": "PASS",
            "quantization_contract": {
                "layer_manifests": layer_manifests,
                "layer_manifest_sha256": layer_manifest_hashes,
                "source_quantization_vectors_in_layer_manifests": True,
            },
            "structural_status": "STRUCTURAL_PASS",
            "numerical_status": "GOLDEN_EXECUTION_UNAVAILABLE",
            "mismatch_count": None,
            "host_verification_status": "NOT_RUN",
            "npu_verification_status": "NPU_UNVERIFIED",
            "golden_report_sha256": None,
            "auto_repair_report": {
                "status": "BLOCKED",
                "attempts": [],
                "first_failure": "GOLDEN_EXECUTION_UNAVAILABLE",
            },
            "numerical_gate": {
                "code": "GOLDEN_EXECUTION_UNAVAILABLE",
                "reason": "No verified canonical fixture runner is registered.",
                "probes": golden_probes,
            },
            "release_status": "NOT_ELIGIBLE",
        }
        if degree == 2:
            aggregate["collective_contract"]["quantized_partial_semantics"] = {
                "status": "UNPROVEN",
                "proof_sha256": None,
                "per_layer": row_reduce_semantics_by_layer,
                "reason": "No bit-exact proof covers both O and Down row-reduce boundaries for every layer.",
            }
            aggregate["numerical_gate"]["row_reduce_semantics"] = "UNPROVEN"
        checksum_rows = [
            (row["output_sha256"], row["path"])
            for row in all_models
        ] + [
            (digest, relative)
            for relative, digest in layer_manifest_hashes.items()
        ]
        (staged_output / "SHA256SUMS").write_text(
            "".join("{}  {}\n".format(digest, relative) for digest, relative in sorted(checksum_rows, key=lambda row: row[1])),
            encoding="utf-8",
        )
        (staged_output / "MANIFEST.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(str(staged_output), str(output_root))
        return aggregate
    finally:
        shutil.rmtree(str(temporary_root), ignore_errors=True)


def build_tp4_directory(
    source_directory,
    output_directory,
    profile="D1",
    expected_layers=32,
    schema_module=None,
    discovery_report=None,
    fc_width_padding_multiplier=None,
    mlp_gate_dequant_width_padding_multiplier=None,
):
    """Build a TP4 profile using four rank-local output-channel shards."""
    return _build_tp_directory(
        source_directory,
        output_directory,
        profile,
        expected_layers,
        degree=4,
        schema_module=schema_module,
        discovery_report=discovery_report,
        fc_width_padding_multiplier=fc_width_padding_multiplier,
        mlp_gate_dequant_width_padding_multiplier=mlp_gate_dequant_width_padding_multiplier,
    )


def build_tp2_directory(
    source_directory,
    output_directory,
    profile="D1",
    expected_layers=32,
    schema_module=None,
    discovery_report=None,
):
    """Build a TP2 candidate using input-channel O/Down and row-reduce contracts."""
    return _build_tp_directory(
        source_directory,
        output_directory,
        profile,
        expected_layers,
        degree=2,
        schema_module=schema_module,
        discovery_report=discovery_report,
    )
