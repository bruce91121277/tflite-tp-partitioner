"""Packed TFLite INT4 tensor slicing and reconstruction helpers."""

import math

import numpy as np

from .errors import ToolError
from .model import QuantizationInfo


def _validate_shape(shape):
    if not shape or any(int(dimension) <= 0 for dimension in shape):
        raise ToolError("INT4_SHAPE_INVALID", "INT4 slicing requires a non-empty static shape")
    return tuple(int(dimension) for dimension in shape)


def _unpack_nibbles(payload, value_count):
    raw = np.frombuffer(payload, dtype=np.uint8)
    values = np.empty(raw.size * 2, dtype=np.uint8)
    values[0::2] = raw & 0x0F
    values[1::2] = raw >> 4
    return values[:value_count]


def _pack_nibbles(values):
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if values.size % 2:
        values = np.pad(values, (0, 1), mode="constant")
    packed = values[0::2] | (values[1::2] << 4)
    return packed.tobytes()


def _validate_unused_nibble(payload, element_count, label):
    if element_count % 2 and payload and (payload[-1] & 0xF0):
        raise ToolError(
            "INT4_PADDING_NIBBLE_NONZERO",
            "{} has a non-zero unused high nibble; exact byte reconstruction is not defined".format(label),
        )


def slice_packed_int4(payload, shape, axis, start, count):
    """Slice a TFLite INT4 buffer along one logical tensor axis.

    TFLite stores the first logical value in each byte's low nibble. Slicing
    operates on unpacked values, then repacks from the first nibble so odd
    shard sizes do not inherit a misaligned source nibble offset.
    """
    shape = _validate_shape(shape)
    if axis < 0 or axis >= len(shape):
        raise ToolError("INT4_AXIS_INVALID", "axis {} is outside rank {}".format(axis, len(shape)))
    if start < 0 or count <= 0 or start + count > shape[axis]:
        raise ToolError("INT4_SLICE_RANGE_INVALID", "slice [{}, {}) is outside axis length {}".format(start, start + count, shape[axis]))
    element_count = math.prod(shape)
    expected_bytes = (element_count + 1) // 2
    if len(payload) != expected_bytes:
        raise ToolError(
            "INT4_BUFFER_LENGTH_MISMATCH",
            "buffer has {} bytes; shape {} requires {}".format(len(payload), shape, expected_bytes),
        )
    _validate_unused_nibble(payload, element_count, "source tensor")
    values = _unpack_nibbles(payload, element_count).reshape(shape)
    selected = np.take(values, np.arange(start, start + count), axis=axis).reshape(-1)
    return _pack_nibbles(selected)


def reconstruct_packed_int4(shards, source_shape, axis):
    """Reconstruct a tensor from rank-ordered (payload, shard_shape) pairs."""
    source_shape = _validate_shape(source_shape)
    if axis < 0 or axis >= len(source_shape):
        raise ToolError("INT4_AXIS_INVALID", "axis {} is outside rank {}".format(axis, len(source_shape)))
    if not shards:
        raise ToolError("INT4_SHARDS_EMPTY", "at least one shard is required")
    arrays = []
    for rank, (payload, shard_shape) in enumerate(shards):
        shard_shape = _validate_shape(shard_shape)
        if len(shard_shape) != len(source_shape):
            raise ToolError("INT4_SHARD_RANK_MISMATCH", "rank {} shard rank differs from source".format(rank))
        for dim, (actual, expected) in enumerate(zip(shard_shape, source_shape)):
            if dim != axis and actual != expected:
                raise ToolError("INT4_SHARD_SHAPE_MISMATCH", "rank {} differs on non-sharded axis {}".format(rank, dim))
        if len(payload) != (math.prod(shard_shape) + 1) // 2:
            raise ToolError("INT4_BUFFER_LENGTH_MISMATCH", "rank {} packed buffer length does not match its shape".format(rank))
        _validate_unused_nibble(payload, math.prod(shard_shape), "rank {} shard".format(rank))
        arrays.append(_unpack_nibbles(payload, math.prod(shard_shape)).reshape(shard_shape))
    result = np.concatenate(arrays, axis=axis)
    if tuple(result.shape) != source_shape:
        raise ToolError(
            "INT4_RECONSTRUCTION_SHAPE_MISMATCH",
            "reconstructed shape {} differs from source {}".format(tuple(result.shape), source_shape),
        )
    return _pack_nibbles(result.reshape(-1))


def slice_quantization(info, shape, axis, start, count):
    """Slice per-axis or aligned per-group quantization metadata.

    A multi-scale vector on another axis remains unchanged. If it is on the
    sliced axis, the vector must be per-channel or divide the axis into
    uniform groups that are not split across rank boundaries.
    """
    if info is None:
        return None
    shape = tuple(int(dimension) for dimension in shape)
    if axis < 0 or axis >= len(shape) or start < 0 or count <= 0 or start + count > shape[axis]:
        raise ToolError("QUANT_SLICE_RANGE_INVALID", "quantization slice is outside the tensor shape")
    scale_count = len(info.scales)
    for name, values in (("zero point", info.zero_points), ("minimum", info.minimum), ("maximum", info.maximum)):
        if values and (not scale_count or len(values) not in (1, scale_count)):
            code = (
                "QUANT_ZERO_POINT_COUNT_INVALID"
                if name == "zero point"
                else "QUANT_RANGE_METADATA_LENGTH_MISMATCH"
            )
            raise ToolError(code, "{} count does not match scale count".format(name))
    if scale_count > 1 and (info.quantized_dimension < 0 or info.quantized_dimension >= len(shape)):
        raise ToolError("QUANT_AXIS_OUT_OF_RANGE", "quantized dimension is outside tensor rank")
    if not info.scales or len(info.scales) == 1 or info.quantized_dimension != axis:
        return info
    if scale_count == shape[axis]:
        scale_start, scale_end = start, start + count
    elif shape[axis] % scale_count == 0:
        group_size = shape[axis] // scale_count
        if start % group_size or count % group_size:
            raise ToolError(
                "QUANT_GROUP_BOUNDARY_SPLIT",
                "slice boundary splits a quantization group of {} values".format(group_size),
            )
        scale_start, scale_end = start // group_size, (start + count) // group_size
    else:
        raise ToolError(
            "QUANT_SCALE_LAYOUT_UNSUPPORTED",
            "{} scales do not map to {} channels or uniform groups".format(scale_count, shape[axis]),
        )

    def slice_optional(values, name):
        if not values or len(values) == 1:
            return values
        if len(values) == scale_count:
            return values[scale_start:scale_end]
        code = (
            "QUANT_ZERO_POINT_COUNT_INVALID"
            if name == "zero point"
            else "QUANT_RANGE_METADATA_LENGTH_MISMATCH"
        )
        raise ToolError(code, "{} vector length differs from scale count".format(name))

    if info.zero_points and len(info.zero_points) not in (1, scale_count):
        raise ToolError("QUANT_ZERO_POINT_COUNT_INVALID", "zero-point count differs from scale count")
    return QuantizationInfo(
        scales=info.scales[scale_start:scale_end],
        zero_points=slice_optional(info.zero_points, "zero point"),
        quantized_dimension=info.quantized_dimension,
        minimum=slice_optional(info.minimum, "minimum"),
        maximum=slice_optional(info.maximum, "maximum"),
    )
