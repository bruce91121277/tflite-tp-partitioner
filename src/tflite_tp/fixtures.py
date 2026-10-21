"""Deterministic, metadata-aware TFLite input fixtures for host Golden runs."""

from .errors import ToolError


_NUMPY_DTYPES = {
    "BOOL": "bool",
    "INT8": "int8",
    "UINT8": "uint8",
    "INT16": "int16",
    "UINT16": "uint16",
    "INT32": "int32",
    "UINT32": "uint32",
    "INT64": "int64",
    "UINT64": "uint64",
    "FLOAT16": "float16",
    "FLOAT32": "float32",
    "FLOAT64": "float64",
}


def _quantize_real(tensor, real_values, dtype, np):
    quant = tensor.quantization
    if quant is None or not quant.scales:
        return np.asarray(real_values, dtype=dtype)
    scales = np.asarray(quant.scales, dtype=np.float64)
    zero_points = np.asarray(quant.zero_points or (0,), dtype=np.float64)
    if np.any(scales <= 0):
        raise ToolError("GOLDEN_FIXTURE_QUANTIZATION_INVALID", tensor.name)
    if len(scales) == 1:
        real_quant = np.asarray(real_values, dtype=np.float64) / scales[0] + zero_points[0]
    else:
        axis = int(quant.quantized_dimension)
        if axis < 0 or axis >= len(tensor.shape) or len(scales) != int(tensor.shape[axis]):
            raise ToolError(
                "GOLDEN_FIXTURE_QUANTIZATION_INVALID",
                "{}: scales={} axis={} shape={}".format(tensor.name, len(scales), axis, tensor.shape),
            )
        if len(zero_points) == 1:
            zero_points = np.repeat(zero_points, len(scales))
        if len(zero_points) != len(scales):
            raise ToolError("GOLDEN_FIXTURE_QUANTIZATION_INVALID", tensor.name)
        broadcast = [1] * len(tensor.shape)
        broadcast[axis] = len(scales)
        real_quant = (
            np.asarray(real_values, dtype=np.float64) / scales.reshape(broadcast)
            + zero_points.reshape(broadcast)
        )
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        real_quant = np.clip(np.rint(real_quant), limits.min, limits.max)
    return np.asarray(real_quant, dtype=dtype)


def make_deterministic_tflite_inputs(graph, seed=0):
    """Create deterministic valid-shape inputs using each graph's tensor metadata.

    Masks are all-visible (real value zero); rotary embeddings encode cos=1,
    sin=0 when represented as a two-lane input. Cache and hidden tensors use
    bounded seeded values so the Golden exercises non-zero paths.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ToolError("GOLDEN_NUMPY_UNAVAILABLE", str(exc))
    if not isinstance(seed, int):
        raise ToolError("GOLDEN_FIXTURE_SEED_INVALID", repr(seed))
    rng = np.random.default_rng(seed)
    fixtures = {}
    for tensor_index in graph.inputs:
        tensor = graph.tensors[tensor_index]
        shape = tuple(int(dimension) for dimension in tensor.shape)
        if any(dimension <= 0 for dimension in shape):
            raise ToolError(
                "GOLDEN_FIXTURE_SHAPE_DYNAMIC",
                "{}: {}".format(tensor.name, shape),
            )
        dtype_name = str(tensor.dtype).upper()
        if dtype_name not in _NUMPY_DTYPES:
            raise ToolError("GOLDEN_FIXTURE_DTYPE_UNSUPPORTED", "{}: {}".format(tensor.name, dtype_name))
        dtype = np.dtype(_NUMPY_DTYPES[dtype_name])
        lower_name = tensor.name.lower()
        if "mask" in lower_name:
            values = np.zeros(shape, dtype=np.float64)
            fixtures[tensor.name] = _quantize_real(tensor, values, dtype, np)
        elif "pos_emb" in lower_name or "position_embedding" in lower_name or "rotary" in lower_name:
            values = np.zeros(shape, dtype=np.float64)
            if len(shape) >= 2 and shape[1] == 2:
                cosine = [slice(None)] * len(shape)
                cosine[1] = 0
                values[tuple(cosine)] = 1.0
            fixtures[tensor.name] = _quantize_real(tensor, values, dtype, np)
        elif "position_ids" in lower_name or "position_id" in lower_name:
            fixtures[tensor.name] = np.zeros(shape, dtype=dtype)
        elif np.issubdtype(dtype, np.bool_):
            fixtures[tensor.name] = np.zeros(shape, dtype=dtype)
        elif np.issubdtype(dtype, np.integer):
            limits = np.iinfo(dtype)
            low = max(int(limits.min), -16) if np.issubdtype(dtype, np.signedinteger) else 0
            high = min(int(limits.max), 16)
            fixtures[tensor.name] = rng.integers(low, high + 1, size=shape, dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            fixtures[tensor.name] = rng.uniform(-0.125, 0.125, size=shape).astype(dtype)
        else:
            raise ToolError("GOLDEN_FIXTURE_DTYPE_UNSUPPORTED", "{}: {}".format(tensor.name, dtype_name))
    if len(fixtures) != len(graph.inputs):
        raise ToolError("GOLDEN_FIXTURE_INPUT_COVERAGE_INVALID", graph.name)
    return fixtures
