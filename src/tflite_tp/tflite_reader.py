"""Read-only TFLite FlatBuffer metadata reader.

The generated TFLite schema module and FlatBuffers runtime are runtime
dependencies. They are imported lazily so graph-planning unit tests can run
offline without a TFLite SDK installation.
"""

import hashlib
import importlib
from pathlib import Path

from .errors import TFLiteDependencyError, ToolError
from .model import GraphInfo, ModelInfo, OperatorInfo, QuantizationInfo, TensorInfo


def _decode(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _vector(reader, field):
    length_method = getattr(reader, field + "Length", None)
    item_method = getattr(reader, field, None)
    if length_method is None or item_method is None:
        return ()
    length = int(length_method())
    numpy_method = getattr(reader, field + "AsNumpy", None)
    if numpy_method is not None:
        array = numpy_method()
        # Generated FlatBuffer accessors use integer 0 as the default for an
        # absent vector. Accept vector-like results, but fall back to the
        # accessor when the generated module returns that scalar sentinel.
        if hasattr(array, "tolist"):
            values = array.tolist()
            if isinstance(values, list):
                return tuple(values)
            if length == 1:
                return (values,)
        elif isinstance(array, (list, tuple)):
            return tuple(array)
    return tuple(item_method(index) for index in range(length))


def _enum_names(enum_class):
    result = {}
    for name, value in vars(enum_class).items():
        if name.startswith("_") or not isinstance(value, int):
            continue
        result.setdefault(int(value), name)
    return result


def _enum_label(enum_class, value, prefix):
    return _enum_names(enum_class).get(int(value), "{}_{}".format(prefix, value))


def _quantization(tensor):
    reader = tensor.Quantization()
    if reader is None:
        return None
    return QuantizationInfo(
        scales=tuple(float(value) for value in _vector(reader, "Scale")),
        zero_points=tuple(int(value) for value in _vector(reader, "ZeroPoint")),
        quantized_dimension=int(reader.QuantizedDimension()),
        minimum=tuple(float(value) for value in _vector(reader, "Min")),
        maximum=tuple(float(value) for value in _vector(reader, "Max")),
    )


def _buffer_size(buffer):
    method = getattr(buffer, "DataLength", None)
    return int(method()) if method is not None else 0


def _has_sparsity(tensor):
    method = getattr(tensor, "Sparsity", None)
    return method is not None and method() is not None


def _read_model(raw, source_path, source_sha256, schema):
    root = schema.Model.GetRootAsModel(raw, 0)
    tensor_types = _enum_names(schema.TensorType)
    builtin_ops = _enum_names(schema.BuiltinOperator)
    option_types = _enum_names(schema.BuiltinOptions)
    buffer_count = int(root.BuffersLength())

    custom_codes = []
    operator_codes = []
    for index in range(int(root.OperatorCodesLength())):
        code = root.OperatorCodes(index)
        builtin = int(code.BuiltinCode())
        placeholder = getattr(schema.BuiltinOperator, "PLACEHOLDER_FOR_GREATER_OP_CODES", -999)
        if builtin == int(placeholder):
            builtin = int(code.DeprecatedBuiltinCode())
        custom = _decode(code.CustomCode())
        custom_codes.append(custom)
        operator_codes.append(custom if custom else builtin_ops.get(builtin, "BUILTIN_{}".format(builtin)))

    graphs = []
    for graph_index in range(int(root.SubgraphsLength())):
        subgraph = root.Subgraphs(graph_index)
        tensors = []
        for tensor_index in range(int(subgraph.TensorsLength())):
            tensor = subgraph.Tensors(tensor_index)
            buffer_index = int(tensor.Buffer())
            buffer_bytes = 0
            if 0 <= buffer_index < buffer_count:
                buffer_bytes = _buffer_size(root.Buffers(buffer_index))
            tensors.append(
                TensorInfo(
                    index=tensor_index,
                    name=_decode(tensor.Name()),
                    shape=tuple(int(value) for value in _vector(tensor, "Shape")),
                    shape_signature=tuple(int(value) for value in _vector(tensor, "ShapeSignature")),
                    dtype=tensor_types.get(int(tensor.Type()), "TENSOR_TYPE_{}".format(tensor.Type())),
                    buffer_index=buffer_index,
                    buffer_bytes=buffer_bytes,
                    quantization=_quantization(tensor),
                    is_variable=bool(tensor.IsVariable()),
                    is_sparse=_has_sparsity(tensor),
                )
            )

        operators = []
        for op_index in range(int(subgraph.OperatorsLength())):
            op = subgraph.Operators(op_index)
            opcode_index = int(op.OpcodeIndex())
            if opcode_index < 0 or opcode_index >= len(operator_codes):
                opcode = "OPCODE_INDEX_{}".format(opcode_index)
            else:
                opcode = operator_codes[opcode_index]
            option_type_id = int(op.BuiltinOptionsType())
            option_type = option_types.get(option_type_id, "OPTIONS_{}".format(option_type_id))
            operators.append(
                OperatorInfo(
                    index=op_index,
                    opcode=opcode,
                    inputs=tuple(int(value) for value in _vector(op, "Inputs")),
                    outputs=tuple(int(value) for value in _vector(op, "Outputs")),
                    options_type=option_type,
                )
            )

        graphs.append(
            GraphInfo(
                index=graph_index,
                name=_decode(subgraph.Name()),
                tensors=tuple(tensors),
                inputs=tuple(int(value) for value in _vector(subgraph, "Inputs")),
                outputs=tuple(int(value) for value in _vector(subgraph, "Outputs")),
                operators=tuple(operators),
            )
        )

    return ModelInfo(
        source_path=str(source_path),
        source_sha256=source_sha256,
        source_bytes=len(raw),
        version=int(root.Version()),
        buffer_count=buffer_count,
        graphs=tuple(graphs),
    )


def read_tflite_bytes(raw, source_name="<memory>", schema_module=None):
    """Parse TFLite bytes and retain graph and quantization metadata only."""
    if len(raw) < 8 or raw[4:8] != b"TFL3":
        raise ToolError("TFLITE_IDENTIFIER_INVALID", "{} is not a TFL3 FlatBuffer".format(source_name))

    if schema_module is None:
        try:
            schema_module = importlib.import_module("tflite")
        except ImportError as exc:
            raise TFLiteDependencyError(
                "TFLITE_SCHEMA_UNAVAILABLE",
                "install the generated 'tflite' schema module and FlatBuffers runtime to read model files",
            ) from exc

    try:
        digest = hashlib.sha256(raw).hexdigest()
        return _read_model(raw, source_name, digest, schema_module)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError("TFLITE_PARSE_FAILED", "{}: {}".format(source_name, exc)) from exc


def read_tflite(path, schema_module=None):
    """Parse one TFLite file and retain only graph and quantization metadata."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ToolError("SOURCE_READ_FAILED", "{}: {}".format(path.name, exc)) from exc
    return read_tflite_bytes(raw, source_name=str(path), schema_module=schema_module)
