"""Generic TFLite schema object reader and FlatBuffer serializer.

The generated ``tflite`` schema package exposes table readers and Add methods,
but not all releases ship object API classes. This adapter builds object tables
from those accessors, including populated option unions, and fails closed when
an unknown union or unsupported vector type is encountered.
"""

import copy
import importlib
import os
from pathlib import Path

import numpy as np

from .errors import TFLiteDependencyError, ToolError


def _members(schema, table_name):
    prefix = table_name + "Add"
    return sorted(
        (name[len(prefix) :], function)
        for name, function in vars(schema).items()
        if name.startswith(prefix) and callable(function)
    )


def _attribute(field_name):
    return field_name[0].lower() + field_name[1:]


class _ObjectTable:
    def __init__(self, table_name, adapter):
        self._table_name = table_name
        self._adapter = adapter

    def __deepcopy__(self, memo):
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            setattr(clone, name, value if name == "_adapter" else copy.deepcopy(value, memo))
        return clone

    def Pack(self, builder):
        fields = _members(self._adapter.schema, self._table_name)
        prepared = {}
        for field_name, _ in fields:
            value = getattr(self, _attribute(field_name), None)
            if value is None:
                continue
            if isinstance(value, _ObjectTable):
                value = value.Pack(builder)
            elif isinstance(value, (bytes, str)):
                value = builder.CreateString(value)
            elif isinstance(value, np.ndarray):
                value = builder.CreateNumpyVector(value)
            elif isinstance(value, (list, tuple)):
                if not value:
                    continue
                if all(isinstance(item, _ObjectTable) for item in value):
                    offsets = [item.Pack(builder) for item in value]
                    getattr(self._adapter.schema, self._table_name + "Start" + field_name + "Vector")(
                        builder, len(offsets)
                    )
                    for offset in reversed(offsets):
                        builder.PrependUOffsetTRelative(offset)
                    value = builder.EndVector()
                elif all(isinstance(item, (bytes, str)) for item in value):
                    offsets = [builder.CreateString(item) for item in value]
                    getattr(self._adapter.schema, self._table_name + "Start" + field_name + "Vector")(
                        builder, len(offsets)
                    )
                    for offset in reversed(offsets):
                        builder.PrependUOffsetTRelative(offset)
                    value = builder.EndVector()
                else:
                    raise ToolError(
                        "SCHEMA_SCALAR_VECTOR_UNSUPPORTED",
                        "{} has a non-array scalar vector {}".format(self._table_name, field_name),
                    )
            else:
                # Numeric and enum scalar fields are consumed directly by AddX.
                pass
            prepared[field_name] = value

        getattr(self._adapter.schema, self._table_name + "Start")(builder)
        for field_name, add_function in fields:
            if field_name in prepared:
                add_function(builder, prepared[field_name])
        return getattr(self._adapter.schema, self._table_name + "End")(builder)


def normalize_host_scalar_constant_shapes(model, graph, schema_module):
    """Normalize safe one-element constants for the MTK Host importer.

    The Host importer accepts these importer-sensitive scalar operands as
    shape ``[1]``. Their bytes and quantization are not changed. Only scalar
    constants used as a SPLIT axis or a broadcast MUL operand are eligible.
    """
    split_code = int(getattr(schema_module.BuiltinOperator, "SPLIT"))
    mul_code = int(getattr(schema_module.BuiltinOperator, "MUL"))
    int16_type = int(getattr(schema_module.TensorType, "INT16"))
    int32_type = int(getattr(schema_module.TensorType, "INT32"))
    consumers = {}
    placeholder = int(getattr(schema_module.BuiltinOperator, "PLACEHOLDER_FOR_GREATER_OP_CODES", -999))
    for op_index, op in enumerate(graph.operators):
        if int(op.opcodeIndex) < 0 or int(op.opcodeIndex) >= len(model.operatorCodes):
            continue
        opcode = model.operatorCodes[int(op.opcodeIndex)]
        builtin = int(opcode.builtinCode)
        if builtin == placeholder:
            builtin = int(opcode.deprecatedBuiltinCode)
        for input_position, tensor_index in enumerate(op.inputs):
            tensor_index = int(tensor_index)
            if tensor_index >= 0:
                consumers.setdefault(tensor_index, []).append((builtin, input_position, op_index))

    normalized = []
    for tensor_index, tensor in enumerate(graph.tensors):
        shape = tuple(int(value) for value in np.asarray(tensor.shape if tensor.shape is not None else []).reshape(-1))
        if shape or int(tensor.buffer) <= 0 or int(tensor.buffer) >= len(model.buffers):
            continue
        tensor_type = int(tensor.type)
        if tensor_type not in (int16_type, int32_type):
            continue
        data = getattr(model.buffers[int(tensor.buffer)], "data", None)
        if data is None:
            continue
        raw = np.asarray(data, dtype=np.uint8).reshape(-1)
        if raw.size != (2 if tensor_type == int16_type else 4):
            continue
        uses = consumers.get(tensor_index, [])
        if not uses or not all(
            (builtin == split_code and input_position == 0) or builtin == mul_code
            for builtin, input_position, _ in uses
        ):
            continue
        tensor.shape = np.asarray([1], dtype=np.int32)
        normalized.append({"tensor_index": tensor_index, "consumer_operators": [item[2] for item in uses]})
    return normalized


class TFLiteObjectAdapter:
    def __init__(self, schema_module=None):
        if schema_module is None:
            try:
                schema_module = importlib.import_module("tflite")
            except ImportError as exc:
                raise TFLiteDependencyError(
                    "TFLITE_SCHEMA_UNAVAILABLE",
                    "install the generated 'tflite' schema module and FlatBuffers runtime to rewrite model files",
                ) from exc
        self.schema = schema_module

    def _union(self, reader, field_name, table):
        if table is None:
            return None
        type_reader = getattr(reader, field_name + "Type", None)
        if type_reader is None:
            raise ToolError("SCHEMA_UNION_TYPE_ACCESSOR_MISSING", field_name)
        enum_value = int(type_reader())
        if not enum_value:
            return None
        enum_class = getattr(self.schema, field_name, None)
        if enum_class is None:
            raise ToolError("SCHEMA_UNION_ENUM_MISSING", field_name)
        union_name = next(
            (
                name
                for name, value in vars(enum_class).items()
                if not name.startswith("_") and isinstance(value, int) and value == enum_value
            ),
            None,
        )
        if union_name is None or not hasattr(self.schema, union_name):
            raise ToolError(
                "SCHEMA_UNION_UNSUPPORTED",
                "populated {} union type {} has no generated reader".format(field_name, enum_value),
            )
        union_reader = getattr(self.schema, union_name)()
        union_reader.Init(table.Bytes, table.Pos)
        return self._unpack(union_reader)

    def _unpack(self, reader):
        if reader is None:
            return None
        table_name = reader.__class__.__name__
        result = _ObjectTable(table_name, self)
        for field_name, _ in _members(self.schema, table_name):
            method = getattr(reader, field_name, None)
            if method is None:
                raise ToolError(
                    "SCHEMA_FIELD_ACCESSOR_MISSING",
                    "{} has no {} accessor".format(table_name, field_name),
                )
            length_method = getattr(reader, field_name + "Length", None)
            if length_method is not None:
                length = int(length_method())
                if not length:
                    value = None
                else:
                    numpy_method = getattr(reader, field_name + "AsNumpy", None)
                    array = numpy_method() if numpy_method is not None else None
                    if isinstance(array, np.ndarray):
                        value = np.array(array, copy=True)
                    elif isinstance(array, (list, tuple)):
                        value = np.asarray(array)
                    else:
                        values = []
                        for index in range(length):
                            item = method(index)
                            if hasattr(item, "_tab"):
                                item = self._unpack(item)
                            elif hasattr(item, "Bytes") and hasattr(item, "Pos"):
                                item = self._union(reader, field_name, item)
                            values.append(item)
                        value = values
            else:
                value = method()
                if isinstance(value, bytes) or value is None:
                    pass
                elif hasattr(value, "_tab"):
                    value = self._unpack(value)
                elif hasattr(value, "Bytes") and hasattr(value, "Pos"):
                    value = self._union(reader, field_name, value)
            setattr(result, _attribute(field_name), value)
        return result

    def load_bytes(self, raw, source_name="model.tflite"):
        if len(raw) < 8 or raw[4:8] != b"TFL3":
            raise ToolError("TFLITE_IDENTIFIER_INVALID", "{} is not a TFL3 FlatBuffer".format(source_name))
        try:
            reader = self.schema.Model.GetRootAsModel(raw, 0)
            return self._unpack(reader)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError("TFLITE_PARSE_FAILED", "{}: {}".format(source_name, exc)) from exc

    def load(self, path):
        path = Path(path)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ToolError("SOURCE_READ_FAILED", "{}: {}".format(path.name, exc)) from exc
        return self.load_bytes(raw, source_name=path.name)

    def export(self, model):
        try:
            import flatbuffers
        except ImportError as exc:
            raise TFLiteDependencyError(
                "FLATBUFFERS_UNAVAILABLE",
                "install the flatbuffers runtime to serialize model files",
            ) from exc
        try:
            builder = flatbuffers.Builder(1 << 20)
            builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
            return bytes(builder.Output())
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError("TFLITE_SERIALIZATION_FAILED", str(exc)) from exc

    def write(self, model, path, exclusive=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.export(model)
        mode = "xb" if exclusive else "wb"
        created = False
        try:
            with path.open(mode) as stream:
                created = True
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise ToolError("OUTPUT_EXISTS", "refusing to overwrite {}".format(path)) from exc
        except OSError as exc:
            if created and exclusive:
                try:
                    path.unlink()
                except OSError:
                    pass
            raise ToolError("OUTPUT_WRITE_FAILED", "{}: {}".format(path, exc)) from exc


def load_tflite_object(path, schema_module=None):
    return TFLiteObjectAdapter(schema_module=schema_module).load(path)


def export_tflite_object(model, schema_module=None):
    return TFLiteObjectAdapter(schema_module=schema_module).export(model)
