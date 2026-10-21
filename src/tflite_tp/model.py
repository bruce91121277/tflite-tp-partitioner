"""Small immutable graph records used by discovery and synthetic tests.

Tensor payloads are deliberately not retained here. The TFLite reader records
buffer sizes and quantization metadata while leaving packed weights in the
source FlatBuffer.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict, deque
import hashlib
import math
import struct

from .errors import ModelValidationError


@dataclass(frozen=True)
class QuantizationInfo:
    scales: Tuple[float, ...] = ()
    zero_points: Tuple[int, ...] = ()
    quantized_dimension: int = 0
    minimum: Tuple[float, ...] = ()
    maximum: Tuple[float, ...] = ()


@dataclass(frozen=True)
class TensorInfo:
    index: int
    name: str
    shape: Tuple[int, ...]
    shape_signature: Tuple[int, ...]
    dtype: str
    buffer_index: int
    buffer_bytes: int
    quantization: Optional[QuantizationInfo] = None
    is_variable: bool = False
    is_sparse: bool = False


@dataclass(frozen=True)
class OperatorInfo:
    index: int
    opcode: str
    inputs: Tuple[int, ...]
    outputs: Tuple[int, ...]
    options_type: str = ""


@dataclass(frozen=True)
class GraphInfo:
    index: int
    name: str
    tensors: Tuple[TensorInfo, ...]
    inputs: Tuple[int, ...]
    outputs: Tuple[int, ...]
    operators: Tuple[OperatorInfo, ...]


@dataclass(frozen=True)
class ModelInfo:
    source_path: str
    source_sha256: str
    source_bytes: int
    version: int
    buffer_count: int
    graphs: Tuple[GraphInfo, ...]


@dataclass(frozen=True)
class ProjectionMatch:
    role: str
    tensor_index: int
    tensor_name: str
    shape: Tuple[int, ...]
    dtype: str
    operator_index: int
    operator: str
    activation_tensor_index: int
    output_tensor_indices: Tuple[int, ...]
    bias_tensor_index: Optional[int]
    quantization: Optional[QuantizationInfo]
    semantic_layer_id: Optional[int]
    weight_buffer_index: Optional[int] = None
    bias_shape: Tuple[int, ...] = ()
    bias_dtype: Optional[str] = None
    bias_buffer_index: Optional[int] = None
    bias_quantization: Optional[QuantizationInfo] = None
    weight_quantization_layout: str = "UNRESOLVED"
    operator_input_tensor_indices: Tuple[int, ...] = ()
    output_shapes: Tuple[Tuple[int, ...], ...] = ()


@dataclass
class GraphAnalysis:
    graph_index: int
    layer_id: Optional[int] = None
    tensor_producers: Dict[int, int] = field(default_factory=dict)
    tensor_consumers: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    operator_dependencies: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    topological_order: Tuple[int, ...] = ()
    input_roles: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    output_roles: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    features: Dict[str, bool] = field(default_factory=dict)
    evidence: Dict[str, object] = field(default_factory=dict)
    issues: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self):
        return {
            "graph_index": self.graph_index,
            "layer_id": self.layer_id,
            "tensor_producers": {str(key): value for key, value in sorted(self.tensor_producers.items())},
            "tensor_consumers": {str(key): list(value) for key, value in sorted(self.tensor_consumers.items())},
            "operator_dependencies": {str(key): list(value) for key, value in sorted(self.operator_dependencies.items())},
            "topological_order": list(self.topological_order),
            "input_roles": {key: list(value) for key, value in sorted(self.input_roles.items())},
            "output_roles": {key: list(value) for key, value in sorted(self.output_roles.items())},
            "features": self.features,
            "evidence": self.evidence,
            "issues": [{"code": code, "message": message} for code, message in self.issues],
        }


@dataclass
class LayerDiscovery:
    layer_id: int
    graph_index: int
    profile: str
    source_name: str
    source_sha256: str
    roles: Dict[str, ProjectionMatch] = field(default_factory=dict)
    issues: List[Tuple[str, str]] = field(default_factory=list)
    graph_features: Dict[str, bool] = field(default_factory=dict)

    @property
    def complete(self):
        return not self.issues and len(self.roles) == len(PROJECTION_ROLES)


@dataclass
class SourceDiscovery:
    source_name: str
    source_sha256: str
    source_bytes: int
    version: int
    graph_count: int
    profiles: List[str] = field(default_factory=list)
    layers: List[LayerDiscovery] = field(default_factory=list)
    issues: List[Tuple[str, str]] = field(default_factory=list)
    graph_analyses: List[GraphAnalysis] = field(default_factory=list)
    model_info: Optional[ModelInfo] = None


@dataclass
class DiscoveryReport:
    expected_layers: int
    sources: List[SourceDiscovery] = field(default_factory=list)
    coverage: Dict[str, Dict[str, object]] = field(default_factory=dict)
    issues: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def structural_pass(self):
        return not self.issues and all(layer.complete for source in self.sources for layer in source.layers)

    def to_dict(self, include_quant_values=False):
        return {
            "schema": 1,
            "stage": "M1_GRAPH_DISCOVERY",
            "structural_status": "STRUCTURAL_PASS" if self.structural_pass else "STRUCTURAL_FAIL",
            "numerical_status": "NOT_RUN",
            "release_status": "NOT_ELIGIBLE",
            "expected_layers": self.expected_layers,
            "coverage": self.coverage,
            "issues": [{"code": code, "message": message} for code, message in self.issues],
            "sources": [
                {
                    "source": source.source_name,
                    "source_sha256": source.source_sha256,
                    "source_bytes": source.source_bytes,
                    "tflite_version": source.version,
                    "graph_count": source.graph_count,
                    "profiles": source.profiles,
                    "issues": [{"code": code, "message": message} for code, message in source.issues],
                    "graph_analyses": [analysis.to_dict() for analysis in source.graph_analyses],
                    "layers": [
                        {
                            "layer": layer.layer_id,
                            "graph_index": layer.graph_index,
                            "profile": layer.profile,
                            "complete": layer.complete,
                            "roles": {
                                role: {
                                    "weight_tensor": item.tensor_index,
                                    "weight_name": item.tensor_name,
                                    "weight_shape": list(item.shape),
                                    "weight_dtype": item.dtype,
                                    "weight_quantization": quantization_dict(item.quantization, include_values=include_quant_values),
                                    "weight_quantization_layout": item.weight_quantization_layout,
                                    "weight_buffer": item.weight_buffer_index,
                                    "operator_index": item.operator_index,
                                    "operator": item.operator,
                                    "operator_inputs": list(item.operator_input_tensor_indices),
                                    "activation_tensor": item.activation_tensor_index,
                                    "output_tensors": list(item.output_tensor_indices),
                                    "output_shapes": [list(shape) for shape in item.output_shapes],
                                    "bias_tensor": item.bias_tensor_index,
                                    "bias_shape": list(item.bias_shape),
                                    "bias_dtype": item.bias_dtype,
                                    "bias_buffer": item.bias_buffer_index,
                                    "bias_quantization": quantization_dict(item.bias_quantization, include_values=include_quant_values),
                                    "semantic_layer_id": item.semantic_layer_id,
                                }
                                for role, item in sorted(layer.roles.items())
                            },
                            "issues": [{"code": code, "message": message} for code, message in layer.issues],
                            "graph_features": layer.graph_features,
                        }
                        for layer in source.layers
                    ],
                }
                for source in self.sources
            ],
        }


PROJECTION_ROLES = ("q", "k", "v", "o", "gate", "up", "down")

EXPECTED_WEIGHT_SHAPES = {
    "q": (4096, 4096),
    "k": (1024, 4096),
    "v": (1024, 4096),
    "o": (4096, 4096),
    "gate": (14336, 4096),
    "up": (14336, 4096),
    "down": (4096, 14336),
}


def _quant_vector_digest(values, format_character):
    if not values:
        return None
    encoded = struct.pack("<" + format_character * len(values), *values)
    return hashlib.sha256(encoded).hexdigest()


def quantization_dict(item, include_values=False):
    if item is None:
        return None
    result = {
        "scale_count": len(item.scales),
        "scale_sha256_f32le": _quant_vector_digest(item.scales, "f"),
        "scale_range": [min(item.scales), max(item.scales)] if item.scales else None,
        "zero_point_count": len(item.zero_points),
        "zero_point_sha256_i64le": _quant_vector_digest(item.zero_points, "q"),
        "quantized_dimension": item.quantized_dimension,
        "minimum_count": len(item.minimum),
        "minimum_sha256_f32le": _quant_vector_digest(item.minimum, "f"),
        "maximum_count": len(item.maximum),
        "maximum_sha256_f32le": _quant_vector_digest(item.maximum, "f"),
    }
    if include_values:
        result.update({
            "scales": list(item.scales),
            "zero_points": list(item.zero_points),
            "minimum": list(item.minimum),
            "maximum": list(item.maximum),
        })
    return result


def _issue(issues, code, message):
    issues.append((code, message))


def validate_graph(model, graph):
    """Check graph references and acyclicity without trusting operator order."""
    issues = []
    tensor_count = len(graph.tensors)
    graph_inputs = set(graph.inputs)
    graph_outputs = set(graph.outputs)

    for tensor_index in graph.inputs + graph.outputs:
        if tensor_index < 0 or tensor_index >= tensor_count:
            _issue(issues, "PUBLIC_TENSOR_OUT_OF_RANGE", "tensor index {} is outside the graph".format(tensor_index))

    producers = {}
    for tensor in graph.tensors:
        if tensor.index < 0 or tensor.index >= tensor_count:
            _issue(issues, "TENSOR_INDEX_INVALID", "tensor record has index {}".format(tensor.index))
        if tensor.buffer_index < 0 or tensor.buffer_index >= model.buffer_count:
            _issue(issues, "TENSOR_BUFFER_OUT_OF_RANGE", "tensor {} references buffer {}".format(tensor.index, tensor.buffer_index))
        if any(dimension < -1 for dimension in tensor.shape + tensor.shape_signature):
            _issue(issues, "TENSOR_SHAPE_INVALID", "tensor {} has a dimension smaller than -1".format(tensor.index))
        quant = tensor.quantization
        if quant is not None:
            if any(not math.isfinite(scale) or scale <= 0 for scale in quant.scales):
                _issue(issues, "QUANT_SCALE_INVALID", "tensor {} has a non-positive or non-finite scale".format(tensor.index))
            if quant.zero_points and len(quant.zero_points) not in (1, len(quant.scales)):
                _issue(issues, "QUANT_ZERO_POINT_COUNT_INVALID", "tensor {} has mismatched scale and zero-point counts".format(tensor.index))
            if len(quant.scales) > 1 and (quant.quantized_dimension < 0 or quant.quantized_dimension >= len(tensor.shape)):
                _issue(issues, "QUANT_AXIS_OUT_OF_RANGE", "tensor {} quantized dimension is outside its rank".format(tensor.index))
        if tensor.dtype == "INT4" and tensor.buffer_bytes and tensor.shape and not tensor.is_sparse:
            if all(dimension >= 0 for dimension in tensor.shape):
                element_count = math.prod(tensor.shape)
                expected_bytes = (element_count + 1) // 2
                if tensor.buffer_bytes != expected_bytes:
                    _issue(
                        issues,
                        "INT4_BUFFER_LENGTH_MISMATCH",
                        "tensor {} has {} packed bytes; shape {} requires {}".format(
                            tensor.index, tensor.buffer_bytes, tensor.shape, expected_bytes
                        ),
                    )

    operator_indices = set()
    for op in graph.operators:
        if op.index in operator_indices:
            _issue(issues, "OPERATOR_INDEX_DUPLICATE", "operator index {} is duplicated".format(op.index))
        operator_indices.add(op.index)
        for tensor_index in op.inputs:
            if tensor_index == -1:
                continue
            if tensor_index < 0 or tensor_index >= tensor_count:
                _issue(issues, "OP_INPUT_OUT_OF_RANGE", "operator {} input {} is outside the graph".format(op.index, tensor_index))
        for tensor_index in op.outputs:
            if tensor_index < 0 or tensor_index >= tensor_count:
                _issue(issues, "OP_OUTPUT_OUT_OF_RANGE", "operator {} output {} is outside the graph".format(op.index, tensor_index))
                continue
            previous = producers.get(tensor_index)
            if previous is not None:
                _issue(issues, "TENSOR_HAS_MULTIPLE_PRODUCERS", "tensor {} is written by operators {} and {}".format(tensor_index, previous, op.index))
            else:
                producers[tensor_index] = op.index

    for tensor_index in graph_inputs:
        if tensor_index in producers:
            _issue(issues, "GRAPH_INPUT_HAS_PRODUCER", "graph input tensor {} is also produced by an operator".format(tensor_index))

    operator_by_index = {op.index: op for op in graph.operators}
    dependencies = {op.index: set() for op in graph.operators}
    users = defaultdict(set)
    for op in graph.operators:
        for tensor_index in op.inputs:
            if tensor_index < 0 or tensor_index >= tensor_count:
                continue
            producer = producers.get(tensor_index)
            if producer is not None:
                dependencies[op.index].add(producer)
                users[producer].add(op.index)
            else:
                tensor = graph.tensors[tensor_index]
                is_constant = tensor.buffer_bytes > 0
                if tensor_index not in graph_inputs and not is_constant and not tensor.is_variable:
                    _issue(
                        issues,
                        "GRAPH_UNBOUND_OPERATOR_INPUT",
                        "operator {} reads tensor {} without a graph input, constant buffer, or producer".format(
                            op.index, tensor_index
                        ),
                    )

    ready = deque(index for index, deps in dependencies.items() if not deps)
    visited = set()
    while ready:
        index = ready.popleft()
        if index in visited:
            continue
        visited.add(index)
        for user in users.get(index, ()):
            dependencies[user].discard(index)
            if not dependencies[user]:
                ready.append(user)
    if len(visited) != len(operator_by_index):
        cycle = sorted(set(operator_by_index) - visited)
        _issue(issues, "GRAPH_CYCLE", "operator dependency cycle includes {}".format(cycle))

    for tensor_index in graph_outputs:
        if tensor_index < 0 or tensor_index >= tensor_count:
            continue
        tensor = graph.tensors[tensor_index]
        if tensor_index not in producers and tensor_index not in graph_inputs and tensor.buffer_bytes <= 0 and not tensor.is_variable:
            _issue(issues, "GRAPH_UNBOUND_PUBLIC_OUTPUT", "graph output tensor {} has no producer or input".format(tensor_index))

    if issues:
        raise ModelValidationError(issues)
