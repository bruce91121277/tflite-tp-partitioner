"""Automatic graph-boundary discovery for TP4 TFLite components."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

from .errors import ToolError
from .model import GraphInfo, LayerDiscovery, ModelInfo
from .planning import make_partition_plan


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    operator_indices: Tuple[int, ...]
    input_tensor_indices: Tuple[int, ...]
    output_tensor_indices: Tuple[int, ...]

    def to_dict(self):
        return {
            "name": self.name,
            "operator_indices": list(self.operator_indices),
            "input_tensor_indices": list(self.input_tensor_indices),
            "output_tensor_indices": list(self.output_tensor_indices),
        }


def _graph_index(graph):
    producers = {}
    consumers = {tensor.index: [] for tensor in graph.tensors}
    operators = {operator.index: operator for operator in graph.operators}
    for operator in graph.operators:
        for tensor_index in operator.outputs:
            if tensor_index in producers:
                raise ToolError(
                    "TENSOR_HAS_MULTIPLE_PRODUCERS",
                    "tensor {} has multiple graph producers".format(tensor_index),
                )
            producers[tensor_index] = operator.index
        for tensor_index in operator.inputs:
            if tensor_index >= 0:
                consumers.setdefault(tensor_index, []).append(operator.index)
    return operators, producers, consumers


def _ancestor_operators(tensor_index, operators, producers):
    result = set()
    pending = [tensor_index]
    visited_tensors = set()
    while pending:
        current = pending.pop()
        if current in visited_tensors:
            continue
        visited_tensors.add(current)
        producer_index = producers.get(current)
        if producer_index is None or producer_index in result:
            continue
        result.add(producer_index)
        pending.extend(index for index in operators[producer_index].inputs if index >= 0)
    return result


def _layer_boundary_tensors(layer, graph, operators, producers, graph_inputs, tensors):
    """Resolve this block's hidden input and post-MLP residual from graph edges."""
    q_projection = layer.roles.get("q")
    down_projection = layer.roles.get("down")
    if q_projection is None or down_projection is None:
        raise ToolError("LAYER_BOUNDARY_UNRESOLVED", "Q and Down roles are required")

    pending = [q_projection.activation_tensor_index]
    visited_tensors = set()
    norm_candidates = []
    while pending:
        tensor_index = pending.pop()
        if tensor_index in visited_tensors:
            continue
        visited_tensors.add(tensor_index)
        producer_index = producers.get(tensor_index)
        producer = operators.get(producer_index)
        if producer is None:
            continue
        if "NORM" in producer.opcode.upper():
            norm_candidates.append(producer)
            continue
        pending.extend(index for index in producer.inputs if index >= 0)
    if len(norm_candidates) != 1:
        raise ToolError(
            "LAYER_INPUT_NORM_UNRESOLVED",
            "Q activation reaches {} nearest normalization operators".format(len(norm_candidates)),
        )
    norm_operator = norm_candidates[0]
    activation_shape = tensors[q_projection.activation_tensor_index].shape
    input_candidates = [
        tensor_index
        for tensor_index in norm_operator.inputs
        if tensor_index >= 0
        and tensor_index in tensors
        and tensors[tensor_index].shape == activation_shape
        and (tensor_index in graph_inputs or producers.get(tensor_index) is not None)
    ]
    if len(input_candidates) != 1:
        raise ToolError(
            "LAYER_INPUT_HIDDEN_AMBIGUOUS",
            "normalization input candidates are {}".format(input_candidates),
        )
    hidden_input = input_candidates[0]

    down_outputs = set(down_projection.output_tensor_indices)
    residual_outputs = [
        output_index
        for operator in graph.operators
        if "ADD" in operator.opcode.upper()
        and down_outputs.intersection(index for index in operator.inputs if index >= 0)
        for output_index in operator.outputs
    ]
    if len(residual_outputs) != 1 or residual_outputs[0] not in tensors:
        raise ToolError(
            "LAYER_OUTPUT_RESIDUAL_UNRESOLVED",
            "Down output does not lead to one unique residual tensor",
        )
    return hidden_input, residual_outputs[0]


def _layer_kv_outputs(layer, graph, graph_analysis):
    """Select only the current block's public KV outputs in a grouped graph."""
    semantic_ids = graph_analysis.evidence.get("semantic_layer_ids", [])
    key_outputs = list(graph_analysis.output_roles.get("current_key", ()))
    value_outputs = list(graph_analysis.output_roles.get("current_value", ()))
    if len(semantic_ids) <= 1:
        return key_outputs, value_outputs

    semantic_id = layer.roles["q"].semantic_layer_id
    if semantic_id is None:
        raise ToolError("GROUPED_KV_LAYER_SCOPE_UNRESOLVED", str(layer.layer_id))

    tensor_by_index = {tensor.index: tensor for tensor in graph.tensors}
    scope_pattern = re.compile(
        r"(?:^|/)(?:layers?|blocks?)/{}/(?:|.*)".format(int(semantic_id)),
        re.IGNORECASE,
    )

    def belongs(index):
        tensor = tensor_by_index.get(index)
        if tensor is None:
            return False
        name = tensor.name.lower().replace("\\", "/")
        if scope_pattern.search(name):
            return True
        # Some exporters put the semantic layer number on the current-cache
        # tensor suffix instead of a layers/<id>/ path.
        if any(token in name for token in ("curr_key", "current_key", "curr_value", "current_value")):
            return bool(re.search(r"(?:_|/){:d}$".format(int(semantic_id)), name))
        return False

    scoped_keys = [index for index in key_outputs if belongs(index)]
    scoped_values = [index for index in value_outputs if belongs(index)]
    if len(scoped_keys) != 1 or len(scoped_values) != 1:
        raise ToolError(
            "GROUPED_KV_LAYER_SCOPE_AMBIGUOUS",
            "layer {} key outputs={} value outputs={}".format(layer.layer_id, scoped_keys, scoped_values),
        )
    return scoped_keys, scoped_values


def _component_ports(name, operator_indices, operators, producers, consumers, graph_inputs, graph_outputs, tensors, order):
    operator_indices = set(operator_indices)
    if not operator_indices:
        raise ToolError("COMPONENT_OPERATOR_SET_EMPTY", "{} has no operators".format(name))
    inputs = []
    outputs = []
    for operator_index in order:
        if operator_index not in operator_indices:
            continue
        operator = operators[operator_index]
        for tensor_index in operator.inputs:
            if tensor_index < 0:
                continue
            producer_index = producers.get(tensor_index)
            if producer_index in operator_indices:
                continue
            tensor = tensors[tensor_index]
            if tensor_index in graph_inputs or producer_index is not None:
                if tensor_index not in inputs:
                    inputs.append(tensor_index)
            elif tensor.buffer_bytes <= 0 and not tensor.is_variable:
                raise ToolError(
                    "COMPONENT_INPUT_UNRESOLVED",
                    "{} input tensor {} has no producer, source input, or constant".format(name, tensor_index),
                )
        for tensor_index in operator.outputs:
            external_users = any(user not in operator_indices for user in consumers.get(tensor_index, ()))
            if (external_users or tensor_index in graph_outputs) and tensor_index not in outputs:
                outputs.append(tensor_index)
    if not outputs:
        raise ToolError("COMPONENT_OUTPUT_UNRESOLVED", "{} has no external output boundary".format(name))
    return ComponentSpec(
        name=name,
        operator_indices=tuple(index for index in order if index in operator_indices),
        input_tensor_indices=tuple(inputs),
        output_tensor_indices=tuple(outputs),
    )


def derive_tp_components(model: ModelInfo, layer: LayerDiscovery, graph_analysis, degree=4) -> List[ComponentSpec]:
    """Derive six stage components for TP2 or TP4 from tensor dependencies.

    The operator storage order is ignored. Grouped exports are handled by
    isolating the selected block between its normalization input and its
    post-Down residual using producer/consumer dependencies.
    """
    if not layer.complete:
        raise ToolError("LAYER_DISCOVERY_INCOMPLETE", "layer {} is incomplete".format(layer.layer_id))
    if layer.profile not in ("P128", "D1"):
        raise ToolError("PROFILE_UNSUPPORTED", layer.profile)
    if graph_analysis.layer_id != layer.layer_id:
        raise ToolError("GRAPH_ANALYSIS_LAYER_MISMATCH", "analysis and layer IDs differ")
    if not all(graph_analysis.features.get(name, False) for name in (
        "rmsnorm", "rotary", "attention", "swiglu", "residual", "mask", "position", "kv_cache", "layer_boundaries"
    )):
        raise ToolError("UNSUPPORTED_GRAPH_VARIANT", "required transformer graph features were not all proven")
    if degree not in (2, 4):
        raise ToolError("TP_DEGREE_UNSUPPORTED", "supported tensor-parallel degrees are 2 and 4")
    make_partition_plan(layer, degree=degree)
    if layer.graph_index < 0 or layer.graph_index >= len(model.graphs):
        raise ToolError("GRAPH_INDEX_OUT_OF_RANGE", str(layer.graph_index))
    graph = model.graphs[layer.graph_index]
    operators, producers, consumers = _graph_index(graph)
    tensors = {tensor.index: tensor for tensor in graph.tensors}
    layer_key_outputs, layer_value_outputs = _layer_kv_outputs(layer, graph, graph_analysis)
    layer_input_tensor, layer_output_tensor = _layer_boundary_tensors(
        layer,
        graph,
        operators,
        producers,
        set(graph.inputs),
        tensors,
    )
    input_ancestors = _ancestor_operators(layer_input_tensor, operators, producers)
    layer_operator_indices = _ancestor_operators(layer_output_tensor, operators, producers)
    for tensor_index in layer_key_outputs + layer_value_outputs:
        layer_operator_indices.update(_ancestor_operators(tensor_index, operators, producers))
    layer_operator_indices -= input_ancestors

    o_projection = layer.roles["o"]
    gate_projection = layer.roles["gate"]
    down_projection = layer.roles["down"]
    attention_ids = _ancestor_operators(
        o_projection.activation_tensor_index, operators, producers
    ) - input_ancestors
    for tensor_index in layer_key_outputs + layer_value_outputs:
        attention_ids.update(_ancestor_operators(tensor_index, operators, producers) - input_ancestors)
    o_ids = {o_projection.operator_index}
    o_output = o_projection.output_tensor_indices[0]
    pre_mlp_ids = _ancestor_operators(
        gate_projection.activation_tensor_index, operators, producers
    ) - input_ancestors
    o_path_ids = _ancestor_operators(o_output, operators, producers) - input_ancestors
    suffix_attn_ids = pre_mlp_ids - o_path_ids
    down_input_ids = _ancestor_operators(
        down_projection.activation_tensor_index, operators, producers
    ) - input_ancestors
    mlp_ids = down_input_ids - pre_mlp_ids
    down_ids = {down_projection.operator_index}
    down_output = down_projection.output_tensor_indices[0]
    final_ids = _ancestor_operators(layer_output_tensor, operators, producers) - _ancestor_operators(
        down_output, operators, producers
    )

    stage_sets = [attention_ids, o_ids, suffix_attn_ids, mlp_ids, down_ids, final_ids]
    stage_names = ("attn", "o", "suffix_attn", "mlp", "down", "suffix_final")
    owner = {}
    for stage_name, stage_set in zip(stage_names, stage_sets):
        if not stage_set:
            raise ToolError("COMPONENT_OPERATOR_SET_EMPTY", "{} has no operators".format(stage_name))
        for operator_index in stage_set:
            if operator_index in owner:
                raise ToolError(
                    "COMPONENT_OPERATOR_OVERLAP",
                    "operator {} belongs to {} and {}".format(operator_index, owner[operator_index], stage_name),
                )
            owner[operator_index] = stage_name
    unassigned_layer_operators = sorted(layer_operator_indices - set(owner))
    operators_outside_layer = sorted(set(owner) - layer_operator_indices)
    if unassigned_layer_operators or operators_outside_layer:
        raise ToolError(
            "LAYER_OPERATOR_COVERAGE_INVALID",
            "unassigned layer operators {}; operators outside layer {}".format(
                unassigned_layer_operators,
                operators_outside_layer,
            ),
        )

    topological_order = graph_analysis.topological_order
    if set(topological_order) != set(operators):
        raise ToolError("GRAPH_TOPOLOGICAL_ORDER_INVALID", "not all graph operators have a topological position")
    graph_inputs = set(graph.inputs)
    graph_outputs = set(graph.outputs)
    return [
        _component_ports(
            stage_name,
            stage_set,
            operators,
            producers,
            consumers,
            graph_inputs,
            graph_outputs,
            tensors,
            topological_order,
        )
        for stage_name, stage_set in zip(stage_names, stage_sets)
    ]


def derive_tp4_components(model: ModelInfo, layer: LayerDiscovery, graph_analysis) -> List[ComponentSpec]:
    """Backward-compatible name for the TP4 component discovery path."""
    return derive_tp_components(model, layer, graph_analysis, degree=4)
