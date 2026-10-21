"""Dependency and Llama graph-feature analysis independent of operator order."""

import heapq
import re
from collections import defaultdict, deque

from .model import GraphAnalysis, validate_graph


def _port_role(name, direction):
    name = name.lower().replace(".", "/").replace("\\", "/")
    if "mask" in name:
        return "mask"
    if "pos_emb" in name or "position_embedding" in name or "position_ids" in name:
        return "position"
    if "past_key" in name or "past_keys" in name:
        return "past_key"
    if "past_value" in name or "past_values" in name:
        return "past_value"
    if "curr_key" in name or "current_key" in name:
        return "current_key"
    if "curr_value" in name or "current_value" in name:
        return "current_value"
    if "input_embeds" in name or "hidden_states" in name or "hidden_state" in name:
        return "hidden_state"
    if direction == "output" and "logits" in name:
        return "logits"
    return None


def _semantic_layer_ids(graph):
    result = set()
    pattern = re.compile(r"(?:^|/)(?:layers?|blocks?)/(\d+)(?:/|$)", re.IGNORECASE)
    for tensor in graph.tensors:
        normalized = tensor.name.lower().replace(".", "/").replace("\\", "/")
        match = pattern.search(normalized)
        if match:
            result.add(int(match.group(1)))
    return sorted(result)


def _operator_semantics(opcode):
    name = opcode.upper()
    return {
        "normalization": "NORM" in name,
        "rmsnorm": "RMS" in name and "NORM" in name,
        "batch_matmul": "BATCH_MATMUL" in name or "BATCHMATMUL" in name,
        "softmax": "SOFTMAX" in name,
        "silu": "SILU" in name or "SWISH" in name,
        "multiply": name == "MUL" or name.endswith("_MUL"),
        "add": name == "ADD" or name.endswith("_ADD"),
    }


def analyze_graph(model, graph, projection_roles, layer_id=None):
    """Build producer/consumer maps, stable topological order and semantic evidence.

    The analyzer records unresolved semantics instead of selecting an operator
    based on its storage index. Tensor names, role discovery, shapes and graph
    reachability provide the evidence for each feature.
    """
    issues = []
    try:
        validate_graph(model, graph)
    except Exception as exc:
        if hasattr(exc, "issues"):
            issues.extend(exc.issues)
        else:
            raise

    operators = {operator.index: operator for operator in graph.operators}
    producers = {}
    consumers = defaultdict(list)
    for operator in graph.operators:
        for tensor_index in operator.outputs:
            producers[tensor_index] = operator.index
        for tensor_index in operator.inputs:
            if tensor_index >= 0:
                consumers[tensor_index].append(operator.index)

    dependencies = {operator.index: set() for operator in graph.operators}
    users = defaultdict(set)
    for operator in graph.operators:
        for tensor_index in operator.inputs:
            producer = producers.get(tensor_index)
            if producer is not None:
                dependencies[operator.index].add(producer)
                users[producer].add(operator.index)

    remaining = {index: set(deps) for index, deps in dependencies.items()}
    ready = [index for index, deps in remaining.items() if not deps]
    heapq.heapify(ready)
    topological_order = []
    while ready:
        index = heapq.heappop(ready)
        topological_order.append(index)
        for user in sorted(users.get(index, ())):
            remaining[user].discard(index)
            if not remaining[user]:
                heapq.heappush(ready, user)

    def reaches(source_tensor, destination_tensor):
        if source_tensor == destination_tensor:
            return True
        queue = deque([source_tensor])
        visited = {source_tensor}
        while queue:
            tensor_index = queue.popleft()
            for operator_index in consumers.get(tensor_index, ()):
                for output_index in operators[operator_index].outputs:
                    if output_index == destination_tensor:
                        return True
                    if output_index not in visited:
                        visited.add(output_index)
                        queue.append(output_index)
        return False

    input_roles = defaultdict(list)
    output_roles = defaultdict(list)
    for tensor_index in graph.inputs:
        role = _port_role(graph.tensors[tensor_index].name, "input")
        if role:
            input_roles[role].append(tensor_index)
    for tensor_index in graph.outputs:
        role = _port_role(graph.tensors[tensor_index].name, "output")
        if role:
            output_roles[role].append(tensor_index)

    semantic_ops = {index: _operator_semantics(op.opcode) for index, op in operators.items()}
    norms = sorted(index for index, info in semantic_ops.items() if info["normalization"])
    rmsnorms = sorted(index for index, info in semantic_ops.items() if info["rmsnorm"])
    batch_matmuls = sorted(index for index, info in semantic_ops.items() if info["batch_matmul"])
    softmaxes = sorted(index for index, info in semantic_ops.items() if info["softmax"])
    silus = sorted(index for index, info in semantic_ops.items() if info["silu"])
    multiply_ops = sorted(index for index, info in semantic_ops.items() if info["multiply"])
    add_ops = sorted(index for index, info in semantic_ops.items() if info["add"])

    role_outputs = {
        role: tuple(match.output_tensor_indices)
        for role, match in projection_roles.items()
    }
    qkv_reach = {}
    for role in ("q", "k", "v"):
        qkv_reach[role] = any(
            reaches(source, tensor_index)
            for source in role_outputs.get(role, ())
            for op_index in batch_matmuls
            for tensor_index in operators[op_index].inputs
            if tensor_index >= 0
        )
    attention = False
    for softmax_index in softmaxes:
        softmax = operators[softmax_index]
        if len(softmax.inputs) != 1 or not softmax.outputs:
            continue
        q_before = any(reaches(source, softmax.inputs[0]) for source in role_outputs.get("q", ()))
        k_before = any(reaches(source, softmax.inputs[0]) for source in role_outputs.get("k", ()))
        if not (q_before and k_before):
            continue
        probabilities = softmax.outputs[0]
        for op_index in batch_matmuls:
            operator = operators[op_index]
            if len(operator.inputs) < 2:
                continue
            softmax_branch = any(reaches(probabilities, tensor_index) for tensor_index in operator.inputs if tensor_index >= 0)
            value_branch = any(reaches(source, tensor_index) for source in role_outputs.get("v", ()) for tensor_index in operator.inputs if tensor_index >= 0)
            if softmax_branch and value_branch:
                attention = True
                break
        if attention:
            break

    position_inputs = input_roles.get("position", ())
    position_reaches_attention = any(
        reaches(position, tensor_index)
        for position in position_inputs
        for op_index in batch_matmuls
        for tensor_index in operators[op_index].inputs
        if tensor_index >= 0
    )
    q_rotary_path = any(
        reaches(position, tensor_index) and any(reaches(source, tensor_index) for source in role_outputs.get("q", ()))
        for position in position_inputs
        for op_index in batch_matmuls
        for tensor_index in operators[op_index].inputs
        if tensor_index >= 0
    )
    k_rotary_path = any(
        reaches(position, tensor_index) and any(reaches(source, tensor_index) for source in role_outputs.get("k", ()))
        for position in position_inputs
        for op_index in batch_matmuls
        for tensor_index in operators[op_index].inputs
        if tensor_index >= 0
    )
    rotary = bool(position_inputs and q_rotary_path and k_rotary_path)

    mask_inputs = input_roles.get("mask", ())
    mask_reaches_softmax = any(
        reaches(mask, tensor_index)
        for mask in mask_inputs
        for op_index in softmaxes
        for tensor_index in operators[op_index].inputs
        if tensor_index >= 0
    )

    gate = projection_roles.get("gate")
    up = projection_roles.get("up")
    down = projection_roles.get("down")
    swiglu_mul = []
    gate_silu_operators = []
    if gate and up and down:
        for op_index in multiply_ops:
            operator = operators[op_index]
            if len(operator.inputs) < 2 or len(operator.outputs) != 1:
                continue
            branch_pairs = [
                (left, right)
                for left in range(len(operator.inputs))
                for right in range(len(operator.inputs))
                if left != right
                and any(reaches(source, operator.inputs[left]) for source in gate.output_tensor_indices)
                and any(reaches(source, operator.inputs[right]) for source in up.output_tensor_indices)
            ]
            matching_silu = [
                silu_index
                for silu_index in silus
                if any(
                    reaches(source, tensor_index)
                    for source in gate.output_tensor_indices
                    for tensor_index in operators[silu_index].inputs
                    if tensor_index >= 0
                )
                and any(
                    reaches(tensor_index, operator.inputs[left])
                    for tensor_index in operators[silu_index].outputs
                    for left, _ in branch_pairs
                )
            ]
            if branch_pairs and matching_silu and reaches(operator.outputs[0], down.activation_tensor_index):
                swiglu_mul.append(op_index)
                gate_silu_operators.extend(matching_silu)
    swiglu = bool(swiglu_mul and gate_silu_operators and gate and up and down)

    residual_ops = []
    for op_index in add_ops:
        operator = operators[op_index]
        if len(operator.outputs) != 1:
            continue
        if len(graph.tensors[operator.outputs[0]].shape) < 1 or graph.tensors[operator.outputs[0]].shape[-1] != 4096:
            continue
        projection_boundary_inputs = set()
        if "o" in projection_roles:
            projection_boundary_inputs.update(projection_roles["o"].output_tensor_indices)
        if "down" in projection_roles:
            projection_boundary_inputs.update(projection_roles["down"].output_tensor_indices)
        if any(tensor_index in projection_boundary_inputs for tensor_index in operator.inputs):
            residual_ops.append(op_index)

    kv_cache = all(input_roles.get(role) for role in ("past_key", "past_value")) and all(
        output_roles.get(role) for role in ("current_key", "current_value")
    )
    semantic_layers = _semantic_layer_ids(graph)
    features = {
        "rmsnorm": bool(rmsnorms),
        "rotary": rotary,
        "attention": attention,
        "swiglu": swiglu,
        "residual": len(residual_ops) >= 2,
        "mask": bool(mask_inputs and mask_reaches_softmax),
        "position": bool(position_inputs),
        "kv_cache": bool(kv_cache),
        "layer_boundaries": bool(semantic_layers or projection_roles),
        "grouped_export": len(semantic_layers) > 1,
    }
    required = ("rmsnorm", "rotary", "attention", "swiglu", "residual", "mask", "position", "kv_cache", "layer_boundaries")
    for feature in required:
        if not features[feature]:
            issues.append(("GRAPH_FEATURE_UNRESOLVED", "graph {} did not prove {} from names, shapes, and dependency paths".format(graph.index, feature)))

    evidence = {
        "semantic_layer_ids": semantic_layers,
        "normalization_operators": norms,
        "rmsnorm_operators": rmsnorms,
        "batch_matmul_operators": batch_matmuls,
        "softmax_operators": softmaxes,
        "silu_operators": silus,
        "gate_silu_operators": sorted(set(gate_silu_operators)),
        "swiglu_multiply_operators": swiglu_mul,
        "residual_add_operators": residual_ops,
        "qkv_reach_attention": qkv_reach,
        "position_reaches_attention": position_reaches_attention,
        "mask_reaches_softmax": mask_reaches_softmax,
    }
    return GraphAnalysis(
        graph_index=graph.index,
        layer_id=layer_id,
        tensor_producers=producers,
        tensor_consumers={index: tuple(sorted(values)) for index, values in consumers.items()},
        operator_dependencies={index: tuple(sorted(values)) for index, values in dependencies.items()},
        topological_order=tuple(topological_order),
        input_roles={role: tuple(sorted(values)) for role, values in input_roles.items()},
        output_roles={role: tuple(sorted(values)) for role, values in output_roles.items()},
        features=features,
        evidence=evidence,
        issues=issues,
    )
