"""Inspect Llama geometry without changing a source graph."""

import re

from .discovery import _head_geometry, _weight_matches
from .errors import ToolError
from .model import PROJECTION_ROLES
from .tflite_reader import read_tflite


def inspect_llama_source(path, tokens):
    if tokens <= 0:
        raise ToolError("TOKEN_COUNT_INVALID", "token count must be positive")
    model = read_tflite(path)
    if len(model.graphs) != 1:
        raise ToolError("GRAPH_COUNT_UNSUPPORTED", "expected a single-subgraph source")
    graph = model.graphs[0]
    matches, issues = _weight_matches(graph)
    roles = {}
    for _, role, tensor, _ in matches:
        roles.setdefault(role, set()).add(tensor.shape)
    if issues or set(roles) != set(PROJECTION_ROLES) or any(len(v) != 1 for v in roles.values()):
        raise ToolError("PROJECTION_ROLES_UNRESOLVED", "seven uniquely shaped named projections are required")
    shapes = {role: next(iter(value)) for role, value in roles.items()}
    if any(len(shape) != 2 or min(shape) <= 0 for shape in shapes.values()):
        raise ToolError("PROJECTION_SHAPE_INVALID", "projection weights must be positive rank-two matrices")
    hidden = shapes["q"][1]
    qwidth, kvwidth, intermediate = shapes["q"][0], shapes["k"][0], shapes["gate"][0]
    expected = {"q": (qwidth, hidden), "k": (kvwidth, hidden), "v": (kvwidth, hidden),
                "o": (hidden, qwidth), "gate": (intermediate, hidden),
                "up": (intermediate, hidden), "down": (hidden, intermediate)}
    if shapes != expected:
        raise ToolError("PROJECTION_GEOMETRY_INVALID", "projection shapes do not form a consistent Llama block")
    inputs = [graph.tensors[i] for i in graph.inputs]
    hidden_inputs = [t for t in inputs if len(t.shape) == 3 and t.shape[0] == 1 and t.shape[-1] == hidden]
    if len(hidden_inputs) != 1 or hidden_inputs[0].shape[1] != tokens:
        raise ToolError("TOKEN_COUNT_MISMATCH", "requested {} tokens; hidden input shapes are {}".format(tokens, [t.shape for t in hidden_inputs]))
    caches = [t for t in inputs if len(t.shape) == 4 and "past" in t.name.lower()
              and ("key" in t.name.lower() or "value" in t.name.lower())]
    keys = [t for t in caches if "key" in t.name.lower()]
    values = [t for t in caches if "value" in t.name.lower()]
    key_layers = [re.sub(r"keys?|values?", "", t.name.lower()) for t in keys]
    value_layers = [re.sub(r"keys?|values?", "", t.name.lower()) for t in values]
    if (not keys or len(keys) != len(values) or len(set(key_layers)) != len(keys)
            or set(key_layers) != set(value_layers)):
        raise ToolError("KV_HEADS_UNRESOLVED", "cannot uniquely identify compatible past key and value heads")
    geometry = _head_geometry(graph, shapes)
    if geometry is None:
        raise ToolError("KV_HEADS_UNRESOLVED", "source graph has no past key/value head geometry")
    q_heads, kv_heads, head_dim = geometry
    degrees = [degree for degree in (2, 4, 8)
               if q_heads % degree == 0 and kv_heads % degree == 0
               and all(dimension % degree == 0 for dimension in (hidden, qwidth, kvwidth, intermediate))]
    return {"source_sha256": model.source_sha256, "tokens": tokens, "hidden_size": hidden,
            "intermediate_size": intermediate, "query_heads": q_heads, "kv_heads": kv_heads,
            "head_dim": head_dim, "eligible_tp_degrees": degrees,
            "status": "GEOMETRY_ONLY", "partition_or_numerical_validation": "NOT_RUN"}
