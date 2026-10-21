"""Automatic graph, layer, profile, and projection discovery."""

from collections import defaultdict
from pathlib import Path
import re

from .errors import ModelValidationError, ToolError
from .model import (
    EXPECTED_WEIGHT_SHAPES,
    PROJECTION_ROLES,
    DiscoveryReport,
    LayerDiscovery,
    ProjectionMatch,
    SourceDiscovery,
    validate_graph,
)
from .analysis import analyze_graph
from .tflite_reader import read_tflite


ROLE_ALIASES = {
    "q": {"q_proj", "query", "wq"},
    "k": {"k_proj", "key", "wk"},
    "v": {"v_proj", "value", "wv"},
    "o": {"o_proj", "out_proj", "wo"},
    "gate": {"gate_proj", "w1"},
    "up": {"up_proj", "w3"},
    "down": {"down_proj", "w2"},
}

_SEMANTIC_LAYER = re.compile(r"(?:^|/)(?:layers?/|blocks?/|layer|block)(\d+)(?:/|$)", re.IGNORECASE)
_SOURCE_LAYER = re.compile(r"(?:^|[^a-z0-9])layer[_-]?(\d{1,2})(?:[^0-9]|$)", re.IGNORECASE)
_SOURCE_INDEXED_LAYER = re.compile(r"(?:^|[_-])1layer_1t1024c_(\d{1,2})(?:\.tflite)?$", re.IGNORECASE)


def source_layer_id(source_path):
    """Get an explicit layer index from a layerNN source filename or directory."""
    for component in reversed(Path(source_path).parts):
        indexed = _SOURCE_INDEXED_LAYER.search(component)
        if indexed:
            return int(indexed.group(1))
        match = _SOURCE_LAYER.search(component)
        if match:
            return int(match.group(1))
    return None


def semantic_layer_id(tensor_name):
    normalized = tensor_name.lower().replace(".", "/").replace("\\", "/")
    match = _SEMANTIC_LAYER.search(normalized)
    return int(match.group(1)) if match else None


def role_from_tensor_name(tensor_name):
    normalized = tensor_name.lower().replace(".", "/").replace("\\", "/")
    tokens = [token for token in normalized.split("/") if token]
    if not tokens or tokens[-1] not in {"weight", "kernel", "filter"}:
        return None
    matches = [role for role, aliases in ROLE_ALIASES.items() if any(token in aliases for token in tokens[:-1])]
    return matches[0] if len(matches) == 1 else None


def classify_profile(graph):
    token_counts = set()
    for tensor_index in graph.inputs:
        if tensor_index < 0 or tensor_index >= len(graph.tensors):
            continue
        tensor = graph.tensors[tensor_index]
        for shape in (tensor.shape, tensor.shape_signature):
            if len(shape) == 3 and shape[0] in (1, -1) and shape[-1] == 4096:
                if shape[1] in (128, 1):
                    token_counts.add(shape[1])
                    break
    if token_counts == {128}:
        return "P128"
    if token_counts == {1}:
        return "D1"
    if token_counts:
        return "MIXED_OR_UNSUPPORTED"
    return "UNKNOWN"


def _role_operators(graph):
    by_weight = defaultdict(list)
    for operator in graph.operators:
        if len(operator.inputs) > 1 and operator.inputs[1] >= 0:
            by_weight[operator.inputs[1]].append(operator)
    return by_weight


def _weight_matches(graph):
    by_weight = _role_operators(graph)
    matches = []
    issues = []
    for tensor in graph.tensors:
        role = role_from_tensor_name(tensor.name)
        if role is None:
            continue
        operators = by_weight.get(tensor.index, [])
        if not operators:
            issues.append(("ROLE_WEIGHT_UNBOUND", "named {} weight tensor {} is not a linear filter input".format(role, tensor.index)))
            continue
        layer_id = semantic_layer_id(tensor.name)
        for operator in operators:
            matches.append((layer_id, role, tensor, operator))
    return matches, issues


def _is_fully_connected_operator(opcode):
    """Recognize builtin and vendor-prefixed fully connected op names."""
    normalized = opcode.upper()
    return normalized == "FULLY_CONNECTED" or normalized.endswith("_FULLY_CONNECTED")


def _quantization_layout(tensor):
    quant = tensor.quantization
    if quant is None or not quant.scales:
        return "MISSING"
    if len(quant.scales) == 1:
        return "PER_TENSOR"
    axis = quant.quantized_dimension
    if axis < 0 or axis >= len(tensor.shape):
        return "AXIS_OUT_OF_RANGE"
    extent = tensor.shape[axis]
    if len(quant.scales) == extent:
        return "PER_CHANNEL"
    if extent > 0 and extent % len(quant.scales) == 0:
        return "UNIFORM_GROUPS_ON_QUANT_AXIS"
    return "UNRESOLVED"


def _validate_projection(layer, graph, role, tensor, operator, semantic_id):
    expected = EXPECTED_WEIGHT_SHAPES[role]
    if not _is_fully_connected_operator(operator.opcode):
        layer.issues.append(
            ("PROJECTION_OPERATOR_UNSUPPORTED", "{} uses {} at operator {}".format(role, operator.opcode, operator.index))
        )
        return
    if tensor.shape != expected:
        layer.issues.append(
            ("PROJECTION_WEIGHT_SHAPE_MISMATCH", "{} weight is {}; expected {}".format(role, tensor.shape, expected))
        )
    if tensor.dtype != "INT4":
        layer.issues.append(
            ("PROJECTION_WEIGHT_NOT_INT4", "{} weight tensor {} is {}; W4A16 requires INT4".format(role, tensor.index, tensor.dtype))
        )
    if tensor.buffer_bytes <= 0:
        layer.issues.append(("PROJECTION_WEIGHT_NOT_CONSTANT", "{} weight tensor {} has no constant payload".format(role, tensor.index)))
    if tensor.quantization is None or not tensor.quantization.scales:
        layer.issues.append(("PROJECTION_QUANTIZATION_MISSING", "{} weight tensor {} has no scales".format(role, tensor.index)))
    if len(operator.inputs) < 2 or operator.inputs[0] < 0:
        layer.issues.append(("PROJECTION_ACTIVATION_MISSING", "{} operator {} has no activation input".format(role, operator.index)))
        return
    if len(operator.outputs) != 1:
        layer.issues.append(("PROJECTION_OUTPUT_ARITY_UNSUPPORTED", "{} operator {} has {} outputs".format(role, operator.index, len(operator.outputs))))
        return

    activation = graph.tensors[operator.inputs[0]]
    output = graph.tensors[operator.outputs[0]]
    if activation.shape and activation.shape[-1] not in (-1, expected[1]):
        layer.issues.append(
            ("PROJECTION_ACTIVATION_SHAPE_MISMATCH", "{} activation ends in {}; expected {}".format(role, activation.shape, expected[1]))
        )
    if output.shape and output.shape[-1] not in (-1, expected[0]):
        layer.issues.append(
            ("PROJECTION_OUTPUT_SHAPE_MISMATCH", "{} output ends in {}; expected {}".format(role, output.shape, expected[0]))
        )

    bias_index = None
    bias_shape = ()
    bias_dtype = None
    bias_buffer_index = None
    bias_quantization = None
    if len(operator.inputs) > 2 and operator.inputs[2] >= 0:
        bias_index = operator.inputs[2]
        bias_tensor = graph.tensors[bias_index]
        bias_shape = bias_tensor.shape
        bias_dtype = bias_tensor.dtype
        bias_buffer_index = bias_tensor.buffer_index
        bias_quantization = bias_tensor.quantization
        if bias_shape and bias_shape not in ((expected[0],), (1, expected[0])):
            layer.issues.append(
                ("PROJECTION_BIAS_SHAPE_MISMATCH", "{} bias shape is {}".format(role, bias_shape))
            )

    layer.roles[role] = ProjectionMatch(
        role=role,
        tensor_index=tensor.index,
        tensor_name=tensor.name,
        shape=tensor.shape,
        dtype=tensor.dtype,
        operator_index=operator.index,
        operator=operator.opcode,
        activation_tensor_index=operator.inputs[0],
        output_tensor_indices=operator.outputs,
        bias_tensor_index=bias_index,
        quantization=tensor.quantization,
        semantic_layer_id=semantic_id,
        weight_buffer_index=tensor.buffer_index,
        bias_shape=bias_shape,
        bias_dtype=bias_dtype,
        bias_buffer_index=bias_buffer_index,
        bias_quantization=bias_quantization,
        weight_quantization_layout=_quantization_layout(tensor),
        operator_input_tensor_indices=operator.inputs,
        output_shapes=tuple(graph.tensors[index].shape for index in operator.outputs),
    )


def _discover_graph_layers(model, graph, source_name, profile):
    issues = []
    layers = []
    analyses = []
    try:
        validate_graph(model, graph)
    except ModelValidationError as exc:
        issues.extend(exc.issues)

    matches, match_issues = _weight_matches(graph)
    issues.extend(match_issues)
    explicit_id = source_layer_id(source_name)
    semantic_ids = {layer_id for layer_id, _, _, _ in matches if layer_id is not None}

    if semantic_ids:
        # Per-layer exporter files often retain a local "layers/0" tensor scope;
        # an explicit layerNN source filename supplies the original layer index.
        if len(semantic_ids) == 1 and explicit_id is not None:
            layer_mapping = {next(iter(semantic_ids)): explicit_id}
        else:
            layer_mapping = {layer_id: layer_id for layer_id in semantic_ids}
        if any(layer_id is None for layer_id, _, _, _ in matches):
            issues.append(("LAYER_SCOPE_INCONSISTENT", "some projection weights lack a layer name in a multi-layer graph"))
    elif explicit_id is not None:
        layer_mapping = {None: explicit_id}
    else:
        if matches:
            issues.append(("LAYER_ID_UNRESOLVED", "projection names have no layer index and source name has no layerNN index"))
        layer_mapping = {}

    grouped = defaultdict(lambda: defaultdict(list))
    for semantic_id, role, tensor, operator in matches:
        if semantic_id not in layer_mapping:
            continue
        layer_id = layer_mapping[semantic_id]
        grouped[layer_id][role].append((tensor, operator, semantic_id))

    for layer_id in sorted(grouped):
        layer = LayerDiscovery(
            layer_id=layer_id,
            graph_index=graph.index,
            profile=profile,
            source_name=source_name,
            source_sha256=model.source_sha256,
        )
        for code, message in issues:
            layer.issues.append((code, message))
        role_groups = grouped[layer_id]
        for role in PROJECTION_ROLES:
            candidates = role_groups.get(role, [])
            if not candidates:
                layer.issues.append(("PROJECTION_ROLE_MISSING", "layer {} has no {} projection".format(layer_id, role)))
                continue
            if len(candidates) != 1:
                indexes = [candidate[0].index for candidate in candidates]
                layer.issues.append(("PROJECTION_ROLE_AMBIGUOUS", "layer {} has {} candidates {}".format(layer_id, role, indexes)))
                continue
            tensor, operator, semantic_id = candidates[0]
            _validate_projection(layer, graph, role, tensor, operator, semantic_id)
        analysis = analyze_graph(model, graph, layer.roles, layer_id=layer_id)
        layer.graph_features = analysis.features
        analyses.append(analysis)
        issues.extend(analysis.issues)
        layers.append(layer)
    return layers, issues, analyses


def discover_models(entries, expected_layers=32):
    """Analyze already parsed models; useful for offline synthetic graph tests."""
    report = DiscoveryReport(expected_layers=expected_layers)
    for model, source_name in entries:
        source = SourceDiscovery(
            source_name=Path(source_name).as_posix(),
            source_sha256=model.source_sha256,
            source_bytes=model.source_bytes,
            version=model.version,
            graph_count=len(model.graphs),
            model_info=model,
        )
        for graph in model.graphs:
            profile = classify_profile(graph)
            if profile not in ("P128", "D1"):
                source.issues.append(("PROFILE_UNRESOLVED", "graph {} profile is {}".format(graph.index, profile)))
            if profile not in source.profiles:
                source.profiles.append(profile)
            graph_layers, graph_issues, graph_analyses = _discover_graph_layers(model, graph, source_name, profile)
            source.layers.extend(graph_layers)
            source.issues.extend(graph_issues)
            source.graph_analyses.extend(graph_analyses)
        report.sources.append(source)
        report.issues.extend(source.issues)
        for layer in source.layers:
            report.issues.extend(layer.issues)

    by_profile = defaultdict(list)
    for source in report.sources:
        for layer in source.layers:
            if layer.profile in ("P128", "D1"):
                by_profile[layer.profile].append(layer.layer_id)

    profiles = sorted(set(by_profile) | {profile for source in report.sources for profile in source.profiles if profile in ("P128", "D1")})
    for profile in profiles:
        layer_ids = by_profile.get(profile, [])
        duplicates = sorted(index for index in set(layer_ids) if layer_ids.count(index) > 1)
        expected = list(range(expected_layers))
        missing = sorted(set(expected) - set(layer_ids))
        unexpected = sorted(set(layer_ids) - set(expected))
        coverage = {
            "present_layers": sorted(set(layer_ids)),
            "missing_layers": missing,
            "duplicate_layers": duplicates,
            "unexpected_layers": unexpected,
            "covered_count": len(set(layer_ids) & set(expected)),
            "expected_count": expected_layers,
            "complete": not missing and not duplicates and not unexpected,
        }
        report.coverage[profile] = coverage
        if missing:
            report.issues.append(("LAYER_COVERAGE_INCOMPLETE", "{} is missing layers {}".format(profile, missing)))
        if duplicates:
            report.issues.append(("LAYER_COVERAGE_DUPLICATE", "{} has duplicate layers {}".format(profile, duplicates)))
        if unexpected:
            report.issues.append(("LAYER_COVERAGE_UNEXPECTED", "{} has out-of-range layers {}".format(profile, unexpected)))

    if not report.sources:
        report.issues.append(("NO_TFLITE_SOURCES", "no TFLite models were analyzed"))
    return report


def discover_path(source_path, expected_layers=32, schema_module=None):
    """Scan a TFLite file or directory and produce a source/layer audit."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise ToolError("SOURCE_NOT_FOUND", "source path does not exist: {}".format(source_path))
    if source_path.is_file():
        paths = [source_path] if source_path.suffix.lower() == ".tflite" else []
        root = source_path.parent
    else:
        root = source_path
        paths = sorted(path for path in source_path.rglob("*") if path.is_file() and path.suffix.lower() == ".tflite")
    if not paths:
        report = DiscoveryReport(expected_layers=expected_layers)
        report.issues.append(("NO_TFLITE_SOURCES", "no .tflite files found under {}".format(source_path.name)))
        return report

    entries = []
    failed_sources = []
    for path in paths:
        try:
            model = read_tflite(path, schema_module=schema_module)
            relative = path.name if source_path.is_file() else path.relative_to(root).as_posix()
            entries.append((model, relative))
        except ToolError as exc:
            failed_sources.append((path.name, exc.code, exc.message))
    report = discover_models(entries, expected_layers=expected_layers)
    for source_name, code, message in failed_sources:
        report.issues.append((code, "{}: {}".format(source_name, message)))
        report.sources.append(
            SourceDiscovery(
                source_name=source_name,
                source_sha256="",
                source_bytes=0,
                version=0,
                graph_count=0,
                issues=[(code, message)],
            )
        )
    return report
