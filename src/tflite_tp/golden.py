"""Host Golden backend probes and exact comparison primitives.

The optional MTK Converter runner executes a single TFLite graph. It is not,
by itself, proof of TP equivalence: callers must still stitch all ranks and
layers and compare their public outputs against the canonical graph.
"""

import hashlib
import importlib.util
import os
import re
import tempfile
from pathlib import Path

from .errors import ToolError


_HOST_PACKAGES = (
    "tflite_runtime.interpreter",
    "tensorflow.lite",
    "ai_edge_litert.interpreter",
    "mtk_converter.python.tools.tflite_executor",
)


def probe_golden_backend(model_info, custom_op_library=None):
    """Report whether a trustworthy interpreter integration is configured."""
    packages = {}
    for package in _HOST_PACKAGES:
        try:
            packages[package] = importlib.util.find_spec(package) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            packages[package] = False
    custom_ops = sorted({
        operator.opcode
        for graph in model_info.graphs
        for operator in graph.operators
        if operator.opcode.startswith("MTKEXT_")
    })
    configured_library = custom_op_library or os.environ.get("GENIO_TFLITE_CUSTOM_OP_LIBRARY")
    library_exists = bool(configured_library and Path(configured_library).is_file())
    interpreter_available = any(packages.values())
    mtk_executor_available = packages.get("mtk_converter.python.tools.tflite_executor", False)
    if mtk_executor_available:
        reason = (
            "MTK Converter TFLiteExecutor is available for individual graphs; "
            "a full TP rank/component stitch verifier is not registered."
        )
    elif not interpreter_available:
        reason = "No supported host TFLite interpreter package is installed."
    elif custom_ops and not library_exists:
        reason = "Source uses custom operators without a configured and existing custom-op library."
    elif custom_ops:
        reason = "A custom-op library path exists, but this tool has no verified resolver/ABI integration for it."
    else:
        reason = "An interpreter is present, but no canonical fixture runner has been registered."
    return {
        "status": "GOLDEN_EXECUTION_UNAVAILABLE",
        "code": "GOLDEN_EXECUTION_UNAVAILABLE",
        "reason": reason,
        "host_packages": packages,
        "interpreter_available": interpreter_available,
        "mtk_tflite_executor_available": mtk_executor_available,
        "candidate_backends": (
            ["mtk_converter.python.tools.tflite_executor"] if mtk_executor_available else []
        ),
        "custom_operators": custom_ops,
        "custom_op_library": str(configured_library) if configured_library else None,
        "custom_op_library_exists": library_exists,
        "verified_fixture_runner": False,
    }


class MTKConverterTFLiteRunner:
    """Execute one TFLite graph through MTK Converter's host interpreter.

    This adapter deliberately does not claim TP correctness. It returns named
    outputs so a topology-aware stitcher can compare canonical and rank graphs.
    ``simulate_fp16`` is explicit in the result because its value affects the
    host computation contract.
    """

    def __init__(self, custom_op_lib_map=None, simulate_fp16=False, executor_factory=None):
        if executor_factory is None:
            try:
                from mtk_converter.python.tools.tflite_executor import TFLiteExecutor
            except (ImportError, ModuleNotFoundError) as exc:
                raise ToolError(
                    "GOLDEN_EXECUTION_UNAVAILABLE",
                    "MTK Converter TFLiteExecutor is not importable: {}".format(exc),
                )
            executor_factory = TFLiteExecutor
        self._executor_factory = executor_factory
        self._custom_op_lib_map = custom_op_lib_map
        self._simulate_fp16 = bool(simulate_fp16)

    def run(self, model_path, inputs, output_names=None):
        """Run a model with name-keyed inputs and optionally expose internal tensors."""
        path = Path(model_path)
        if not path.is_file():
            raise ToolError("GOLDEN_MODEL_MISSING", str(path))
        if not isinstance(inputs, dict) or not inputs:
            raise ToolError("GOLDEN_FIXTURE_INVALID", "inputs must be a non-empty name-to-array map")
        requested_names = None
        if output_names is not None:
            requested_names = list(dict.fromkeys(output_names))
            if not requested_names or any(not isinstance(name, str) or not name for name in requested_names):
                raise ToolError("GOLDEN_OUTPUT_NAMES_INVALID", "output_names must be non-empty tensor names")
        temporary_path = None
        scalar_shape_compatibility = []
        try:
            try:
                executor = self._executor_factory(
                    str(path),
                    custom_op_lib_map=self._custom_op_lib_map,
                    simulate_fp16=self._simulate_fp16,
                )
            except Exception as initial_error:
                if "the tensor has no shape" not in str(initial_error).lower():
                    raise
                from .flatbuffer_adapter import TFLiteObjectAdapter, normalize_host_scalar_constant_shapes

                adapter = TFLiteObjectAdapter()
                model = adapter.load(path)
                for graph_index, graph in enumerate(model.subgraphs):
                    rows = normalize_host_scalar_constant_shapes(model, graph, adapter.schema)
                    scalar_shape_compatibility.extend(
                        {"subgraph": graph_index, **row} for row in rows
                    )
                if not scalar_shape_compatibility:
                    raise initial_error
                fd, temporary_path = tempfile.mkstemp(prefix="tflite-tp-host-shape-", suffix=".tflite")
                with os.fdopen(fd, "wb") as target:
                    target.write(adapter.export(model))
                executor = self._executor_factory(
                    temporary_path,
                    custom_op_lib_map=self._custom_op_lib_map,
                    simulate_fp16=self._simulate_fp16,
                )
            model_output_names = list(executor.model_outputs)
            result_names = requested_names if requested_names is not None else model_output_names
            if requested_names is None:
                output_values = executor.run(inputs)
            else:
                output_values = executor.run(inputs, output_names=requested_names)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError("GOLDEN_EXECUTION_FAILED", "{}: {}".format(path.name, exc))
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
        if len(result_names) != len(output_values):
            raise ToolError(
                "GOLDEN_OUTPUT_MAPPING_INVALID",
                "interpreter returned {} values for {} output names".format(
                    len(output_values), len(result_names)
                ),
            )
        outputs = dict(zip(result_names, output_values))
        source_hash = _hash_file(path)
        fixture_hash = _hash_fixture(inputs)
        return {
            "outputs": outputs,
            "model_output_names": model_output_names,
            "requested_output_names": result_names,
            "source_sha256": source_hash,
            "fixture_sha256": fixture_hash,
            "backend": "mtk_converter.python.tools.tflite_executor.TFLiteExecutor",
            "simulate_fp16": self._simulate_fp16,
            "host_scalar_shape_compatibility": scalar_shape_compatibility,
        }


def _hash_fixture(inputs):
    """Hash fixture names, dtypes, shapes, and bytes in stable name order."""
    digest = hashlib.sha256()
    try:
        import numpy as np
    except ImportError as exc:
        raise ToolError("GOLDEN_NUMPY_UNAVAILABLE", str(exc))
    for name in sorted(inputs):
        value = np.ascontiguousarray(inputs[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_named_outputs(expected, actual):
    """Compare named numeric TFLite outputs bit-for-bit and locate mismatches."""
    try:
        import numpy as np
    except ImportError as exc:
        raise ToolError("GOLDEN_NUMPY_UNAVAILABLE", str(exc))
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise ToolError("GOLDEN_OUTPUT_MAPPING_INVALID", "expected and actual must be mappings")
    mismatches = []
    names = sorted(set(expected) | set(actual))
    mismatch_count = 0
    max_absolute_difference = 0.0
    difference_available = True
    for name in names:
        if name not in expected or name not in actual:
            mismatch_count += 1
            difference_available = False
            mismatches.append({
                "tensor_name": name,
                "category": "OUTPUT_SET_MISMATCH",
                "expected_present": name in expected,
                "actual_present": name in actual,
            })
            continue
        left = np.asarray(expected[name])
        right = np.asarray(actual[name])
        if left.shape != right.shape:
            mismatch_count += 1
            difference_available = False
            mismatches.append({
                "tensor_name": name,
                "category": "OUTPUT_SHAPE_MISMATCH",
                "expected_shape": list(left.shape),
                "actual_shape": list(right.shape),
            })
            continue
        if left.dtype != right.dtype:
            mismatch_count += 1
            difference_available = False
            mismatches.append({
                "tensor_name": name,
                "category": "OUTPUT_DTYPE_MISMATCH",
                "expected_dtype": str(left.dtype),
                "actual_dtype": str(right.dtype),
            })
            continue
        if left.dtype.hasobject or not (
            np.issubdtype(left.dtype, np.number) or np.issubdtype(left.dtype, np.bool_)
        ):
            raise ToolError(
                "GOLDEN_OUTPUT_DTYPE_UNSUPPORTED",
                "{}: {}".format(name, left.dtype),
            )
        if left.size == 0:
            continue
        item_size = max(1, left.dtype.itemsize)
        left_bytes = np.frombuffer(np.ascontiguousarray(left).tobytes(), dtype=np.uint8).reshape(-1, item_size)
        right_bytes = np.frombuffer(np.ascontiguousarray(right).tobytes(), dtype=np.uint8).reshape(-1, item_size)
        different = np.any(left_bytes != right_bytes, axis=1)
        count = int(np.count_nonzero(different))
        if count:
            mismatch_count += count
            flat_index = int(np.flatnonzero(different)[0])
            first_index = tuple(int(i) for i in np.unravel_index(flat_index, left.shape))
            left_flat = left.reshape(-1)
            right_flat = right.reshape(-1)
            mismatch_max = None
            try:
                if np.issubdtype(left.dtype, np.integer) or np.issubdtype(left.dtype, np.bool_):
                    left_numeric = left_flat.astype(np.int64, copy=False)
                    right_numeric = right_flat.astype(np.int64, copy=False)
                    delta = np.abs(left_numeric[different] - right_numeric[different])
                else:
                    delta = np.abs(
                        left_flat[different].astype(np.float64)
                        - right_flat[different].astype(np.float64)
                    )
                if delta.size and np.all(np.isfinite(delta)):
                    mismatch_max = float(np.max(delta))
                    max_absolute_difference = max(max_absolute_difference, mismatch_max)
                else:
                    difference_available = False
            except (TypeError, ValueError, OverflowError):
                difference_available = False
            mismatches.append({
                "tensor_name": name,
                "category": "OUTPUT_BIT_MISMATCH",
                "mismatch_count": count,
                "first_mismatch_index": list(first_index),
                "expected": left[first_index].item(),
                "actual": right[first_index].item(),
                "max_absolute_difference": mismatch_max,
            })
    return {
        "mismatch_count": mismatch_count,
        "max_absolute_difference": max_absolute_difference if difference_available else None,
        "mismatches": mismatches,
    }


_TP4_LAYER_COMPONENTS = ("attn", "o", "suffix_attn", "mlp", "down", "suffix_final")
_TP4_GOLDEN_STAGES = (
    "Q",
    "K",
    "V",
    "ATTENTION_CONTEXT",
    "O",
    "SWIGLU",
    "DOWN",
    "RESIDUAL_HIDDEN",
    "KV_OUTPUT",
)


def _shape_size(shape):
    size = 1
    for dimension in shape:
        size *= int(dimension)
    return size


def _slice_input_for_rank(value, target_shape, rank, degree, tensor_name):
    import numpy as np

    value = np.asarray(value)
    target_shape = tuple(int(dimension) for dimension in target_shape)
    if tuple(value.shape) == target_shape:
        return value
    if value.ndim != len(target_shape):
        raise ToolError(
            "GOLDEN_INPUT_SHAPE_UNSUPPORTED",
            "{}: {} -> {}".format(tensor_name, tuple(value.shape), target_shape),
        )
    axes = [
        axis
        for axis, (source_size, local_size) in enumerate(zip(value.shape, target_shape))
        if source_size == local_size * degree
        and all(value.shape[i] == target_shape[i] for i in range(value.ndim) if i != axis)
    ]
    if len(axes) != 1:
        raise ToolError(
            "GOLDEN_INPUT_SHARD_AXIS_AMBIGUOUS",
            "{}: {} -> {}".format(tensor_name, tuple(value.shape), target_shape),
        )
    axis = axes[0]
    start = rank * target_shape[axis]
    indices = [slice(None)] * value.ndim
    indices[axis] = slice(start, start + target_shape[axis])
    return value[tuple(indices)]


def _combine_tp4_rank_outputs(name, rank_values, target_shape):
    import numpy as np

    degree = 4
    target_shape = tuple(int(dimension) for dimension in target_shape)
    values = [np.asarray(rank_values[rank]) for rank in range(degree)]
    if all(tuple(value.shape) == target_shape for value in values):
        baseline = values[0]
        failures = []
        mismatch_count = 0
        for rank, value in enumerate(values[1:], start=1):
            compared = compare_named_outputs({name: baseline}, {name: value})
            if compared["mismatch_count"]:
                mismatch_count += compared["mismatch_count"]
                failures.append({"rank": rank, **compared["mismatches"][0]})
        return baseline, mismatch_count, failures

    shapes = {tuple(value.shape) for value in values}
    if len(shapes) != 1:
        raise ToolError("GOLDEN_RANK_OUTPUT_SHAPES_DIFFER", name)
    local_shape = next(iter(shapes))
    axes = [
        axis
        for axis, (full_size, local_size) in enumerate(zip(target_shape, local_shape))
        if full_size == local_size * degree
        and all(target_shape[i] == local_shape[i] for i in range(len(target_shape)) if i != axis)
    ]
    if len(axes) != 1:
        raise ToolError(
            "GOLDEN_OUTPUT_SHARD_AXIS_AMBIGUOUS",
            "{}: {} -> {}".format(name, local_shape, target_shape),
        )
    combined = np.concatenate(values, axis=axes[0])
    if tuple(combined.shape) != target_shape:
        raise ToolError("GOLDEN_OUTPUT_GATHER_SHAPE_INVALID", name)
    return combined, 0, []


def _graph_tensor_by_name(source_graph, tensor_name):
    matches = [tensor for tensor in source_graph.tensors if tensor.name == tensor_name]
    if len(matches) != 1:
        raise ToolError(
            "GOLDEN_BOUNDARY_MAPPING_UNRESOLVED",
            "expected one source tensor named {!r}, found {}".format(tensor_name, len(matches)),
        )
    return matches[0]


def derive_tp4_boundary_contract(source_graph, source_layer):
    """Resolve the nine required numerical boundaries from graph connectivity."""
    tensors = {tensor.index: tensor for tensor in source_graph.tensors}
    roles = source_layer.roles

    def role_tensor(role, member):
        match = roles.get(role)
        if match is None:
            raise ToolError("GOLDEN_BOUNDARY_MAPPING_UNRESOLVED", "missing projection role {}".format(role))
        if member == "activation":
            index = match.activation_tensor_index
        else:
            indices = match.output_tensor_indices
            if len(indices) != 1:
                raise ToolError(
                    "GOLDEN_BOUNDARY_MAPPING_UNRESOLVED",
                    "projection {} has {} output tensors".format(role, len(indices)),
                )
            index = indices[0]
        tensor = tensors.get(index)
        if tensor is None:
            raise ToolError("GOLDEN_BOUNDARY_MAPPING_UNRESOLVED", "tensor index {} is missing".format(index))
        return tensor.name

    def residual_tensor_after(role):
        match = roles.get(role)
        if match is None:
            raise ToolError("GOLDEN_BOUNDARY_MAPPING_UNRESOLVED", "missing projection role {}".format(role))
        source_indices = set(match.output_tensor_indices)
        add_ops = [
            operator
            for operator in source_graph.operators
            if "ADD" in operator.opcode.upper()
            and source_indices.intersection(index for index in operator.inputs if index >= 0)
        ]
        outputs = [index for operator in add_ops for index in operator.outputs]
        if len(outputs) != 1 or outputs[0] not in tensors:
            raise ToolError(
                "GOLDEN_BOUNDARY_MAPPING_UNRESOLVED",
                "could not resolve the residual output after {}".format(role),
            )
        return tensors[outputs[0]].name

    hidden_output = residual_tensor_after("down")
    public_output_tensors = [tensors[index] for index in source_graph.outputs if index in tensors]
    current_keys = [
        tensor.name for tensor in public_output_tensors
        if "curr" in tensor.name.lower() and "key" in tensor.name.lower()
    ]
    current_values = [
        tensor.name for tensor in public_output_tensors
        if "curr" in tensor.name.lower() and "value" in tensor.name.lower()
    ]
    semantic_id = getattr(roles.get("q"), "semantic_layer_id", None) if roles.get("q") else None
    if semantic_id is not None:
        scope = re.compile(
            r"(?:^|/)(?:layers?|blocks?)/{}/(?:|.*)".format(int(semantic_id)),
            re.IGNORECASE,
        )

        def scoped(names):
            result = []
            for name in names:
                normalized = name.lower().replace("\\", "/")
                if scope.search(normalized) or re.search(r"(?:_|/){:d}$".format(int(semantic_id)), normalized):
                    result.append(name)
            return result

        current_keys = scoped(current_keys)
        current_values = scoped(current_values)
    if len(current_keys) != 1 or len(current_values) != 1:
        raise ToolError(
            "GOLDEN_BOUNDARY_MAPPING_UNRESOLVED",
            "layer {} does not resolve one public current key and value output (keys={}, values={})".format(
                source_layer.layer_id, current_keys, current_values
            ),
        )

    def one(stage, component, source_name, distribution):
        return [{
            "component": component,
            "source_name": source_name,
            "candidate_name": source_name,
            "distribution": distribution,
        }]

    qkv = {
        role.upper(): one(role.upper(), "attn", role_tensor(role, "output"), "rank_shard")
        for role in ("q", "k", "v")
    }
    contract = {
        **qkv,
        "ATTENTION_CONTEXT": one(
            "ATTENTION_CONTEXT", "attn", role_tensor("o", "activation"), "rank_shard"
        ),
        "O": one("O", "o", role_tensor("o", "output"), "rank_shard"),
        "SWIGLU": one("SWIGLU", "mlp", role_tensor("down", "activation"), "rank_shard"),
        "DOWN": one("DOWN", "down", role_tensor("down", "output"), "rank_shard"),
        "RESIDUAL_HIDDEN": [
            {
                "component": "suffix_attn",
                "source_name": residual_tensor_after("o"),
                "candidate_name": residual_tensor_after("o"),
                "distribution": "replicated",
            },
            {
                "component": "suffix_final",
                "source_name": hidden_output,
                "candidate_name": hidden_output,
                "distribution": "replicated",
            },
        ],
        "KV_OUTPUT": [
            {
                "component": "attn",
                "source_name": current_keys[0],
                "candidate_name": current_keys[0],
                "distribution": "rank_shard",
            },
            {
                "component": "attn",
                "source_name": current_values[0],
                "candidate_name": current_values[0],
                "distribution": "rank_shard",
            },
        ],
    }
    return contract


def _validate_boundary_contract(boundary_contract):
    if not isinstance(boundary_contract, dict) or set(boundary_contract) != set(_TP4_GOLDEN_STAGES):
        raise ToolError("GOLDEN_STAGE_COVERAGE_MISSING", "TP4 boundary contract must define all nine stages")
    for stage in _TP4_GOLDEN_STAGES:
        checks = boundary_contract[stage]
        if not isinstance(checks, (list, tuple)) or not checks:
            raise ToolError("GOLDEN_STAGE_COVERAGE_MISSING", "{} has no tensor checks".format(stage))
        for check in checks:
            if (
                not isinstance(check, dict)
                or check.get("component") not in _TP4_LAYER_COMPONENTS
                or not isinstance(check.get("source_name"), str)
                or not check["source_name"]
                or not isinstance(check.get("candidate_name"), str)
                or not check["candidate_name"]
                or check.get("distribution") not in ("rank_shard", "replicated")
            ):
                raise ToolError("GOLDEN_BOUNDARY_CONTRACT_INVALID", repr((stage, check)))


def _source_tensor_quant_domain(source_graph, name):
    if source_graph is None:
        return None
    tensor = _graph_tensor_by_name(source_graph, name)
    quant = tensor.quantization
    return {
        "dtype": tensor.dtype,
        "shape": list(tensor.shape),
        "scales": list(quant.scales) if quant else [],
        "zero_points": list(quant.zero_points) if quant else [],
        "quantized_dimension": quant.quantized_dimension if quant else None,
    }


def _run_named(runner, model_path, inputs, output_names=None):
    if output_names is None:
        return runner.run(model_path, inputs)
    try:
        return runner.run(model_path, inputs, output_names=output_names)
    except TypeError as exc:
        raise ToolError(
            "GOLDEN_INTERNAL_OUTPUT_UNSUPPORTED",
            "runner cannot expose requested boundary tensors: {}".format(exc),
        )


def run_tp4_layer_golden(
    source_model,
    candidate_root,
    candidate_manifest,
    fixtures,
    layer_id,
    runner,
    source_graph=None,
    source_layer=None,
    boundary_contract=None,
    public_source_output_names=None,
    source_result=None,
    candidate_input_overrides=None,
):
    """Stitch one TP4 layer and compare public outputs plus all nine boundaries.

    The Canonical graph and rank graphs are executed by the same host backend.
    Internal tensor names come from graph discovery, not hand-entered tensor
    IDs. When no boundary contract is supplied, this remains a low-level public
    output stitch check and does not satisfy the release gate.
    """
    import numpy as np

    if not hasattr(runner, "run") or not callable(runner.run):
        raise ToolError("GOLDEN_EXECUTION_UNAVAILABLE", "runner has no TFLite run method")
    if candidate_manifest.get("structural_status") != "STRUCTURAL_PASS":
        raise ToolError("GOLDEN_CANDIDATE_NOT_STRUCTURAL", str(layer_id))
    contract = candidate_manifest.get("collective_contract", {})
    if contract.get("architecture") != "FOUR_STAGE_ALL_GATHER":
        raise ToolError("GOLDEN_TP4_CONTRACT_INVALID", "expected FOUR_STAGE_ALL_GATHER")
    if candidate_manifest.get("rank_order") != [0, 1, 2, 3]:
        raise ToolError("GOLDEN_RANK_ORDER_INVALID", str(candidate_manifest.get("rank_order")))
    stages = [item.get("stage") for item in contract.get("all_gather_boundaries", [])]
    if stages != ["AG0", "AG1", "AG2", "AG3"]:
        raise ToolError("GOLDEN_ALL_GATHER_ORDER_INVALID", repr(stages))
    if boundary_contract is None and source_graph is not None and source_layer is not None:
        boundary_contract = derive_tp4_boundary_contract(source_graph, source_layer)
    if boundary_contract is not None:
        _validate_boundary_contract(boundary_contract)
        if public_source_output_names is None and source_graph is not None:
            tensor_by_index = {tensor.index: tensor for tensor in source_graph.tensors}
            public_source_output_names = [
                tensor_by_index[index].name
                for index in source_graph.outputs
                if index in tensor_by_index
            ]
        if not public_source_output_names:
            raise ToolError(
                "GOLDEN_PUBLIC_OUTPUT_MAPPING_UNRESOLVED",
                "the source graph's public output names are required with boundary checks",
            )
        public_source_output_names = list(dict.fromkeys(public_source_output_names))
        requested_source_outputs = list(public_source_output_names)
        requested_source_outputs.extend(
            check["source_name"]
            for stage in _TP4_GOLDEN_STAGES
            for check in boundary_contract[stage]
        )
        requested_source_outputs = list(dict.fromkeys(requested_source_outputs))
    else:
        requested_source_outputs = None

    profile = candidate_manifest.get("profile")
    layer_rows = [
        row for row in candidate_manifest.get("models", [])
        if row.get("layer") == int(layer_id) and row.get("profile") == profile
    ]
    grouped = {component: {} for component in _TP4_LAYER_COMPONENTS}
    for row in layer_rows:
        component, rank = row.get("component"), row.get("rank")
        if component not in grouped or rank not in range(4) or rank in grouped[component]:
            raise ToolError("GOLDEN_COMPONENT_COVERAGE_INVALID", repr((component, rank)))
        grouped[component][rank] = row
    if any(set(grouped[component]) != set(range(4)) for component in _TP4_LAYER_COMPONENTS):
        raise ToolError("GOLDEN_COMPONENT_COVERAGE_INVALID", str(layer_id))

    if source_result is None:
        source_result = _run_named(runner, source_model, fixtures, requested_source_outputs)
    canonical_values = source_result.get("outputs")
    if not isinstance(canonical_values, dict):
        raise ToolError("GOLDEN_REPORT_INVALID", "source runner returned no named outputs")
    source_hash = str(source_result.get("source_sha256", "")).lower()
    if len(source_hash) != 64:
        raise ToolError("GOLDEN_REPORT_INVALID", "source result has no SHA256")
    if boundary_contract is not None:
        missing_source = set(requested_source_outputs) - set(canonical_values)
        if missing_source:
            raise ToolError("GOLDEN_SOURCE_BOUNDARY_OUTPUT_MISSING", repr(sorted(missing_source)))
        expected_outputs = {
            name: canonical_values[name]
            for name in public_source_output_names
        }
    else:
        expected_outputs = canonical_values

    target_shapes = {}
    for row in layer_rows:
        for port in row.get("io", {}).get("inputs", []) + row.get("io", {}).get("outputs", []):
            name = port.get("name")
            if name:
                target_shapes.setdefault(name, set()).add(tuple(int(x) for x in port.get("shape", [])))
    for name, value in canonical_values.items():
        target_shapes.setdefault(name, set()).add(tuple(np.asarray(value).shape))
    full_shapes = {}
    for name, shapes in target_shapes.items():
        largest = max(_shape_size(shape) for shape in shapes)
        candidates = {shape for shape in shapes if _shape_size(shape) == largest}
        if len(candidates) != 1:
            raise ToolError("GOLDEN_TENSOR_SHAPE_AMBIGUOUS", name)
        full_shapes[name] = next(iter(candidates))

    extra_outputs_by_component = {component: set() for component in _TP4_LAYER_COMPONENTS}
    if boundary_contract is not None:
        for stage in _TP4_GOLDEN_STAGES:
            for check in boundary_contract[stage]:
                extra_outputs_by_component[check["component"]].add(check["candidate_name"])

    candidate_root = Path(candidate_root)
    global_outputs = {}
    candidate_hashes = {}
    component_rank_outputs = {component: {} for component in _TP4_LAYER_COMPONENTS}
    replicated_mismatches = []
    replicated_mismatch_count = 0
    for component in _TP4_LAYER_COMPONENTS:
        public_names = None
        for rank in range(4):
            row = grouped[component][rank]
            if str(row.get("source_sha256", "")).lower() != source_hash:
                raise ToolError("GOLDEN_CANDIDATE_SOURCE_SHA_MISMATCH", str(row.get("path")))
            model_path = candidate_root / row["path"]
            if not model_path.is_file():
                raise ToolError("GOLDEN_CANDIDATE_MISSING", str(model_path))
            output_hash = _hash_file(model_path)
            if output_hash != str(row.get("output_sha256", "")).lower():
                raise ToolError("GOLDEN_CANDIDATE_SHA_MISMATCH", str(row["path"]))
            candidate_hashes[row["path"]] = output_hash
            inputs = {}
            for port in row.get("io", {}).get("inputs", []):
                name = port["name"]
                value = global_outputs.get(
                    name,
                    (candidate_input_overrides or {}).get(name, fixtures.get(name)),
                )
                if value is None and component == "attn":
                    # TP4's attention component starts after the canonical
                    # pre-attention normalization. That intermediate is not a
                    # public fixture or another candidate component output;
                    # use the same tensor from the canonical graph execution.
                    value = canonical_values.get(name)
                if value is None:
                    raise ToolError("GOLDEN_INPUT_MISSING", name)
                inputs[name] = _slice_input_for_rank(value, port["shape"], rank, 4, name)
            names = {port.get("name") for port in row.get("io", {}).get("outputs", [])}
            if public_names is None:
                public_names = names
            elif names != public_names:
                raise ToolError("GOLDEN_RANK_OUTPUT_SET_MISMATCH", component)
            requested = sorted(public_names | extra_outputs_by_component[component])
            component_result = _run_named(
                runner,
                model_path,
                inputs,
                requested if requested != sorted(public_names) else None,
            )
            outputs = component_result.get("outputs")
            expected_names = public_names | extra_outputs_by_component[component]
            if not isinstance(outputs, dict) or set(outputs) != expected_names:
                raise ToolError("GOLDEN_COMPONENT_OUTPUT_MAPPING_INVALID", str(row["path"]))
            component_rank_outputs[component][rank] = outputs

        for name in sorted(public_names or ()):
            if name not in full_shapes:
                raise ToolError("GOLDEN_OUTPUT_SHAPE_UNRESOLVED", name)
            values = {
                rank: component_rank_outputs[component][rank][name]
                for rank in range(4)
            }
            combined, mismatch_count, mismatches = _combine_tp4_rank_outputs(
                name, values, full_shapes[name]
            )
            global_outputs[name] = combined
            if mismatch_count:
                replicated_mismatch_count += mismatch_count
                replicated_mismatches.extend(
                    {
                        "layer": int(layer_id),
                        "rank": item.get("rank"),
                        "stage": component,
                        "operator": component,
                        "tensor_name": name,
                        "component": component,
                        **item,
                    }
                    for item in mismatches
                )

    candidate_outputs = {
        name: global_outputs[name]
        for name in expected_outputs
        if name in global_outputs
    }
    compared = compare_named_outputs(expected_outputs, candidate_outputs)
    boundary_results = []
    boundary_mismatch_count = 0
    boundary_mismatches = []
    if boundary_contract is not None:
        for rank in range(4):
            for stage in _TP4_GOLDEN_STAGES:
                check_results = []
                stage_mismatch_count = 0
                stage_max_difference = 0.0
                stage_difference_available = True
                for check in boundary_contract[stage]:
                    component = check["component"]
                    source_name = check["source_name"]
                    candidate_name = check["candidate_name"]
                    actual_outputs = component_rank_outputs[component][rank]
                    if candidate_name not in actual_outputs:
                        raise ToolError(
                            "GOLDEN_CANDIDATE_BOUNDARY_OUTPUT_MISSING",
                            "{}:{}:{}".format(component, rank, candidate_name),
                        )
                    if source_name not in canonical_values:
                        raise ToolError("GOLDEN_SOURCE_BOUNDARY_OUTPUT_MISSING", source_name)
                    expected_value = np.asarray(canonical_values[source_name])
                    actual_value = np.asarray(actual_outputs[candidate_name])
                    if check["distribution"] == "rank_shard":
                        expected_value = _slice_input_for_rank(
                            expected_value,
                            actual_value.shape,
                            rank,
                            4,
                            source_name,
                        )
                    compared_boundary = compare_named_outputs(
                        {"boundary": expected_value},
                        {"boundary": actual_value},
                    )
                    stage_mismatch_count += compared_boundary["mismatch_count"]
                    difference = compared_boundary["max_absolute_difference"]
                    if difference is None:
                        stage_difference_available = False
                    else:
                        stage_max_difference = max(stage_max_difference, difference)
                    quant_domain = _source_tensor_quant_domain(source_graph, source_name)
                    detailed = {
                        "component": component,
                        "source_tensor": source_name,
                        "candidate_tensor": candidate_name,
                        "distribution": check["distribution"],
                        "source_shape": list(np.asarray(canonical_values[source_name]).shape),
                        "actual_shape": list(actual_value.shape),
                        "dtype": str(actual_value.dtype),
                        "quant_domain": quant_domain,
                        "mismatch_count": compared_boundary["mismatch_count"],
                        "max_absolute_difference": difference,
                        "mismatches": compared_boundary["mismatches"],
                    }
                    check_results.append(detailed)
                    if compared_boundary["mismatch_count"]:
                        boundary_mismatches.extend(
                            {
                                "layer": int(layer_id),
                                "rank": rank,
                                "stage": stage,
                                "operator": component,
                                "tensor_name": source_name,
                                "candidate_tensor": candidate_name,
                                "dtype": str(actual_value.dtype),
                                "shape": list(actual_value.shape),
                                "quant_domain": quant_domain,
                                **item,
                            }
                            for item in compared_boundary["mismatches"]
                        )
                boundary_mismatch_count += stage_mismatch_count
                boundary_results.append({
                    "layer": int(layer_id),
                    "rank": rank,
                    "stage": stage,
                    "mismatch_count": stage_mismatch_count,
                    "max_absolute_difference": (
                        stage_max_difference if stage_difference_available else None
                    ),
                    "checks": check_results,
                })

    mismatches = replicated_mismatches + boundary_mismatches + compared["mismatches"]
    mismatch_count = (
        replicated_mismatch_count
        + boundary_mismatch_count
        + compared["mismatch_count"]
    )
    return {
        "profile": profile,
        "layer": int(layer_id),
        "topology": "TP4_FOUR_STAGE_ALL_GATHER",
        "backend": source_result.get("backend"),
        "simulate_fp16": source_result.get("simulate_fp16"),
        "source_sha256": source_hash,
        "fixture_sha256": source_result.get("fixture_sha256"),
        "candidate_sha256": candidate_hashes,
        "stitched_outputs": global_outputs,
        "rank_component_execution_count": 24,
        "required_stages": list(_TP4_GOLDEN_STAGES) if boundary_contract is not None else [],
        "boundary_results": boundary_results,
        "public_output_mismatch_count": compared["mismatch_count"],
        "rank_replication_mismatch_count": replicated_mismatch_count,
        "mismatch_count": mismatch_count,
        "mismatches": mismatches,
        "numerical_status": "NUMERICAL_PASS" if mismatch_count == 0 else "NUMERICAL_FAIL",
    }


def _source_graph_hidden_input(source_graph):
    tensor_by_index = {tensor.index: tensor for tensor in source_graph.tensors}
    candidates = [
        tensor_by_index[index].name
        for index in source_graph.inputs
        if index in tensor_by_index
        and any(token in tensor_by_index[index].name.lower() for token in ("input_embeds", "hidden_states", "hidden_state"))
    ]
    if len(candidates) != 1:
        raise ToolError(
            "GOLDEN_HIDDEN_INPUT_MAPPING_UNRESOLVED",
            "expected one hidden-state graph input, found {}".format(candidates),
        )
    return candidates[0]


def run_tp4_profile_golden(source_report, candidate_root, candidate_manifest, profile, runner, seed=24031984):
    """Run deterministic canonical-to-rank Golden checks across a TP4 profile.

    Canonical graphs may contain one or several semantic layers. A grouped
    graph runs once per graph and exposes layer boundaries by tensor name;
    candidate layers still execute and chain in semantic order.
    """
    import numpy as np

    from .fixtures import make_deterministic_tflite_inputs
    from .partition import _graph_index, _layer_boundary_tensors

    if profile not in ("P128", "D1"):
        raise ToolError("GOLDEN_PROFILE_UNSUPPORTED", str(profile))
    if candidate_manifest.get("profile") != profile:
        raise ToolError("GOLDEN_PROFILE_MISMATCH", str(candidate_manifest.get("profile")))
    expected_layers = int(candidate_manifest.get("expected_layers", 0))
    if expected_layers <= 0 or source_report.expected_layers != expected_layers:
        raise ToolError("GOLDEN_LAYER_COVERAGE_INVALID", str(expected_layers))
    coverage = source_report.coverage.get(profile, {})
    if coverage.get("complete") is not True or coverage.get("present_layers") != list(range(expected_layers)):
        raise ToolError("GOLDEN_COVERAGE_INCOMPLETE", repr(coverage))

    layer_sources = {}
    source_hashes = {}
    graph_owners = {}
    for source in source_report.sources:
        if profile not in source.profiles:
            continue
        if source.model_info is None:
            raise ToolError("GOLDEN_SOURCE_MODEL_INFO_MISSING", source.source_name)
        if source.source_name in source_hashes:
            raise ToolError("GOLDEN_SOURCE_NAME_DUPLICATE", source.source_name)
        source_hashes[source.source_name] = source.source_sha256
        for layer in source.layers:
            if layer.profile != profile:
                continue
            key = (source.source_name, layer.graph_index)
            graph_owners.setdefault(key, []).append(layer.layer_id)
            if layer.layer_id in layer_sources:
                raise ToolError("GOLDEN_LAYER_DUPLICATE", str(layer.layer_id))
            layer_sources[layer.layer_id] = (source, layer)
    missing_layers = sorted(set(range(expected_layers)) - set(layer_sources))
    if missing_layers:
        raise ToolError("GOLDEN_COVERAGE_INCOMPLETE", repr(missing_layers))
    source_rows = {
        row.get("source"): row.get("source_sha256")
        for row in candidate_manifest.get("source_audit", {}).get("sources", [])
        if profile in row.get("profiles", [])
    }
    if source_rows != source_hashes:
        raise ToolError("GOLDEN_SOURCE_MAPPING_MISMATCH", repr({"manifest": source_rows, "discovery": source_hashes}))

    candidate_root = Path(candidate_root)
    layer_reports = []
    fixture_hashes = {}
    candidate_input_fixture_hashes = {}
    canonical_hidden = None
    candidate_hidden = None
    graph_contexts = {}
    for layer_id in range(expected_layers):
        source, layer = layer_sources[layer_id]
        if len(source.model_info.graphs) != 1:
            raise ToolError(
                "GOLDEN_SOURCE_GRAPH_UNSUPPORTED",
                "{} has {} subgraphs".format(source.source_name, len(source.model_info.graphs)),
            )
        graph = source.model_info.graphs[layer.graph_index]
        source_path = Path(source.model_info.source_path)
        if not source_path.is_file():
            raise ToolError("GOLDEN_SOURCE_FILE_MISSING", str(source_path))
        context_key = (source.source_name, layer.graph_index)
        context = graph_contexts.get(context_key)
        if context is None:
            graph_layers = [
                item for item in source.layers
                if item.profile == profile and item.graph_index == layer.graph_index
            ]
            fixtures = make_deterministic_tflite_inputs(
                graph,
                seed=seed + min(item.layer_id for item in graph_layers) * 104729,
            )
            graph_input_hidden_name = _source_graph_hidden_input(graph)
            if canonical_hidden is not None:
                graph_input_tensor = next(
                    (graph.tensors[index] for index in graph.inputs if graph.tensors[index].name == graph_input_hidden_name),
                    None,
                )
                if graph_input_tensor is None or tuple(np.asarray(canonical_hidden).shape) != tuple(graph_input_tensor.shape):
                    raise ToolError(
                        "GOLDEN_LAYER_CHAIN_SHAPE_MISMATCH",
                        "layer {} cannot accept preceding canonical hidden state".format(layer_id),
                    )
                fixtures[graph_input_hidden_name] = canonical_hidden
            contracts = {
                item.layer_id: derive_tp4_boundary_contract(graph, item)
                for item in graph_layers
            }
            requested_names = [graph.tensors[index].name for index in graph.outputs]
            requested_names.extend(
                check["source_name"]
                for item_contract in contracts.values()
                for stage in _TP4_GOLDEN_STAGES
                for check in item_contract[stage]
            )
            requested_names.extend(
                _tp4_attention_intermediate_inputs(
                    candidate_manifest,
                    profile,
                    {item.layer_id for item in graph_layers},
                    graph,
                )
            )
            requested_names = list(dict.fromkeys(requested_names))
            source_result = _run_named(runner, source_path, fixtures, requested_names)
            context = {
                "fixtures": fixtures,
                "source_result": source_result,
                "contracts": contracts,
                "graph": graph,
                "graph_input_hidden_name": graph_input_hidden_name,
                "graph_layers": {item.layer_id for item in graph_layers},
            }
            graph_contexts[context_key] = context

        fixtures = context["fixtures"]
        source_result = context["source_result"]
        boundary_contract = context["contracts"].get(layer_id)
        if boundary_contract is None:
            raise ToolError("GOLDEN_BOUNDARY_MAPPING_UNRESOLVED", "layer {} has no graph contract".format(layer_id))
        public_outputs = [
            boundary_contract["RESIDUAL_HIDDEN"][-1]["source_name"],
            *[check["source_name"] for check in boundary_contract["KV_OUTPUT"]],
        ]
        public_outputs = list(dict.fromkeys(public_outputs))

        graph_operators, graph_producers, _graph_consumers = _graph_index(graph)
        try:
            hidden_input_index, hidden_output_index = _layer_boundary_tensors(
                layer,
                graph,
                graph_operators,
                graph_producers,
                set(graph.inputs),
                {tensor.index: tensor for tensor in graph.tensors},
            )
        except ToolError:
            if len(context["graph_layers"]) != 1:
                raise
            # Legacy single-layer test/source graphs may expose the block
            # boundary without retaining the normalization operator itself.
            hidden_input_name = _source_graph_hidden_input(graph)
            hidden_input_index = next(
                index for index in graph.inputs if graph.tensors[index].name == hidden_input_name
            )
            hidden_output_name = boundary_contract["RESIDUAL_HIDDEN"][-1]["source_name"]
            hidden_output_index = next(
                tensor.index for tensor in graph.tensors if tensor.name == hidden_output_name
            )
        hidden_input_name = graph.tensors[hidden_input_index].name
        hidden_output_name = graph.tensors[hidden_output_index].name
        canonical_values = source_result.get("outputs", {})
        if hidden_output_name not in canonical_values:
            raise ToolError("GOLDEN_HIDDEN_OUTPUT_MISSING", hidden_output_name)
        canonical_hidden = np.asarray(canonical_values[hidden_output_name])
        fixture_hashes[str(layer_id)] = source_result.get("fixture_sha256")

        candidate_overrides = {}
        candidate_input_fixture_hash = None
        if candidate_hidden is not None:
            expected_shape = tuple(graph.tensors[hidden_input_index].shape)
            if tuple(np.asarray(candidate_hidden).shape) != expected_shape:
                raise ToolError(
                    "GOLDEN_CANDIDATE_LAYER_CHAIN_SHAPE_MISMATCH",
                    "layer {}: {} -> {}".format(layer_id, tuple(np.asarray(candidate_hidden).shape), expected_shape),
                )
            candidate_overrides[hidden_input_name] = candidate_hidden
            derived_inputs, candidate_input_fixture_hash = _tp4_candidate_attention_input_overrides(
                candidate_manifest,
                profile,
                layer_id,
                graph,
                context["graph_layers"],
                context["graph_input_hidden_name"],
                hidden_input_index,
                candidate_hidden,
                fixtures,
                source_path,
                runner,
            )
            candidate_overrides.update(derived_inputs)
            if candidate_input_fixture_hash is not None:
                candidate_input_fixture_hashes[str(layer_id)] = candidate_input_fixture_hash

        layer_report = run_tp4_layer_golden(
            source_path,
            candidate_root,
            candidate_manifest,
            fixtures,
            layer_id,
            runner,
            source_graph=graph,
            source_layer=layer,
            boundary_contract=boundary_contract,
            public_source_output_names=public_outputs,
            source_result=source_result,
            candidate_input_overrides=candidate_overrides,
        )
        stitched_outputs = layer_report.pop("stitched_outputs", {})
        if hidden_output_name not in stitched_outputs:
            raise ToolError("GOLDEN_CANDIDATE_HIDDEN_OUTPUT_MISSING", hidden_output_name)
        candidate_hidden = np.asarray(stitched_outputs[hidden_output_name])
        layer_report["source_name"] = source.source_name
        layer_source_sha = source_hashes[source.source_name]
        for mismatch in layer_report.get("mismatches", []):
            mismatch.setdefault("source_sha256", layer_source_sha)
        layer_report["seed"] = seed + min(context["graph_layers"]) * 104729
        layer_report["canonical_graph_group_layers"] = sorted(context["graph_layers"])
        if candidate_input_fixture_hash is not None:
            layer_report["candidate_input_fixture_sha256"] = candidate_input_fixture_hash
        layer_reports.append(layer_report)

    boundary_results = [
        row
        for layer_report in layer_reports
        for row in layer_report.get("boundary_results", [])
    ]
    mismatches = [
        row
        for layer_report in layer_reports
        for row in layer_report.get("mismatches", [])
    ]
    mismatch_count = sum(report.get("mismatch_count", 0) for report in layer_reports)
    return {
        "schema": 1,
        "profile": profile,
        "topology": "TP4_FOUR_STAGE_ALL_GATHER",
        "backend": layer_reports[0].get("backend") if layer_reports else None,
        "simulate_fp16": layer_reports[0].get("simulate_fp16") if layer_reports else None,
        "seed": seed,
        "required_stages": list(_TP4_GOLDEN_STAGES),
        "source_sha256_by_name": source_hashes,
        "fixture_sha256_by_layer": fixture_hashes,
        "candidate_input_fixture_sha256_by_layer": candidate_input_fixture_hashes,
        "candidate_sha256": {
            key: value
            for report in layer_reports
            for key, value in report.get("candidate_sha256", {}).items()
        },
        "layer_reports": layer_reports,
        "boundary_results": boundary_results,
        "golden_coverage": {
            "complete": len(layer_reports) == expected_layers and not missing_layers,
            "layers": [report["layer"] for report in layer_reports],
            "ranks": list(range(4)),
            "components": list(_TP4_LAYER_COMPONENTS),
        },
        "mismatch_count": mismatch_count,
        "mismatches": mismatches,
        "numerical_status": "NUMERICAL_PASS" if mismatch_count == 0 and len(layer_reports) == expected_layers else "NUMERICAL_FAIL",
    }


def _tp2_single_quantization(tensor, role, rank):
    quantization = tensor.quantization
    if quantization is None or len(quantization.scales) != 1:
        raise ToolError(
            "TP2_GOLDEN_QUANTIZATION_UNSUPPORTED",
            "{} rank {} requires scalar per-tensor quantization".format(role, rank),
        )
    zero_points = tuple(quantization.zero_points or (0,))
    if len(zero_points) != 1 or float(quantization.scales[0]) <= 0:
        raise ToolError(
            "TP2_GOLDEN_QUANTIZATION_UNSUPPORTED",
            "{} rank {} has invalid scalar quantization".format(role, rank),
        )
    return float(quantization.scales[0]), int(zero_points[0])


def _tp2_reduce_quantized_partials(partials, partial_tensors, output_tensor, role):
    """Sum input-shard outputs in their declared quantization domains.

    This is a diagnostic reference reduction. It intentionally does not claim
    that this host policy matches the target collective's accumulator or
    requantization implementation.
    """
    import numpy as np

    if not partials or len(partials) != len(partial_tensors):
        raise ToolError("TP2_GOLDEN_PARTIAL_COVERAGE_INVALID", role)
    output_scale, output_zero_point = _tp2_single_quantization(output_tensor, role, "source")
    output_dtype_by_name = {
        "INT8": np.dtype(np.int8),
        "UINT8": np.dtype(np.uint8),
        "INT16": np.dtype(np.int16),
        "UINT16": np.dtype(np.uint16),
        "INT32": np.dtype(np.int32),
        "UINT32": np.dtype(np.uint32),
    }
    output_dtype = output_dtype_by_name.get(str(output_tensor.dtype).upper())
    if output_dtype is None or not np.issubdtype(output_dtype, np.integer):
        raise ToolError("TP2_GOLDEN_DTYPE_UNSUPPORTED", "{} partials must be integer".format(role))
    accumulated = np.zeros(np.asarray(partials[0]).shape, dtype=np.float64)
    for rank, (partial, tensor) in enumerate(zip(partials, partial_tensors)):
        partial = np.asarray(partial)
        if partial.shape != accumulated.shape or not np.issubdtype(partial.dtype, np.integer):
            raise ToolError("TP2_GOLDEN_PARTIAL_SHAPE_OR_DTYPE_INVALID", "{} rank {}".format(role, rank))
        partial_scale, partial_zero_point = _tp2_single_quantization(tensor, role, rank)
        accumulated += (partial.astype(np.float64) - partial_zero_point) * (partial_scale / output_scale)
    requantized = np.rint(accumulated + output_zero_point)
    limits = np.iinfo(output_dtype)
    return np.clip(requantized, limits.min, limits.max).astype(output_dtype)


def _tp2_partial_range_diagnostic(rank_results):
    """Report integer rank partials that reach a representable endpoint.

    Endpoint equality is evidence of possible clipping, not proof that the
    kernel saturated an out-of-range mathematical value.
    """
    limit_hits = []
    for result in rank_results:
        at_min = int(result.get("values_at_dtype_min", 0))
        at_max = int(result.get("values_at_dtype_max", 0))
        if at_min or at_max:
            limit_hits.append({
                "rank": result.get("rank"),
                "dtype": result.get("output_dtype"),
                "dtype_min": result.get("output_dtype_min"),
                "dtype_max": result.get("output_dtype_max"),
                "values_at_dtype_min": at_min,
                "values_at_dtype_max": at_max,
            })
    return {
        "status": "RANGE_LIMIT_REACHED" if limit_hits else "WITHIN_DTYPE_RANGE",
        "rank_limit_hits": limit_hits,
        "root_cause_confirmed": False,
        "safe_repair_available": False,
    }


def run_tp2_row_reduce_profile_golden(
    source_report,
    candidate_root,
    candidate_manifest,
    profile,
    runner,
    seed=24031984,
    fixture_code_scale_factors=(64,),
    source_execution_paths=None,
    layer_ids=None,
    candidate_model_paths=None,
):
    """Numerically probe every TP2 O/Down row-reduce boundary in a profile.

    Canonical activations are fed to each row-sharded projection so this run
    isolates the two quantized row-reduce boundaries per layer. Other TP2
    components and cross-layer candidate chaining are reported as uncovered;
    a zero count in this probe therefore can never satisfy the full release
    gate.
    """
    import numpy as np

    from .fixtures import make_deterministic_tflite_inputs
    from .tflite_reader import read_tflite

    if profile not in ("P128", "D1"):
        raise ToolError("GOLDEN_PROFILE_UNSUPPORTED", str(profile))
    if candidate_manifest.get("profile") != profile:
        raise ToolError("GOLDEN_PROFILE_MISMATCH", str(candidate_manifest.get("profile")))
    if candidate_manifest.get("topology") != "TP2_ROW_REDUCE":
        raise ToolError("GOLDEN_TP2_CONTRACT_INVALID", str(candidate_manifest.get("topology")))
    if candidate_manifest.get("structural_status") != "STRUCTURAL_PASS":
        raise ToolError("GOLDEN_CANDIDATE_NOT_STRUCTURAL", profile)
    if not hasattr(runner, "run") or not callable(runner.run):
        raise ToolError("GOLDEN_EXECUTION_UNAVAILABLE", "runner has no TFLite run method")
    if not all(hasattr(source_report, name) for name in ("expected_layers", "coverage", "sources")):
        raise ToolError("GOLDEN_SOURCE_REPORT_INVALID", "source discovery report is incomplete")
    expected_layers = int(candidate_manifest.get("expected_layers", 0))
    coverage = source_report.coverage.get(profile, {})
    if (
        expected_layers <= 0
        or source_report.expected_layers != expected_layers
        or coverage.get("complete") is not True
        or coverage.get("present_layers") != list(range(expected_layers))
    ):
        raise ToolError("GOLDEN_COVERAGE_INCOMPLETE", repr(coverage))
    selected_layers = set(range(expected_layers)) if layer_ids is None else {int(value) for value in layer_ids}
    if not selected_layers or any(value < 0 or value >= expected_layers for value in selected_layers):
        raise ToolError("GOLDEN_LAYER_SELECTION_INVALID", repr(layer_ids))
    factors = tuple(int(value) for value in fixture_code_scale_factors)
    if not factors or any(value < 1 for value in factors) or len(set(factors)) != len(factors):
        raise ToolError("GOLDEN_FIXTURE_SCALE_INVALID", repr(fixture_code_scale_factors))

    from collections import defaultdict

    layer_sources = {}
    graph_groups = defaultdict(list)
    source_hashes = {}
    for source in source_report.sources:
        if profile not in source.profiles:
            continue
        if source.model_info is None or source.source_name in source_hashes:
            raise ToolError("GOLDEN_SOURCE_MODEL_INFO_MISSING", source.source_name)
        source_path = Path(source.model_info.source_path)
        if not source_path.is_file() or _hash_file(source_path) != source.source_sha256:
            raise ToolError("GOLDEN_SOURCE_SHA_MISMATCH", source.source_name)
        source_hashes[source.source_name] = source.source_sha256
        for layer in source.layers:
            if layer.profile != profile or layer.layer_id not in selected_layers:
                continue
            if layer.layer_id in layer_sources:
                raise ToolError("GOLDEN_LAYER_DUPLICATE", str(layer.layer_id))
            layer_sources[layer.layer_id] = (source, layer)
            graph_groups[(source.source_name, layer.graph_index)].append((source, layer))
    missing_layers = sorted(selected_layers - set(layer_sources))
    if missing_layers:
        raise ToolError("GOLDEN_COVERAGE_INCOMPLETE", repr(missing_layers))

    declared_sources = {
        row.get("source"): row.get("source_sha256")
        for row in candidate_manifest.get("source_audit", {}).get("sources", [])
        if profile in row.get("profiles", [])
    }
    if declared_sources != source_hashes:
        raise ToolError(
            "GOLDEN_SOURCE_MAPPING_MISMATCH",
            repr({"manifest": declared_sources, "discovery": source_hashes}),
        )

    candidate_root = Path(candidate_root)
    model_rows = [
        row for row in candidate_manifest.get("models", [])
        if row.get("profile") == profile
    ]
    rows_by_layer_component_rank = {}
    for row in model_rows:
        key = (int(row.get("layer", -1)), row.get("component"), int(row.get("rank", -1)))
        if key in rows_by_layer_component_rank:
            raise ToolError("GOLDEN_COMPONENT_COVERAGE_INVALID", repr(key))
        rows_by_layer_component_rank[key] = row

    layer_reports = []
    fixture_hashes = {}
    candidate_hashes = {}
    mismatch_rows = []
    total_mismatch_count = 0
    boundary_count = 0
    backends = set()
    simulate_fp16_values = set()
    required_components = ("attn", "o", "suffix_attn", "mlp", "down", "suffix_final")

    for group in graph_groups.values():
        source, _ = group[0]
        graph_index = group[0][1].graph_index
        graph = source.model_info.graphs[graph_index]
        tensor_by_index = {tensor.index: tensor for tensor in graph.tensors}
        requested_names = []
        for _group_source, layer in group:
            for role in ("o", "down"):
                projection = layer.roles.get(role)
                if projection is None or len(projection.output_tensor_indices) != 1:
                    raise ToolError("GOLDEN_PROJECTION_MAPPING_UNRESOLVED", "{} layer {}".format(role, layer.layer_id))
                for tensor_index in (projection.activation_tensor_index, projection.output_tensor_indices[0]):
                    tensor = tensor_by_index.get(tensor_index)
                    if tensor is None:
                        raise ToolError("GOLDEN_PROJECTION_TENSOR_MISSING", "{} layer {}".format(role, layer.layer_id))
                    requested_names.append(tensor.name)
        requested_names = list(dict.fromkeys(requested_names))
        canonical_path = Path(source.model_info.source_path)
        execution_path = Path((source_execution_paths or {}).get(source.source_name, canonical_path))
        if not execution_path.is_file():
            raise ToolError("GOLDEN_SOURCE_FILE_MISSING", str(execution_path))
        group_layers = sorted(layer.layer_id for _group_source, layer in group)
        group_fixture_hashes = {}

        for scale_factor in factors:
            fixtures = make_deterministic_tflite_inputs(
                graph,
                seed=seed + min(group_layers) * 104729,
            )
            clipped_counts = {}
            if scale_factor != 1:
                for name, value in list(fixtures.items()):
                    lower_name = name.lower()
                    if any(marker in lower_name for marker in ("mask", "pos_emb", "position")):
                        continue
                    if np.issubdtype(value.dtype, np.integer):
                        limits = np.iinfo(value.dtype)
                        scaled = value.astype(np.int64) * scale_factor
                        clipped_counts[name] = int(np.count_nonzero((scaled < limits.min) | (scaled > limits.max)))
                        fixtures[name] = np.clip(scaled, limits.min, limits.max).astype(value.dtype)
            source_result = _run_named(runner, execution_path, fixtures, requested_names)
            if source_result.get("backend") is not None:
                backends.add(source_result["backend"])
            if "simulate_fp16" in source_result:
                simulate_fp16_values.add(bool(source_result["simulate_fp16"]))
            canonical_sha = source.source_sha256
            execution_sha = source_result.get("source_sha256")
            if not isinstance(execution_sha, str) or len(execution_sha) != 64:
                raise ToolError("GOLDEN_REPORT_INVALID", "source execution result has no SHA256")
            fixture_hash = source_result.get("fixture_sha256")
            if not isinstance(fixture_hash, str) or len(fixture_hash) != 64:
                raise ToolError("GOLDEN_REPORT_INVALID", "source execution result has no fixture SHA256")
            source_outputs = source_result.get("outputs")
            if not isinstance(source_outputs, dict) or set(requested_names) - set(source_outputs):
                raise ToolError("GOLDEN_SOURCE_BOUNDARY_OUTPUT_MISSING", repr(requested_names))

            for _group_source, layer in group:
                layer_fixture_key = "{}:{}".format(layer.layer_id, scale_factor)
                fixture_hashes[layer_fixture_key] = fixture_hash
                group_fixture_hashes[str(scale_factor)] = fixture_hash
                layer_report = {
                    "layer": layer.layer_id,
                    "source": source.source_name,
                    "canonical_source_sha256": canonical_sha,
                    "execution_source_sha256": execution_sha,
                    "fixture_sha256": fixture_hash,
                    "fixture_code_scale_factor": scale_factor,
                    "fixture_clipped_values_by_input": clipped_counts,
                    "row_reduce_boundaries": {},
                }
                for role in ("o", "down"):
                    projection = layer.roles[role]
                    source_input = tensor_by_index[projection.activation_tensor_index]
                    source_output = tensor_by_index[projection.output_tensor_indices[0]]
                    full_input = np.asarray(source_outputs[source_input.name])
                    expected = np.asarray(source_outputs[source_output.name])
                    if full_input.dtype.name.upper() != source_input.dtype or expected.dtype.name.upper() != source_output.dtype:
                        raise ToolError("GOLDEN_SOURCE_DTYPE_MISMATCH", "{} layer {}".format(role, layer.layer_id))
                    partial_outputs = []
                    partial_tensors = []
                    rank_results = []
                    for rank in range(2):
                        row = rows_by_layer_component_rank.get((layer.layer_id, role, rank))
                        if row is None:
                            raise ToolError("GOLDEN_COMPONENT_COVERAGE_INVALID", "{} layer {} rank {}".format(role, layer.layer_id, rank))
                        if str(row.get("source_sha256", "")).lower() != canonical_sha:
                            raise ToolError("GOLDEN_CANDIDATE_SOURCE_SHA_MISMATCH", str(row.get("path")))
                        ports_in = row.get("io", {}).get("inputs", [])
                        ports_out = row.get("io", {}).get("outputs", [])
                        if len(ports_in) != 1 or len(ports_out) != 1:
                            raise ToolError("GOLDEN_COMPONENT_IO_UNSUPPORTED", str(row.get("path")))
                        input_port, output_port = ports_in[0], ports_out[0]
                        input_value = _slice_input_for_rank(
                            full_input,
                            input_port.get("shape", []),
                            rank,
                            2,
                            source_input.name,
                        )
                        if input_port.get("name") != source_input.name or input_port.get("dtype") != source_input.dtype:
                            raise ToolError("GOLDEN_CANDIDATE_INPUT_CONTRACT_MISMATCH", str(row.get("path")))
                        if output_port.get("name") != source_output.name or tuple(output_port.get("shape", [])) != tuple(expected.shape):
                            raise ToolError("GOLDEN_CANDIDATE_OUTPUT_CONTRACT_MISMATCH", str(row.get("path")))

                        model_path = Path(
                            (candidate_model_paths or {}).get(
                                row["path"], candidate_root / row["path"]
                            )
                        )
                        if not model_path.is_file():
                            raise ToolError("GOLDEN_CANDIDATE_MISSING", str(model_path))
                        output_hash = _hash_file(model_path)
                        if output_hash != str(row.get("output_sha256", "")).lower():
                            raise ToolError("GOLDEN_CANDIDATE_SHA_MISMATCH", str(row["path"]))
                        candidate_hashes[row["path"]] = output_hash
                        candidate_info = read_tflite(model_path)
                        if len(candidate_info.graphs) != 1:
                            raise ToolError("GOLDEN_CANDIDATE_GRAPH_UNSUPPORTED", str(row["path"]))
                        candidate_graph = candidate_info.graphs[0]
                        candidate_inputs = [candidate_graph.tensors[index] for index in candidate_graph.inputs]
                        candidate_outputs = [candidate_graph.tensors[index] for index in candidate_graph.outputs]
                        if len(candidate_inputs) != 1 or len(candidate_outputs) != 1:
                            raise ToolError("GOLDEN_CANDIDATE_IO_INVALID", str(row["path"]))
                        candidate_input, candidate_output = candidate_inputs[0], candidate_outputs[0]
                        if (
                            candidate_input.name != source_input.name
                            or tuple(candidate_input.shape) != tuple(input_value.shape)
                            or candidate_input.dtype != source_input.dtype
                            or candidate_input.quantization != source_input.quantization
                            or candidate_output.name != source_output.name
                            or tuple(candidate_output.shape) != tuple(expected.shape)
                            or candidate_output.dtype not in ("INT16", "INT32")
                        ):
                            raise ToolError("GOLDEN_CANDIDATE_TENSOR_CONTRACT_MISMATCH", str(row["path"]))
                        result = _run_named(
                            runner,
                            model_path,
                            {input_port["name"]: input_value},
                            [output_port["name"]],
                        )
                        if result.get("source_sha256") != output_hash:
                            raise ToolError("GOLDEN_CANDIDATE_EXECUTION_SHA_MISMATCH", str(row["path"]))
                        if result.get("backend") is not None:
                            backends.add(result["backend"])
                        if "simulate_fp16" in result:
                            simulate_fp16_values.add(bool(result["simulate_fp16"]))
                        values = result.get("outputs")
                        if not isinstance(values, dict) or output_port["name"] not in values:
                            raise ToolError("GOLDEN_COMPONENT_OUTPUT_MAPPING_INVALID", str(row["path"]))
                        actual = np.asarray(values[output_port["name"]])
                        expected_dtype = {"INT16": np.dtype(np.int16), "INT32": np.dtype(np.int32)}[candidate_output.dtype]
                        if actual.shape != expected.shape or actual.dtype != expected_dtype:
                            raise ToolError("GOLDEN_CANDIDATE_OUTPUT_ARRAY_INVALID", str(row["path"]))
                        dtype_limits = np.iinfo(actual.dtype)
                        partial_outputs.append(actual)
                        partial_tensors.append(candidate_output)
                        rank_results.append({
                            "rank": rank,
                            "candidate_sha256": output_hash,
                            "input_shape": list(input_value.shape),
                            "input_sha256": hashlib.sha256(np.ascontiguousarray(input_value).tobytes()).hexdigest(),
                            "output_shape": list(actual.shape),
                            "output_dtype": str(actual.dtype),
                            "output_min": int(actual.min()),
                            "output_max": int(actual.max()),
                            "output_dtype_min": int(dtype_limits.min),
                            "output_dtype_max": int(dtype_limits.max),
                            "values_at_dtype_min": int(np.count_nonzero(actual == dtype_limits.min)),
                            "values_at_dtype_max": int(np.count_nonzero(actual == dtype_limits.max)),
                        })
                    reduced = _tp2_reduce_quantized_partials(partial_outputs, partial_tensors, source_output, role)
                    compared = compare_named_outputs({source_output.name: expected}, {source_output.name: reduced})
                    boundary_count += 1
                    total_mismatch_count += compared["mismatch_count"]
                    boundary = {
                        "stage": "{}_ROW_REDUCE".format(role.upper()),
                        "source_input_tensor": source_input.name,
                        "source_output_tensor": source_output.name,
                        "input_shape": list(full_input.shape),
                        "output_shape": list(expected.shape),
                        "source_input_dtype": source_input.dtype,
                        "source_output_dtype": source_output.dtype,
                        "reduction_policy": "host_float64_scale_ratio_round_to_even_saturate",
                        "source_output_quantization": {
                            "scales": list(source_output.quantization.scales) if source_output.quantization else [],
                            "zero_points": list(source_output.quantization.zero_points) if source_output.quantization else [],
                        },
                        "rank_partials": rank_results,
                        "partial_range_diagnostic": _tp2_partial_range_diagnostic(rank_results),
                        "mismatch_count": compared["mismatch_count"],
                        "max_absolute_difference": compared["max_absolute_difference"],
                        "mismatches": compared["mismatches"],
                    }
                    layer_report["row_reduce_boundaries"][role] = boundary
                    mismatch_rows.extend({
                        "topology": "TP2_ROW_REDUCE",
                        "source": source.source_name,
                        "source_sha256": canonical_sha,
                        "layer": layer.layer_id,
                        "rank": None,
                        "stage": boundary["stage"],
                        "operator": role,
                        "tensor_name": source_output.name,
                        "partial_range_diagnostic": boundary["partial_range_diagnostic"],
                        **mismatch,
                    } for mismatch in compared["mismatches"])
                layer_reports.append(layer_report)

    layer_reports.sort(key=lambda item: (item["layer"], item["fixture_code_scale_factor"]))
    checked_layers = sorted({item["layer"] for item in layer_reports})
    numerical_status = "NUMERICAL_FAIL" if total_mismatch_count else "NUMERICAL_INCOMPLETE"
    first_failure = "TP2_ROW_REDUCE_MISMATCH" if total_mismatch_count else "GOLDEN_COVERAGE_INCOMPLETE"
    return {
        "schema": 1,
        "profile": profile,
        "topology": "TP2_ROW_REDUCE",
        "backend": next(iter(backends)) if len(backends) == 1 else sorted(backends) if backends else None,
        "simulate_fp16": (
            next(iter(simulate_fp16_values))
            if len(simulate_fp16_values) == 1
            else sorted(simulate_fp16_values)
            if simulate_fp16_values
            else None
        ),
        "seed": seed,
        "fixture_code_scale_factors": list(factors),
        "required_stages": ["O_ROW_REDUCE", "DOWN_ROW_REDUCE"],
        "source_sha256_by_name": source_hashes,
        "fixture_sha256_by_layer": fixture_hashes,
        "candidate_sha256": candidate_hashes,
        "layer_reports": layer_reports,
        "golden_coverage": {
            "complete": False,
            "layers": checked_layers,
            "selected_layers": sorted(selected_layers),
            "expected_layers": expected_layers,
            "ranks": [0, 1],
            "checked_components": ["o", "down"],
            "required_components": list(required_components),
            "cross_layer_candidate_chain": False,
            "checked_boundary_count": boundary_count,
            "expected_row_reduce_boundary_count": expected_layers * 2 * len(factors),
        },
        "row_reduce_semantics_status": "UNPROVEN",
        "mismatch_count": total_mismatch_count,
        "mismatches": mismatch_rows,
        "numerical_status": numerical_status,
        "first_failure": first_failure,
        "release_eligible": False,
    }


def _tp4_attention_intermediate_inputs(candidate_manifest, profile, layer_ids, source_graph):
    """Find canonical internal tensors needed to seed rank-local attention."""
    tensor_indices = {tensor.name: tensor.index for tensor in source_graph.tensors}
    public_inputs = set(source_graph.inputs)
    names = set()
    layer_ids = set(layer_ids)
    for row in candidate_manifest.get("models", []):
        if row.get("profile") != profile or row.get("layer") not in layer_ids:
            continue
        if row.get("component") != "attn":
            continue
        for port in row.get("io", {}).get("inputs", []):
            name = port.get("name")
            tensor_index = tensor_indices.get(name)
            if tensor_index is not None and tensor_index not in public_inputs:
                names.add(name)
    return sorted(names)


def _tp4_candidate_attention_input_overrides(
    candidate_manifest,
    profile,
    layer_id,
    source_graph,
    graph_layers,
    graph_input_hidden_name,
    hidden_input_index,
    candidate_hidden,
    fixtures,
    source_path,
    runner,
):
    """Recompute rank-local attention inputs from a chained candidate hidden state.

    Some canonical layer files expose the raw INT16 hidden input as a graph
    input, then dequantize it to the FLOAT32 tensor consumed by the attention
    component. In that layout, passing the preceding INT16 candidate output
    directly to the FLOAT32 component input is incorrect. Re-run the canonical
    layer with the candidate hidden input to derive those internal activations.
    """
    graph_input_indices = set(source_graph.inputs)
    if len(graph_layers) != 1 or hidden_input_index in graph_input_indices:
        return {}, None
    graph_input_index = next(
        (
            index
            for index in graph_input_indices
            if source_graph.tensors[index].name == graph_input_hidden_name
        ),
        None,
    )
    if graph_input_index is None:
        return {}, None
    attention_inputs = _tp4_attention_intermediate_inputs(
        candidate_manifest,
        profile,
        {layer_id},
        source_graph,
    )
    if not attention_inputs:
        return {}, None
    import numpy as np

    candidate_hidden = np.asarray(candidate_hidden)
    graph_input_shape = tuple(source_graph.tensors[graph_input_index].shape)
    if tuple(candidate_hidden.shape) != graph_input_shape:
        raise ToolError(
            "GOLDEN_CANDIDATE_LAYER_CHAIN_SHAPE_MISMATCH",
            "layer {} graph input {}: {} -> {}".format(
                layer_id,
                graph_input_hidden_name,
                tuple(candidate_hidden.shape),
                graph_input_shape,
            ),
        )
    candidate_fixtures = dict(fixtures)
    candidate_fixtures[graph_input_hidden_name] = candidate_hidden
    candidate_preprocess = _run_named(
        runner,
        source_path,
        candidate_fixtures,
        attention_inputs,
    )
    derived_inputs = candidate_preprocess.get("outputs", {})
    if not isinstance(derived_inputs, dict) or set(derived_inputs) != set(attention_inputs):
        raise ToolError(
            "GOLDEN_CANDIDATE_INPUT_PREPROCESS_INVALID",
            "layer {} did not produce all candidate attention inputs".format(layer_id),
        )
    return derived_inputs, candidate_preprocess.get("fixture_sha256")


def run_canonical_golden(source_model, fixtures, backend_runner=None):
    """Run only a registered exact backend; never substitute NumPy arithmetic."""
    if not callable(backend_runner):
        raise ToolError(
            "GOLDEN_EXECUTION_UNAVAILABLE",
            "No verified host runner is registered for canonical TFLite execution.",
        )
    result = backend_runner(source_model=source_model, fixtures=fixtures)
    if not isinstance(result, dict):
        raise ToolError("GOLDEN_REPORT_INVALID", "backend runner did not return a report object")
    if "mismatch_count" not in result or not isinstance(result["mismatch_count"], int):
        raise ToolError("GOLDEN_REPORT_INVALID", "backend report must include an integer mismatch_count")
    if not result.get("source_sha256") or not result.get("fixture_sha256"):
        raise ToolError("GOLDEN_REPORT_INVALID", "backend report must bind source and fixtures by SHA256")
    if result["mismatch_count"] == 0:
        result["numerical_status"] = "NUMERICAL_PASS"
    else:
        result["numerical_status"] = "NUMERICAL_FAIL"
    return result
