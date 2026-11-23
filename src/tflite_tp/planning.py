"""Automatic rank-slice plans for the distinct TP2 and TP4 contracts."""

from dataclasses import dataclass
from typing import Dict, List

from .errors import ToolError
from .model import PROJECTION_ROLES, LayerDiscovery, profile_token_count


@dataclass(frozen=True)
class PartitionPlan:
    topology: str
    degree: int
    layer: int
    profile: str
    source_name: str
    source_sha256: str
    collective_contract: Dict[str, object]
    head_geometry: Dict[str, object]
    projections: List[Dict[str, object]]

    def to_dict(self):
        return {
            "schema": 1,
            "topology": self.topology,
            "tp_degree": self.degree,
            "layer": self.layer,
            "profile": self.profile,
            "source": self.source_name,
            "source_sha256": self.source_sha256,
            "head_geometry": self.head_geometry,
            "collective_contract": self.collective_contract,
            "projections": self.projections,
            "structural_status": "PLAN_ONLY",
            "numerical_status": "NOT_RUN",
            "release_status": "NOT_ELIGIBLE",
        }


def _collective_contract(degree, tokens, hidden_size, intermediate_size, q_width, topology=None):
    topology = resolve_topology(degree, topology)
    if topology == "row-reduce":
        return {
            "architecture": "ROW_REDUCE",
            "rank_order": [0, 1],
            "all_gather_boundaries": [],
            "row_reduce_boundaries": [
                {"name": "O", "operation": "ALL_REDUCE_SUM", "shape": [1, tokens, hidden_size]},
                {"name": "Down", "operation": "ALL_REDUCE_SUM", "shape": [1, tokens, hidden_size]},
            ],
            "residual_policy": "ADD_ONCE_AFTER_EACH_REDUCE",
            "kv_cache_policy": "RANK_LOCAL",
            "numerical_policy": "BIT_EXACT_ROW_REDUCE_NOT_PROVEN",
        }
    return {
        "architecture": "FOUR_STAGE_ALL_GATHER",
        "rank_order": list(range(degree)),
        "all_gather_boundaries": [
            {"stage": "AG0", "tensor": "attention_context", "local_channels": q_width // degree, "gathered_channels": q_width},
            {"stage": "AG1", "tensor": "o_projection", "local_channels": hidden_size // degree, "gathered_channels": hidden_size},
            {"stage": "AG2", "tensor": "swiglu", "local_channels": intermediate_size // degree, "gathered_channels": intermediate_size},
            {"stage": "AG3", "tensor": "down_projection", "local_channels": hidden_size // degree, "gathered_channels": hidden_size},
        ],
        "row_reduce_boundaries": [],
        "residual_policy": "ADD_ONCE_AT_DOCUMENTED_BOUNDARIES",
        "kv_cache_policy": "RANK_LOCAL",
        "numerical_policy": "REQUIRES_HOST_GOLDEN",
    }


def resolve_topology(degree, topology=None):
    if degree not in (2, 4, 8):
        raise ToolError("TP_DEGREE_UNSUPPORTED", "supported tensor-parallel degrees are 2, 4 and 8")
    topology = topology or "all-gather"
    if topology != "all-gather":
        raise ToolError("TOPOLOGY_UNSUPPORTED", "only all-gather is supported")
    return topology


# One artifact version for every build: six components per rank and layer, with the
# byte-identical suffix_attn/suffix_final compiled once per layer and shared by all ranks.
ARTIFACT_VERSION = "genio-tp-allgather-v1"
ARTIFACT_LAYOUT = "shared-suffix"
DEPRECATED_LAYOUT_ALIASES = ("legacy18", "rank-local24")


def resolve_artifact_layout(layout=None):
    """Return (layout, deprecated_alias_or_None); every accepted name maps to the single layout."""
    if layout in (None, ARTIFACT_LAYOUT):
        return ARTIFACT_LAYOUT, None
    if layout in DEPRECATED_LAYOUT_ALIASES:
        return ARTIFACT_LAYOUT, layout
    raise ToolError("ARTIFACT_LAYOUT_UNSUPPORTED", "{} (supported: {})".format(layout, ARTIFACT_LAYOUT))


# Q/K/V 2x output-width padding has Host/NPU evidence only for these fixed-token profiles.
VERIFIED_QKV_PADDING_PROFILES = ("P128", "D1")
QKV_PADDING_DEGREES = (4, 8)


def resolve_qkv_padding(degree, profile, requested=None, structurally_supported=True):
    """Single policy for the TP4/TP8 Q/K/V width padding, shared by every build entry point.

    requested=None pads only verified profiles; 2 is an explicit opt-in (any profile whose
    Q/K/V FCs support the rewrite); 0 disables padding. Returns (multiplier or None, decision).
    """
    if requested not in (None, 0, 2):
        raise ToolError(
            "FC_WIDTH_PADDING_MULTIPLIER_UNVERIFIED",
            "only 2 or 0 (explicitly disabled for diagnostics) is accepted",
        )
    verified = profile in VERIFIED_QKV_PADDING_PROFILES
    multiplier = None
    if degree not in QKV_PADDING_DEGREES:
        decision = "DEGREE_NOT_PADDED"
    elif requested == 0:
        decision = "EXPLICIT_DISABLED"
    elif requested == 2:
        if not structurally_supported:
            raise ToolError(
                "FC_WIDTH_PADDING_STRUCTURE_UNSUPPORTED",
                "TP{} {} Q/K/V projections do not support width padding".format(degree, profile),
            )
        multiplier, decision = 2, "EXPLICIT_OPT_IN"
    elif not verified:
        decision = "AUTO_UNVERIFIED_PROFILE"
    elif not structurally_supported:
        decision = "AUTO_STRUCTURE_UNSUPPORTED"
    else:
        multiplier, decision = 2, "AUTO_VERIFIED_PROFILE"
    return multiplier, {
        "degree": degree,
        "profile": profile,
        "requested": requested,
        "multiplier": multiplier,
        "decision": decision,
        "verified_profile": verified,
        "verified_profiles": list(VERIFIED_QKV_PADDING_PROFILES),
    }


def qkv_padding_status(degree, profile, multiplier):
    """Manifest status naming the profile the padding was actually applied to."""
    if multiplier is None:
        return "DISABLED"
    prefix = "APPLIED" if degree == 4 else "CANDIDATE_APPLIED"
    suffix = "" if profile in VERIFIED_QKV_PADDING_PROFILES else "_UNVERIFIED_PROFILE"
    return "{}_TO_TP{}_{}_QKV{}".format(prefix, degree, profile, suffix)


def make_partition_plan(layer, degree, topology=None):
    """Build a tensor-ID-free partition plan from automatically discovered roles."""
    topology = resolve_topology(degree, topology)
    if not layer.complete:
        raise ToolError("LAYER_DISCOVERY_INCOMPLETE", "layer {} has unresolved or incomplete projection roles".format(layer.layer_id))
    if set(layer.roles) != set(PROJECTION_ROLES):
        raise ToolError("PROJECTION_ROLE_SET_INVALID", "layer {} does not contain exactly seven projections".format(layer.layer_id))

    tokens = profile_token_count(layer.profile)
    if tokens is None:
        raise ToolError("PROFILE_UNSUPPORTED", "profile {} has no fixed token extent".format(layer.profile))

    shapes = {role: tuple(layer.roles[role].shape) for role in PROJECTION_ROLES}
    if any(len(shape) != 2 or min(shape) <= 0 for shape in shapes.values()):
        raise ToolError("PROJECTION_SHAPE_INVALID", "all projection weights must be positive rank-two matrices")
    q_width, hidden_size = shapes["q"]
    kv_width = shapes["k"][0]
    intermediate_size = shapes["gate"][0]
    expected_shapes = {
        "q": (q_width, hidden_size), "k": (kv_width, hidden_size), "v": (kv_width, hidden_size),
        "o": (hidden_size, q_width), "gate": (intermediate_size, hidden_size),
        "up": (intermediate_size, hidden_size), "down": (hidden_size, intermediate_size),
    }
    if shapes != expected_shapes:
        raise ToolError("PROJECTION_GEOMETRY_INVALID", "projection shapes do not form a consistent Llama block")
    if not all(value is not None and value > 0 for value in (layer.query_heads, layer.kv_heads, layer.head_dim)):
        raise ToolError("HEAD_GEOMETRY_UNRESOLVED", "query heads, KV heads and head dimension must resolve before partitioning")
    if (q_width != layer.query_heads * layer.head_dim
            or kv_width != layer.kv_heads * layer.head_dim
            or layer.query_heads % layer.kv_heads):
        raise ToolError("HEAD_GEOMETRY_INVALID", "projection widths do not match the discovered query/KV head geometry")
    if layer.query_heads % degree or layer.kv_heads % degree:
        raise ToolError("HEADS_NOT_DIVISIBLE", "{} query heads and {} KV heads cannot be evenly split over TP{}".format(
            layer.query_heads, layer.kv_heads, degree))

    projections = []
    for role in PROJECTION_ROLES:
        match = layer.roles[role]
        expected_shape = shapes[role]
        if match.dtype != "INT4":
            raise ToolError("PROJECTION_DTYPE_UNSUPPORTED", "{} is {}; expected INT4".format(role, match.dtype))
        if topology == "row-reduce" and role in ("o", "down"):
            axis = 1
            partition_kind = "INPUT_CHANNEL_ROW_REDUCE"
            bias_policy = "RANK0_ONLY"
        else:
            axis = 0
            partition_kind = "OUTPUT_CHANNEL_SHARD"
            bias_policy = "SLICE_WITH_OUTPUT_CHANNELS"

        extent = expected_shape[axis]
        if extent % degree:
            raise ToolError("PROJECTION_NOT_DIVISIBLE", "{} axis {} extent {} is not divisible by {}".format(role, axis, extent, degree))
        count = extent // degree
        shards = [
            {
                "rank": rank,
                "start": rank * count,
                "stop": (rank + 1) * count,
                "count": count,
            }
            for rank in range(degree)
        ]
        projections.append(
            {
                "role": role,
                "source_weight_tensor": match.tensor_index,
                "source_operator": match.operator_index,
                "source_shape": list(expected_shape),
                "slice_axis": axis,
                "slice_axis_name": "output_channels" if axis == 0 else "input_channels",
                "partition_kind": partition_kind,
                "bias_policy": bias_policy,
                "rank_shards": shards,
                "rank_weight_shape": [count if dim == axis else value for dim, value in enumerate(expected_shape)],
            }
        )

    topology_name = "TP2_ROW_REDUCE" if topology == "row-reduce" else "TP{}_FOUR_STAGE_ALL_GATHER".format(degree)
    return PartitionPlan(
        topology=topology_name,
        degree=degree,
        layer=layer.layer_id,
        profile=layer.profile,
        source_name=layer.source_name,
        source_sha256=layer.source_sha256,
        collective_contract=_collective_contract(degree, tokens, hidden_size, intermediate_size, q_width, topology),
        head_geometry={"query_heads": layer.query_heads, "kv_heads": layer.kv_heads,
                       "head_dim": layer.head_dim},
        projections=projections,
    )
