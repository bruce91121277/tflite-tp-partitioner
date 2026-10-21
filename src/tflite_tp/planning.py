"""Automatic rank-slice plans for the distinct TP2 and TP4 contracts."""

from dataclasses import dataclass
from typing import Dict, List

from .errors import ToolError
from .model import EXPECTED_WEIGHT_SHAPES, PROJECTION_ROLES, LayerDiscovery


@dataclass(frozen=True)
class PartitionPlan:
    topology: str
    degree: int
    layer: int
    profile: str
    source_name: str
    source_sha256: str
    collective_contract: Dict[str, object]
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
            "collective_contract": self.collective_contract,
            "projections": self.projections,
            "structural_status": "PLAN_ONLY",
            "numerical_status": "NOT_RUN",
            "release_status": "NOT_ELIGIBLE",
        }


def _collective_contract(degree, tokens):
    if degree == 2:
        return {
            "architecture": "ROW_REDUCE",
            "rank_order": [0, 1],
            "all_gather_boundaries": [],
            "row_reduce_boundaries": [
                {"name": "O", "operation": "ALL_REDUCE_SUM", "shape": [1, tokens, 4096]},
                {"name": "Down", "operation": "ALL_REDUCE_SUM", "shape": [1, tokens, 4096]},
            ],
            "residual_policy": "ADD_ONCE_AFTER_EACH_REDUCE",
            "kv_cache_policy": "RANK_LOCAL",
            "numerical_policy": "BIT_EXACT_ROW_REDUCE_NOT_PROVEN",
        }
    return {
        "architecture": "FOUR_STAGE_ALL_GATHER",
        "rank_order": [0, 1, 2, 3],
        "all_gather_boundaries": [
            {"stage": "AG0", "tensor": "attention_context", "local_channels": 1024, "gathered_channels": 4096},
            {"stage": "AG1", "tensor": "o_projection", "local_channels": 1024, "gathered_channels": 4096},
            {"stage": "AG2", "tensor": "swiglu", "local_channels": 3584, "gathered_channels": 14336},
            {"stage": "AG3", "tensor": "down_projection", "local_channels": 1024, "gathered_channels": 4096},
        ],
        "row_reduce_boundaries": [],
        "residual_policy": "ADD_ONCE_AT_DOCUMENTED_BOUNDARIES",
        "kv_cache_policy": "RANK_LOCAL",
        "numerical_policy": "REQUIRES_HOST_GOLDEN",
    }


def make_partition_plan(layer, degree):
    """Build a tensor-ID-free partition plan from automatically discovered roles."""
    if degree not in (2, 4):
        raise ToolError("TP_DEGREE_UNSUPPORTED", "supported tensor-parallel degrees are 2 and 4")
    if not layer.complete:
        raise ToolError("LAYER_DISCOVERY_INCOMPLETE", "layer {} has unresolved or incomplete projection roles".format(layer.layer_id))
    if set(layer.roles) != set(PROJECTION_ROLES):
        raise ToolError("PROJECTION_ROLE_SET_INVALID", "layer {} does not contain exactly seven projections".format(layer.layer_id))

    tokens = 128 if layer.profile == "P128" else 1 if layer.profile == "D1" else 0
    if not tokens:
        raise ToolError("PROFILE_UNSUPPORTED", "profile {} is not P128 or D1".format(layer.profile))

    projections = []
    for role in PROJECTION_ROLES:
        match = layer.roles[role]
        expected_shape = EXPECTED_WEIGHT_SHAPES[role]
        if match.shape != expected_shape:
            raise ToolError(
                "PROJECTION_SHAPE_UNSUPPORTED",
                "{} has shape {}; expected {}".format(role, match.shape, expected_shape),
            )
        if match.dtype != "INT4":
            raise ToolError("PROJECTION_DTYPE_UNSUPPORTED", "{} is {}; expected INT4".format(role, match.dtype))
        if degree == 2 and role in ("o", "down"):
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

    topology = "TP2_ROW_REDUCE" if degree == 2 else "TP4_FOUR_STAGE_ALL_GATHER"
    return PartitionPlan(
        topology=topology,
        degree=degree,
        layer=layer.layer_id,
        profile=layer.profile,
        source_name=layer.source_name,
        source_sha256=layer.source_sha256,
        collective_contract=_collective_contract(degree, tokens),
        projections=projections,
    )
