import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from tflite_tp.errors import ToolError  # noqa: E402
from tflite_tp.model import (  # noqa: E402
    EXPECTED_WEIGHT_SHAPES,
    PROJECTION_ROLES,
    LayerDiscovery,
    ProjectionMatch,
    QuantizationInfo,
    TensorInfo,
)
from tflite_tp.packed_int4 import (  # noqa: E402
    reconstruct_packed_int4,
    slice_packed_int4,
    slice_quantization,
)
from tflite_tp.discovery import discover_models  # noqa: E402
from tflite_tp.partition import derive_tp4_components  # noqa: E402
from tflite_tp.planning import make_partition_plan  # noqa: E402
from tflite_tp.tflite_rewriter import _tp2_row_reduce_semantics  # noqa: E402


def pack_values(values):
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if values.size % 2:
        values = np.pad(values, (0, 1), mode="constant")
    return (values[0::2] | (values[1::2] << 4)).tobytes()


def decode_values(payload, count):
    raw = np.frombuffer(payload, dtype=np.uint8)
    values = np.empty(raw.size * 2, dtype=np.uint8)
    values[0::2] = raw & 0x0F
    values[1::2] = raw >> 4
    return values[:count]


def complete_layer(profile="D1"):
    layer = LayerDiscovery(
        layer_id=4,
        graph_index=0,
        profile=profile,
        source_name="layer04_1t1024c.tflite",
        source_sha256="a" * 64,
    )
    for index, role in enumerate(PROJECTION_ROLES):
        layer.roles[role] = ProjectionMatch(
            role=role,
            tensor_index=20 + index,
            tensor_name="layers/4/{}.weight".format(role),
            shape=EXPECTED_WEIGHT_SHAPES[role],
            dtype="INT4",
            operator_index=40 + index,
            operator="FULLY_CONNECTED",
            activation_tensor_index=index,
            output_tensor_indices=(60 + index,),
            bias_tensor_index=80 + index,
            quantization=QuantizationInfo(scales=(0.01,), zero_points=(0,), quantized_dimension=0),
            semantic_layer_id=4,
        )
    return layer


class PackedInt4Tests(unittest.TestCase):
    def test_slice_and_reconstruct_odd_axis_shards(self):
        source = np.arange(15, dtype=np.uint8).reshape(3, 5) % 16
        payload = pack_values(source)
        pieces = []
        for start, count in ((0, 1), (1, 2), (3, 2)):
            piece = slice_packed_int4(payload, source.shape, axis=1, start=start, count=count)
            pieces.append((piece, (3, count)))
            np.testing.assert_array_equal(decode_values(piece, 3 * count), source[:, start : start + count].reshape(-1))
        rebuilt = reconstruct_packed_int4(pieces, source.shape, axis=1)
        np.testing.assert_array_equal(decode_values(rebuilt, source.size), source.reshape(-1))

    def test_per_group_quantization_slices_only_whole_groups(self):
        quant = QuantizationInfo(
            scales=(0.1, 0.2, 0.3, 0.4),
            zero_points=(0, 1, 2, 3),
            quantized_dimension=0,
        )
        sliced = slice_quantization(quant, (8, 4), axis=0, start=2, count=4)
        self.assertEqual(sliced.scales, (0.2, 0.3))
        self.assertEqual(sliced.zero_points, (1, 2))
        with self.assertRaises(ToolError) as raised:
            slice_quantization(quant, (8, 4), axis=0, start=1, count=2)
        self.assertEqual(raised.exception.code, "QUANT_GROUP_BOUNDARY_SPLIT")

    def test_malformed_payload_fails_closed(self):
        with self.assertRaises(ToolError) as raised:
            slice_packed_int4(b"\x00", (2, 2), axis=0, start=0, count=1)
        self.assertEqual(raised.exception.code, "INT4_BUFFER_LENGTH_MISMATCH")

    def test_nonzero_unused_padding_nibble_fails_closed(self):
        with self.assertRaises(ToolError) as raised:
            slice_packed_int4(b"\x21\xF3", (1, 3), axis=1, start=0, count=2)
        self.assertEqual(raised.exception.code, "INT4_PADDING_NIBBLE_NONZERO")

    def test_tflite_low_nibble_first_slice_preserves_signed_bit_patterns(self):
        source = b"\xF1"  # logical INT4 values +1, -1 (two's-complement nibble 0xF)
        low = slice_packed_int4(source, (1, 2), axis=1, start=0, count=1)
        high = slice_packed_int4(source, (1, 2), axis=1, start=1, count=1)
        self.assertEqual(low, b"\x01")
        self.assertEqual(high, b"\x0F")
        rebuilt = reconstruct_packed_int4([(low, (1, 1)), (high, (1, 1))], (1, 2), axis=1)
        self.assertEqual(rebuilt, source)

    def test_quant_metadata_lengths_are_checked_even_when_axis_is_unchanged(self):
        quant = QuantizationInfo(
            scales=(0.1,),
            zero_points=(0, 1),
            quantized_dimension=0,
        )
        with self.assertRaises(ToolError) as raised:
            slice_quantization(quant, (4, 8), axis=1, start=0, count=4)
        self.assertEqual(raised.exception.code, "QUANT_ZERO_POINT_COUNT_INVALID")

    def test_quantization_range_metadata_length_error_is_not_mislabeled_zero_point(self):
        quant = QuantizationInfo(
            scales=(0.1,),
            minimum=(-8, -7),
            quantized_dimension=0,
        )
        with self.assertRaises(ToolError) as raised:
            slice_quantization(quant, (4, 8), axis=1, start=0, count=4)
        self.assertEqual(raised.exception.code, "QUANT_RANGE_METADATA_LENGTH_MISMATCH")


class PartitionPlanningTests(unittest.TestCase):
    def test_tp4_components_are_derived_from_dependencies_not_operator_ranges(self):
        from test_discovery import synthetic_transformer_model

        model = synthetic_transformer_model(reverse_operator_storage=True)
        report = discover_models([(model, "layer00_1t1024c.tflite")], expected_layers=1)
        layer = report.sources[0].layers[0]
        analysis = report.sources[0].graph_analyses[0]
        components = derive_tp4_components(model, layer, analysis)
        self.assertEqual([component.name for component in components], [
            "attn", "o", "suffix_attn", "mlp", "down", "suffix_final"
        ])
        self.assertEqual(sum(len(component.operator_indices) for component in components), len(model.graphs[0].operators))
        self.assertEqual(len(set(index for component in components for index in component.operator_indices)), len(model.graphs[0].operators))
        self.assertEqual(
            set(components[0].output_tensor_indices),
            {layer.roles["o"].activation_tensor_index}
            | set(analysis.output_roles["current_key"])
            | set(analysis.output_roles["current_value"]),
        )
        self.assertEqual(components[1].input_tensor_indices, (layer.roles["o"].activation_tensor_index,))

    def test_grouped_graph_components_are_scoped_to_each_semantic_layer(self):
        from test_discovery import synthetic_grouped_transformer_model

        model = synthetic_grouped_transformer_model()
        report = discover_models([(model, "grouped_1t.tflite")], expected_layers=2)
        source = report.sources[0]
        self.assertEqual([layer.layer_id for layer in source.layers], [0, 1])
        self.assertTrue(all(layer.complete for layer in source.layers))
        self.assertTrue(all("GRAPH_FEATURE_UNRESOLVED" not in [code for code, _ in source.graph_analyses[i].issues]
                            for i in range(2)))

        layer_operator_sets = []
        for layer, analysis in zip(source.layers, source.graph_analyses):
            components = derive_tp4_components(model, layer, analysis)
            self.assertEqual([component.name for component in components], [
                "attn", "o", "suffix_attn", "mlp", "down", "suffix_final"
            ])
            layer_operator_sets.append(set(
                index for component in components for index in component.operator_indices
            ))

        self.assertFalse(layer_operator_sets[0] & layer_operator_sets[1])
        self.assertEqual(len(layer_operator_sets[0] | layer_operator_sets[1]), len(model.graphs[0].operators))

    def test_tp2_uses_row_reduce_for_o_and_down(self):
        plan = make_partition_plan(complete_layer(), degree=2).to_dict()
        roles = {item["role"]: item for item in plan["projections"]}
        self.assertEqual(plan["topology"], "TP2_ROW_REDUCE")
        self.assertEqual(roles["q"]["slice_axis"], 0)
        self.assertEqual(roles["o"]["slice_axis"], 1)
        self.assertEqual(roles["o"]["bias_policy"], "RANK0_ONLY")
        self.assertEqual(roles["down"]["slice_axis"], 1)
        self.assertEqual(roles["down"]["bias_policy"], "RANK0_ONLY")
        self.assertEqual(len(plan["collective_contract"]["row_reduce_boundaries"]), 2)
        self.assertEqual(plan["collective_contract"]["all_gather_boundaries"], [])
        self.assertEqual(plan["release_status"], "NOT_ELIGIBLE")

    def test_tp2_manifest_records_observed_partial_contract_without_claiming_proof(self):
        quantization = QuantizationInfo(
            scales=(0.125, 0.25),
            zero_points=(0, 0),
            quantized_dimension=2,
        )
        graph = SimpleNamespace(tensors=(
            TensorInfo(10, "o_partial", (1, 1, 4096), (1, 1, 4096), "INT16", 0, 0, quantization),
            TensorInfo(20, "down_partial", (1, 1, 4096), (1, 1, 4096), "INT16", 0, 0, quantization),
        ))
        layer = SimpleNamespace(roles={
            "o": SimpleNamespace(output_tensor_indices=(0,)),
            "down": SimpleNamespace(output_tensor_indices=(1,)),
        })

        semantics = _tp2_row_reduce_semantics(graph, layer)

        self.assertEqual(semantics["status"], "UNPROVEN")
        self.assertIsNone(semantics["proof_sha256"])
        self.assertEqual(semantics["partial_output_dtype"], {"o": "INT16", "down": "INT16"})
        self.assertEqual(semantics["partial_outputs"]["o"]["tensor_name"], "o_partial")
        self.assertEqual(semantics["partial_outputs"]["o"]["quantization"]["scale_count"], 2)
        self.assertEqual(semantics["accumulator_width"], "UNPROVEN")
        self.assertIn("O=INT16, DOWN=INT16", semantics["reason"])
        self.assertNotIn("FLOAT16", semantics["reason"])

    def test_tp4_uses_four_ordered_all_gather_stages(self):
        plan = make_partition_plan(complete_layer("P128"), degree=4).to_dict()
        self.assertEqual(plan["topology"], "TP4_FOUR_STAGE_ALL_GATHER")
        self.assertTrue(all(item["slice_axis"] == 0 for item in plan["projections"]))
        self.assertEqual(
            [item["stage"] for item in plan["collective_contract"]["all_gather_boundaries"]],
            ["AG0", "AG1", "AG2", "AG3"],
        )
        self.assertEqual(plan["collective_contract"]["rank_order"], [0, 1, 2, 3])

    def test_incomplete_layer_cannot_be_planned(self):
        layer = complete_layer()
        del layer.roles["down"]
        with self.assertRaises(ToolError) as raised:
            make_partition_plan(layer, degree=2)
        self.assertEqual(raised.exception.code, "LAYER_DISCOVERY_INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
