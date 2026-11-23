import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np



from tflite_tp.errors import ToolError  # noqa: E402
from tflite_tp.model import (  # noqa: E402
    PROJECTION_ROLES,
    LayerDiscovery,
    GraphInfo,
    ModelInfo,
    OperatorInfo,
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
from tflite_tp.partition import derive_tp4_components, derive_tp_components  # noqa: E402
from tflite_tp.planning import make_partition_plan  # noqa: E402
from tflite_tp.tflite_rewriter import _tp2_row_reduce_semantics  # noqa: E402
from tflite_tp_testdata import TEST_WEIGHT_SHAPES  # noqa: E402


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
        query_heads=32,
        kv_heads=8,
        head_dim=128,
    )
    for index, role in enumerate(PROJECTION_ROLES):
        layer.roles[role] = ProjectionMatch(
            role=role,
            tensor_index=20 + index,
            tensor_name="layers/4/{}.weight".format(role),
            shape=TEST_WEIGHT_SHAPES[role],
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


class QkvPaddingPolicyTests(unittest.TestCase):
    def test_auto_pads_only_verified_profiles(self):
        from tflite_tp.planning import resolve_qkv_padding
        cases = (
            (4, "P128", 2, "AUTO_VERIFIED_PROFILE"),
            (4, "D1", 2, "AUTO_VERIFIED_PROFILE"),
            (8, "P128", 2, "AUTO_VERIFIED_PROFILE"),
            (4, "P64", None, "AUTO_UNVERIFIED_PROFILE"),
            (4, "D32", None, "AUTO_UNVERIFIED_PROFILE"),
            (2, "P128", None, "DEGREE_NOT_PADDED"),
        )
        for degree, profile, multiplier, decision in cases:
            with self.subTest(degree=degree, profile=profile):
                chosen, record = resolve_qkv_padding(degree, profile)
                self.assertEqual(chosen, multiplier)
                self.assertEqual(record["decision"], decision)
                self.assertEqual(record["verified_profile"], profile in ("P128", "D1"))

    def test_explicit_requests(self):
        from tflite_tp.planning import resolve_qkv_padding
        self.assertEqual(resolve_qkv_padding(4, "P64", 2)[0], 2)
        self.assertEqual(resolve_qkv_padding(4, "P64", 2)[1]["decision"], "EXPLICIT_OPT_IN")
        self.assertEqual(resolve_qkv_padding(4, "P128", 0), (None, resolve_qkv_padding(4, "P128", 0)[1]))
        self.assertEqual(resolve_qkv_padding(4, "P128", 0)[1]["decision"], "EXPLICIT_DISABLED")
        self.assertIsNone(resolve_qkv_padding(4, "P128", structurally_supported=False)[0])
        with self.assertRaises(ToolError) as error:
            resolve_qkv_padding(4, "P64", 3)
        self.assertEqual(error.exception.code, "FC_WIDTH_PADDING_MULTIPLIER_UNVERIFIED")
        with self.assertRaises(ToolError) as error:
            resolve_qkv_padding(4, "P64", 2, structurally_supported=False)
        self.assertEqual(error.exception.code, "FC_WIDTH_PADDING_STRUCTURE_UNSUPPORTED")

    def test_manifest_status_names_the_real_profile(self):
        from tflite_tp.planning import qkv_padding_status
        self.assertEqual(qkv_padding_status(4, "P128", 2), "APPLIED_TO_TP4_P128_QKV")
        self.assertEqual(qkv_padding_status(4, "P64", 2), "APPLIED_TO_TP4_P64_QKV_UNVERIFIED_PROFILE")
        self.assertEqual(qkv_padding_status(8, "D1", 2), "CANDIDATE_APPLIED_TO_TP8_D1_QKV")
        self.assertEqual(qkv_padding_status(4, "P64", None), "DISABLED")


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
    def test_attention_gather_uses_query_width_when_hidden_differs(self):
        layer = complete_layer()
        shapes = {
            'q': (2048, 4096), 'k': (1024, 4096), 'v': (1024, 4096),
            'o': (4096, 2048), 'gate': (14336, 4096),
            'up': (14336, 4096), 'down': (4096, 14336),
        }
        layer.query_heads = 16
        for role, shape in shapes.items():
            layer.roles[role] = replace(layer.roles[role], shape=shape)
        for degree in (4, 8):
            plan = make_partition_plan(layer, degree)
            attention, projection = plan.collective_contract['all_gather_boundaries'][:2]
            self.assertEqual((attention['local_channels'], attention['gathered_channels']),
                             (2048 // degree, 2048))
            self.assertEqual((projection['local_channels'], projection['gathered_channels']),
                             (4096 // degree, 4096))

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

    def test_tp8_attention_keeps_int16_input_dequantize_on_device(self):
        from test_discovery import synthetic_transformer_model

        model = synthetic_transformer_model()
        graph = model.graphs[0]
        hidden = graph.inputs[0]
        dequantized = len(graph.tensors)
        hidden_tensor = graph.tensors[hidden]
        tensors = graph.tensors + (
            TensorInfo(
                dequantized,
                "input_embeds_dequantized",
                hidden_tensor.shape,
                hidden_tensor.shape_signature,
                "FLOAT32",
                0,
                0,
            ),
        )
        operators = [OperatorInfo(0, "DEQUANTIZE", (hidden,), (dequantized,))]
        for operator in graph.operators:
            inputs = (dequantized,) + operator.inputs[1:] if operator.index == 0 else operator.inputs
            operators.append(replace(operator, index=operator.index + 1, inputs=inputs))
        model = replace(model, graphs=(replace(graph, tensors=tensors, operators=tuple(operators)),))
        report = discover_models([(model, "layer00_1t1024c.tflite")], expected_layers=1)
        layer = report.sources[0].layers[0]
        analysis = report.sources[0].graph_analyses[0]

        tp4_attn = derive_tp4_components(model, layer, analysis)[0]
        tp8_attn = derive_tp_components(model, layer, analysis, degree=8)[0]
        self.assertIn(dequantized, tp4_attn.input_tensor_indices)
        self.assertNotIn(0, tp4_attn.operator_indices)
        self.assertIn(hidden, tp8_attn.input_tensor_indices)
        self.assertNotIn(dequantized, tp8_attn.input_tensor_indices)
        self.assertIn(0, tp8_attn.operator_indices)

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

    def test_tp2_defaults_to_all_gather(self):
        from tflite_tp.planning import resolve_topology
        self.assertEqual(resolve_topology(2), "all-gather")
        self.assertEqual(make_partition_plan(complete_layer(), degree=2).to_dict()["topology"], "TP2_FOUR_STAGE_ALL_GATHER")
        self.assertEqual(resolve_topology(2, "row-reduce"), "row-reduce")
        with self.assertRaises(ToolError):
            resolve_topology(4, "row-reduce")

    def test_tp2_explicit_row_reduce_for_o_and_down(self):
        plan = make_partition_plan(complete_layer(), degree=2, topology="row-reduce").to_dict()
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

    def test_tp2_all_gather_plan(self):
        plan = make_partition_plan(complete_layer(), degree=2, topology="all-gather").to_dict()
        self.assertEqual(plan["topology"], "TP2_FOUR_STAGE_ALL_GATHER")
        self.assertEqual(plan["collective_contract"]["rank_order"], [0, 1])
        self.assertEqual(plan["collective_contract"]["row_reduce_boundaries"], [])
        boundaries = plan["collective_contract"]["all_gather_boundaries"]
        self.assertEqual([(x["stage"], x["local_channels"], x["gathered_channels"]) for x in boundaries],
                         [("AG0", 2048, 4096), ("AG1", 2048, 4096),
                          ("AG2", 7168, 14336), ("AG3", 2048, 4096)])
        for projection in plan["projections"]:
            self.assertEqual(projection["slice_axis"], 0)
            self.assertEqual(projection["bias_policy"], "SLICE_WITH_OUTPUT_CHANNELS")
            shards = projection["rank_shards"]
            self.assertEqual(shards[0]["start"], 0)
            self.assertEqual(shards[0]["stop"], shards[1]["start"])
            self.assertEqual(shards[1]["stop"], projection["source_shape"][0])

    def test_all_gather_combines_rank_counts(self):
        import numpy as np
        from tflite_tp.golden import _combine_tp4_rank_outputs

        for degree in (2, 4, 8):
            shards = {rank: np.array([[rank]], dtype=np.int16) for rank in range(degree)}
            combined, mismatches, details = _combine_tp4_rank_outputs("tensor", shards, (1, degree))
            np.testing.assert_array_equal(combined, np.arange(degree, dtype=np.int16).reshape(1, degree))
            self.assertEqual((mismatches, details), (0, []))

    def test_token_extent_equal_to_head_count_fails_before_shape_rewrite(self):
        from tflite_tp.tflite_rewriter import _adjust_attention_shapes
        layer = SimpleNamespace(profile='P32', query_heads=32, kv_heads=8)
        component = SimpleNamespace(input_tensor_indices=(), output_tensor_indices=(), operator_indices=())
        with self.assertRaisesRegex(ToolError, 'ATTENTION_HEAD_AXIS_AMBIGUOUS'):
            _adjust_attention_shapes(None, None, layer, component, 0, 2)

    def test_head_axis_trace_distinguishes_equal_token_and_head_dimensions(self):
        import numpy as np
        from unittest.mock import patch
        from tflite_tp.tflite_rewriter import _attention_head_axes, _adjust_attention_shapes

        shapes = [(1, 32, 4096), (1, 32, 32, 128), (1, 32, 32, 128),
                  (1, 1, 32, 128), (1, 32, 32, 128), (1, 8, 1024, 128), (4,), (1, 32, 4096)]
        names = ['q_projection', 'reshaped', 'transposed', 'position', 'rotated', 'past_keys', 'perm', 'context']
        tensors = tuple(SimpleNamespace(name=name, shape=shape) for name, shape in zip(names, shapes))
        operators = (SimpleNamespace(index=0, opcode='RESHAPE', inputs=(0,), outputs=(1,)),
                     SimpleNamespace(index=1, opcode='TRANSPOSE', inputs=(1, 6), outputs=(2,)),
                     SimpleNamespace(index=2, opcode='MUL', inputs=(2, 3), outputs=(4,)))
        graph = SimpleNamespace(tensors=tensors, operators=operators)
        roles = {role: SimpleNamespace(output_tensor_indices=(0,)) for role in ('q', 'k', 'v')}
        roles['o'] = SimpleNamespace(activation_tensor_index=7, shape=(4096, 4096))
        layer = SimpleNamespace(profile='P32', query_heads=32, kv_heads=8, head_dim=128, roles=roles)
        component = SimpleNamespace(input_tensor_indices=(3, 5), output_tensor_indices=(4, 5),
                                    operator_indices=(0, 1, 2))
        with patch('tflite_tp.tflite_rewriter._tensor_data',
                   return_value=np.array([0, 2, 1, 3], dtype='<i4').tobytes()):
            axes = _attention_head_axes(None, graph, graph, layer, component)
        self.assertEqual(axes, {1: 2, 2: 1, 4: 1, 5: 1})
        self.assertNotIn(3, axes)
        mutable_tensors = [SimpleNamespace(shape=np.array(shape, dtype=np.int32)) for shape in shapes]
        mutable_graph = SimpleNamespace(tensors=mutable_tensors, operators=operators)
        with patch('tflite_tp.tflite_rewriter._tensor_data',
                   return_value=np.array([0, 2, 1, 3], dtype='<i4').tobytes()):
            _adjust_attention_shapes(None, mutable_graph, layer, component, 0, 2, source_graph=graph)
        self.assertEqual(tuple(mutable_tensors[1].shape), (1, 32, 16, 128))
        self.assertEqual(tuple(mutable_tensors[2].shape), (1, 16, 32, 128))
        self.assertEqual(tuple(mutable_tensors[3].shape), (1, 1, 32, 128))
        self.assertEqual(tuple(mutable_tensors[5].shape), (1, 4, 1024, 128))
        self.assertEqual(tuple(mutable_tensors[7].shape), (1, 32, 2048))

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

    def test_tp8_uses_eight_ranks_and_correct_gather_widths(self):
        plan = make_partition_plan(complete_layer("D1"), degree=8).to_dict()
        self.assertEqual(plan["topology"], "TP8_FOUR_STAGE_ALL_GATHER")
        self.assertEqual(plan["collective_contract"]["rank_order"], list(range(8)))
        boundaries = plan["collective_contract"]["all_gather_boundaries"]
        self.assertEqual([b["local_channels"] for b in boundaries], [512, 512, 1792, 512])
        self.assertEqual([b["local_channels"] * 8 for b in boundaries],
                         [b["gathered_channels"] for b in boundaries])

    def test_incomplete_layer_cannot_be_planned(self):
        layer = complete_layer()
        del layer.roles["down"]
        with self.assertRaises(ToolError) as raised:
            make_partition_plan(layer, degree=2)
        self.assertEqual(raised.exception.code, "LAYER_DISCOVERY_INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
