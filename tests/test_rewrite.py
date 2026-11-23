import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np



try:
    import tflite
except ImportError:  # pragma: no cover - exercised in minimal install environments
    tflite = None

from tflite_tp.errors import ToolError  # noqa: E402
from tflite_tp.flatbuffer_adapter import TFLiteObjectAdapter, _ObjectTable  # noqa: E402
from tflite_tp.tflite_rewriter import (  # noqa: E402
    _apply_mlp_gate_dequant_width_padding_rule,
    _apply_fc_width_padding_rule,
    _pad_rank_local_fc_output,
    _validate_serialized_component,
)


@unittest.skipIf(tflite is None, "TFLite schema dependency is not installed")
class FCWidthPaddingRewriteTests(unittest.TestCase):
    def setUp(self):
        self.adapter = TFLiteObjectAdapter(tflite)

    def _object(self, table_name, **fields):
        result = _ObjectTable(table_name, self.adapter)
        for name, value in fields.items():
            setattr(result, name, value)
        return result

    def _quantization(self, scales, zero_points=None, minimum=(), maximum=(), dimension=0):
        if zero_points is None:
            zero_points = [0] * len(scales)
        return self._object(
            "QuantizationParameters",
            scale=np.asarray(scales, dtype=np.float32),
            zeroPoint=np.asarray(zero_points, dtype=np.int64),
            min=np.asarray(minimum, dtype=np.float32),
            max=np.asarray(maximum, dtype=np.float32),
            quantizedDimension=dimension,
        )

    def _make_fc_model(self, roles=("q",), tokens=128):
        """Build a tiny real TFLite FlatBuffer with MTKEXT FC operators."""
        tensors = []
        operators = []
        buffers = [self._object("Buffer", data=np.asarray([], dtype=np.uint8))]
        role_operator_indices = {}
        role_output_indices = {}
        for role_index, role in enumerate(roles):
            activation_index = len(tensors)
            activation_shape = (1, tokens, 4)
            tensors.append(
                self._object(
                    "Tensor",
                    shape=np.asarray(activation_shape, dtype=np.int32),
                    shapeSignature=np.asarray(activation_shape, dtype=np.int32),
                    type=tflite.TensorType.INT16,
                    buffer=0,
                    name="{}_activation".format(role),
                    quantization=self._quantization([0.05]),
                    isVariable=False,
                )
            )

            weight_index = len(tensors)
            # Three output channels exercise odd-N INT4 row packing.
            weight_shape = (3, 4)
            weight_payload = bytes((0x21 + role_index, 0x43, 0x65, 0x87, 0xA9, 0xCB))
            weight_buffer = len(buffers)
            buffers.append(self._object("Buffer", data=np.frombuffer(weight_payload, dtype=np.uint8).copy()))
            weight_scales = [0.1 + role_index, 0.2 + role_index, 0.3 + role_index]
            tensors.append(
                self._object(
                    "Tensor",
                    shape=np.asarray(weight_shape, dtype=np.int32),
                    shapeSignature=np.asarray(weight_shape, dtype=np.int32),
                    type=tflite.TensorType.INT4,
                    buffer=weight_buffer,
                    name="{}_weight".format(role),
                    quantization=self._quantization(
                        weight_scales,
                        zero_points=[0, 0, 0],
                        minimum=[-7.0, -7.0, -7.0],
                        maximum=[7.0, 7.0, 7.0],
                    ),
                    isVariable=False,
                )
            )

            bias_index = len(tensors)
            bias_shape = (3,)
            bias_payload = np.asarray([role_index + 1, -2, 3], dtype="<i4").tobytes()
            bias_buffer = len(buffers)
            buffers.append(self._object("Buffer", data=np.frombuffer(bias_payload, dtype=np.uint8).copy()))
            tensors.append(
                self._object(
                    "Tensor",
                    shape=np.asarray(bias_shape, dtype=np.int32),
                    shapeSignature=np.asarray(bias_shape, dtype=np.int32),
                    type=tflite.TensorType.INT32,
                    buffer=bias_buffer,
                    name="{}_bias".format(role),
                    quantization=self._quantization(
                        [0.01, 0.02, 0.03],
                        minimum=[-100.0, -100.0, -100.0],
                        maximum=[100.0, 100.0, 100.0],
                    ),
                    isVariable=False,
                )
            )

            output_index = len(tensors)
            output_shape = (1, tokens, 3)
            tensors.append(
                self._object(
                    "Tensor",
                    shape=np.asarray(output_shape, dtype=np.int32),
                    shapeSignature=np.asarray(output_shape, dtype=np.int32),
                    type=tflite.TensorType.INT16,
                    buffer=0,
                    name="{}_output".format(role),
                    quantization=self._quantization([0.05]),
                    isVariable=False,
                )
            )
            role_operator_indices[role] = len(operators)
            role_output_indices[role] = output_index
            operators.append(
                self._object(
                    "Operator",
                    opcodeIndex=0,
                    inputs=np.asarray([activation_index, weight_index, bias_index], dtype=np.int32),
                    outputs=np.asarray([output_index], dtype=np.int32),
                    customOptions=b"{}",
                    customOptionsFormat=tflite.CustomOptionsFormat.FLEXBUFFERS,
                )
            )

        graph = self._object(
            "SubGraph",
            tensors=tensors,
            inputs=np.asarray([4 * index for index in range(len(roles))], dtype=np.int32),
            outputs=np.asarray([role_output_indices[role] for role in roles], dtype=np.int32),
            operators=operators,
            name="main",
        )
        operator_code = self._object(
            "OperatorCode",
            builtinCode=tflite.BuiltinOperator.CUSTOM,
            deprecatedBuiltinCode=tflite.BuiltinOperator.CUSTOM,
            customCode="MTKEXT_FULLY_CONNECTED",
            version=1,
        )
        model = self._object(
            "Model",
            version=3,
            operatorCodes=[operator_code],
            subgraphs=[graph],
            buffers=buffers,
            description="synthetic FC width padding regression fixture",
        )
        return model, role_operator_indices, role_output_indices

    def _round_trip(self, model):
        raw = self.adapter.export(model)
        self.assertEqual(raw[4:8], b"TFL3")
        loaded = self.adapter.load_bytes(raw, "fc_width_padding_input.tflite")
        return loaded

    def _make_gate_dequant_model(self, tokens=128):
        activation = self._object(
            "Tensor",
            shape=np.asarray([1, tokens, 4], dtype=np.int32),
            shapeSignature=np.asarray([1, tokens, 4], dtype=np.int32),
            type=tflite.TensorType.INT16,
            buffer=0,
            name="layers/0/post_attention_norm/mul/mul",
            quantization=self._quantization([0.001]),
            isVariable=False,
        )
        weight_payload = bytes((0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB))
        bias_payload = np.asarray([1, -2, 3], dtype="<i4").tobytes()
        buffers = [
            self._object("Buffer", data=np.asarray([], dtype=np.uint8)),
            self._object("Buffer", data=np.frombuffer(weight_payload, dtype=np.uint8).copy()),
            self._object("Buffer", data=np.frombuffer(bias_payload, dtype=np.uint8).copy()),
        ]
        weight = self._object(
            "Tensor",
            shape=np.asarray([3, 4], dtype=np.int32),
            shapeSignature=np.asarray([3, 4], dtype=np.int32),
            type=tflite.TensorType.INT4,
            buffer=1,
            name="layers/0/mlp/gate_proj/weight",
            quantization=self._quantization([0.1, 0.2, 0.3], minimum=[-7.0] * 3, maximum=[7.0] * 3),
            isVariable=False,
        )
        bias = self._object(
            "Tensor",
            shape=np.asarray([3], dtype=np.int32),
            shapeSignature=np.asarray([3], dtype=np.int32),
            type=tflite.TensorType.INT32,
            buffer=2,
            name="layers/0/mlp/gate_proj/bias",
            quantization=self._quantization([0.0001, 0.0002, 0.0003]),
            isVariable=False,
        )
        gate_int16 = self._object(
            "Tensor",
            shape=np.asarray([1, tokens, 3], dtype=np.int32),
            shapeSignature=np.asarray([1, tokens, 3], dtype=np.int32),
            type=tflite.TensorType.INT16,
            buffer=0,
            name="layers/0/mlp/gate_proj/fully_connected",
            quantization=self._quantization([0.05]),
            isVariable=False,
        )
        gate_fp32 = self._object(
            "Tensor",
            shape=np.asarray([1, tokens, 3], dtype=np.int32),
            shapeSignature=np.asarray([1, tokens, 3], dtype=np.int32),
            type=tflite.TensorType.FLOAT32,
            buffer=0,
            name="layers/0/mlp/gate_proj/dequantize",
            quantization=None,
            isVariable=False,
        )
        fc = self._object(
            "Operator",
            opcodeIndex=0,
            inputs=np.asarray([0, 1, 2], dtype=np.int32),
            outputs=np.asarray([3], dtype=np.int32),
            customOptions=b"{}",
            customOptionsFormat=tflite.CustomOptionsFormat.FLEXBUFFERS,
        )
        dequant = self._object(
            "Operator",
            opcodeIndex=1,
            inputs=np.asarray([3], dtype=np.int32),
            outputs=np.asarray([4], dtype=np.int32),
        )
        graph = self._object(
            "SubGraph",
            tensors=[activation, weight, bias, gate_int16, gate_fp32],
            inputs=np.asarray([0], dtype=np.int32),
            outputs=np.asarray([4], dtype=np.int32),
            operators=[fc, dequant],
            name="main",
        )
        model = self._object(
            "Model",
            version=3,
            operatorCodes=[
                self._object(
                    "OperatorCode",
                    builtinCode=tflite.BuiltinOperator.CUSTOM,
                    deprecatedBuiltinCode=tflite.BuiltinOperator.CUSTOM,
                    customCode="MTKEXT_FULLY_CONNECTED",
                    version=1,
                ),
                self._object(
                    "OperatorCode",
                    builtinCode=tflite.BuiltinOperator.DEQUANTIZE,
                    deprecatedBuiltinCode=tflite.BuiltinOperator.DEQUANTIZE,
                    version=2,
                ),
            ],
            subgraphs=[graph],
            buffers=buffers,
            description="synthetic TP4 MLP Gate Dequant width rewrite fixture",
        )
        return model

    def test_mlp_gate_dequant_width_padding_slices_after_dequant(self):
        model = self._round_trip(self._make_gate_dequant_model())
        graph = model.subgraphs[0]
        original_output = int(graph.outputs[0])
        layer = SimpleNamespace(profile="P128", layer_id=0)
        component = SimpleNamespace(name="mlp")

        applied = _apply_mlp_gate_dequant_width_padding_rule(
            model, graph, layer, component, 4, 2, self.adapter
        )

        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["padded_output_width"], 6)
        self.assertEqual(applied[0]["zero_padding_int4_bytes"], 6)
        self.assertEqual([model.operatorCodes[int(op.opcodeIndex)].customCode or "BUILTIN" for op in graph.operators], [
            b"MTKEXT_FULLY_CONNECTED", "BUILTIN", "BUILTIN"
        ])
        fc, dequant, slice_op = graph.operators
        padded_gate = int(fc.outputs[0])
        padded_fp32 = int(dequant.outputs[0])
        self.assertEqual(tuple(int(x) for x in graph.tensors[padded_gate].shape), (1, 128, 6))
        self.assertEqual(tuple(int(x) for x in graph.tensors[padded_fp32].shape), (1, 128, 6))
        self.assertEqual(int(dequant.inputs[0]), padded_gate)
        self.assertEqual(tuple(int(x) for x in slice_op.inputs[:1]), (padded_fp32,))
        self.assertEqual(tuple(int(x) for x in slice_op.outputs), (original_output,))
        self.assertEqual(tuple(int(x) for x in graph.tensors[original_output].shape), (1, 128, 3))
        self.assertEqual(tuple(int(x) for x in graph.outputs), (original_output,))
        padded_weight = graph.tensors[int(fc.inputs[1])]
        self.assertEqual(bytes(model.buffers[int(padded_weight.buffer)].data), bytes((0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB)) + bytes(6))

        serialized = self.adapter.export(model)
        parsed = _validate_serialized_component(serialized, "mlp_gate_dequant_width.tflite", tflite)
        self.assertEqual([op.opcode for op in parsed.graphs[0].operators], [
            "MTKEXT_FULLY_CONNECTED", "DEQUANTIZE", "SLICE"
        ])
        self.assertEqual(parsed.graphs[0].outputs, (original_output,))

    def test_mlp_gate_dequant_width_padding_is_opt_in_and_profile_guarded(self):
        model = self._round_trip(self._make_gate_dequant_model())
        graph = model.subgraphs[0]
        layer = SimpleNamespace(profile="P128", layer_id=0)
        component = SimpleNamespace(name="mlp")
        self.assertEqual(
            _apply_mlp_gate_dequant_width_padding_rule(model, graph, layer, component, 4, None, self.adapter),
            [],
        )
        with self.assertRaisesRegex(ToolError, "MLP_GATE_WIDTH_PADDING_MULTIPLIER_UNVERIFIED"):
            _apply_mlp_gate_dequant_width_padding_rule(model, graph, layer, component, 4, 3, self.adapter)
        with self.assertRaisesRegex(ToolError, "MLP_GATE_WIDTH_PADDING_PROFILE_UNSUPPORTED"):
            _apply_mlp_gate_dequant_width_padding_rule(model, graph, layer, component, 2, 2, self.adapter)

    def test_mlp_gate_dequant_padding_builds_unverified_p128_layer_candidate(self):
        component = SimpleNamespace(name="mlp")
        layer = SimpleNamespace(profile="P128", layer_id=1)
        model = self._round_trip(self._make_gate_dequant_model())
        applied = _apply_mlp_gate_dequant_width_padding_rule(
            model, model.subgraphs[0], layer, component, 4, 2, self.adapter
        )
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["padded_output_width"], 6)
        self.assertEqual(
            applied[0]["candidate_numerical_status"],
            "NOT_RUN_REQUIRES_SHA_BOUND_NPU_GOLDEN",
        )
        self.assertEqual(
            applied[0]["formal_profile_status"],
            "CANDIDATE_REQUIRES_SHA_BOUND_NPU_GOLDEN",
        )

        d1 = SimpleNamespace(profile="D1", layer_id=0)
        model = self._round_trip(self._make_gate_dequant_model())
        with self.assertRaisesRegex(ToolError, "FC_WIDTH_PADDING_PROFILE_UNSUPPORTED"):
            _apply_mlp_gate_dequant_width_padding_rule(
                model, model.subgraphs[0], d1, component, 4, 2, self.adapter
            )

        for degree, profile in ((8, "P128"), (8, "D1")):
            model = self._round_trip(self._make_gate_dequant_model(tokens=128 if profile == "P128" else 1))
            applied = _apply_mlp_gate_dequant_width_padding_rule(
                model, model.subgraphs[0], SimpleNamespace(profile=profile, layer_id=0),
                component, degree, 2, self.adapter,
            )
            self.assertEqual(applied[0]["padded_output_width"], 6)

    def test_p128_fc_padding_serializes_and_preserves_rank_output(self):
        model, _, outputs = self._make_fc_model(tokens=128)
        model = self._round_trip(model)
        graph = model.subgraphs[0]
        original_output_index = outputs["q"]
        original_output = graph.tensors[original_output_index]
        expected_output_shape = tuple(int(x) for x in original_output.shape)
        weight_index = int(graph.operators[0].inputs[1])
        bias_index = int(graph.operators[0].inputs[2])
        weight = graph.tensors[weight_index]
        bias = graph.tensors[bias_index]
        original_weight = bytes(model.buffers[int(weight.buffer)].data)
        original_bias = bytes(model.buffers[int(bias.buffer)].data)

        result = _pad_rank_local_fc_output(model, graph, 0, "q", 2, self.adapter, "P128")

        self.assertEqual(tuple(int(x) for x in weight.shape), (6, 4))
        self.assertEqual(bytes(model.buffers[int(weight.buffer)].data), original_weight + bytes(6))
        np.testing.assert_array_equal(
            weight.quantization.scale,
            np.asarray([0.1, 0.2, 0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        )
        self.assertEqual(tuple(int(x) for x in weight.quantization.zeroPoint), (0, 0, 0, 0, 0, 0))
        self.assertEqual(tuple(float(x) for x in weight.quantization.min), (-7.0,) * 6)
        self.assertEqual(tuple(float(x) for x in weight.quantization.max), (7.0,) * 6)
        self.assertEqual(tuple(int(x) for x in bias.shape), (6,))
        self.assertEqual(bytes(model.buffers[int(bias.buffer)].data), original_bias + bytes(12))
        np.testing.assert_array_equal(
            bias.quantization.scale,
            np.asarray([0.01, 0.02, 0.03, 0.03, 0.03, 0.03], dtype=np.float32),
        )

        fc_op, slice_op = graph.operators
        padded_output_index = int(fc_op.outputs[0])
        self.assertEqual(tuple(int(x) for x in graph.tensors[padded_output_index].shape), (1, 128, 6))
        self.assertEqual(tuple(int(x) for x in slice_op.inputs), (padded_output_index, len(graph.tensors) - 2, len(graph.tensors) - 1))
        self.assertEqual(tuple(int(x) for x in slice_op.outputs), (original_output_index,))
        self.assertEqual(tuple(int(x) for x in original_output.shape), expected_output_shape)
        self.assertEqual(int(graph.outputs[0]), original_output_index)
        self.assertEqual(int(model.operatorCodes[int(slice_op.opcodeIndex)].builtinCode), tflite.BuiltinOperator.SLICE)
        self.assertEqual(result["rank_local_int4_nonzero_payload_bytes"], 6)
        self.assertEqual(result["serialized_int4_weight_bytes"], 12)
        self.assertEqual(result["zero_padding_int4_bytes"], 6)

        serialized = self.adapter.export(model)
        parsed = _validate_serialized_component(serialized, "padded_q.tflite", tflite)
        parsed_graph = parsed.graphs[0]
        self.assertEqual(len(parsed_graph.operators), 2)
        self.assertEqual(parsed_graph.operators[0].opcode, "MTKEXT_FULLY_CONNECTED")
        self.assertEqual(parsed_graph.operators[1].opcode, "SLICE")
        self.assertEqual(parsed_graph.tensors[original_output_index].shape, expected_output_shape)
        self.assertEqual(parsed_graph.outputs, (original_output_index,))

    def test_d1_and_all_qkv_roles_are_padded_by_rank_generic_rule(self):
        roles = ("q", "k", "v")
        model, operator_indices, output_indices = self._make_fc_model(roles=roles, tokens=1)
        model = self._round_trip(model)
        graph = model.subgraphs[0]
        layer = SimpleNamespace(
            profile="D1",
            roles={role: SimpleNamespace(operator_index=operator_indices[role]) for role in roles},
        )
        component = SimpleNamespace(name="attn", operator_indices=tuple(operator_indices[role] for role in roles))

        applied = _apply_fc_width_padding_rule(model, graph, layer, component, 4, 2, self.adapter)

        self.assertEqual([item["role"] for item in applied], ["q", "k", "v"])
        self.assertEqual(len(graph.operators), 6)
        self.assertEqual(tuple(int(x) for x in graph.outputs), tuple(output_indices[r] for r in roles))
        for output_index in output_indices.values():
            self.assertEqual(tuple(int(x) for x in graph.tensors[output_index].shape), (1, 1, 3))
        serialized = self.adapter.export(model)
        parsed = _validate_serialized_component(serialized, "padded_qkv_d1.tflite", tflite)
        self.assertEqual(len(parsed.graphs[0].operators), 6)
        self.assertEqual(sum(op.opcode == "SLICE" for op in parsed.graphs[0].operators), 3)

    def test_tp8_attention_qkv_can_use_shape_preserving_width_padding(self):
        roles = ("q", "k", "v")
        model, operator_indices, output_indices = self._make_fc_model(roles=roles, tokens=1)
        model = self._round_trip(model)
        graph = model.subgraphs[0]
        layer = SimpleNamespace(
            profile="D1",
            roles={role: SimpleNamespace(operator_index=operator_indices[role]) for role in roles},
        )
        component = SimpleNamespace(name="attn", operator_indices=tuple(operator_indices[role] for role in roles))

        applied = _apply_fc_width_padding_rule(model, graph, layer, component, 8, 2, self.adapter)

        self.assertEqual([item["role"] for item in applied], ["q", "k", "v"])
        self.assertEqual(tuple(int(x) for x in graph.outputs), tuple(output_indices[role] for role in roles))
        parsed = _validate_serialized_component(self.adapter.export(model), "padded_tp8_qkv_d1.tflite", tflite)
        self.assertEqual(sum(op.opcode == "SLICE" for op in parsed.graphs[0].operators), 3)
        self.assertEqual(
            tuple(parsed.graphs[0].tensors[index].shape for index in parsed.graphs[0].outputs),
            ((1, 1, 3),) * 3,
        )

    def test_padding_rejects_unverified_or_incompatible_contracts(self):
        model, _, _ = self._make_fc_model(tokens=128)
        model = self._round_trip(model)
        with self.assertRaisesRegex(ToolError, "FC_WIDTH_PADDING_MULTIPLIER_UNVERIFIED"):
            _pad_rank_local_fc_output(model, model.subgraphs[0], 0, "q", 3, self.adapter, "P128")

        model, _, _ = self._make_fc_model(tokens=128)
        model = self._round_trip(model)
        with self.assertRaisesRegex(ToolError, "FC_WIDTH_PADDING_PROFILE_UNSUPPORTED"):
            _pad_rank_local_fc_output(model, model.subgraphs[0], 0, "q", 2, self.adapter, "D1")

        model, _, _ = self._make_fc_model(tokens=128)
        model = self._round_trip(model)
        weight = model.subgraphs[0].tensors[int(model.subgraphs[0].operators[0].inputs[1])]
        weight.quantization.scale = np.asarray([0.1, 0.2], dtype=np.float32)
        with self.assertRaisesRegex(ToolError, "FC_WIDTH_PADDING_QUANTIZATION_SHAPE_UNSUPPORTED"):
            _pad_rank_local_fc_output(model, model.subgraphs[0], 0, "q", 2, self.adapter, "P128")

    def test_padding_requires_custom_fc_and_valid_int4_buffer_layout(self):
        model, _, _ = self._make_fc_model(tokens=128)
        model = self._round_trip(model)
        model.operatorCodes[0].customCode = "OTHER_CUSTOM_OP"
        with self.assertRaisesRegex(ToolError, "FC_WIDTH_PADDING_OPERATOR_UNSUPPORTED"):
            _pad_rank_local_fc_output(model, model.subgraphs[0], 0, "q", 2, self.adapter, "P128")

        model, _, _ = self._make_fc_model(tokens=128)
        model = self._round_trip(model)
        weight = model.subgraphs[0].tensors[int(model.subgraphs[0].operators[0].inputs[1])]
        weight.shape = np.asarray([3, 3], dtype=np.int32)
        with self.assertRaisesRegex(ToolError, "FC_WIDTH_PADDING_INT4_LAYOUT_UNSUPPORTED"):
            _pad_rank_local_fc_output(model, model.subgraphs[0], 0, "q", 2, self.adapter, "P128")


if __name__ == "__main__":
    unittest.main()
