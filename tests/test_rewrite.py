import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

try:
    import tflite
except ImportError:  # pragma: no cover - exercised in minimal install environments
    tflite = None

from tflite_tp.errors import ToolError  # noqa: E402
from tflite_tp.flatbuffer_adapter import TFLiteObjectAdapter, _ObjectTable  # noqa: E402
from tflite_tp.tflite_rewriter import (  # noqa: E402
    _apply_attention_context_head_padding,
    _apply_fc_width_padding_rule,
    _apply_mlp_gate_dequant_width_padding_rule,
    _pad_rank_local_fc_output,
    _prune_model,
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

    def _make_gate_dequant_model(self):
        model, role_operator_indices, role_output_indices = self._make_fc_model(roles=("q", "k", "v"), tokens=128)
        graph = model.subgraphs[0]
        fc = graph.operators[role_operator_indices["v"]]
        weight_index = int(fc.inputs[1])
        graph.tensors[weight_index].name = "mlp/gate_proj/weight"
        gate_output_index = int(fc.outputs[0])
        gate_output = graph.tensors[gate_output_index]
        fp32_shape = tuple(int(value) for value in gate_output.shape)
        fp32_index = len(graph.tensors)
        graph.tensors.append(
            self._object(
                "Tensor",
                shape=np.asarray(fp32_shape, dtype=np.int32),
                shapeSignature=np.asarray(fp32_shape, dtype=np.int32),
                type=tflite.TensorType.FLOAT32,
                buffer=0,
                name="mlp/gate_proj/dequant",
                quantization=None,
                isVariable=False,
            )
        )
        dequant_opcode_index = len(model.operatorCodes)
        model.operatorCodes.append(
            self._object(
                "OperatorCode",
                builtinCode=tflite.BuiltinOperator.DEQUANTIZE,
                deprecatedBuiltinCode=tflite.BuiltinOperator.DEQUANTIZE,
                customCode=None,
                version=2,
            )
        )
        graph.operators.append(
            self._object(
                "Operator",
                opcodeIndex=dequant_opcode_index,
                inputs=np.asarray([gate_output_index], dtype=np.int32),
                outputs=np.asarray([fp32_index], dtype=np.int32),
                customOptions=b"",
                customOptionsFormat=tflite.CustomOptionsFormat.FLEXBUFFERS,
            )
        )
        graph.outputs = np.asarray([fp32_index], dtype=np.int32)
        return model, weight_index, fp32_index

    def _make_attention_head_padding_model(self):
        """Build a synthetic P128 attention graph with reusable TILE/SLICE templates."""
        shape = (1, 8, 128, 128)
        buffers = [self._object("Buffer", data=np.asarray([], dtype=np.uint8))]
        tensors = []

        def tensor(name, dimensions, tensor_type, buffer=0, quantized=False):
            quantization = self._quantization([1.0 / 32768.0]) if quantized else None
            tensors.append(self._object(
                "Tensor",
                shape=np.asarray(dimensions, dtype=np.int32),
                shapeSignature=np.asarray(dimensions, dtype=np.int32),
                type=tensor_type,
                buffer=buffer,
                name=name,
                quantization=quantization,
                isVariable=False,
            ))
            return len(tensors) - 1

        input_slice = tensor("layers/0/self_attn/slice", shape, tflite.TensorType.INT16, quantized=True)
        values = tensor("layers/0/self_attn/view.9", shape, tflite.TensorType.INT16, quantized=True)
        matmul_output = tensor("layers/0/self_attn/matmul4/matmul", shape, tflite.TensorType.INT16, quantized=True)
        public_output = tensor("attention_output", shape, tflite.TensorType.INT16, quantized=True)

        def constant(name, values):
            payload = np.asarray(values, dtype="<i4").tobytes()
            buffers.append(self._object("Buffer", data=np.frombuffer(payload, dtype=np.uint8).copy()))
            return tensor(name, (len(values),), tflite.TensorType.INT32, len(buffers) - 1)

        tile_multiples = constant("tile_multiples", (1, 1, 1, 1))
        slice_begin = constant("slice_begin", (0, 0, 0, 0))
        slice_size = constant("slice_size", shape)
        tile_template_output = tensor("tile_template_output", shape, tflite.TensorType.INT16, quantized=True)
        slice_template_output = tensor("slice_template_output", shape, tflite.TensorType.INT16, quantized=True)

        operator_codes = []
        for builtin, custom_code in (
            (tflite.BuiltinOperator.CUSTOM, "MTKEXT_TILE"),
            (tflite.BuiltinOperator.SLICE, None),
            (tflite.BuiltinOperator.BATCH_MATMUL, None),
            (tflite.BuiltinOperator.ADD, None),
        ):
            operator_codes.append(self._object(
                "OperatorCode",
                builtinCode=builtin,
                deprecatedBuiltinCode=builtin,
                customCode=custom_code,
                version=1,
            ))

        def operator(opcode, inputs, outputs, builtin_options_type=0, builtin_options=None):
            return self._object(
                "Operator",
                opcodeIndex=opcode,
                inputs=np.asarray(inputs, dtype=np.int32),
                outputs=np.asarray(outputs, dtype=np.int32),
                customOptions=b"{}" if opcode == 0 else None,
                customOptionsFormat=tflite.CustomOptionsFormat.FLEXBUFFERS if opcode == 0 else 0,
                builtinOptionsType=builtin_options_type,
                builtinOptions=builtin_options,
            )

        operators = [
            operator(0, (input_slice, tile_multiples), (tile_template_output,)),
            operator(1, (tile_template_output, slice_begin, slice_size), (slice_template_output,)),
            operator(2, (input_slice, values), (matmul_output,)),
            operator(3, (matmul_output, input_slice), (public_output,)),
        ]
        graph = self._object(
            "SubGraph",
            tensors=tensors,
            inputs=np.asarray([input_slice, values], dtype=np.int32),
            outputs=np.asarray([public_output], dtype=np.int32),
            operators=operators,
            name="main",
        )
        model = self._object(
            "Model",
            version=3,
            operatorCodes=operator_codes,
            subgraphs=[graph],
            buffers=buffers,
            description="synthetic attention context head-padding regression fixture",
        )
        return model

    def _prune_gate_component(self, model, weight_index, output_index, layer_id=0):
        graph = model.subgraphs[0]
        gate_operator_index = next(
            index for index, op in enumerate(graph.operators)
            if len(op.inputs) == 3 and int(op.inputs[1]) == weight_index
        )
        gate_output = int(graph.operators[gate_operator_index].outputs[0])
        dequant_operator_index = next(
            index for index, op in enumerate(graph.operators)
            if len(op.inputs) == 1 and int(op.inputs[0]) == gate_output
        )
        component = SimpleNamespace(
            name="mlp",
            operator_indices=(gate_operator_index, dequant_operator_index),
            input_tensor_indices=(int(graph.operators[gate_operator_index].inputs[0]),),
            output_tensor_indices=(output_index,),
        )
        layer = SimpleNamespace(
            profile="P128",
            layer_id=layer_id,
            roles={"gate": SimpleNamespace(
                operator_index=gate_operator_index,
                tensor_index=weight_index,
                tensor_name="mlp/gate_proj/weight",
            )},
        )
        _prune_model(model, graph, component)
        self.assertNotEqual(gate_operator_index, 0, "fixture must exercise operator-index remapping")
        self.assertNotEqual(int(graph.operators[0].inputs[1]), weight_index, "fixture must exercise tensor-index remapping")
        return graph, layer, component

    def _round_trip(self, model):
        raw = self.adapter.export(model)
        self.assertEqual(raw[4:8], b"TFL3")
        loaded = self.adapter.load_bytes(raw, "fc_width_padding_input.tflite")
        return loaded

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

    def test_mlp_gate_dequant_padding_slices_fp32_after_dequant(self):
        model, weight_index, expected_output = self._make_gate_dequant_model()
        model = self._round_trip(model)
        graph, layer, component = self._prune_gate_component(model, weight_index, expected_output)
        expected_output = int(graph.outputs[0])
        original_weight = graph.tensors[int(graph.operators[0].inputs[1])]
        expected_weight_payload = bytes(model.buffers[int(original_weight.buffer)].data) + bytes(6)
        original_weight_scales = np.asarray(original_weight.quantization.scale, dtype=np.float32).copy()
        expected_weight_scales = np.concatenate(
            [original_weight_scales, np.repeat(original_weight_scales[-1], 3)]
        )

        applied = _apply_mlp_gate_dequant_width_padding_rule(model, graph, layer, component, 4, 2, self.adapter)

        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["stage"], "GATE_FC_PADDED_INT16_DEQUANTIZED_BEFORE_FP32_SLICE")
        self.assertEqual(applied[0]["zero_padding_int4_bytes"], 6)
        fc, dequant, slice_op = graph.operators
        padded_weight = graph.tensors[int(fc.inputs[1])]
        self.assertEqual(tuple(int(value) for value in padded_weight.shape), (6, 4))
        self.assertEqual(bytes(model.buffers[int(padded_weight.buffer)].data), expected_weight_payload)
        np.testing.assert_array_equal(padded_weight.quantization.scale, expected_weight_scales)
        padded_bias = graph.tensors[int(fc.inputs[2])]
        self.assertEqual(bytes(model.buffers[int(padded_bias.buffer)].data)[-12:], bytes(12))
        self.assertEqual(tuple(int(value) for value in graph.tensors[int(fc.outputs[0])].shape), (1, 128, 6))
        self.assertEqual(tuple(int(value) for value in graph.tensors[int(dequant.outputs[0])].shape), (1, 128, 6))
        self.assertEqual(tuple(int(value) for value in dequant.inputs), tuple(int(value) for value in fc.outputs))
        self.assertEqual(tuple(int(value) for value in slice_op.inputs[:1]), tuple(int(value) for value in dequant.outputs))
        self.assertEqual(tuple(int(value) for value in slice_op.outputs), (expected_output,))
        self.assertEqual(tuple(int(value) for value in graph.tensors[expected_output].shape), (1, 128, 3))
        parsed = _validate_serialized_component(self.adapter.export(model), "mlp_gate_padded.tflite", tflite)
        self.assertEqual(
            [op.opcode for op in parsed.graphs[0].operators],
            ["MTKEXT_FULLY_CONNECTED", "DEQUANTIZE", "SLICE"],
        )
        self.assertEqual(parsed.graphs[0].outputs, (expected_output,))

    def test_attention_context_head_padding_tiles_batch_matmul_and_slices_back(self):
        model = self._round_trip(self._make_attention_head_padding_model())
        graph = model.subgraphs[0]
        # Exporter tensor names are not graph identity and may be arbitrary.
        for index, tensor in enumerate(graph.tensors):
            tensor.name = "tensor_{}".format(index).encode("utf-8")
        layer = SimpleNamespace(profile="P128", layer_id=0)
        component = SimpleNamespace(name="attn")
        input_qparams = np.asarray(graph.tensors[0].quantization.scale, dtype=np.float32).copy()
        output_qparams = np.asarray(graph.tensors[2].quantization.scale, dtype=np.float32).copy()

        applied = _apply_attention_context_head_padding(model, graph, layer, component, 4, 4, self.adapter)

        self.assertEqual(
            applied["status"],
            "APPLIED_TP4_P128_LAYER0_ATTENTION_CONTEXT_BATCHMATMUL_HEAD_PADDING",
        )
        self.assertEqual(applied["tile_multiples"], [1, 4, 1, 1])
        self.assertEqual(applied["target_head_count"], 32)
        self.assertEqual(tuple(int(value) for value in graph.tensors[int(graph.operators[4].outputs[0])].shape), (1, 32, 128, 128))
        self.assertEqual(tuple(int(value) for value in graph.tensors[int(graph.operators[5].outputs[0])].shape), (1, 8, 128, 128))
        self.assertEqual(tuple(int(value) for value in graph.operators[4].inputs), (12, 13))
        self.assertEqual(tuple(int(value) for value in graph.operators[5].outputs), (15,))
        self.assertEqual(tuple(int(value) for value in graph.operators[6].inputs), (15, 0))
        np.testing.assert_array_equal(graph.tensors[0].quantization.scale, input_qparams)
        np.testing.assert_array_equal(graph.tensors[2].quantization.scale, output_qparams)
        self.assertEqual(applied["npu_verification_status"], "NOT_VERIFIED_FOR_NEW_CANDIDATE_SHA")

        serialized = self.adapter.export(model)
        loaded = self.adapter.load_bytes(serialized, "attention_head_padding.tflite")
        self.assertEqual(len(loaded.subgraphs[0].operators), 7)
        self.assertEqual(tuple(int(value) for value in loaded.subgraphs[0].outputs), (3,))
        bmm = loaded.subgraphs[0].operators[4]
        self.assertEqual(int(loaded.operatorCodes[int(bmm.opcodeIndex)].builtinCode), tflite.BuiltinOperator.BATCH_MATMUL)

    def test_attention_context_head_padding_fails_closed_on_ambiguous_shape_match(self):
        model = self._round_trip(self._make_attention_head_padding_model())
        graph = model.subgraphs[0]
        duplicate_output = copy.deepcopy(graph.tensors[2])
        duplicate_output.name = b"unrelated_exporter_name"
        graph.tensors.append(duplicate_output)
        duplicate = copy.deepcopy(graph.operators[2])
        duplicate.outputs = np.asarray([len(graph.tensors) - 1], dtype=np.int32)
        graph.operators.append(duplicate)
        duplicate_add_output = copy.deepcopy(graph.tensors[3])
        duplicate_add_output.name = b"another_unrelated_name"
        graph.tensors.append(duplicate_add_output)
        duplicate_add = copy.deepcopy(graph.operators[3])
        duplicate_add.inputs = np.asarray([len(graph.tensors) - 2, 0], dtype=np.int32)
        duplicate_add.outputs = np.asarray([len(graph.tensors) - 1], dtype=np.int32)
        graph.operators.append(duplicate_add)

        with self.assertRaisesRegex(ToolError, "ATTENTION_CONTEXT_PADDING_MATMUL_AMBIGUOUS"):
            _apply_attention_context_head_padding(
                model,
                graph,
                SimpleNamespace(profile="P128", layer_id=0),
                SimpleNamespace(name="attn"),
                4,
                4,
                self.adapter,
            )

    def test_attention_context_head_padding_ignores_same_shape_concat_product(self):
        model = self._round_trip(self._make_attention_head_padding_model())
        graph = model.subgraphs[0]
        # Multiple equivalent operator templates are common in real graphs.
        for template_index, template_output in ((0, 4), (1, 5)):
            duplicate_template = copy.deepcopy(graph.operators[template_index])
            duplicate_tensor = copy.deepcopy(graph.tensors[template_output])
            duplicate_tensor.name = "template_output_{}".format(template_index).encode("utf-8")
            graph.tensors.append(duplicate_tensor)
            duplicate_template.outputs = np.asarray([len(graph.tensors) - 1], dtype=np.int32)
            graph.operators.append(duplicate_template)
        distractor_output = copy.deepcopy(graph.tensors[2])
        distractor_output.name = b"score_product_output"
        graph.tensors.append(distractor_output)
        distractor_product = copy.deepcopy(graph.operators[2])
        distractor_product.outputs = np.asarray([len(graph.tensors) - 1], dtype=np.int32)
        graph.operators.append(distractor_product)

        concat_code_index = len(model.operatorCodes)
        model.operatorCodes.append(self._object(
            "OperatorCode",
            builtinCode=tflite.BuiltinOperator.CONCATENATION,
            deprecatedBuiltinCode=tflite.BuiltinOperator.CONCATENATION,
            customCode=None,
            version=1,
        ))
        concat_output = copy.deepcopy(graph.tensors[2])
        concat_output.name = b"concatenated_score_output"
        concat_output.shape = np.asarray((1, 8, 128, 256), dtype=np.int32)
        concat_output.shapeSignature = np.asarray((1, 8, 128, 256), dtype=np.int32)
        graph.tensors.append(concat_output)
        graph.operators.append(self._object(
            "Operator",
            opcodeIndex=concat_code_index,
            inputs=np.asarray([len(graph.tensors) - 2, 1], dtype=np.int32),
            outputs=np.asarray([len(graph.tensors) - 1], dtype=np.int32),
            customOptions=b"",
            customOptionsFormat=tflite.CustomOptionsFormat.FLEXBUFFERS,
        ))

        applied = _apply_attention_context_head_padding(
            model,
            graph,
            SimpleNamespace(profile="P128", layer_id=0),
            SimpleNamespace(name="attn"),
            4,
            4,
            self.adapter,
        )
        self.assertEqual(applied["target_head_count"], 32)

    def test_attention_context_head_padding_fails_closed_outside_verified_scope(self):
        model = self._make_attention_head_padding_model()
        with self.assertRaisesRegex(ToolError, "ATTENTION_CONTEXT_PADDING_MULTIPLIER_UNVERIFIED"):
            _apply_attention_context_head_padding(
                model,
                model.subgraphs[0],
                SimpleNamespace(profile="P128", layer_id=0),
                SimpleNamespace(name="attn"),
                4,
                2,
                self.adapter,
            )

        model = self._make_attention_head_padding_model()
        with self.assertRaisesRegex(ToolError, "ATTENTION_CONTEXT_PADDING_SCOPE_UNVERIFIED"):
            _apply_attention_context_head_padding(
                model, model.subgraphs[0], SimpleNamespace(profile="D1", layer_id=0),
                SimpleNamespace(name="attn"), 4, 4, self.adapter
            )

    def test_mlp_gate_width_rewrite_accepts_p128_layer1_as_unverified_candidate(self):
        model, weight_index, expected_output = self._make_gate_dequant_model()
        model = self._round_trip(model)
        graph, layer, component = self._prune_gate_component(model, weight_index, expected_output, layer_id=1)
        applied = _apply_mlp_gate_dequant_width_padding_rule(
            model, graph, layer, component, 4, 2, self.adapter
        )
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["npu_verification_status"], "NOT_VERIFIED_FOR_NEW_CANDIDATE_SHA")
        self.assertEqual(applied[0]["rank_local_int4_nonzero_payload_bytes"], 6)

    def test_mlp_gate_width_rewrite_fails_closed_for_other_profiles_and_tp(self):
        component = SimpleNamespace(name="mlp")
        for layer, degree in (
            (SimpleNamespace(profile="D1", layer_id=0, roles={}), 4),
            (SimpleNamespace(profile="P128", layer_id=0, roles={}), 2),
        ):
            model, _, _ = self._make_gate_dequant_model()
            model = self._round_trip(model)
            with self.subTest(profile=layer.profile, layer_id=layer.layer_id, degree=degree):
                with self.assertRaisesRegex(ToolError, "MLP_GATE_WIDTH_PADDING"):
                    _apply_mlp_gate_dequant_width_padding_rule(
                        model, model.subgraphs[0], layer, component, degree, 2, self.adapter
                    )

    def test_mlp_gate_width_rewrite_is_opt_in(self):
        model, weight_index, _ = self._make_gate_dequant_model()
        model = self._round_trip(model)
        layer = SimpleNamespace(
            profile="P128",
            layer_id=0,
            roles={"gate": SimpleNamespace(tensor_index=weight_index)},
        )
        self.assertEqual(
            _apply_mlp_gate_dequant_width_padding_rule(
                model, model.subgraphs[0], layer, SimpleNamespace(name="mlp"), 4, None, self.adapter
            ),
            [],
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


class AttentionContextPaddingSemanticsTests(unittest.TestCase):
    def test_repeated_batch_matmul_heads_then_slice_preserves_local_result(self):
        # Small synthetic integer arithmetic checks the repeat-and-slice
        # contract. It does not emulate MTK rounding or prove model/NPU exactness.
        rng = np.random.default_rng(2410)
        query = rng.integers(-3, 4, size=(1, 8, 5, 7), dtype=np.int16)
        key = rng.integers(-3, 4, size=(1, 8, 7, 6), dtype=np.int16)

        local = np.matmul(query.astype(np.int32), key.astype(np.int32))
        tiled_query = np.tile(query, (1, 4, 1, 1)).astype(np.int32)
        tiled_key = np.tile(key, (1, 4, 1, 1)).astype(np.int32)
        padded = np.matmul(tiled_query, tiled_key)
        restored = padded[:, :8, :, :]

        self.assertEqual(tiled_query.shape, (1, 32, 5, 7))
        self.assertEqual(restored.shape, (1, 8, 5, 6))
        np.testing.assert_array_equal(restored, local)


if __name__ == "__main__":
    unittest.main()
