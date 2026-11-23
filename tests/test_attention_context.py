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
class AttentionContextRewriteTests(unittest.TestCase):

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
