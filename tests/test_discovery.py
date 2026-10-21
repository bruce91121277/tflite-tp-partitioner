import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from tflite_tp.discovery import classify_profile, discover_models, source_layer_id  # noqa: E402
from tflite_tp.errors import ModelValidationError, ToolError  # noqa: E402
from tflite_tp.model import (  # noqa: E402
    EXPECTED_WEIGHT_SHAPES,
    GraphInfo,
    ModelInfo,
    OperatorInfo,
    QuantizationInfo,
    TensorInfo,
    validate_graph,
)
from tflite_tp.tflite_reader import _vector, read_tflite  # noqa: E402


def synthetic_model(layer_id=0, tokens=128, duplicate_q=False, group_scales=False, operator_name="FULLY_CONNECTED"):
    role_list = ("q", "k", "v", "o", "gate", "up", "down")
    tensors = []
    inputs = []
    outputs = []
    operators = []
    quant_scales = (0.01, 0.02, 0.03, 0.04) if group_scales else (0.01,)
    names = {
        "q": "self_attn.q_proj",
        "k": "self_attn.k_proj",
        "v": "self_attn.v_proj",
        "o": "self_attn.o_proj",
        "gate": "mlp.gate_proj",
        "up": "mlp.up_proj",
        "down": "mlp.down_proj",
    }

    for role in role_list:
        activation_index = len(tensors)
        input_width = EXPECTED_WEIGHT_SHAPES[role][1]
        tensors.append(
            TensorInfo(
                index=activation_index,
                name="activation_{}".format(role),
                shape=(1, tokens, input_width),
                shape_signature=(1, tokens, input_width),
                dtype="INT16",
                buffer_index=0,
                buffer_bytes=0,
            )
        )
        inputs.append(activation_index)

    for role_index, role in enumerate(role_list):
        shape = EXPECTED_WEIGHT_SHAPES[role]
        tensors.append(
            TensorInfo(
                index=len(tensors),
                name="layers/{}/{}.weight".format(layer_id, names[role]),
                shape=shape,
                shape_signature=shape,
                dtype="INT4",
                buffer_index=role_index + 1,
                buffer_bytes=(shape[0] * shape[1] + 1) // 2,
                quantization=QuantizationInfo(
                    scales=quant_scales,
                    zero_points=tuple(0 for _ in quant_scales),
                    quantized_dimension=0,
                ),
            )
        )

    for role_index, role in enumerate(role_list):
        output_index = len(tensors)
        output_width = EXPECTED_WEIGHT_SHAPES[role][0]
        tensors.append(
            TensorInfo(
                index=output_index,
                name="{}_output".format(role),
                shape=(1, tokens, output_width),
                shape_signature=(1, tokens, output_width),
                dtype="INT16",
                buffer_index=0,
                buffer_bytes=0,
            )
        )
        outputs.append(output_index)
        operators.append(
            OperatorInfo(
                index=role_index,
                opcode=operator_name,
                inputs=(role_index, len(role_list) + role_index, -1),
                outputs=(output_index,),
                options_type="FullyConnectedOptions",
            )
        )

    if duplicate_q:
        original = tensors[len(role_list)]
        duplicate_index = len(tensors)
        duplicate_output = duplicate_index + 1
        tensors.append(
            TensorInfo(
                index=duplicate_index,
                name=original.name,
                shape=original.shape,
                shape_signature=original.shape_signature,
                dtype=original.dtype,
                buffer_index=8,
                buffer_bytes=original.buffer_bytes,
                quantization=original.quantization,
            )
        )
        tensors.append(
            TensorInfo(
                index=duplicate_output,
                name="duplicate_q_output",
                shape=tensors[14].shape,
                shape_signature=tensors[14].shape_signature,
                dtype="INT16",
                buffer_index=0,
                buffer_bytes=0,
            )
        )
        outputs.append(duplicate_output)
        operators.append(
            OperatorInfo(
                index=len(operators),
                opcode="FULLY_CONNECTED",
                inputs=(0, duplicate_index, -1),
                outputs=(duplicate_output,),
            )
        )

    graph = GraphInfo(
        index=0,
        name="synthetic_llama_layer",
        tensors=tuple(tensors),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        operators=tuple(operators),
    )
    return ModelInfo(
        source_path="synthetic",
        source_sha256="a" * 64,
        source_bytes=1234,
        version=3,
        buffer_count=9 if duplicate_q else 8,
        graphs=(graph,),
    )


def synthetic_transformer_model(reverse_operator_storage=False):
    tensors = []

    def tensor(name, shape, dtype="INT16", buffer_index=0, buffer_bytes=0, quantization=None):
        index = len(tensors)
        tensors.append(
            TensorInfo(
                index=index,
                name=name,
                shape=tuple(shape),
                shape_signature=tuple(shape),
                dtype=dtype,
                buffer_index=buffer_index,
                buffer_bytes=buffer_bytes,
                quantization=quantization,
            )
        )
        return index

    hidden = tensor("input_embeds", (1, 1, 4096))
    mask = tensor("mask", (1, 1, 1, 1025))
    position = tensor("pos_emb", (1, 2, 1, 128))
    past_key = tensor("past_keys_0", (1, 8, 1024, 128))
    past_value = tensor("past_values_0", (1, 8, 1024, 128))
    norm0 = tensor("layers/0/input_norm/output", (1, 1, 4096))
    operators = [OperatorInfo(0, "MTKEXT_RMS_NORMALIZATION", (hidden, -1, -1, -1), (norm0,))]
    role_names = {
        "q": "self_attn.q_proj",
        "k": "self_attn.k_proj",
        "v": "self_attn.v_proj",
        "o": "self_attn.o_proj",
        "gate": "mlp.gate_proj",
        "up": "mlp.up_proj",
        "down": "mlp.down_proj",
    }
    role_activations = {}
    role_outputs = {}
    role_operators = {}

    def add_projection(role, activation, output_shape):
        weight_shape = EXPECTED_WEIGHT_SHAPES[role]
        weight_index = tensor(
            "layers/0/{}/weight".format(role_names[role]),
            weight_shape,
            dtype="INT4",
            buffer_index=1 + len([item for item in tensors if item.dtype == "INT4"]),
            buffer_bytes=(weight_shape[0] * weight_shape[1] + 1) // 2,
            quantization=QuantizationInfo(scales=(0.01,), zero_points=(0,), quantized_dimension=0),
        )
        output = tensor("layers/0/{}/linear".format(role_names[role]), output_shape)
        op_index = len(operators)
        operators.append(OperatorInfo(op_index, "MTKEXT_FULLY_CONNECTED", (activation, weight_index, -1), (output,)))
        role_activations[role] = activation
        role_outputs[role] = output
        role_operators[role] = op_index
        return output

    q_linear = add_projection("q", norm0, (1, 1, 4096))
    k_linear = add_projection("k", norm0, (1, 1, 1024))
    v_linear = add_projection("v", norm0, (1, 1, 1024))
    q_rope = tensor("layers/0/self_attn/q_rope", (1, 32, 1, 128))
    k_rope = tensor("layers/0/self_attn/k_rope", (1, 8, 1, 128))
    operators.extend(
        [
            OperatorInfo(len(operators), "ADD", (q_linear, position), (q_rope,)),
            OperatorInfo(len(operators) + 1, "ADD", (k_linear, position), (k_rope,)),
        ]
    )
    scores = tensor("layers/0/self_attn/scores", (1, 32, 1, 1024))
    operators.append(OperatorInfo(len(operators), "BATCH_MATMUL", (q_rope, k_rope), (scores,)))
    masked = tensor("layers/0/self_attn/masked_scores", (1, 32, 1, 1025))
    operators.append(OperatorInfo(len(operators), "ADD", (scores, mask), (masked,)))
    probabilities = tensor("layers/0/self_attn/probabilities", (1, 32, 1, 1025))
    operators.append(OperatorInfo(len(operators), "SOFTMAX", (masked,), (probabilities,)))
    context = tensor("layers/0/self_attn/context", (1, 1, 4096))
    operators.append(OperatorInfo(len(operators), "BATCH_MATMUL", (probabilities, v_linear), (context,)))
    o_linear = add_projection("o", context, (1, 1, 4096))
    attn_residual = tensor("layers/0/attention_residual", (1, 1, 4096))
    operators.append(OperatorInfo(len(operators), "ADD", (hidden, o_linear), (attn_residual,)))
    norm1 = tensor("layers/0/post_attention_norm/output", (1, 1, 4096))
    operators.append(OperatorInfo(len(operators), "MTKEXT_RMS_NORMALIZATION", (attn_residual, -1, -1, -1), (norm1,)))
    gate_linear = add_projection("gate", norm1, (1, 1, 14336))
    up_linear = add_projection("up", norm1, (1, 1, 14336))
    gate_silu = tensor("layers/0/mlp/gate_silu", (1, 1, 14336))
    operators.append(OperatorInfo(len(operators), "MTKEXT_SILU", (gate_linear,), (gate_silu,)))
    swiglu = tensor("layers/0/mlp/swiglu", (1, 1, 14336))
    operators.append(OperatorInfo(len(operators), "MUL", (gate_silu, up_linear), (swiglu,)))
    down_linear = add_projection("down", swiglu, (1, 1, 4096))
    hidden_out = tensor("hidden_states_0", (1, 1, 4096))
    operators.append(OperatorInfo(len(operators), "ADD", (attn_residual, down_linear), (hidden_out,)))
    current_key = tensor("curr_keys_0", (1, 8, 1, 128))
    operators.append(OperatorInfo(len(operators), "IDENTITY", (past_key,), (current_key,)))
    current_value = tensor("curr_values_0", (1, 8, 1, 128))
    operators.append(OperatorInfo(len(operators), "IDENTITY", (past_value,), (current_value,)))

    graph_operators = tuple(reversed(operators)) if reverse_operator_storage else tuple(operators)
    graph = GraphInfo(
        index=0,
        name="synthetic_llama_transformer_layer",
        tensors=tuple(tensors),
        inputs=(hidden, mask, position, past_key, past_value),
        outputs=(hidden_out, current_key, current_value),
        operators=graph_operators,
    )
    return ModelInfo(
        source_path="synthetic_transformer",
        source_sha256="b" * 64,
        source_bytes=12345,
        version=3,
        buffer_count=8,
        graphs=(graph,),
    )


def synthetic_grouped_transformer_model(layer_count=2):
    """Build a sequential grouped graph with shared mask/position inputs."""
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    base_model = synthetic_transformer_model()
    base = base_model.graphs[0]
    tensors = []
    operators = []
    graph_inputs = []
    shared_map = {}
    for source_index in base.inputs[:3]:
        source = base.tensors[source_index]
        index = len(tensors)
        tensors.append(replace(source, index=index))
        shared_map[source_index] = index
        graph_inputs.append(index)

    previous_hidden = shared_map[base.inputs[0]]
    layer_kv_outputs = []
    past_input_names = ("past_keys_{}", "past_values_{}")
    for layer_index in range(layer_count):
        remap = dict(shared_map)
        remap[base.inputs[0]] = previous_hidden
        for source_index, name_template in zip(base.inputs[3:5], past_input_names):
            source = base.tensors[source_index]
            index = len(tensors)
            tensors.append(replace(source, index=index, name=name_template.format(layer_index)))
            remap[source_index] = index
            graph_inputs.append(index)

        for source in base.tensors:
            if source.index in remap:
                continue
            name = source.name.replace("layers/0/", "layers/{}/".format(layer_index))
            for stem in ("curr_keys", "curr_values", "hidden_states"):
                name = name.replace(stem + "_0", stem + "_{}".format(layer_index))
            index = len(tensors)
            tensors.append(replace(source, index=index, name=name))
            remap[source.index] = index

        for source_op in base.operators:
            operators.append(
                OperatorInfo(
                    index=len(operators),
                    opcode=source_op.opcode,
                    inputs=tuple(remap[index] if index >= 0 else index for index in source_op.inputs),
                    outputs=tuple(remap[index] for index in source_op.outputs),
                    options_type=source_op.options_type,
                )
            )
        previous_hidden = remap[base.outputs[0]]
        layer_kv_outputs.append((remap[base.outputs[1]], remap[base.outputs[2]]))

    graph = GraphInfo(
        index=0,
        name="synthetic_grouped_llama_transformer",
        tensors=tuple(tensors),
        inputs=tuple(graph_inputs),
        outputs=(previous_hidden,) + tuple(index for pair in layer_kv_outputs for index in pair),
        operators=tuple(operators),
    )
    return ModelInfo(
        source_path="synthetic_grouped_transformer",
        source_sha256="c" * 64,
        source_bytes=base_model.source_bytes * layer_count,
        version=3,
        buffer_count=base_model.buffer_count,
        graphs=(graph,),
    )


class DiscoveryTests(unittest.TestCase):
    def test_vector_reader_handles_flatbuffer_zero_sentinel(self):
        class FakeVectorReader:
            def ShapeLength(self):
                return 2

            def ShapeAsNumpy(self):
                return 0

            def Shape(self, index):
                return (1, 128)[index]

        self.assertEqual(_vector(FakeVectorReader(), "Shape"), (1, 128))

    def test_vector_reader_handles_empty_flatbuffer_vector(self):
        class FakeVectorReader:
            def ShapeSignatureLength(self):
                return 0

            def ShapeSignatureAsNumpy(self):
                return 0

            def ShapeSignature(self, index):
                raise AssertionError("empty vector accessor must not be called")

        self.assertEqual(_vector(FakeVectorReader(), "ShapeSignature"), ())

    def test_one_layer_discovers_all_roles_and_group_quant_metadata(self):
        model = synthetic_model(group_scales=True)
        report = discover_models([(model, "layer00_128t1024c.tflite")], expected_layers=1)
        self.assertTrue(report.sources[0].layers[0].complete)
        self.assertFalse(report.structural_pass)  # This projection-only fixture omits transformer semantics.
        self.assertEqual(report.coverage["P128"]["present_layers"], [0])
        layer = report.sources[0].layers[0]
        self.assertEqual(set(layer.roles), {"q", "k", "v", "o", "gate", "up", "down"})
        self.assertEqual(layer.roles["q"].quantization.scales, (0.01, 0.02, 0.03, 0.04))
        self.assertEqual(layer.roles["q"].dtype, "INT4")
        compact_quant = report.to_dict()["sources"][0]["layers"][0]["roles"]["q"]["weight_quantization"]
        full_quant = report.to_dict(include_quant_values=True)["sources"][0]["layers"][0]["roles"]["q"]["weight_quantization"]
        self.assertEqual(compact_quant["scale_count"], 4)
        self.assertNotIn("scales", compact_quant)
        self.assertEqual(full_quant["scales"], [0.01, 0.02, 0.03, 0.04])

    def test_vendor_fused_fully_connected_operator_is_discovered(self):
        report = discover_models(
            [(synthetic_model(operator_name="MTKEXT_FULLY_CONNECTED"), "layer00_128t1024c.tflite")],
            expected_layers=1,
        )
        self.assertTrue(report.sources[0].layers[0].complete)
        self.assertFalse(report.structural_pass)
        self.assertEqual(set(report.sources[0].layers[0].roles), {"q", "k", "v", "o", "gate", "up", "down"})

    def test_explicit_source_layer_overrides_single_local_layer_scope(self):
        model = synthetic_model(layer_id=0)
        report = discover_models([(model, "prefill/layer17_128t1024c.tflite")], expected_layers=18)
        self.assertEqual(report.sources[0].layers[0].layer_id, 17)
        self.assertEqual(report.sources[0].layers[0].roles["q"].semantic_layer_id, 0)
        self.assertIn(0, report.coverage["P128"]["missing_layers"])

    def test_full_32_layer_decode_coverage(self):
        entries = [
            (synthetic_model(layer_id=layer, tokens=1), "decode/layer{:02d}_1t.tflite".format(layer))
            for layer in range(32)
        ]
        report = discover_models(entries, expected_layers=32)
        self.assertEqual(report.coverage["D1"]["covered_count"], 32)
        self.assertEqual(report.coverage["D1"]["missing_layers"], [])
        self.assertTrue(report.coverage["D1"]["complete"])
        self.assertFalse(report.structural_pass)  # Coverage alone is not a semantic graph pass.

    def test_full_32_layer_prefill_coverage(self):
        entries = [
            (synthetic_model(layer_id=layer, tokens=128), "prefill/layer{:02d}_128t.tflite".format(layer))
            for layer in range(32)
        ]
        report = discover_models(entries, expected_layers=32)
        self.assertEqual(report.coverage["P128"]["covered_count"], 32)
        self.assertEqual(report.coverage["P128"]["missing_layers"], [])
        self.assertTrue(report.coverage["P128"]["complete"])
        self.assertFalse(report.structural_pass)  # Coverage alone is not a semantic graph pass.

    def test_grouped_32_layer_graph_discovers_all_transformer_layers(self):
        model = synthetic_grouped_transformer_model(layer_count=32)
        report = discover_models([(model, "grouped_decode_1t.tflite")], expected_layers=32)

        source = report.sources[0]
        self.assertTrue(report.structural_pass, report.to_dict())
        self.assertEqual([layer.layer_id for layer in source.layers], list(range(32)))
        self.assertEqual(len(source.graph_analyses), 32)
        self.assertTrue(all(layer.complete for layer in source.layers))
        self.assertTrue(all(layer.profile == "D1" for layer in source.layers))
        self.assertTrue(all(analysis.features["grouped_export"] for analysis in source.graph_analyses))
        self.assertEqual(report.coverage["D1"]["covered_count"], 32)
        self.assertEqual(report.coverage["D1"]["missing_layers"], [])
        self.assertEqual(
            [layer.roles["q"].semantic_layer_id for layer in source.layers],
            list(range(32)),
        )

    def test_duplicate_projection_role_fails_closed(self):
        report = discover_models([(synthetic_model(duplicate_q=True), "layer00_128t.tflite")], expected_layers=1)
        self.assertFalse(report.structural_pass)
        self.assertIn("PROJECTION_ROLE_AMBIGUOUS", [code for code, _ in report.issues])

    def test_incomplete_layer_coverage_fails_closed(self):
        report = discover_models([(synthetic_model(), "layer00_128t.tflite")], expected_layers=32)
        self.assertFalse(report.structural_pass)
        self.assertEqual(len(report.coverage["P128"]["missing_layers"]), 31)

    def test_profile_and_source_layer_inference(self):
        self.assertEqual(classify_profile(synthetic_model(tokens=128).graphs[0]), "P128")
        self.assertEqual(classify_profile(synthetic_model(tokens=1).graphs[0]), "D1")
        self.assertEqual(source_layer_id("prefill/layer_31_128t.tflite"), 31)
        self.assertEqual(
            source_layer_id(
                "incoming/auto_fetch/hash__Meta-Llama-3.1-8B-Instruct_1layer_1t1024c_15.tflite"
            ),
            15,
        )

    def test_transformer_features_and_dependencies_ignore_operator_storage_order(self):
        model = synthetic_transformer_model(reverse_operator_storage=True)
        report = discover_models([(model, "layer00_1t1024c.tflite")], expected_layers=1)
        self.assertTrue(report.structural_pass)
        analysis = report.sources[0].graph_analyses[0]
        self.assertTrue(all(analysis.features[name] for name in (
            "rmsnorm", "rotary", "attention", "swiglu", "residual", "mask", "position", "kv_cache", "layer_boundaries"
        )))
        self.assertEqual(len(analysis.evidence["residual_add_operators"]), 2)
        graph = model.graphs[0]
        past_key = next(tensor.index for tensor in graph.tensors if tensor.name == "past_keys_0")
        current_value = next(tensor.index for tensor in graph.tensors if tensor.name == "curr_values_0")
        self.assertEqual(analysis.input_roles["past_key"], (past_key,))
        self.assertEqual(analysis.output_roles["current_value"], (current_value,))
        self.assertNotEqual(graph.operators[0].index, graph.operators[-1].index)
        producer_index = analysis.tensor_producers[current_value]
        self.assertEqual(next(op.opcode for op in graph.operators if op.index == producer_index), "IDENTITY")
        self.assertIn(producer_index, analysis.topological_order)

    def test_graph_validation_ignores_operator_storage_order(self):
        tensors = (
            TensorInfo(0, "x", (1,), (1,), "INT8", 0, 0),
            TensorInfo(1, "middle", (1,), (1,), "INT8", 0, 0),
            TensorInfo(2, "y", (1,), (1,), "INT8", 0, 0),
        )
        graph = GraphInfo(
            0,
            "out_of_order",
            tensors,
            (0,),
            (2,),
            (
                OperatorInfo(1, "ADD", (1, -1), (2,)),
                OperatorInfo(0, "ADD", (0, -1), (1,)),
            ),
        )
        validate_graph(ModelInfo("fixture", "", 1, 3, 1, (graph,)), graph)

    def test_graph_validation_rejects_multiple_producers(self):
        tensors = (
            TensorInfo(0, "x", (1,), (1,), "INT8", 0, 0),
            TensorInfo(1, "y", (1,), (1,), "INT8", 0, 0),
        )
        graph = GraphInfo(
            0,
            "duplicate_producer",
            tensors,
            (0,),
            (1,),
            (
                OperatorInfo(0, "ADD", (0, -1), (1,)),
                OperatorInfo(1, "MUL", (0, -1), (1,)),
            ),
        )
        with self.assertRaises(ModelValidationError) as raised:
            validate_graph(ModelInfo("fixture", "", 1, 3, 1, (graph,)), graph)
        self.assertIn("TENSOR_HAS_MULTIPLE_PRODUCERS", [code for code, _ in raised.exception.issues])

    def test_graph_validation_rejects_dependency_cycle(self):
        tensors = (
            TensorInfo(0, "a", (1,), (1,), "INT8", 0, 0),
            TensorInfo(1, "b", (1,), (1,), "INT8", 0, 0),
        )
        graph = GraphInfo(
            0,
            "cycle",
            tensors,
            (),
            (1,),
            (
                OperatorInfo(0, "ADD", (1, -1), (0,)),
                OperatorInfo(1, "ADD", (0, -1), (1,)),
            ),
        )
        with self.assertRaises(ModelValidationError) as raised:
            validate_graph(ModelInfo("fixture", "", 1, 3, 1, (graph,)), graph)
        self.assertIn("GRAPH_CYCLE", [code for code, _ in raised.exception.issues])

    def test_packed_int4_payload_length_is_checked(self):
        model = synthetic_model()
        tensors = list(model.graphs[0].tensors)
        tensors[7] = TensorInfo(
            index=7,
            name=tensors[7].name,
            shape=tensors[7].shape,
            shape_signature=tensors[7].shape_signature,
            dtype="INT4",
            buffer_index=tensors[7].buffer_index,
            buffer_bytes=tensors[7].buffer_bytes - 1,
            quantization=tensors[7].quantization,
        )
        graph = GraphInfo(0, "bad_int4", tuple(tensors), model.graphs[0].inputs, model.graphs[0].outputs, model.graphs[0].operators)
        with self.assertRaises(ModelValidationError) as raised:
            validate_graph(model, graph)
        self.assertIn("INT4_BUFFER_LENGTH_MISMATCH", [code for code, _ in raised.exception.issues])

    def test_bad_identifier_fails_before_schema_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.tflite"
            path.write_bytes(b"not-a-tflite-file")
            with self.assertRaises(ToolError) as raised:
                read_tflite(path)
        self.assertEqual(raised.exception.code, "TFLITE_IDENTIFIER_INVALID")


if __name__ == "__main__":
    unittest.main()
