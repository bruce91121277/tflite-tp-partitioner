import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tflite_tp.cli import build_parser
from tflite_tp.discovery import discover_models
from tflite_tp.diagnostics import (
    AUTO_REPAIR_RULES,
    AutoRepairEngine,
    annotate_tp4_golden_diagnostics,
    classify_mismatch,
    diagnose_tp4_mismatch,
)
from tflite_tp.errors import ToolError
from tflite_tp.golden import (
    MTKConverterTFLiteRunner,
    _tp4_attention_intermediate_inputs,
    _tp4_candidate_attention_input_overrides,
    compare_named_outputs,
    derive_tp4_boundary_contract,
    probe_golden_backend,
    run_tp4_layer_golden,
    run_tp4_profile_golden,
)
from tflite_tp.fixtures import make_deterministic_tflite_inputs
from tflite_tp.pipeline import (
    _auto_repair_tp2_candidate,
    _auto_repair_tp4_candidate,
    _finalize_tp4_golden,
    _record_golden_execution_failure,
)
from tflite_tp.partition import derive_tp4_components
from tflite_tp.release import _validate_golden_report, evaluate_release, publish_release


class GoldenInputMappingTests(unittest.TestCase):
    def test_tp4_attention_uses_only_source_internal_inputs_for_selected_layers(self):
        graph = SimpleNamespace(
            tensors=(
                SimpleNamespace(index=0, name="input_embeds"),
                SimpleNamespace(index=1, name="layers/0/argument_1.1_dequantized"),
                SimpleNamespace(index=2, name="layers/1/argument_1.1_dequantized"),
            ),
            inputs=(0,),
        )
        manifest = {
            "models": [
                {
                    "profile": "D1",
                    "layer": 0,
                    "component": "attn",
                    "io": {"inputs": [
                        {"name": "input_embeds"},
                        {"name": "layers/0/argument_1.1_dequantized"},
                        {"name": "generated_unmatched"},
                    ]},
                },
                {
                    "profile": "D1",
                    "layer": 1,
                    "component": "attn",
                    "io": {"inputs": [{"name": "layers/1/argument_1.1_dequantized"}]},
                },
                {
                    "profile": "P128",
                    "layer": 0,
                    "component": "attn",
                    "io": {"inputs": [{"name": "layers/0/argument_1.1_dequantized"}]},
                },
                {
                    "profile": "D1",
                    "layer": 0,
                    "component": "mlp",
                    "io": {"inputs": [{"name": "layers/0/argument_1.1_dequantized"}]},
                },
            ]
        }

        names = _tp4_attention_intermediate_inputs(manifest, "D1", {0}, graph)

        self.assertEqual(names, ["layers/0/argument_1.1_dequantized"])

    def test_tp4_layer_chain_recomputes_float_attention_input_from_candidate_hidden(self):
        hidden_name = "layers/0/argument_1.1_dequantized"
        graph = SimpleNamespace(
            tensors=(
                SimpleNamespace(index=0, name="input_embeds", shape=(1, 1, 4)),
                SimpleNamespace(index=1, name=hidden_name, shape=(1, 1, 4)),
            ),
            inputs=(0,),
        )
        manifest = {
            "models": [{
                "profile": "D1",
                "layer": 1,
                "component": "attn",
                "io": {"inputs": [{"name": hidden_name}]},
            }]
        }
        candidate_hidden = np.array([[[2, 4, 6, 8]]], dtype=np.int16)

        class FakeRunner:
            def __init__(self):
                self.input_dtype = None

            def run(self, model_path, inputs, output_names=None):
                self.input_dtype = np.asarray(inputs["input_embeds"]).dtype
                return {
                    "outputs": {hidden_name: np.asarray(inputs["input_embeds"], dtype=np.float32) / 2},
                    "fixture_sha256": "a" * 64,
                }

        runner = FakeRunner()
        overrides, fixture_sha = _tp4_candidate_attention_input_overrides(
            manifest,
            "D1",
            1,
            graph,
            {1},
            "input_embeds",
            1,
            candidate_hidden,
            {"input_embeds": np.zeros((1, 1, 4), dtype=np.int16)},
            "layer01.tflite",
            runner,
        )

        self.assertEqual(runner.input_dtype, np.dtype(np.int16))
        self.assertEqual(overrides[hidden_name].dtype, np.dtype(np.float32))
        self.assertTrue(np.array_equal(overrides[hidden_name], candidate_hidden.astype(np.float32) / 2))
        self.assertEqual(fixture_sha, "a" * 64)


class MismatchClassificationTests(unittest.TestCase):
    def test_known_negative_fixture_categories(self):
        fixtures = (
            ({"message": "rank weight mismatch", "projection": "q"}, "WEIGHT_SLICE_ERROR"),
            ({"message": "INT4 nibble packing differs"}, "INT4_PACKING_ERROR"),
            ({"code": "QUANT_SCALE_LAYOUT_UNSUPPORTED"}, "QUANT_SCALE_MISMATCH"),
            ({"message": "zero point differs"}, "QUANT_ZERO_POINT_MISMATCH"),
            ({"code": "BIAS_OWNERSHIP_RECONSTRUCTION_MISMATCH"}, "BIAS_OWNERSHIP_ERROR"),
            ({"stage": "residual add/add"}, "RESIDUAL_DUPLICATION"),
            ({"code": "REQUANT_ROUNDING_MISMATCH"}, "REQUANT_ROUNDING_MISMATCH"),
            ({"code": "GATHER_ORDER_MISMATCH"}, "GATHER_CHANNEL_ORDER_ERROR"),
            ({"code": "MASK_SHAPE_MISMATCH"}, "MASK_SHAPE_MISMATCH"),
            ({"tensor_name": "position_rope_cos"}, "POSITION_ROPE_MISMATCH"),
            ({"tensor_name": "past_key_cache"}, "KV_LAYOUT_MISMATCH"),
            ({"message": "PAD4608 alignment mismatch"}, "PAD_ALIGNMENT_MISMATCH"),
            ({"message": "PAD6144 alignment mismatch"}, "PAD_ALIGNMENT_MISMATCH"),
            ({"operator": "unknown_custom_op"}, "UNSUPPORTED_OPERATOR_SEMANTICS"),
        )
        for record, expected in fixtures:
            with self.subTest(record=record):
                self.assertEqual(classify_mismatch(record)["category"], expected)

    def test_quantization_metadata_errors_are_classified_from_specific_evidence(self):
        cases = (
            ("QUANT_SCALE_INVALID", "QUANT_SCALE_MISMATCH"),
            ("QUANT_ZERO_POINT_COUNT_INVALID", "QUANT_ZERO_POINT_MISMATCH"),
            ("QUANT_RANGE_METADATA_LENGTH_MISMATCH", "QUANT_METADATA_MISMATCH"),
            ("QUANT_METADATA_LENGTH_MISMATCH", "QUANT_METADATA_MISMATCH"),
            ("QUANT_AXIS_OUT_OF_RANGE", "QUANT_METADATA_MISMATCH"),
        )
        for code, expected_category in cases:
            with self.subTest(code=code):
                diagnosis = classify_mismatch({"code": code})
                self.assertEqual(diagnosis["category"], expected_category)
                if expected_category == "QUANT_METADATA_MISMATCH":
                    self.assertIsNone(AUTO_REPAIR_RULES.get(expected_category))

    def test_rank_partial_at_dtype_endpoint_is_diagnosed_without_unsafe_repair(self):
        record = {
            "category": "OUTPUT_BIT_MISMATCH",
            "layer": 20,
            "rank": None,
            "stage": "DOWN_ROW_REDUCE",
            "operator": "down",
            "tensor_name": "layers/20/mlp/down_proj/linear",
            "expected": -10,
            "actual": -20,
            "partial_range_diagnostic": {
                "status": "RANGE_LIMIT_REACHED",
                "rank_limit_hits": [{
                    "rank": 0,
                    "dtype": "int32",
                    "dtype_min": -2147483648,
                    "dtype_max": 2147483647,
                    "values_at_dtype_min": 12,
                    "values_at_dtype_max": 0,
                }],
                "root_cause_confirmed": False,
                "safe_repair_available": False,
            },
        }

        diagnosis = classify_mismatch(record)

        self.assertEqual(diagnosis["category"], "PARTIAL_OUTPUT_RANGE_LIMIT_REACHED")
        self.assertFalse(diagnosis["classification_evidence"]["root_cause_confirmed"])
        self.assertFalse(diagnosis["classification_evidence"]["safe_repair_available"])
        self.assertEqual(diagnosis["coordinates"]["partial_range_diagnostic"], record["partial_range_diagnostic"])

    def test_coordinates_preserve_first_mismatch_evidence(self):
        record = {
            "layer": 21,
            "rank": 1,
            "stage": "attention",
            "operator": 17,
            "tensor_name": "layers/21/self_attn/mask",
            "projection": "q",
            "slice_range": [1024, 2048],
            "quant_domain": "FLOAT16",
            "dtype": "FLOAT16",
            "shape": [1, 1, 1, 1025],
            "first_mismatch_index": [0, 0, 0, 1024],
            "expected": 0.0,
            "actual": 1.0,
            "max_absolute_difference": 1.0,
            "mismatch_count": 1,
        }
        classified = classify_mismatch(record)
        self.assertEqual(classified["category"], "MASK_SHAPE_MISMATCH")
        self.assertEqual(classified["coordinates"]["first_mismatch_index"], [0, 0, 0, 1024])
        self.assertEqual(classified["coordinates"]["expected"], 0.0)
        self.assertEqual(classified["coordinates"]["actual"], 1.0)

    def test_one_unit_quantized_output_mismatch_gets_projection_slice_diagnosis(self):
        record = {
            "category": "OUTPUT_BIT_MISMATCH",
            "layer": 0,
            "rank": 3,
            "stage": "V",
            "operator": "attn",
            "tensor_name": "boundary",
            "candidate_tensor": "layers/0/self_attn/v_proj/linear",
            "dtype": "int16",
            "quant_domain": {"dtype": "INT16", "scales": [0.125], "zero_points": [0]},
            "shape": [1, 128, 256],
            "first_mismatch_index": [0, 56, 188],
            "expected": 10,
            "actual": 11,
            "max_absolute_difference": 1.0,
            "mismatch_count": 1,
        }
        layer_manifest = {
            "weight_partitions": {
                "v": {
                    "source_tensor": 21,
                    "source_buffer": 10,
                    "source_shape": [1024, 4096],
                    "slice_axis": 0,
                    "slice_axis_name": "output_channels",
                    "slice_ranges": [
                        {"rank": 0, "start": 0, "stop": 256},
                        {"rank": 1, "start": 256, "stop": 512},
                        {"rank": 2, "start": 512, "stop": 768},
                        {"rank": 3, "start": 768, "stop": 1024},
                    ],
                    "source_sha256": "a" * 64,
                    "weight_shards": [{"rank": 3, "sha256": "b" * 64}],
                    "reconstruction_equal": True,
                }
            }
        }
        diagnosed = diagnose_tp4_mismatch(record, layer_manifest)
        self.assertEqual(diagnosed["category"], "REQUANT_ROUNDING_MISMATCH")
        self.assertEqual(diagnosed["projection"], "v")
        self.assertEqual(diagnosed["slice_range"], [768, 1024])
        self.assertEqual(diagnosed["source_slice"]["source_tensor_id"], 21)
        self.assertEqual(diagnosed["source_slice"]["rank_weight_sha256"], "b" * 64)
        self.assertTrue(diagnosed["source_slice"]["reconstruction_equal"])
        self.assertFalse(diagnosed["classification_evidence"]["root_cause_confirmed"])

    def test_one_unit_quantized_mismatch_fails_closed_without_safe_repair(self):
        source_sha = "c" * 64
        record = {
            "category": "OUTPUT_BIT_MISMATCH",
            "layer": 0,
            "rank": 0,
            "stage": "Q",
            "operator": "attn",
            "tensor_name": "boundary",
            "candidate_tensor": "layers/0/self_attn/q_proj/linear",
            "dtype": "int16",
            "quant_domain": {"dtype": "INT16", "scales": [0.25], "zero_points": [0]},
            "first_mismatch_index": [0, 52, 884],
            "expected": -1,
            "actual": -2,
            "max_absolute_difference": 1.0,
            "mismatch_count": 1,
            "source_sha256": source_sha,
        }
        calls = []
        result = AutoRepairEngine(max_repairs=2).run(
            {
                "numerical_status": "NUMERICAL_FAIL",
                "mismatch_count": 1,
                "source_sha256": source_sha,
                "fixture_sha256": "f" * 64,
                "mismatches": [record],
            },
            lambda *_args, **_kwargs: calls.append("rebuild"),
            lambda *_args, **_kwargs: calls.append("verify"),
        )
        self.assertEqual(result["status"], "REPAIR_BLOCKED")
        self.assertEqual(result["failure"], "NO_SAFE_REPAIR_RULE_FOR_REQUANT_ROUNDING_MISMATCH")
        self.assertEqual(result["diagnosed_mismatch"]["projection"], "q")
        self.assertEqual(calls, [])

    def test_tp4_public_kv_mismatch_is_linked_to_rank_boundary_slice(self):
        normalized_source_sha = "c" * 64
        boundary = {
            "category": "OUTPUT_BIT_MISMATCH",
            "layer": 0,
            "rank": 3,
            "stage": "KV_OUTPUT",
            "operator": "attn",
            "tensor_name": "boundary",
            "candidate_tensor": "curr_values_0",
            "dtype": "int16",
            "quant_domain": {"dtype": "INT16", "scales": [0.125], "zero_points": [0]},
            "first_mismatch_index": [0, 1, 56, 60],
            "expected": 10,
            "actual": 11,
            "max_absolute_difference": 1.0,
            "mismatch_count": 1,
            "source_sha256": normalized_source_sha,
        }
        public_output = {
            "category": "OUTPUT_BIT_MISMATCH",
            "tensor_name": "curr_values_0",
            "first_mismatch_index": [0, 1, 56, 60],
            "expected": 10,
            "actual": 11,
            "max_absolute_difference": 1.0,
            "mismatch_count": 1,
            "source_sha256": normalized_source_sha,
        }
        layer_manifest = {
            "layer": 0,
            "source_sha256": "a" * 64,
            "weight_partitions": {
                "v": {
                    "source_tensor": 21,
                    "source_buffer": 10,
                    "source_shape": [1024, 4096],
                    "slice_axis": 0,
                    "slice_axis_name": "output_channels",
                    "slice_ranges": [{"rank": 3, "start": 768, "stop": 1024}],
                    "source_sha256": "d" * 64,
                    "weight_shards": [{"rank": 3, "sha256": "e" * 64}],
                    "reconstruction_equal": True,
                }
            },
        }
        report = {"mismatches": [boundary, public_output], "layer_reports": []}
        annotate_tp4_golden_diagnostics(
            report,
            {0: layer_manifest},
            source_sha256_to_layer={normalized_source_sha: 0},
        )
        diagnosed_output = report["mismatches"][1]
        self.assertEqual(diagnosed_output["layer"], 0)
        self.assertEqual(diagnosed_output["affected_ranks"], [3])
        self.assertEqual(diagnosed_output["diagnostic_category"], "REQUANT_ROUNDING_MISMATCH")
        self.assertEqual(diagnosed_output["diagnosis"]["source_slices"][0]["slice_range"], [768, 1024])
        self.assertEqual(report["diagnostic_summary"]["mismatch_count"], 2)

    def test_fault_injected_tp_mismatches_are_detected_classified_and_retested(self):
        source_sha = "e" * 64
        scenarios = (
            (
                "rank_weight_swap",
                np.array([1, 2, 3, 4], dtype=np.int8),
                np.array([3, 4, 1, 2], dtype=np.int8),
                {"tensor_name": "layers/0/q_proj/weight", "stage": "Q_WEIGHT", "projection": "q"},
                "WEIGHT_SLICE_ERROR",
            ),
            (
                "quant_scale_change",
                np.array([0.25], dtype=np.float32),
                np.array([0.5], dtype=np.float32),
                {"tensor_name": "layers/0/q_proj/scale", "stage": "Q_WEIGHT", "quant_field": "scale"},
                "QUANT_SCALE_MISMATCH",
            ),
            (
                "duplicated_bias",
                np.array([10, 11], dtype=np.int16),
                np.array([14, 15], dtype=np.int16),
                {"tensor_name": "layers/0/o_proj/bias_output", "stage": "O_BIAS"},
                "BIAS_OWNERSHIP_ERROR",
            ),
            (
                "gather_rank_order",
                np.arange(8, dtype=np.int16),
                np.array([4, 5, 6, 7, 0, 1, 2, 3], dtype=np.int16),
                {"tensor_name": "layers/0/all_gather_output", "stage": "AG1_GATHER"},
                "GATHER_CHANNEL_ORDER_ERROR",
            ),
            (
                "mask_shape",
                np.zeros((1, 1, 1, 4), dtype=np.int16),
                np.zeros((1, 1, 1, 3), dtype=np.int16),
                {"tensor_name": "layers/0/attention_mask", "stage": "ATTENTION_MASK"},
                "MASK_SHAPE_MISMATCH",
            ),
            (
                "kv_position_shift",
                np.arange(4, dtype=np.int16),
                np.roll(np.arange(4, dtype=np.int16), 1),
                {"tensor_name": "layers/0/current_key_cache_position", "stage": "KV_CACHE_UPDATE"},
                "KV_LAYOUT_MISMATCH",
            ),
        )

        for name, expected_values, mutated_values, coordinates, expected_category in scenarios:
            with self.subTest(fault=name):
                comparison = compare_named_outputs(
                    {"fault_boundary": expected_values},
                    {"fault_boundary": mutated_values},
                )
                self.assertGreater(comparison["mismatch_count"], 0)
                mismatch = dict(comparison["mismatches"][0])
                mismatch.update(coordinates)
                mismatch.update({"layer": 0, "rank": 1, "operator": 7, "mismatch_count": comparison["mismatch_count"]})
                classified = classify_mismatch(mismatch)
                self.assertEqual(classified["category"], expected_category)

                repair_calls = []

                def rebuild(_mismatch, expected_source_sha256, repair_rule):
                    repair_calls.append((expected_source_sha256, repair_rule))
                    return {
                        "rebuilt_from_canonical": True,
                        "repair_applied": True,
                        "source_sha256": source_sha,
                        "repair_rule": repair_rule,
                        "explanation": "Rebuilt this fault fixture from its canonical expected boundary.",
                        "repair_evidence": {
                            "action": "REBUILD_RANK_COMPONENTS_FROM_CANONICAL",
                            "requested_rule": repair_rule,
                            "candidate_inventory_before_sha256": "b" * 64,
                            "candidate_inventory_after_sha256": "c" * 64,
                            "changed_candidate_artifact_count": 1,
                        },
                        "outputs": {"fault_boundary": expected_values.copy()},
                    }

                def verify_again(rebuilt, _mismatch):
                    retest = compare_named_outputs(
                        {"fault_boundary": expected_values}, rebuilt["outputs"]
                    )
                    return {
                        "source_sha256": source_sha,
                        "fixture_sha256": "f" * 64,
                        "numerical_status": "NUMERICAL_PASS" if retest["mismatch_count"] == 0 else "NUMERICAL_FAIL",
                        "mismatch_count": retest["mismatch_count"],
                        "mismatches": retest["mismatches"],
                    }

                report = AutoRepairEngine(max_repairs=1).run(
                    {
                        "source_sha256": source_sha,
                        "fixture_sha256": "f" * 64,
                        "numerical_status": "NUMERICAL_FAIL",
                        "mismatch_count": comparison["mismatch_count"],
                        "mismatches": [mismatch],
                    },
                    rebuild,
                    verify_again,
                )
                self.assertEqual(report["status"], "REPAIRED_AND_VERIFIED")
                self.assertEqual(len(repair_calls), 1)
                self.assertEqual(repair_calls[0][0], source_sha)
                self.assertEqual(report["final_report"]["mismatch_count"], 0)


class AutoRepairTests(unittest.TestCase):
    def test_tp2_auto_rebuild_retests_partial_probe_without_promoting_candidate(self):
        import json
        import tflite_tp.pipeline as pipeline

        source_sha = "b" * 64
        fixture_sha = "f" * 64
        source_row = {"source": "layer00.tflite", "source_sha256": source_sha, "profiles": ["D1"]}
        source_report = SimpleNamespace(
            to_dict=lambda include_quant_values=False: {"sources": [source_row]}
        )
        initial_golden = {
            "source_sha256_by_name": {"layer00.tflite": source_sha},
            "fixture_sha256_by_layer": {"0:1": fixture_sha},
            "numerical_status": "NUMERICAL_FAIL",
            "mismatch_count": 1,
            "mismatches": [{
                "topology": "TP2_ROW_REDUCE",
                "source": "layer00.tflite",
                "source_sha256": source_sha,
                "layer": 0,
                "rank": None,
                "stage": "O_ROW_REDUCE",
                "operator": "o",
                "tensor_name": "layers/0/self_attn/o_proj/output",
                "expected": 7,
                "actual": 8,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            candidate_root = Path(directory) / "candidate"
            candidate_root.mkdir()
            original_payload = b"stale-tp2-rank-component"
            (candidate_root / "model.tflite").write_bytes(original_payload)
            (candidate_root / "TP2_ROW_REDUCE_GOLDEN.json").write_text(
                json.dumps(initial_golden), encoding="utf-8"
            )
            manifest = {
                "profile": "D1",
                "topology": "TP2_ROW_REDUCE",
                "expected_layers": 1,
                "golden_report_path": "TP2_ROW_REDUCE_GOLDEN.json",
                "checksums_file": "SHA256SUMS",
                "layer_manifest_sha256": {},
                "collective_contract": {"quantized_partial_semantics": {"status": "UNPROVEN"}},
                "source_audit": {"sources": [source_row]},
                "models": [{
                    "path": "model.tflite",
                    "output_sha256": hashlib.sha256(original_payload).hexdigest(),
                }],
            }

            def rebuild(_source_path, output_directory, **_kwargs):
                output_directory.mkdir()
                rebuilt_payload = b"canonical-rebuilt-tp2-rank-component"
                (output_directory / "model.tflite").write_bytes(rebuilt_payload)
                return {
                    "profile": "D1",
                    "topology": "TP2_ROW_REDUCE",
                    "expected_layers": 1,
                    "source_audit": {"sources": [source_row]},
                    "collective_contract": {"quantized_partial_semantics": {"status": "UNPROVEN"}},
                    "layer_manifest_sha256": {},
                    "checksums_file": "SHA256SUMS",
                    "models": [{
                        "path": "model.tflite",
                        "output_sha256": hashlib.sha256(rebuilt_payload).hexdigest(),
                    }],
                }

            partial_probe_pass = {
                "source_sha256_by_name": {"layer00.tflite": source_sha},
                "fixture_sha256_by_layer": {"0:1": fixture_sha},
                "numerical_status": "NUMERICAL_INCOMPLETE",
                "mismatch_count": 0,
                "mismatches": [],
                "row_reduce_semantics_status": "UNPROVEN",
                "golden_coverage": {
                    "complete": False,
                    "selected_layers": [0],
                    "checked_boundary_count": 2,
                    "expected_row_reduce_boundary_count": 2,
                },
            }
            with patch.object(pipeline, "build_tp2_directory", side_effect=rebuild), patch.object(
                pipeline, "run_tp2_row_reduce_profile_golden", return_value=partial_probe_pass
            ):
                finalized = _auto_repair_tp2_candidate(
                    "canonical-d1",
                    candidate_root,
                    manifest,
                    source_report,
                    "D1",
                    1,
                    object(),
                    2,
                )

            original_bytes_after = (candidate_root / "model.tflite").read_bytes()
            repair_root = Path(finalized["auto_repair_report"]["candidate_roots"][0])
            rebuilt_bytes = (repair_root / "model.tflite").read_bytes()
            report_path = candidate_root / finalized["auto_repair_report_path"]
            report_exists = report_path.is_file()

        report = finalized["auto_repair_report"]
        self.assertEqual(report["status"], "REBUILT_AND_RETESTED_PARTIAL_SCOPE")
        self.assertFalse(report["candidate_promoted"])
        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["final_report"]["mismatch_count"], 0)
        self.assertEqual(report["final_report"]["numerical_status"], "NUMERICAL_INCOMPLETE")
        self.assertTrue(report["final_report"]["row_reduce_probe_exact"])
        self.assertEqual(original_bytes_after, original_payload)
        self.assertEqual(rebuilt_bytes, b"canonical-rebuilt-tp2-rank-component")
        self.assertTrue(report_exists)
        self.assertEqual(finalized["release_status"], "NOT_ELIGIBLE")

    def test_pipeline_promotes_only_canonical_rebuild_that_passes_golden(self):
        import json
        import tflite_tp.pipeline as pipeline

        source_sha = "a" * 64
        source_row = {"source": "layer00.tflite", "source_sha256": source_sha, "profiles": ["D1"]}
        source_report = SimpleNamespace(
            to_dict=lambda include_quant_values=False: {"sources": [source_row]}
        )
        initial_golden = {
            "source_sha256": source_sha,
            "fixture_sha256_by_layer": {"0": "f" * 64},
            "numerical_status": "NUMERICAL_FAIL",
            "mismatch_count": 1,
            "mismatches": [{
                "source_sha256": source_sha,
                "layer": 0,
                "rank": 0,
                "stage": "Q",
                "operator": "attn",
                "tensor_name": "layers/0/q_proj/weight",
                "projection": "q",
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            candidate_root = Path(directory) / "candidate"
            candidate_root.mkdir()
            old_payload = b"stale-rank-component"
            (candidate_root / "model.tflite").write_bytes(old_payload)
            (candidate_root / "GOLDEN_REPORT.json").write_text(json.dumps(initial_golden), encoding="utf-8")
            manifest = {
                "profile": "D1",
                "expected_layers": 1,
                "golden_report_path": "GOLDEN_REPORT.json",
                "layer_manifest_sha256": {},
                "checksums_file": "SHA256SUMS",
                "models": [{
                    "path": "model.tflite",
                    "output_sha256": hashlib.sha256(old_payload).hexdigest(),
                }],
            }

            def rebuild(source_path, output_directory, **kwargs):
                output_directory.mkdir()
                new_payload = b"canonical-rebuilt-rank-component"
                (output_directory / "model.tflite").write_bytes(new_payload)
                rebuilt = {
                    "profile": "D1",
                    "topology": "TP4_FOUR_STAGE_ALL_GATHER",
                    "expected_layers": 1,
                    "source_audit": {"sources": [source_row]},
                    "layer_manifest_sha256": {},
                    "checksums_file": "SHA256SUMS",
                    "models": [{
                        "path": "model.tflite",
                        "output_sha256": hashlib.sha256(new_payload).hexdigest(),
                    }],
                }
                return rebuilt

            golden_pass = {
                "fixture_sha256_by_layer": {"0": "f" * 64},
                "numerical_status": "NUMERICAL_PASS",
                "mismatch_count": 0,
                "mismatches": [],
                "backend": "synthetic",
            }
            with patch.object(pipeline, "build_tp4_directory", side_effect=rebuild), patch.object(
                pipeline, "run_tp4_profile_golden", return_value=golden_pass
            ):
                finalized = _auto_repair_tp4_candidate(
                    "canonical-d1",
                    candidate_root,
                    manifest,
                    source_report,
                    "D1",
                    1,
                    object(),
                    2,
                )

            report_path = candidate_root / finalized["auto_repair_report_path"]
            checksums = (candidate_root / "SHA256SUMS").read_text(encoding="utf-8")
            backup = Path(str(candidate_root) + ".pre-repair")
            report_exists = report_path.is_file()
            backup_exists = backup.is_dir()

        self.assertEqual(finalized["auto_repair_report"]["status"], "REPAIRED_AND_VERIFIED")
        self.assertTrue(backup_exists)
        self.assertTrue(report_exists)
        self.assertIn(finalized["auto_repair_report_sha256"], checksums)

    def test_rebuild_from_canonical_is_retested_before_pass(self):
        source_sha = "a" * 64
        initial = {
            "source_sha256": source_sha,
            "fixture_sha256": "f" * 64,
            "numerical_status": "NUMERICAL_FAIL",
            "mismatch_count": 1,
            "mismatches": [{
                "layer": 0,
                "rank": 1,
                "stage": "attention",
                "operator": 3,
                "tensor_name": "layers/0/self_attn/q_proj/weight",
                "projection": "q",
                "message": "rank weight mismatch",
            }],
        }
        calls = []

        def rebuild(mismatch, expected_source_sha256, repair_rule):
            calls.append(("rebuild", mismatch["category"]))
            return {
                "rebuilt_from_canonical": True,
                "repair_applied": True,
                "source_sha256": expected_source_sha256,
                "repair_rule": repair_rule,
                "explanation": "Regenerated the Q shard from the canonical output-channel range.",
                "repair_evidence": {
                    "action": "REBUILD_RANK_COMPONENTS_FROM_CANONICAL",
                    "requested_rule": repair_rule,
                    "candidate_inventory_before_sha256": "b" * 64,
                    "candidate_inventory_after_sha256": "c" * 64,
                    "changed_candidate_artifact_count": 1,
                },
            }

        def verify(rebuild_result, mismatch):
            calls.append(("verify", mismatch["category"]))
            return {
                "source_sha256": source_sha,
                "fixture_sha256": "f" * 64,
                "numerical_status": "NUMERICAL_PASS",
                "mismatch_count": 0,
                "mismatches": [],
            }

        report = AutoRepairEngine(max_repairs=2).run(initial, rebuild, verify)
        self.assertEqual(report["status"], "REPAIRED_AND_VERIFIED")
        self.assertEqual(calls, [("rebuild", "WEIGHT_SLICE_ERROR"), ("verify", "WEIGHT_SLICE_ERROR")])
        self.assertTrue(report["attempts"][0]["retested"])
        self.assertEqual(report["attempts"][0]["after_mismatch_count"], 0)
        self.assertEqual(report["attempts"][0]["fixture_identity_field"], "fixture_sha256")
        self.assertEqual(report["attempts"][0]["fixture_evidence_sha256"], "f" * 64)

    def test_rebuild_without_changed_candidate_artifacts_cannot_be_called_repair(self):
        source_sha = "a" * 64
        initial = {
            "source_sha256": source_sha,
            "fixture_sha256": "f" * 64,
            "numerical_status": "NUMERICAL_FAIL",
            "mismatch_count": 1,
            "mismatches": [{
                "layer": 0,
                "rank": 0,
                "stage": "Q",
                "operator": "attn",
                "tensor_name": "layers/0/q/weight",
                "projection": "q",
            }],
        }
        retests = []

        def rebuild(_mismatch, expected_source_sha256, repair_rule):
            return {
                "rebuilt_from_canonical": True,
                "repair_applied": False,
                "source_sha256": expected_source_sha256,
                "repair_rule": repair_rule,
                "explanation": "The canonical rebuild produced identical component files.",
                "repair_evidence": {
                    "action": "REBUILD_RANK_COMPONENTS_FROM_CANONICAL",
                    "requested_rule": repair_rule,
                    "candidate_inventory_before_sha256": "b" * 64,
                    "candidate_inventory_after_sha256": "b" * 64,
                    "changed_candidate_artifact_count": 0,
                },
            }

        report = AutoRepairEngine(max_repairs=1).run(
            initial,
            rebuild,
            lambda *_args: retests.append(True),
        )
        self.assertEqual(report["status"], "REPAIR_FAILED")
        self.assertEqual(report["failure"], "REBUILD_DID_NOT_CHANGE_CANDIDATE_ARTIFACTS")
        self.assertEqual(retests, [])
        self.assertFalse(report["attempts"][0]["applied"])

    def test_repair_fails_closed_when_initial_fixture_identity_is_missing(self):
        initial = {
            "source_sha256": "a" * 64,
            "numerical_status": "NUMERICAL_FAIL",
            "mismatch_count": 1,
            "mismatches": [{
                "layer": 0,
                "rank": 0,
                "stage": "attention",
                "operator": 3,
                "tensor_name": "layers/0/self_attn/q_proj/weight",
            }],
        }
        callbacks = []
        report = AutoRepairEngine(max_repairs=1).run(
            initial,
            lambda *_args, **_kwargs: callbacks.append("rebuild"),
            lambda *_args, **_kwargs: callbacks.append("verify"),
        )
        self.assertEqual(report["status"], "REPAIR_BLOCKED")
        self.assertEqual(report["failure"], "GOLDEN_FIXTURE_IDENTITY_MISSING")
        self.assertEqual(callbacks, [])

    def test_changed_retest_fixture_cannot_be_reported_as_repaired(self):
        source_sha = "a" * 64
        mismatch = {
            "layer": 0,
            "rank": 1,
            "stage": "attention",
            "operator": 3,
            "tensor_name": "layers/0/self_attn/q_proj/weight",
        }
        initial = {
            "source_sha256": source_sha,
            "fixture_sha256_by_layer": {"0": "f" * 64},
            "numerical_status": "NUMERICAL_FAIL",
            "mismatch_count": 1,
            "mismatches": [mismatch],
        }

        def rebuild(_mismatch, expected_source_sha256, repair_rule):
            return {
                "rebuilt_from_canonical": True,
                "repair_applied": True,
                "source_sha256": expected_source_sha256,
                "repair_rule": repair_rule,
                "explanation": "Rebuilt from the canonical graph and recorded weight ranges.",
                "repair_evidence": {
                    "action": "REBUILD_RANK_COMPONENTS_FROM_CANONICAL",
                    "requested_rule": repair_rule,
                    "candidate_inventory_before_sha256": "b" * 64,
                    "candidate_inventory_after_sha256": "c" * 64,
                    "changed_candidate_artifact_count": 1,
                },
            }

        def verify(_rebuild, _mismatch):
            return {
                "source_sha256": source_sha,
                "fixture_sha256_by_layer": {"0": "e" * 64},
                "numerical_status": "NUMERICAL_PASS",
                "mismatch_count": 0,
                "mismatches": [],
            }

        report = AutoRepairEngine(max_repairs=1).run(initial, rebuild, verify)
        self.assertEqual(report["status"], "REPAIR_FAILED")
        self.assertEqual(report["failure"], "RETEST_FIXTURE_IDENTITY_MISMATCH")
        self.assertEqual(report["attempts"][0]["fixture_identity_field"], "fixture_sha256_by_layer")
        self.assertNotEqual(
            report["attempts"][0]["fixture_evidence_sha256"],
            report["attempts"][0]["after_fixture_evidence_sha256"],
        )

    def test_repair_fails_closed_without_full_graph_coordinates(self):
        initial = {
            "source_sha256": "a" * 64,
            "fixture_sha256": "f" * 64,
            "numerical_status": "NUMERICAL_FAIL",
            "mismatch_count": 1,
            "mismatches": [{"stage": "attention", "tensor_name": "q"}],
        }
        report = AutoRepairEngine(max_repairs=3).run(initial, lambda *_a, **_k: {}, lambda *_a: {})
        self.assertEqual(report["status"], "REPAIR_BLOCKED")
        self.assertEqual(report["failure"], "MISMATCH_SOURCE_MAPPING_INCOMPLETE")
        self.assertEqual(report["attempts"], [])

    def test_repair_attempt_limit_is_enforced(self):
        source_sha = "b" * 64
        mismatch = {
            "layer": 0,
            "rank": 0,
            "stage": "mask",
            "operator": 9,
            "tensor_name": "mask",
            "message": "mask shape mismatch",
        }
        initial = {"source_sha256": source_sha, "fixture_sha256": "f" * 64, "numerical_status": "NUMERICAL_FAIL", "mismatch_count": 1, "mismatches": [mismatch]}
        verifies = []

        def rebuild(_mismatch, expected_source_sha256, repair_rule):
            return {
                "rebuilt_from_canonical": True,
                "repair_applied": True,
                "source_sha256": expected_source_sha256,
                "repair_rule": repair_rule,
                "explanation": "Rebuilt mask mapping from the source tensor contract.",
                "repair_evidence": {
                    "action": "REBUILD_RANK_COMPONENTS_FROM_CANONICAL",
                    "requested_rule": repair_rule,
                    "candidate_inventory_before_sha256": "b" * 64,
                    "candidate_inventory_after_sha256": "c" * 64,
                    "changed_candidate_artifact_count": 1,
                },
            }

        def verify(_rebuild, _mismatch):
            verifies.append(True)
            return {"source_sha256": source_sha, "fixture_sha256": "f" * 64, "numerical_status": "NUMERICAL_FAIL", "mismatch_count": 1, "mismatches": [mismatch]}

        report = AutoRepairEngine(max_repairs=1).run(initial, rebuild, verify)
        self.assertEqual(report["status"], "REPAIR_LIMIT_REACHED")
        self.assertEqual(len(report["attempts"]), 1)
        self.assertEqual(len(verifies), 1)

    def test_invalid_max_repairs_is_rejected(self):
        with self.assertRaises(ToolError):
            AutoRepairEngine(max_repairs=-1)

    def test_unknown_requant_rounding_is_not_auto_modified(self):
        initial = {
            "source_sha256": "d" * 64,
            "fixture_sha256": "f" * 64,
            "numerical_status": "NUMERICAL_FAIL",
            "mismatch_count": 1,
            "mismatches": [{
                "layer": 0,
                "rank": 0,
                "stage": "o",
                "operator": 4,
                "tensor_name": "layers/0/self_attn/o_proj/output",
                "message": "requant rounding mismatch",
            }],
        }
        called = []
        report = AutoRepairEngine(max_repairs=3).run(
            initial,
            lambda *_args, **_kwargs: called.append(True),
            lambda *_args, **_kwargs: {},
        )
        self.assertEqual(report["status"], "REPAIR_BLOCKED")
        self.assertEqual(report["failure"], "NO_SAFE_REPAIR_RULE_FOR_REQUANT_ROUNDING_MISMATCH")
        self.assertEqual(called, [])

    def test_ambiguous_quantization_metadata_is_not_auto_modified(self):
        initial = {
            "source_sha256": "c" * 64,
            "fixture_sha256": "f" * 64,
            "numerical_status": "NUMERICAL_FAIL",
            "mismatch_count": 1,
            "mismatches": [{
                "code": "QUANT_METADATA_LENGTH_MISMATCH",
                "layer": 0,
                "rank": 0,
                "stage": "Q_WEIGHT",
                "operator": 4,
                "tensor_name": "layers/0/self_attn/q_proj/weight",
                "message": "quantization metadata vector lengths differ",
            }],
        }
        called = []
        report = AutoRepairEngine(max_repairs=3).run(
            initial,
            lambda *_args, **_kwargs: called.append("rebuild"),
            lambda *_args, **_kwargs: called.append("verify"),
        )
        self.assertEqual(report["status"], "REPAIR_BLOCKED")
        self.assertEqual(report["failure"], "NO_SAFE_REPAIR_RULE_FOR_QUANT_METADATA_MISMATCH")
        self.assertEqual(report["diagnosed_mismatch"]["category"], "QUANT_METADATA_MISMATCH")
        self.assertEqual(called, [])


class ReleaseGateTests(unittest.TestCase):
    def test_release_golden_requires_fixture_identity_for_every_layer(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_sources = {"layer00.tflite": "a" * 64}
            report = {
                "numerical_status": "NUMERICAL_PASS",
                "mismatch_count": 0,
                "source_sha256_by_name": expected_sources,
                "fixture_sha256_by_layer": {"0": "f" * 64},
            }
            report_path = root / "GOLDEN_REPORT.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            manifest = {
                "golden_report_path": report_path.name,
                "golden_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "expected_layers": 2,
            }
            failure = _validate_golden_report(
                root,
                manifest,
                expected_sources,
                4,
                ("attn", "o", "suffix_attn", "mlp", "down", "suffix_final"),
            )
        self.assertEqual(failure, "GOLDEN_FIXTURE_COVERAGE_INCOMPLETE")

    def test_golden_unavailable_fails_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "candidate"
            root.mkdir()
            rows = []
            checksums = []
            components = ("attn", "o", "suffix_attn", "mlp", "down", "suffix_final")
            for rank in range(4):
                for component in components:
                    relative = "rank{}/{}.tflite".format(rank, component)
                    model = root / relative
                    model.parent.mkdir(parents=True, exist_ok=True)
                    model.write_bytes(("candidate graph {} {}".format(rank, component)).encode("ascii"))
                    digest = hashlib.sha256(model.read_bytes()).hexdigest()
                    checksums.append("{}  {}".format(digest, relative))
                    rows.append({
                        "path": relative,
                        "source_sha256": "c" * 64,
                        "output_sha256": digest,
                        "structural_status": "STRUCTURAL_PASS",
                        "layer": 0,
                        "rank": rank,
                        "component": component,
                    })
            (root / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
            manifest = {
                "topology": "TP4_FOUR_STAGE_ALL_GATHER",
                "source": "layer00.tflite",
                "source_sha256": "c" * 64,
                "expected_layers": 1,
                "model_count": len(rows),
                "rank_order": [0, 1, 2, 3],
                "collective_contract": {
                    "all_gather_boundaries": [
                        {"stage": "AG0"},
                        {"stage": "AG1"},
                        {"stage": "AG2"},
                        {"stage": "AG3"},
                    ],
                    "row_reduce_boundaries": [],
                    "kv_cache_policy": "RANK_LOCAL",
                },
                "structural_status": "STRUCTURAL_PASS",
                "source_coverage": {"complete": True},
                "weight_reconstruction": "PASS",
                "quantization_contract": {"source": "canonical"},
                "models": rows,
                "checksums_file": "SHA256SUMS",
                "numerical_status": "GOLDEN_EXECUTION_UNAVAILABLE",
                "numerical_gate": {"code": "GOLDEN_EXECUTION_UNAVAILABLE"},
                "mismatch_count": None,
                "golden_report_sha256": None,
                "auto_repair_report": {"status": "BLOCKED"},
                "release_status": "NOT_ELIGIBLE",
            }
            candidate = {"root": str(root), "manifest": manifest, "label": "tp4_d1"}
            gate = evaluate_release([candidate])
            self.assertEqual(gate["release_gate"], "FAIL")
            self.assertEqual(gate["first_failure"], "GOLDEN_EXECUTION_UNAVAILABLE")
            release_path = Path(directory) / "release"
            with self.assertRaises(ToolError) as raised:
                publish_release(root, release_path, [candidate])
            self.assertEqual(raised.exception.code, "RELEASE_GATE_FAIL")
            self.assertFalse(release_path.exists())

    def test_empty_candidate_set_cannot_pass(self):
        gate = evaluate_release([])
        self.assertEqual(gate["release_gate"], "FAIL")
        self.assertEqual(gate["first_failure"], "NO_CANDIDATES")


class CliTests(unittest.TestCase):
    def test_top_level_cli_contract(self):
        args = build_parser().parse_args([
            "--prefill", "./canonical/prefill128",
            "--decode", "./canonical/decode1",
            "--tp", "2",
            "--quant", "w4a16",
            "--strict-bitexact",
            "--auto-repair",
            "--max-repairs", "4",
            "--output", "./output/tp2",
        ])
        self.assertIsNone(args.command)
        self.assertEqual(args.tp, "2")
        self.assertTrue(args.strict_bitexact)
        self.assertTrue(args.auto_repair)
        self.assertEqual(args.max_repairs, 4)


class GoldenBackendTests(unittest.TestCase):
    def test_tp2_pipeline_records_golden_unavailable_instead_of_zero_mismatch(self):
        from tflite_tp.pipeline import run_pipeline
        import tflite_tp.pipeline as pipeline
        import json

        def discovered(_path, expected_layers):
            self.assertEqual(expected_layers, 1)
            return SimpleNamespace(
                structural_pass=True,
                issues=(),
                coverage={
                    "P128": {"complete": True},
                    "D1": {"complete": True},
                },
                sources=(),
                to_dict=lambda include_quant_values=False: {"sources": []},
            )

        def build_tp2(_source, output_directory, profile, **_kwargs):
            root = Path(output_directory)
            root.mkdir(parents=True, exist_ok=True)
            return {
                "profile": profile,
                "topology": "TP2_ROW_REDUCE",
                "expected_layers": 1,
                "structural_status": "STRUCTURAL_PASS",
                "source_coverage": {"complete": True},
                "weight_reconstruction": "PASS",
                "quantization_contract": {"source": "canonical"},
                "models": [],
                "model_count": 0,
                "rank_order": [0, 1],
                "collective_contract": {
                    "row_reduce_boundaries": [
                        {"name": "O", "operation": "ALL_REDUCE_SUM"},
                        {"name": "Down", "operation": "ALL_REDUCE_SUM"},
                    ],
                    "all_gather_boundaries": [],
                    "kv_cache_policy": "RANK_LOCAL",
                    "quantized_partial_semantics": {
                        "status": "UNPROVEN",
                        "proof_sha256": None,
                    },
                },
                "source_audit": {"sources": []},
                "layer_manifests": [],
                "layer_manifest_sha256": {},
                "checksums_file": "SHA256SUMS",
                "numerical_status": "GOLDEN_EXECUTION_UNAVAILABLE",
                "mismatch_count": None,
                "numerical_gate": {"code": "GOLDEN_EXECUTION_UNAVAILABLE"},
                "auto_repair_report": {"status": "BLOCKED"},
                "release_status": "NOT_ELIGIBLE",
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(
            pipeline, "discover_path", side_effect=discovered
        ), patch.object(pipeline, "build_tp2_directory", side_effect=build_tp2), patch.object(
            pipeline,
            "MTKConverterTFLiteRunner",
            side_effect=ToolError("GOLDEN_EXECUTION_UNAVAILABLE", "test has no registered interpreter"),
        ):
            result = run_pipeline(
                "canonical-p128",
                "canonical-d1",
                2,
                Path(directory) / "output",
                expected_layers=1,
                auto_repair=True,
            )
            diagnostics = [
                json.loads((Path(item["root"]) / "GOLDEN_DIAGNOSTIC.json").read_text(encoding="utf-8"))
                for item in result["candidate_sets"]
            ]

        self.assertEqual(result["release_gate"], "FAIL")
        self.assertEqual(result["numerical_status"], "GOLDEN_EXECUTION_UNAVAILABLE")
        self.assertIsNone(result["mismatch_count"])
        self.assertEqual(len(diagnostics), 2)
        self.assertTrue(all(item["first_failure"] == "GOLDEN_EXECUTION_UNAVAILABLE" for item in diagnostics))
        self.assertTrue(all(item["quantized_partial_semantics"]["status"] == "UNPROVEN" for item in diagnostics))

    def test_tp2_missing_golden_runner_writes_unproven_row_reduce_diagnostic(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            manifest = {
                "profile": "D1",
                "topology": "TP2_ROW_REDUCE",
                "models": [],
                "layer_manifest_sha256": {},
                "checksums_file": "SHA256SUMS",
                "collective_contract": {
                    "quantized_partial_semantics": {
                        "status": "UNPROVEN",
                        "proof_sha256": None,
                        "reason": "Accumulator and requantization behavior are not verified.",
                    },
                },
            }
            finalized = _record_golden_execution_failure(
                root,
                manifest,
                "GOLDEN_EXECUTION_UNAVAILABLE",
                "No verified TP2 Golden runner is registered.",
            )
            diagnostic = json.loads((root / "GOLDEN_DIAGNOSTIC.json").read_text(encoding="utf-8"))
            checksums = (root / "SHA256SUMS").read_text(encoding="utf-8")
        self.assertEqual(finalized["numerical_status"], "GOLDEN_EXECUTION_UNAVAILABLE")
        self.assertIsNone(finalized["mismatch_count"])
        self.assertEqual(finalized["numerical_gate"]["code"], "GOLDEN_EXECUTION_UNAVAILABLE")
        self.assertEqual(finalized["numerical_gate"]["row_reduce_semantics"], "UNPROVEN")
        self.assertEqual(diagnostic["quantized_partial_semantics"]["status"], "UNPROVEN")
        self.assertIn(finalized["golden_diagnostic_sha256"], checksums)

    def test_tp4_golden_finalizer_binds_report_sha_and_release_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "profile": "D1",
                "expected_layers": 1,
                "models": [],
                "layer_manifest_sha256": {},
                "checksums_file": "SHA256SUMS",
            }
            report = {
                "fixture_sha256_by_layer": {"0": "f" * 64},
                "numerical_status": "NUMERICAL_PASS",
                "mismatch_count": 0,
                "backend": "synthetic",
                "mismatches": [],
            }
            finalized = _finalize_tp4_golden(root, manifest, report)
            report_path = root / "GOLDEN_REPORT.json"
            digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            checksum_rows = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        self.assertEqual(finalized["numerical_status"], "NUMERICAL_PASS")
        self.assertEqual(finalized["release_status"], "RELEASE_PASS")
        self.assertEqual(finalized["release_verification_scope"], "HOST_CANONICAL_GOLDEN")
        self.assertEqual(finalized["npu_verification_status"], "NPU_UNVERIFIED")
        self.assertEqual(finalized["golden_report_sha256"], digest)
        self.assertIn("{}  GOLDEN_REPORT.json".format(digest), checksum_rows)

    def test_tp4_golden_finalizer_blocks_zero_mismatch_without_fixture_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "profile": "D1",
                "expected_layers": 1,
                "models": [],
                "layer_manifest_sha256": {},
            }
            report = {
                "numerical_status": "NUMERICAL_PASS",
                "mismatch_count": 0,
                "backend": "synthetic",
                "mismatches": [],
            }
            finalized = _finalize_tp4_golden(root, manifest, report)
        self.assertEqual(finalized["numerical_status"], "NUMERICAL_PASS")
        self.assertEqual(finalized["release_status"], "NOT_ELIGIBLE")
        self.assertEqual(finalized["numerical_gate"]["code"], "GOLDEN_FIXTURE_IDENTITY_MISSING")
        self.assertEqual(finalized["auto_repair_report"]["status"], "BLOCKED")

    def test_tp4_golden_finalizer_keeps_mismatches_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"profile": "D1", "models": [], "layer_manifest_sha256": {}}
            report = {
                "numerical_status": "NUMERICAL_FAIL",
                "mismatch_count": 1,
                "backend": "synthetic",
                "mismatches": [{
                    "layer": 3,
                    "rank": 1,
                    "stage": "Q",
                    "operator": "attn",
                    "tensor_name": "layers/3/q",
                    "projection": "q",
                }],
            }
            finalized = _finalize_tp4_golden(root, manifest, report)
        self.assertEqual(finalized["mismatch_count"], 1)
        self.assertEqual(finalized["release_status"], "NOT_ELIGIBLE")
        self.assertEqual(finalized["auto_repair_report"]["status"], "BLOCKED")

    def test_tp4_boundary_contract_uses_discovered_graph_roles_and_add_edges(self):
        names = (
            "q_out", "k_out", "v_out", "context", "o_out", "swiglu", "down_out",
            "attn_residual", "hidden_states_0", "curr_keys_0", "curr_values_0",
        )
        tensors = tuple(
            SimpleNamespace(index=index, name=name, shape=(1, 1, 4), dtype="INT16", quantization=None)
            for index, name in enumerate(names)
        )
        graph = SimpleNamespace(
            tensors=tensors,
            inputs=(),
            outputs=(8, 9, 10),
            operators=(
                SimpleNamespace(index=0, opcode="ADD", inputs=(4, 99), outputs=(7,)),
                SimpleNamespace(index=1, opcode="ADD", inputs=(6, 7), outputs=(8,)),
            ),
        )

        def role(output=None, activation=None):
            return SimpleNamespace(
                output_tensor_indices=() if output is None else (output,),
                activation_tensor_index=activation,
            )

        layer = SimpleNamespace(roles={
            "q": role(output=0),
            "k": role(output=1),
            "v": role(output=2),
            "o": role(output=4, activation=3),
            "down": role(output=6, activation=5),
        })
        contract = derive_tp4_boundary_contract(graph, layer)
        self.assertEqual(contract["Q"][0]["source_name"], "q_out")
        self.assertEqual(contract["ATTENTION_CONTEXT"][0]["source_name"], "context")
        self.assertEqual(contract["RESIDUAL_HIDDEN"][0]["source_name"], "attn_residual")
        self.assertEqual(contract["RESIDUAL_HIDDEN"][1]["source_name"], "hidden_states_0")
        self.assertEqual(contract["KV_OUTPUT"][0]["source_name"], "curr_keys_0")

    def test_historical_layer21_attention_and_layer31_final_hidden_scope_by_semantics(self):
        from tflite_tp.discovery import discover_models
        from test_discovery import synthetic_transformer_model

        for semantic_layer in (21, 31):
            with self.subTest(semantic_layer=semantic_layer):
                source = synthetic_transformer_model()
                source_graph = source.graphs[0]

                def rename(name):
                    name = name.replace("layers/0/", "layers/{}/".format(semantic_layer))
                    for stem in ("curr_keys", "curr_values", "hidden_states"):
                        name = name.replace(stem + "_0", stem + "_{}".format(semantic_layer))
                    return name

                graph = replace(
                    source_graph,
                    tensors=tuple(replace(tensor, name=rename(tensor.name)) for tensor in source_graph.tensors),
                )
                model = replace(source, graphs=(graph,))
                report = discover_models(
                    [(model, "historical/layer{}_1t.tflite".format(semantic_layer))],
                    expected_layers=32,
                )

                layer = report.sources[0].layers[0]
                self.assertEqual(layer.layer_id, semantic_layer)
                self.assertTrue(layer.complete, layer.issues)
                self.assertFalse(report.structural_pass)  # One historical layer must not satisfy Full32 coverage.
                self.assertEqual(
                    [code for code, _ in report.issues],
                    ["LAYER_COVERAGE_INCOMPLETE"],
                )
                self.assertEqual(layer.roles["q"].semantic_layer_id, semantic_layer)
                analysis = report.sources[0].graph_analyses[0]
                self.assertEqual(analysis.evidence["semantic_layer_ids"], [semantic_layer])
                self.assertTrue(analysis.features["attention"])
                self.assertTrue(analysis.features["mask"])
                mask = next(tensor for tensor in graph.tensors if tensor.name == "mask")
                self.assertEqual(mask.shape[-1], 1025)

                contract = derive_tp4_boundary_contract(graph, layer)
                self.assertEqual(
                    contract["KV_OUTPUT"][0]["source_name"],
                    "curr_keys_{}".format(semantic_layer),
                )
                self.assertEqual(
                    contract["KV_OUTPUT"][1]["source_name"],
                    "curr_values_{}".format(semantic_layer),
                )
                self.assertEqual(
                    contract["RESIDUAL_HIDDEN"][1]["source_name"],
                    "hidden_states_{}".format(semantic_layer),
                )

    def test_deterministic_fixtures_respect_mask_and_quantized_rope_contracts(self):
        tensors = (
            SimpleNamespace(index=0, name="input_embeds", shape=(1, 2), dtype="INT16", quantization=None),
            SimpleNamespace(
                index=1, name="mask", shape=(1, 3), dtype="INT16",
                quantization=SimpleNamespace(scales=(0.25,), zero_points=(7,), quantized_dimension=0),
            ),
            SimpleNamespace(
                index=2, name="pos_emb", shape=(1, 2, 1, 2), dtype="INT16",
                quantization=SimpleNamespace(scales=(1.0 / 16384.0,), zero_points=(0,), quantized_dimension=0),
            ),
            SimpleNamespace(index=3, name="past_keys_0", shape=(1, 4), dtype="INT16", quantization=None),
        )
        graph = SimpleNamespace(tensors=tensors, inputs=(0, 1, 2, 3), name="fixture_graph")
        first = make_deterministic_tflite_inputs(graph, seed=9)
        again = make_deterministic_tflite_inputs(graph, seed=9)
        other = make_deterministic_tflite_inputs(graph, seed=10)
        self.assertTrue(np.array_equal(first["mask"], np.full((1, 3), 7, dtype=np.int16)))
        self.assertTrue(np.array_equal(first["pos_emb"][:, 0], np.full((1, 1, 2), 16384, dtype=np.int16)))
        self.assertTrue(np.array_equal(first["pos_emb"][:, 1], np.zeros((1, 1, 2), dtype=np.int16)))
        for name in first:
            self.assertTrue(np.array_equal(first[name], again[name]))
        self.assertFalse(np.array_equal(first["input_embeds"], other["input_embeds"]))

    def test_tp4_layer_runner_gathers_rank_outputs_and_matches_canonical(self):
        components = {
            "attn": (
                [("x", [1, 1, 4]), ("past", [1, 4, 1, 1])],
                [("ctx", [1, 1, 1]), ("keys", [1, 1, 1, 1]), ("values", [1, 1, 1, 1])],
            ),
            "o": ([("ctx", [1, 1, 4])], [("o", [1, 1, 1])]),
            "suffix_attn": (
                [("x", [1, 1, 4]), ("o", [1, 1, 4])],
                [("res", [1, 1, 4]), ("norm", [1, 1, 4])],
            ),
            "mlp": ([("norm", [1, 1, 4])], [("mlp", [1, 1, 1])]),
            "down": ([("mlp", [1, 1, 4])], [("down", [1, 1, 1])]),
            "suffix_final": (
                [("res", [1, 1, 4]), ("down", [1, 1, 4])],
                [("hidden", [1, 1, 4])],
            ),
        }
        x = np.array([[[1, 2, 3, 4]]], dtype=np.int16)
        expected = {
            "hidden": 21 * x + 6,
            "keys": np.arange(10, 14, dtype=np.int16).reshape(1, 4, 1, 1),
            "values": np.arange(20, 24, dtype=np.int16).reshape(1, 4, 1, 1),
            "q": x + 10,
            "k": x + 20,
            "v": x + 30,
            "ctx": x,
            "o": 2 * x,
            "swiglu": 9 * x + 3,
            "down_full": 18 * x + 6,
            "attn_res": 3 * x,
        }

        class FakeRunner:
            def __init__(self, source_path, source_hash):
                self.source_path = str(source_path)
                self.source_hash = source_hash

            def run(self, model_path, inputs, output_names=None):
                path = Path(model_path)
                if str(path) == self.source_path:
                    outputs = expected
                    if output_names is not None:
                        outputs = {name: outputs[name] for name in output_names}
                    return {
                        "outputs": outputs,
                        "source_sha256": self.source_hash,
                        "fixture_sha256": "f" * 64,
                        "backend": "synthetic",
                        "simulate_fp16": False,
                    }
                rank = int(path.parent.parent.name.replace("rank", ""))
                if path.stem == "attn":
                    outputs = {
                        "ctx": inputs["x"][..., rank:rank + 1],
                        "keys": np.array([[[[rank + 10]]]], dtype=np.int16),
                        "values": np.array([[[[rank + 20]]]], dtype=np.int16),
                        "q": inputs["x"][..., rank:rank + 1] + 10,
                        "k": inputs["x"][..., rank:rank + 1] + 20,
                        "v": inputs["x"][..., rank:rank + 1] + 30,
                    }
                elif path.stem == "o":
                    outputs = {"o": inputs["ctx"][..., rank:rank + 1] * 2}
                elif path.stem == "suffix_attn":
                    residual = inputs["x"] + inputs["o"]
                    outputs = {"res": residual, "norm": residual + 1}
                elif path.stem == "mlp":
                    outputs = {"mlp": inputs["norm"][..., rank:rank + 1] * 3}
                elif path.stem == "down":
                    outputs = {"down": inputs["mlp"][..., rank:rank + 1] * 2}
                else:
                    outputs = {"hidden": inputs["res"] + inputs["down"]}
                return {"outputs": outputs}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tflite"
            source.write_bytes(b"canonical source")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            candidate_root = root / "candidate"
            rows = []
            for component, (input_specs, output_specs) in components.items():
                for rank in range(4):
                    relative = "d1/rank{}/layer00/{}.tflite".format(rank, component)
                    target = candidate_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes("{}-{}".format(component, rank).encode("ascii"))
                    rows.append({
                        "profile": "D1",
                        "layer": 0,
                        "rank": rank,
                        "component": component,
                        "path": relative,
                        "source_sha256": source_hash,
                        "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "io": {
                            "inputs": [{"name": name, "shape": shape} for name, shape in input_specs],
                            "outputs": [{"name": name, "shape": shape} for name, shape in output_specs],
                        },
                    })
            manifest = {
                "profile": "D1",
                "structural_status": "STRUCTURAL_PASS",
                "rank_order": [0, 1, 2, 3],
                "collective_contract": {
                    "architecture": "FOUR_STAGE_ALL_GATHER",
                    "all_gather_boundaries": [{"stage": stage} for stage in ("AG0", "AG1", "AG2", "AG3")],
                },
                "models": rows,
            }
            boundary_contract = {
                "Q": [{"component": "attn", "source_name": "q", "candidate_name": "q", "distribution": "rank_shard"}],
                "K": [{"component": "attn", "source_name": "k", "candidate_name": "k", "distribution": "rank_shard"}],
                "V": [{"component": "attn", "source_name": "v", "candidate_name": "v", "distribution": "rank_shard"}],
                "ATTENTION_CONTEXT": [{"component": "attn", "source_name": "ctx", "candidate_name": "ctx", "distribution": "rank_shard"}],
                "O": [{"component": "o", "source_name": "o", "candidate_name": "o", "distribution": "rank_shard"}],
                "SWIGLU": [{"component": "mlp", "source_name": "swiglu", "candidate_name": "mlp", "distribution": "rank_shard"}],
                "DOWN": [{"component": "down", "source_name": "down_full", "candidate_name": "down", "distribution": "rank_shard"}],
                "RESIDUAL_HIDDEN": [
                    {"component": "suffix_attn", "source_name": "attn_res", "candidate_name": "res", "distribution": "replicated"},
                    {"component": "suffix_final", "source_name": "hidden", "candidate_name": "hidden", "distribution": "replicated"},
                ],
                "KV_OUTPUT": [
                    {"component": "attn", "source_name": "keys", "candidate_name": "keys", "distribution": "rank_shard"},
                    {"component": "attn", "source_name": "values", "candidate_name": "values", "distribution": "rank_shard"},
                ],
            }
            report = run_tp4_layer_golden(
                source,
                candidate_root,
                manifest,
                {"x": x, "past": np.arange(4, dtype=np.int16).reshape(1, 4, 1, 1)},
                0,
                FakeRunner(source, source_hash),
                boundary_contract=boundary_contract,
                public_source_output_names=["hidden", "keys", "values"],
            )

        self.assertEqual(report["mismatch_count"], 0)
        self.assertEqual(report["numerical_status"], "NUMERICAL_PASS")
        self.assertEqual(report["rank_component_execution_count"], 24)
        self.assertEqual(len(report["boundary_results"]), 4 * 9)
        self.assertEqual(report["required_stages"], [
            "Q", "K", "V", "ATTENTION_CONTEXT", "O", "SWIGLU", "DOWN", "RESIDUAL_HIDDEN", "KV_OUTPUT"
        ])
        self.assertTrue(all(row["mismatch_count"] == 0 for row in report["boundary_results"]))

    def test_tp4_profile_golden_chains_canonical_hidden_and_covers_every_stage(self):
        component_specs = {
            "attn": (
                [("input_embeds", [1, 1, 4]), ("mask", [1, 1, 1, 5]), ("pos_emb", [1, 2, 1, 1]),
                 ("past_keys_0", [1, 1, 1, 1]), ("past_values_0", [1, 1, 1, 1])],
                [("ctx", [1, 1, 1]), ("curr_keys_0", [1, 1, 1, 1]), ("curr_values_0", [1, 1, 1, 1])],
            ),
            "o": ([("ctx", [1, 1, 4])], [("o", [1, 1, 1])]),
            "suffix_attn": (
                [("input_embeds", [1, 1, 4]), ("o", [1, 1, 4])],
                [("attn_residual", [1, 1, 4]), ("norm", [1, 1, 4])],
            ),
            "mlp": ([("norm", [1, 1, 4])], [("swiglu", [1, 1, 1])]),
            "down": ([("swiglu", [1, 1, 4])], [("down", [1, 1, 1])]),
            "suffix_final": (
                [("attn_residual", [1, 1, 4]), ("down", [1, 1, 4])],
                [("hidden_states_0", [1, 1, 4])],
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "sources"
            candidate_root = root / "candidate"
            source_dir.mkdir()
            source_objects = []
            source_manifest_rows = []
            source_paths = {}
            candidate_rows = []

            for layer_id in range(2):
                source_name = "layer{:02d}.tflite".format(layer_id)
                source_path = source_dir / source_name
                source_path.write_bytes(("canonical-{}".format(layer_id)).encode("ascii"))
                source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
                source_paths[str(source_path)] = source_hash
                shapes = {
                    "input_embeds": (1, 1, 4), "mask": (1, 1, 1, 5), "pos_emb": (1, 2, 1, 1),
                    "past_keys_0": (1, 4, 1, 1), "past_values_0": (1, 4, 1, 1),
                    "q": (1, 1, 4), "k": (1, 1, 4), "v": (1, 1, 4), "ctx": (1, 1, 4),
                    "o": (1, 1, 4), "swiglu": (1, 1, 4), "down": (1, 1, 4),
                    "attn_residual": (1, 1, 4), "hidden_states_0": (1, 1, 4),
                    "curr_keys_0": (1, 4, 1, 1), "curr_values_0": (1, 4, 1, 1),
                }
                tensors = tuple(
                    SimpleNamespace(index=index, name=name, shape=shape, dtype="INT16", quantization=None)
                    for index, (name, shape) in enumerate(shapes.items())
                )
                tensor_index = {tensor.name: tensor.index for tensor in tensors}
                graph = SimpleNamespace(
                    name="synthetic_d1_layer",
                    tensors=tensors,
                    inputs=tuple(tensor_index[name] for name in (
                        "input_embeds", "mask", "pos_emb", "past_keys_0", "past_values_0"
                    )),
                    outputs=tuple(tensor_index[name] for name in (
                        "hidden_states_0", "curr_keys_0", "curr_values_0"
                    )),
                    operators=(
                        SimpleNamespace(index=0, opcode="ADD", inputs=(tensor_index["o"], -1), outputs=(tensor_index["attn_residual"],)),
                        SimpleNamespace(index=1, opcode="ADD", inputs=(tensor_index["down"], tensor_index["attn_residual"]), outputs=(tensor_index["hidden_states_0"],)),
                    ),
                )

                def projection(output=None, activation=None):
                    return SimpleNamespace(
                        output_tensor_indices=() if output is None else (tensor_index[output],),
                        activation_tensor_index=tensor_index[activation] if activation else None,
                    )

                layer = SimpleNamespace(
                    layer_id=layer_id,
                    graph_index=0,
                    profile="D1",
                    roles={
                        "q": projection(output="q"), "k": projection(output="k"), "v": projection(output="v"),
                        "o": projection(output="o", activation="ctx"),
                        "down": projection(output="down", activation="swiglu"),
                    },
                )
                model_info = SimpleNamespace(source_path=str(source_path), graphs=(graph,))
                source_objects.append(SimpleNamespace(
                    source_name=source_name,
                    source_sha256=source_hash,
                    profiles=["D1"],
                    model_info=model_info,
                    layers=[layer],
                ))
                source_manifest_rows.append({
                    "source": source_name,
                    "source_sha256": source_hash,
                    "profiles": ["D1"],
                })

                for component, (inputs, outputs) in component_specs.items():
                    for rank in range(4):
                        relative = "d1/rank{}/layer{:02d}/{}.tflite".format(rank, layer_id, component)
                        model_path = candidate_root / relative
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        model_path.write_bytes("{}-{}-{}".format(layer_id, rank, component).encode("ascii"))
                        candidate_rows.append({
                            "profile": "D1", "layer": layer_id, "rank": rank, "component": component,
                            "path": relative,
                            "source_sha256": source_hash,
                            "output_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                            "io": {
                                "inputs": [{"name": name, "shape": shape} for name, shape in inputs],
                                "outputs": [{"name": name, "shape": shape} for name, shape in outputs],
                            },
                        })

            source_report = SimpleNamespace(
                expected_layers=2,
                coverage={"D1": {"complete": True, "present_layers": [0, 1]}},
                sources=source_objects,
            )
            manifest = {
                "profile": "D1",
                "expected_layers": 2,
                "structural_status": "STRUCTURAL_PASS",
                "rank_order": [0, 1, 2, 3],
                "source_audit": {"sources": source_manifest_rows},
                "collective_contract": {
                    "architecture": "FOUR_STAGE_ALL_GATHER",
                    "all_gather_boundaries": [{"stage": stage} for stage in ("AG0", "AG1", "AG2", "AG3")],
                },
                "models": candidate_rows,
            }

            class FakeRunner:
                def __init__(self):
                    self.canonical_inputs = {}

                def run(self, model_path, inputs, output_names=None):
                    path = str(model_path)
                    if path in source_paths:
                        x = np.asarray(inputs["input_embeds"])
                        self.canonical_inputs[path] = x.copy()
                        outputs = {
                            "q": x + 10,
                            "k": x + 20,
                            "v": x + 30,
                            "ctx": x,
                            "o": x * 2,
                            "swiglu": x * 9 + 3,
                            "down": x * 18 + 6,
                            "attn_residual": x * 3,
                            "hidden_states_0": x * 21 + 6,
                            "curr_keys_0": inputs["past_keys_0"],
                            "curr_values_0": inputs["past_values_0"],
                        }
                        if output_names is not None:
                            outputs = {name: outputs[name] for name in output_names}
                        return {
                            "outputs": outputs,
                            "source_sha256": source_paths[path],
                            "fixture_sha256": "f" * 64,
                            "backend": "synthetic",
                            "simulate_fp16": False,
                        }
                    rank = int(Path(path).parent.parent.name.replace("rank", ""))
                    component = Path(path).stem
                    if component == "attn":
                        x = inputs["input_embeds"]
                        shard = x[..., rank:rank + 1]
                        outputs = {
                            "ctx": shard,
                            "curr_keys_0": inputs["past_keys_0"],
                            "curr_values_0": inputs["past_values_0"],
                            "q": shard + 10, "k": shard + 20, "v": shard + 30,
                        }
                    elif component == "o":
                        outputs = {"o": inputs["ctx"][..., rank:rank + 1] * 2}
                    elif component == "suffix_attn":
                        residual = inputs["input_embeds"] + inputs["o"]
                        outputs = {"attn_residual": residual, "norm": residual + 1}
                    elif component == "mlp":
                        outputs = {"swiglu": inputs["norm"][..., rank:rank + 1] * 3}
                    elif component == "down":
                        outputs = {"down": inputs["swiglu"][..., rank:rank + 1] * 2}
                    else:
                        outputs = {"hidden_states_0": inputs["attn_residual"] + inputs["down"]}
                    if output_names is not None:
                        outputs = {name: outputs[name] for name in output_names}
                    return {"outputs": outputs}

            runner = FakeRunner()
            report = run_tp4_profile_golden(source_report, candidate_root, manifest, "D1", runner, seed=44)

        self.assertEqual(report["numerical_status"], "NUMERICAL_PASS")
        self.assertEqual(report["mismatch_count"], 0)
        self.assertEqual(report["golden_coverage"]["layers"], [0, 1])
        self.assertEqual(len(report["boundary_results"]), 2 * 4 * 9)
        self.assertTrue(all(row["mismatch_count"] == 0 for row in report["boundary_results"]))
        second_input = runner.canonical_inputs[str(source_objects[1].model_info.source_path)]
        first_input = runner.canonical_inputs[str(source_objects[0].model_info.source_path)]
        self.assertTrue(np.array_equal(second_input, first_input * 21 + 6))

    def test_tp4_profile_golden_executes_grouped_canonical_graph_once_and_chains_candidates(self):
        import json
        from test_discovery import synthetic_grouped_transformer_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "grouped.tflite"
            source_path.write_bytes(b"grouped canonical source")
            source_bytes = source_path.read_bytes()
            source_sha = hashlib.sha256(source_bytes).hexdigest()
            model = replace(
                synthetic_grouped_transformer_model(),
                source_path=str(source_path),
                source_sha256=source_sha,
                source_bytes=len(source_bytes),
            )
            source_report = discover_models([(model, source_path.name)], expected_layers=2)
            self.assertTrue(source_report.structural_pass)
            graph = model.graphs[0]
            source_tensors = {tensor.name: tensor for tensor in graph.tensors}
            candidate_root = root / "candidate"
            candidate_rows = []
            candidate_paths = {}
            distribution_by_layer = {}

            for source in source_report.sources:
                for layer, analysis in zip(source.layers, source.graph_analyses):
                    components = derive_tp4_components(model, layer, analysis)
                    contract = derive_tp4_boundary_contract(graph, layer)
                    distribution = {
                        (check["component"], check["candidate_name"]): check["distribution"]
                        for stage in contract.values()
                        for check in stage
                    }
                    distribution_by_layer[layer.layer_id] = distribution
                    for rank in range(4):
                        for component in components:
                            relative = "d1/rank{}/layer{:02d}/{}.tflite".format(
                                rank, layer.layer_id, component.name
                            )
                            target = candidate_root / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes("{}:{}:{}".format(layer.layer_id, rank, component.name).encode("ascii"))
                            output_ports = []
                            for tensor_index in component.output_tensor_indices:
                                tensor = graph.tensors[tensor_index]
                                output_ports.append({"name": tensor.name, "shape": list(tensor.shape)})
                            input_ports = [
                                {"name": graph.tensors[index].name, "shape": list(graph.tensors[index].shape)}
                                for index in component.input_tensor_indices
                            ]
                            row = {
                                "profile": "D1",
                                "layer": layer.layer_id,
                                "rank": rank,
                                "component": component.name,
                                "path": relative,
                                "source_sha256": source_sha,
                                "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                                "io": {"inputs": input_ports, "outputs": output_ports},
                            }
                            candidate_rows.append(row)
                            candidate_paths[str(target)] = row

            manifest = {
                "profile": "D1",
                "expected_layers": 2,
                "structural_status": "STRUCTURAL_PASS",
                "rank_order": [0, 1, 2, 3],
                "source_audit": source_report.to_dict(include_quant_values=False),
                "collective_contract": {
                    "architecture": "FOUR_STAGE_ALL_GATHER",
                    "all_gather_boundaries": [
                        {"stage": stage} for stage in ("AG0", "AG1", "AG2", "AG3")
                    ],
                },
                "models": candidate_rows,
            }

            class FakeRunner:
                def __init__(self):
                    self.source_calls = 0
                    self.layer1_rank0_hidden_input = None

                def _tensor_shape(self, name):
                    return tuple(source_tensors[name].shape)

                def _rank_shape(self, shape, name):
                    shape = list(shape)
                    if "curr_key" in name.lower() or "curr_value" in name.lower():
                        axis = 1
                    else:
                        axes = [index for index, extent in enumerate(shape) if extent > 1 and extent % 4 == 0]
                        if len(axes) != 1:
                            return tuple(shape)
                        axis = axes[0]
                    shape[axis] //= 4
                    return tuple(shape)

                def run(self, model_path, inputs, output_names=None):
                    path = str(model_path)
                    if path == str(source_path):
                        self.source_calls += 1
                        outputs = {}
                        for name in output_names or ():
                            tensor = source_tensors[name]
                            fill = 1 if "hidden_states_" in name else 0
                            outputs[name] = np.full(tensor.shape, fill, dtype=np.int16)
                        return {
                            "outputs": outputs,
                            "source_sha256": source_sha,
                            "fixture_sha256": "f" * 64,
                            "backend": "synthetic_grouped",
                            "simulate_fp16": False,
                        }

                    row = candidate_paths[path]
                    rank = row["rank"]
                    layer_id = row["layer"]
                    component = row["component"]
                    if layer_id == 1 and rank == 0 and component == "attn":
                        self.layer1_rank0_hidden_input = np.asarray(inputs["hidden_states_0"]).copy()
                    port_shapes = {port["name"]: tuple(port["shape"]) for port in row["io"]["outputs"]}
                    outputs = {}
                    requested_names = output_names or list(port_shapes)
                    for name in requested_names:
                        shape = port_shapes.get(name, self._tensor_shape(name))
                        if (component, name) in distribution_by_layer[layer_id] and distribution_by_layer[layer_id][(component, name)] == "rank_shard":
                            shape = self._rank_shape(self._tensor_shape(name), name)
                        fill = 1 if "hidden_states_" in name else 0
                        outputs[name] = np.full(shape, fill, dtype=np.int16)
                    return {"outputs": outputs}

            runner = FakeRunner()
            report = run_tp4_profile_golden(
                source_report, candidate_root, manifest, "D1", runner, seed=121
            )

        self.assertEqual(runner.source_calls, 1)
        self.assertEqual(report["numerical_status"], "NUMERICAL_PASS")
        self.assertEqual(report["mismatch_count"], 0)
        self.assertEqual(report["golden_coverage"]["layers"], [0, 1])
        self.assertEqual(len(report["boundary_results"]), 2 * 4 * 9)
        self.assertTrue(np.all(runner.layer1_rank0_hidden_input == 1))
        json.dumps(report)

    def test_tp4_directory_builder_accepts_grouped_single_subgraph_sources(self):
        from test_discovery import synthetic_grouped_transformer_model
        import tflite_tp.tflite_rewriter as rewriter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "sources"
            source_root.mkdir()
            source_path = source_root / "grouped.tflite"
            source_path.write_bytes(b"grouped canonical source")
            source_data = source_path.read_bytes()
            source_sha = hashlib.sha256(source_data).hexdigest()
            model = replace(
                synthetic_grouped_transformer_model(),
                source_path=str(source_path),
                source_sha256=source_sha,
                source_bytes=len(source_data),
            )
            source_report = discover_models([(model, source_path.name)], expected_layers=2)
            output_root = root / "tp4"

            class FakeAdapter:
                def __init__(self, schema_module=None):
                    self.schema = object()

                def load(self, path):
                    return SimpleNamespace(subgraphs=[object()])

            def fake_build_layer(model_object, model_info, layer, analysis, output_directory, degree, schema_module=None):
                self.assertEqual(degree, 4)
                self.assertEqual(analysis.layer_id, layer.layer_id)
                models = []
                for rank in range(4):
                    for component in ("attn", "o", "suffix_attn", "mlp", "down", "suffix_final"):
                        relative = "d1/rank{}/layer{:02d}/{}.tflite".format(rank, layer.layer_id, component)
                        path = Path(output_directory) / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes("{}:{}:{}".format(layer.layer_id, rank, component).encode("ascii"))
                        models.append({
                            "profile": "D1",
                            "layer": layer.layer_id,
                            "rank": rank,
                            "component": component,
                            "path": relative,
                            "source": layer.source_name,
                            "source_sha256": source_sha,
                            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        })
                return {
                    "models": models,
                    "numerical_gate": {"status": "GOLDEN_EXECUTION_UNAVAILABLE"},
                    "quantization_contract": {"verified": True},
                    "weight_partitions": {},
                }

            with patch.object(rewriter, "TFLiteObjectAdapter", FakeAdapter), patch.object(
                rewriter, "_build_tp_layer_into", side_effect=fake_build_layer
            ):
                aggregate = rewriter.build_tp4_directory(
                    source_root,
                    output_root,
                    profile="D1",
                    expected_layers=2,
                    discovery_report=source_report,
                )

        self.assertEqual(aggregate["model_count"], 2 * 4 * 6)
        self.assertEqual([row["layer"] for row in aggregate["models"][:24]], [0] * 24)
        self.assertEqual([row["layer"] for row in aggregate["models"][24:]], [1] * 24)
        self.assertEqual(aggregate["structural_status"], "STRUCTURAL_PASS")

    def test_tp2_directory_builder_preserves_unproven_row_reduce_evidence_per_layer(self):
        from test_discovery import synthetic_grouped_transformer_model
        import tflite_tp.tflite_rewriter as rewriter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "sources"
            source_root.mkdir()
            source_path = source_root / "grouped.tflite"
            source_path.write_bytes(b"grouped canonical source")
            source_data = source_path.read_bytes()
            source_sha = hashlib.sha256(source_data).hexdigest()
            model = replace(
                synthetic_grouped_transformer_model(),
                source_path=str(source_path),
                source_sha256=source_sha,
                source_bytes=len(source_data),
            )
            source_report = discover_models([(model, source_path.name)], expected_layers=2)
            output_root = root / "tp2"

            class FakeAdapter:
                def __init__(self, schema_module=None):
                    self.schema = object()

                def load(self, path):
                    return SimpleNamespace(subgraphs=[object()])

            def fake_build_layer(model_object, model_info, layer, analysis, output_directory, degree, schema_module=None):
                self.assertEqual(degree, 2)
                models = []
                for rank in range(2):
                    for component in ("attn", "o", "suffix_attn", "mlp", "down", "suffix_final"):
                        relative = "d1/rank{}/layer{:02d}/{}.tflite".format(rank, layer.layer_id, component)
                        path = Path(output_directory) / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes("{}:{}:{}".format(layer.layer_id, rank, component).encode("ascii"))
                        models.append({
                            "profile": "D1",
                            "layer": layer.layer_id,
                            "rank": rank,
                            "component": component,
                            "path": relative,
                            "source": layer.source_name,
                            "source_sha256": source_sha,
                            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        })
                return {
                    "models": models,
                    "numerical_gate": {"status": "GOLDEN_EXECUTION_UNAVAILABLE"},
                    "quantization_contract": {"verified": True},
                    "weight_partitions": {},
                    "collective_contract": {
                        "quantized_partial_semantics": {
                            "status": "UNPROVEN",
                            "proof_sha256": None,
                            "partial_output_dtype": {"o": "INT16", "down": "INT16"},
                            "layer": layer.layer_id,
                        },
                    },
                }

            with patch.object(rewriter, "TFLiteObjectAdapter", FakeAdapter), patch.object(
                rewriter, "_build_tp_layer_into", side_effect=fake_build_layer
            ):
                aggregate = rewriter.build_tp2_directory(
                    source_root,
                    output_root,
                    profile="D1",
                    expected_layers=2,
                    discovery_report=source_report,
                )

        partial_semantics = aggregate["collective_contract"]["quantized_partial_semantics"]
        self.assertEqual(aggregate["model_count"], 2 * 2 * 6)
        self.assertEqual(partial_semantics["status"], "UNPROVEN")
        self.assertIsNone(partial_semantics["proof_sha256"])
        self.assertEqual(sorted(partial_semantics["per_layer"]), ["0", "1"])
        self.assertEqual(partial_semantics["per_layer"]["1"]["partial_output_dtype"]["o"], "INT16")

    def test_mtk_executor_is_reported_as_candidate_not_tp_golden_pass(self):
        mtk_module = "mtk_converter.python.tools.tflite_executor"

        def fake_find_spec(name):
            return object() if name == mtk_module else None

        with patch("tflite_tp.golden.importlib.util.find_spec", side_effect=fake_find_spec):
            report = probe_golden_backend(SimpleNamespace(graphs=[]))
        self.assertTrue(report["mtk_tflite_executor_available"])
        self.assertFalse(report["verified_fixture_runner"])
        self.assertEqual(report["status"], "GOLDEN_EXECUTION_UNAVAILABLE")

    def test_mtk_converter_runner_maps_named_outputs_and_records_hashes(self):
        calls = []

        class FakeExecutor:
            model_outputs = ["hidden", "keys"]

            def __init__(self, path, custom_op_lib_map=None, simulate_fp16=False):
                calls.append((path, custom_op_lib_map, simulate_fp16))

            def run(self, inputs, output_names=None):
                self_inputs.append(inputs)
                values = {
                    "hidden": np.array([1, 2], dtype=np.int16),
                    "keys": np.array([3], dtype=np.int16),
                    "internal": np.array([4], dtype=np.int16),
                }
                names = output_names or self.model_outputs
                return [values[name] for name in names]

        self_inputs = []
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.tflite"
            source.write_bytes(b"canonical tflite")
            inputs = {"x": np.array([[4, 5]], dtype=np.int16)}
            runner = MTKConverterTFLiteRunner(
                custom_op_lib_map={"MTKEXT_TEST": "/opt/mtkext.so"},
                simulate_fp16=True,
                executor_factory=FakeExecutor,
            )
            result = runner.run(source, inputs)
            internal_result = runner.run(source, inputs, output_names=["internal", "hidden"])

        self.assertEqual(result["outputs"]["hidden"].tolist(), [1, 2])
        self.assertEqual(internal_result["outputs"]["internal"].tolist(), [4])
        self.assertEqual(internal_result["requested_output_names"], ["internal", "hidden"])
        self.assertEqual(result["outputs"]["keys"].tolist(), [3])
        self.assertEqual(result["backend"], "mtk_converter.python.tools.tflite_executor.TFLiteExecutor")
        self.assertTrue(result["simulate_fp16"])
        self.assertEqual(result["source_sha256"], hashlib.sha256(b"canonical tflite").hexdigest())
        self.assertEqual(len(result["fixture_sha256"]), 64)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self_inputs, [inputs, inputs])

    def test_mtk_converter_runner_rejects_output_mapping_mismatch(self):
        class FakeExecutor:
            model_outputs = ["hidden", "keys"]

            def __init__(self, *_args, **_kwargs):
                pass

            def run(self, _inputs):
                return [np.array([1], dtype=np.int16)]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.tflite"
            source.write_bytes(b"canonical tflite")
            runner = MTKConverterTFLiteRunner(executor_factory=FakeExecutor)
            with self.assertRaises(ToolError) as raised:
                runner.run(source, {"x": np.array([0], dtype=np.int16)})
        self.assertEqual(raised.exception.code, "GOLDEN_OUTPUT_MAPPING_INVALID")

    def test_named_output_compare_is_bit_exact(self):
        result = compare_named_outputs(
            {"hidden": np.array([1.0, -0.0], dtype=np.float16)},
            {"hidden": np.array([1.0, 0.0], dtype=np.float16)},
        )
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["mismatches"][0]["category"], "OUTPUT_BIT_MISMATCH")
        self.assertEqual(result["mismatches"][0]["first_mismatch_index"], [1])

    def test_named_output_compare_reports_multidimensional_bit_mismatches(self):
        result = compare_named_outputs(
            {"hidden": np.zeros((1, 4), dtype=np.int16)},
            {"hidden": np.array([[0, 0, 0, 7]], dtype=np.int16)},
        )
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["mismatches"][0]["category"], "OUTPUT_BIT_MISMATCH")
        self.assertEqual(result["mismatches"][0]["first_mismatch_index"], [0, 3])
        self.assertEqual(result["mismatches"][0]["max_absolute_difference"], 7.0)

    def test_named_output_compare_fails_on_shape_and_dtype_changes(self):
        result = compare_named_outputs(
            {
                "hidden": np.zeros((2,), dtype=np.int16),
                "keys": np.zeros((1,), dtype=np.int16),
            },
            {
                "hidden": np.zeros((1, 2), dtype=np.int16),
                "keys": np.zeros((1,), dtype=np.float16),
            },
        )
        self.assertEqual(result["mismatch_count"], 2)
        self.assertEqual(
            [row["category"] for row in result["mismatches"]],
            ["OUTPUT_SHAPE_MISMATCH", "OUTPUT_DTYPE_MISMATCH"],
        )


if __name__ == "__main__":
    unittest.main()
