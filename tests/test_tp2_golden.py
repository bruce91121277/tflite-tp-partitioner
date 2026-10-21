import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from tflite_tp.golden import (
    _tp2_partial_range_diagnostic,
    _tp2_reduce_quantized_partials,
    run_tp2_row_reduce_profile_golden,
)
from tflite_tp.pipeline import _finalize_tp2_row_reduce_golden
from tflite_tp.model import (
    DiscoveryReport,
    GraphInfo,
    ModelInfo,
    QuantizationInfo,
    SourceDiscovery,
    TensorInfo,
)


def _tensor(index, name, shape, dtype="INT16", scale=1.0):
    return TensorInfo(
        index=index,
        name=name,
        shape=tuple(shape),
        shape_signature=tuple(shape),
        dtype=dtype,
        buffer_index=0,
        buffer_bytes=0,
        quantization=QuantizationInfo(
            scales=(scale,),
            zero_points=(0,),
            quantized_dimension=0,
        ),
    )


class _FakeRunner:
    def __init__(self, source_path, source_outputs, candidate_values):
        self.source_path = Path(source_path)
        self.source_outputs = source_outputs
        self.candidate_values = candidate_values
        self.inputs_by_model = {}

    def run(self, model_path, inputs, output_names=None):
        path = Path(model_path)
        if path == self.source_path:
            selected = self.source_outputs
            outputs = selected if output_names is None else {name: selected[name] for name in output_names}
            return {
                "outputs": outputs,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fixture_sha256": "f" * 64,
            }
        self.inputs_by_model[path.stem] = inputs
        output_name, value = self.candidate_values[path.stem]
        return {
            "outputs": {output_name: value},
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fixture_sha256": "f" * 64,
        }


class TP2GoldenTests(unittest.TestCase):
    def test_partial_range_diagnostic_counts_endpoint_values_without_claiming_saturation(self):
        diagnostic = _tp2_partial_range_diagnostic([
            {
                "rank": 0,
                "output_dtype": "int32",
                "output_dtype_min": -2147483648,
                "output_dtype_max": 2147483647,
                "values_at_dtype_min": 7,
                "values_at_dtype_max": 2,
            },
            {
                "rank": 1,
                "output_dtype": "int32",
                "output_dtype_min": -2147483648,
                "output_dtype_max": 2147483647,
                "values_at_dtype_min": 0,
                "values_at_dtype_max": 0,
            },
        ])

        self.assertEqual(diagnostic["status"], "RANGE_LIMIT_REACHED")
        self.assertEqual(diagnostic["rank_limit_hits"], [{
            "rank": 0,
            "dtype": "int32",
            "dtype_min": -2147483648,
            "dtype_max": 2147483647,
            "values_at_dtype_min": 7,
            "values_at_dtype_max": 2,
        }])
        self.assertFalse(diagnostic["root_cause_confirmed"])
        self.assertFalse(diagnostic["safe_repair_available"])

    def test_partial_range_diagnostic_is_clear_when_no_rank_hits_integer_endpoints(self):
        diagnostic = _tp2_partial_range_diagnostic([
            {
                "rank": 0,
                "output_dtype": "int32",
                "values_at_dtype_min": 0,
                "values_at_dtype_max": 0,
            }
        ])

        self.assertEqual(diagnostic["status"], "WITHIN_DTYPE_RANGE")
        self.assertEqual(diagnostic["rank_limit_hits"], [])


    def test_reduction_requantizes_and_saturates_to_canonical_dtype(self):
        output = _tensor(0, "out", (1, 1, 1), dtype="INT16", scale=1.0)
        partials = [
            _tensor(0, "partial0", (1, 1, 1), dtype="INT32", scale=0.5),
            _tensor(0, "partial1", (1, 1, 1), dtype="INT32", scale=0.5),
        ]

        reduced = _tp2_reduce_quantized_partials(
            [np.array([[[40000]]], dtype=np.int32), np.array([[[40000]]], dtype=np.int32)],
            partials,
            output,
            "o",
        )

        self.assertEqual(reduced.dtype, np.dtype(np.int16))
        self.assertEqual(int(reduced[0, 0, 0]), np.iinfo(np.int16).max)

    def test_profile_runner_executes_each_rank_and_marks_partial_scope_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "layer00.tflite"
            source_path.write_bytes(b"canonical")
            source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

            source_tensors = (
                _tensor(0, "o_input", (1, 1, 4), scale=0.25),
                _tensor(1, "o_output", (1, 1, 2), scale=1.0),
                _tensor(2, "down_input", (1, 1, 4), scale=0.25),
                _tensor(3, "down_output", (1, 1, 2), scale=1.0),
            )
            source_graph = GraphInfo(
                index=0,
                name="layer0",
                tensors=source_tensors,
                inputs=(0, 2),
                outputs=(1, 3),
                operators=(),
            )
            model_info = ModelInfo(
                source_path=str(source_path),
                source_sha256=source_hash,
                source_bytes=source_path.stat().st_size,
                version=3,
                buffer_count=1,
                graphs=(source_graph,),
            )
            layer = SimpleNamespace(
                layer_id=0,
                graph_index=0,
                profile="P128",
                roles={
                    "o": SimpleNamespace(activation_tensor_index=0, output_tensor_indices=(1,)),
                    "down": SimpleNamespace(activation_tensor_index=2, output_tensor_indices=(3,)),
                },
            )
            source = SourceDiscovery(
                source_name="layer00.tflite",
                source_sha256=source_hash,
                source_bytes=source_path.stat().st_size,
                version=3,
                graph_count=1,
                profiles=["P128"],
                layers=[layer],
                model_info=model_info,
            )
            source_report = DiscoveryReport(
                expected_layers=1,
                sources=[source],
                coverage={"P128": {"complete": True, "present_layers": [0]}},
            )

            candidate_root = root / "candidate"
            candidate_root.mkdir()
            models = []
            candidate_values = {
                "o-r0": ("o_output", np.array([[[-32768, 2]]], dtype=np.int16)),
                "o-r1": ("o_output", np.array([[[3, 4]]], dtype=np.int16)),
                "down-r0": ("down_output", np.array([[[2, 3]]], dtype=np.int16)),
                "down-r1": ("down_output", np.array([[[4, 5]]], dtype=np.int16)),
            }
            for component in ("o", "down"):
                input_name = "{}_input".format(component)
                output_name = "{}_output".format(component)
                for rank in range(2):
                    stem = "{}-r{}".format(component, rank)
                    model_path = candidate_root / (stem + ".tflite")
                    model_path.write_bytes(stem.encode("ascii"))
                    models.append({
                        "profile": "P128",
                        "layer": 0,
                        "component": component,
                        "rank": rank,
                        "path": model_path.name,
                        "source_sha256": source_hash,
                        "output_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        "io": {
                            "inputs": [{"name": input_name, "shape": [1, 1, 2], "dtype": "INT16"}],
                            "outputs": [{"name": output_name, "shape": [1, 1, 2], "dtype": "INT16"}],
                        },
                    })

            candidate_manifest = {
                "profile": "P128",
                "topology": "TP2_ROW_REDUCE",
                "structural_status": "STRUCTURAL_PASS",
                "expected_layers": 1,
                "source_audit": {
                    "sources": [{
                        "source": "layer00.tflite",
                        "source_sha256": source_hash,
                        "profiles": ["P128"],
                    }],
                },
                "models": models,
            }
            source_outputs = {
                "o_input": np.array([[[10, 11, 12, 13]]], dtype=np.int16),
                "o_output": np.array([[[-32765, 6]]], dtype=np.int16),
                "down_input": np.array([[[20, 21, 22, 23]]], dtype=np.int16),
                "down_output": np.array([[[6, 8]]], dtype=np.int16),
            }
            runner = _FakeRunner(source_path, source_outputs, candidate_values)

            def fake_read_tflite(path):
                component = Path(path).stem.split("-", 1)[0]
                input_name = component + "_input"
                output_name = component + "_output"
                graph = GraphInfo(
                    index=0,
                    name=component,
                    tensors=(
                        _tensor(0, input_name, (1, 1, 2), scale=0.25),
                        _tensor(1, output_name, (1, 1, 2), scale=1.0),
                    ),
                    inputs=(0,),
                    outputs=(1,),
                    operators=(),
                )
                return SimpleNamespace(graphs=(graph,))

            with patch("tflite_tp.tflite_reader.read_tflite", side_effect=fake_read_tflite):
                report = run_tp2_row_reduce_profile_golden(
                    source_report,
                    candidate_root,
                    candidate_manifest,
                    "P128",
                    runner,
                    fixture_code_scale_factors=(1,),
                )

            self.assertEqual(report["mismatch_count"], 0)
            self.assertEqual(report["numerical_status"], "NUMERICAL_INCOMPLETE")
            self.assertFalse(report["golden_coverage"]["complete"])
            self.assertEqual(report["golden_coverage"]["checked_boundary_count"], 2)
            o_boundary = report["layer_reports"][0]["row_reduce_boundaries"]["o"]
            self.assertEqual(o_boundary["rank_partials"][0]["values_at_dtype_min"], 1)
            self.assertEqual(o_boundary["partial_range_diagnostic"]["status"], "RANGE_LIMIT_REACHED")
            down_boundary = report["layer_reports"][0]["row_reduce_boundaries"]["down"]
            self.assertEqual(down_boundary["partial_range_diagnostic"]["status"], "WITHIN_DTYPE_RANGE")
            self.assertIn("o-r0", runner.inputs_by_model)
            self.assertEqual(runner.inputs_by_model["o-r0"]["o_input"].tolist(), [[[10, 11]]])
            self.assertEqual(runner.inputs_by_model["o-r1"]["o_input"].tolist(), [[[12, 13]]])

    def test_finalizer_keeps_zero_mismatch_partial_probe_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = {
                "collective_contract": {"quantized_partial_semantics": {"status": "UNPROVEN"}},
                "models": [],
                "layer_manifest_sha256": {},
                "checksums_file": "SHA256SUMS",
            }
            report = {
                "mismatch_count": 0,
                "numerical_status": "NUMERICAL_INCOMPLETE",
                "first_failure": "GOLDEN_COVERAGE_INCOMPLETE",
                "golden_coverage": {"checked_boundary_count": 2},
                "mismatches": [],
            }

            finalized = _finalize_tp2_row_reduce_golden(temporary, manifest, report)

            self.assertEqual(finalized["numerical_status"], "NUMERICAL_INCOMPLETE")
            self.assertEqual(finalized["mismatch_count"], 0)
            self.assertEqual(finalized["release_status"], "NOT_ELIGIBLE")
            self.assertEqual(
                finalized["collective_contract"]["quantized_partial_semantics"]["status"],
                "UNPROVEN",
            )
            self.assertTrue((Path(temporary) / "TP2_ROW_REDUCE_GOLDEN.json").is_file())

    def test_finalizer_blocks_nonzero_mismatch_and_does_not_auto_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = {
                "collective_contract": {"quantized_partial_semantics": {"status": "UNPROVEN"}},
                "models": [],
                "layer_manifest_sha256": {},
                "checksums_file": "SHA256SUMS",
            }
            report = {
                "mismatch_count": 1,
                "numerical_status": "NUMERICAL_FAIL",
                "first_failure": "TP2_ROW_REDUCE_MISMATCH",
                "golden_coverage": {"checked_boundary_count": 2},
                "mismatches": [{
                    "operator": "o",
                    "stage": "O_ROW_REDUCE",
                    "tensor_name": "o_output",
                    "expected": 0,
                    "actual": 1,
                    "index": [0, 0, 0],
                    "mismatch_count": 1,
                    "max_absolute_difference": 1.0,
                }],
            }

            finalized = _finalize_tp2_row_reduce_golden(temporary, manifest, report)

            self.assertEqual(finalized["numerical_status"], "NUMERICAL_FAIL")
            self.assertEqual(finalized["mismatch_count"], 1)
            self.assertEqual(finalized["release_status"], "NOT_ELIGIBLE")
            self.assertEqual(finalized["auto_repair_report"]["status"], "BLOCKED")
            self.assertFalse(finalized["auto_repair_report"]["diagnosed_mismatch"]["root_cause_confirmed"])

if __name__ == "__main__":
    unittest.main()
