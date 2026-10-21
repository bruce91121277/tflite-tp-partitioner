import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from tflite_tp_mediatek.npu_verify.capture_adapter import import_canonical_capture, import_canonical_prefix_capture
from tflite_tp_mediatek.npu_verify.exact import compare_arrays, compare_npu_runs, stitch_tp4_outputs, tp2_row_reduce_gate
from tflite_tp_mediatek.npu_verify.contracts import sha256_file
from tflite_tp_mediatek.npu_verify.report_generator import derive_release_gate
from tflite_tp_mediatek.npu_verify.transport import device_gate, invoke_npu


class NpuVerifyTests(unittest.TestCase):
    def test_invoke_accepts_its_own_initial_gate_evidence_in_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "candidate.tflite"
            dla = root / "candidate.dla"
            model.write_bytes(b"tflite")
            dla.write_bytes(b"dla")
            model_sha = sha256_file(model)
            dla_sha = sha256_file(dla)
            compile_manifest = root / "compile.json"
            compile_manifest.write_text(json.dumps({
                "status": "PASS", "source_tflite_sha256": model_sha, "dla_sha256": dla_sha,
            }), encoding="utf-8")
            input_dir = root / "inputs"
            input_dir.mkdir()
            input_file = input_dir / "input0.bin"
            input_file.write_bytes(b"x")
            input_manifest = root / "input.json"
            input_manifest.write_text(json.dumps({
                "status": "PASS_CONTRACT_AND_SHA_VERIFIED",
                "candidate_tflite_sha256": model_sha,
                "fixture_id": "fixture-a", "fixture_sha256": "f" * 64,
                "profile": "P128", "tp_degree": 4, "rank": 0,
                "inputs": [{
                    "handle": 0, "semantic_tensor": "input", "bytes": 1,
                    "sha256": sha256_file(input_file),
                }],
            }), encoding="utf-8")
            output_dir = root / "run"
            contract = {
                "sha256": model_sha,
                "inputs": [{"name": "input"}],
                "outputs": [{
                    "name": "output", "shape": [1], "dtype": "INT16", "quantization": None,
                }],
            }

            def fake_run(command, timeout, cwd=None):
                if "-r" in command:
                    collected = Path(command[-1])
                    output = collected / "output0.bin"
                    output.write_bytes(b"\x01\x00")
                    (collected / "OUTPUT_SHA256SUMS").write_text(
                        "{}  output0.bin\n".format(sha256_file(output)), encoding="ascii",
                    )
                    return SimpleNamespace(returncode=0, stdout="")
                if "ssh.exe" in str(command[0]).lower() or str(command[0]).lower().endswith("ssh"):
                    return SimpleNamespace(
                        returncode=0,
                        stdout="NEURONRT_START_EPOCH_NS=100\nNEURONRT_END_EPOCH_NS=200\nNEURONRT_EXIT_CODE=0\n",
                    )
                return SimpleNamespace(returncode=0, stdout="")

            with (
                patch("tflite_tp_mediatek.npu_verify.transport.load_tflite_contract", return_value=contract),
                patch("tflite_tp_mediatek.npu_verify.transport.device_gate", return_value={"gate": "PASS"}) as gate,
                patch("tflite_tp_mediatek.npu_verify.transport._check_program", side_effect=lambda name: name + ".exe"),
                patch("tflite_tp_mediatek.npu_verify.transport._ssh", return_value=""),
                patch("tflite_tp_mediatek.npu_verify.transport._run", side_effect=fake_run),
            ):
                result = invoke_npu(
                    model, dla, compile_manifest, input_dir, input_manifest,
                    "test-device.invalid", "/tmp/tflite-tp-test", output_dir,
                    "phase3_invoke_test", timeout_seconds=30, profile="P128", rank=0,
                )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(gate.call_count, 2)
            self.assertTrue((output_dir / "EVK_IDLE_GATE_INITIAL.json").is_file())

    @patch("tflite_tp_mediatek.npu_verify.transport._ssh")
    def test_device_gate_checks_parent_storage_for_new_remote_run_root(self, ssh):
        ssh.return_value = """EVK_HOSTNAME=test-device.invalid
NEURONRT_PATH=/usr/sbin/neuronrt
NEURONRT_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
NEURONRT_VERSION_BEGIN
8.2.16
NEURONRT_VERSION_END
TIMEOUT_PATH=/usr/bin/timeout
BUSY_SCAN_BEGIN
BUSY_SCAN_END
EVK_STORAGE_CHECK_PATH=/tmp
EVK_TMP_AVAILABLE_KB=4000000
EVK_MEM_AVAILABLE_KB=8000000
EVK_UNAME=Linux test-device.invalid
"""
        result = device_gate("test-device.invalid", "/tmp/tflite-tp-phase3-test", 1024, 1024)
        self.assertEqual(result["gate"], "PASS")
        self.assertEqual(result["storage_check_path"], "/tmp")
        remote_script = ssh.call_args.args[1]
        self.assertIn('while [ ! -d "$storage_path" ]', remote_script)
        self.assertIn('df -Pk "$storage_path"', remote_script)

    def test_release_gate_does_not_promote_missing_or_unrun_exact_statuses(self):
        gate = derive_release_gate({
            "profile_gates": {"TP4_P128_EXACT": "PASS"},
            "host_verified": "NOT_RUN",
            "tp2_row_reduce_contract": "UNPROVEN",
            "distributed_runtime_verified": "NOT_VERIFIED",
        })
        self.assertEqual(gate["release_gate"], "BLOCKED")
        self.assertFalse(gate["eligible"])
        self.assertEqual(gate["profile_gates"]["TP2_P128_EXACT"], "NOT_RUN")
        self.assertGreaterEqual(len(gate["blockers"]), 4)

    def test_release_gate_requires_profile_host_and_row_reduce_gates(self):
        gate = derive_release_gate({
            "profile_gates": {
                "TP2_P128_EXACT": "PASS", "TP2_D1_EXACT": "PASS",
                "TP4_P128_EXACT": "PASS", "TP4_D1_EXACT": "PASS",
            },
            "host_verified": "PASS",
            "tp2_row_reduce_contract": "PROVEN",
            "distributed_runtime_verified": "NOT_VERIFIED",
        })
        self.assertEqual(gate["release_gate"], "PASS")
        self.assertTrue(gate["eligible"])
        self.assertTrue(gate["distributed_runtime_is_separate_gate"])

    def test_integer_comparison_is_elementwise_exact_and_locates_first_failure(self):
        expected = np.array([[1, -2, 3]], dtype=np.dtype("<i2"))
        actual = np.array([[1, -1, 3]], dtype=np.dtype("<i2"))
        exact = compare_arrays(expected, expected.copy(), "INT16", "stage/out")
        mismatch = compare_arrays(actual, expected, "INT16", "stage/out")
        self.assertEqual(exact["mismatch_count"], 0)
        self.assertEqual(mismatch["mismatch_count"], 1)
        self.assertEqual(mismatch["max_abs_diff"], 1)
        self.assertEqual(mismatch["first_failure"]["index"], [0, 1])
        self.assertEqual(mismatch["first_failure"]["expected"], -2)
        self.assertEqual(mismatch["first_failure"]["actual"], -1)

    def test_quantization_contract_mismatch_fails_closed(self):
        values = np.array([4], dtype=np.dtype("<i2"))
        with self.assertRaisesRegex(ValueError, "COMPARE_QUANTIZATION_MISMATCH"):
            compare_arrays(
                values, values.copy(), "INT16", "q",
                {"scales": [0.25], "zero_points": [0], "quantized_dimension": 0},
                {"scales": [0.5], "zero_points": [0], "quantized_dimension": 0},
            )

    def test_tp2_row_reduce_without_evidence_is_blocked(self):
        result = tp2_row_reduce_gate({})
        self.assertEqual(result["status"], "UNPROVEN")
        self.assertFalse(result["release_eligible"])
        self.assertEqual(result["first_failure"], "TP2_ROW_REDUCE_SEMANTICS_UNPROVEN")

    def test_candidate_compares_rank_slice_to_verified_same_fixture_npu_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_model = root / "candidate.tflite"
            reference_model = root / "reference.tflite"
            candidate_model.write_bytes(b"candidate")
            reference_model.write_bytes(b"reference")
            q = {"scales": [0.25], "zero_points": [0], "quantized_dimension": 0}
            contracts = {
                str(candidate_model): {"sha256": sha256_file(candidate_model), "outputs": [
                    {"name": "projection", "shape": [1, 2], "dtype": "INT16", "quantization": q},
                ]},
                str(reference_model): {"sha256": sha256_file(reference_model), "outputs": [
                    {"name": "projection", "shape": [1, 8], "dtype": "INT16", "quantization": q},
                ]},
            }
            candidate_out = root / "candidate_run"
            reference_out = root / "reference_run"
            candidate_out.mkdir()
            reference_out.mkdir()
            candidate_values = np.array([[5, 6]], dtype=np.dtype("<i2"))
            reference_values = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.dtype("<i2"))
            (candidate_out / "output0.bin").write_bytes(candidate_values.tobytes())
            (reference_out / "output0.bin").write_bytes(reference_values.tobytes())
            candidate_run = {
                "status": "PASS", "fixture_sha256": "f" * 64,
                "source_tflite_sha256": contracts[str(candidate_model)]["sha256"],
                "outputs": [{"handle_index": 0, "bytes": 4, "sha256": sha256_file(candidate_out / "output0.bin"), "status": "PASS"}],
            }
            reference_run = {
                "status": "CAPTURE_SHA_VERIFIED", "reference_eligible": True, "fixture_sha256": "f" * 64,
                "source_tflite_sha256": contracts[str(reference_model)]["sha256"],
                "outputs": [{"handle_index": 0, "bytes": 16, "sha256": sha256_file(reference_out / "output0.bin"), "status": "PASS"}],
            }
            candidate_manifest = root / "candidate_run.json"
            reference_manifest = root / "reference_run.json"
            candidate_manifest.write_text(json.dumps(candidate_run), encoding="utf-8")
            reference_manifest.write_text(json.dumps(reference_run), encoding="utf-8")
            with patch("tflite_tp_mediatek.npu_verify.exact.load_tflite_contract", side_effect=lambda path: contracts[str(path)]):
                report = compare_npu_runs(
                    candidate_model, candidate_out, candidate_manifest,
                    reference_model, reference_out, reference_manifest, 4, 2,
                )
            self.assertEqual(report["mismatch_count"], 0)
            self.assertEqual(report["exact_gate"], "PASS")
            self.assertEqual(report["comparisons"][0]["mapping"]["start"], 4)

            candidate_values[0, 1] = 7
            (candidate_out / "output0.bin").write_bytes(candidate_values.tobytes())
            candidate_run["outputs"][0]["sha256"] = sha256_file(candidate_out / "output0.bin")
            candidate_manifest.write_text(json.dumps(candidate_run), encoding="utf-8")
            with patch("tflite_tp_mediatek.npu_verify.exact.load_tflite_contract", side_effect=lambda path: contracts[str(path)]):
                report = compare_npu_runs(
                    candidate_model, candidate_out, candidate_manifest,
                    reference_model, reference_out, reference_manifest, 4, 2,
                )
            self.assertEqual(report["mismatch_count"], 1)
            self.assertEqual(report["max_abs_diff"], 1)

    def test_tp4_stitch_compares_rank_order_and_all_output_elements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q = {"scales": [0.5], "zero_points": [0], "quantized_dimension": 0}
            rank_models, rank_outputs, rank_runs = [], [], []
            contracts = {}
            fixture_sha = "a" * 64
            full_values = np.array([[10, 20, 30, 40]], dtype=np.dtype("<i2"))
            for rank, value in enumerate((10, 20, 30, 40)):
                model = root / "rank{}.tflite".format(rank)
                model.write_bytes(("rank{}".format(rank)).encode())
                output_dir = root / "rank{}_outputs".format(rank)
                output_dir.mkdir()
                output = output_dir / "output0.bin"
                output.write_bytes(np.array([[value]], dtype=np.dtype("<i2")).tobytes())
                run_path = root / "rank{}_run.json".format(rank)
                model_sha = sha256_file(model)
                run = {
                    "status": "PASS", "rank": rank, "fixture_sha256": fixture_sha,
                    "source_tflite_sha256": model_sha,
                    "outputs": [{"handle_index": 0, "bytes": 2, "sha256": sha256_file(output), "status": "PASS"}],
                }
                run_path.write_text(json.dumps(run), encoding="utf-8")
                rank_models.append(model)
                rank_outputs.append(output_dir)
                rank_runs.append(run_path)
                contracts[str(model)] = {"sha256": model_sha, "outputs": [
                    {"name": "hidden", "shape": [1, 1], "dtype": "INT16", "quantization": q},
                ]}
            reference_model = root / "reference.tflite"
            reference_model.write_bytes(b"reference")
            ref_outputs = root / "reference_outputs"
            ref_outputs.mkdir()
            ref_output = ref_outputs / "output0.bin"
            ref_output.write_bytes(full_values.tobytes())
            reference_sha = sha256_file(reference_model)
            contracts[str(reference_model)] = {"sha256": reference_sha, "outputs": [
                {"name": "hidden", "shape": [1, 4], "dtype": "INT16", "quantization": q},
            ]}
            ref_run = root / "reference_run.json"
            ref_run.write_text(json.dumps({
                "status": "CAPTURE_SHA_VERIFIED", "reference_eligible": True,
                "fixture_sha256": fixture_sha, "source_tflite_sha256": reference_sha,
                "outputs": [{"handle_index": 0, "bytes": 8, "sha256": sha256_file(ref_output), "status": "PASS"}],
            }), encoding="utf-8")
            with patch("tflite_tp_mediatek.npu_verify.exact.load_tflite_contract", side_effect=lambda path: contracts[str(path)]):
                report = stitch_tp4_outputs(
                    rank_models, rank_outputs, rank_runs, "hidden",
                    reference_model, ref_outputs, ref_run, axis=1,
                )
            self.assertEqual(report["mismatch_count"], 0)
            self.assertEqual(report["exact_gate"], "PASS")

    def test_imported_capture_is_reference_only_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "canonical.tflite"
            model.write_bytes(b"model")
            output_dir = root / "outputs"
            output_dir.mkdir()
            output_bytes = np.array([7, -8], dtype=np.dtype("<i2")).tobytes()
            output = output_dir / "output0.bin"
            output.write_bytes(output_bytes)
            sha = hashlib.sha256(output_bytes).hexdigest()
            model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
            capture = {
                "schema": "test.capture",
                "profile": "P128",
                "fixture_id": "fixture-a",
                "fixture_sha256": "a" * 64,
                "source_model_sha256": "b" * 64,
                "canonical_tflite_sha256": model_sha,
                "canonical_dla": {"sha256": "c" * 64, "target_arch": "mdla5.3,edma3.6"},
                "evk": {"hostname": "test-device.invalid"},
                "invoke": {"exit_code": 0, "status": "USER_REPORTED_SUCCESS", "exact_command": "NOT_CAPTURED"},
                "outputs": [{
                    "handle_index": 0,
                    "semantic_tensor": "stage/out",
                    "shape": [1, 2],
                    "dtype": "INT16",
                    "bytes": len(output_bytes),
                    "sha256": sha,
                }],
            }
            capture_path = root / "capture.json"
            capture_path.write_text(json.dumps(capture), encoding="utf-8")
            contract = {
                "sha256": model_sha,
                "outputs": [{
                    "name": "stage/out", "shape": [1, 2], "dtype": "INT16",
                    "quantization": {"scales": [1.0], "zero_points": [0], "quantized_dimension": 0},
                }],
            }
            destination = root / "NPU_RUN_MANIFEST.json"
            with patch("tflite_tp_mediatek.npu_verify.capture_adapter.load_tflite_contract", return_value=contract):
                result = import_canonical_capture(model, capture_path, output_dir, destination)
            self.assertEqual(result["status"], "CAPTURE_SHA_VERIFIED")
            self.assertTrue(result["reference_eligible"])
            self.assertEqual(result["exact_command"], "NOT_CAPTURED")
            self.assertEqual(result["outputs"][0]["sha256"], sha)

            output.write_bytes(b"tampered")
            with patch("tflite_tp_mediatek.npu_verify.capture_adapter.load_tflite_contract", return_value=contract):
                with self.assertRaisesRegex(ValueError, "CAPTURE_OUTPUT_SHA_OR_SIZE_MISMATCH"):
                    import_canonical_capture(model, capture_path, output_dir, root / "bad.json")

    def test_import_prefix_capture_accepts_d1_comparison_manifest_field_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix_model = root / "prefix.tflite"
            prefix_model.write_bytes(b"prefix-model")
            output_dir = root / "prefix_outputs"
            output_dir.mkdir()
            output = output_dir / "output0.bin"
            output.write_bytes(np.array([[3, -2]], dtype=np.dtype("<i2")).tobytes())
            model_sha = sha256_file(prefix_model)
            canonical_sha = "c" * 64
            fixture_sha = "f" * 64
            graph_path = root / "graph.json"
            graph_path.write_text(json.dumps({
                "destination_sha256": model_sha,
                "source_sha256": canonical_sha,
            }), encoding="utf-8")
            comparison_path = root / "comparison.json"
            comparison_path.write_text(json.dumps({
                "prefix_graph_sha256": model_sha,
                "public_stability_gate": "PASS",
                "fixture_sha256": fixture_sha,
                "prefix_outputs_verified": [{
                    "file": "output0.bin", "bytes": output.stat().st_size,
                    "sha256": sha256_file(output),
                }],
            }), encoding="utf-8")
            canonical_path = root / "canonical_capture.json"
            canonical_path.write_text(json.dumps({
                "canonical_tflite_sha256": canonical_sha,
                "fixture_sha256": fixture_sha,
                "profile": "D1", "fixture_id": "d1_fixture",
                "source_model_sha256": "a" * 64,
            }), encoding="utf-8")
            contract = {
                "sha256": model_sha,
                "outputs": [{
                    "name": "stage/output", "shape": [1, 2], "dtype": "INT16",
                    "quantization": {"scales": [0.5], "zero_points": [0], "quantized_dimension": 0},
                }],
            }
            destination = root / "reference.json"
            with patch("tflite_tp_mediatek.npu_verify.capture_adapter.load_tflite_contract", return_value=contract):
                result = import_canonical_prefix_capture(
                    prefix_model, graph_path, comparison_path, output_dir, destination, canonical_path,
                )
            self.assertEqual(result["status"], "CAPTURE_SHA_VERIFIED")
            self.assertTrue(result["reference_eligible"])
            self.assertEqual(result["source_comparison_output_sha_field"], "prefix_outputs_verified")
            self.assertEqual(result["source_comparison_gate_field"], "public_stability_gate")


if __name__ == "__main__":
    unittest.main()
