import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch



from tflite_tp.cli import main
from tflite_tp.tflite_rewriter import build_tp8_directory


class AutoBuildCliTests(unittest.TestCase):
    @unittest.skip("NCC compilation and automatic NPU repair are outside this repository")
    def test_build_supports_one_profile_and_all_degrees(self):
        output = io.StringIO()
        arguments = ['build', '--decode', 'decode', '--tp', 'all', '--compiler', 'ncc', '--output', 'out']
        with patch('tflite_tp.batch_dla.build_auto', return_value={'status': 'PASS'}) as build, \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(arguments), 0)
        build.assert_called_once_with(None, 'decode', 'all', 'out', 'ncc', None, 32, 'all-gather', None,
                                      fc_width_padding_multiplier=None)

    @unittest.skip("NCC compilation and automatic NPU repair are outside this repository")
    def test_build_forwards_explicit_qkv_padding_and_rejects_unverified_multiplier(self):
        arguments = ['build', '--prefill', 'p64', '--tp', '4', '--compiler', 'ncc', '--output', 'out',
                     '--fc-width-padding-multiplier', '2']
        with patch('tflite_tp.batch_dla.build_auto', return_value={'status': 'PASS'}) as build, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(arguments), 0)
        self.assertEqual(build.call_args.kwargs['fc_width_padding_multiplier'], 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(arguments[:-1] + ['3'])
        self.assertEqual(error.exception.code, 2)

    @unittest.skip("NCC compilation and automatic NPU repair are outside this repository")
    def test_build_reports_compile_failure_as_nonzero(self):
        output = io.StringIO()
        arguments = ['build', '--prefill', 'prefill', '--tp', '2', '--compiler', 'ncc', '--output', 'out']
        with patch('tflite_tp.batch_dla.build_auto', return_value={'status': 'FAIL'}), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(arguments), 1)


class PartitionerAutoRepairCliTests(unittest.TestCase):
    @unittest.skip("NCC compilation and automatic NPU repair are outside this repository")
    def test_auto_repair_command_is_exposed_by_partitioner_and_passes_evidence(self):
        arguments = [
            "auto-repair-tp4-mlp",
            "--canonical-source", "canonical.tflite",
            "--candidate-manifest", "candidate/MANIFEST.json",
            "--comparison-report", "npu/compare.json",
            "--root-cause-report", "npu/baseline.json",
            "--gate-trace", "npu/gate-trace.json",
            "--npu-evidence", "npu/regression.json",
            "--npu-evidence-checksums", "npu/SHA256SUMS",
            "--output", "rebuilt",
        ]
        expected = {
            "status": "NPU_COMPONENT_EXACT_SAVED_IDENTICAL_SHA",
            "selected_rule": "TP4_MLP_GATE_DEQUANT_WIDTH_PAD_2X",
            "release_gate": "BLOCKED",
        }
        stdout = io.StringIO()
        with patch("tflite_tp.cli.auto_repair_tp4_mlp_candidate", return_value=expected) as repair:
            with contextlib.redirect_stdout(stdout):
                result = main(arguments)

        self.assertEqual(result, 0)
        repair.assert_called_once_with(
            "canonical.tflite", "candidate/MANIFEST.json", "npu/compare.json",
            "npu/baseline.json", "npu/gate-trace.json", "rebuilt",
            "npu/regression.json", "npu/SHA256SUMS",
        )
        self.assertEqual(json.loads(stdout.getvalue()), expected)

    @unittest.skip("NCC compilation and automatic NPU repair are outside this repository")
    def test_auto_repair_failure_is_reported_as_blocked(self):
        stdout = io.StringIO()
        arguments = [
            "auto-repair-tp4-mlp",
            "--canonical-source", "canonical.tflite",
            "--candidate-manifest", "candidate/MANIFEST.json",
            "--comparison-report", "npu/compare.json",
            "--root-cause-report", "npu/baseline.json",
            "--gate-trace", "npu/gate-trace.json",
            "--output", "rebuilt",
        ]
        with patch("tflite_tp.cli.auto_repair_tp4_mlp_candidate", side_effect=ValueError("SHA_MISMATCH")):
            with contextlib.redirect_stdout(stdout):
                result = main(arguments)

        self.assertEqual(result, 2)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["status"], "AUTO_REPAIR_BLOCKED")
        self.assertEqual(report["release_gate"], "BLOCKED")
        self.assertEqual(report["first_failure"], "SHA_MISMATCH")


class Tp8BuildCliTests(unittest.TestCase):
    def test_tp8_directory_forwards_mlp_gate_padding(self):
        expected = {"structural_status": "STRUCTURAL_PASS"}
        with patch("tflite_tp.tflite_rewriter._build_tp_directory", return_value=expected) as builder:
            report = build_tp8_directory(
                "source", "output", profile="P128", expected_layers=1,
                mlp_gate_dequant_width_padding_multiplier=2,
            )

        self.assertIs(report, expected)
        self.assertEqual(builder.call_args.kwargs["degree"], 8)
        self.assertEqual(builder.call_args.kwargs["mlp_gate_dequant_width_padding_multiplier"], 2)

    def test_build_tp8_defaults_to_width_padding_for_prefill_and_decode(self):
        report = {
            "structural_status": "STRUCTURAL_PASS",
            "numerical_status": "NOT_RUN",
            "release_status": "NOT_ELIGIBLE",
            "profile": "P128",
            "expected_layers": 1,
            "model_count": 48,
            "source_coverage": {"complete": True},
            "mismatch_count": None,
            "auto_repair_report": {},
        }
        for profile in ("P128", "D1"):
            stdout = io.StringIO()
            with self.subTest(profile=profile), patch(
                "tflite_tp.cli.discover_path", return_value=object()
            ), patch("tflite_tp.cli.build_tp8_directory", return_value=report) as builder, \
                 contextlib.redirect_stdout(stdout):
                result = main([
                    "build-tp8", "--source", "source", "--output", "out",
                    "--profile", profile, "--expected-layers", "1",
                ])
            self.assertEqual(result, 0)
            self.assertEqual(builder.call_args.kwargs["fc_width_padding_multiplier"], 2)

    def test_build_tp8_passes_opt_in_mlp_option(self):
        report = {
            "structural_status": "STRUCTURAL_PASS",
            "numerical_status": "NOT_RUN",
            "release_status": "NOT_ELIGIBLE",
            "profile": "P128",
            "expected_layers": 1,
            "model_count": 48,
            "source_coverage": {"complete": True},
            "mismatch_count": None,
            "auto_repair_report": {},
        }
        stdout = io.StringIO()
        with patch("tflite_tp.cli.discover_path", return_value=object()), patch(
            "tflite_tp.cli.build_tp8_directory", return_value=report
        ) as builder, contextlib.redirect_stdout(stdout):
            result = main([
                "build-tp8", "--source", "source", "--output", "out",
                "--profile", "P128", "--expected-layers", "1",
                "--fc-width-padding-multiplier", "0",
                "--mlp-gate-dequant-width-padding-multiplier", "2",
            ])

        self.assertEqual(result, 0)
        self.assertEqual(builder.call_args.kwargs["mlp_gate_dequant_width_padding_multiplier"], 2)
        self.assertEqual(builder.call_args.kwargs["fc_width_padding_multiplier"], None)
        self.assertEqual(json.loads(stdout.getvalue())["structural_status"], "STRUCTURAL_PASS")

    def test_build_tp8_rejects_unknown_mlp_padding_multiplier(self):
        stdout = io.StringIO()
        with patch("tflite_tp.cli.discover_path", return_value=object()), contextlib.redirect_stdout(stdout):
            result = main([
                "build-tp8", "--source", "source", "--output", "out",
                "--profile", "P128", "--expected-layers", "1",
                "--mlp-gate-dequant-width-padding-multiplier", "3",
            ])

        self.assertEqual(result, 2)
        self.assertEqual(json.loads(stdout.getvalue())["error"], "MLP_GATE_WIDTH_PADDING_MULTIPLIER_UNVERIFIED")


class BatchDlaCliTests(unittest.TestCase):
    @unittest.skip("NCC compilation and automatic NPU repair are outside this repository")
    def test_build_dla_forwards_independent_prefill_and_decode_contexts(self):
        stdout = io.StringIO()
        expected = {"status": "PASS", "results": [1], "numerical_status": "NOT_VERIFIED"}
        with patch("tflite_tp.batch_dla.build_dla", return_value=expected) as build, \
             contextlib.redirect_stdout(stdout):
            result = main([
                "build-dla", "--prefill", "prefill", "--decode", "decode", "--tp", "4",
                "--prefill-tokens", "64", "--decode-tokens", "7", "--expected-layers", "16",
                "--output", "out", "--compiler", "ncc", "--tail", "tail",
            ])

        self.assertEqual(result, 0)
        build.assert_called_once_with("prefill", "decode", 4, 64, 7, 16, "out", "ncc", None, "tail")
        self.assertEqual(json.loads(stdout.getvalue()), {
            "status": "PASS", "compiled_models": 1, "numerical_status": "NOT_VERIFIED",
        })


if __name__ == "__main__":
    unittest.main()
