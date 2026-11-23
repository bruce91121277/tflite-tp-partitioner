import contextlib
import io
import sys
import unittest
from pathlib import Path


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from tflite_tp.cli import build_parser, main  # noqa: E402


class AttentionContextPaddingCliTests(unittest.TestCase):
    def test_build_tp4_exposes_opt_in_attention_context_rule(self):
        args = build_parser().parse_args([
            "build-tp4",
            "--source", "canonical.tflite",
            "--output", "out",
            "--profile", "P128",
            "--expected-layers", "1",
            "--attention-context-head-padding-multiplier", "4",
        ])
        self.assertEqual(args.attention_context_head_padding_multiplier, 4)

    def test_build_tp4_rejects_unverified_attention_multiplier(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = main([
                "build-tp4",
                "--source", "canonical.tflite",
                "--output", "out",
                "--profile", "P128",
                "--expected-layers", "1",
                "--attention-context-head-padding-multiplier", "2",
            ])
        self.assertEqual(result, 2)
        self.assertIn("ATTENTION_CONTEXT_PADDING_MULTIPLIER_UNVERIFIED", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
