import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from tflite_tp_testdata import WorkspaceTemporaryDirectory as TemporaryDirectory
from tflite_tp.errors import ToolError
from tflite_tp.tail import (_set_tail_input_scale, build_tail_tp4,
                                compare_tail_logits)


class TailSafetyTests(unittest.TestCase):
    def test_tail_input_scale_updates_symmetric_int16_quantization(self):
        quantization = SimpleNamespace(
            scale=np.asarray([0.0002645707572810352], dtype=np.float32),
            zeroPoint=np.asarray([0], dtype=np.int64),
            min=np.asarray([-8.67], dtype=np.float32),
            max=np.asarray([8.67], dtype=np.float32),
            quantizedDimension=0,
        )
        tensor = SimpleNamespace(quantization=quantization)

        result = _set_tail_input_scale(tensor, 0.002868709620088339)

        self.assertAlmostEqual(float(quantization.scale[0]), 0.002868709620088339, places=9)
        self.assertEqual(quantization.zeroPoint.tolist(), [0])
        self.assertAlmostEqual(float(quantization.min[0]), -32768 * float(quantization.scale[0]), places=4)
        self.assertAlmostEqual(float(quantization.max[0]), 32767 * float(quantization.scale[0]), places=4)
        self.assertEqual(result["dtype"], "INT16")
        self.assertEqual(result["zero_point"], 0)

    def test_external_input_scale_override_leaves_normalized_fc_bias_scale_alone(self):
        def quant(scale):
            return SimpleNamespace(
                scale=np.asarray([scale], dtype=np.float32),
                zeroPoint=np.asarray([0], dtype=np.int64),
                min=np.asarray([], dtype=np.float32),
                max=np.asarray([], dtype=np.float32),
                quantizedDimension=0,
            )

        external_input = SimpleNamespace(quantization=quant(0.001))
        normalized_fc_input = SimpleNamespace(quantization=quant(0.00129119656))
        fc_bias = SimpleNamespace(quantization=quant(0.00129119656 * 0.01325334795))
        bias_scale = float(fc_bias.quantization.scale[0])

        _set_tail_input_scale(external_input, 0.002868709620088339)

        self.assertAlmostEqual(float(external_input.quantization.scale[0]), 0.002868709620088339, places=9)
        self.assertAlmostEqual(float(normalized_fc_input.quantization.scale[0]), 0.00129119656, places=9)
        self.assertEqual(float(fc_bias.quantization.scale[0]), bias_scale)

    def test_tail_input_scale_rejects_invalid_or_asymmetric_quantization(self):
        for scale in (0, -1, float("nan"), float("inf")):
            with self.subTest(scale=scale):
                tensor = SimpleNamespace(quantization=SimpleNamespace(
                    scale=np.asarray([0.1], dtype=np.float32),
                    zeroPoint=np.asarray([0], dtype=np.int64),
                    min=np.asarray([-1], dtype=np.float32),
                    max=np.asarray([1], dtype=np.float32),
                    quantizedDimension=0,
                ))
                with self.assertRaises(ToolError) as raised:
                    _set_tail_input_scale(tensor, scale)
                self.assertEqual(raised.exception.code, "TAIL_INPUT_SCALE_INVALID")

        asymmetric = SimpleNamespace(quantization=SimpleNamespace(
            scale=np.asarray([0.1], dtype=np.float32),
            zeroPoint=np.asarray([1], dtype=np.int64),
            min=np.asarray([-1], dtype=np.float32),
            max=np.asarray([1], dtype=np.float32),
            quantizedDimension=0,
        ))
        with self.assertRaises(ToolError) as raised:
            _set_tail_input_scale(asymmetric, 0.2)
        self.assertEqual(raised.exception.code, "TAIL_INPUT_QUANTIZATION_UNSUPPORTED")

    def test_logits_compare_supports_tp2_and_tp8(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            hidden = root / 'hidden.bin'
            hidden.write_bytes(b'identical tail input')
            gold = np.arange(128256, dtype=np.int32).astype('<i2')
            gold.tofile(root / 'gold.bin')
            for degree in (2, 8):
                count = len(gold) // degree
                ranks = []
                for rank in range(degree):
                    path = root / ('tp%d_rank%d.bin' % (degree, rank))
                    gold[rank * count:(rank + 1) * count].tofile(path)
                    ranks.append(path)
                result = compare_tail_logits(root / 'gold.bin', ranks, hidden, hidden)
                self.assertEqual(result['status'], 'PASS')
                self.assertEqual(result['input_sha256'], hashlib.sha256(hidden.read_bytes()).hexdigest())
                self.assertEqual(result['per_rank_mismatch'], [0] * degree)

    def test_logits_compare_uses_model_vocabulary_size(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            hidden = root / 'hidden.bin'
            hidden.write_bytes(b'identical tail input')
            gold = np.arange(24, dtype='<i2')
            gold.tofile(root / 'gold.bin')
            for degree in (2, 4, 8):
                count = len(gold) // degree
                ranks = []
                for rank in range(degree):
                    path = root / ('tp%d_rank%d.bin' % (degree, rank))
                    gold[rank * count:(rank + 1) * count].tofile(path)
                    ranks.append(path)
                self.assertEqual(compare_tail_logits(root / 'gold.bin', ranks, hidden, hidden)['status'], 'PASS')

    def test_logits_rejects_unsupported_rank_count(self):
        with self.assertRaises(ToolError) as raised:
            compare_tail_logits('unused', ['a', 'b', 'c'], 'unused', 'unused')
        self.assertEqual(raised.exception.code, 'TAIL_RANK_COUNT_INVALID')

    def test_logits_report_mismatch_and_token(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            hidden = root / 'hidden.bin'
            hidden.write_bytes(b'identical tail input')
            gold = np.zeros(128256, dtype='<i2')
            gold[3] = 8
            gold.tofile(root / 'gold.bin')
            actual = gold.copy()
            actual[32065] = 9
            ranks = []
            for rank in range(4):
                path = root / ('rank%d.bin' % rank)
                actual[rank * 32064:(rank + 1) * 32064].tofile(path)
                ranks.append(path)
            result = compare_tail_logits(root / 'gold.bin', ranks, hidden, hidden)
            self.assertEqual(result['per_rank_mismatch'], [0, 1, 0, 0])
            self.assertEqual((result['candidate_token'], result['golden_token']), (32065, 3))

    def test_logits_reject_different_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            golden_input = root / 'golden_hidden.bin'
            candidate_input = root / 'candidate_hidden.bin'
            golden_input.write_bytes(b'golden hidden')
            candidate_input.write_bytes(b'candidate hidden')
            with self.assertRaises(ToolError) as raised:
                compare_tail_logits('unused', ['a', 'b'], golden_input, candidate_input)
            self.assertEqual(raised.exception.code, 'TAIL_INPUT_PROVENANCE_MISMATCH')

    def test_unknown_source_is_rejected(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.tflite'
            source.write_bytes(b'unknown')
            target = Path(directory) / 'output'
            with self.assertRaises(ToolError):
                build_tail_tp4(source, target)
            self.assertFalse(target.exists())

    def test_existing_output_is_preserved(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.tflite'
            source.write_bytes(b'fixture')
            target = Path(directory) / 'output'
            target.mkdir()
            marker = target / 'marker'
            marker.write_bytes(b'keep')
            with self.assertRaises(ToolError):
                build_tail_tp4(source, target)
            self.assertEqual(marker.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
