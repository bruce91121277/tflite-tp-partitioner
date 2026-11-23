import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from tflite_tp.capabilities import inspect_llama_source
from tflite_tp.errors import ToolError


class CapabilityTests(unittest.TestCase):
    def inspect(self, kv_heads=8, tokens=16, requested=16, layers=1, value_layers=None):
        shapes = {"q": (2048, 2048), "k": (kv_heads * 64, 2048),
                  "v": (kv_heads * 64, 2048), "o": (2048, 2048),
                  "gate": (8192, 2048), "up": (8192, 2048), "down": (2048, 8192)}
        tensors = [SimpleNamespace(name="hidden", shape=(1, tokens, 2048))]
        value_layers = layers if value_layers is None else value_layers
        for layer in range(layers):
            suffix = "_{}".format(layer) if layers > 1 else ""
            tensors.append(SimpleNamespace(name="past_keys{}".format(suffix), shape=(1, kv_heads, 1024, 64)))
        for layer in range(value_layers):
            suffix = "_{}".format(layer) if layers > 1 else ""
            tensors.append(SimpleNamespace(name="past_values{}".format(suffix), shape=(1, kv_heads, 1024, 64)))
        graph = SimpleNamespace(tensors=tensors, inputs=tuple(range(len(tensors))))
        model = SimpleNamespace(graphs=[graph], source_sha256="a" * 64)
        matches = [(0, role, SimpleNamespace(shape=shape), None) for role, shape in shapes.items()]
        with patch("tflite_tp.capabilities.read_tflite", return_value=model), \
             patch("tflite_tp.capabilities._weight_matches", return_value=(matches, [])):
            return inspect_llama_source("source.tflite", requested)

    def test_head_geometry_limits_tp(self):
        result = self.inspect(kv_heads=8)
        self.assertEqual((result["query_heads"], result["kv_heads"], result["head_dim"]), (32, 8, 64))
        self.assertEqual(result["eligible_tp_degrees"], [2, 4, 8])
        self.assertEqual(self.inspect(kv_heads=4)["eligible_tp_degrees"], [2, 4])
        self.assertEqual(self.inspect(kv_heads=1)["eligible_tp_degrees"], [])

    def test_rejects_token_length_mismatch(self):
        with self.assertRaises(ToolError) as raised:
            self.inspect(tokens=16, requested=8)
        self.assertEqual(raised.exception.code, "TOKEN_COUNT_MISMATCH")

    def test_accepts_matching_per_layer_cache_pairs(self):
        result = self.inspect(layers=16)
        self.assertEqual(result["eligible_tp_degrees"], [2, 4, 8])

    def test_rejects_unpaired_layer_cache(self):
        with self.assertRaises(ToolError) as raised:
            self.inspect(layers=16, value_layers=15)
        self.assertEqual(raised.exception.code, "KV_HEADS_UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
