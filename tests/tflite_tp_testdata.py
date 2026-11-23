import shutil
import tempfile
from pathlib import Path


class WorkspaceTemporaryDirectory:
    def __init__(self):
        self.path = Path(tempfile.mkdtemp(prefix="tflite_tp_test_"))

    def __enter__(self):
        return str(self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.path)


TEST_WEIGHT_SHAPES = {
    "q": (4096, 4096),
    "k": (1024, 4096),
    "v": (1024, 4096),
    "o": (4096, 4096),
    "gate": (14336, 4096),
    "up": (14336, 4096),
    "down": (4096, 14336),
}
