"""Import smoke tests for the interactive benchmark entry points."""

import unittest

import main
from oakforge.render.gpu_metrics import GpuTelemetry
from oakforge.render.opengl_viewer import OpenGLViewer


class ViewerImportTests(unittest.TestCase):
    def test_interactive_entry_points_are_importable_without_opening_a_window(self):
        self.assertTrue(callable(main.main))
        self.assertTrue(callable(OpenGLViewer))
        self.assertTrue(callable(GpuTelemetry))


if __name__ == "__main__":
    unittest.main()
