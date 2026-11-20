"""Tests for the data aggregation used by the canopy A/B benchmark."""

import unittest

from oakforge.render.opengl_viewer import (
    _average_frame_metrics,
    _average_gpu_metrics,
)


class RepresentationBenchmarkTests(unittest.TestCase):
    def test_frame_averages_use_elapsed_wall_time(self):
        fps, frame_ms = _average_frame_metrics(120, 2.0)
        self.assertEqual(fps, 60.0)
        self.assertAlmostEqual(frame_ms, 1000.0 / 60.0)

    def test_empty_frame_window_has_no_estimated_metrics(self):
        self.assertEqual(_average_frame_metrics(0, 0.0), (None, None))
        self.assertEqual(_average_frame_metrics(10, 0.0), (None, None))

    def test_gpu_samples_are_scoped_to_representation_generation(self):
        samples = [(1, 2.0), (1, 4.0), (2, 10.0)]
        self.assertEqual(_average_gpu_metrics(samples, 1), 3.0)
        self.assertEqual(_average_gpu_metrics(samples, 2), 10.0)
        self.assertIsNone(_average_gpu_metrics(samples, 3))

    def test_gpu_metric_is_unavailable_without_timer_samples(self):
        self.assertIsNone(_average_gpu_metrics([], 1))


if __name__ == "__main__":
    unittest.main()
