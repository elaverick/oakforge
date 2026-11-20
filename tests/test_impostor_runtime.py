"""Tests for runtime canopy impostor selection and depth reconstruction."""

import unittest

import numpy as np

from oakforge.render.impostor_runtime import (
    nearest_view_index,
    reconstruct_depth_offset,
)


class RuntimeImpostorMathTests(unittest.TestCase):
    def test_nearest_view_selects_the_most_aligned_capture(self):
        views = [
            {"direction": [1.0, 0.0, 0.0]},
            {"direction": [0.0, 1.0, 0.0]},
            {"direction": [-1.0, 0.0, 0.0]},
        ]
        self.assertEqual(nearest_view_index(views, np.array([0.9, 0.1, 0.0])), 0)
        self.assertEqual(nearest_view_index(views, np.array([0.0, 1.0, 0.0])), 1)
        self.assertEqual(nearest_view_index(views, np.array([-1.0, 0.0, 0.0])), 2)

    def test_depth_reconstruction_maps_near_and_far_planes(self):
        near, far = 0.01, 60.0
        self.assertAlmostEqual(
            reconstruct_depth_offset(0.0, near, far), far * 0.5 - near
        )
        self.assertAlmostEqual(
            reconstruct_depth_offset(1.0, near, far), -far * 0.5
        )

    def test_depth_reconstruction_midpoint_is_near_origin(self):
        near, far = 0.01, 60.0
        self.assertAlmostEqual(
            reconstruct_depth_offset(0.5, near, far), -near * 0.5
        )

    def test_empty_view_set_is_rejected(self):
        with self.assertRaises(ValueError):
            nearest_view_index([], np.array([0.0, 1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
