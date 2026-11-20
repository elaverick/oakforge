"""Tests for coherent capture selection and depth reconstruction."""

import unittest

import numpy as np

from oakforge.render.impostor_runtime import (
    IMPOSTOR_STYLES,
    DepthImpostorRenderer,
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

    def test_impostor_style_modes_and_validation(self):
        renderer = object.__new__(DepthImpostorRenderer)
        for style in IMPOSTOR_STYLES:
            renderer.set_style(style)
            self.assertEqual(renderer.style, style)
        with self.assertRaises(ValueError):
            renderer.set_style("blur")

    def test_live_foliage_controls_are_clamped(self):
        renderer = object.__new__(DepthImpostorRenderer)
        renderer.noise_intensity = 1.35
        renderer.noise_scale = 28.0
        renderer.angularity = 0.8
        renderer.splat_scale = 2.6

        self.assertAlmostEqual(renderer.adjust_noise_intensity(0.25), 1.6)
        self.assertEqual(renderer.adjust_noise_intensity(5.0), 2.5)
        self.assertEqual(renderer.adjust_noise_intensity(-5.0), 0.0)

        self.assertAlmostEqual(renderer.adjust_noise_scale(-3.0), 25.0)
        self.assertEqual(renderer.adjust_noise_scale(-100.0), 1.0)
        self.assertEqual(renderer.adjust_noise_scale(200.0), 80.0)

        self.assertAlmostEqual(renderer.adjust_angularity(0.2), 1.0)
        self.assertEqual(renderer.adjust_angularity(-5.0), 0.0)
        self.assertEqual(renderer.adjust_angularity(5.0), 1.0)

        self.assertAlmostEqual(renderer.adjust_splat_scale(0.4), 3.0)
        self.assertEqual(renderer.adjust_splat_scale(-5.0), 0.8)
        self.assertEqual(renderer.adjust_splat_scale(20.0), 6.0)



if __name__ == "__main__":
    unittest.main()
