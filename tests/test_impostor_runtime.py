"""Tests for runtime canopy impostor selection and depth reconstruction."""

import unittest

import numpy as np

from oakforge.render.impostor_runtime import (
    nearest_view_index,
    reconstruct_depth_offset,
    select_view_blend,
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

    def test_spherical_view_blending_uses_barycentric_weights(self):
        from oakforge.render.impostor import fibonacci_directions

        directions = fibonacci_directions(64).astype(np.float64)
        target = directions[7] * 0.55 + directions[23] * 0.45
        target /= np.linalg.norm(target)
        views = [{"direction": direction.tolist()} for direction in directions]
        indices, weights = select_view_blend(views, target)

        self.assertEqual(len(indices), 3)
        self.assertEqual(weights.shape, (3,))
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=7)
        reconstructed = weights @ directions[indices]
        reconstructed /= np.linalg.norm(reconstructed)
        self.assertGreater(float(np.dot(reconstructed, target)), 0.999999)

    def test_spherical_view_blending_rejects_too_few_captures(self):
        with self.assertRaises(ValueError):
            select_view_blend(
                [{"direction": [1.0, 0.0, 0.0]}, {"direction": [0.0, 1.0, 0.0]}],
                np.asarray([1.0, 0.0, 0.0]),
            )

    def test_precomputed_spherical_capture_data_matches_dynamic_selection(self):
        from oakforge.render.impostor import fibonacci_directions
        from oakforge.render.impostor_runtime import _build_spherical_view_triangles

        directions = fibonacci_directions(64).astype(np.float64)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        views = [{"direction": direction.tolist()} for direction in directions]
        target = directions[11] * 0.4 + directions[39] * 0.6
        target /= np.linalg.norm(target)

        dynamic_indices, dynamic_weights = select_view_blend(views, target)
        cached_triangles = _build_spherical_view_triangles(directions)
        cached_indices, cached_weights = select_view_blend(
            views, target, directions, cached_triangles
        )

        self.assertEqual(cached_indices, dynamic_indices)
        np.testing.assert_allclose(cached_weights, dynamic_weights, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
