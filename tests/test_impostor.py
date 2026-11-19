"""Unit tests for the renderer-independent parts of impostor baking."""

import unittest

import numpy as np

from oakforge.render.impostor import (
    ImpostorSettings,
    atlas_dimensions,
    fibonacci_directions,
    view_basis,
)


class ImpostorMathTests(unittest.TestCase):
    def test_default_atlas_is_eight_by_eight_tiles(self):
        self.assertEqual(atlas_dimensions(ImpostorSettings()), (1024, 1024))

    def test_view_directions_are_unit_length_and_distributed(self):
        directions = fibonacci_directions(64)
        self.assertEqual(directions.shape, (64, 3))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)
        self.assertGreater(float(np.min(directions[:, 1])), -1.0)
        self.assertLess(float(np.max(directions[:, 1])), 1.0)

    def test_view_basis_is_orthonormal_near_poles_and_equator(self):
        for direction in ([0, 1, 0], [0, -1, 0], [1, 0, 0], [0.3, 0.4, 0.5]):
            right, up, forward = view_basis(direction)
            basis = np.stack((right, up, forward))
            np.testing.assert_allclose(basis @ basis.T, np.identity(3), atol=1e-6)

    def test_invalid_tile_size_is_rejected(self):
        with self.assertRaises(ValueError):
            atlas_dimensions(ImpostorSettings(tile_size=100))


if __name__ == "__main__":
    unittest.main()
