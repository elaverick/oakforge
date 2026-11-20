"""Unit tests for the renderer-independent parts of impostor baking."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from oakforge.render.impostor import (
    ImpostorSettings,
    atlas_dimensions,
    fibonacci_directions,
    view_basis,
    write_png,
)


class ImpostorMathTests(unittest.TestCase):
    def test_default_atlas_is_eight_by_eight_tiles(self):
        self.assertEqual(atlas_dimensions(ImpostorSettings()), (3104, 3104))

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
            atlas_dimensions(ImpostorSettings(tile_size=24))

    def test_depth_atlas_png_preserves_16_bit_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "depth.png"
            source = np.asarray([[0, 12345, 65535], [65534, 32768, 1]], dtype=np.uint16)
            write_png(path, source)
            with Image.open(path) as image:
                decoded = np.asarray(image).astype(np.uint16)
            np.testing.assert_array_equal(decoded, source)

    def test_multiple_depth_layers_are_bounded(self):
        with self.assertRaises(ValueError):
            atlas_dimensions(ImpostorSettings(layers=5))


if __name__ == "__main__":
    unittest.main()
