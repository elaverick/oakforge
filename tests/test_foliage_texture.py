"""Tests for the procedural oak leaf texture atlas."""

import unittest

import numpy as np

from oakforge.render.foliage_texture import (
    build_oak_cluster_atlas,
    build_oak_leaf_atlas,
)


class OakLeafTextureTests(unittest.TestCase):
    def test_atlas_contains_transparent_lobed_leaf_silhouettes(self):
        atlas = build_oak_leaf_atlas()
        pixels = np.asarray(atlas, dtype=np.uint8)
        self.assertEqual(atlas.size, (176, 256))
        self.assertEqual(pixels.shape, (256, 176, 4))
        self.assertGreater(np.count_nonzero(pixels[:, :, 3] == 0), 10_000)
        self.assertGreater(np.count_nonzero(pixels[:, :, 3] > 240), 10_000)
        self.assertGreater(
            len(np.unique(pixels[pixels[:, :, 3] > 0, :3], axis=0)),
            32,
        )

    def test_cluster_atlas_contains_dense_transparent_leaf_sprays(self):
        atlas = build_oak_cluster_atlas()
        pixels = np.asarray(atlas, dtype=np.uint8)
        self.assertEqual(atlas.size, (256, 256))
        self.assertEqual(pixels.shape, (256, 256, 4))
        self.assertGreater(np.count_nonzero(pixels[:, :, 3] == 0), 8_000)
        self.assertGreater(np.count_nonzero(pixels[:, :, 3] > 240), 18_000)

    def test_cluster_atlas_generation_is_deterministic(self):
        first = np.asarray(build_oak_cluster_atlas())
        second = np.asarray(build_oak_cluster_atlas())
        np.testing.assert_array_equal(first, second)

    def test_atlas_generation_is_deterministic(self):
        first = np.asarray(build_oak_leaf_atlas())
        second = np.asarray(build_oak_leaf_atlas())
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
