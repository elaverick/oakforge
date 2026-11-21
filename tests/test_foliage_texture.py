"""Tests for the procedural oak leaf texture atlas."""

import unittest

import numpy as np

from oakforge.render.foliage_texture import build_oak_leaf_atlas


class OakLeafTextureTests(unittest.TestCase):
    def test_atlas_contains_transparent_lobed_leaf_silhouettes(self):
        atlas = build_oak_leaf_atlas()
        pixels = np.asarray(atlas, dtype=np.uint8)
        self.assertEqual(atlas.size, (352, 512))
        self.assertEqual(pixels.shape, (512, 352, 4))
        self.assertGreater(np.count_nonzero(pixels[:, :, 3] == 0), 20_000)
        self.assertGreater(np.count_nonzero(pixels[:, :, 3] > 240), 20_000)
        self.assertGreater(
            len(np.unique(pixels[pixels[:, :, 3] > 0, :3], axis=0)),
            500,
        )

    def test_atlas_generation_is_deterministic(self):
        first = np.asarray(build_oak_leaf_atlas())
        second = np.asarray(build_oak_leaf_atlas())
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
