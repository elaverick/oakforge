"""Regression checks for first-visible and refined tree representations."""

import unittest
from pathlib import Path

import numpy as np

from oakforge.generation import generate_from_file
from oakforge.render.opengl_viewer import OpenGLViewer
from oakforge.progressive import (
    build_cluster_leaves,
    build_lod_representation,
    generate_coarse_tree,
    _tree_for_level,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPE = PROJECT_ROOT / "configs" / "english_oak.yaml"


class ProgressiveTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full_tree = generate_from_file(RECIPE)
        cls.coarse_tree = generate_coarse_tree(RECIPE)

    def test_initial_tree_has_scaffold_and_a_nonempty_cluster_canopy(self):
        roles = {branch.role for branch in self.coarse_tree.branches}
        self.assertIn("trunk", roles)
        self.assertIn("primary", roles)
        self.assertIn("leader", roles)
        self.assertIn("secondary", roles)
        self.assertNotIn("tertiary", roles)
        self.assertNotIn("twig", roles)
        self.assertNotIn("spray", roles)
        self.assertEqual(len(self.coarse_tree.leaves), 6000)
        self.assertEqual(self.coarse_tree.metadata["progressive_level"], 3)

    def test_coarse_scaffold_matches_the_same_seed_full_recipe(self):
        full_by_id = {branch.branch_id: branch for branch in self.full_tree.branches}
        for branch in self.coarse_tree.branches:
            self.assertIn(branch.branch_id, full_by_id)
            expected = full_by_id[branch.branch_id]
            self.assertEqual(branch.role, expected.role)
            self.assertEqual(len(branch.points), len(expected.points))
            np.testing.assert_allclose(
                np.asarray(branch.points), np.asarray(expected.points), atol=1e-9
            )

    def test_cluster_leaf_placement_is_seed_deterministic(self):
        first = build_cluster_leaves(self.coarse_tree, target_count=250, seed_offset=31)
        second = build_cluster_leaves(self.coarse_tree, target_count=250, seed_offset=31)
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            np.testing.assert_allclose(left.base_position, right.base_position, atol=0.0)
            np.testing.assert_allclose(left.growth_direction, right.growth_direction, atol=0.0)
            self.assertAlmostEqual(left.length, right.length)
            self.assertAlmostEqual(left.width, right.width)

    def test_initial_lod_has_valid_wood_and_foliage_buffers(self):
        representation = build_lod_representation(self.coarse_tree, 3)
        self.assertEqual(representation.level, 3)
        self.assertGreater(representation.wood_mesh.triangle_count, 1000)
        self.assertGreater(representation.foliage_mesh.triangle_count, 1000)
        expected_cpu_bytes = 0
        for mesh in (representation.wood_mesh, representation.foliage_mesh):
            self.assertEqual(len(mesh.positions), len(mesh.normals))
            self.assertEqual(len(mesh.positions), len(mesh.colours))
            self.assertEqual(len(mesh.positions) % 9, 0)
            self.assertTrue(np.isfinite(np.frombuffer(mesh.positions, dtype=np.float32)).all())
            expected_cpu_bytes += sum(
                len(buffer) * buffer.itemsize
                for buffer in (mesh.positions, mesh.normals, mesh.colours)
            )
        self.assertEqual(representation.cpu_bytes, expected_cpu_bytes)

    def test_lod_two_uses_tertiary_structure_and_clustered_foliage(self):
        representation = _tree_for_level(self.full_tree, 2)
        roles = {branch.role for branch in representation.branches}
        self.assertIn("tertiary", roles)
        self.assertNotIn("twig", roles)
        self.assertNotIn("spray", roles)
        self.assertEqual(len(representation.leaves), 18000)
        self.assertEqual(representation.metadata["progressive_level"], 2)



    def test_near_lods_retain_full_branch_hierarchy_and_progressively_add_real_leaves(self):
        lod1 = _tree_for_level(self.full_tree, 1)
        lod0 = _tree_for_level(self.full_tree, 0)
        self.assertEqual(len(lod1.branches), len(self.full_tree.branches))
        self.assertEqual(len(lod0.branches), len(self.full_tree.branches))
        self.assertEqual(len(lod1.leaves), (len(self.full_tree.leaves) + 7) // 8)
        self.assertEqual(len(lod0.leaves), len(self.full_tree.leaves))
        self.assertEqual(lod1.metadata["progressive_level"], 1)
        self.assertEqual(lod0.metadata["progressive_level"], 0)

    def test_distance_lod_selection_uses_hysteresis(self):
        viewer = OpenGLViewer.__new__(OpenGLViewer)
        viewer.active_lod = 3
        viewer.camera_distance = 45.0
        self.assertEqual(viewer._desired_progressive_lod(), 3)
        viewer.camera_distance = 43.9
        self.assertEqual(viewer._desired_progressive_lod(), 2)
        # Model the viewer completing the transition before testing the
        # outward hysteresis edge for the newly active LOD2.
        viewer.active_lod = 2
        viewer.camera_distance = 48.0
        self.assertEqual(viewer._desired_progressive_lod(), 2)
        viewer.camera_distance = 48.1
        self.assertEqual(viewer._desired_progressive_lod(), 3)
        viewer.active_lod = 0
        viewer.camera_distance = 15.0
        self.assertEqual(viewer._desired_progressive_lod(), 0)
        viewer.camera_distance = 15.1
        self.assertEqual(viewer._desired_progressive_lod(), 1)

if __name__ == "__main__":
    unittest.main()
