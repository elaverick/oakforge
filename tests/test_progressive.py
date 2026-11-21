"""Regression checks for first-visible and refined tree representations."""

import unittest
from pathlib import Path

import numpy as np

from oakforge.core.mesh import FAR_LEAF_PROFILE, MID_LEAF_PROFILE, NEAR_LEAF_PROFILE, OAK_LEAF_PROFILE, build_foliage_mesh
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
        self.assertIn("tertiary", roles)
        self.assertNotIn("twig", roles)
        self.assertNotIn("spray", roles)
        self.assertEqual(len(self.coarse_tree.leaves), 35520)
        self.assertEqual(self.coarse_tree.metadata["progressive_level"], 3)

    def test_far_lod_fuses_roots_and_major_scaffold_limbs(self):
        surface = self.coarse_tree.metadata.get("wood_surface")
        self.assertIsInstance(surface, dict)
        self.assertGreater(surface["triangle_count"], 1000)

        fused_branch_ids = set(surface["branch_ids"])
        trunk = next(branch for branch in self.coarse_tree.branches if branch.role == "trunk")
        roots = [branch for branch in self.coarse_tree.branches if branch.role == "root"]
        self.assertGreater(len(roots), 7)
        self.assertIn(trunk.branch_id, fused_branch_ids)
        self.assertTrue(
            {branch.branch_id for branch in roots}.issubset(fused_branch_ids),
            "Every primary, lateral and feeder root must join the fused basal surface",
        )

        # The distant implicit surface must resolve root tips beyond a single voxel.
        self.assertGreaterEqual(
            surface["root_tip_radius_minimum"], surface["voxel_size"] * 1.25
        )
        major_limb_ids = {
            branch.branch_id
            for branch in self.coarse_tree.branches
            if branch.role in {"primary", "leader", "secondary"}
        }
        self.assertTrue(
            major_limb_ids.issubset(fused_branch_ids),
            "LOD3 must retain the fused major-limb surface used by refined LODs",
        )
        full_surface_ids = set(self.full_tree.metadata["wood_surface"]["branch_ids"])
        self.assertEqual(
            fused_branch_ids,
            full_surface_ids,
            "LOD3 and refined LODs must use the same fused trunk/root/major-limb network",
        )

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
            np.testing.assert_allclose(
                np.asarray(branch.radii), np.asarray(expected.radii), atol=1e-9
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
        self.assertIn("twig", roles)
        self.assertIn("spray", roles)
        self.assertEqual(len(representation.branches), len(self.full_tree.branches))
        self.assertEqual(len(representation.leaves), (len(self.full_tree.leaves) + 3) // 4)
        self.assertEqual(representation.metadata["progressive_level"], 2)



    def test_near_lods_retain_full_branch_hierarchy_and_progressively_add_real_leaves(self):
        lod1 = _tree_for_level(self.full_tree, 1)
        lod0 = _tree_for_level(self.full_tree, 0)
        self.assertEqual(len(lod1.branches), len(self.full_tree.branches))
        self.assertEqual(len(lod0.branches), len(self.full_tree.branches))
        self.assertIn("twig", {branch.role for branch in lod1.branches})
        self.assertIn("spray", {branch.role for branch in lod1.branches})
        self.assertEqual(len(lod1.leaves), (len(self.full_tree.leaves) + 1) // 2)
        self.assertEqual(len(lod0.leaves), len(self.full_tree.leaves))
        self.assertEqual(lod1.metadata["progressive_level"], 1)
        self.assertEqual(lod0.metadata["progressive_level"], 0)


    def test_foliage_mesh_profiles_reduce_triangles_at_each_lod(self):
        lod3_mesh = build_foliage_mesh(self.coarse_tree, leaf_profile=FAR_LEAF_PROFILE)
        lod2_tree = _tree_for_level(self.full_tree, 2)
        lod1_tree = _tree_for_level(self.full_tree, 1)
        lod0_tree = _tree_for_level(self.full_tree, 0)
        lod2_mesh = build_foliage_mesh(lod2_tree, leaf_profile=MID_LEAF_PROFILE)
        lod1_mesh = build_foliage_mesh(lod1_tree, leaf_profile=NEAR_LEAF_PROFILE)
        lod0_mesh = build_foliage_mesh(lod0_tree, leaf_profile=OAK_LEAF_PROFILE)
        self.assertEqual(lod3_mesh.triangle_count, len(self.coarse_tree.leaves) * (2 * len(FAR_LEAF_PROFILE) - 1))
        self.assertEqual(lod2_mesh.triangle_count, len(lod2_tree.leaves) * (2 * len(MID_LEAF_PROFILE) - 1))
        self.assertEqual(lod1_mesh.triangle_count, len(lod1_tree.leaves) * (2 * len(NEAR_LEAF_PROFILE) - 1))
        self.assertEqual(lod0_mesh.triangle_count, len(lod0_tree.leaves) * (2 * len(OAK_LEAF_PROFILE) - 1))
        self.assertLess(lod3_mesh.triangle_count, lod2_mesh.triangle_count)
        self.assertLess(lod2_mesh.triangle_count, lod1_mesh.triangle_count)
        self.assertLess(lod1_mesh.triangle_count, lod0_mesh.triangle_count)


    def test_lod_two_uses_enlarged_leaves_to_preserve_canopy_coverage(self):
        lod2_tree = _tree_for_level(self.full_tree, 2)
        regular = build_foliage_mesh(lod2_tree, leaf_profile=MID_LEAF_PROFILE)
        enlarged = build_foliage_mesh(
            lod2_tree, leaf_profile=MID_LEAF_PROFILE, leaf_size_scale=1.20
        )
        regular_positions = np.frombuffer(regular.positions, dtype=np.float32).reshape(-1, 3)
        enlarged_positions = np.frombuffer(enlarged.positions, dtype=np.float32).reshape(-1, 3)
        self.assertEqual(regular.triangle_count, enlarged.triangle_count)
        self.assertGreater(
            float(np.linalg.norm(np.ptp(enlarged_positions, axis=0))),
            float(np.linalg.norm(np.ptp(regular_positions, axis=0))),
        )

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
