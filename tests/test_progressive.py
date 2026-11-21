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
        self.assertEqual(len(self.coarse_tree.leaves), 3072)
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

    def test_cluster_sprays_keep_leaf_bases_compactly_grouped(self):
        leaves = build_cluster_leaves(
            self.coarse_tree,
            target_count=48,
            seed_offset=19,
            leaf_size=0.14,
            cluster_size=6,
        )
        self.assertEqual(len(leaves), 48)
        groups = {}
        for leaf in leaves:
            cluster_id = leaf.leaf_id.rsplit("_", 1)[0]
            groups.setdefault(cluster_id, []).append(leaf)
        self.assertEqual(len(groups), 8)
        for cluster_id, cluster_leaves in groups.items():
            self.assertEqual(len(cluster_leaves), 6, cluster_id)
            positions = np.asarray([leaf.base_position for leaf in cluster_leaves])
            distances = np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], axis=2
            )
            self.assertLess(
                float(np.max(distances)),
                0.32,
                f"{cluster_id} should form a compact spray rather than isolated cards",
            )

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


    def test_intermediate_lods_enlarge_real_leaves_for_canopy_coverage(self):
        lod2_tree = _tree_for_level(self.full_tree, 2)
        lod1_tree = _tree_for_level(self.full_tree, 1)
        for tree, profile, scale in (
            (lod2_tree, MID_LEAF_PROFILE, 1.34),
            (lod1_tree, NEAR_LEAF_PROFILE, 1.08),
        ):
            regular = build_foliage_mesh(tree, leaf_profile=profile)
            enlarged = build_foliage_mesh(
                tree, leaf_profile=profile, leaf_size_scale=scale
            )
            regular_positions = np.frombuffer(
                regular.positions, dtype=np.float32
            ).reshape(-1, 3)
            enlarged_positions = np.frombuffer(
                enlarged.positions, dtype=np.float32
            ).reshape(-1, 3)
            self.assertEqual(regular.triangle_count, enlarged.triangle_count)
            self.assertGreater(
                float(np.linalg.norm(np.ptp(enlarged_positions, axis=0))),
                float(np.linalg.norm(np.ptp(regular_positions, axis=0))),
            )

    def test_canopy_surface_area_budget_increases_at_each_finer_lod(self):
        lod2_tree = _tree_for_level(self.full_tree, 2)
        lod1_tree = _tree_for_level(self.full_tree, 1)

        def foliage_area(leaves, count, scale=1.0):
            selected_indices = np.linspace(
                0, len(leaves) - 1, min(count, len(leaves)), dtype=np.int64
            )
            selected = [leaves[int(index)] for index in selected_indices]
            return sum(leaf.length * leaf.width for leaf in selected) * scale ** 2

        # The proxy clusters start larger, but actual leaf coverage increases
        # steadily as real foliage is introduced and its count is raised.
        area_lod3 = foliage_area(self.coarse_tree.leaves, 3072)
        area_lod2 = foliage_area(lod2_tree.leaves, 8192, 1.80)
        area_lod1 = foliage_area(lod1_tree.leaves, 16384, 1.35)
        area_lod0 = foliage_area(self.full_tree.leaves, 40960)
        self.assertLess(area_lod3, area_lod2)
        self.assertLess(area_lod2, area_lod1)
        self.assertLess(area_lod1, area_lod0)

    def test_runtime_leaf_cards_have_consistent_winding_and_normals(self):
        mesh = build_foliage_mesh(
            self.coarse_tree, max_leaf_count=32, card_mode=True
        )
        positions = np.frombuffer(mesh.positions, dtype=np.float32).reshape(-1, 3, 3)
        normals = np.frombuffer(mesh.normals, dtype=np.float32).reshape(-1, 3, 3)
        geometric_normals = np.cross(
            positions[:, 1] - positions[:, 0],
            positions[:, 2] - positions[:, 0],
        )
        average_normals = normals.mean(axis=1)
        self.assertTrue(
            np.all(np.einsum("ij,ij->i", geometric_normals, average_normals) > 0.0),
            "Two-triangle leaf cards must face the same direction as their vertex normals",
        )

    def test_runtime_lods_obey_per_tree_geometry_budgets(self):
        expected_leaf_counts = {3: 3072, 2: 8192, 1: 16384, 0: 40960}
        representations = {
            3: build_lod_representation(self.coarse_tree, 3),
        }
        lod2 = build_lod_representation(self.full_tree, 2)
        representations[2] = lod2
        representations[1] = build_lod_representation(
            self.full_tree, 1, shared_wood_mesh=lod2.wood_mesh
        )
        representations[0] = build_lod_representation(
            self.full_tree, 0, shared_wood_mesh=lod2.wood_mesh
        )

        for level, expected_leaves in expected_leaf_counts.items():
            representation = representations[level]
            self.assertEqual(representation.foliage_mesh.leaf_count, expected_leaves)
            self.assertEqual(
                representation.foliage_mesh.triangle_count,
                expected_leaves * 2,
                "Runtime leaves must remain two-triangle cards",
            )
            self.assertTrue(representation.foliage_mesh.uses_leaf_texture)
            self.assertEqual(
                len(representation.foliage_mesh.texcoords),
                expected_leaves * 6 * 2,
                "Every card vertex must carry its oak-atlas UV",
            )
            self.assertLess(
                representation.triangle_count,
                200_000,
                f"LOD{level} exceeds the 200k-triangle per-tree runtime budget",
            )
            self.assertEqual(
                len(representation.wood_mesh.positions),
                len(representation.wood_mesh.normals),
            )
            self.assertEqual(
                len(representation.wood_mesh.positions),
                len(representation.wood_mesh.colours),
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
