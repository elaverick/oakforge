"""Tests for the canopy-only mesh path used by depth impostors."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from oakforge.core.mesh import (
    build_crown_proxy_mesh,
    build_foliage_cluster_mesh,
    build_foliage_mesh,
    build_tree_mesh,
    build_wood_mesh,
)
from oakforge.core.models import Branch, Leaf, TreeModel


def make_branch(role: str, branch_id: str) -> Branch:
    return Branch(
        branch_id=branch_id,
        parent_id=None,
        role=role,
        depth=1,
        points=[
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ],
        radii=[0.05, 0.01],
    )


class FoliageMeshTests(unittest.TestCase):
    def setUp(self):
        leaf = Leaf(
            leaf_id="leaf-1",
            base_position=np.array([0.0, 1.0, 0.0]),
            growth_direction=np.array([0.0, 1.0, 0.0]),
            width_direction=np.array([1.0, 0.0, 0.0]),
            length=0.25,
            width=0.12,
        )
        self.tree = TreeModel(
            name="Test oak",
            branches=[
                make_branch("primary", "primary-1"),
                make_branch("twig", "twig-1"),
                make_branch("spray", "spray-1"),
            ],
            leaves=[leaf],
        )

    def test_default_foliage_mesh_contains_leaves_only(self):
        foliage = build_foliage_mesh(self.tree)
        self.assertEqual(foliage.triangle_count, 21)

    def test_runtime_leaf_cards_have_uvs_for_alpha_texture(self):
        cards = build_foliage_mesh(self.tree, card_mode=True)
        self.assertEqual(cards.triangle_count, 2)
        self.assertTrue(cards.uses_leaf_texture)
        self.assertEqual(len(cards.texcoords), 6 * 2)
        self.assertEqual(len(cards.texcoords), (len(cards.positions) // 3) * 2)
        self.assertTrue(
            np.isfinite(np.frombuffer(cards.texcoords, dtype=np.float32)).all()
        )

    def test_textured_leaf_obj_exports_uvs_and_material_texture(self):
        foliage = build_foliage_mesh(self.tree, card_mode=True)
        with TemporaryDirectory() as temporary_directory:
            obj_path = Path(temporary_directory) / "LOD0_foliage.obj"
            from oakforge.core.mesh import write_obj

            write_obj(foliage, str(obj_path))
            obj_text = obj_path.read_text(encoding="utf-8")
            mtl_path = obj_path.with_suffix(".mtl")
            mtl_text = mtl_path.read_text(encoding="utf-8")
        self.assertIn("mtllib LOD0_foliage.mtl", obj_text)
        self.assertIn("vt ", obj_text)
        self.assertIn("usemtl OakLeaf", obj_text)
        self.assertIn("map_Kd oak_leaf_atlas.png", mtl_text)
        self.assertIn("map_d oak_leaf_atlas.png", mtl_text)

    def test_far_crown_proxy_is_an_opaque_volume_not_leaf_cards(self):
        crown = build_crown_proxy_mesh(self.tree)
        self.assertEqual(crown.canopy_kind, "crown")
        self.assertEqual(crown.texture_kind, "none")
        self.assertFalse(crown.uses_leaf_texture)
        self.assertGreater(crown.triangle_count, 500)
        self.assertEqual(crown.leaf_count, 0)
        self.assertEqual(len(crown.texcoords), (len(crown.positions) // 3) * 2)
        self.assertTrue(np.isfinite(np.frombuffer(crown.positions, dtype=np.float32)).all())

    def test_middle_lod_builds_crossed_branch_attached_leaf_clumps(self):
        clumps = build_foliage_cluster_mesh(
            self.tree,
            cluster_count=12,
            cluster_scale=0.34,
            seed_offset=7,
        )
        self.assertEqual(clumps.canopy_kind, "clusters")
        self.assertEqual(clumps.texture_kind, "cluster")
        self.assertTrue(clumps.uses_leaf_texture)
        self.assertEqual(clumps.cluster_count, 12)
        self.assertEqual(clumps.triangle_count, 12 * 3 * 2)
        self.assertEqual(len(clumps.texcoords), (len(clumps.positions) // 3) * 2)
        self.assertTrue(np.isfinite(np.frombuffer(clumps.positions, dtype=np.float32)).all())

    def test_clump_cards_have_consistent_winding_and_normals(self):
        clumps = build_foliage_cluster_mesh(
            self.tree, cluster_count=8, cluster_scale=0.32, seed_offset=9
        )
        positions = np.frombuffer(
            clumps.positions, dtype=np.float32
        ).reshape(-1, 3, 3)
        normals = np.frombuffer(
            clumps.normals, dtype=np.float32
        ).reshape(-1, 3, 3)
        geometric_normals = np.cross(
            positions[:, 1] - positions[:, 0],
            positions[:, 2] - positions[:, 0],
        )
        average_normals = normals.mean(axis=1)
        self.assertTrue(
            np.all(np.einsum("ij,ij->i", geometric_normals, average_normals) > 0.0)
        )

    def test_clump_obj_exports_the_cluster_material(self):
        clumps = build_foliage_cluster_mesh(self.tree, cluster_count=3)
        with TemporaryDirectory() as temporary_directory:
            obj_path = Path(temporary_directory) / "LOD2_foliage.obj"
            from oakforge.core.mesh import write_obj

            write_obj(clumps, str(obj_path))
            obj_text = obj_path.read_text(encoding="utf-8")
            mtl_text = obj_path.with_suffix(".mtl").read_text(encoding="utf-8")
        self.assertIn("usemtl OakCluster", obj_text)
        self.assertIn("map_Kd oak_foliage_cluster_atlas.png", mtl_text)
        self.assertIn("map_d oak_foliage_cluster_atlas.png", mtl_text)

    def test_optional_terminal_twigs_are_added_without_structural_limbs(self):
        leaves_only = build_foliage_mesh(self.tree)
        canopy_with_twigs = build_foliage_mesh(self.tree, include_twigs=True)

        # Each two-point terminal tube has 12 sides, two end caps and 48 triangles.
        self.assertEqual(canopy_with_twigs.triangle_count - leaves_only.triangle_count, 96)

    def test_wood_mesh_excludes_leaves_and_can_drop_terminal_twigs(self):
        wood_with_twigs = build_wood_mesh(self.tree)
        structural_wood = build_wood_mesh(self.tree, include_terminal_twigs=False)
        self.assertEqual(wood_with_twigs.triangle_count - structural_wood.triangle_count, 96)

    def test_hybrid_meshes_recompose_the_legacy_full_tree(self):
        full_tree = build_tree_mesh(self.tree)
        wood = build_wood_mesh(self.tree)
        leaves = build_foliage_mesh(self.tree)
        wood_without_twigs = build_wood_mesh(self.tree, include_terminal_twigs=False)
        canopy_with_twigs = build_foliage_mesh(self.tree, include_twigs=True)

        self.assertEqual(full_tree.triangle_count, wood.triangle_count + leaves.triangle_count)
        self.assertEqual(
            full_tree.triangle_count,
            wood_without_twigs.triangle_count + canopy_with_twigs.triangle_count,
        )


if __name__ == "__main__":
    unittest.main()
