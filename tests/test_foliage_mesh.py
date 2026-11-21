"""Tests for the canopy-only mesh path used by depth impostors."""

import unittest

import numpy as np

from oakforge.core.mesh import build_foliage_mesh, build_tree_mesh, build_wood_mesh
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
